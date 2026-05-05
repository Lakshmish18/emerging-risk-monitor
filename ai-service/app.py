import time
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request

from services.categoriser import categorise_text
from services.chroma_store import init_collection
from services.groq_client import GROQ_MODEL_NAME
from services.query_service import answer_query
from services.query_service import get_query_cache_stats
from services.report_job import get_job_record, start_report_job
from services.response_meta import build_meta, json_ok
from services.runtime_metrics import get_runtime_stats

app = Flask(__name__)

_HEALTH_DOC_CACHE: Dict[str, Any] = {"ts": 0.0, "count": 0}
_HEALTH_DOC_TTL_SEC = 60.0


def _cached_chroma_doc_count() -> int:
    now = time.perf_counter()
    if now - float(_HEALTH_DOC_CACHE["ts"]) < _HEALTH_DOC_TTL_SEC:
        return int(_HEALTH_DOC_CACHE["count"])
    try:
        count = int(init_collection().count())
    except Exception:
        count = 0
    _HEALTH_DOC_CACHE["ts"] = now
    _HEALTH_DOC_CACHE["count"] = count
    return count


@app.get("/")
def root() -> Any:
    t0 = time.perf_counter()
    body = {
        "service": "emerging-risk-monitor-ai",
        "status": "running",
        "endpoints": [
            "GET /health",
            "POST /categorise",
            "POST /query",
            "POST /generate-report",
            "GET /generate-report/<job_id>",
        ],
    }
    meta = build_meta(
        response_time_ms=(time.perf_counter() - t0) * 1000,
        model_used=GROQ_MODEL_NAME,
        tokens_used=None,
        cached=False,
        is_fallback=False,
        confidence=None,
    )
    return jsonify(json_ok(body, meta))


def _err(msg: str, t0: float, code: int = 400) -> Tuple[Any, int]:
    ms = (time.perf_counter() - t0) * 1000
    return (
        jsonify(
            {
                "error": msg,
                "meta": build_meta(
                    response_time_ms=ms,
                    model_used=GROQ_MODEL_NAME,
                    tokens_used=None,
                    cached=False,
                    is_fallback=False,
                    confidence=None,
                ),
            }
        ),
        code,
    )


@app.post("/categorise")
def categorise() -> Any:
    t0 = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return _err("Request JSON must include a non-empty 'text' field.", t0, 400)

    result = categorise_text(text=text.strip())
    wall_ms = (time.perf_counter() - t0) * 1000
    meta = build_meta(
        response_time_ms=wall_ms,
        model_used=result["model_used"],
        tokens_used=result["tokens_used"],
        cached=False,
        is_fallback=result["is_fallback"],
        confidence=result["confidence"],
    )
    body: Dict[str, Any] = {
        "category": result["category"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }
    return jsonify(json_ok(body, meta))


@app.post("/query")
def query() -> Any:
    t0 = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return _err("Request JSON must include a non-empty 'question' field.", t0, 400)

    fresh = (
        request.args.get("fresh", "").lower() in ("1", "true", "yes")
        or request.headers.get("X-Fresh-Request", "").strip().lower() == "true"
        or payload.get("fresh") is True
    )

    result = answer_query(question=question.strip(), top_k=3, skip_cache=fresh)
    wall_ms = (time.perf_counter() - t0) * 1000
    meta = build_meta(
        response_time_ms=wall_ms,
        model_used=result["model_used"],
        tokens_used=result["tokens_used"],
        cached=result["cached"],
        is_fallback=result["is_fallback"],
        confidence=result.get("confidence"),
    )
    body = {
        "answer": result["answer"],
        "sources": result["sources"],
    }
    return jsonify(json_ok(body, meta))


@app.post("/generate-report")
def generate_report() -> Any:
    t0 = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    brief = payload.get("brief")
    webhook_url = payload.get("webhook_url")

    if not isinstance(brief, str) or not brief.strip():
        return _err("Request JSON must include a non-empty 'brief' field.", t0, 400)
    if webhook_url is not None and (not isinstance(webhook_url, str) or not webhook_url.strip()):
        return _err("webhook_url must be a non-empty string when provided.", t0, 400)

    job_id = start_report_job(
        brief.strip(),
        webhook_url.strip() if isinstance(webhook_url, str) and webhook_url.strip() else None,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    meta = build_meta(
        response_time_ms=wall_ms,
        model_used=GROQ_MODEL_NAME,
        tokens_used=0,
        cached=False,
        is_fallback=False,
        confidence=None,
    )
    return jsonify(
        json_ok(
            {
                "job_id": job_id,
                "status": "queued",
            },
            meta,
        )
    )


@app.get("/generate-report/<job_id>")
def generate_report_status(job_id: str) -> Any:
    t0 = time.perf_counter()
    rec = get_job_record(job_id)
    if not rec:
        return _err("Unknown job_id.", t0, 404)

    wall_ms = (time.perf_counter() - t0) * 1000
    r = rec.get("result") or {}
    meta = build_meta(
        response_time_ms=wall_ms,
        model_used=str(r.get("model_used") or GROQ_MODEL_NAME),
        tokens_used=r.get("tokens_used"),
        cached=False,
        is_fallback=bool(r.get("is_fallback")) if rec.get("status") == "completed" else False,
        confidence=None,
    )
    body: Dict[str, Any] = {
        "job_id": job_id,
        "status": rec.get("status"),
        "result": rec.get("result"),
        "error": rec.get("error"),
    }
    return jsonify(json_ok(body, meta))


@app.get("/health")
def health() -> Any:
    t0 = time.perf_counter()
    runtime = get_runtime_stats()
    cache_stats = get_query_cache_stats()

    doc_count = _cached_chroma_doc_count()

    wall_ms = (time.perf_counter() - t0) * 1000
    meta = build_meta(
        response_time_ms=wall_ms,
        model_used=GROQ_MODEL_NAME,
        tokens_used=None,
        cached=False,
        is_fallback=False,
        confidence=None,
    )
    body = {
        "status": "ok",
        "model_name": GROQ_MODEL_NAME,
        "avg_groq_latency_ms_last_10": runtime["avg_response_time_ms_last_10"],
        "chroma_doc_count": doc_count,
        "uptime": {
            "seconds": runtime["uptime_seconds"],
            "human": runtime["uptime_human"],
        },
        "cache_stats": cache_stats,
    }
    return jsonify(json_ok(body, meta))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
