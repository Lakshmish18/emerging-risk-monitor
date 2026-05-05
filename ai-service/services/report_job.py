import json
import logging
import threading
import time
import uuid
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from services.fallback_responses import REPORT_MARKDOWN
from services.groq_client import call_groq

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}


def create_job_record() -> str:
    job_id = str(uuid.uuid4())
    with _LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "result": None,
            "error": None,
        }
    return job_id


def get_job_record(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        j = _JOBS.get(job_id)
        return dict(j) if j else None


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(kwargs)


def _post_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Webhook delivery failed: %s", exc)


def _run_report(job_id: str, brief: str, webhook_url: Optional[str]) -> None:
    try:
        _update_job(job_id, status="running")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior risk analyst. Produce a markdown report with sections: "
                    "## Executive summary\n## Key risks\n## Recommended actions\n"
                    "Use bullet points where helpful. Ground statements in the brief; if data is missing, "
                    "state assumptions explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Risk report brief:\n\n{brief}",
            },
        ]
        result_llm = call_groq(messages, temperature=0.25, max_tokens=2000)
        is_fb = bool(result_llm.get("is_fallback"))
        markdown = REPORT_MARKDOWN if is_fb else (result_llm.get("content") or "").strip()
        if not markdown:
            markdown = REPORT_MARKDOWN
            is_fb = True

        result_payload = {
            "markdown": markdown,
            "model_used": result_llm.get("model"),
            "tokens_used": result_llm.get("tokens_used"),
            "is_fallback": is_fb,
        }
        _update_job(job_id, status="completed", result=result_payload, error=None)

        if webhook_url:
            _post_webhook(
                webhook_url,
                {
                    "job_id": job_id,
                    "status": "completed",
                    "result": result_payload,
                },
            )
    except Exception as exc:
        logger.exception("Report job failed: %s", exc)
        _update_job(job_id, status="failed", error=str(exc))
        if webhook_url:
            _post_webhook(
                webhook_url,
                {"job_id": job_id, "status": "failed", "error": str(exc)},
            )


def start_report_job(brief: str, webhook_url: Optional[str]) -> str:
    job_id = create_job_record()
    thread = threading.Thread(
        target=_run_report,
        args=(job_id, brief, webhook_url),
        daemon=True,
        name=f"report-{job_id[:8]}",
    )
    thread.start()
    return job_id
