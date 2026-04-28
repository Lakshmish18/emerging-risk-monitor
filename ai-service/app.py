from flask import Flask, jsonify, request

from services.categoriser import categorise_text
from services.chroma_store import init_collection
from services.groq_client import GROQ_MODEL_NAME
from services.query_service import answer_query
from services.query_service import get_query_cache_stats
from services.runtime_metrics import get_runtime_stats

app = Flask(__name__)


@app.post("/categorise")
def categorise():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request JSON must include a non-empty 'text' field."}), 400

    result = categorise_text(text=text.strip())
    return jsonify(
        {
            "category": result["category"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
        }
    )


@app.post("/query")
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Request JSON must include a non-empty 'question' field."}), 400

    result = answer_query(question=question.strip(), top_k=3)
    return jsonify(
        {
            "answer": result["answer"],
            "sources": result["sources"],
        }
    )


@app.get("/health")
def health():
    runtime = get_runtime_stats()
    cache_stats = get_query_cache_stats()

    doc_count = 0
    try:
        doc_count = int(init_collection().count())
    except Exception:
        doc_count = 0

    return jsonify(
        {
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
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
