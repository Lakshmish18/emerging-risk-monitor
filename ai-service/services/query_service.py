import time
from typing import Any, Dict, List, Optional

from services.ai_cache import (
    get_cache_stats,
    record_cache_miss,
    set_cached,
    try_get_cached,
)
from services.chroma_store import init_collection, query_text
from services.fallback_responses import QUERY_ANSWER
from services.groq_client import GROQ_MODEL_NAME, call_groq

_CACHE_META_KEY = "__meta"


def _build_sources(chroma_result: Dict) -> List[Dict]:
    ids = chroma_result.get("ids", [[]])
    docs = chroma_result.get("documents", [[]])
    metas = chroma_result.get("metadatas", [[]])
    distances = chroma_result.get("distances", [[]])

    first_ids = ids[0] if ids else []
    first_docs = docs[0] if docs else []
    first_metas = metas[0] if metas else []
    first_distances = distances[0] if distances else []

    sources = []
    for index, doc_id in enumerate(first_ids):
        distance = first_distances[index] if index < len(first_distances) else None
        similarity = None
        if isinstance(distance, (float, int)):
            similarity = 1.0 / (1.0 + float(distance))

        sources.append(
            {
                "id": doc_id,
                "content": first_docs[index] if index < len(first_docs) else "",
                "metadata": first_metas[index] if index < len(first_metas) else {},
                "distance": distance,
                "similarity": similarity,
            }
        )
    return sources


def _avg_source_confidence(sources: List[Dict]) -> Optional[float]:
    sims = [s.get("similarity") for s in sources if isinstance(s.get("similarity"), (int, float))]
    if not sims:
        return None
    return max(0.0, min(1.0, sum(float(x) for x in sims) / len(sims)))


def answer_query(question: str, top_k: int = 3, skip_cache: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cached = False

    if not skip_cache:
        raw_cached = try_get_cached(question, top_k)
        if raw_cached is not None:
            cached = True
            meta = raw_cached.get(_CACHE_META_KEY) or {}
            answer = raw_cached.get("answer", "")
            sources = raw_cached.get("sources", [])
            elapsed = (time.perf_counter() - t0) * 1000
            return {
                "answer": answer,
                "sources": sources,
                "model_used": str(meta.get("model_used") or GROQ_MODEL_NAME),
                "tokens_used": int(meta.get("tokens_used") or 0),
                "llm_time_ms": float(meta.get("llm_time_ms") or 0.0),
                "response_time_ms": round(elapsed, 2),
                "cached": True,
                "is_fallback": bool(meta.get("is_fallback", False)),
                "confidence": meta.get("confidence"),
            }
        record_cache_miss()

    collection = init_collection()
    result = query_text(collection=collection, query=question, n_results=top_k)
    sources = _build_sources(result)

    if not sources:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "answer": "No relevant context found in the knowledge base.",
            "sources": [],
            "model_used": GROQ_MODEL_NAME,
            "tokens_used": 0,
            "llm_time_ms": 0.0,
            "response_time_ms": round(elapsed, 2),
            "cached": False,
            "is_fallback": False,
            "confidence": None,
        }

    context_lines = []
    for idx, source in enumerate(sources, start=1):
        context_lines.append(
            f"[{idx}] id={source['id']} metadata={source.get('metadata', {})}\n"
            f"{source.get('content', '')}"
        )

    context_block = "\n\n".join(context_lines)
    system_prompt = (
        "You are a helpful risk-analysis assistant. Use ONLY the provided context to answer "
        "the question. If context is insufficient, explicitly say so. "
        "Reply in 2-6 sentences, plain language, no markdown fences."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Return a concise answer grounded in the context."
    )

    llm_result = call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    is_fb = bool(llm_result.get("is_fallback"))
    if is_fb:
        answer = QUERY_ANSWER
    else:
        answer = (llm_result.get("content") or "").strip() or "Unable to generate answer right now."

    conf = _avg_source_confidence(sources)
    elapsed = (time.perf_counter() - t0) * 1000
    llm_time = float(llm_result.get("latency_ms") or 0.0)

    out = {
        "answer": answer,
        "sources": sources,
        "model_used": str(llm_result.get("model") or GROQ_MODEL_NAME),
        "tokens_used": int(llm_result.get("tokens_used") or 0),
        "llm_time_ms": llm_time,
        "response_time_ms": round(elapsed, 2),
        "cached": False,
        "is_fallback": is_fb,
        "confidence": conf,
    }

    if not skip_cache:
        to_store = {
            "answer": out["answer"],
            "sources": out["sources"],
            _CACHE_META_KEY: {
                "model_used": out["model_used"],
                "tokens_used": out["tokens_used"],
                "llm_time_ms": out["llm_time_ms"],
                "is_fallback": out["is_fallback"],
                "confidence": out["confidence"],
            },
        }
        set_cached(question, top_k, to_store)

    return out


def get_query_cache_stats() -> Dict:
    return get_cache_stats()
