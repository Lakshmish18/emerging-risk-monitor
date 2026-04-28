import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from services.chroma_store import init_collection, query_text
from services.groq_client import call_groq

MAX_CACHE_SIZE = 100
CACHE_TTL = 900  # 15 minutes

_QUERY_CACHE: "OrderedDict[str, Tuple[Dict, float]]" = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0


def get_cached(key: str) -> Optional[Dict]:
    if key in _QUERY_CACHE:
        value, timestamp = _QUERY_CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            _QUERY_CACHE.move_to_end(key)
            return value
        del _QUERY_CACHE[key]
    return None


def set_cached(key: str, value: Dict) -> None:
    if key in _QUERY_CACHE:
        _QUERY_CACHE.move_to_end(key)
    _QUERY_CACHE[key] = (value, time.time())
    if len(_QUERY_CACHE) > MAX_CACHE_SIZE:
        _QUERY_CACHE.popitem(last=False)


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


def answer_query(question: str, top_k: int = 3) -> Dict:
    global _CACHE_HITS, _CACHE_MISSES
    cache_key = question.strip().lower()
    cached = get_cached(cache_key)
    if cached is not None:
        _CACHE_HITS += 1
        return cached
    _CACHE_MISSES += 1

    collection = init_collection()
    result = query_text(collection=collection, query=question, n_results=top_k)
    sources = _build_sources(result)

    if not sources:
        return {
            "answer": "No relevant context found in the knowledge base.",
            "sources": [],
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
        "the question. If context is insufficient, explicitly say so."
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

    answer = (
        llm_result.get("content", "").strip()
        if llm_result and llm_result.get("content")
        else "Unable to generate answer right now."
    )

    final_result = {
        "answer": answer,
        "sources": sources,
    }
    set_cached(cache_key, final_result)
    return final_result


def get_query_cache_stats() -> Dict:
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = (_CACHE_HITS / total) if total else 0.0
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "size": len(_QUERY_CACHE),
        "hit_rate": round(hit_rate, 3),
    }
