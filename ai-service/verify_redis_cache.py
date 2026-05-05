"""Smoke test: SHA256 key, TTL set via cache layer, hit/miss counters."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.ai_cache import (
    cache_key_sha256,
    get_cache_stats,
    record_cache_miss,
    set_cached,
    try_get_cached,
)


def main() -> None:
    question = "smoke test cache verification question"
    top_k = 3

    digest = cache_key_sha256(question, top_k)
    assert len(digest) == 64, "SHA256 hex length"

    assert try_get_cached(question, top_k) is None
    record_cache_miss()

    payload = {
        "answer": "cached answer",
        "sources": [{"id": "t1", "content": "x", "metadata": {}, "distance": None, "similarity": None}],
        "__meta": {
            "model_used": "test-model",
            "tokens_used": 1,
            "llm_time_ms": 2.0,
            "is_fallback": False,
            "confidence": 0.9,
        },
    }
    set_cached(question, top_k, payload)

    cached = try_get_cached(question, top_k)
    assert cached == payload

    stats = get_cache_stats()
    print("OK — cache round-trip works.")
    print(f"backend: {stats.get('backend')}")
    print(f"hits: {stats['hits']}  misses: {stats['misses']}  size: {stats['size']}  ttl_seconds: {stats.get('ttl_seconds')}")

    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


if __name__ == "__main__":
    main()
