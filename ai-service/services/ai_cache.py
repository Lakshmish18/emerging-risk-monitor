import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_PREFIX = "emerging-risk:query:"
HITS_KEY = "emerging-risk:cache:hits"
MISSES_KEY = "emerging-risk:cache:misses"
TTL_SECONDS = 900  # 15 minutes

_redis_client: Any = None
_redis_checked = False

# Fallback when Redis is unavailable (dev): SHA256-keyed entries + TTL
_fallback_store: Dict[str, tuple] = {}
_fallback_hits = 0
_fallback_misses = 0


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _get_redis():
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis

        client = redis.from_url(_redis_url(), decode_responses=True)
        client.ping()
        _redis_client = client
        logger.info("Redis AI cache connected.")
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable; using in-memory AI cache fallback: %s", exc)
        _redis_client = None
        return None


def cache_key_sha256(question: str, top_k: int) -> str:
    raw = f"{question.strip().lower()}|{top_k}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _fallback_get(sha_hex: str) -> Optional[Dict]:
    global _fallback_hits
    now = time.time()
    if sha_hex not in _fallback_store:
        return None
    data, expires_at = _fallback_store[sha_hex]
    if now >= expires_at:
        del _fallback_store[sha_hex]
        return None
    _fallback_hits += 1
    return json.loads(data) if isinstance(data, str) else data


def _fallback_set(sha_hex: str, payload: Dict) -> None:
    encoded = json.dumps(payload)
    _fallback_store[sha_hex] = (encoded, time.time() + TTL_SECONDS)


def _fallback_record_miss() -> None:
    global _fallback_misses
    _fallback_misses += 1


def try_get_cached(question: str, top_k: int) -> Optional[Dict]:
    """Return cached query result or None. Increments hit counter on hit."""
    sha_hex = cache_key_sha256(question, top_k)
    r = _get_redis()
    if r:
        key = CACHE_PREFIX + sha_hex
        raw = r.get(key)
        if raw is not None:
            r.incr(HITS_KEY)
            return json.loads(raw)
        return None
    return _fallback_get(sha_hex)


def record_cache_miss() -> None:
    """Call when cache lookup misses before computing."""
    r = _get_redis()
    if r:
        r.incr(MISSES_KEY)
    else:
        _fallback_record_miss()


def set_cached(question: str, top_k: int, payload: Dict) -> None:
    sha_hex = cache_key_sha256(question, top_k)
    r = _get_redis()
    if r:
        key = CACHE_PREFIX + sha_hex
        r.setex(key, TTL_SECONDS, json.dumps(payload))
        return
    _fallback_set(sha_hex, payload)


def get_cache_stats() -> Dict:
    r = _get_redis()
    if r:
        try:
            hits = int(r.get(HITS_KEY) or 0)
            misses = int(r.get(MISSES_KEY) or 0)
            size = sum(1 for _ in r.scan_iter(match=f"{CACHE_PREFIX}*", count=500))
        except Exception as exc:
            logger.error("Redis cache stats failed: %s", exc)
            hits = misses = size = 0
        total = hits + misses
        hit_rate = (hits / total) if total else 0.0
        return {
            "backend": "redis",
            "hits": hits,
            "misses": misses,
            "size": size,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": TTL_SECONDS,
        }

    total = _fallback_hits + _fallback_misses
    hit_rate = (_fallback_hits / total) if total else 0.0
    return {
        "backend": "memory",
        "hits": _fallback_hits,
        "misses": _fallback_misses,
        "size": len(_fallback_store),
        "hit_rate": round(hit_rate, 3),
        "ttl_seconds": TTL_SECONDS,
    }
