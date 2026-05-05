from typing import Any, Dict, Optional


def build_meta(
    *,
    response_time_ms: float,
    model_used: str,
    tokens_used: Optional[int],
    cached: bool,
    is_fallback: bool,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Standard meta block for all API JSON responses."""
    out: Dict[str, Any] = {
        "confidence": confidence,
        "model_used": model_used,
        "tokens_used": int(tokens_used) if tokens_used is not None else None,
        "response_time_ms": round(response_time_ms, 2),
        "cached": bool(cached),
        "is_fallback": bool(is_fallback),
    }
    if out["confidence"] is not None:
        try:
            c = float(out["confidence"])
            out["confidence"] = max(0.0, min(1.0, c))
        except (TypeError, ValueError):
            out["confidence"] = None
    return out


def json_ok(body: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {**body, "meta": meta}
