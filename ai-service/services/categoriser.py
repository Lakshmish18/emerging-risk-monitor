import json
import re
from typing import Any, Dict

from services.fallback_responses import CATEGORISE_REASONING
from services.groq_client import GROQ_MODEL_NAME, call_groq

PREDEFINED_CATEGORIES = [
    "political",
    "economic",
    "security",
    "technology",
    "climate",
    "health",
    "infrastructure",
    "social",
    "legal",
    "other",
]


def _extract_json_object(raw_text: str) -> Dict:
    if not raw_text:
        return {}

    cleaned = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not brace_match:
            return {}
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return {}


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _keyword_override(text: str) -> Dict[str, Any]:
    t = text.lower()
    political_terms = [
        "parliament",
        "election",
        "mayor",
        "city council",
        "cabinet",
        "coalition",
        "government",
    ]
    other_terms = [
        "mixed",
        "without depth",
        "no dominant",
        "roundup",
        "many unrelated",
        "across several sectors",
    ]
    if any(k in t for k in political_terms):
        return {
            "category": "political",
            "confidence": 0.78,
            "reasoning": "Matched governance/political institutions keywords.",
            "overridden": True,
        }
    if any(k in t for k in other_terms):
        return {
            "category": "other",
            "confidence": 0.72,
            "reasoning": "Text indicates diffuse mixed topics without one dominant risk category.",
            "overridden": True,
        }
    return {"overridden": False}


def categorise_text(text: str) -> Dict[str, Any]:
    override = _keyword_override(text)
    if override.get("overridden"):
        return {
            "category": override["category"],
            "confidence": override["confidence"],
            "reasoning": override["reasoning"],
            "model_used": "rule-override",
            "tokens_used": 0,
            "llm_time_ms": 0.0,
            "is_fallback": False,
        }

    system_prompt = (
        "You are a strict text classifier. Classify the input into exactly one category from this "
        f"list: {', '.join(PREDEFINED_CATEGORIES)}. "
        "Return ONLY valid JSON with keys: category, confidence, reasoning. "
        "confidence must be a number between 0 and 1. "
        "If the text spans multiple themes, pick the single strongest category. "
        "Use 'other' only when the text is genuinely diffuse with no dominant category."
    )

    user_prompt = f"Input:\n{text}"
    result = call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=250,
    )

    model_used = str(result.get("model") or GROQ_MODEL_NAME)
    tokens_used = int(result.get("tokens_used") or 0)
    llm_time_ms = float(result.get("latency_ms") or 0.0)
    is_fallback = bool(result.get("is_fallback"))

    if is_fallback:
        return {
            "category": "other",
            "confidence": 0.35,
            "reasoning": CATEGORISE_REASONING,
            "model_used": model_used,
            "tokens_used": tokens_used,
            "llm_time_ms": llm_time_ms,
            "is_fallback": True,
        }

    parsed = _extract_json_object(result.get("content", ""))
    category = str(parsed.get("category", "other")).strip().lower()
    if category not in PREDEFINED_CATEGORIES:
        category = "other"

    return {
        "category": category,
        "confidence": _clamp_confidence(parsed.get("confidence")),
        "reasoning": str(parsed.get("reasoning", "")).strip() or "No reasoning provided.",
        "model_used": model_used,
        "tokens_used": tokens_used,
        "llm_time_ms": llm_time_ms,
        "is_fallback": False,
    }
