import json
import re
from typing import Dict

from services.groq_client import call_groq

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


def _clamp_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def categorise_text(text: str) -> Dict:
    system_prompt = (
        "You are a strict text classifier. Classify the input into exactly one category from this "
        f"list: {', '.join(PREDEFINED_CATEGORIES)}. "
        "Return ONLY valid JSON with keys: category, confidence, reasoning. "
        "confidence must be a number between 0 and 1."
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

    if not result:
        return {
            "category": "other",
            "confidence": 0.0,
            "reasoning": "Classification provider unavailable.",
        }

    parsed = _extract_json_object(result.get("content", ""))
    category = str(parsed.get("category", "other")).strip().lower()
    if category not in PREDEFINED_CATEGORIES:
        category = "other"

    return {
        "category": category,
        "confidence": _clamp_confidence(parsed.get("confidence")),
        "reasoning": str(parsed.get("reasoning", "")).strip() or "No reasoning provided.",
    }
