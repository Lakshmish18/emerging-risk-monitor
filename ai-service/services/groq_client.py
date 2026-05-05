import json
import logging
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

from services.runtime_metrics import record_latency_ms

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT_SEC = float(os.getenv("GROQ_TIMEOUT_SEC", "60"))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _empty_fallback_result(latency_ms: float) -> Dict[str, Any]:
    return {
        "content": "",
        "parsed": {"raw_text": ""},
        "model": GROQ_MODEL_NAME,
        "tokens_used": 0,
        "is_fallback": True,
        "latency_ms": round(latency_ms, 2),
    }


def call_groq(
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Calls Groq chat completions. On repeated failure, returns a structured fallback
    dict (never None) with is_fallback=True for downstream template responses.
    """
    started_at = time.perf_counter()

    for attempt in range(1, retries + 1):
        try:
            create_kwargs = dict(
                model=GROQ_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                response = client.chat.completions.create(
                    **create_kwargs,
                    timeout=GROQ_TIMEOUT_SEC,
                )
            except TypeError:
                response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_text": content}

            latency_ms = (time.perf_counter() - started_at) * 1000
            record_latency_ms(latency_ms)
            tokens_used: Optional[int] = None
            try:
                if response.usage and response.usage.total_tokens is not None:
                    tokens_used = int(response.usage.total_tokens)
            except Exception:
                pass

            logger.info("Groq call successful on attempt %s", attempt)
            return {
                "content": content,
                "parsed": parsed,
                "model": getattr(response, "model", None) or GROQ_MODEL_NAME,
                "tokens_used": tokens_used if tokens_used is not None else 0,
                "is_fallback": False,
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as e:
            logger.error("Attempt %s failed: %s", attempt, e)
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                logger.error("All retries exhausted; returning AI fallback envelope.")
                latency_ms = (time.perf_counter() - started_at) * 1000
                record_latency_ms(latency_ms)
                fb = _empty_fallback_result(latency_ms)
                return fb

    # Unreachable; satisfy type checkers
    return _empty_fallback_result((time.perf_counter() - started_at) * 1000)
