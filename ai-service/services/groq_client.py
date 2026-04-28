import os
import time
import logging
import json
from groq import Groq
from dotenv import load_dotenv
from services.runtime_metrics import record_latency_ms

load_dotenv()
logger = logging.getLogger(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

def call_groq(messages: list, temperature: float = 0.3, max_tokens: int = 1000, retries: int = 3):
    started_at = time.time()
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_text": content}
            logger.info(f"Groq call successful on attempt {attempt}")
            record_latency_ms((time.time() - started_at) * 1000)
            return {
                "content": content,
                "parsed": parsed,
                "model": response.model,
                "tokens_used": response.usage.total_tokens
            }
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # 2s -> 4s -> 8s backoff
            else:
                logger.error("All retries exhausted. Returning fallback.")
                record_latency_ms((time.time() - started_at) * 1000)
                return None
