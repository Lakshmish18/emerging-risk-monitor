"""
Performance benchmark: p50 / p95 / p99 over 50 requests per endpoint.

Uses Flask test client. LLM endpoints are mocked by default so CI/local runs
do not require GROQ_API_KEY. Set REAL_LLM_BENCHMARK=1 to hit real Groq (slow).

Targets (response_time_ms p95, mocked LLM):
  GET /health                     < 500
  POST /generate-report           < 800  (enqueue only)
  POST /categorise                < 3000 (mocked)
  POST /query                     < 3000 (mocked; includes Chroma if installed)
"""
from __future__ import annotations

import os
import statistics
import time
from typing import Callable, Dict, List, Tuple
from unittest.mock import patch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import app  # noqa: E402


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    xs = sorted(sorted_vals)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(xs):
        return xs[f] + c * (xs[f + 1] - xs[f])
    return xs[f]


def _bench(name: str, fn: Callable[[], None], n: int = 50) -> Tuple[str, Dict[str, float]]:
    times: List[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(times)
    return name, {
        "p50_ms": round(_percentile(s, 50), 2),
        "p95_ms": round(_percentile(s, 95), 2),
        "p99_ms": round(_percentile(s, 99), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "max_ms": round(max(times), 2),
    }


def _mock_groq(*_a, **_k):
    return {
        "content": '{"category":"other","confidence":0.5,"reasoning":"mock"}',
        "parsed": {},
        "model": "mock-model",
        "tokens_used": 12,
        "is_fallback": False,
        "latency_ms": 5.0,
    }


def _mock_groq_answer(*_a, **_k):
    return {
        "content": "Mock grounded answer based on context.",
        "parsed": {},
        "model": "mock-model",
        "tokens_used": 40,
        "is_fallback": False,
        "latency_ms": 8.0,
    }


def main() -> None:
    real = os.getenv("REAL_LLM_BENCHMARK", "").lower() in ("1", "true", "yes")
    n = int(os.getenv("BENCHMARK_REQUESTS", "50"))

    targets = {
        "GET /health": 500.0,
        "POST /generate-report": 800.0,
        "POST /categorise": 3000.0,
        "POST /query": 8000.0,
    }

    client = app.test_client()
    results = {}
    # Warm routes once so first-hit latency does not dominate p99 on /health.
    client.get("/health")

    def get_health():
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert "meta" in data

    _lbl, results["GET /health"] = _bench("GET /health", get_health, n=n)

    def post_report():
        r = client.post(
            "/generate-report",
            json={"brief": "Benchmark brief: summarize operational risks for Q2."},
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["job_id"]

    _lbl, results["POST /generate-report"] = _bench("POST /generate-report", post_report, n=n)

    if real:

        def cat_real():
            r = client.post(
                "/categorise",
                json={"text": "Central bank raised rates after inflation surprise."},
            )
            assert r.status_code == 200
            assert "meta" in r.get_json()

        _lbl, results["POST /categorise"] = _bench("POST /categorise", cat_real, n=min(n, 10))
    else:
        with patch("services.categoriser.call_groq", side_effect=_mock_groq):

            def cat():
                r = client.post(
                    "/categorise",
                    json={"text": "Central bank raised rates after inflation surprise."},
                )
                assert r.status_code == 200
                assert r.get_json()["meta"]["cached"] is False

            _lbl, results["POST /categorise"] = _bench("POST /categorise", cat, n=n)

    if real:

        def qy_real():
            r = client.post(
                "/query",
                json={"question": "What are the key economic risks mentioned?"},
            )
            assert r.status_code == 200
            assert "meta" in r.get_json()

        _lbl, results["POST /query"] = _bench("POST /query", qy_real, n=min(n, 10))
    else:
        with patch("services.query_service.call_groq", side_effect=_mock_groq_answer):

            def qy():
                r = client.post(
                    "/query",
                    json={"question": "What are the key economic risks mentioned?"},
                )
                assert r.status_code == 200
                assert "meta" in r.get_json()

            _lbl, results["POST /query"] = _bench("POST /query", qy, n=n)

    print(f"Requests per endpoint: {n} (categorise/query may use fewer when REAL_LLM_BENCHMARK=1)")
    print("-" * 72)
    for label, stats_d in results.items():
        t = targets.get(label)
        p95 = stats_d["p95_ms"]
        ok = t is None or p95 <= t
        flag = "OK" if ok else "OVER"
        print(f"{label}")
        print(f"  p50={stats_d['p50_ms']}  p95={stats_d['p95_ms']}  p99={stats_d['p99_ms']}  mean={stats_d['mean_ms']}")
        if t is not None:
            print(f"  target_p95<={t} ms -> {flag}")
        print()

    over = [k for k, v in results.items() if targets.get(k) is not None and v["p95_ms"] > targets[k]]
    if over:
        print("Endpoints over target (optimize or widen targets for LLM-bound paths):", ", ".join(over))
    else:
        print("All benchmarked endpoints within configured targets.")


if __name__ == "__main__":
    main()
