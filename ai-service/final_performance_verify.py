"""
Final performance checklist (manual + scripted hints):

1. Run mocked benchmark (no Groq cost):
     python benchmark_performance.py

2. Run with live LLM (optional, slower):
     set REAL_LLM_BENCHMARK=1
     python benchmark_performance.py

3. Confirm Redis cache:
     start Redis, set REDIS_URL, call POST /query twice same body;
     second response meta.cached should be true (unless fresh=1).

4. Confirm AI fallback:
     unset GROQ_API_KEY or block network; POST /categorise and /query;
     meta.is_fallback should be true and body uses template messaging.

5. Docker:
     docker build -t erm-ai-service -f Dockerfile .
     docker run --rm -e GROQ_API_KEY -e REDIS_URL -p 5000:5000 erm-ai-service
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    bench = os.path.join(root, "benchmark_performance.py")
    rc = subprocess.call([sys.executable, bench], cwd=root)
    if rc != 0:
        return rc
    print("\nBenchmark OK. Follow comments in this file for Redis/fallback/Docker checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
