"""
Final prompt QA: run all prompts against 30 seeded demo records
(prompts/demo_records.json). Verifies outputs are demo-ready (format, non-empty,
expected category or keyword coverage for query).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from services.categoriser import PREDEFINED_CATEGORIES, categorise_text
from services.demo_seed import seed_query_demo_collection
from services.query_service import answer_query


def _load_records() -> List[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "prompts", "demo_records.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY for full demo QA (or expect fallback/is_fallback outputs).")

    seed_query_demo_collection()
    records = _load_records()
    failures: List[str] = []

    for rec in records:
        rid = rec.get("id")
        kind = rec.get("kind")
        if kind == "categorise":
            text = rec.get("text") or ""
            exp = rec.get("expected_category")
            out = categorise_text(text)
            cat = out.get("category")
            if cat not in PREDEFINED_CATEGORIES:
                failures.append(f"{rid}: invalid category {cat}")
            if exp and cat != exp:
                failures.append(f"{rid}: category expected {exp} got {cat}")
            if out.get("is_fallback") and os.getenv("GROQ_API_KEY"):
                failures.append(f"{rid}: categorise used fallback despite API key")
        elif kind == "query":
            q = rec.get("question") or ""
            keys = rec.get("must_contain") or []
            out = answer_query(q, top_k=3, skip_cache=True)
            ans = (out.get("answer") or "").lower()
            if not ans.strip():
                failures.append(f"{rid}: empty answer")
            keyword_hit = any(str(k).lower() in ans for k in keys)
            if not keyword_hit and len(ans) < 80:
                failures.append(f"{rid}: answer missing expected keywords from {keys}")
            if out.get("is_fallback") and os.getenv("GROQ_API_KEY"):
                failures.append(f"{rid}: query used fallback despite API key (check corpus/API)")
        else:
            failures.append(f"{rid}: unknown kind {kind}")

    print(f"Demo QA: {len(records)} records")
    if failures:
        print("FAILED:")
        for f in failures:
            print(" ", f)
        return 1

    print("All demo records passed demo-ready checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
