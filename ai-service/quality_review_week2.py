"""
Week 2 AI quality review: 10 fresh categorise + 10 fresh query inputs.
Scores 1-5 per item; target average >= 4.0/5.0 on weighted accuracy + format.

Run with GROQ_API_KEY set for live scoring; otherwise exits with notice.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from services.categoriser import PREDEFINED_CATEGORIES, categorise_text
from services.demo_seed import seed_query_demo_collection
from services.query_service import answer_query

# 10 hold-out classification cases (not in original prompt_eval set).
CAT_SAMPLES = [
    ("Emergency fiscal package cleared parliament after currency pressure.", "economic"),
    ("Mayoral runoff debate drew record turnout in metro districts.", "political"),
    ("APT campaign leveraged stolen certs against VPN gateways.", "security"),
    ("Open-source LLM runtime patched for side-channel hardening.", "technology"),
    ("Monsoon drainage upgrades prioritized low-income wards.", "climate"),
    ("ICU capacity triggers activated amid respiratory surge.", "health"),
    ("Tunnel ventilation outage halted commuter rail briefly.", "infrastructure"),
    ("Volunteer networks coordinated shelter staffing shortages.", "social"),
    ("Appeals court narrowed scope of warrantless device searches.", "legal"),
    ("Newsletter roundup without a dominant storyline.", "other"),
]

# 10 fresh query prompts aligned with seeded corpus (see demo_seed).
QUERY_SAMPLES = [
    ("What happened along the coast after the cyclone?", "cyclone", "coastal"),
    ("Why were benchmark rates increased?", "rates", "inflation"),
    ("What kind of cyber incident affected hospitals?", "ransomware", "encrypted"),
    ("What parliamentary action was taken?", "procurement", "Parliament"),
    ("Describe drought effects on agriculture.", "drought", "crop"),
    ("What changed at the port for containers?", "automation", "turnaround"),
    ("Why were buses disrupted in the capital?", "strike", "bus"),
    ("What happened with facial recognition rules?", "court", "suspended"),
    ("How was disease surge addressed?", "measles", "vaccination"),
    ("How was internet capacity restored?", "cable", "outage"),
]


def _score_categorise(text: str, expected: str) -> tuple[int, str]:
    r = categorise_text(text)
    cat = r.get("category")
    conf = float(r.get("confidence") or 0.0)
    ok = cat == expected
    if ok and conf >= 0.6:
        s = 5
    elif ok:
        s = 4
    elif cat in PREDEFINED_CATEGORIES and not ok:
        s = 2
    else:
        s = 1
    note = f"expected={expected} got={cat} conf={conf:.2f}"
    return s, note


def _score_query(question: str, k1: str, k2: str) -> tuple[int, str]:
    r = answer_query(question, top_k=3, skip_cache=True)
    ans = (r.get("answer") or "").lower()
    hit = (k1.lower() in ans) or (k2.lower() in ans)
    s = 5 if hit and len(ans) > 40 else (4 if hit else 2)
    return s, f"keywords={k1},{k2} len={len(ans)}"


def main() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY to run live Week 2 quality review.")
        return 1

    seed_query_demo_collection()

    cat_scores = []
    for text, exp in CAT_SAMPLES:
        s, _ = _score_categorise(text, exp)
        cat_scores.append(s)

    q_scores = []
    for q, k1, k2 in QUERY_SAMPLES:
        s, _ = _score_query(q, k1, k2)
        q_scores.append(s)

    all_s = cat_scores + q_scores
    avg = sum(all_s) / len(all_s)
    print("Week 2 quality review (1-5 per item, 20 items)")
    print(f"  categorise avg: {sum(cat_scores)/len(cat_scores):.2f}/5.0")
    print(f"  query      avg: {sum(q_scores)/len(q_scores):.2f}/5.0")
    print(f"  overall    avg: {avg:.2f}/5.0  (target >= 4.0)")

    if avg < 4.0:
        print("  Below target: tighten categoriser disambiguation prompts and query grounding instructions.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
