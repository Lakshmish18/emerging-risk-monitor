import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from services.categoriser import PREDEFINED_CATEGORIES, categorise_text
from services.chroma_store import init_collection, upsert_texts
from services.query_service import answer_query


@dataclass
class EvalResult:
    accuracy_score: float
    format_score: float
    total_score: float
    details: List[str]


def evaluate_categoriser() -> EvalResult:
    samples: List[Tuple[str, str]] = [
        ("Parliament passed emergency election reforms after coalition talks collapsed.", "political"),
        ("Fuel prices surged as shipping costs increased across key trade routes.", "economic"),
        ("A malware attack disrupted payment systems at two national banks.", "security"),
        ("Telecoms rolled out 6G pilot towers with AI-managed network optimization.", "technology"),
        ("A heatwave and wildfire forced evacuation of three districts.", "climate"),
        ("Hospitals reported a spike in dengue cases after heavy rain.", "health"),
        ("Bridge corrosion shut down a major freight corridor for repairs.", "infrastructure"),
        ("Large protests erupted over rising food prices in urban centers.", "social"),
        ("Supreme court struck down a controversial surveillance law.", "legal"),
        ("The article discusses mixed developments across several sectors.", "other"),
    ]

    correct = 0
    valid_format = 0
    details: List[str] = []

    for idx, (text, expected_category) in enumerate(samples, start=1):
        result = categorise_text(text)
        category = result.get("category")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning")

        is_correct = category == expected_category
        has_valid_format = (
            isinstance(category, str)
            and category in PREDEFINED_CATEGORIES
            and isinstance(confidence, (int, float))
            and 0.0 <= float(confidence) <= 1.0
            and isinstance(reasoning, str)
            and len(reasoning.strip()) > 0
        )

        correct += int(is_correct)
        valid_format += int(has_valid_format)

        details.append(
            f"[categoriser #{idx}] expected={expected_category} got={category} "
            f"correct={is_correct} format_ok={has_valid_format}"
        )

    accuracy_score = round((correct / len(samples)) * 10, 2)
    format_score = round((valid_format / len(samples)) * 10, 2)
    total_score = round((accuracy_score + format_score) / 2, 2)
    return EvalResult(accuracy_score, format_score, total_score, details)


def _seed_query_collection() -> None:
    collection = init_collection()
    docs = [
        "Cyclone Nivar damaged coastal substations and flooded roads in the delta region.",
        "Central bank increased benchmark rates by 50 basis points to curb inflation.",
        "A ransomware group encrypted records at a major city hospital network.",
        "Parliament passed emergency procurement powers after weeks of debate.",
        "A prolonged drought reduced reservoir levels and crop output in the north.",
        "Port crane automation improved turnaround times for container shipments.",
        "A new labor strike disrupted bus services in the capital for three days.",
        "The high court suspended implementation of a facial-recognition policy.",
        "A measles outbreak prompted emergency vaccination drives in two provinces.",
        "Undersea cable repairs restored internet capacity after a regional outage.",
    ]
    ids = [f"seed-{i}" for i in range(1, len(docs) + 1)]
    metadatas = [{"source": f"report-{i}"} for i in range(1, len(docs) + 1)]
    upsert_texts(collection=collection, ids=ids, documents=docs, metadatas=metadatas)


def evaluate_query_prompt() -> EvalResult:
    _seed_query_collection()
    samples: List[Tuple[str, str]] = [
        ("What happened to coastal power and roads after the cyclone?", "seed-1"),
        ("Why did the central bank raise rates?", "seed-2"),
        ("Which sector was impacted by ransomware?", "seed-3"),
        ("What legal action was taken on facial-recognition policy?", "seed-8"),
        ("How was internet capacity restored after outage?", "seed-10"),
        ("What disrupted bus services in the capital?", "seed-7"),
        ("What public health response followed the measles outbreak?", "seed-9"),
        ("What climate impact affected crop output in the north?", "seed-5"),
        ("What improved container shipment turnaround?", "seed-6"),
        ("What political step followed parliamentary debate?", "seed-4"),
    ]

    correct = 0
    valid_format = 0
    details: List[str] = []

    for idx, (question, expected_source_id) in enumerate(samples, start=1):
        result = answer_query(question=question, top_k=3)
        answer = result.get("answer")
        sources = result.get("sources")

        source_ids = []
        if isinstance(sources, list):
            source_ids = [s.get("id") for s in sources if isinstance(s, dict)]

        source_hit = expected_source_id in source_ids
        answer_ok = isinstance(answer, str) and len(answer.strip()) > 0
        has_valid_format = isinstance(sources, list) and len(sources) <= 3 and answer_ok

        # Accuracy proxy: expected doc appears in retrieved sources and answer is non-empty.
        is_correct = source_hit and answer_ok

        correct += int(is_correct)
        valid_format += int(has_valid_format)

        details.append(
            f"[query #{idx}] expected_source={expected_source_id} got_sources={source_ids} "
            f"correct={is_correct} format_ok={has_valid_format}"
        )

    accuracy_score = round((correct / len(samples)) * 10, 2)
    format_score = round((valid_format / len(samples)) * 10, 2)
    total_score = round((accuracy_score + format_score) / 2, 2)
    return EvalResult(accuracy_score, format_score, total_score, details)


def main() -> int:
    categoriser = evaluate_categoriser()
    query = evaluate_query_prompt()

    print("Prompt Evaluation Results")
    print("=" * 80)
    print(
        f"categoriser -> accuracy={categoriser.accuracy_score}/10 "
        f"format={categoriser.format_score}/10 total={categoriser.total_score}/10"
    )
    print(
        f"query       -> accuracy={query.accuracy_score}/10 "
        f"format={query.format_score}/10 total={query.total_score}/10"
    )
    print("-" * 80)

    for line in categoriser.details:
        print(line)
    for line in query.details:
        print(line)

    # Non-zero exit if any prompt falls below threshold.
    if categoriser.total_score < 7.0 or query.total_score < 7.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
