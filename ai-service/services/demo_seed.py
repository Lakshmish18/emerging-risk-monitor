"""ChromaDB seed documents for demo / QA retrieval."""

from services.chroma_store import init_collection, upsert_texts


def seed_query_demo_collection() -> None:
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
