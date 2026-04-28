import os
from typing import Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DEFAULT_COLLECTION_NAME = "risk_documents"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def get_persistent_client(persist_directory: Optional[str] = None) -> chromadb.PersistentClient:
    persist_path = persist_directory or os.getenv("CHROMA_PERSIST_DIR") or "./chroma_data"
    os.makedirs(persist_path, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def init_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBED_MODEL,
) -> Collection:
    client = get_persistent_client(persist_directory=persist_directory)
    try:
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    except ValueError:
        # Fallback keeps retrieval functional when sentence-transformers is unavailable.
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )


def upsert_texts(
    collection: Collection,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
) -> None:
    if len(ids) != len(documents):
        raise ValueError("ids and documents must have the same length")
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("metadatas must match the number of documents")

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )


def query_text(collection: Collection, query: str, n_results: int = 1) -> Dict:
    return collection.query(
        query_texts=[query],
        n_results=n_results,
    )
