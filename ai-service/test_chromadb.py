import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from services.chroma_store import init_collection, query_text, upsert_texts


class TestChromaPersistenceAndQuery(unittest.TestCase):
    def test_embed_and_query_returns_expected_document(self):
        tmp_dir = tempfile.mkdtemp()
        collection = None
        try:
            collection = init_collection(
                collection_name="test_risk_collection",
                persist_directory=tmp_dir,
            )

            upsert_texts(
                collection=collection,
                ids=["doc-1", "doc-2", "doc-3"],
                documents=[
                    "A severe cyclone disrupted coastal transport and power lines.",
                    "Central bank policy rates were raised after inflation data.",
                    "A ransomware incident impacted a major regional hospital.",
                ],
                metadatas=[
                    {"category": "climate"},
                    {"category": "economic"},
                    {"category": "security"},
                ],
            )

            result = query_text(
                collection=collection,
                query="storm damage to coastal infrastructure",
                n_results=1,
            )

            self.assertIn("ids", result)
            self.assertEqual(result["ids"][0][0], "doc-1")
        finally:
            # Release sqlite handles on Windows before deleting temp files.
            if collection is not None:
                del collection
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
