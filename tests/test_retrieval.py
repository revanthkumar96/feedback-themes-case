import unittest
from pathlib import Path

from feedback_themes.domain import Taxonomy
from feedback_themes.retrieval import (
    FastEmbedThemeRetriever,
    RetrievalError,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingBackend:
    def passage_embed(self, texts):
        self.passages = list(texts)
        return [
            (
                [1.0, 0.0]
                if "Review authenticity" in text
                else [0.7, 0.7]
                if "Support responsiveness" in text
                else [0.0, 1.0]
            )
            for text in self.passages
        ]

    def query_embed(self, texts):
        self.queries = list(texts)
        return [[0.95, 0.05]]


class InvalidEmbeddingBackend(FakeEmbeddingBackend):
    def query_embed(self, texts):
        return []


class RetrievalTests(unittest.TestCase):
    def test_ranks_taxonomy_leaves_without_assigning_them(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        backend = FakeEmbeddingBackend()
        retriever = FastEmbedThemeRetriever(
            model_name="test-encoder",
            backend=backend,
        )

        rankings = retriever.retrieve(
            [
                {
                    "id": "review-1",
                    "title": "Questionable reviews",
                    "content_en": "The five-star reviews look manufactured.",
                }
            ],
            taxonomy,
            top_k=2,
        )

        self.assertEqual(
            ["review_authenticity", "support_responsiveness"],
            [
                candidate.specific_theme_id
                for candidate in rankings["review-1"]
            ],
        )
        self.assertEqual(10, len(backend.passages))
        self.assertIn("Questionable reviews", backend.queries[0])

    def test_rejects_invalid_top_k(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        retriever = FastEmbedThemeRetriever(
            model_name="test-encoder",
            backend=FakeEmbeddingBackend(),
        )

        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.retrieve([], taxonomy, top_k=0)

    def test_rejects_backend_result_count_mismatch(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        retriever = FastEmbedThemeRetriever(
            model_name="test-encoder",
            backend=InvalidEmbeddingBackend(),
        )

        with self.assertRaisesRegex(
            RetrievalError, "wrong number of review vectors"
        ):
            retriever.retrieve(
                [{"id": "review-1", "content_en": "A review."}],
                taxonomy,
                top_k=2,
            )
