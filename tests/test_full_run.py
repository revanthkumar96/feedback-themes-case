import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.full_run import run_full_classification
from feedback_themes.groq import Completion, GroqError
from feedback_themes.retrieval import ThemeCandidate


ROOT = Path(__file__).resolve().parents[1]


class PromptAwareClassifier:
    model = "openai/gpt-oss-20b"
    reasoning_effort = "medium"
    max_completion_tokens = 4096
    rate_limit_retry_count = 0

    def __init__(self):
        self.calls = 0

    def classify(self, prompt, schema):
        self.calls += 1
        request = json.loads(prompt.split("\n\nYour previous")[0])
        results = []
        for review in request["reviews"]:
            if "five-star reviews" in review["content_en"]:
                assignments = [
                    {
                        "specific_theme_id": "review_authenticity",
                        "evidence": "five-star reviews",
                    }
                ]
                reason = None
            else:
                assignments = []
                reason = "no_relevant_theme"
            results.append(
                {
                    "review_id": review["review_id"],
                    "assignments": assignments,
                    "no_assignment_reason": reason,
                }
            )
        return Completion(
            content=json.dumps({"results": results}),
            model=self.model,
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )


class GenerationRetryClassifier(PromptAwareClassifier):
    def classify(self, prompt, schema):
        if self.calls == 0:
            self.calls += 1
            raise GroqError(
                "strict generation failed",
                status_code=400,
                error_code="json_validate_failed",
            )
        return super().classify(prompt, schema)


class FakeRetriever:
    model_name = "test-small-encoder"

    def retrieve(self, reviews, taxonomy, *, top_k):
        self.top_k = top_k
        return {
            review["id"]: [
                ThemeCandidate("review_authenticity", 0.8),
                ThemeCandidate("fee_transparency", 0.4),
            ]
            for review in reviews
        }


class FallbackAwareClassifier(PromptAwareClassifier):
    def classify(self, prompt, schema):
        self.calls += 1
        request = json.loads(prompt.split("\n\nYour previous")[0])
        results = []
        for review in request["reviews"]:
            assignments = []
            if (
                "candidate_specific_theme_ids" not in review
                and "portal is slow" in review["content_en"]
            ):
                assignments = [
                    {
                        "specific_theme_id": "portal_performance",
                        "evidence": "portal is slow",
                    }
                ]
            results.append(
                {
                    "review_id": review["review_id"],
                    "assignments": assignments,
                }
            )
        return Completion(
            content=json.dumps({"results": results}),
            model=self.model,
            usage={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )


class FullRunTests(unittest.TestCase):
    def test_hybrid_retrieval_audits_abstentions_with_full_taxonomy(self):
        classifier = FallbackAwareClassifier()
        retriever = FakeRetriever()
        reviews = [
            {
                "id": "review-1",
                "rating": 1,
                "title": "",
                "content_en": "The portal is slow.",
            },
            {
                "id": "review-2",
                "rating": 5,
                "title": "",
                "content_en": "Recommended.",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            reviews_path = temporary / "reviews.json"
            reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
            summary = run_full_classification(
                reviews_path=reviews_path,
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_dir=temporary / "out",
                batch_size=2,
                classifier=classifier,
                retriever=retriever,
                retrieval_top_k=2,
                progress=lambda _: None,
            )
            rich = json.loads(Path(summary["results_path"]).read_text("utf-8"))

        self.assertEqual(2, classifier.calls)
        self.assertEqual(2, summary["fallback_review_count"])
        self.assertEqual(1, summary["fallback_recovered_count"])
        self.assertEqual(1, summary["assignment_count"])
        self.assertEqual(
            "embedding-assisted-frozen-taxonomy-classification",
            rich["run"]["pipeline"],
        )
        self.assertEqual(
            "test-small-encoder",
            rich["run"]["retrieval"]["model"],
        )
        self.assertTrue(
            rich["review_results"][0]["routing"]["fallback_used"]
        )
        self.assertEqual(
            "review_authenticity",
            rich["review_results"][0]["routing"][
                "semantic_candidates"
            ][0]["specific_theme_id"],
        )

    def test_batches_all_reviews_and_writes_both_outputs(self):
        classifier = PromptAwareClassifier()
        progress = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            summary = run_full_classification(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_dir=temporary / "out",
                batch_size=30,
                classifier=classifier,
                progress=progress.append,
                checkpoint_dir=temporary / "checkpoints",
            )
            rich = json.loads(Path(summary["results_path"]).read_text("utf-8"))
            flat = json.loads(Path(summary["flat_path"]).read_text("utf-8"))

        self.assertEqual(223, summary["review_count"])
        self.assertEqual(8, summary["batch_count"])
        self.assertEqual(8, classifier.calls)
        self.assertEqual(8, len(progress))
        self.assertEqual(800, rich["run"]["usage"]["input_tokens"])
        self.assertEqual("Review authenticity", flat[0]["specific_theme"])
        self.assertEqual(223, len(rich["review_results"]))

    def test_resumes_only_matching_checkpoints_without_model_calls(self):
        first_classifier = PromptAwareClassifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            arguments = {
                "reviews_path": ROOT / "data" / "reviews.json",
                "taxonomy_path": ROOT / "data" / "slice1_taxonomy.json",
                "output_dir": temporary / "out",
                "batch_size": 30,
                "progress": lambda _: None,
                "checkpoint_dir": temporary / "checkpoints",
            }
            run_full_classification(
                classifier=first_classifier,
                **arguments,
            )
            resumed_classifier = PromptAwareClassifier()
            summary = run_full_classification(
                classifier=resumed_classifier,
                resume=True,
                **arguments,
            )

        self.assertEqual(0, resumed_classifier.calls)
        self.assertEqual(223, summary["review_count"])

    def test_resume_recomputes_stale_checkpoint_instead_of_failing(self):
        first_classifier = PromptAwareClassifier()
        progress: list[str] = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            arguments = {
                "reviews_path": ROOT / "data" / "reviews.json",
                "taxonomy_path": ROOT / "data" / "slice1_taxonomy.json",
                "output_dir": temporary / "out",
                "batch_size": 30,
                "checkpoint_dir": temporary / "checkpoints",
            }
            run_full_classification(
                classifier=first_classifier,
                progress=lambda _: None,
                **arguments,
            )
            stale_path = temporary / "checkpoints" / "batch-003.json"
            stale = json.loads(stale_path.read_text("utf-8"))
            stale["identity"]["prompt_version"] = "some-older-prompt"
            stale_path.write_text(json.dumps(stale), "utf-8")

            resumed_classifier = PromptAwareClassifier()
            summary = run_full_classification(
                classifier=resumed_classifier,
                resume=True,
                progress=progress.append,
                **arguments,
            )

        self.assertEqual(1, resumed_classifier.calls)
        self.assertEqual(223, summary["review_count"])
        self.assertTrue(
            any("recomputing" in message for message in progress)
        )

    def test_subset_classifies_only_requested_reviews(self):
        classifier = PromptAwareClassifier()
        all_reviews = json.loads(
            (ROOT / "data" / "reviews.json").read_text("utf-8")
        )
        subset_ids = {review["id"] for review in all_reviews[:7]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = run_full_classification(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_dir=Path(temporary_directory) / "out",
                batch_size=10,
                classifier=classifier,
                progress=lambda _: None,
                review_ids=subset_ids,
            )
            rich = json.loads(Path(summary["results_path"]).read_text("utf-8"))

        self.assertEqual(7, summary["review_count"])
        self.assertEqual(1, summary["batch_count"])
        self.assertEqual(7, rich["run"]["review_subset_size"])
        self.assertEqual(
            subset_ids,
            {result["review_id"] for result in rich["review_results"]},
        )

    def test_subset_with_unknown_id_is_rejected(self):
        from feedback_themes.domain import ContractError

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ContractError):
                run_full_classification(
                    reviews_path=ROOT / "data" / "reviews.json",
                    taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                    output_dir=Path(temporary_directory) / "out",
                    batch_size=10,
                    classifier=PromptAwareClassifier(),
                    progress=lambda _: None,
                    review_ids={"rev-does-not-exist"},
                )

    def test_retries_provider_schema_generation_failure(self):
        classifier = GenerationRetryClassifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = run_full_classification(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_dir=Path(temporary_directory) / "out",
                batch_size=30,
                classifier=classifier,
                progress=lambda _: None,
            )

        self.assertEqual(1, summary["generation_retries"])
        self.assertEqual(9, classifier.calls)
