import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.full_run import run_full_classification
from feedback_themes.groq import Completion, GroqError


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


class FullRunTests(unittest.TestCase):
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
