import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.groq import Completion
from feedback_themes.domain import Taxonomy
from feedback_themes.pipeline import PROMPT_VERSION, build_prompt, run_slice1


ROOT = Path(__file__).resolve().parents[1]


class FakeClassifier:
    model = "openai/gpt-oss-20b"
    reasoning_effort = "low"
    max_completion_tokens = 2048

    def classify(self, prompt, schema):
        self.prompt = prompt
        self.schema = schema
        return Completion(
            content=json.dumps(
                {
                    "results": [
                        {
                            "review_id": "rev-76c3011b2ca9",
                            "assignments": [
                                {
                                    "specific_theme_id": "review_authenticity",
                                    "evidence": "five-star reviews arrived in the same week",
                                }
                            ],
                            "no_assignment_reason": None,
                        },
                        {
                            "review_id": "rev-f04f75b142cc",
                            "assignments": [],
                            "no_assignment_reason": "no_relevant_theme",
                        },
                    ]
                }
            ),
            model=self.model,
            usage={
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
            },
        )


class PipelineTests(unittest.TestCase):
    def test_hybrid_prompt_keeps_full_taxonomy_and_adds_ranked_candidates(
        self,
    ) -> None:
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        reviews = [
            {
                "id": "review-1",
                "title": "",
                "content_en": "The portal is slow.",
            },
            {
                "id": "review-2",
                "title": "",
                "content_en": "Support never answered.",
            },
        ]

        prompt = json.loads(
            build_prompt(
                reviews,
                taxonomy,
                {
                    "review-1": ["portal_performance", "portal_usability"],
                    "review-2": ["support_responsiveness"],
                },
            )
        )

        self.assertEqual(10, len(prompt["taxonomy"]))
        self.assertEqual(
            ["portal_performance", "portal_usability"],
            prompt["reviews"][0]["candidate_specific_theme_ids"],
        )

    def test_end_to_end_writes_rich_and_flat_outputs(self) -> None:
        classifier = FakeClassifier()
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = run_slice1(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_dir=temporary_directory,
                limit=2,
                classifier=classifier,
            )
            rich = json.loads(Path(summary["results_path"]).read_text("utf-8"))
            flat = json.loads(Path(summary["flat_path"]).read_text("utf-8"))

        self.assertEqual(2, summary["review_count"])
        self.assertEqual(1, summary["assignment_count"])
        self.assertEqual(PROMPT_VERSION, rich["run"]["prompt_version"])
        self.assertEqual("low", rich["run"]["reasoning_effort"])
        self.assertEqual(2048, rich["run"]["max_completion_tokens"])
        self.assertEqual(0.000105, rich["run"]["estimated_cost_usd"])
        self.assertEqual("Review authenticity", flat[0]["specific_theme"])
        self.assertNotIn("rating", classifier.prompt)
        self.assertNotIn(
            "minItems", classifier.schema["properties"]["results"]
        )
