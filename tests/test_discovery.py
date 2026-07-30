import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.discovery import (
    run_discovery,
    select_stratified_sample,
    taxonomy_schema,
)
from feedback_themes.groq import Completion


ROOT = Path(__file__).resolve().parents[1]


class FakeGenerator:
    model = "openai/gpt-oss-20b"
    reasoning_effort = "medium"
    max_completion_tokens = 4096

    def classify(self, prompt, schema):
        self.prompt = prompt
        self.schema = schema
        taxonomy = json.loads(
            (ROOT / "data" / "slice1_taxonomy.json").read_text("utf-8")
        )
        return Completion(
            content=json.dumps(
                {"strategic_themes": taxonomy["strategic_themes"]}
            ),
            model=self.model,
            usage={
                "input_tokens": 2000,
                "output_tokens": 1000,
                "total_tokens": 3000,
            },
        )


class CorrectingGenerator(FakeGenerator):
    def __init__(self):
        self.calls = 0

    def classify(self, prompt, schema):
        self.calls += 1
        completion = super().classify(prompt, schema)
        if self.calls == 1:
            payload = json.loads(completion.content)
            payload["strategic_themes"][0]["id"] = "pålitelighet"
            return Completion(
                content=json.dumps(payload),
                model=completion.model,
                usage=completion.usage,
            )
        return completion


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.reviews = json.loads(
            (ROOT / "data" / "reviews.json").read_text("utf-8")
        )

    def test_sample_is_deterministic_and_rating_stratified(self):
        first = select_stratified_sample(self.reviews, 40)
        second = select_stratified_sample(self.reviews, 40)
        holdout = select_stratified_sample(self.reviews, 40, phase=1)
        self.assertEqual(
            [review["id"] for review in first],
            [review["id"] for review in second],
        )
        counts = {
            rating: sum(review["rating"] == rating for review in first)
            for rating in range(1, 6)
        }
        self.assertEqual({1: 8, 2: 8, 3: 8, 4: 8, 5: 8}, counts)
        self.assertTrue(
            {review["id"] for review in first}.isdisjoint(
                review["id"] for review in holdout
            )
        )

    def test_schema_closes_every_object(self):
        def check(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                self.assertFalse(node.get("additionalProperties"))
                self.assertEqual(
                    set(node["properties"]), set(node["required"])
                )
            for value in node.values():
                if isinstance(value, dict):
                    check(value)
                elif isinstance(value, list):
                    for item in value:
                        check(item)

        check(taxonomy_schema())

    def test_discovery_writes_valid_taxonomy_and_metadata(self):
        generator = FakeGenerator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            summary = run_discovery(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_output=temporary / "themes.json",
                metadata_output=temporary / "taxonomy_run.json",
                sample_size=50,
                sample_phase=0,
                generator=generator,
            )
            taxonomy = json.loads(
                Path(summary["taxonomy_path"]).read_text("utf-8")
            )
            metadata = json.loads(
                Path(summary["metadata_path"]).read_text("utf-8")
            )

        self.assertEqual("v1", taxonomy["version"])
        self.assertEqual(50, metadata["sample_size"])
        self.assertEqual(4, summary["strategic_count"])
        self.assertEqual(10, summary["specific_count"])
        self.assertIn("recurring subject", generator.prompt)

    def test_discovery_retries_one_semantically_invalid_taxonomy(self):
        generator = CorrectingGenerator()
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            summary = run_discovery(
                reviews_path=ROOT / "data" / "reviews.json",
                taxonomy_output=temporary / "themes.json",
                metadata_output=temporary / "taxonomy_run.json",
                sample_size=50,
                sample_phase=0,
                generator=generator,
            )
            metadata = json.loads(
                Path(summary["metadata_path"]).read_text("utf-8")
            )

        self.assertEqual(2, generator.calls)
        self.assertEqual(1, metadata["validation_retries"])
        self.assertEqual(4000, summary["usage"]["input_tokens"])
