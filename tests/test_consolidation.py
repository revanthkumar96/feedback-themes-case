import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.consolidation import (
    build_consolidation_prompt,
    run_consolidation,
)
from feedback_themes.domain import Taxonomy
from feedback_themes.groq import Completion


ROOT = Path(__file__).resolve().parents[1]


class FakeConsolidator:
    model = "openai/gpt-oss-120b"
    reasoning_effort = "low"
    max_completion_tokens = 5000

    def classify(self, prompt, schema):
        self.prompt = prompt
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


class ConsolidationTests(unittest.TestCase):
    def test_prompt_uses_compact_leaf_paths_and_polarity_rule(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        prompt = build_consolidation_prompt([taxonomy, taxonomy])
        self.assertIn("Merge opposite states", prompt)
        self.assertIn("collections handling", prompt)
        self.assertNotIn('"version": "slice1-v1"', prompt)

    def test_consolidation_writes_valid_frozen_taxonomy(self):
        generator = FakeConsolidator()
        candidate = ROOT / "data" / "slice1_taxonomy.json"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            summary = run_consolidation(
                candidate_paths=[candidate, candidate],
                taxonomy_output=temporary / "themes.json",
                metadata_output=temporary / "consolidation_run.json",
                generator=generator,
            )
            taxonomy = Taxonomy.load(summary["taxonomy_path"])
            metadata = json.loads(
                Path(summary["metadata_path"]).read_text("utf-8")
            )

        self.assertEqual(10, len(taxonomy.leaves))
        self.assertEqual("openai/gpt-oss-120b", metadata["model"])
        self.assertEqual(2, len(metadata["candidate_taxonomy_hashes"]))
