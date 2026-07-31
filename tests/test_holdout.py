import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.discovery import load_reviews
from feedback_themes.domain import ContractError, Taxonomy
from feedback_themes.holdout import (
    build_annotation_template,
    load_excluded_ids,
    run_holdout_selection,
    select_holdout,
)

ROOT = Path(__file__).resolve().parents[1]


class SelectHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviews = load_reviews(ROOT / "data" / "reviews.json")

    def test_selection_is_deterministic_and_sized(self):
        first = select_holdout(self.reviews, 50, set())
        second = select_holdout(self.reviews, 50, set())
        self.assertEqual(50, len(first))
        self.assertEqual(
            [review["id"] for review in first],
            [review["id"] for review in second],
        )

    def test_selection_prefers_unseen_reviews(self):
        excluded = {
            review["id"]
            for review in self.reviews
            if review["rating"] == 5
        }
        excluded = set(sorted(excluded)[:40])
        holdout = select_holdout(self.reviews, 50, excluded)
        self.assertEqual(50, len(holdout))
        self.assertFalse(
            excluded & {review["id"] for review in holdout}
        )

    def test_selection_covers_every_rating(self):
        holdout = select_holdout(self.reviews, 50, set())
        ratings = {review["rating"] for review in holdout}
        self.assertEqual({1, 2, 3, 4, 5}, ratings)

    def test_backfills_depleted_rating_stratum(self):
        rating_three_ids = {
            review["id"]
            for review in self.reviews
            if review["rating"] == 3
        }
        holdout = select_holdout(self.reviews, 50, rating_three_ids)
        selected_threes = [
            review for review in holdout if review["rating"] == 3
        ]
        self.assertGreaterEqual(len(selected_threes), 4)
        self.assertTrue(
            all(
                review["id"] in rating_three_ids
                for review in selected_threes
            )
        )
        other_ratings = {
            review["rating"] for review in holdout if review["rating"] != 3
        }
        self.assertEqual({1, 2, 4, 5}, other_ratings)

    def test_quotas_follow_pool_proportions(self):
        holdout = select_holdout(self.reviews, 50, set())
        pool_share = sum(
            1 for review in self.reviews if review["rating"] == 1
        ) / len(self.reviews)
        holdout_share = sum(
            1 for review in holdout if review["rating"] == 1
        ) / len(holdout)
        self.assertAlmostEqual(pool_share, holdout_share, delta=0.05)

    def test_rejects_oversized_selection(self):
        with self.assertRaises(ValueError):
            select_holdout(self.reviews, len(self.reviews) + 1, set())

    def test_rejects_tiny_selection(self):
        with self.assertRaises(ValueError):
            select_holdout(self.reviews, 5, set())


class AnnotationTemplateTests(unittest.TestCase):
    def test_template_binds_taxonomy_and_starts_unannotated(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        reviews = load_reviews(ROOT / "data" / "reviews.json")[:12]
        template = build_annotation_template(reviews, taxonomy)

        self.assertEqual(taxonomy.content_hash, template["taxonomy_hash"])
        self.assertEqual(len(taxonomy.leaves), len(template["theme_reference"]))
        self.assertEqual(12, len(template["annotations"]))
        for entry in template["annotations"]:
            self.assertIsNone(entry["reference"]["specific_theme_ids"])
            self.assertIn("content_en", entry)


class RunHoldoutSelectionTests(unittest.TestCase):
    def test_writes_template_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            exclude = temporary / "discovery.json"
            exclude.write_text(
                json.dumps({"sample_review_ids": []}), encoding="utf-8"
            )
            arguments = {
                "reviews_path": ROOT / "data" / "reviews.json",
                "taxonomy_path": ROOT / "data" / "slice1_taxonomy.json",
                "output_path": temporary / "annotations.json",
                "metadata_output": temporary / "selection.json",
                "size": 50,
                "exclude_metadata": [exclude],
            }
            summary = run_holdout_selection(**arguments)
            template = json.loads(
                (temporary / "annotations.json").read_text("utf-8")
            )
            metadata = json.loads(
                (temporary / "selection.json").read_text("utf-8")
            )

            self.assertEqual(50, summary["holdout_size"])
            self.assertEqual(50, len(template["annotations"]))
            self.assertEqual(50, len(metadata["holdout_review_ids"]))
            with self.assertRaises(ContractError):
                run_holdout_selection(**arguments)
            run_holdout_selection(**arguments, force=True)

    def test_rejects_malformed_exclusion_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            bad = Path(temporary_directory) / "bad.json"
            bad.write_text(
                json.dumps({"sample_review_ids": [1, 2]}), encoding="utf-8"
            )
            with self.assertRaises(ContractError):
                load_excluded_ids([bad])


if __name__ == "__main__":
    unittest.main()
