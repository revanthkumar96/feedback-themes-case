import json
import tempfile
import unittest
from pathlib import Path

from feedback_themes.domain import ContractError, Taxonomy
from feedback_themes.evaluation import (
    compare_runs,
    evaluate,
    load_annotations,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]

REVIEWS = [
    {
        "id": "review-a",
        "rating": 1,
        "title": "",
        "content_en": "The five-star reviews look fake.",
    },
    {
        "id": "review-b",
        "rating": 2,
        "title": "",
        "content_en": "Hidden fees everywhere and the portal is confusing.",
    },
    {
        "id": "review-c",
        "rating": 5,
        "title": "",
        "content_en": "Great!",
    },
    {
        "id": "review-d",
        "rating": 3,
        "title": "",
        "content_en": "Just a question about opening hours.",
    },
]

REFERENCES = {
    "review-a": {"review_authenticity"},
    "review-b": {"fee_transparency", "portal_usability"},
    "review-c": set(),
    "review-d": set(),
}

PREDICTIONS = {
    "review-a": [
        {
            "specific_theme_id": "review_authenticity",
            "evidence": "five-star reviews look fake",
        }
    ],
    "review-b": [
        {
            "specific_theme_id": "fee_transparency",
            "evidence": "Hidden fees",
        }
    ],
    "review-c": [],
    "review-d": [
        {
            "specific_theme_id": "payout_timing",
            "evidence": "text that is not in the review",
        }
    ],
}


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        cls.content_by_id = {
            review["id"]: review["content_en"] for review in REVIEWS
        }

    def _report(self):
        return evaluate(
            references=REFERENCES,
            predictions=PREDICTIONS,
            taxonomy=self.taxonomy,
            content_by_id=self.content_by_id,
        )

    def test_micro_precision_recall_f1(self):
        report = self._report()
        self.assertEqual(
            {"true_positive": 2, "false_positive": 1, "false_negative": 1},
            report["assignment_counts"],
        )
        self.assertAlmostEqual(2 / 3, report["precision"])
        self.assertAlmostEqual(2 / 3, report["recall"])
        self.assertAlmostEqual(2 / 3, report["micro_f1"])

    def test_exact_match_and_multi_subject_recall(self):
        report = self._report()
        self.assertAlmostEqual(0.5, report["exact_match_rate"])
        self.assertEqual(1, report["multi_subject"]["review_count"])
        self.assertAlmostEqual(0.5, report["multi_subject"]["recall"])

    def test_abstention_metrics(self):
        report = self._report()
        self.assertEqual(2, report["abstention"]["reference_count"])
        self.assertEqual(1, report["abstention"]["predicted_count"])
        self.assertAlmostEqual(1.0, report["abstention"]["precision"])
        self.assertAlmostEqual(0.5, report["abstention"]["recall"])

    def test_evidence_validity_and_unsupported_rate(self):
        report = self._report()
        self.assertAlmostEqual(2 / 3, report["evidence_validity_rate"])
        self.assertAlmostEqual(1 / 3, report["unsupported_assignment_rate"])
        self.assertTrue(report["hierarchy"]["valid"])

    def test_unknown_predicted_theme_fails_hierarchy(self):
        predictions = dict(PREDICTIONS)
        predictions["review-c"] = [
            {"specific_theme_id": "made_up_theme", "evidence": "Great!"}
        ]
        report = evaluate(
            references=REFERENCES,
            predictions=predictions,
            taxonomy=self.taxonomy,
            content_by_id=self.content_by_id,
        )
        self.assertFalse(report["hierarchy"]["valid"])
        self.assertEqual(
            ["made_up_theme"], report["hierarchy"]["unknown_theme_ids"]
        )

    def test_missing_annotated_review_is_rejected(self):
        predictions = {
            review_id: rows
            for review_id, rows in PREDICTIONS.items()
            if review_id != "review-b"
        }
        with self.assertRaises(ContractError):
            evaluate(
                references=REFERENCES,
                predictions=predictions,
                taxonomy=self.taxonomy,
                content_by_id=self.content_by_id,
            )


class LoadAnnotationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")

    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        return handle.name

    def test_rejects_unannotated_entries(self):
        path = self._write(
            {
                "taxonomy_hash": self.taxonomy.content_hash,
                "annotations": [
                    {
                        "review_id": "review-a",
                        "reference": {"specific_theme_ids": None},
                    }
                ],
            }
        )
        with self.assertRaises(ContractError) as context:
            load_annotations(path, self.taxonomy, {"review-a"})
        self.assertIn("not annotated", str(context.exception))

    def test_rejects_taxonomy_hash_mismatch(self):
        path = self._write({"taxonomy_hash": "stale", "annotations": []})
        with self.assertRaises(ContractError):
            load_annotations(path, self.taxonomy, {"review-a"})

    def test_rejects_unknown_theme_ids(self):
        path = self._write(
            {
                "taxonomy_hash": self.taxonomy.content_hash,
                "annotations": [
                    {
                        "review_id": "review-a",
                        "reference": {"specific_theme_ids": ["nope"]},
                    }
                ],
            }
        )
        with self.assertRaises(ContractError):
            load_annotations(path, self.taxonomy, {"review-a"})

    def test_accepts_complete_annotations(self):
        path = self._write(
            {
                "taxonomy_hash": self.taxonomy.content_hash,
                "annotations": [
                    {
                        "review_id": "review-a",
                        "reference": {
                            "specific_theme_ids": ["review_authenticity"]
                        },
                    },
                    {
                        "review_id": "review-c",
                        "reference": {"specific_theme_ids": []},
                    },
                ],
            }
        )
        references = load_annotations(
            path, self.taxonomy, {"review-a", "review-c"}
        )
        self.assertEqual(
            {"review-a": {"review_authenticity"}, "review-c": set()},
            references,
        )


class CompareRunsTests(unittest.TestCase):
    def test_identical_runs_score_one(self):
        stability = compare_runs(PREDICTIONS, PREDICTIONS)
        self.assertEqual(4, stability["shared_review_count"])
        self.assertAlmostEqual(1.0, stability["identical_assignment_rate"])
        self.assertAlmostEqual(1.0, stability["mean_jaccard"])

    def test_partial_disagreement(self):
        other = dict(PREDICTIONS)
        other["review-a"] = []
        stability = compare_runs(PREDICTIONS, other)
        self.assertAlmostEqual(0.75, stability["identical_assignment_rate"])
        self.assertAlmostEqual(0.75, stability["mean_jaccard"])


class RunEvaluationTests(unittest.TestCase):
    def test_end_to_end_report(self):
        taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            (temporary / "reviews.json").write_text(
                json.dumps(REVIEWS), encoding="utf-8"
            )
            (temporary / "annotations.json").write_text(
                json.dumps(
                    {
                        "taxonomy_hash": taxonomy.content_hash,
                        "annotations": [
                            {
                                "review_id": review_id,
                                "reference": {
                                    "specific_theme_ids": sorted(reference)
                                },
                            }
                            for review_id, reference in REFERENCES.items()
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (temporary / "results.json").write_text(
                json.dumps(
                    {
                        "review_results": [
                            {
                                "review_id": review_id,
                                "assignments": assignments,
                            }
                            for review_id, assignments in PREDICTIONS.items()
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = run_evaluation(
                annotations_path=temporary / "annotations.json",
                results_path=temporary / "results.json",
                reviews_path=temporary / "reviews.json",
                taxonomy_path=ROOT / "data" / "slice1_taxonomy.json",
                output_path=temporary / "evaluation.json",
                baseline_results_path=temporary / "results.json",
            )
            written = json.loads(
                (temporary / "evaluation.json").read_text("utf-8")
            )

        self.assertAlmostEqual(2 / 3, report["micro_f1"])
        self.assertAlmostEqual(1.0, report["stability"]["mean_jaccard"])
        self.assertEqual(report["taxonomy_hash"], written["taxonomy_hash"])


if __name__ == "__main__":
    unittest.main()
