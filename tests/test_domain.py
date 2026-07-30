import copy
import json
import unittest
from pathlib import Path

from feedback_themes.domain import (
    ContractError,
    Taxonomy,
    build_flat_projection,
    validate_model_output,
)


ROOT = Path(__file__).resolve().parents[1]


class TaxonomyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")

    def test_loads_unique_leaf_paths(self) -> None:
        path = self.taxonomy.leaves["portal_performance"]
        self.assertEqual("Digital experience", path.strategic_label)
        self.assertEqual("Online portal", path.midlevel_label)
        self.assertEqual("Portal performance", path.specific_label)

    def test_rejects_duplicate_ids_across_tiers(self) -> None:
        broken = copy.deepcopy(self.taxonomy.source)
        broken["strategic_themes"][0]["midlevel_themes"][0]["id"] = (
            "trust_and_transparency"
        )
        with self.assertRaisesRegex(ContractError, "duplicate theme id"):
            Taxonomy.from_dict(broken)

    def test_rejects_duplicate_labels_that_would_break_flat_tree(self) -> None:
        broken = copy.deepcopy(self.taxonomy.source)
        broken["strategic_themes"][1]["midlevel_themes"][0]["label"] = (
            broken["strategic_themes"][0]["midlevel_themes"][0]["label"]
        )
        with self.assertRaisesRegex(ContractError, "duplicate midlevel"):
            Taxonomy.from_dict(broken)

    def test_schema_restricts_leaf_ids_without_unsupported_array_keywords(self) -> None:
        schema = self.taxonomy.model_schema(3)
        results = schema["properties"]["results"]
        leaf_schema = (
            results["items"]["properties"]["assignments"]["items"]["properties"][
                "specific_theme_id"
            ]
        )
        self.assertEqual(sorted(self.taxonomy.leaves), leaf_schema["enum"])
        self.assertNotIn("minItems", results)
        self.assertNotIn("maxItems", results)

    def test_every_schema_object_is_closed_and_all_fields_are_required(self) -> None:
        schema = self.taxonomy.model_schema(3)

        def check(node):
            if not isinstance(node, dict):
                return
            if node.get("type") == "object":
                self.assertFalse(node.get("additionalProperties"))
                self.assertEqual(
                    set(node.get("properties", {})),
                    set(node.get("required", [])),
                )
            for value in node.values():
                if isinstance(value, dict):
                    check(value)
                elif isinstance(value, list):
                    for item in value:
                        check(item)

        check(schema)


class AssignmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.taxonomy = Taxonomy.load(ROOT / "data" / "slice1_taxonomy.json")
        self.reviews = [
            {
                "id": "review-1",
                "content_en": "The portal is slow and support does not answer.",
            },
            {"id": "review-2", "content_en": "Recommended."},
        ]

    def _valid_payload(self):
        return {
            "results": [
                {
                    "review_id": "review-1",
                    "assignments": [
                        {
                            "specific_theme_id": "portal_performance",
                            "evidence": "portal is slow",
                        },
                        {
                            "specific_theme_id": "support_responsiveness",
                            "evidence": "support does not answer",
                        },
                    ],
                    "no_assignment_reason": None,
                },
                {
                    "review_id": "review-2",
                    "assignments": [],
                    "no_assignment_reason": "no_relevant_theme",
                },
            ]
        }

    def test_validates_evidence_and_builds_deterministic_parents(self) -> None:
        results = validate_model_output(
            self._valid_payload(), self.reviews, self.taxonomy
        )
        rows = build_flat_projection(results, self.taxonomy)
        self.assertEqual(
            {
                "review_id": "review-1",
                "strategic_theme": "Digital experience",
                "midlevel_theme": "Online portal",
                "specific_theme": "Portal performance",
            },
            rows[0],
        )
        self.assertEqual(2, len(rows))

    def test_rejects_evidence_not_present_in_review(self) -> None:
        payload = self._valid_payload()
        payload["results"][0]["assignments"][0]["evidence"] = "very slow portal"
        with self.assertRaisesRegex(ContractError, "exact substring"):
            validate_model_output(payload, self.reviews, self.taxonomy)

    def test_canonicalizes_one_unambiguous_case_only_evidence_mismatch(self) -> None:
        payload = self._valid_payload()
        payload["results"][0]["assignments"][0]["evidence"] = "the portal is slow"
        results = validate_model_output(payload, self.reviews, self.taxonomy)
        self.assertEqual(
            "The portal is slow",
            results[0]["assignments"][0]["evidence"],
        )

    def test_rejects_missing_or_reordered_reviews(self) -> None:
        payload = self._valid_payload()
        payload["results"].reverse()
        with self.assertRaisesRegex(ContractError, "every requested review"):
            validate_model_output(payload, self.reviews, self.taxonomy)

    def test_derives_reason_for_empty_assignments(self) -> None:
        payload = self._valid_payload()
        payload["results"][1]["no_assignment_reason"] = None
        results = validate_model_output(payload, self.reviews, self.taxonomy)
        self.assertEqual(
            "no_relevant_theme", results[1]["no_assignment_reason"]
        )
