from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when taxonomy or model output violates the pipeline contract."""


def _required_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{location} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ThemePath:
    strategic_id: str
    strategic_label: str
    midlevel_id: str
    midlevel_label: str
    specific_id: str
    specific_label: str


@dataclass(frozen=True)
class Taxonomy:
    version: str
    source: dict[str, Any]
    leaves: dict[str, ThemePath]

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, data: Any) -> "Taxonomy":
        if not isinstance(data, dict):
            raise ContractError("taxonomy must be a JSON object")

        version = _required_text(data.get("version"), "taxonomy.version")
        strategic_themes = data.get("strategic_themes")
        if not isinstance(strategic_themes, list) or not strategic_themes:
            raise ContractError("taxonomy.strategic_themes must be a non-empty list")

        seen_ids: set[str] = set()
        seen_labels = {
            "strategic": set(),
            "midlevel": set(),
            "specific": set(),
        }
        leaves: dict[str, ThemePath] = {}

        def unique_id(value: Any, location: str) -> str:
            theme_id = _required_text(value, location)
            if theme_id in seen_ids:
                raise ContractError(f"duplicate theme id: {theme_id!r}")
            seen_ids.add(theme_id)
            return theme_id

        def unique_label(value: Any, tier: str, location: str) -> str:
            label = _required_text(value, location)
            normalized = label.casefold()
            if normalized in seen_labels[tier]:
                raise ContractError(f"duplicate {tier} theme label: {label!r}")
            seen_labels[tier].add(normalized)
            return label

        for strategic_index, strategic in enumerate(strategic_themes):
            strategic_location = f"strategic_themes[{strategic_index}]"
            if not isinstance(strategic, dict):
                raise ContractError(f"{strategic_location} must be an object")
            strategic_id = unique_id(strategic.get("id"), f"{strategic_location}.id")
            strategic_label = unique_label(
                strategic.get("label"),
                "strategic",
                f"{strategic_location}.label",
            )
            _required_text(strategic.get("definition"), f"{strategic_location}.definition")
            midlevels = strategic.get("midlevel_themes")
            if not isinstance(midlevels, list) or not midlevels:
                raise ContractError(
                    f"{strategic_location}.midlevel_themes must be a non-empty list"
                )

            for midlevel_index, midlevel in enumerate(midlevels):
                midlevel_location = (
                    f"{strategic_location}.midlevel_themes[{midlevel_index}]"
                )
                if not isinstance(midlevel, dict):
                    raise ContractError(f"{midlevel_location} must be an object")
                midlevel_id = unique_id(midlevel.get("id"), f"{midlevel_location}.id")
                midlevel_label = unique_label(
                    midlevel.get("label"),
                    "midlevel",
                    f"{midlevel_location}.label",
                )
                _required_text(
                    midlevel.get("definition"), f"{midlevel_location}.definition"
                )
                specifics = midlevel.get("specific_themes")
                if not isinstance(specifics, list) or not specifics:
                    raise ContractError(
                        f"{midlevel_location}.specific_themes must be a non-empty list"
                    )

                for specific_index, specific in enumerate(specifics):
                    specific_location = (
                        f"{midlevel_location}.specific_themes[{specific_index}]"
                    )
                    if not isinstance(specific, dict):
                        raise ContractError(f"{specific_location} must be an object")
                    specific_id = unique_id(
                        specific.get("id"), f"{specific_location}.id"
                    )
                    specific_label = unique_label(
                        specific.get("label"),
                        "specific",
                        f"{specific_location}.label",
                    )
                    _required_text(
                        specific.get("definition"), f"{specific_location}.definition"
                    )
                    leaves[specific_id] = ThemePath(
                        strategic_id=strategic_id,
                        strategic_label=strategic_label,
                        midlevel_id=midlevel_id,
                        midlevel_label=midlevel_label,
                        specific_id=specific_id,
                        specific_label=specific_label,
                    )

        return cls(version=version, source=data, leaves=leaves)

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(
            self.source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def model_schema(self, review_count: int) -> dict[str, Any]:
        """Strict JSON schema for one result per review.

        Review identity, evidence support, and assignment count are also checked
        semantically after parsing; JSON Schema alone cannot express all of them.
        """
        if review_count < 1:
            raise ValueError("review_count must be positive")
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": (
                        f"Exactly {review_count} results, one per supplied review, "
                        "in input order."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "review_id": {"type": "string"},
                            "assignments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "specific_theme_id": {
                                            "type": "string",
                                            "enum": sorted(self.leaves),
                                        },
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["specific_theme_id", "evidence"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["review_id", "assignments"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }


def validate_model_output(
    payload: Any,
    reviews: list[dict[str, Any]],
    taxonomy: Taxonomy,
) -> list[dict[str, Any]]:
    """Validate semantic constraints not guaranteed by JSON Schema."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ContractError("model output must contain a results list")

    expected_ids: list[str] = []
    content_by_id: dict[str, str] = {}
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            raise ContractError(f"reviews[{index}] must be an object")
        review_id = _required_text(review.get("id"), f"reviews[{index}].id")
        content = _required_text(
            review.get("content_en"), f"reviews[{index}].content_en"
        )
        if review_id in content_by_id:
            raise ContractError(f"duplicate input review id: {review_id!r}")
        expected_ids.append(review_id)
        content_by_id[review_id] = content

    results = payload["results"]
    actual_ids = [
        result.get("review_id") if isinstance(result, dict) else None
        for result in results
    ]
    if actual_ids != expected_ids:
        raise ContractError(
            "model results must contain every requested review exactly once "
            f"and in input order; expected {expected_ids!r}, got {actual_ids!r}"
        )

    validated: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        review_id = expected_ids[index]
        assignments = result.get("assignments")
        reason = result.get("no_assignment_reason")
        if not isinstance(assignments, list):
            raise ContractError(f"{review_id}: assignments must be a list")
        if len(assignments) > 5:
            raise ContractError(f"{review_id}: at most five assignments are allowed")
        if assignments and reason is not None:
            raise ContractError(
                f"{review_id}: no_assignment_reason must be null when themes exist"
            )
        if not assignments and reason not in {None, "no_relevant_theme"}:
            raise ContractError(
                f"{review_id}: invalid no-assignment reason {reason!r}"
            )

        seen_leaf_ids: set[str] = set()
        normalized_assignments: list[dict[str, str]] = []
        for assignment_index, assignment in enumerate(assignments):
            location = f"{review_id}.assignments[{assignment_index}]"
            if not isinstance(assignment, dict):
                raise ContractError(f"{location} must be an object")
            leaf_id = _required_text(
                assignment.get("specific_theme_id"),
                f"{location}.specific_theme_id",
            )
            evidence = _required_text(
                assignment.get("evidence"), f"{location}.evidence"
            )
            if leaf_id not in taxonomy.leaves:
                raise ContractError(f"{location}: unknown theme id {leaf_id!r}")
            if leaf_id in seen_leaf_ids:
                raise ContractError(
                    f"{location}: duplicate theme assignment {leaf_id!r}"
                )
            review_content = content_by_id[review_id]
            if evidence not in review_content:
                lowered_content = review_content.lower()
                lowered_evidence = evidence.lower()
                start = lowered_content.find(lowered_evidence)
                unique_case_insensitive_match = (
                    start >= 0
                    and lowered_content.find(
                        lowered_evidence, start + len(lowered_evidence)
                    )
                    == -1
                )
                if not unique_case_insensitive_match:
                    raise ContractError(
                        f"{location}: evidence {evidence!r} must be an exact "
                        "substring of content_en"
                    )
                evidence = review_content[start : start + len(evidence)]
            seen_leaf_ids.add(leaf_id)
            normalized_assignments.append(
                {"specific_theme_id": leaf_id, "evidence": evidence}
            )

        validated.append(
            {
                "review_id": review_id,
                "assignments": normalized_assignments,
                "no_assignment_reason": (
                    None if assignments else "no_relevant_theme"
                ),
            }
        )
    return validated


def build_flat_projection(
    results: list[dict[str, Any]], taxonomy: Taxonomy
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results:
        for assignment in result["assignments"]:
            path = taxonomy.leaves[assignment["specific_theme_id"]]
            rows.append(
                {
                    "review_id": result["review_id"],
                    "strategic_theme": path.strategic_label,
                    "midlevel_theme": path.midlevel_label,
                    "specific_theme": path.specific_label,
                }
            )
    return rows
