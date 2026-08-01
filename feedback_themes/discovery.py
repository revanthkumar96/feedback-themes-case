from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .domain import ContractError, Taxonomy, write_json
from .groq import Completion
from .pipeline import estimated_cost_usd

DISCOVERY_PROMPT_VERSION = "taxonomy-discovery-v1"
SNAKE_CASE_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class TaxonomyGenerator(Protocol):
    model: str
    reasoning_effort: str
    max_completion_tokens: int

    def classify(self, prompt: str, schema: dict[str, Any]) -> Completion: ...


def load_reviews(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        reviews = json.load(handle)
    if not isinstance(reviews, list) or not reviews:
        raise ContractError("reviews file must contain a non-empty JSON list")
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            raise ContractError(f"reviews[{index}] must be an object")
        if not isinstance(review.get("id"), str) or not review["id"].strip():
            raise ContractError(f"reviews[{index}].id must be a non-empty string")
        if (
            not isinstance(review.get("rating"), int)
            or review["rating"] not in {1, 2, 3, 4, 5}
        ):
            raise ContractError(f"reviews[{index}].rating must be from 1 to 5")
        if (
            not isinstance(review.get("content_en"), str)
            or not review["content_en"].strip()
        ):
            raise ContractError(
                f"reviews[{index}].content_en must be a non-empty string"
            )
    return reviews


def select_stratified_sample(
    reviews: list[dict[str, Any]], sample_size: int, phase: int = 0
) -> list[dict[str, Any]]:
    """Select equal rating strata while covering short and long reviews."""
    if sample_size < 5:
        raise ValueError("sample_size must be at least 5")
    if sample_size > len(reviews):
        raise ValueError("sample_size cannot exceed the review count")
    if phase not in {0, 1}:
        raise ValueError("phase must be 0 or 1")

    ratings = [1, 2, 3, 4, 5]
    buckets = {
        rating: sorted(
            (review for review in reviews if review["rating"] == rating),
            key=lambda review: (len(review["content_en"]), review["id"]),
        )
        for rating in ratings
    }
    if any(not bucket for bucket in buckets.values()):
        raise ContractError("every rating from 1 to 5 must have at least one review")

    base, remainder = divmod(sample_size, len(ratings))
    quotas = {
        rating: base + (1 if index < remainder else 0)
        for index, rating in enumerate(ratings)
    }
    if any(quotas[rating] > len(buckets[rating]) for rating in ratings):
        raise ValueError("sample_size cannot be evenly stratified across ratings")
    if phase == 1 and any(
        2 * quotas[rating] > len(buckets[rating]) for rating in ratings
    ):
        raise ValueError(
            "sample_size is too large for a disjoint second rating-stratified phase"
        )

    selected_ids: set[str] = set()
    for rating in ratings:
        bucket = buckets[rating]
        quota = quotas[rating]
        first_phase_indices = [
            min(
                ((2 * position + 1) * len(bucket)) // (2 * quota),
                len(bucket) - 1,
            )
            for position in range(quota)
        ]
        if phase == 0:
            selected_indices = first_phase_indices
        else:
            first_phase_index_set = set(first_phase_indices)
            remaining_indices = [
                index
                for index in range(len(bucket))
                if index not in first_phase_index_set
            ]
            selected_indices = [
                remaining_indices[
                    min(
                        ((2 * position + 1) * len(remaining_indices))
                        // (2 * quota),
                        len(remaining_indices) - 1,
                    )
                ]
                for position in range(quota)
            ]
        for index in selected_indices:
            selected_ids.add(bucket[index]["id"])

    if len(selected_ids) != sample_size:
        raise ContractError("stratified sampling produced duplicate selections")
    return [review for review in reviews if review["id"] in selected_ids]


def taxonomy_schema() -> dict[str, Any]:
    specific = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "definition": {"type": "string"},
        },
        "required": ["id", "label", "definition"],
        "additionalProperties": False,
    }
    midlevel = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "definition": {"type": "string"},
            "specific_themes": {"type": "array", "items": specific},
        },
        "required": ["id", "label", "definition", "specific_themes"],
        "additionalProperties": False,
    }
    strategic = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "definition": {"type": "string"},
            "midlevel_themes": {"type": "array", "items": midlevel},
        },
        "required": ["id", "label", "definition", "midlevel_themes"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "strategic_themes": {"type": "array", "items": strategic},
        },
        "required": ["strategic_themes"],
        "additionalProperties": False,
    }


def build_discovery_prompt(sample: list[dict[str, Any]]) -> str:
    payload = {
        "task": (
            "Design a reusable three-tier taxonomy of subjects customers discuss "
            "in these business-banking and invoice-financing reviews."
        ),
        "rules": [
            "A theme is a recurring subject, not sentiment, a review summary, or an isolated incident.",
            "All tiers are subjects at different resolution; each child must narrow exactly one parent.",
            "Keep polarity out of labels so positive and negative feedback share the same subject theme.",
            "Specific themes must be reusable across reviews and mutually distinguishable.",
            "Use concise noun-phrase labels and one-sentence boundary definitions.",
            "Do not create Other, General experience, Positive feedback, Negative feedback, or rating themes.",
            "Avoid composite labels joined by and or slash; split independent subjects.",
            "Use globally unique lowercase snake_case IDs.",
            "Aim for 3-6 strategic themes, 7-15 midlevel themes, and 15-30 specific themes.",
            "Prefer a coherent compact taxonomy over exhaustive one-off labels.",
        ],
        "reviews": [
            {
                "review_id": review["id"],
                "title": review.get("title") or "",
                "content_en": review["content_en"],
            }
            for review in sample
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def validate_discovered_taxonomy(payload: Any, version: str) -> Taxonomy:
    if not isinstance(payload, dict):
        raise ContractError("taxonomy model output must be an object")
    data = {
        "version": version,
        "strategic_themes": payload.get("strategic_themes"),
    }
    taxonomy = Taxonomy.from_dict(data)

    strategic_count = len(data["strategic_themes"])
    midlevel_count = sum(
        len(strategic["midlevel_themes"])
        for strategic in data["strategic_themes"]
    )
    specific_count = len(taxonomy.leaves)
    if not 3 <= strategic_count <= 8:
        raise ContractError(
            f"discovered taxonomy has {strategic_count} strategic themes; expected 3-8"
        )
    if not 6 <= midlevel_count <= 20:
        raise ContractError(
            f"discovered taxonomy has {midlevel_count} midlevel themes; expected 6-20"
        )
    if not 8 <= specific_count <= 40:
        raise ContractError(
            f"discovered taxonomy has {specific_count} specific themes; expected 8-40"
        )

    invalid_ids = [
        theme_id
        for theme_id in _all_theme_ids(data)
        if not SNAKE_CASE_ID.fullmatch(theme_id)
    ]
    if invalid_ids:
        raise ContractError(
            f"theme IDs must be lowercase snake_case; invalid: {invalid_ids[0]!r}"
        )
    return taxonomy


def _all_theme_ids(taxonomy: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for strategic in taxonomy["strategic_themes"]:
        ids.append(strategic["id"])
        for midlevel in strategic["midlevel_themes"]:
            ids.append(midlevel["id"])
            ids.extend(
                specific["id"] for specific in midlevel["specific_themes"]
            )
    return ids


def run_discovery(
    *,
    reviews_path: str | Path,
    taxonomy_output: str | Path,
    metadata_output: str | Path,
    sample_size: int,
    sample_phase: int,
    generator: TaxonomyGenerator,
    version: str = "v1",
) -> dict[str, Any]:
    reviews = load_reviews(reviews_path)
    sample = select_stratified_sample(reviews, sample_size, sample_phase)
    base_prompt = build_discovery_prompt(sample)
    started = time.perf_counter()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    validation_retries = 0
    completion: Completion | None = None
    taxonomy: Taxonomy | None = None
    validation_error: ContractError | None = None

    for attempt in range(2):
        prompt = base_prompt
        if validation_error is not None:
            prompt += (
                "\n\nThe previous taxonomy was rejected by deterministic "
                f"validation: {validation_error}. Return the complete corrected "
                "taxonomy. In particular, every ID must contain ASCII lowercase "
                "letters, digits, and underscores only."
            )
        completion = generator.classify(prompt, taxonomy_schema())
        for key in total_usage:
            total_usage[key] += completion.usage[key]
        try:
            payload = json.loads(completion.content)
            taxonomy = validate_discovered_taxonomy(payload, version)
            break
        except json.JSONDecodeError:
            validation_error = ContractError(
                "taxonomy model content is not valid JSON"
            )
        except ContractError as error:
            validation_error = error
        validation_retries += 1

    elapsed_seconds = round(time.perf_counter() - started, 3)
    if taxonomy is None or completion is None:
        raise validation_error or ContractError("taxonomy discovery failed")

    taxonomy_path = Path(taxonomy_output)
    metadata_path = Path(metadata_output)
    cost = estimated_cost_usd(completion.model, total_usage)
    metadata = {
        "prompt_version": DISCOVERY_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "groq",
        "model": completion.model,
        "reasoning_effort": generator.reasoning_effort,
        "max_completion_tokens": generator.max_completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "usage": total_usage,
        "estimated_cost_usd": cost,
        "request_count": validation_retries + 1,
        "validation_retries": validation_retries,
        "rate_limit_retries": getattr(generator, "rate_limit_retry_count", 0),
        "sample_size": len(sample),
        "sample_phase": sample_phase,
        "sample_review_ids": [review["id"] for review in sample],
        "taxonomy_hash": taxonomy.content_hash,
    }
    write_json(taxonomy_path, taxonomy.source)
    write_json(metadata_path, metadata)

    strategic_count = len(taxonomy.source["strategic_themes"])
    midlevel_count = sum(
        len(strategic["midlevel_themes"])
        for strategic in taxonomy.source["strategic_themes"]
    )
    return {
        "taxonomy_path": taxonomy_path,
        "metadata_path": metadata_path,
        "strategic_count": strategic_count,
        "midlevel_count": midlevel_count,
        "specific_count": len(taxonomy.leaves),
        "elapsed_seconds": elapsed_seconds,
        "usage": total_usage,
        "estimated_cost_usd": cost,
    }
