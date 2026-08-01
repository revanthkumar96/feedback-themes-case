"""Stratified holdout selection and annotation-template generation.

The holdout is used only for evaluation and prompt/model selection. It is
selected deterministically, stratified by rating, and disjoint from the
taxonomy-discovery samples so the taxonomy is never evaluated on the reviews
it was designed from.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import load_reviews
from .domain import ContractError, Taxonomy, write_json

RATINGS = (1, 2, 3, 4, 5)


def load_excluded_ids(metadata_paths: list[str | Path]) -> set[str]:
    """Collect ``sample_review_ids`` from discovery metadata files."""
    excluded: set[str] = set()
    for metadata_path in metadata_paths:
        path = Path(metadata_path)
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        sample_ids = metadata.get("sample_review_ids")
        if not isinstance(sample_ids, list) or not all(
            isinstance(review_id, str) and review_id.strip()
            for review_id in sample_ids
        ):
            raise ContractError(
                f"{path}: sample_review_ids must be a list of review ids"
            )
        excluded.update(sample_ids)
    return excluded


def select_holdout(
    reviews: list[dict[str, Any]],
    size: int,
    excluded_ids: set[str],
    min_per_rating: int = 4,
) -> list[dict[str, Any]]:
    """Deterministic rating-stratified selection.

    Quotas are proportional to each rating's share of the full dataset
    (largest-remainder rounding) with a per-rating floor so no stratum
    disappears. Within a rating, reviews the taxonomy has never seen are
    preferred and picked evenly across the short-to-long length spectrum;
    discovery-seen reviews are used only when a stratum is otherwise
    depleted.
    """
    if size < 10:
        raise ValueError("holdout size must be at least 10")
    if size > len(reviews):
        raise ValueError(
            f"holdout size {size} exceeds the {len(reviews)} reviews"
        )

    def ordered(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            subset,
            key=lambda review: (len(review["content_en"]), review["id"]),
        )

    fresh = {
        rating: ordered(
            [
                review
                for review in reviews
                if review["rating"] == rating
                and review["id"] not in excluded_ids
            ]
        )
        for rating in RATINGS
    }
    seen = {
        rating: ordered(
            [
                review
                for review in reviews
                if review["rating"] == rating and review["id"] in excluded_ids
            ]
        )
        for rating in RATINGS
    }
    capacity = {
        rating: len(fresh[rating]) + len(seen[rating]) for rating in RATINGS
    }

    exact = {
        rating: size * capacity[rating] / len(reviews) for rating in RATINGS
    }
    quotas = {rating: int(exact[rating]) for rating in RATINGS}
    remainder_order = sorted(
        RATINGS, key=lambda rating: (quotas[rating] - exact[rating], rating)
    )
    for rating in remainder_order:
        if sum(quotas.values()) == size:
            break
        quotas[rating] += 1

    floor = {
        rating: min(min_per_rating, capacity[rating]) for rating in RATINGS
    }
    for rating in RATINGS:
        quotas[rating] = min(quotas[rating], capacity[rating])
        if quotas[rating] < floor[rating]:
            quotas[rating] = floor[rating]
    while sum(quotas.values()) > size:
        donor = max(
            RATINGS,
            key=lambda rating: (quotas[rating] - floor[rating], quotas[rating], rating),
        )
        if quotas[donor] <= floor[donor]:
            raise ContractError("holdout size is too small for rating floors")
        quotas[donor] -= 1
    while sum(quotas.values()) < size:
        receiver = max(
            RATINGS,
            key=lambda rating: (capacity[rating] - quotas[rating], rating),
        )
        if quotas[receiver] >= capacity[receiver]:
            raise ContractError("holdout quotas could not be satisfied")
        quotas[receiver] += 1

    def spaced(bucket: list[dict[str, Any]], quota: int) -> list[str]:
        return [
            bucket[
                min(
                    ((2 * position + 1) * len(bucket)) // (2 * quota),
                    len(bucket) - 1,
                )
            ]["id"]
            for position in range(quota)
        ]

    selected_ids: set[str] = set()
    for rating in RATINGS:
        quota = quotas[rating]
        fresh_quota = min(quota, len(fresh[rating]))
        if fresh_quota:
            selected_ids.update(spaced(fresh[rating], fresh_quota))
        backfill = quota - fresh_quota
        if backfill:
            selected_ids.update(spaced(seen[rating], backfill))
    if len(selected_ids) != size:
        raise ContractError("holdout sampling produced duplicate selections")
    return [review for review in reviews if review["id"] in selected_ids]


def build_annotation_template(
    holdout: list[dict[str, Any]],
    taxonomy: Taxonomy,
    discovery_seen_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Annotation file the human fills in.

    ``reference.specific_theme_ids`` starts as ``null`` (not annotated). The
    annotator replaces it with a list of leaf IDs — an empty list is a
    deliberate abstention. Evaluation rejects any entry still at ``null``.
    """
    theme_reference = [
        {
            "specific_theme_id": leaf_id,
            "path": (
                f"{path.strategic_label} > {path.midlevel_label} > "
                f"{path.specific_label}"
            ),
            "definition": taxonomy.leaf_definitions[leaf_id],
        }
        for leaf_id, path in taxonomy.leaves.items()
    ]
    annotations = [
        {
            "review_id": review["id"],
            "rating": review["rating"],
            "title": review.get("title") or "",
            "content_en": review["content_en"],
            "seen_during_discovery": review["id"]
            in (discovery_seen_ids or set()),
            "reference": {
                "specific_theme_ids": None,
                "notes": "",
            },
        }
        for review in holdout
    ]
    return {
        "purpose": (
            "Human reference annotations for semantic evaluation. Replace "
            "each null specific_theme_ids with the complete list of "
            "supported leaf IDs; use an empty list for a correct abstention."
        ),
        "taxonomy_version": taxonomy.version,
        "taxonomy_hash": taxonomy.content_hash,
        "theme_reference": theme_reference,
        "annotations": annotations,
    }


def run_holdout_selection(
    *,
    reviews_path: str | Path,
    taxonomy_path: str | Path,
    output_path: str | Path,
    metadata_output: str | Path,
    size: int,
    exclude_metadata: list[str | Path],
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not force:
        raise ContractError(
            f"{output} already exists; it may contain human annotations. "
            "Pass --force to overwrite."
        )
    reviews = load_reviews(reviews_path)
    taxonomy = Taxonomy.load(taxonomy_path)
    excluded_ids = load_excluded_ids(exclude_metadata)
    holdout = select_holdout(reviews, size, excluded_ids)
    template = build_annotation_template(
        holdout, taxonomy, discovery_seen_ids=excluded_ids
    )
    write_json(output, template)

    discovery_overlap = [
        review["id"] for review in holdout if review["id"] in excluded_ids
    ]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "holdout_size": len(holdout),
        "excluded_review_count": len(excluded_ids),
        "eligible_review_count": len(reviews) - len(
            excluded_ids & {review["id"] for review in reviews}
        ),
        "discovery_overlap_count": len(discovery_overlap),
        "discovery_overlap_review_ids": discovery_overlap,
        "rating_distribution": {
            str(rating): sum(
                1 for review in holdout if review["rating"] == rating
            )
            for rating in RATINGS
        },
        "taxonomy_hash": taxonomy.content_hash,
        "holdout_review_ids": [review["id"] for review in holdout],
    }
    write_json(Path(metadata_output), metadata)
    return {
        "holdout_size": len(holdout),
        "output_path": output,
        "metadata_path": Path(metadata_output),
        "rating_distribution": metadata["rating_distribution"],
        "discovery_overlap_count": len(discovery_overlap),
    }
