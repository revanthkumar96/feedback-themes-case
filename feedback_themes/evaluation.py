"""Semantic evaluation of classification results against human annotations.

Implements the holdout metrics from the revised plan: assignment precision
and recall, micro/macro F1, exact set match, multi-subject recall,
abstention precision and recall, evidence-substring validity,
unsupported-assignment rate, hierarchy validity, and run-to-run agreement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import load_reviews
from .domain import ContractError, Taxonomy, write_json


def load_annotations(
    path: str | Path,
    taxonomy: Taxonomy,
    review_ids: set[str],
) -> dict[str, set[str]]:
    """Return {review_id: reference leaf-id set}. Rejects unannotated entries."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("annotations"), list
    ):
        raise ContractError("annotations file must contain an annotations list")
    annotation_hash = payload.get("taxonomy_hash")
    if annotation_hash != taxonomy.content_hash:
        raise ContractError(
            "annotations were made against a different taxonomy "
            f"(hash {annotation_hash!r})"
        )

    references: dict[str, set[str]] = {}
    unannotated: list[str] = []
    for index, entry in enumerate(payload["annotations"]):
        location = f"annotations[{index}]"
        if not isinstance(entry, dict):
            raise ContractError(f"{location} must be an object")
        review_id = entry.get("review_id")
        if not isinstance(review_id, str) or review_id not in review_ids:
            raise ContractError(f"{location}: unknown review id {review_id!r}")
        if review_id in references:
            raise ContractError(f"{location}: duplicate review id {review_id!r}")
        reference = entry.get("reference")
        if not isinstance(reference, dict):
            raise ContractError(f"{location}: reference must be an object")
        leaf_ids = reference.get("specific_theme_ids")
        if leaf_ids is None:
            unannotated.append(review_id)
            continue
        if not isinstance(leaf_ids, list):
            raise ContractError(
                f"{location}: specific_theme_ids must be a list or null"
            )
        unknown = [
            leaf_id for leaf_id in leaf_ids if leaf_id not in taxonomy.leaves
        ]
        if unknown:
            raise ContractError(
                f"{location}: unknown theme ids {sorted(unknown)!r}"
            )
        if len(set(leaf_ids)) != len(leaf_ids):
            raise ContractError(f"{location}: duplicate theme ids")
        references[review_id] = set(leaf_ids)

    if unannotated:
        raise ContractError(
            f"{len(unannotated)} holdout reviews are not annotated yet "
            f"(e.g. {unannotated[0]!r}); replace null specific_theme_ids "
            "with a list before evaluating"
        )
    if not references:
        raise ContractError("annotations file contains no annotated reviews")
    return references


def load_predictions(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Return {review_id: [assignment, ...]} from a rich results file."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("review_results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ContractError("results file must contain review_results")
    predictions: dict[str, list[dict[str, str]]] = {}
    for index, result in enumerate(results):
        location = f"review_results[{index}]"
        if not isinstance(result, dict):
            raise ContractError(f"{location} must be an object")
        review_id = result.get("review_id")
        assignments = result.get("assignments")
        if not isinstance(review_id, str) or not review_id.strip():
            raise ContractError(f"{location}: review_id must be a string")
        if review_id in predictions:
            raise ContractError(f"{location}: duplicate review id {review_id!r}")
        if not isinstance(assignments, list):
            raise ContractError(f"{location}: assignments must be a list")
        predictions[review_id] = assignments
    return predictions


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(
    *,
    references: dict[str, set[str]],
    predictions: dict[str, list[dict[str, str]]],
    taxonomy: Taxonomy,
    content_by_id: dict[str, str],
) -> dict[str, Any]:
    missing = sorted(set(references) - set(predictions))
    if missing:
        raise ContractError(
            f"results are missing {len(missing)} annotated reviews "
            f"(e.g. {missing[0]!r})"
        )

    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0
    multi_tp = 0
    multi_fn = 0
    multi_review_count = 0
    abstain_tp = 0
    abstain_pred = 0
    abstain_ref = 0
    per_leaf: dict[str, dict[str, int]] = {}
    evidence_total = 0
    evidence_valid = 0
    invalid_hierarchy_ids: set[str] = set()

    for review_id, reference in references.items():
        assignments = predictions[review_id]
        predicted: set[str] = set()
        for assignment in assignments:
            leaf_id = assignment.get("specific_theme_id")
            if leaf_id not in taxonomy.leaves:
                invalid_hierarchy_ids.add(str(leaf_id))
                continue
            predicted.add(leaf_id)
            evidence_total += 1
            evidence = assignment.get("evidence")
            if (
                isinstance(evidence, str)
                and evidence in content_by_id.get(review_id, "")
            ):
                evidence_valid += 1

        true_positive += len(predicted & reference)
        false_positive += len(predicted - reference)
        false_negative += len(reference - predicted)
        exact_matches += predicted == reference
        if len(reference) >= 2:
            multi_review_count += 1
            multi_tp += len(predicted & reference)
            multi_fn += len(reference - predicted)
        if not predicted:
            abstain_pred += 1
        if not reference:
            abstain_ref += 1
        if not predicted and not reference:
            abstain_tp += 1
        for leaf_id in predicted | reference:
            counts = per_leaf.setdefault(
                leaf_id, {"tp": 0, "fp": 0, "fn": 0}
            )
            if leaf_id in predicted and leaf_id in reference:
                counts["tp"] += 1
            elif leaf_id in predicted:
                counts["fp"] += 1
            else:
                counts["fn"] += 1

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    leaf_f1_scores = []
    for counts in per_leaf.values():
        leaf_precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
        leaf_recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
        score = _f1(leaf_precision, leaf_recall)
        leaf_f1_scores.append(score if score is not None else 0.0)

    return {
        "review_count": len(references),
        "assignment_counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "precision": precision,
        "recall": recall,
        "micro_f1": _f1(precision, recall),
        "macro_f1": (
            sum(leaf_f1_scores) / len(leaf_f1_scores)
            if leaf_f1_scores
            else None
        ),
        "exact_match_rate": _ratio(exact_matches, len(references)),
        "multi_subject": {
            "review_count": multi_review_count,
            "recall": _ratio(multi_tp, multi_tp + multi_fn),
        },
        "abstention": {
            "reference_count": abstain_ref,
            "predicted_count": abstain_pred,
            "precision": _ratio(abstain_tp, abstain_pred),
            "recall": _ratio(abstain_tp, abstain_ref),
        },
        "evidence_validity_rate": _ratio(evidence_valid, evidence_total),
        "unsupported_assignment_rate": _ratio(
            false_positive, true_positive + false_positive
        ),
        "hierarchy": {
            "valid": not invalid_hierarchy_ids,
            "unknown_theme_ids": sorted(invalid_hierarchy_ids),
        },
    }


def compare_runs(
    predictions_a: dict[str, list[dict[str, str]]],
    predictions_b: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Run-to-run stability: assignment agreement between two result files."""
    shared = sorted(set(predictions_a) & set(predictions_b))
    if not shared:
        raise ContractError("the two result files share no reviews")
    jaccard_total = 0.0
    identical = 0
    for review_id in shared:
        set_a = {
            assignment.get("specific_theme_id")
            for assignment in predictions_a[review_id]
        }
        set_b = {
            assignment.get("specific_theme_id")
            for assignment in predictions_b[review_id]
        }
        union = set_a | set_b
        jaccard_total += len(set_a & set_b) / len(union) if union else 1.0
        identical += set_a == set_b
    return {
        "shared_review_count": len(shared),
        "identical_assignment_rate": identical / len(shared),
        "mean_jaccard": jaccard_total / len(shared),
    }


def run_evaluation(
    *,
    annotations_path: str | Path,
    results_path: str | Path,
    reviews_path: str | Path,
    taxonomy_path: str | Path,
    output_path: str | Path,
    baseline_results_path: str | Path | None = None,
) -> dict[str, Any]:
    taxonomy = Taxonomy.load(taxonomy_path)
    reviews = load_reviews(reviews_path)
    content_by_id = {review["id"]: review["content_en"] for review in reviews}
    references = load_annotations(
        annotations_path, taxonomy, set(content_by_id)
    )
    predictions = load_predictions(results_path)
    report = evaluate(
        references=references,
        predictions=predictions,
        taxonomy=taxonomy,
        content_by_id=content_by_id,
    )
    if baseline_results_path is not None:
        report["stability"] = compare_runs(
            predictions, load_predictions(baseline_results_path)
        )
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["annotations_path"] = str(annotations_path)
    report["results_path"] = str(results_path)
    report["taxonomy_hash"] = taxonomy.content_hash
    write_json(Path(output_path), report)
    report["output_path"] = Path(output_path)
    return report
