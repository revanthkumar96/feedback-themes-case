from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .discovery import load_reviews
from .domain import (
    ContractError,
    Taxonomy,
    build_flat_projection,
    validate_model_output,
    write_json,
)
from .groq import GroqError
from .pipeline import (
    PROMPT_VERSION,
    Classifier,
    build_prompt,
    estimated_call_cost_usd,
    estimated_cost_usd,
)
from .retrieval import ThemeCandidate, ThemeRetriever

Progress = Callable[[str], None]


class BudgetError(RuntimeError):
    """Raised when the projected next call would cross the cost ceiling."""


class CostBudget:
    """Thread-safe hard spend ceiling, checked before every provider call."""

    def __init__(self, ceiling_usd: float) -> None:
        if ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be positive")
        self.ceiling_usd = ceiling_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, projected_usd: float | None) -> None:
        """Fail loudly if the projected call would cross the ceiling.

        A ``None`` projection (unknown model pricing) is let through; the
        guard can only be as good as the price list.
        """
        if projected_usd is None:
            return
        with self._lock:
            if self.spent_usd + projected_usd > self.ceiling_usd:
                raise BudgetError(
                    f"projected next call (~${projected_usd:.4f}) would take "
                    f"spend past the ${self.ceiling_usd:.2f} ceiling "
                    f"(${self.spent_usd:.4f} spent so far); completed "
                    "batches are checkpointed, rerun with --resume and a "
                    "higher --max-cost-usd"
                )

    def record(self, actual_usd: float | None) -> None:
        if actual_usd is None:
            return
        with self._lock:
            self.spent_usd += actual_usd


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in total:
        total[key] += usage[key]


def _classify_batch(
    *,
    reviews: list[dict[str, Any]],
    taxonomy: Taxonomy,
    classifier: Classifier,
    candidate_ids_by_review: dict[str, list[str]] | None = None,
    budget: CostBudget | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    base_prompt = build_prompt(
        reviews,
        taxonomy,
        candidate_ids_by_review=candidate_ids_by_review,
    )
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    validation_error: ContractError | None = None
    validation_retries = 0
    generation_retries = 0

    # Three attempts total, shared between provider-side generation failures
    # and deterministic contract violations.
    for _ in range(3):
        prompt = base_prompt
        if validation_error is not None:
            prompt += (
                "\n\nYour previous response was rejected by deterministic "
                f"validation: {validation_error}. Return the complete batch "
                "again, correcting that issue without changing supported "
                "assignments."
            )
        if budget is not None:
            budget.reserve(
                estimated_call_cost_usd(
                    classifier.model,
                    prompt,
                    getattr(classifier, "max_completion_tokens", None),
                )
            )
        try:
            completion = classifier.classify(
                prompt,
                taxonomy.model_schema(len(reviews)),
            )
        except GroqError as error:
            if (
                error.status_code == 400
                and error.error_code == "json_validate_failed"
                and validation_retries + generation_retries < 2
            ):
                generation_retries += 1
                validation_error = ContractError(
                    "provider failed to generate its strict JSON schema"
                )
                continue
            raise
        _add_usage(total_usage, completion.usage)
        if budget is not None:
            budget.record(
                estimated_cost_usd(completion.model, completion.usage)
            )
        try:
            payload = json.loads(completion.content)
            results = validate_model_output(
                payload,
                reviews,
                taxonomy,
            )
            return results, total_usage, validation_retries, generation_retries
        except json.JSONDecodeError:
            validation_error = ContractError("model content is not valid JSON")
        except ContractError as error:
            validation_error = error
        validation_retries += 1

    raise validation_error or ContractError("batch classification failed")


def _review_content_hash(reviews: list[dict[str, Any]]) -> str:
    """Hash the texts the model actually sees, so a changed review under an
    unchanged ID cannot silently reuse a stale checkpoint."""
    canonical = json.dumps(
        [
            [review["id"], review.get("title") or "", review["content_en"]]
            for review in reviews
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checkpoint_identity(
    *,
    batch_index: int,
    reviews: list[dict[str, Any]],
    taxonomy: Taxonomy,
    classifier: Classifier,
    batch_size: int,
    candidate_ids_by_review: dict[str, list[str]] | None,
    embedding_model: str | None,
    retrieval_top_k: int | None,
    audit_unassigned: bool,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "review_ids": [review["id"] for review in reviews],
        "review_content_hash": _review_content_hash(reviews),
        "taxonomy_hash": taxonomy.content_hash,
        "prompt_version": PROMPT_VERSION,
        "model": classifier.model,
        "reasoning_effort": getattr(classifier, "reasoning_effort", None),
        "max_completion_tokens": getattr(
            classifier, "max_completion_tokens", None
        ),
        "batch_size": batch_size,
        "embedding_model": embedding_model,
        "retrieval_top_k": retrieval_top_k,
        "candidate_ids_by_review": candidate_ids_by_review,
        "audit_unassigned": audit_unassigned,
    }


@dataclass
class _BatchOutcome:
    """Everything one batch contributes to the run, however it was obtained."""

    results: list[dict[str, Any]]
    usage: dict[str, int]
    validation_retries: int
    generation_retries: int
    rate_limit_retries: int
    elapsed_seconds: float
    fallback_review_ids: list[str]
    fallback_recovered_count: int
    resumed: bool = False


def _load_reusable_checkpoint(
    batch_checkpoint: Path | None,
    identity: dict[str, Any],
    *,
    resume: bool,
    progress: Progress,
    batch_index: int,
    batch_count: int,
) -> dict[str, Any] | None:
    if not resume or batch_checkpoint is None or not batch_checkpoint.exists():
        return None
    with batch_checkpoint.open(encoding="utf-8") as handle:
        candidate = json.load(handle)
    if candidate.get("identity") == identity:
        return candidate
    progress(
        f"Batch {batch_index}/{batch_count}: checkpoint was "
        "written under a different configuration; recomputing"
    )
    return None


def _outcome_from_checkpoint(
    checkpoint: dict[str, Any],
    batch: list[dict[str, Any]],
    taxonomy: Taxonomy,
) -> _BatchOutcome:
    return _BatchOutcome(
        results=validate_model_output(
            {"results": checkpoint["results"]}, batch, taxonomy
        ),
        usage=checkpoint["usage"],
        validation_retries=checkpoint["validation_retries"],
        generation_retries=checkpoint["generation_retries"],
        rate_limit_retries=checkpoint["rate_limit_retries"],
        elapsed_seconds=float(checkpoint["elapsed_seconds"]),
        fallback_review_ids=checkpoint.get("fallback_review_ids", []),
        fallback_recovered_count=int(
            checkpoint.get("fallback_recovered_count", 0)
        ),
        resumed=True,
    )


def _classify_batch_with_fallback(
    *,
    batch: list[dict[str, Any]],
    taxonomy: Taxonomy,
    classifier: Classifier,
    candidate_ids_by_review: dict[str, list[str]] | None,
    run_fallback_audit: bool,
    budget: CostBudget | None = None,
) -> _BatchOutcome:
    """Classify one batch; re-check shortlist abstentions on the full taxonomy."""
    rate_retries_before = getattr(classifier, "rate_limit_retry_count", 0)
    started = time.perf_counter()
    results, usage, validation_retries, generation_retries = _classify_batch(
        reviews=batch,
        taxonomy=taxonomy,
        classifier=classifier,
        candidate_ids_by_review=candidate_ids_by_review,
        budget=budget,
    )
    fallback_review_ids: list[str] = []
    fallback_recovered_count = 0
    if run_fallback_audit:
        primary_by_id = {result["review_id"]: result for result in results}
        fallback_reviews = [
            review
            for review in batch
            if not primary_by_id[review["id"]]["assignments"]
        ]
        fallback_review_ids = [review["id"] for review in fallback_reviews]
        if fallback_reviews:
            (
                fallback_results,
                fallback_usage,
                fallback_validation_retries,
                fallback_generation_retries,
            ) = _classify_batch(
                reviews=fallback_reviews,
                taxonomy=taxonomy,
                classifier=classifier,
                budget=budget,
            )
            _add_usage(usage, fallback_usage)
            validation_retries += fallback_validation_retries
            generation_retries += fallback_generation_retries
            fallback_by_id = {
                result["review_id"]: result for result in fallback_results
            }
            fallback_recovered_count = sum(
                bool(result["assignments"]) for result in fallback_results
            )
            results = [
                fallback_by_id.get(result["review_id"], result)
                for result in results
            ]
    return _BatchOutcome(
        results=results,
        usage=usage,
        validation_retries=validation_retries,
        generation_retries=generation_retries,
        rate_limit_retries=(
            getattr(classifier, "rate_limit_retry_count", 0)
            - rate_retries_before
        ),
        elapsed_seconds=round(time.perf_counter() - started, 3),
        fallback_review_ids=fallback_review_ids,
        fallback_recovered_count=fallback_recovered_count,
    )


def run_full_classification(
    *,
    reviews_path: str | Path,
    taxonomy_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    classifier: Classifier,
    progress: Progress = print,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    retriever: ThemeRetriever | None = None,
    retrieval_top_k: int = 12,
    audit_unassigned: bool = True,
    review_ids: set[str] | None = None,
    concurrency: int = 1,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 30:
        raise ValueError("batch_size must be between 1 and 30")
    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be positive")
    if concurrency < 1 or concurrency > 8:
        raise ValueError("concurrency must be between 1 and 8")
    budget = CostBudget(max_cost_usd) if max_cost_usd is not None else None
    started = time.perf_counter()
    reviews = load_reviews(reviews_path)
    if review_ids is not None:
        if not review_ids:
            raise ContractError("review subset must not be empty")
        known_ids = {review["id"] for review in reviews}
        unknown_ids = review_ids - known_ids
        if unknown_ids:
            raise ContractError(
                f"subset contains unknown review ids: {sorted(unknown_ids)[:3]!r}"
            )
        reviews = [
            review for review in reviews if review["id"] in review_ids
        ]
    taxonomy = Taxonomy.load(taxonomy_path)
    retrieval_started = time.perf_counter()
    candidate_rankings: dict[str, list[ThemeCandidate]] | None = None
    if retriever is not None:
        progress(
            f"Embedding {len(reviews)} reviews and "
            f"{len(taxonomy.leaves)} themes with {retriever.model_name}"
        )
        candidate_rankings = retriever.retrieve(
            reviews,
            taxonomy,
            top_k=retrieval_top_k,
        )
    retrieval_elapsed_seconds = round(
        time.perf_counter() - retrieval_started,
        3,
    )
    if retriever is not None:
        progress(
            f"Semantic retrieval complete in "
            f"{retrieval_elapsed_seconds:.3f}s"
        )

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    all_results: list[dict[str, Any]] = []
    routing_by_review: dict[str, dict[str, Any]] = {}
    validation_retries = 0
    generation_retries = 0
    fallback_review_count = 0
    fallback_recovered_count = 0
    active_elapsed_seconds = retrieval_elapsed_seconds
    batch_count = (len(reviews) + batch_size - 1) // batch_size
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    process_rate_retries_before = getattr(
        classifier, "rate_limit_retry_count", 0
    )

    batch_specs: list[
        tuple[int, list[dict[str, Any]], dict[str, list[str]] | None]
    ] = []
    for batch_index, start in enumerate(range(0, len(reviews), batch_size), 1):
        batch = reviews[start : start + batch_size]
        candidate_ids_by_review = (
            {
                review["id"]: [
                    candidate.specific_theme_id
                    for candidate in candidate_rankings[review["id"]]
                ]
                for review in batch
            }
            if candidate_rankings is not None
            else None
        )
        batch_specs.append((batch_index, batch, candidate_ids_by_review))

    def _obtain_outcome(
        spec: tuple[int, list[dict[str, Any]], dict[str, list[str]] | None],
    ) -> _BatchOutcome:
        batch_index, batch, candidate_ids_by_review = spec
        identity = _checkpoint_identity(
            batch_index=batch_index,
            reviews=batch,
            taxonomy=taxonomy,
            classifier=classifier,
            batch_size=batch_size,
            candidate_ids_by_review=candidate_ids_by_review,
            embedding_model=(
                retriever.model_name if retriever is not None else None
            ),
            retrieval_top_k=(
                retrieval_top_k if retriever is not None else None
            ),
            audit_unassigned=audit_unassigned,
        )
        batch_checkpoint = (
            checkpoint_path / f"batch-{batch_index:03d}.json"
            if checkpoint_path
            else None
        )
        checkpoint = _load_reusable_checkpoint(
            batch_checkpoint,
            identity,
            resume=resume,
            progress=progress,
            batch_index=batch_index,
            batch_count=batch_count,
        )
        if checkpoint is not None:
            outcome = _outcome_from_checkpoint(checkpoint, batch, taxonomy)
            progress(
                f"Batch {batch_index}/{batch_count}: resumed "
                f"{len(batch)} reviews from checkpoint"
            )
            return outcome
        outcome = _classify_batch_with_fallback(
            batch=batch,
            taxonomy=taxonomy,
            classifier=classifier,
            candidate_ids_by_review=candidate_ids_by_review,
            run_fallback_audit=(
                candidate_rankings is not None and audit_unassigned
            ),
            budget=budget,
        )
        if batch_checkpoint:
            write_json(
                batch_checkpoint,
                {
                    "identity": identity,
                    "results": outcome.results,
                    "usage": outcome.usage,
                    "elapsed_seconds": outcome.elapsed_seconds,
                    "validation_retries": outcome.validation_retries,
                    "generation_retries": outcome.generation_retries,
                    "rate_limit_retries": outcome.rate_limit_retries,
                    "fallback_review_ids": outcome.fallback_review_ids,
                    "fallback_recovered_count": (
                        outcome.fallback_recovered_count
                    ),
                },
            )
        progress(
            f"Batch {batch_index}/{batch_count}: {len(batch)} reviews, "
            f"{sum(len(item['assignments']) for item in outcome.results)} "
            f"assignments, {len(outcome.fallback_review_ids)} fallback "
            f"audits, {outcome.elapsed_seconds:.1f}s"
        )
        return outcome

    if concurrency == 1:
        outcomes = [_obtain_outcome(spec) for spec in batch_specs]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(_obtain_outcome, batch_specs))

    for (_, batch, _), outcome in zip(batch_specs, outcomes, strict=True):
        all_results.extend(outcome.results)
        for review in batch:
            review_id = review["id"]
            route: dict[str, Any] = {
                "fallback_used": review_id in outcome.fallback_review_ids,
            }
            if candidate_rankings is not None:
                route["semantic_candidates"] = [
                    candidate.as_dict()
                    for candidate in candidate_rankings[review_id]
                ]
            routing_by_review[review_id] = route
        _add_usage(total_usage, outcome.usage)
        validation_retries += outcome.validation_retries
        generation_retries += outcome.generation_retries
        fallback_review_count += len(outcome.fallback_review_ids)
        fallback_recovered_count += outcome.fallback_recovered_count
        active_elapsed_seconds += outcome.elapsed_seconds

    # Resumed batches report the retries recorded in their checkpoints;
    # fresh batches are counted from the shared client counter, which stays
    # exact even when batches run concurrently.
    rate_limit_retries = sum(
        outcome.rate_limit_retries for outcome in outcomes if outcome.resumed
    ) + (
        getattr(classifier, "rate_limit_retry_count", 0)
        - process_rate_retries_before
    )

    process_wall_seconds = round(time.perf_counter() - started, 3)
    elapsed_seconds = round(active_elapsed_seconds, 3)
    flat = build_flat_projection(all_results, taxonomy)
    model = classifier.model
    cost = estimated_cost_usd(model, total_usage)
    published_results = []
    for result in all_results:
        published_result = dict(result)
        if retriever is not None:
            published_result["routing"] = routing_by_review[
                result["review_id"]
            ]
        published_results.append(published_result)

    rich_output = {
        "run": {
            "pipeline": (
                "embedding-assisted-frozen-taxonomy-classification"
                if retriever is not None
                else "frozen-taxonomy-classification"
            ),
            "provider": "groq",
            "model": model,
            "reasoning_effort": getattr(classifier, "reasoning_effort", None),
            "max_completion_tokens": getattr(
                classifier, "max_completion_tokens", None
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "process_wall_seconds": process_wall_seconds,
            "review_subset_size": (
                len(reviews) if review_ids is not None else None
            ),
            "batch_size": batch_size,
            "batch_count": batch_count,
            "concurrency": concurrency,
            "cost_ceiling_usd": max_cost_usd,
            "usage": total_usage,
            "estimated_cost_usd": cost,
            "validation_retries": validation_retries,
            "generation_retries": generation_retries,
            "rate_limit_retries": rate_limit_retries,
            "retrieval": (
                {
                    "model": retriever.model_name,
                    "top_k": retrieval_top_k,
                    "elapsed_seconds": retrieval_elapsed_seconds,
                    "role": "soft_candidate_ranking",
                }
                if retriever is not None
                else None
            ),
            "abstention_audit": (
                {
                    "enabled": audit_unassigned,
                    "review_count": fallback_review_count,
                    "recovered_count": fallback_recovered_count,
                }
                if retriever is not None
                else None
            ),
            "resumed": resume,
            "taxonomy_hash": taxonomy.content_hash,
        },
        "taxonomy": taxonomy.source,
        "review_results": published_results,
    }
    output_path = Path(output_dir)
    results_path = output_path / "results.json"
    flat_path = output_path / "flat.json"
    write_json(results_path, rich_output)
    write_json(flat_path, flat)

    return {
        "review_count": len(all_results),
        "assignment_count": len(flat),
        "unassigned_count": sum(
            not result["assignments"] for result in all_results
        ),
        "results_path": results_path,
        "flat_path": flat_path,
        "elapsed_seconds": elapsed_seconds,
        "batch_count": batch_count,
        "usage": total_usage,
        "estimated_cost_usd": cost,
        "validation_retries": validation_retries,
        "generation_retries": generation_retries,
        "rate_limit_retries": rate_limit_retries,
        "retrieval_elapsed_seconds": retrieval_elapsed_seconds,
        "fallback_review_count": fallback_review_count,
        "fallback_recovered_count": fallback_recovered_count,
    }
