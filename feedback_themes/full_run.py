from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .discovery import load_reviews
from .domain import (
    ContractError,
    Taxonomy,
    build_flat_projection,
    validate_model_output,
)
from .groq import GroqError
from .pipeline import (
    PROMPT_VERSION,
    Classifier,
    build_prompt,
    estimated_cost_usd,
)
from .retrieval import ThemeCandidate, ThemeRetriever

Progress = Callable[[str], None]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in total:
        total[key] += usage[key]


def _classify_batch(
    *,
    reviews: list[dict[str, Any]],
    taxonomy: Taxonomy,
    classifier: Classifier,
    candidate_ids_by_review: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    base_prompt = build_prompt(
        reviews,
        taxonomy,
        candidate_ids_by_review=candidate_ids_by_review,
    )
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    validation_error: ContractError | None = None
    generation_retries = 0

    for attempt in range(3):
        prompt = base_prompt
        if validation_error is not None:
            prompt += (
                "\n\nYour previous response was rejected by deterministic "
                f"validation: {validation_error}. Return the complete batch "
                "again, correcting that issue without changing supported "
                "assignments."
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
                and attempt < 2
            ):
                generation_retries += 1
                validation_error = ContractError(
                    "provider failed to generate its strict JSON schema"
                )
                continue
            raise
        _add_usage(total_usage, completion.usage)
        try:
            payload = json.loads(completion.content)
            results = validate_model_output(
                payload,
                reviews,
                taxonomy,
            )
            return results, total_usage, attempt - generation_retries, generation_retries
        except json.JSONDecodeError:
            validation_error = ContractError("model content is not valid JSON")
        except ContractError as error:
            validation_error = error

    raise validation_error or ContractError("batch classification failed")


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
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 30:
        raise ValueError("batch_size must be between 1 and 30")
    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be positive")
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
    rate_limit_retries = 0
    fallback_review_count = 0
    fallback_recovered_count = 0
    active_elapsed_seconds = retrieval_elapsed_seconds
    batch_count = (len(reviews) + batch_size - 1) // batch_size
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None

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
        checkpoint: dict[str, Any] | None = None
        if resume and batch_checkpoint and batch_checkpoint.exists():
            with batch_checkpoint.open(encoding="utf-8") as handle:
                candidate_checkpoint = json.load(handle)
            if candidate_checkpoint.get("identity") == identity:
                checkpoint = candidate_checkpoint
            else:
                progress(
                    f"Batch {batch_index}/{batch_count}: checkpoint was "
                    "written under a different configuration; recomputing"
                )
        if checkpoint is not None:
            results = validate_model_output(
                {"results": checkpoint["results"]}, batch, taxonomy
            )
            usage = checkpoint["usage"]
            batch_validation_retries = checkpoint["validation_retries"]
            batch_generation_retries = checkpoint["generation_retries"]
            batch_rate_limit_retries = checkpoint["rate_limit_retries"]
            batch_elapsed = float(checkpoint["elapsed_seconds"])
            fallback_review_ids = checkpoint.get("fallback_review_ids", [])
            batch_fallback_recovered_count = int(
                checkpoint.get("fallback_recovered_count", 0)
            )
            progress(
                f"Batch {batch_index}/{batch_count}: resumed "
                f"{len(batch)} reviews from checkpoint"
            )
        else:
            rate_retries_before = getattr(
                classifier, "rate_limit_retry_count", 0
            )
            batch_started = time.perf_counter()
            (
                results,
                usage,
                batch_validation_retries,
                batch_generation_retries,
            ) = _classify_batch(
                reviews=batch,
                taxonomy=taxonomy,
                classifier=classifier,
                candidate_ids_by_review=candidate_ids_by_review,
            )
            fallback_review_ids: list[str] = []
            batch_fallback_recovered_count = 0
            if candidate_rankings is not None and audit_unassigned:
                primary_by_id = {
                    result["review_id"]: result for result in results
                }
                fallback_reviews = [
                    review
                    for review in batch
                    if not primary_by_id[review["id"]]["assignments"]
                ]
                fallback_review_ids = [
                    review["id"] for review in fallback_reviews
                ]
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
                    )
                    _add_usage(usage, fallback_usage)
                    batch_validation_retries += fallback_validation_retries
                    batch_generation_retries += fallback_generation_retries
                    fallback_by_id = {
                        result["review_id"]: result
                        for result in fallback_results
                    }
                    batch_fallback_recovered_count = sum(
                        bool(result["assignments"])
                        for result in fallback_results
                    )
                    results = [
                        fallback_by_id.get(result["review_id"], result)
                        for result in results
                    ]
            batch_elapsed = round(time.perf_counter() - batch_started, 3)
            batch_rate_limit_retries = (
                getattr(classifier, "rate_limit_retry_count", 0)
                - rate_retries_before
            )
            if batch_checkpoint:
                _write_json(
                    batch_checkpoint,
                    {
                        "identity": identity,
                        "results": results,
                        "usage": usage,
                        "elapsed_seconds": batch_elapsed,
                        "validation_retries": batch_validation_retries,
                        "generation_retries": batch_generation_retries,
                        "rate_limit_retries": batch_rate_limit_retries,
                        "fallback_review_ids": fallback_review_ids,
                        "fallback_recovered_count": (
                            batch_fallback_recovered_count
                        ),
                    },
                )
            progress(
                f"Batch {batch_index}/{batch_count}: {len(batch)} reviews, "
                f"{sum(len(item['assignments']) for item in results)} "
                f"assignments, {len(fallback_review_ids)} fallback audits, "
                f"{batch_elapsed:.1f}s"
            )
        all_results.extend(results)
        for review in batch:
            review_id = review["id"]
            route: dict[str, Any] = {
                "fallback_used": review_id in fallback_review_ids,
            }
            if candidate_rankings is not None:
                route["semantic_candidates"] = [
                    candidate.as_dict()
                    for candidate in candidate_rankings[review_id]
                ]
            routing_by_review[review_id] = route
        _add_usage(total_usage, usage)
        validation_retries += batch_validation_retries
        generation_retries += batch_generation_retries
        rate_limit_retries += batch_rate_limit_retries
        fallback_review_count += len(fallback_review_ids)
        fallback_recovered_count += batch_fallback_recovered_count
        active_elapsed_seconds += batch_elapsed

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
    _write_json(results_path, rich_output)
    _write_json(flat_path, flat)

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
