from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .domain import (
    ContractError,
    Taxonomy,
    build_flat_projection,
    validate_model_output,
)
from .groq import Completion

PROMPT_VERSION = "classification-v6"
MODEL_PRICING_USD_PER_MILLION = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}


class Classifier(Protocol):
    model: str

    def classify(self, prompt: str, schema: dict[str, Any]) -> Completion: ...


def build_prompt(
    reviews: list[dict[str, Any]],
    taxonomy: Taxonomy,
    candidate_ids_by_review: dict[str, list[str]] | None = None,
) -> str:
    if candidate_ids_by_review is not None:
        for review in reviews:
            review_id = review["id"]
            candidates = candidate_ids_by_review.get(review_id)
            if not candidates:
                raise ContractError(
                    f"semantic candidates missing for review {review_id!r}"
                )
            unknown_ids = set(candidates) - set(taxonomy.leaves)
            if unknown_ids:
                raise ContractError(
                    f"unknown semantic candidates for {review_id!r}: "
                    f"{sorted(unknown_ids)!r}"
                )

    taxonomy_for_model = []
    for leaf_id, path in taxonomy.leaves.items():
        definition = _find_leaf_definition(taxonomy.source, leaf_id)
        taxonomy_for_model.append(
            {
                "specific_theme_id": leaf_id,
                "path": [
                    path.strategic_label,
                    path.midlevel_label,
                    path.specific_label,
                ],
                "definition": definition,
            }
        )

    reviews_for_model = []
    for review in reviews:
        review_payload = {
            "review_id": review["id"],
            "title": review.get("title") or "",
            "content_en": review["content_en"],
        }
        if candidate_ids_by_review is not None:
            review_payload["candidate_specific_theme_ids"] = (
                candidate_ids_by_review[review["id"]]
            )
        reviews_for_model.append(review_payload)

    rules = [
        "Classify the subjects the customer explicitly discusses; sentiment alone never justifies a subject theme.",
        "Use only the supplied specific_theme_id values.",
        "Do not force a subject assignment when no supplied theme is supported.",
        "A verdict on the company or service that names no concrete subject belongs to exactly one general-sentiment leaf (positive endorsement, negative verdict, or neutral remark), never to a subject theme.",
        "Use the general-sentiment leaves only when the whole review names no concrete subject; when any subject theme applies, do not add a general-sentiment assignment alongside it.",
        "When a cost complaint does not identify whether fees or interest are meant, assign unattributed price remarks rather than guessing a fee or interest theme.",
        "Distinguish the interest subjects: the size or competitiveness of a rate, how interest or effective rates are calculated or explained, and how rate changes are announced are different themes.",
        "Evaluate every independent sentence and clause; return every distinct supported theme rather than only the most prominent one.",
        "A multi-topic review should normally have multiple assignments when separate clauses support separate themes.",
        "Return at most five assignments per review; when more are supported, keep the five with the most explicit evidence.",
        "For every assignment, copy the shortest self-contained verbatim substring from content_en that identifies both the subject and the behavior or property supporting the theme.",
        "Evidence must remain understandable on its own; include important qualifiers such as delays, negations, timing, or missing communication.",
        "Evidence must be one contiguous substring copied without paraphrasing, changed capitalization, omitted words, or inserted ellipses.",
        "Never use the title as evidence.",
        "Do not assign the same specific theme twice to one review.",
        "Return every review once and preserve input order.",
        "Return an empty assignments list when no supplied theme is explicitly supported.",
    ]
    if candidate_ids_by_review is not None:
        rules.insert(
            2,
            "For each review, inspect its ordered candidate_specific_theme_ids first. A local semantic encoder produced this ranking, but it is only a recall aid: you may assign any supplied theme, and review meaning plus evidence must decide.",
        )
    instructions = {
        "task": "Assign zero or more fixed specific themes to each review.",
        "rules": rules,
        "taxonomy": taxonomy_for_model,
        "reviews": reviews_for_model,
    }
    return json.dumps(instructions, ensure_ascii=False, indent=2)


def _find_leaf_definition(taxonomy: dict[str, Any], leaf_id: str) -> str:
    for strategic in taxonomy["strategic_themes"]:
        for midlevel in strategic["midlevel_themes"]:
            for specific in midlevel["specific_themes"]:
                if specific["id"] == leaf_id:
                    return specific["definition"]
    raise ContractError(f"taxonomy definition missing for {leaf_id!r}")


def _load_reviews(path: str | Path, limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ContractError("reviews file must contain a JSON list")
    selected = payload[:limit]
    if len(selected) < limit:
        raise ContractError(
            f"requested {limit} reviews but input contains only {len(payload)}"
        )
    for index, review in enumerate(selected):
        if not isinstance(review, dict):
            raise ContractError(f"reviews[{index}] must be an object")
        if not isinstance(review.get("id"), str) or not review["id"].strip():
            raise ContractError(f"reviews[{index}].id must be a non-empty string")
        if (
            not isinstance(review.get("content_en"), str)
            or not review["content_en"].strip()
        ):
            raise ContractError(
                f"reviews[{index}].content_en must be a non-empty string"
            )
    return selected


def estimated_cost_usd(model: str, usage: dict[str, int]) -> float | None:
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)
    if pricing is None:
        return None
    return round(
        (
            usage["input_tokens"] * pricing["input"]
            + usage["output_tokens"] * pricing["output"]
        )
        / 1_000_000,
        6,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def run_slice1(
    *,
    reviews_path: str | Path,
    taxonomy_path: str | Path,
    output_dir: str | Path,
    limit: int,
    classifier: Classifier,
) -> dict[str, Any]:
    taxonomy = Taxonomy.load(taxonomy_path)
    reviews = _load_reviews(reviews_path, limit)
    prompt = build_prompt(reviews, taxonomy)
    started = time.perf_counter()
    completion = classifier.classify(prompt, taxonomy.model_schema(len(reviews)))
    elapsed_seconds = round(time.perf_counter() - started, 3)
    try:
        raw_output = json.loads(completion.content)
    except json.JSONDecodeError as error:
        raise ContractError("model content is not valid JSON") from error

    results = validate_model_output(raw_output, reviews, taxonomy)
    flat = build_flat_projection(results, taxonomy)
    estimated_cost = estimated_cost_usd(completion.model, completion.usage)
    rich_output = {
        "run": {
            "slice": "classification-contract",
            "prompt_version": PROMPT_VERSION,
            "provider": "groq",
            "model": completion.model,
            "reasoning_effort": getattr(classifier, "reasoning_effort", None),
            "max_completion_tokens": getattr(
                classifier, "max_completion_tokens", None
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "usage": completion.usage,
            "estimated_cost_usd": estimated_cost,
            "taxonomy_hash": taxonomy.content_hash,
        },
        "taxonomy": taxonomy.source,
        "review_results": results,
    }

    output_path = Path(output_dir)
    _write_json(output_path / "results.json", rich_output)
    _write_json(output_path / "flat.json", flat)
    return {
        "review_count": len(results),
        "assignment_count": len(flat),
        "results_path": output_path / "results.json",
        "flat_path": output_path / "flat.json",
        "elapsed_seconds": elapsed_seconds,
        "usage": completion.usage,
        "estimated_cost_usd": estimated_cost,
    }
