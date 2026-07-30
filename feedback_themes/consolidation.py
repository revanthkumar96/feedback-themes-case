from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import TaxonomyGenerator, taxonomy_schema, validate_discovered_taxonomy
from .domain import ContractError, Taxonomy
from .groq import Completion
from .pipeline import estimated_cost_usd

CONSOLIDATION_PROMPT_VERSION = "taxonomy-consolidation-v1"


def _candidate_summary(taxonomy: Taxonomy) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for strategic in taxonomy.source["strategic_themes"]:
        for midlevel in strategic["midlevel_themes"]:
            for specific in midlevel["specific_themes"]:
                summaries.append(
                    {
                        "path": [
                            strategic["label"],
                            midlevel["label"],
                            specific["label"],
                        ],
                        "definition": specific["definition"],
                    }
                )
    return summaries


def build_consolidation_prompt(candidates: list[Taxonomy]) -> str:
    payload = {
        "task": (
            "Consolidate two independently discovered candidate trees into one "
            "compact, polarity-neutral, reusable three-tier taxonomy."
        ),
        "non_negotiable_rules": [
            "Every theme is a subject, never praise, criticism, success, failure, fast, slow, high, low, present, or missing.",
            "Merge opposite states into one subject: Fast approval and Slow approval become Approval time.",
            "Do not encode a support channel and response state together unless the channel itself is the recurring subject.",
            "Specific themes at the same level must be distinct, reusable, and similarly granular.",
            "Each child narrows exactly one parent; labels must be globally unique within each tier.",
            "Definitions must be polarity-neutral and state what belongs inside the theme.",
            "Use globally unique ASCII lowercase snake_case IDs.",
            "Do not create Other, General experience, Product fit, or sentiment themes.",
            "Aim for 4-6 strategic, 9-16 midlevel, and 18-30 specific themes.",
        ],
        "reviewer_observed_subjects_that_need_coverage": [
            "application requirements, processing time, and decision explanations",
            "invoice approval and payout timing",
            "credit-line suitability and adjustment flexibility",
            "repayment flexibility, payment deferrals, and collections handling",
            "fee level, fee transparency, interest calculation, and rate changes",
            "support responsiveness, follow-up, advisor expertise, staff conduct, and contact continuity",
            "portal access, usability, performance, mobile access, accounting integrations, balance accuracy, statements, and data export",
            "clarity and consistency of information across contracts, portal, email, chatbot, and advisors",
            "account or credit-line closure",
            "institutional credibility and review authenticity",
        ],
        "candidate_leaf_paths": [
            _candidate_summary(candidate) for candidate in candidates
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run_consolidation(
    *,
    candidate_paths: list[str | Path],
    taxonomy_output: str | Path,
    metadata_output: str | Path,
    generator: TaxonomyGenerator,
    version: str = "v1",
) -> dict[str, Any]:
    if len(candidate_paths) < 2:
        raise ValueError("at least two candidate taxonomies are required")
    candidates = [Taxonomy.load(path) for path in candidate_paths]
    base_prompt = build_consolidation_prompt(candidates)
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
                "\n\nThe previous consolidated taxonomy was rejected by "
                f"deterministic validation: {validation_error}. Return the "
                "complete corrected taxonomy. Merge adjacent midlevel buckets "
                "and remove one-child or over-specific groupings before adding "
                "anything new."
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
                "consolidation model content is not valid JSON"
            )
        except ContractError as error:
            validation_error = error
        validation_retries += 1

    elapsed_seconds = round(time.perf_counter() - started, 3)
    if taxonomy is None or completion is None:
        raise validation_error or ContractError("taxonomy consolidation failed")

    cost = estimated_cost_usd(completion.model, total_usage)
    metadata = {
        "prompt_version": CONSOLIDATION_PROMPT_VERSION,
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
        "candidate_taxonomy_hashes": [
            candidate.content_hash for candidate in candidates
        ],
        "taxonomy_hash": taxonomy.content_hash,
    }
    taxonomy_path = Path(taxonomy_output)
    metadata_path = Path(metadata_output)
    _write_json(taxonomy_path, taxonomy.source)
    _write_json(metadata_path, metadata)

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
