from __future__ import annotations

import argparse
import os
import sys

from .consolidation import run_consolidation
from .discovery import run_discovery
from .domain import ContractError
from .full_run import run_full_classification
from .groq import GroqClient, GroqError
from .pipeline import run_slice1


def _add_provider_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str = "openai/gpt-oss-20b",
    default_reasoning_effort: str = "medium",
    default_max_completion_tokens: int | None = None,
) -> None:
    parser.add_argument(
        "--model",
        choices=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        default=default_model,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=default_reasoning_effort,
    )
    parser.add_argument(
        "--api-base", default="https://api.groq.com/openai/v1"
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=default_max_completion_tokens,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feedback-themes",
        description="Evidence-backed customer feedback theme extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    slice1 = subparsers.add_parser(
        "slice1",
        help="Run the fixed-taxonomy Groq contract against a small review slice",
    )
    slice1.add_argument("--reviews", default="data/reviews.json")
    slice1.add_argument("--taxonomy", default="data/slice1_taxonomy.json")
    slice1.add_argument("--output-dir", default="out")
    slice1.add_argument("--limit", type=int, default=10)
    _add_provider_arguments(slice1)

    discover = subparsers.add_parser(
        "discover",
        help="Build a candidate taxonomy from a deterministic review sample",
    )
    discover.add_argument("--reviews", default="data/reviews.json")
    discover.add_argument("--sample-size", type=int, default=40)
    discover.add_argument("--sample-phase", type=int, choices=[0, 1], default=0)
    discover.add_argument("--version", default="v1")
    discover.add_argument("--output", default="themes.json")
    discover.add_argument(
        "--metadata-output", default="artifacts/taxonomy_run.json"
    )
    _add_provider_arguments(
        discover,
        default_reasoning_effort="low",
        default_max_completion_tokens=5000,
    )

    consolidate = subparsers.add_parser(
        "consolidate",
        help="Merge independently discovered candidate taxonomies",
    )
    consolidate.add_argument(
        "--candidates",
        nargs="+",
        default=["themes.json", "artifacts/themes_phase1.json"],
    )
    consolidate.add_argument("--version", default="v1")
    consolidate.add_argument("--output", default="themes_final.json")
    consolidate.add_argument(
        "--metadata-output", default="artifacts/consolidation_run.json"
    )
    _add_provider_arguments(
        consolidate,
        default_model="openai/gpt-oss-120b",
        default_reasoning_effort="low",
        default_max_completion_tokens=4700,
    )

    run = subparsers.add_parser(
        "run",
        help="Classify the full corpus against the frozen taxonomy",
    )
    run.add_argument("--reviews", default="data/reviews.json")
    run.add_argument("--taxonomy", default="themes.json")
    run.add_argument("--output-dir", default="out")
    run.add_argument("--batch-size", type=int, default=10)
    run.add_argument("--checkpoint-dir", default="out/checkpoints")
    run.add_argument("--resume", action="store_true")
    _add_provider_arguments(
        run,
        default_model="openai/gpt-oss-120b",
        default_reasoning_effort="low",
        default_max_completion_tokens=3000,
    )
    return parser


def _print_usage(summary: dict) -> None:
    usage = summary["usage"]
    print(
        f"Tokens: {usage['input_tokens']} input, "
        f"{usage['output_tokens']} output."
    )
    if summary["estimated_cost_usd"] is not None:
        print(f"Estimated API cost: ${summary['estimated_cost_usd']:.6f}.")
    else:
        print("Estimated API cost: unavailable for the selected model.")


def main() -> int:
    args = _parser().parse_args()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "error: GROQ_API_KEY is not set; see .env.example",
            file=sys.stderr,
        )
        return 2

    try:
        classifier = GroqClient(
            api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
            base_url=args.api_base,
            timeout_seconds=args.timeout_seconds,
        )
        if args.command == "slice1":
            summary = run_slice1(
                reviews_path=args.reviews,
                taxonomy_path=args.taxonomy,
                output_dir=args.output_dir,
                limit=args.limit,
                classifier=classifier,
            )
        elif args.command == "discover":
            summary = run_discovery(
                reviews_path=args.reviews,
                taxonomy_output=args.output,
                metadata_output=args.metadata_output,
                sample_size=args.sample_size,
                sample_phase=args.sample_phase,
                version=args.version,
                generator=classifier,
            )
        elif args.command == "consolidate":
            summary = run_consolidation(
                candidate_paths=args.candidates,
                taxonomy_output=args.output,
                metadata_output=args.metadata_output,
                version=args.version,
                generator=classifier,
            )
        elif args.command == "run":
            summary = run_full_classification(
                reviews_path=args.reviews,
                taxonomy_path=args.taxonomy,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                classifier=classifier,
                progress=lambda message: print(message, flush=True),
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
            )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (ContractError, GroqError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.command == "slice1":
        print(
            f"Classified {summary['review_count']} reviews into "
            f"{summary['assignment_count']} assignments in "
            f"{summary['elapsed_seconds']:.3f}s."
        )
        _print_usage(summary)
        print(f"Rich output: {summary['results_path']}")
        print(f"Flat output: {summary['flat_path']}")
    elif args.command == "discover":
        print(
            f"Discovered {summary['strategic_count']} strategic, "
            f"{summary['midlevel_count']} midlevel, and "
            f"{summary['specific_count']} specific themes in "
            f"{summary['elapsed_seconds']:.3f}s."
        )
        _print_usage(summary)
        print(f"Taxonomy: {summary['taxonomy_path']}")
        print(f"Discovery metadata: {summary['metadata_path']}")
    elif args.command == "consolidate":
        print(
            f"Consolidated {summary['strategic_count']} strategic, "
            f"{summary['midlevel_count']} midlevel, and "
            f"{summary['specific_count']} specific themes in "
            f"{summary['elapsed_seconds']:.3f}s."
        )
        _print_usage(summary)
        print(f"Taxonomy: {summary['taxonomy_path']}")
        print(f"Consolidation metadata: {summary['metadata_path']}")
    else:
        print(
            f"Classified {summary['review_count']} reviews into "
            f"{summary['assignment_count']} assignments in "
            f"{summary['elapsed_seconds']:.3f}s across "
            f"{summary['batch_count']} batches."
        )
        print(f"Unassigned reviews: {summary['unassigned_count']}.")
        _print_usage(summary)
        print(
            f"Retries: {summary['generation_retries']} generation, "
            f"{summary['validation_retries']} validation, "
            f"{summary['rate_limit_retries']} rate-limit."
        )
        print(f"Rich output: {summary['results_path']}")
        print(f"Flat output: {summary['flat_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
