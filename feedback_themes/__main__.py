from __future__ import annotations

import argparse
import json
import os
import sys

from .consolidation import run_consolidation
from .discovery import run_discovery
from .domain import ContractError
from .evaluation import run_evaluation
from .full_run import run_full_classification
from .groq import GroqClient, GroqError
from .holdout import run_holdout_selection
from .pipeline import run_slice1
from .retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
    FastEmbedThemeRetriever,
    RetrievalError,
)


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
    run.add_argument(
        "--subset",
        default=None,
        help=(
            "Classify only the reviews listed in this JSON file (a holdout "
            "annotations file or a selection metadata file); used for "
            "model/prompt comparison, never for the submitted full run"
        ),
    )
    run.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "Experimental: rank candidate themes with a local embedding "
            "model as a soft prior (requires the optional fastembed "
            "dependency; not the submitted configuration)"
        ),
    )
    run.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Local FastEmbed model used to rank candidate themes",
    )
    run.add_argument(
        "--embedding-cache-dir",
        default="out/model-cache",
        help="Directory for downloaded ONNX embedding weights",
    )
    run.add_argument(
        "--retrieval-top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help="Candidate themes ranked per review as a soft LLM prior",
    )
    run.add_argument(
        "--no-abstention-audit",
        action="store_true",
        help="Do not recheck shortlist abstentions against the full taxonomy",
    )
    _add_provider_arguments(
        run,
        default_model="openai/gpt-oss-120b",
        default_reasoning_effort="low",
        default_max_completion_tokens=3000,
    )

    holdout = subparsers.add_parser(
        "holdout",
        help=(
            "Select a stratified evaluation holdout and write the "
            "annotation template (no API key required)"
        ),
    )
    holdout.add_argument("--reviews", default="data/reviews.json")
    holdout.add_argument("--taxonomy", default="themes.json")
    holdout.add_argument("--size", type=int, default=50)
    holdout.add_argument(
        "--exclude-metadata",
        nargs="*",
        default=[
            "artifacts/taxonomy_run.json",
            "artifacts/taxonomy_phase1_run.json",
        ],
        help="Discovery metadata files whose sample reviews are excluded",
    )
    holdout.add_argument("--output", default="data/holdout_annotations.json")
    holdout.add_argument(
        "--metadata-output", default="artifacts/holdout_selection.json"
    )
    holdout.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing annotations file",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Score results against human holdout annotations "
            "(no API key required)"
        ),
    )
    evaluate.add_argument(
        "--annotations", default="data/holdout_annotations.json"
    )
    evaluate.add_argument("--results", default="out/results.json")
    evaluate.add_argument("--reviews", default="data/reviews.json")
    evaluate.add_argument("--taxonomy", default="themes.json")
    evaluate.add_argument("--output", default="out/evaluation.json")
    evaluate.add_argument(
        "--baseline-results",
        default=None,
        help="Second results file for run-to-run stability comparison",
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


def _load_subset_ids(path: str) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        listed = payload.get("holdout_review_ids")
        if isinstance(listed, list):
            return {str(review_id) for review_id in listed}
        annotations = payload.get("annotations")
        if isinstance(annotations, list):
            return {
                str(entry["review_id"])
                for entry in annotations
                if isinstance(entry, dict) and "review_id" in entry
            }
    raise ContractError(
        f"{path} contains neither holdout_review_ids nor annotations"
    )


def _run_offline(args: argparse.Namespace) -> int:
    try:
        if args.command == "holdout":
            summary = run_holdout_selection(
                reviews_path=args.reviews,
                taxonomy_path=args.taxonomy,
                output_path=args.output,
                metadata_output=args.metadata_output,
                size=args.size,
                exclude_metadata=args.exclude_metadata,
                force=args.force,
            )
            print(
                f"Selected {summary['holdout_size']} holdout reviews "
                f"(ratings {summary['rating_distribution']})."
            )
            if summary["discovery_overlap_count"]:
                print(
                    f"{summary['discovery_overlap_count']} reviews were "
                    "seen during taxonomy discovery (depleted rating "
                    "strata); they are flagged seen_during_discovery."
                )
            print(f"Annotation template: {summary['output_path']}")
            print(f"Selection metadata: {summary['metadata_path']}")
            print(
                "Replace each null specific_theme_ids with the supported "
                "leaf IDs; an empty list records a correct abstention."
            )
        else:
            report = run_evaluation(
                annotations_path=args.annotations,
                results_path=args.results,
                reviews_path=args.reviews,
                taxonomy_path=args.taxonomy,
                output_path=args.output,
                baseline_results_path=args.baseline_results,
            )
            _print_evaluation(report)
    except (ContractError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _print_evaluation(report: dict) -> None:
    counts = report["assignment_counts"]
    print(f"Annotated reviews: {report['review_count']}.")
    print(
        f"Assignments: {counts['true_positive']} correct, "
        f"{counts['false_positive']} unsupported, "
        f"{counts['false_negative']} missed."
    )
    print(
        f"Precision {_format_metric(report['precision'])}, "
        f"recall {_format_metric(report['recall'])}, "
        f"micro-F1 {_format_metric(report['micro_f1'])}, "
        f"macro-F1 {_format_metric(report['macro_f1'])}."
    )
    print(
        f"Exact theme-set match: {_format_metric(report['exact_match_rate'])}."
    )
    multi = report["multi_subject"]
    print(
        f"Multi-subject recall: {_format_metric(multi['recall'])} "
        f"over {multi['review_count']} reviews."
    )
    abstention = report["abstention"]
    print(
        f"Abstention precision {_format_metric(abstention['precision'])}, "
        f"recall {_format_metric(abstention['recall'])} "
        f"({abstention['reference_count']} reference abstentions)."
    )
    print(
        f"Evidence validity: "
        f"{_format_metric(report['evidence_validity_rate'])}; "
        f"unsupported-assignment rate: "
        f"{_format_metric(report['unsupported_assignment_rate'])}."
    )
    hierarchy = report["hierarchy"]
    print(
        "Hierarchy: "
        + ("valid" if hierarchy["valid"] else f"INVALID {hierarchy['unknown_theme_ids']!r}")
        + "."
    )
    if "stability" in report:
        stability = report["stability"]
        print(
            f"Stability vs baseline: identical assignments on "
            f"{stability['identical_assignment_rate']:.3f} of "
            f"{stability['shared_review_count']} reviews, "
            f"mean Jaccard {stability['mean_jaccard']:.3f}."
        )
    print(f"Report: {report['output_path']}")


def main() -> int:
    args = _parser().parse_args()
    if args.command in {"holdout", "evaluate"}:
        return _run_offline(args)

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
            retriever = (
                FastEmbedThemeRetriever(
                    model_name=args.embedding_model,
                    cache_dir=args.embedding_cache_dir,
                )
                if args.hybrid
                else None
            )
            summary = run_full_classification(
                reviews_path=args.reviews,
                taxonomy_path=args.taxonomy,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                classifier=classifier,
                progress=lambda message: print(message, flush=True),
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                retriever=retriever,
                retrieval_top_k=args.retrieval_top_k,
                audit_unassigned=not args.no_abstention_audit,
                review_ids=(
                    _load_subset_ids(args.subset) if args.subset else None
                ),
            )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (
        ContractError,
        GroqError,
        OSError,
        RetrievalError,
        ValueError,
    ) as error:
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
        if args.hybrid:
            print(
                f"Local retrieval: {args.embedding_model}, top "
                f"{args.retrieval_top_k}, "
                f"{summary['retrieval_elapsed_seconds']:.3f}s."
            )
            print(
                f"Full-taxonomy abstention audits: "
                f"{summary['fallback_review_count']}; recovered "
                f"{summary['fallback_recovered_count']}."
            )
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
