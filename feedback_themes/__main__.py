from __future__ import annotations

import argparse
import os
import sys

from .domain import ContractError
from .groq import GroqClient, GroqError
from .pipeline import run_slice1


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
    slice1.add_argument(
        "--model",
        choices=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        default="openai/gpt-oss-20b",
    )
    slice1.add_argument(
        "--api-base", default="https://api.groq.com/openai/v1"
    )
    slice1.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command != "slice1":
        raise AssertionError(f"unhandled command: {args.command}")

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
            base_url=args.api_base,
            timeout_seconds=args.timeout_seconds,
        )
        summary = run_slice1(
            reviews_path=args.reviews,
            taxonomy_path=args.taxonomy,
            output_dir=args.output_dir,
            limit=args.limit,
            classifier=classifier,
        )
    except (ContractError, GroqError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Classified {summary['review_count']} reviews into "
        f"{summary['assignment_count']} assignments in "
        f"{summary['elapsed_seconds']:.3f}s."
    )
    usage = summary["usage"]
    print(
        f"Tokens: {usage['input_tokens']} input, "
        f"{usage['output_tokens']} output."
    )
    if summary["estimated_cost_usd"] is not None:
        print(f"Estimated API cost: ${summary['estimated_cost_usd']:.6f}.")
    else:
        print("Estimated API cost: unavailable for the selected model.")
    print(f"Rich output: {summary['results_path']}")
    print(f"Flat output: {summary['flat_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
