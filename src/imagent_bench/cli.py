from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import BenchmarkRunError, run
from .try_agent import add_arguments as add_try_arguments
from .try_agent import run_once


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagent-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Score an agent repository against a benchmark suite")
    run_parser.add_argument("--repository", required=True, help="Local path or git URL for the candidate repository")
    run_parser.add_argument("--commit", default=None, help="Commit SHA to record or checkout for git URLs")
    run_parser.add_argument("--config", default=None, help="Benchmark config JSON path")
    run_parser.add_argument("--output-dir", default="benchmark-output", help="Directory for reports and artifacts")
    run_parser.add_argument("--baseline-score", type=float, default=None, help="Current top merged baseline score")
    run_parser.add_argument("--baseline-commit", default=None, help="Current top merged baseline commit")
    run_parser.add_argument("--fail-on-policy", action="store_true", help="Exit non-zero when policy fails")

    try_parser = subparsers.add_parser("try", help="Run an agent against one ad-hoc prompt (development only)")
    add_try_arguments(try_parser)

    args = parser.parse_args(argv)

    if args.command == "try":
        return run_once(args)

    if args.command == "run":
        try:
            result = run(
                repository=args.repository,
                commit=args.commit,
                config=args.config,
                output_dir=args.output_dir,
                baseline_score=args.baseline_score,
                baseline_commit=args.baseline_commit,
            )
        except BenchmarkRunError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        report_path = Path(args.output_dir) / "benchmark-report.json"
        print(json.dumps({"status": result.status, "overall_score": result.overall_score, "report": str(report_path)}))
        if args.fail_on_policy and result.status != "pass":
            return 1
        return 0

    return 2
