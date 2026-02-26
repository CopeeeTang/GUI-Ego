"""
CLI entry point for the evaluation framework.

Usage:
  # Evaluate a single result file (metrics only, fast)
  python3 -m agent.ar_proactive.eval result.json --skip-judge

  # Full evaluation with Gemini content judge
  python3 -m agent.ar_proactive.eval result.json

  # Evaluate multiple files
  python3 -m agent.ar_proactive.eval result1.json result2.json

  # Custom judge model and tolerance
  python3 -m agent.ar_proactive.eval result.json --judge-model gemini:gemini-2.5-flash --tolerance 5.0

  # Save report
  python3 -m agent.ar_proactive.eval result.json -o eval_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .runner import EvalRunner
from .report import EvalReport


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Proactive AR Agent results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to result JSON file(s) from process_egtea_session",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini:gemini-3-flash",
        help="Model spec for content quality judge (default: gemini:gemini-3-flash)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Boundary matching tolerance in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM content quality evaluation (faster)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save report to JSON file",
    )
    parser.add_argument(
        "--markdown",
        help="Export report as markdown file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    runner = EvalRunner(
        judge_model=args.judge_model,
        tolerance_sec=args.tolerance,
        skip_judge=args.skip_judge,
    )

    if len(args.results) == 1:
        # Single file evaluation
        report_data = runner.evaluate_result_file(args.results[0])
        report = EvalReport(report_data)
        report.print_summary()
    else:
        # Multi-file aggregate
        report_data = runner.evaluate_multiple(args.results)
        report = EvalReport(report_data)
        # Print each session
        for session in report_data.get("sessions", []):
            EvalReport(session).print_summary()
        report.print_aggregate()

    # Save outputs
    if args.output:
        report.save(args.output)
        print(f"Report saved to: {args.output}")

    if args.markdown:
        if len(args.results) == 1:
            md = report.to_markdown()
            Path(args.markdown).write_text(md)
            print(f"Markdown saved to: {args.markdown}")


if __name__ == "__main__":
    main()
