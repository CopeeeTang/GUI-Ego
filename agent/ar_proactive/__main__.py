"""CLI entry point: python3 -m agent.ar_proactive"""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ARAgentConfig
from .agent import ProactiveARAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proactive AR Smart Glasses Agent — Observe → Think → Act",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # EGTEA session (new mode)
  python3 -m agent.ar_proactive \\
      --egtea-session P01-R01-PastaSalad --egtea-max-clips 5 -v

  # EGTEA with ground truth steps
  python3 -m agent.ar_proactive \\
      --egtea-session P01-R01-PastaSalad --use-gt-steps --egtea-max-clips 10

  # Legacy: single sample
  python3 -m agent.ar_proactive \\
      --sample agent/example/Task2.1/P10_Ernesto/sample_001 -v

  # Legacy: multiple samples for a participant
  python3 -m agent.ar_proactive \\
      --task Task2.1 --participant-id P10_Ernesto --limit 3
        """,
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--egtea-session",
        type=str,
        help="EGTEA session name (e.g., P01-R01-PastaSalad)",
    )
    input_group.add_argument(
        "--sample",
        type=str,
        help="Path to a single sample directory (legacy mode)",
    )
    input_group.add_argument(
        "--participant-id",
        type=str,
        help="Participant ID for legacy mode (requires --task)",
    )

    # EGTEA options
    parser.add_argument(
        "--egtea-data-root",
        type=str,
        default="data/EGTEA_Gaze_Plus",
        help="EGTEA dataset root (default: data/EGTEA_Gaze_Plus)",
    )
    parser.add_argument(
        "--egtea-max-clips",
        type=int,
        default=None,
        help="Max clips to process in EGTEA session (default: all)",
    )
    parser.add_argument(
        "--use-gt-steps",
        action="store_true",
        help="Use EGTEA ground truth actions as task steps",
    )

    # Processing options
    parser.add_argument(
        "--model",
        type=str,
        default="claude:claude-sonnet-4-5",
        help="LLM model spec (default: claude:claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="Seconds between frame extractions (default: 1.0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds between interventions (default: 5.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Min confidence for intervention (default: 0.5)",
    )

    # Legacy options
    parser.add_argument("--task", type=str, default="Task2.1")
    parser.add_argument("--data-root", type=str, default="agent/example")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--importance-threshold", type=float, default=0.4)

    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ARAgentConfig(
        model_spec=args.model,
        frame_interval_sec=args.frame_interval,
        cooldown_sec=args.cooldown,
        min_confidence=args.min_confidence,
        importance_threshold=args.importance_threshold,
        egtea_data_root=args.egtea_data_root,
        egtea_max_clips=args.egtea_max_clips,
        use_egtea_gt_steps=args.use_gt_steps,
        data_root=args.data_root,
        verbose=args.verbose,
    )

    agent = ProactiveARAgent(config)

    if args.egtea_session:
        # EGTEA mode
        from .data.egtea_loader import EGTEALoader

        loader = EGTEALoader(args.egtea_data_root)
        session = loader.get_session(args.egtea_session)
        if session is None:
            print(f"Error: EGTEA session not found: {args.egtea_session}", file=sys.stderr)
            print(f"Available sessions ({len(loader.list_sessions())}):")
            for name in loader.list_sessions()[:10]:
                print(f"  {name}")
            print("  ...")
            sys.exit(1)

        # Print session info
        stats = loader.get_session_stats(session)
        print(f"EGTEA Session: {stats['session']}")
        print(f"  Recipe: {stats['recipe']}")
        print(f"  Clips: {stats['num_clips']}, Actions: {stats['num_actions']}")
        print(f"  Duration: {stats['duration_sec']:.0f}s")
        print(f"  Has gaze: {stats['has_gaze']}")
        if args.egtea_max_clips:
            print(f"  Processing first {args.egtea_max_clips} clips")
        print()

        results = agent.process_egtea_session(session, loader)
        _print_egtea_summary(results)

    elif args.sample:
        sample_dir = Path(args.sample)
        if not sample_dir.exists():
            print(f"Error: sample directory not found: {sample_dir}", file=sys.stderr)
            sys.exit(1)
        results = agent.process_sample(sample_dir)
        _print_legacy_summary(results)

    else:
        participant_dir = Path(args.data_root) / args.task / args.participant_id
        if not participant_dir.exists():
            print(f"Error: participant directory not found: {participant_dir}", file=sys.stderr)
            sys.exit(1)

        sample_dirs = sorted(
            d for d in participant_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
        )
        if args.limit:
            sample_dirs = sample_dirs[:args.limit]

        all_results = []
        for sd in sample_dirs:
            try:
                r = agent.process_sample(sd)
                all_results.append(r)
                _print_legacy_summary(r)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({"sample_dir": str(sd), "error": str(e)})
        results = all_results

    output_path = agent.save_results(results, output_path=args.output)
    print(f"\nResults saved to: {output_path}")


def _print_egtea_summary(result: dict):
    """Print EGTEA session results."""
    print(f"  Frames processed: {result['frames_processed']}")
    print(f"  Processing time: {result['processing_time_sec']}s")
    print(f"  Interventions: {result['intervention_count']}")

    # Print interventions
    for intv in result.get("interventions", []):
        mode = intv.get("intervention_mode", "?")
        itype = intv.get("intervention_type", "?")
        print(
            f"    [{intv['timestamp']:.1f}s] [{mode}] {itype} "
            f"(confidence: {intv['confidence']:.2f})"
        )
        print(f"      → {intv['content']}")

    # Print eval metrics
    ev = result.get("eval", {})
    if ev:
        print(f"\n  Evaluation:")
        print(f"    Total triggers: {ev.get('total_triggers', 0)}")
        print(f"    GT action boundaries: {ev.get('gt_boundary_count', 0)}")
        print(f"    Triggers at boundaries: {ev.get('triggers_at_gt_boundary', 0)}")
        print(f"    Boundary coverage: {ev.get('boundary_coverage', 0):.1%}")
        if ev.get("intervention_mode_distribution"):
            print(f"    Mode distribution: {ev['intervention_mode_distribution']}")
        if ev.get("intervention_type_distribution"):
            print(f"    Type distribution: {ev['intervention_type_distribution']}")

    # Memory stats
    stats = result.get("memory_stats", {})
    if stats:
        print(f"\n  Memory:")
        print(f"    Task: {stats.get('task_goal', '?')}")
        print(f"    Steps: {stats.get('completed_steps', 0)}/{stats.get('total_steps', 0)} completed")
        print(f"    Working frames: {stats.get('working_frames', 0)}/{stats.get('working_capacity', 0)}")
        print(f"    Key events: {stats.get('key_events', 0)}")


def _print_legacy_summary(result: dict):
    """Print legacy sample results."""
    print(f"  Video duration: {result.get('video_duration_sec', '?')}s")
    print(f"  Frames processed: {result.get('frames_processed', '?')}")
    print(f"  Processing time: {result.get('processing_time_sec', '?')}s")
    print(f"  Interventions: {result.get('intervention_count', 0)}")

    for intv in result.get("interventions", []):
        mode = intv.get("intervention_mode", "?")
        print(
            f"    [{intv['timestamp']:.2f}s] [{mode}] {intv['intervention_type']} "
            f"(confidence: {intv['confidence']:.2f})"
        )
        print(f"      → {intv['content']}")


if __name__ == "__main__":
    main()
