#!/usr/bin/env python3
"""Command-line interface for Ego-Dataset sampling.

Usage:
    # Demo mode - generate a single example for confirmation
    python -m agent.src.sampling.main --task 2.1 --demo

    # Batch mode - generate 100 samples
    python -m agent.src.sampling.main --task 2.1 --count 100
    python -m agent.src.sampling.main --task 2.2 --count 100

    # Show statistics
    python -m agent.src.sampling.main --stats
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from agent.src.sampling.sampler import EgoDatasetSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ego-Dataset Sampling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["2.1", "2.2"],
        help="Task to sample from (2.1 or 2.2)",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a single demo sample for confirmation",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show dataset statistics",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/ego-dataset/data",
        help="Root directory of the ego-dataset",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="example",
        help="Root directory for output samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--signal-window",
        type=float,
        default=5.0,
        help="Time window (seconds) for signal extraction (default: ±5s)",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=3,
        help="Number of keyframes to extract per sample (default: 3)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize sampler
    try:
        sampler = EgoDatasetSampler(
            data_root=args.data_root,
            output_root=args.output_root,
            signal_window=args.signal_window,
            num_keyframes=args.num_frames,
        )
    except FileNotFoundError as e:
        logger.error(f"Data root not found: {e}")
        sys.exit(1)

    # Show statistics
    if args.stats:
        stats = sampler.get_statistics()
        print("\n" + "=" * 60)
        print("Ego-Dataset Statistics")
        print("=" * 60)
        print(f"\nTotal Participants: {len(stats['participants'])}")
        print(f"Participants: {', '.join(stats['participants'])}")

        print(f"\nTask 2.1:")
        print(f"  Participants with data: {len(stats['task_2_1']['participants'])}")
        print(f"  Total entries: {stats['task_2_1']['total_entries']}")
        print(f"  Participants: {', '.join(stats['task_2_1']['participants'])}")

        print(f"\nTask 2.2:")
        print(f"  Participants with data: {len(stats['task_2_2']['participants'])}")
        print(f"  Total entries: {stats['task_2_2']['total_entries']}")
        print(f"  Participants: {', '.join(stats['task_2_2']['participants'])}")
        print("=" * 60 + "\n")
        return

    # Validate task argument
    if not args.task:
        parser.error("--task is required for demo or batch generation")

    # Demo mode
    if args.demo:
        print(f"\n{'=' * 60}")
        print(f"Generating demo sample for Task {args.task}")
        print(f"{'=' * 60}\n")

        demo_path = sampler.generate_demo(task=args.task)

        if demo_path:
            print(f"\n{'=' * 60}")
            print("Demo sample generated successfully!")
            print(f"Location: {demo_path}")
            print(f"\nContents:")
            for item in sorted(demo_path.rglob("*")):
                if item.is_file():
                    rel_path = item.relative_to(demo_path)
                    size = item.stat().st_size
                    print(f"  {rel_path} ({size:,} bytes)")
            print(f"\nVerification commands:")
            print(f"  # Check JSON:")
            print(f"  python -c \"import json; print(json.dumps(json.load(open('{demo_path}/rawdata.json')), indent=2, ensure_ascii=False))\"")
            print(f"\n  # Check video:")
            print(f"  ffprobe {demo_path}/video/clip.mp4")
            print(f"\n  # List keyframes:")
            print(f"  ls -la {demo_path}/video/*.jpg")
            print(f"{'=' * 60}\n")
        else:
            logger.error("Failed to generate demo sample")
            sys.exit(1)
        return

    # Batch mode
    print(f"\n{'=' * 60}")
    print(f"Generating {args.count} samples for Task {args.task}")
    print(f"Output directory: {args.output_root}")
    print(f"Random seed: {args.seed}")
    print(f"{'=' * 60}\n")

    generated = sampler.batch_generate(
        task=args.task,
        count=args.count,
        random_seed=args.seed,
    )

    print(f"\n{'=' * 60}")
    print(f"Batch generation complete!")
    print(f"Generated {len(generated)} samples")
    print(f"Output location: {args.output_root}/Task{args.task}/")
    print(f"{'=' * 60}\n")

    # Summary statistics
    if generated:
        total_size = sum(
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            for p in generated
        )
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
        print(f"\nSample distribution by participant:")
        participant_counts = {}
        for p in generated:
            participant = p.parent.name
            participant_counts[participant] = participant_counts.get(participant, 0) + 1
        for participant, count in sorted(participant_counts.items()):
            print(f"  {participant}: {count} samples")


if __name__ == "__main__":
    main()
