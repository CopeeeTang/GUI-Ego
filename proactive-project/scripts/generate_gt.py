#!/usr/bin/env python3
"""
Generate and inspect ground truth triggers for all sessions.

Usage:
  python3 -m scripts.generate_gt
  python3 -m scripts.generate_gt --recipe PastaSalad --sessions 3
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.egtea_loader import EGTEALoader
from src.proactive.gt_generator import GroundTruthGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Ground Truth Triggers")
    parser.add_argument("--data-root", default="/home/v-tangxin/GUI/data/EGTEA_Gaze_Plus")
    parser.add_argument("--recipe", default=None, help="Filter by recipe")
    parser.add_argument("--sessions", type=int, default=None, help="Max sessions")
    parser.add_argument("--output", default=None, help="Save GT to JSON file")
    args = parser.parse_args()

    loader = EGTEALoader(args.data_root, split=1)
    print(loader.summary())
    print()

    gt_gen = GroundTruthGenerator()
    gt = gt_gen.generate_for_dataset(
        loader, recipe=args.recipe, max_sessions=args.sessions
    )

    print(gt_gen.summary(gt))
    print()

    # Show sample
    for sid, triggers in list(gt.items())[:3]:
        session = loader.get_session(sid)
        print(f"\n{'='*60}")
        print(f"Session: {sid} ({session.recipe})")
        print(f"Duration: {session.duration_sec:.0f}s, Actions: {session.num_actions}")
        print(f"Triggers: {len(triggers)}")
        for t in triggers[:8]:
            print(f"  [{t.timestamp:7.1f}s] {t.trigger_type.value:20s} | {t.expected_content[:60]}")
        if len(triggers) > 8:
            print(f"  ... and {len(triggers) - 8} more")

    # Save if requested
    if args.output:
        serialized = {}
        for sid, triggers in gt.items():
            serialized[sid] = [{
                "timestamp": t.timestamp,
                "type": t.trigger_type.value,
                "content": t.expected_content,
                "valid_window": list(t.valid_window),
                "context": t.context,
            } for t in triggers]

        with open(args.output, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"\nGT saved to {args.output}")


if __name__ == "__main__":
    main()
