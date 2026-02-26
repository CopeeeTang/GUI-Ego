#!/usr/bin/env python3
"""
Run Proactive Intervention Experiment (RQ1)

Usage:
  # Baseline: periodic trigger (no VLM needed)
  python3 -m scripts.run_proactive --trigger periodic --sessions 5

  # Oracle: action boundary trigger (uses GT annotations)
  python3 -m scripts.run_proactive --trigger action_boundary --sessions 10

  # VLM-based trigger with GPT-4o
  python3 -m scripts.run_proactive --trigger vlm_delta --model gpt4o --sessions 5

  # VLM-based trigger with local Qwen3-VL
  python3 -m scripts.run_proactive --trigger vlm_delta --model qwen --sessions 5

  # Filter by recipe
  python3 -m scripts.run_proactive --trigger periodic --recipe PastaSalad --sessions 3
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Proactive Intervention Evaluation")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--trigger", default="periodic",
                        choices=["periodic", "action_boundary", "vlm_delta"],
                        help="Trigger detection method")
    parser.add_argument("--model", default=None, choices=["gpt4o", "qwen", None],
                        help="VLM model to use (required for vlm_delta)")
    parser.add_argument("--sessions", type=int, default=5, help="Max sessions to evaluate")
    parser.add_argument("--recipe", default=None, help="Filter by recipe name")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load VLM if needed
    vlm = None
    if args.model == "gpt4o":
        from src.models.gpt4o import GPT4oModel
        vlm = GPT4oModel()
    elif args.model == "qwen":
        from src.models.qwen_vl import QwenVLModel
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        qwen_cfg = config["models"]["qwen"]
        vlm = QwenVLModel(
            model_id=qwen_cfg.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
            device=qwen_cfg.get("device", "cuda"),
            dtype=qwen_cfg.get("dtype", "bfloat16"),
            cache_dir=qwen_cfg.get("cache_dir"),
        )

    if args.trigger == "vlm_delta" and vlm is None:
        print("ERROR: --model required for vlm_delta trigger method")
        sys.exit(1)

    # Run evaluation
    runner = BenchmarkRunner(config_path=args.config)
    results = runner.run_proactive_eval(
        vlm=vlm,
        trigger_method=args.trigger,
        max_sessions=args.sessions,
        recipe=args.recipe,
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Proactive Evaluation Results ({args.trigger})")
    print("=" * 60)

    agg = results.get("aggregate", {})
    print(f"Sessions evaluated: {results['num_sessions']}")
    print(f"Avg Trigger F1:     {agg.get('avg_trigger_f1', 0):.3f}")
    print(f"Avg Timing MAE:     {agg.get('avg_timing_mae', 0):.2f}s")
    print(f"Avg Content Sim:    {agg.get('avg_avg_semantic_similarity', 0):.3f}")
    print(f"Avg Triggers/min:   {agg.get('avg_triggers_per_minute', 0):.2f}")
    print(f"\nResults saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
