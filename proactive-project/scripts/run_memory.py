#!/usr/bin/env python3
"""
Run Memory System Experiment (RQ2)

Usage:
  # Without VLM (Layer 1-3 metrics only, no QA)
  python3 -m scripts.run_memory --sessions 5

  # With GPT-4o for end-to-end QA evaluation
  python3 -m scripts.run_memory --model gpt4o --sessions 5

  # With local Qwen3-VL
  python3 -m scripts.run_memory --model qwen --sessions 5

  # Custom queries per session
  python3 -m scripts.run_memory --model gpt4o --sessions 3 --queries 20
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Memory System Evaluation")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--model", default=None, choices=["gpt4o", "qwen", None],
                        help="VLM model for QA evaluation")
    parser.add_argument("--sessions", type=int, default=5, help="Max sessions to evaluate")
    parser.add_argument("--queries", type=int, default=10, help="Queries per session")
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

    # Run evaluation
    runner = BenchmarkRunner(config_path=args.config)
    results = runner.run_memory_eval(
        vlm=vlm,
        max_sessions=args.sessions,
        queries_per_session=args.queries,
        recipe=args.recipe,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Memory System Evaluation Results")
    print("=" * 60)

    agg = results.get("aggregate", {})
    print(f"Sessions evaluated: {results['num_sessions']}")
    print(f"\nLayer 1 (Task Memory):")
    print(f"  Step Detection Acc:  {agg.get('avg_step_detection_accuracy', 0):.3f}")
    print(f"  Entity Tracking F1:  {agg.get('avg_entity_tracking_f1', 0):.3f}")
    print(f"  Progress MAE:        {agg.get('avg_progress_mae', 0):.1f}%")
    print(f"\nLayer 2 (Event Memory):")
    print(f"  Recall@1/3/5:        {agg.get('avg_event_recall_at_1', 0):.3f} / {agg.get('avg_event_recall_at_3', 0):.3f} / {agg.get('avg_event_recall_at_5', 0):.3f}")
    print(f"  MRR:                 {agg.get('avg_event_mrr', 0):.3f}")
    print(f"  Temporal IoU:        {agg.get('avg_temporal_localization_iou', 0):.3f}")
    print(f"\nLayer 3 (Visual Memory):")
    print(f"  Frame P@1:           {agg.get('avg_frame_retrieval_precision_at_1', 0):.3f}")
    if vlm:
        print(f"\nEnd-to-End QA:")
        print(f"  Accuracy:            {agg.get('avg_qa_accuracy', 0):.3f}")
        print(f"  Avg Score:           {agg.get('avg_qa_avg_score', 0):.2f}/5")
        print(f"  Latency:             {agg.get('avg_avg_response_latency_ms', 0):.0f}ms")
    print(f"\nResults saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
