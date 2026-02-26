"""Threshold sweep for ProAssist Adaptive-D optimization.

Sweeps over multiple not_talk_threshold values on WTAG dataset,
recording per-frame w2t_prob traces and computing metrics at each threshold.

Usage:
    python3 threshold_sweep.py --dataset wtag/dialog_val --mode both
    python3 threshold_sweep.py --dataset wtag/dialog_val --mode threshold
    python3 threshold_sweep.py --dataset wtag/dialog_val --mode diff
"""

import os
import sys
import json
import argparse
import time

sys.path.insert(0, "/home/v-tangxin/GUI/temp/ProAssist")
os.environ.setdefault("DATA_ROOT_DIR", "/home/v-tangxin/GUI/data/ProAssist")

import torch
from mmassist.model import build_from_checkpoint
from mmassist.data import build_eval_datasets
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator
from mmassist.configs import ModelArguments

MODEL_PATH = "/home/v-tangxin/GUI/data/ProAssist/ProAssist-Model-L4096-I1"
DATA_ROOT = os.path.join(os.environ["DATA_ROOT_DIR"], "processed_data")
RESULT_DIR = "/home/v-tangxin/GUI/proassist_experiments/results/threshold_sweep"

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DIFF_THRESHOLDS = ["diff0.05", "diff0.1", "diff0.15", "diff0.2", "diff0.3", "diff0.5"]


def run_sweep(dataset_name, thresholds, model, tokenizer, model_config):
    """Run threshold sweep for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "wtag/dialog_val").
        thresholds: List of threshold values (float or str like "diff0.1").
        model: The loaded ProAssist model.
        tokenizer: The tokenizer.
        model_config: The model config.

    Returns:
        Dict mapping threshold -> metrics dict.
    """
    results = {}
    datasets = build_eval_datasets(
        eval_datasets=dataset_name,
        data_root_dir=DATA_ROOT,
        **model_config.to_dict()
    )
    dataset = list(datasets.values())[0]

    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}, Threshold: {thresh}")
        print(f"{'='*60}")

        t_start = time.time()

        evaluator = StreamEvaluator.build(
            model_path=MODEL_PATH,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            fps=2,
            not_talk_threshold=thresh,
            eval_max_seq_len_str="4k",
            context_handling_method="summarize_and_drop",
            force_rerun=True,
        )

        # Run all predictions
        sample_indices = list(range(len(dataset)))
        evaluator.run_all_predictions(sample_indices, progress_bar=True)

        # Compute metrics
        metrics = evaluator.compute_metrics(must_complete=True)
        elapsed = time.time() - t_start
        metrics["elapsed_seconds"] = elapsed

        results[str(thresh)] = metrics
        print(f"F1: {metrics.get('F1', 'N/A'):.4f}, "
              f"Precision: {metrics.get('precision', 'N/A'):.4f}, "
              f"Recall: {metrics.get('recall', 'N/A'):.4f}, "
              f"Time: {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="ProAssist threshold sweep for Adaptive-D")
    parser.add_argument("--dataset", default="wtag/dialog-klg-sum_val_L4096_I1",
                        help="Dataset to sweep (default: wtag/dialog-klg-sum_val_L4096_I1)")
    parser.add_argument("--mode", default="both", choices=["threshold", "diff", "both"],
                        help="Sweep mode: threshold (float), diff (string), or both")
    parser.add_argument("--save_dir", default=RESULT_DIR,
                        help="Directory to save results")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated custom threshold list (overrides --mode)")
    args = parser.parse_args()

    # Load model once (use sdpa since flash_attention_2 is not installed)
    print("Loading model...")
    model_args = ModelArguments(attn_implementation="sdpa")
    model, tokenizer = build_from_checkpoint(MODEL_PATH, model_args=model_args)
    model_config = model.config

    # Determine which thresholds to sweep
    if args.thresholds:
        thresholds = []
        for t in args.thresholds.split(","):
            t = t.strip()
            if t.startswith("diff"):
                thresholds.append(t)
            else:
                thresholds.append(float(t))
    else:
        thresholds = []
        if args.mode in ["threshold", "both"]:
            thresholds.extend(THRESHOLDS)
        if args.mode in ["diff", "both"]:
            thresholds.extend(DIFF_THRESHOLDS)

    # Run sweep
    dataset_short = args.dataset.replace("/", "_")
    results = run_sweep(args.dataset, thresholds, model, tokenizer, model_config)

    # Save results
    save_dir = os.path.join(args.save_dir, dataset_short)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sweep_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Print summary table
    print(f"\n{'Threshold':>10} | {'F1':>8} | {'Precision':>10} | {'Recall':>8} | {'Missing':>8} | {'Redundant':>9}")
    print("-" * 70)
    for thresh, metrics in sorted(results.items(), key=lambda x: str(x[0])):
        print(f"{thresh:>10} | {metrics.get('F1', 0):.4f}   | {metrics.get('precision', 0):.4f}     | "
              f"{metrics.get('recall', 0):.4f}   | {metrics.get('missing_rate', 0):.4f}   | {metrics.get('redundant_rate', 0):.4f}")


if __name__ == "__main__":
    main()
