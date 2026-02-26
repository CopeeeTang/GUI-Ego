"""Adaptive-D analysis for ProAssist threshold optimization.

Reads threshold sweep results, finds per-dataset optimal thresholds,
computes Adaptive-D aggregate F1, and analyzes w2t_prob distributions.

Usage:
    python3 adaptive_d_analysis.py
    python3 adaptive_d_analysis.py --result_dir /path/to/sweep/results
"""

import os
import sys
import json
import argparse

import numpy as np

RESULT_DIR = "/home/v-tangxin/GUI/proassist_experiments/results/threshold_sweep"
ADAPTIVE_D_DIR = "/home/v-tangxin/GUI/proassist_experiments/results/adaptive_d"

# ProAssist paper baselines (from paper Table 2/3)
# F1 values to be filled after running baseline sweeps
PAPER_BASELINES = {
    "wtag": {"threshold": 0.5, "F1": None},
    "ego4d": {"threshold": 0.3, "F1": None},
    "holoassist": {"threshold": 0.4, "F1": None},
    "epickitchens": {"threshold": 0.4, "F1": None},
    "egoexolearn": {"threshold": 0.4, "F1": None},
    "assembly101": {"threshold": 0.4, "F1": None},
}


def find_optimal_threshold(sweep_results):
    """Find the threshold that maximizes F1.

    Args:
        sweep_results: Dict mapping threshold_str -> metrics dict.

    Returns:
        Tuple of (best_threshold_str, best_f1_score).
    """
    best_thresh = None
    best_f1 = -1
    for thresh, metrics in sweep_results.items():
        f1 = metrics.get("F1", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def find_optimal_by_metric(sweep_results, metric_name="F1", maximize=True):
    """Find the threshold that optimizes a given metric.

    Args:
        sweep_results: Dict mapping threshold_str -> metrics dict.
        metric_name: Name of the metric to optimize.
        maximize: If True, maximize; if False, minimize.

    Returns:
        Tuple of (best_threshold_str, best_metric_value).
    """
    best_thresh = None
    best_val = float('-inf') if maximize else float('inf')
    for thresh, metrics in sweep_results.items():
        val = metrics.get(metric_name, 0)
        if (maximize and val > best_val) or (not maximize and val < best_val):
            best_val = val
            best_thresh = thresh
    return best_thresh, best_val


def analyze_w2t_prob_distribution(dataset_dir):
    """Analyze w2t_prob distribution from prediction files.

    Separates w2t_prob values into talk (generated text) vs no-talk (empty)
    frames and computes distribution statistics.

    Args:
        dataset_dir: Directory containing prediction JSON files.

    Returns:
        Dict with distribution statistics.
    """
    talk_probs = []
    notalk_probs = []

    if not os.path.isdir(dataset_dir):
        return {"error": f"Directory not found: {dataset_dir}"}

    for filename in os.listdir(dataset_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(dataset_dir, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if "predictions" not in data:
            continue

        for pred in data["predictions"]:
            w2t_prob = pred.get("w2t_prob")
            if w2t_prob is None:
                continue
            gen = pred.get("gen", "")
            if gen and gen.strip():
                talk_probs.append(w2t_prob)
            else:
                notalk_probs.append(w2t_prob)

    result = {
        "n_talk": len(talk_probs),
        "n_notalk": len(notalk_probs),
    }

    if talk_probs:
        result["talk_mean"] = float(np.mean(talk_probs))
        result["talk_std"] = float(np.std(talk_probs))
        result["talk_median"] = float(np.median(talk_probs))
        result["talk_q25"] = float(np.percentile(talk_probs, 25))
        result["talk_q75"] = float(np.percentile(talk_probs, 75))
    else:
        result["talk_mean"] = None
        result["talk_std"] = None

    if notalk_probs:
        result["notalk_mean"] = float(np.mean(notalk_probs))
        result["notalk_std"] = float(np.std(notalk_probs))
        result["notalk_median"] = float(np.median(notalk_probs))
        result["notalk_q25"] = float(np.percentile(notalk_probs, 25))
        result["notalk_q75"] = float(np.percentile(notalk_probs, 75))
    else:
        result["notalk_mean"] = None
        result["notalk_std"] = None

    if talk_probs and notalk_probs:
        result["separation"] = float(np.mean(notalk_probs) - np.mean(talk_probs))
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(talk_probs) + np.var(notalk_probs)) / 2)
        if pooled_std > 0:
            result["cohens_d"] = float((np.mean(notalk_probs) - np.mean(talk_probs)) / pooled_std)
        else:
            result["cohens_d"] = None
    else:
        result["separation"] = None
        result["cohens_d"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Adaptive-D analysis for ProAssist")
    parser.add_argument("--result_dir", default=RESULT_DIR,
                        help="Directory containing threshold sweep results")
    parser.add_argument("--output_dir", default=ADAPTIVE_D_DIR,
                        help="Directory to save analysis output")
    parser.add_argument("--analyze_probs", action="store_true",
                        help="Also analyze w2t_prob distributions from prediction files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    adaptive_d_results = {}

    # Process each dataset
    if not os.path.isdir(args.result_dir):
        print(f"Result directory not found: {args.result_dir}")
        print("Please run threshold_sweep.py first.")
        return

    for dataset_dir in sorted(os.listdir(args.result_dir)):
        sweep_file = os.path.join(args.result_dir, dataset_dir, "sweep_results.json")
        if not os.path.exists(sweep_file):
            continue

        with open(sweep_file) as f:
            sweep_results = json.load(f)

        best_thresh, best_f1 = find_optimal_threshold(sweep_results)

        # Separate analysis for float-only and diff-only thresholds
        float_results = {k: v for k, v in sweep_results.items() if not k.startswith("diff")}
        diff_results = {k: v for k, v in sweep_results.items() if k.startswith("diff")}

        best_float_thresh, best_float_f1 = find_optimal_threshold(float_results) if float_results else (None, 0)
        best_diff_thresh, best_diff_f1 = find_optimal_threshold(diff_results) if diff_results else (None, 0)

        # Look up paper baseline
        dataset_base = dataset_dir.split("_")[0]
        baseline = PAPER_BASELINES.get(dataset_base, {})
        baseline_thresh = baseline.get("threshold", "N/A")
        baseline_f1_from_sweep = sweep_results.get(str(baseline_thresh), {}).get("F1", None)

        entry = {
            "optimal_threshold": best_thresh,
            "adaptive_d_f1": best_f1,
            "optimal_float_threshold": best_float_thresh,
            "optimal_float_f1": best_float_f1,
            "optimal_diff_threshold": best_diff_thresh,
            "optimal_diff_f1": best_diff_f1,
            "baseline_threshold": baseline_thresh,
            "baseline_f1": baseline_f1_from_sweep,
            "delta_f1": best_f1 - baseline_f1_from_sweep if isinstance(baseline_f1_from_sweep, (int, float)) else None,
            "all_thresholds": {k: v.get("F1", 0) for k, v in sweep_results.items()},
        }

        # Optionally analyze w2t_prob distributions
        if args.analyze_probs:
            # Use the eval dir for the best threshold to find prediction files
            pred_dir = os.path.join(args.result_dir, dataset_dir, "predictions")
            if os.path.isdir(pred_dir):
                entry["w2t_prob_distribution"] = analyze_w2t_prob_distribution(pred_dir)

        adaptive_d_results[dataset_dir] = entry

    if not adaptive_d_results:
        print("No sweep results found. Run threshold_sweep.py first.")
        return

    # Save analysis
    output_file = os.path.join(args.output_dir, "adaptive_d_analysis.json")
    with open(output_file, "w") as f:
        json.dump(adaptive_d_results, f, indent=2)
    print(f"Analysis saved to {output_file}")

    # Print summary
    print(f"\n{'='*90}")
    print("ADAPTIVE-D ANALYSIS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':>20} | {'BL thresh':>9} | {'BL F1':>8} | {'Best thresh':>11} | {'Best F1':>8} | {'Delta F1':>8}")
    print("-" * 90)
    for dataset, result in adaptive_d_results.items():
        delta = result['delta_f1']
        delta_str = f"{delta:+.4f}" if delta is not None else "  N/A"
        bl_f1 = result['baseline_f1']
        bl_f1_str = f"{bl_f1:.4f}" if isinstance(bl_f1, (int, float)) else "  N/A"
        print(f"{dataset:>20} | {str(result['baseline_threshold']):>9} | {bl_f1_str:>8} | "
              f"{str(result['optimal_threshold']):>11} | {result['adaptive_d_f1']:.4f}   | {delta_str:>8}")

    # Print per-threshold F1 table
    print(f"\n{'='*90}")
    print("PER-THRESHOLD F1 SCORES")
    print(f"{'='*90}")
    for dataset, result in adaptive_d_results.items():
        print(f"\n--- {dataset} ---")
        all_t = result["all_thresholds"]
        # Sort: float thresholds first, then diff thresholds
        float_items = sorted(
            [(k, v) for k, v in all_t.items() if not k.startswith("diff")],
            key=lambda x: float(x[0])
        )
        diff_items = sorted(
            [(k, v) for k, v in all_t.items() if k.startswith("diff")],
            key=lambda x: x[0]
        )
        for thresh, f1 in float_items + diff_items:
            marker = " <-- best" if thresh == result["optimal_threshold"] else ""
            baseline_marker = " (baseline)" if str(thresh) == str(result["baseline_threshold"]) else ""
            print(f"  {thresh:>10}: F1 = {f1:.4f}{baseline_marker}{marker}")

    # Compute aggregate Adaptive-D metrics
    if len(adaptive_d_results) > 1:
        print(f"\n{'='*90}")
        print("AGGREGATE ADAPTIVE-D METRICS")
        print(f"{'='*90}")
        ad_f1s = [r["adaptive_d_f1"] for r in adaptive_d_results.values()]
        bl_f1s = [r["baseline_f1"] for r in adaptive_d_results.values()
                  if isinstance(r["baseline_f1"], (int, float))]
        print(f"  Mean Adaptive-D F1: {np.mean(ad_f1s):.4f}")
        if bl_f1s:
            print(f"  Mean Baseline F1:   {np.mean(bl_f1s):.4f}")
            print(f"  Mean Delta F1:      {np.mean(ad_f1s[:len(bl_f1s)]) - np.mean(bl_f1s):+.4f}")


if __name__ == "__main__":
    main()
