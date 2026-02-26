#!/usr/bin/env python3
"""
Offline Threshold Sweep for Hypothesis D (Logprob-Based Trigger).

Re-simulates different logprob gap thresholds (τ) and cooldown values on
existing inference data (hypothesis_D_raw.json), computing F1/FP/Recall
for each setting WITHOUT any GPU re-inference.

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/threshold_sweep.py

    # Custom ranges:
    python3 proactive-project/experiments/estp_phase3/threshold_sweep.py \
        --tau_min 0.0 --tau_max 6.0 --tau_step 0.25 \
        --cooldown 0 8 12 --verbose
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RAW_FILE = RESULTS_DIR / "hypothesis_D_raw.json"

# ESTP-Bench evaluation parameters
ANTICIPATION = 1
LATENCY = 2


def ceil_time_by_fps(time_val, fps, start, end):
    """Round time to nearest frame boundary."""
    if fps <= 0:
        return time_val
    frame_idx = round(time_val * fps)
    return max(start, min(frame_idx / fps, end))


def simulate_trigger(trigger_log, threshold, cooldown_sec=0.0):
    """Re-simulate trigger decisions for given threshold + cooldown.

    Three-segment rule:
      gap < 0:           REJECT
      0 <= gap <= tau:    REJECT
      gap > tau:          TRIGGER (subject to cooldown)

    Returns list of trigger times.
    """
    trigger_times = []
    last_trigger_time = -float("inf")

    for entry in trigger_log:
        t = entry["time"]
        gap = entry["logprob_gap"]

        # Three-segment rule
        would_trigger = gap > threshold

        # Cooldown
        in_cooldown = (t - last_trigger_time) < cooldown_sec

        if would_trigger and not in_cooldown:
            trigger_times.append(t)
            last_trigger_time = t

    return trigger_times


def compute_metrics(trigger_times, gt_spans, anticipation=ANTICIPATION, latency=LATENCY):
    """Compute F1, precision, recall, FP from trigger times and GT spans."""
    if not trigger_times:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "n_preds": 0, "tp": 0, "fp": 0,
            "n_gt": len(gt_spans), "fn": len(gt_spans),
        }

    # True positives: GT windows hit by at least one prediction
    hits = []
    for gi, (gs, ge) in enumerate(gt_spans):
        for pt in trigger_times:
            if gs - anticipation <= pt <= ge + latency:
                hits.append(gi)
                break

    # False positives: predictions not in any GT window
    fp = 0
    for pt in trigger_times:
        in_any = False
        for gs, ge in gt_spans:
            if gs - anticipation <= pt <= ge + latency:
                in_any = True
                break
        if not in_any:
            fp += 1

    tp = len(hits)
    fn = len(gt_spans) - tp
    n_preds = len(trigger_times)

    precision = tp / n_preds if n_preds > 0 else 0.0
    recall = tp / len(gt_spans) if gt_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "n_preds": n_preds, "tp": tp, "fp": fp,
        "n_gt": len(gt_spans), "fn": fn,
    }


def run_sweep(raw_data, tau_values, cooldown_values, verbose=False):
    """Run full threshold sweep.

    Returns: list of dicts, each with {tau, cooldown, per_case_metrics, aggregate}.
    """
    # Extract valid cases with hypothesis trigger logs
    valid_cases = []
    for r in raw_data:
        if r.get("status") != "ok":
            continue
        hyp = r.get("hypothesis", {})
        tlog = hyp.get("trigger_log", [])
        if not tlog or "logprob_gap" not in tlog[0]:
            continue
        valid_cases.append(r)

    if not valid_cases:
        print("ERROR: No valid cases with logprob data found!")
        return []

    print(f"Found {len(valid_cases)} valid cases with logprob data")
    for i, c in enumerate(valid_cases):
        case = c["case"]
        print(f"  Case {i+1}: {case['task_type']} - {case['question'][:50]}...")
        print(f"    GT windows: {c['n_gt']}, Duration: {c['duration']:.0f}s")

    sweep_results = []

    for cooldown in cooldown_values:
        for tau in tau_values:
            per_case = []
            for case_data in valid_cases:
                hyp_log = case_data["hypothesis"]["trigger_log"]
                gt_spans = [tuple(s) for s in case_data["gt_spans"]]

                trigger_times = simulate_trigger(hyp_log, tau, cooldown)
                metrics = compute_metrics(trigger_times, gt_spans)

                # Also compute baseline for comparison
                baseline_preds = [
                    d["time"] for d in case_data["baseline"]["dialog"]
                    if d.get("role") == "assistant"
                ]
                baseline_metrics = compute_metrics(baseline_preds, gt_spans)

                per_case.append({
                    "case": case_data["case"],
                    "task_type": case_data["case"]["task_type"],
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "delta_f1": metrics["f1"] - baseline_metrics["f1"],
                    "n_triggers": len(trigger_times),
                    "trigger_rate": len(trigger_times) / len(hyp_log) if hyp_log else 0,
                })

            # Aggregate
            f1s = [c["metrics"]["f1"] for c in per_case]
            fps = [c["metrics"]["fp"] for c in per_case]
            recalls = [c["metrics"]["recall"] for c in per_case]
            deltas = [c["delta_f1"] for c in per_case]
            trigger_rates = [c["trigger_rate"] for c in per_case]
            total_preds = sum(c["metrics"]["n_preds"] for c in per_case)
            total_tp = sum(c["metrics"]["tp"] for c in per_case)
            total_gt = sum(c["metrics"]["n_gt"] for c in per_case)

            agg = {
                "tau": tau,
                "cooldown": cooldown,
                "avg_f1": np.mean(f1s),
                "std_f1": np.std(f1s),
                "avg_fp": np.mean(fps),
                "total_fp": sum(fps),
                "avg_recall": np.mean(recalls),
                "avg_delta_f1": np.mean(deltas),
                "avg_trigger_rate": np.mean(trigger_rates),
                "total_preds": total_preds,
                "total_tp": total_tp,
                "total_gt": total_gt,
                "macro_precision": total_tp / total_preds if total_preds > 0 else 0,
                "macro_recall": total_tp / total_gt if total_gt > 0 else 0,
            }

            sweep_results.append({
                "aggregate": agg,
                "per_case": per_case,
            })

            if verbose:
                print(f"  τ={tau:.2f} cd={cooldown:.0f}s: "
                      f"F1={agg['avg_f1']:.3f} FP={agg['total_fp']} "
                      f"Recall={agg['avg_recall']:.3f} "
                      f"TrigRate={agg['avg_trigger_rate']:.1%} "
                      f"Preds={agg['total_preds']}")

    return sweep_results


def format_report(sweep_results, cooldown_values):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Hypothesis D: Logprob Threshold Sweep Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Three-segment rule: gap<0 → REJECT | 0≤gap≤τ → REJECT | gap>τ → TRIGGER")
    lines.append("")

    for cooldown in cooldown_values:
        lines.append(f"--- Cooldown = {cooldown:.0f}s ---")
        lines.append(f"{'τ':>6s} | {'F1':>6s} | {'ΔF1':>7s} | {'FP':>4s} | "
                      f"{'Recall':>6s} | {'Prec':>6s} | {'Preds':>5s} | {'TrigRate':>8s}")
        lines.append("-" * 70)

        cd_results = [r for r in sweep_results if r["aggregate"]["cooldown"] == cooldown]
        # Sort by tau
        cd_results.sort(key=lambda x: x["aggregate"]["tau"])

        best_f1 = 0
        best_tau = None
        for r in cd_results:
            a = r["aggregate"]
            if a["avg_f1"] > best_f1:
                best_f1 = a["avg_f1"]
                best_tau = a["tau"]

            marker = " ★" if a["avg_f1"] == best_f1 and a["tau"] == best_tau else ""
            lines.append(
                f"{a['tau']:6.2f} | {a['avg_f1']:6.3f} | {a['avg_delta_f1']:+7.3f} | "
                f"{a['total_fp']:4d} | {a['avg_recall']:6.3f} | "
                f"{a['macro_precision']:6.3f} | {a['total_preds']:5d} | "
                f"{a['avg_trigger_rate']:7.1%}{marker}"
            )

        if best_tau is not None:
            lines.append(f"\n  Best τ = {best_tau:.2f} (F1 = {best_f1:.3f}) for cd={cooldown:.0f}s")
        lines.append("")

    # Per-case detail for the best overall setting
    all_results = sorted(sweep_results, key=lambda x: -x["aggregate"]["avg_f1"])
    if all_results:
        best = all_results[0]
        ba = best["aggregate"]
        lines.append(f"\n{'='*80}")
        lines.append(f"Best Overall: τ={ba['tau']:.2f}, cooldown={ba['cooldown']:.0f}s")
        lines.append(f"  Avg F1={ba['avg_f1']:.3f}, ΔF1={ba['avg_delta_f1']:+.3f}, "
                      f"FP={ba['total_fp']}, Recall={ba['avg_recall']:.3f}")
        lines.append(f"{'='*80}")
        lines.append("")

        for i, pc in enumerate(best["per_case"]):
            c = pc["case"]
            m = pc["metrics"]
            bm = pc["baseline_metrics"]
            lines.append(f"  Case {i+1}: {c['task_type']}")
            lines.append(f"  Q: {c['question'][:70]}")
            lines.append(f"  Baseline: F1={bm['f1']:.3f}, Preds={bm['n_preds']}, "
                          f"TP={bm['tp']}, FP={bm['fp']}")
            lines.append(f"  Hypo-D:   F1={m['f1']:.3f}, Preds={m['n_preds']}, "
                          f"TP={m['tp']}, FP={m['fp']}")
            lines.append(f"  Delta:    {pc['delta_f1']:+.3f}  "
                          f"TrigRate={pc['trigger_rate']:.1%}")
            lines.append("")

    # Logprob gap distribution analysis
    lines.append(f"\n{'='*80}")
    lines.append("Logprob Gap Distribution (from raw data)")
    lines.append(f"{'='*80}")

    return "\n".join(lines)


def analyze_gap_distribution(raw_data):
    """Analyze logprob gap distribution w.r.t. GT windows."""
    lines = []
    for r in raw_data:
        if r.get("status") != "ok":
            continue
        hyp = r.get("hypothesis", {})
        tlog = hyp.get("trigger_log", [])
        if not tlog or "logprob_gap" not in tlog[0]:
            continue

        gt_spans = [tuple(s) for s in r["gt_spans"]]
        case = r["case"]

        gt_gaps = []
        non_gt_gaps = []

        for entry in tlog:
            t = entry["time"]
            gap = entry["logprob_gap"]
            in_gt = any(gs - ANTICIPATION <= t <= ge + LATENCY for gs, ge in gt_spans)
            if in_gt:
                gt_gaps.append(gap)
            else:
                non_gt_gaps.append(gap)

        lines.append(f"\n  Case: {case['task_type']} - {case['question'][:50]}...")
        if gt_gaps:
            lines.append(f"    GT-window gaps:     n={len(gt_gaps):3d}, "
                          f"mean={np.mean(gt_gaps):+.2f}, "
                          f"std={np.std(gt_gaps):.2f}, "
                          f"range=[{min(gt_gaps):.2f}, {max(gt_gaps):.2f}]")
        else:
            lines.append(f"    GT-window gaps:     n=0 (no frames in GT windows)")
        if non_gt_gaps:
            lines.append(f"    Non-GT-window gaps: n={len(non_gt_gaps):3d}, "
                          f"mean={np.mean(non_gt_gaps):+.2f}, "
                          f"std={np.std(non_gt_gaps):.2f}, "
                          f"range=[{min(non_gt_gaps):.2f}, {max(non_gt_gaps):.2f}]")
        if gt_gaps and non_gt_gaps:
            separation = np.mean(gt_gaps) - np.mean(non_gt_gaps)
            lines.append(f"    Separation (GT - non-GT mean): {separation:+.2f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Offline threshold sweep for Hypothesis D")
    parser.add_argument("--raw_file", type=str, default=str(RAW_FILE),
                        help="Path to hypothesis_D_raw.json")
    parser.add_argument("--tau_min", type=float, default=0.0)
    parser.add_argument("--tau_max", type=float, default=6.0)
    parser.add_argument("--tau_step", type=float, default=0.25)
    parser.add_argument("--cooldown", type=float, nargs="+", default=[0, 8, 12],
                        help="Cooldown values to sweep (seconds)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.raw_file}")
    with open(args.raw_file) as f:
        raw_data = json.load(f)

    tau_values = np.arange(args.tau_min, args.tau_max + args.tau_step / 2, args.tau_step)
    print(f"Sweeping τ from {args.tau_min} to {args.tau_max} (step {args.tau_step}), "
          f"{len(tau_values)} values")
    print(f"Cooldown values: {args.cooldown}")
    print()

    # Run sweep
    sweep_results = run_sweep(raw_data, tau_values, args.cooldown, verbose=args.verbose)

    # Generate report
    report = format_report(sweep_results, args.cooldown)
    gap_analysis = analyze_gap_distribution(raw_data)
    full_report = report + gap_analysis

    # Save
    report_path = RESULTS_DIR / "threshold_sweep_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\nReport saved to: {report_path}")

    # Also save structured results
    json_path = RESULTS_DIR / "threshold_sweep_results.json"
    # Convert to serializable
    serializable = []
    for r in sweep_results:
        sr = {
            "aggregate": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in r["aggregate"].items()},
            "per_case_summary": [
                {
                    "task_type": pc["task_type"],
                    "f1": pc["metrics"]["f1"],
                    "fp": pc["metrics"]["fp"],
                    "recall": pc["metrics"]["recall"],
                    "delta_f1": pc["delta_f1"],
                    "trigger_rate": pc["trigger_rate"],
                }
                for pc in r["per_case"]
            ],
        }
        serializable.append(sr)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Structured results saved to: {json_path}")

    # Print report
    print("\n" + full_report)


if __name__ == "__main__":
    main()
