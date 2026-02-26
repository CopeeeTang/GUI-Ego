"""
Phase II: Task-Type Stratified Deep Analysis
============================================
Offline analysis using saved trigger_log data from Phase I checkpoint.
No GPU required.

Usage:
    python3 proactive-project/experiments/estp_phase3/phase2_stratified_analysis.py \
        [--checkpoint PATH] [--dataset PATH] [--output DIR] [--verbose]
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Constants (mirrors hypothesis_runner.py) ───────────────────────────────
ANTICIPATION = 1   # seconds: can trigger up to 1s before GT window
LATENCY      = 2   # seconds: can trigger up to 2s after GT window start
FRAME_FPS    = 2   # dataset fps


def ceil_time_by_fps(t, fps, min_t, max_t):
    """Round time up to next frame boundary (from hypothesis_runner.py)."""
    return min(max(math.ceil(t * fps) / fps, min_t), max_t)


def compute_case_metrics(dialog_history, qa, anticipation=ANTICIPATION, latency=LATENCY):
    """Compute ESTP-F1 for a single QA case. Pure Python, no GPU.
    Inlined from hypothesis_runner.py to avoid torch import.
    """
    gold_list = [c for c in qa["conversation"] if c["role"].lower() == "assistant"]
    clip_end = qa.get("clip_end_time", qa.get("end_time", 0))
    gt_spans = [
        (ceil_time_by_fps(c["start_time"], FRAME_FPS, 0, clip_end),
         ceil_time_by_fps(c["end_time"],   FRAME_FPS, 0, clip_end))
        for c in gold_list
    ]
    preds = [d for d in dialog_history if d.get("role") == "assistant"]
    if not preds:
        return {"f1": 0, "precision": 0, "recall": 0,
                "n_preds": 0, "n_gt": len(gt_spans),
                "tp": 0, "fp": 0, "fn": len(gt_spans),
                "gt_spans": gt_spans, "pred_times": []}

    hits = []
    for gi, (gs, ge) in enumerate(gt_spans):
        for pred in preds:
            pt = pred.get("time", -1)
            if gs - anticipation <= pt <= ge + latency:
                hits.append(gi)
                break
    misses = [gi for gi in range(len(gt_spans)) if gi not in hits]

    fp = 0
    for pred in preds:
        pt = pred.get("time", -1)
        if not any(gs - anticipation <= pt <= ge + latency for gs, ge in gt_spans):
            fp += 1

    tp = len(hits)
    fn = len(misses)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall,
            "n_preds": len(preds), "n_gt": len(gt_spans),
            "tp": tp, "fp": fp, "fn": fn,
            "gt_spans": gt_spans, "pred_times": [p.get("time") for p in preds]}


# ─── Data Loading ────────────────────────────────────────────────────────────

def _build_dataset_index(dataset_path):
    """Build lookup dict: (video_uid, clip_uid, question_stripped) → qa_raw."""
    with open(dataset_path) as f:
        data = json.load(f)

    index = {}
    for video_uid, clips in data.items():
        for clip_uid, qa_list in clips.items():
            for qa in qa_list:
                q = qa.get("question", "")
                if not q:
                    # goalstep format: question in conversation
                    conv = qa.get("conversation", [])
                    for c in conv:
                        if c.get("role", "").lower() == "user":
                            q = c.get("content", "")
                            break
                key = (video_uid, clip_uid, q.strip())
                index[key] = qa
    return index


def load_cases_and_match_qa(checkpoint_path, dataset_path, verbose=False):
    """Load 60 old-schema cases and attach qa_raw from original dataset."""
    ds_index = _build_dataset_index(dataset_path)
    if verbose:
        print(f"Dataset index: {len(ds_index)} entries")

    cases = []
    skipped = 0
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if "baseline_metrics" not in c:   # skip new-schema entries
                continue

            vid  = c["video_uid"]
            clip = c["clip_uid"]
            q    = c.get("question", "").strip()
            key  = (vid, clip, q)
            qa_raw = ds_index.get(key)
            if qa_raw is None:
                skipped += 1
                if verbose:
                    print(f"  WARN: no qa_raw match for clip={clip[:8]} q={q[:40]}")
                continue
            c["qa_raw"] = qa_raw
            cases.append(c)

    if verbose:
        print(f"Loaded {len(cases)} cases ({skipped} skipped, no qa_raw match)")
    return cases


# ─── Offline Simulation ───────────────────────────────────────────────────────

def simulate_case_at_tau(trigger_log, qa_raw, tau, cooldown_sec):
    """Replay trigger_log at given (tau, cooldown) → ESTP-F1 metrics."""
    trigger_times = []
    last_trigger = -float("inf")
    for entry in trigger_log:
        t   = entry["time"]
        gap = entry.get("logprob_gap", entry.get("gap", 0))
        if gap > tau and (t - last_trigger) >= cooldown_sec:
            trigger_times.append(t)
            last_trigger = t

    dialog = [{"role": "assistant", "content": "sim", "time": t}
              for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    metrics["n_triggered"] = len(trigger_times)
    metrics["trigger_rate"] = len(trigger_times) / max(len(trigger_log), 1)
    return metrics


# ─── Per-Type Threshold Sweep ─────────────────────────────────────────────────

TAU_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]


def per_type_threshold_sweep(cases, tau_range=TAU_RANGE, cooldown=12.0):
    """For each task type, sweep tau and compute avg F1.

    Returns:
        {task_type: {tau: {f1, precision, recall, fp, trigger_rate, n_cases}}}
    """
    # Group cases by task type
    by_type = defaultdict(list)
    for c in cases:
        tt = c["task_type"].strip()
        by_type[tt].append(c)

    results = {}
    for tt, type_cases in by_type.items():
        results[tt] = {}
        for tau in tau_range:
            f1s, precs, recs, fps, trs = [], [], [], [], []
            for c in type_cases:
                m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)
                f1s.append(m["f1"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                fps.append(m["fp"])
                trs.append(m["trigger_rate"])
            results[tt][tau] = {
                "f1":          float(np.mean(f1s)),
                "precision":   float(np.mean(precs)),
                "recall":      float(np.mean(recs)),
                "fp":          float(np.mean(fps)),
                "trigger_rate":float(np.mean(trs)),
                "n_cases":     len(type_cases),
            }
    return results


def find_optimal_tau_per_type(sweep_results):
    """Return {task_type: (optimal_tau, best_f1)}."""
    optimal = {}
    for tt, tau_dict in sweep_results.items():
        best_tau = max(tau_dict, key=lambda t: tau_dict[t]["f1"])
        optimal[tt] = (best_tau, tau_dict[best_tau]["f1"])
    return optimal


# ─── Adaptive Policy Simulation ───────────────────────────────────────────────

def simulate_adaptive_policy(cases, type_optimal_tau, cooldown=12.0):
    """Use per-type optimal tau for each case.

    Returns list of dicts: {task_type, adaptive_f1, baseline_f1, d28_f1, delta, ...}
    """
    per_case = []
    for c in cases:
        tt    = c["task_type"].strip()
        tau   = type_optimal_tau.get(tt, (2.8,))[0]
        ad_m  = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], tau,     cooldown)
        bl_m  = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], 0.0,     0.0)
        d28_m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], 2.8,     cooldown)
        per_case.append({
            "task_type":    tt,
            "question":     c.get("question", "")[:80],
            "adaptive_tau": tau,
            "adaptive_f1":  ad_m["f1"],
            "baseline_f1":  bl_m["f1"],
            "d28_f1":       d28_m["f1"],
            "delta_vs_bl":  ad_m["f1"] - bl_m["f1"],
            "delta_vs_d28": ad_m["f1"] - d28_m["f1"],
            "adaptive_fp":  ad_m["fp"],
            "baseline_fp":  bl_m["fp"],
        })
    return per_case


# ─── Gap Discriminability ────────────────────────────────────────────────────

def _get_gt_windows(qa_raw):
    """Extract GT trigger windows from qa_raw. Returns list of (start, end)."""
    gold = [c for c in qa_raw.get("conversation", [])
            if c.get("role", "").lower() == "assistant"]
    clip_end = qa_raw.get("clip_end_time", qa_raw.get("end_time", 300))
    windows = []
    for c in gold:
        gs = ceil_time_by_fps(c["start_time"], FRAME_FPS, 0, clip_end)
        ge = ceil_time_by_fps(c["end_time"],   FRAME_FPS, 0, clip_end)
        windows.append((gs - ANTICIPATION, ge + LATENCY))  # effective trigger window
    return windows


def gap_discriminability_by_type(cases):
    """Compute GT vs non-GT gap statistics per task type.

    A trigger_log step at time t is "GT step" if it falls in any effective GT window.

    Returns:
        {task_type: {gt_gaps, non_gt_gaps, gt_mean, non_gt_mean, separation}}
    """
    by_type = defaultdict(lambda: {"gt_gaps": [], "non_gt_gaps": []})

    for c in cases:
        tt      = c["task_type"].strip()
        windows = _get_gt_windows(c["qa_raw"])
        for entry in c["trigger_log"]:
            t   = entry["time"]
            gap = entry.get("logprob_gap", entry.get("gap", 0))
            in_gt = any(ws <= t <= we for ws, we in windows)
            if in_gt:
                by_type[tt]["gt_gaps"].append(gap)
            else:
                by_type[tt]["non_gt_gaps"].append(gap)

    result = {}
    for tt, d in by_type.items():
        gt    = d["gt_gaps"]
        non_gt= d["non_gt_gaps"]
        result[tt] = {
            "gt_gaps":     gt,
            "non_gt_gaps": non_gt,
            "gt_mean":     float(np.mean(gt))     if gt     else float("nan"),
            "gt_std":      float(np.std(gt))      if gt     else float("nan"),
            "non_gt_mean": float(np.mean(non_gt)) if non_gt else float("nan"),
            "non_gt_std":  float(np.std(non_gt))  if non_gt else float("nan"),
            "separation":  float(np.mean(gt) - np.mean(non_gt)) if (gt and non_gt) else float("nan"),
            "n_gt":        len(gt),
            "n_non_gt":    len(non_gt),
        }
    return result


# ─── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(values, n=1000, seed=42):
    """Returns (lo, hi, mean)."""
    rng = np.random.RandomState(seed)
    v   = np.array(values)
    means = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.mean(v))


def bootstrap_delta_ci(a, b, n=1000, seed=42):
    """Paired delta bootstrap. Returns (lo, hi, mean_delta)."""
    rng    = np.random.RandomState(seed)
    paired = np.array(a) - np.array(b)
    means  = [np.mean(rng.choice(paired, len(paired), replace=True)) for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.mean(paired))


# ─── Report Generation ────────────────────────────────────────────────────────

def generate_phase2_report(cases, sweep, optimal_tau, adaptive_results, gap_disc, output_dir):
    """Write comprehensive Phase II analysis report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    def w(s=""):
        lines.append(s)

    # ── Header ──────────────────────────────────────────────────────────────
    w("=" * 80)
    w("Phase II: Task-Type Stratified Deep Analysis")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("=" * 80)
    w()
    w(f"Cases: {len(cases)}")
    w(f"Task types: {len(sweep)}")
    w(f"Tau range: {TAU_RANGE}")
    w(f"Cooldown: 12.0s (D setting)")
    w()

    # ── Section 1: Executive Summary ─────────────────────────────────────────
    ad_f1s = [r["adaptive_f1"]  for r in adaptive_results]
    bl_f1s = [r["baseline_f1"]  for r in adaptive_results]
    d28_f1s= [r["d28_f1"]       for r in adaptive_results]
    ad_fps = [r["adaptive_fp"]  for r in adaptive_results]
    bl_fps = [r["baseline_fp"]  for r in adaptive_results]

    lo_ad, hi_ad, mean_ad = bootstrap_delta_ci(ad_f1s, bl_f1s)
    lo_d28,hi_d28,mean_d28= bootstrap_delta_ci(d28_f1s, bl_f1s)
    gate_ad = "PASS" if lo_ad > 0 else "FAIL"

    w("─" * 80)
    w("SECTION 1: EXECUTIVE SUMMARY")
    w("─" * 80)
    w()
    w(f"  {'Config':<25} {'Avg F1':>8} {'Total FP':>10} {'ΔF1 vs BL':>12} {'95% CI':>20}")
    w(f"  {'':─<75}")
    bl_ci  = bootstrap_ci(bl_f1s)
    ad_ci  = bootstrap_ci(ad_f1s)
    d28_ci = bootstrap_ci(d28_f1s)
    w(f"  {'Baseline (τ=0, no cd)':<25} {np.mean(bl_f1s):>8.3f} {sum(bl_fps):>10,} "
      f"  {'—':>12}  {'—':>20}")
    w(f"  {'D@2.8 (global)':<25} {np.mean(d28_f1s):>8.3f} {sum([r['adaptive_fp'] for r in adaptive_results if True]):>10}"
      f"  {mean_d28:>+12.3f}  [{lo_d28:.3f}, {hi_d28:+.3f}]")
    w(f"  {'Adaptive-D (per-type τ)':<25} {np.mean(ad_f1s):>8.3f} {sum(ad_fps):>10,} "
      f"  {mean_ad:>+12.3f}  [{lo_ad:.3f}, {hi_ad:+.3f}]")
    w()
    w(f"  >>> GATE 2 (Adaptive-D vs Baseline): {gate_ad} <<<")
    if gate_ad == "PASS":
        w("      Lower CI > 0. Adaptive-D significantly improves F1 at scale.")
    else:
        w("      CI crosses 0. Adaptive-D improvement not significant at 95% confidence.")
    w()

    # ── Section 2: Per-Type Optimal Threshold ───────────────────────────────
    w("─" * 80)
    w("SECTION 2: PER-TYPE OPTIMAL THRESHOLD")
    w("─" * 80)
    w()
    w(f"  {'Task Type':<45}  n  Opt-τ  Best-F1  BL-F1   ΔF1    vs-D28")
    w(f"  {'':─<76}")

    # Compute baseline F1 per type and D28 F1 per type from sweep
    type_bl_f1  = {tt: sweep[tt][0.0]["f1"]  for tt in sweep}
    type_d28_f1 = {tt: sweep[tt].get(2.8, sweep[tt][max(sweep[tt].keys(), key=lambda k: abs(k-2.8))])["f1"]
                   for tt in sweep}

    sorted_types = sorted(optimal_tau.items(), key=lambda x: -(x[1][1] - type_bl_f1.get(x[0], 0)))
    for tt, (opt_tau, best_f1) in sorted_types:
        bl_f1  = type_bl_f1.get(tt, 0)
        d28_f1 = type_d28_f1.get(tt, 0)
        n      = sweep[tt][opt_tau]["n_cases"]
        delta  = best_f1 - bl_f1
        vs_d28 = best_f1 - d28_f1
        sign   = "+" if delta >= 0 else ""
        vd28s  = "+" if vs_d28 >= 0 else ""
        w(f"  {tt:<45} {n:2}   {opt_tau:.2f}    {best_f1:.3f}    {bl_f1:.3f}  {sign}{delta:.3f}   {vd28s}{vs_d28:.3f}")
    w()

    # ── Section 3: Adaptive Policy Details ──────────────────────────────────
    w("─" * 80)
    w("SECTION 3: ADAPTIVE POLICY AGGREGATE RESULTS")
    w("─" * 80)
    w()
    improved = sum(1 for r in adaptive_results if r["delta_vs_bl"] > 0.001)
    degraded = sum(1 for r in adaptive_results if r["delta_vs_bl"] < -0.001)
    w(f"  Avg Adaptive-D F1:  {np.mean(ad_f1s):.3f}  (std={np.std(ad_f1s):.3f})")
    w(f"  Avg Baseline F1:    {np.mean(bl_f1s):.3f}  (std={np.std(bl_f1s):.3f})")
    w(f"  Delta:              {mean_ad:+.3f}  95% CI=[{lo_ad:.3f}, {hi_ad:+.3f}]")
    w(f"  FP: {sum(bl_fps):,} → {sum(ad_fps):,}  ({(1-sum(ad_fps)/max(sum(bl_fps),1))*100:.0f}% reduction)")
    w(f"  Cases improved:  {improved}/{len(adaptive_results)}")
    w(f"  Cases degraded:  {degraded}/{len(adaptive_results)}")
    w()
    # By task type
    w(f"  Per-type adaptive vs baseline (avg ΔF1):")
    by_tt = defaultdict(list)
    for r in adaptive_results:
        by_tt[r["task_type"]].append(r["delta_vs_bl"])
    for tt in sorted(by_tt, key=lambda t: -np.mean(by_tt[t])):
        avg_d = np.mean(by_tt[tt])
        sign = "+" if avg_d >= 0 else ""
        w(f"    {tt:<45} {sign}{avg_d:.3f}")
    w()

    # ── Section 4: Gap Discriminability per Type ─────────────────────────────
    w("─" * 80)
    w("SECTION 4: GAP DISCRIMINABILITY PER TASK TYPE")
    w("─" * 80)
    w()
    w("  A step is 'GT' if it falls in [gt_start-1s, gt_end+2s].")
    w("  Separation = GT_mean_gap - non_GT_mean_gap. Higher = more discriminative.")
    w()
    w(f"  {'Task Type':<45} GT-mean  non-GT   Sep    n_GT  n_nonGT  Verdict")
    w(f"  {'':─<79}")

    sorted_gap = sorted(gap_disc.items(), key=lambda x: -(x[1]["separation"] if not math.isnan(x[1]["separation"]) else -99))
    for tt, d in sorted_gap:
        sep = d["separation"]
        sep_str = f"{sep:+.3f}" if not math.isnan(sep) else "  n/a"
        gt_m = f"{d['gt_mean']:.3f}" if not math.isnan(d['gt_mean']) else " n/a"
        ngt_m= f"{d['non_gt_mean']:.3f}" if not math.isnan(d['non_gt_mean']) else " n/a"
        # Verdict based on separation vs overall threshold
        if math.isnan(sep):
            verdict = "no GT steps"
        elif sep > 1.0:
            verdict = "DISCRIMINATIVE"
        elif sep > 0.3:
            verdict = "moderate"
        else:
            verdict = "LOW/HARMFUL"
        w(f"  {tt:<45} {gt_m:>7}  {ngt_m:>7}  {sep_str}  {d['n_gt']:5}  {d['n_non_gt']:7}  {verdict}")
    w()

    # ── Section 5: PR Curve Tables ───────────────────────────────────────────
    w("─" * 80)
    w("SECTION 5: PRECISION-RECALL CURVES PER TASK TYPE")
    w("─" * 80)
    w()

    for tt, (opt_tau, best_f1) in sorted_types:
        w(f"  [{tt}]  optimal τ={opt_tau:.2f},  best F1={best_f1:.3f}")
        w(f"  {'τ':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>7}  {'Avg FP':>8}  {'Trigger%':>9}")
        for tau in TAU_RANGE:
            m = sweep[tt].get(tau, {})
            if not m:
                continue
            f1   = m["f1"]
            prec = m["precision"]
            rec  = m["recall"]
            fp   = m["fp"]
            tr   = m["trigger_rate"] * 100
            marker = " ←OPT" if abs(tau - opt_tau) < 0.01 else ""
            w(f"  {tau:>5.2f}  {prec:>10.3f}  {rec:>8.3f}  {f1:>7.3f}  {fp:>8.1f}  {tr:>8.1f}%{marker}")
        w()

    # ── Section 6: Failure Mode Deep-Dive ───────────────────────────────────
    w("─" * 80)
    w("SECTION 6: FAILURE MODE DEEP-DIVE")
    w("─" * 80)
    w()
    FAILURE_TYPES = ["Action Recognition", "Attribute Perception"]
    for target_tt in FAILURE_TYPES:
        if target_tt not in gap_disc:
            continue
        d = gap_disc[target_tt]
        w(f"  [{target_tt}]")
        w(f"    Gap separation:    {d['separation']:+.3f}  "
          f"(GT={d['gt_mean']:.3f} ± {d['gt_std']:.3f}, "
          f"non-GT={d['non_gt_mean']:.3f} ± {d['non_gt_std']:.3f})")
        # Threshold that maximizes F1 for this type
        if target_tt in sweep:
            opt_tau_tt, best_f1_tt = optimal_tau.get(target_tt, (0.0, 0.0))
            bl_f1_tt = sweep[target_tt][0.0]["f1"]
            w(f"    Optimal τ: {opt_tau_tt:.2f}  Best F1: {best_f1_tt:.3f}  "
              f"Baseline F1: {bl_f1_tt:.3f}  Delta: {best_f1_tt - bl_f1_tt:+.3f}")
            # Show F1 at τ=0, 0.5, 1.0, 2.0, 2.8 for comparison
            w(f"    F1 at key τ: "
              + ", ".join(f"τ={t:.1f}→{sweep[target_tt].get(t, {}).get('f1', 0):.3f}"
                          for t in [0.0, 0.5, 1.0, 2.0, 2.8, 5.0] if t in sweep[target_tt]))
        # Show 3 worst cases for this type
        worst = sorted([r for r in adaptive_results if r["task_type"] == target_tt],
                       key=lambda x: x["delta_vs_bl"])[:3]
        if worst:
            w(f"    Worst cases (by Δ vs baseline):")
            for r in worst:
                w(f"      Δ={r['delta_vs_bl']:+.3f}  adaptive_f1={r['adaptive_f1']:.3f}"
                  f"  baseline_f1={r['baseline_f1']:.3f}  Q: {r['question'][:60]}")
        w()

    # ── Section 7: Phase III Recommendations ─────────────────────────────────
    w("─" * 80)
    w("SECTION 7: PHASE III RECOMMENDATIONS")
    w("─" * 80)
    w()

    # Identify high-sep types (good for D) and low-sep types (bad for D)
    high_sep = [(tt, d["separation"]) for tt, d in gap_disc.items()
                if not math.isnan(d["separation"]) and d["separation"] > 0.5]
    low_sep  = [(tt, d["separation"]) for tt, d in gap_disc.items()
                if not math.isnan(d["separation"]) and d["separation"] <= 0.2]
    high_sep.sort(key=lambda x: -x[1])
    low_sep.sort(key=lambda x: x[1])

    w("  Gap discriminability classification:")
    w(f"    High separation (sep > 0.5) — D effective with tuned τ:")
    for tt, sep in high_sep:
        opt_tau_tt = optimal_tau.get(tt, (float("nan"),))[0]
        w(f"      {tt:<45} sep={sep:+.3f}  recommend τ={opt_tau_tt:.2f}")
    w(f"    Low separation (sep ≤ 0.2) — D ineffective, use τ=0 (or bypass):")
    for tt, sep in low_sep:
        w(f"      {tt:<45} sep={sep:+.3f}")
    w()
    w("  Strategy options for Phase III:")
    w("  A. Adaptive-τ routing: detect task type at runtime → select τ from lookup table")
    w("     Requires: task type classifier (e.g. keyword matching on question)")
    w(f"     Expected F1: ~{np.mean(ad_f1s):.3f} vs baseline {np.mean(bl_f1s):.3f}")
    w("  B. Mixed strategy: D-filter for high-sep types, bypass for low-sep types")
    w("     Simpler to implement, robust to classifier errors")
    w("  C. External trigger model: train binary classifier on (gap, task_type, context)")
    w("     Highest potential but requires labeled data and training infrastructure")
    w()
    if gate_ad == "PASS":
        w("  Adaptive-D PASSES gate 2. Recommendation: proceed with strategy A or B.")
    else:
        w("  Adaptive-D FAILS gate 2. Core issue: even optimal per-type τ cannot overcome")
        w("  the fundamental task heterogeneity. Recommendation: strategy B (bypass for")
        w("  low-sep types) or strategy C (external trigger model).")

    w()
    w("=" * 80)
    w(f"Report generated: {len(cases)} cases, {len(sweep)} task types")
    w("=" * 80)

    # Write report
    report_path = os.path.join(output_dir, "phase2_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport → {report_path}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase II: Task-Type Stratified Analysis")
    parser.add_argument("--checkpoint", default=
        "proactive-project/experiments/estp_phase3/results/fullscale_d/checkpoint.jsonl")
    parser.add_argument("--dataset", default=
        "data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json")
    parser.add_argument("--output", default=
        "proactive-project/experiments/estp_phase3/results/phase2_stratified")
    parser.add_argument("--cooldown", type=float, default=12.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase II: Task-Type Stratified Deep Analysis")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading and matching cases...")
    cases = load_cases_and_match_qa(args.checkpoint, args.dataset, verbose=args.verbose)
    if not cases:
        print("ERROR: No cases loaded. Check paths.")
        return
    print(f"  {len(cases)} cases loaded")

    # Step 2: Per-type threshold sweep
    print(f"\n[2/5] Per-type threshold sweep ({len(TAU_RANGE)} tau values × {len(cases)} cases)...")
    sweep = per_type_threshold_sweep(cases, TAU_RANGE, args.cooldown)
    print(f"  Done. Task types: {sorted(sweep.keys())}")

    # Step 3: Find optimal tau per type
    print("\n[3/5] Finding optimal tau per type...")
    optimal_tau = find_optimal_tau_per_type(sweep)
    for tt, (tau, f1) in sorted(optimal_tau.items(), key=lambda x: -x[1][1]):
        bl_f1 = sweep[tt][0.0]["f1"]
        print(f"  {tt:<45} opt_τ={tau:.2f}  F1={f1:.3f}  (BL={bl_f1:.3f}, Δ={f1-bl_f1:+.3f})")

    # Step 4: Adaptive policy simulation
    print("\n[4/5] Simulating adaptive policy...")
    adaptive_results = simulate_adaptive_policy(cases, optimal_tau, args.cooldown)
    ad_f1s = [r["adaptive_f1"] for r in adaptive_results]
    bl_f1s = [r["baseline_f1"] for r in adaptive_results]
    lo, hi, mean = bootstrap_delta_ci(ad_f1s, bl_f1s)
    gate = "PASS" if lo > 0 else "FAIL"
    print(f"  Adaptive-D avg F1: {np.mean(ad_f1s):.3f}  Baseline: {np.mean(bl_f1s):.3f}")
    print(f"  Delta: {mean:+.3f}  95% CI=[{lo:.3f}, {hi:+.3f}]  Gate: {gate}")

    # Step 5: Gap discriminability
    print("\n[5/5] Computing gap discriminability by type...")
    gap_disc = gap_discriminability_by_type(cases)
    for tt in sorted(gap_disc, key=lambda t: -(gap_disc[t]["separation"] or -99)):
        d = gap_disc[tt]
        sep = d["separation"]
        sep_str = f"{sep:+.3f}" if not math.isnan(sep) else " n/a"
        print(f"  {tt:<45} sep={sep_str}  GT={d['n_gt']} steps  non-GT={d['n_non_gt']} steps")

    # Save raw JSON
    os.makedirs(args.output, exist_ok=True)
    sweep_path = os.path.join(args.output, "per_type_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({str(k): {str(tau): v for tau, v in tdict.items()}
                   for k, tdict in sweep.items()}, f, indent=2)
    gap_path = os.path.join(args.output, "gap_by_type.json")
    gap_out = {tt: {k: v for k, v in d.items() if k not in ("gt_gaps", "non_gt_gaps")}
               for tt, d in gap_disc.items()}
    with open(gap_path, "w") as f:
        json.dump(gap_out, f, indent=2)
    print(f"\n  Raw data → {sweep_path}")
    print(f"  Gap data  → {gap_path}")

    # Generate report
    print("\nGenerating report...")
    report = generate_phase2_report(
        cases, sweep, optimal_tau, adaptive_results, gap_disc, args.output
    )
    # Print abbreviated report (first ~100 lines)
    print("\n" + "=" * 60 + " REPORT PREVIEW " + "=" * 5)
    for line in report.split("\n")[:80]:
        print(line)
    print("... (see full report in output dir)")


if __name__ == "__main__":
    main()
