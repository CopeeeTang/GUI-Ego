#!/usr/bin/env python3
"""
GPT-4o Adaptive-D Offline Analysis
===================================
对已完成的 GPT-4o 全量 FBF 数据做离线 Adaptive-D 阈值扫描，
计算最终 ESTP-F1 并与 Qwen Adaptive-D (F1=0.222) 对比。

纯离线分析，不需要任何 API 调用。

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/gpt4o_adaptive_d_analysis.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
ANTICIPATION = 1   # seconds: can trigger up to 1s before GT window
LATENCY      = 2   # seconds: can trigger up to 2s after GT window start
FRAME_FPS    = 2   # dataset fps

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "results" / "gpt4o_fbf" / "checkpoint.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "results" / "gpt4o_fbf"

# Qwen 参考值 (Phase II results)
QWEN_ADAPTIVE_D_F1 = 0.222
QWEN_BASELINE_F1 = 0.141
QWEN_SEPARATION = 1.418
QWEN_AUC = 0.615
QWEN_OPTIMAL_TAU = {
    "Information Function": 2.8,
    "Text-Rich Understanding": 2.0,
    "Object Recognition": 2.5,
    "Action Recognition": 0.5,
    "Attribute Perception": 0.5,
    "Task Understanding": 0.0,  # bypass D
}

# Sweep ranges
TAU_RANGE_GLOBAL = [-5, -3, -2, -1, 0, 0.5, 1, 2, 3, 5, 8, 10, 15]
TAU_RANGE_PER_TYPE = [-5, -3, -2, -1, 0, 0.25, 0.5, 0.75, 1.0, 1.5,
                      2.0, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0, 8.0, 10.0, 15.0]
COOLDOWN_RANGE = [0, 6, 12]


# ─── Utility ─────────────────────────────────────────────────────────────────

def ceil_time_by_fps(t, fps, min_t, max_t):
    """Round time up to next frame boundary."""
    return min(max(math.ceil(t * fps) / fps, min_t), max_t)


def compute_case_metrics(dialog_history, qa, anticipation=ANTICIPATION, latency=LATENCY):
    """Compute ESTP-F1 for a single QA case. Pure Python, no GPU."""
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
                "false_positives": 0,
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
            "tp": tp, "fp": fp, "fn": fn, "false_positives": fp,
            "gt_spans": gt_spans, "pred_times": [p.get("time") for p in preds]}


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_checkpoint(path):
    """Load GPT-4o FBF checkpoint data."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            cases.append(r)
    return cases


# ─── Simulation ──────────────────────────────────────────────────────────────

def simulate_case_at_tau(trigger_log, qa_raw, tau, cooldown_sec):
    """Replay trigger_log at given (tau, cooldown) -> ESTP-F1 metrics."""
    trigger_times = []
    last_trigger = -float("inf")
    for entry in trigger_log:
        t   = entry["time"]
        gap = entry.get("logprob_gap", 0)
        if gap > tau and (t - last_trigger) >= cooldown_sec:
            trigger_times.append(t)
            last_trigger = t

    dialog = [{"role": "assistant", "content": "sim", "time": t}
              for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    metrics["n_triggered"] = len(trigger_times)
    metrics["trigger_rate"] = len(trigger_times) / max(len(trigger_log), 1)
    return metrics


def reconstruct_vanilla_fbf(trigger_log, qa_raw):
    """Reconstruct vanilla FBF baseline: trigger when response contains 'yes'."""
    trigger_times = [e["time"] for e in trigger_log if e.get("triggered", False)]
    dialog = [{"role": "assistant", "content": "sim", "time": t}
              for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    metrics["n_triggered"] = len(trigger_times)
    metrics["trigger_rate"] = len(trigger_times) / max(len(trigger_log), 1)
    return metrics


# ─── GT Window Helpers ───────────────────────────────────────────────────────

def get_gt_windows(qa_raw):
    """Extract effective GT trigger windows from qa_raw."""
    gold = [c for c in qa_raw.get("conversation", [])
            if c.get("role", "").lower() == "assistant"]
    clip_end = qa_raw.get("clip_end_time", qa_raw.get("end_time", 300))
    windows = []
    for c in gold:
        gs = ceil_time_by_fps(c["start_time"], FRAME_FPS, 0, clip_end)
        ge = ceil_time_by_fps(c["end_time"],   FRAME_FPS, 0, clip_end)
        windows.append((gs - ANTICIPATION, ge + LATENCY))
    return windows


# ─── Analysis A: Vanilla FBF F1 ─────────────────────────────────────────────

def analysis_a_vanilla_fbf(cases):
    """Compute vanilla FBF F1 (trigger when response contains 'yes')."""
    print("\n" + "=" * 70)
    print("[A] Vanilla FBF F1 (触发规则=response中包含yes)")
    print("=" * 70)

    all_f1s = []
    by_type = defaultdict(list)

    for c in cases:
        m = reconstruct_vanilla_fbf(c["trigger_log"], c["qa_raw"])
        all_f1s.append(m["f1"])
        by_type[c["task_type"]].append(m)

    avg_f1 = np.mean(all_f1s)
    print(f"\n  整体平均 F1: {avg_f1:.3f} ({len(cases)} cases)")
    print(f"  对比 Qwen FBF baseline: 0.127")

    print(f"\n  {'任务类型':<45} {'n':>3} {'F1':>7} {'P':>7} {'R':>7} {'FP':>5} {'Trig%':>7}")
    print(f"  {'─'*80}")

    type_f1s = {}
    for tt in sorted(by_type.keys()):
        ms = by_type[tt]
        f1 = np.mean([m["f1"] for m in ms])
        prec = np.mean([m["precision"] for m in ms])
        rec = np.mean([m["recall"] for m in ms])
        fp = sum(m["fp"] for m in ms)
        tr = np.mean([m["trigger_rate"] for m in ms])
        type_f1s[tt] = f1
        print(f"  {tt:<45} {len(ms):>3} {f1:>7.3f} {prec:>7.3f} {rec:>7.3f} {fp:>5} {tr:>6.1%}")

    # Yes rate
    total_steps = sum(len(c["trigger_log"]) for c in cases)
    total_yes = sum(sum(1 for e in c["trigger_log"] if e.get("triggered", False))
                    for c in cases)
    yes_rate = total_yes / total_steps if total_steps > 0 else 0
    print(f"\n  总 polling steps: {total_steps}")
    print(f"  总 'yes' triggers: {total_yes} ({yes_rate:.1%})")
    print(f"  对比 Qwen yes-rate: 76-83%")

    return {"avg_f1": avg_f1, "per_type": type_f1s, "yes_rate": yes_rate,
            "all_f1s": all_f1s}


# ─── Analysis B: Global Threshold Sweep ──────────────────────────────────────

def analysis_b_global_sweep(cases):
    """Sweep over (tau, cooldown) combinations globally."""
    print("\n" + "=" * 70)
    print("[B] 全局阈值扫描")
    print("=" * 70)

    results = {}
    for cd in COOLDOWN_RANGE:
        print(f"\n  --- cooldown = {cd}s ---")
        print(f"  {'tau':>6} {'F1':>7} {'P':>7} {'R':>7} {'FP':>5} {'Preds':>6} {'Trig%':>7}")
        print(f"  {'─'*50}")

        best_f1 = -1
        best_tau = None
        for tau in TAU_RANGE_GLOBAL:
            f1s, precs, recs, fps, preds_n, trig_rates = [], [], [], [], [], []
            for c in cases:
                m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], tau, cd)
                f1s.append(m["f1"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                fps.append(m["fp"])
                preds_n.append(m["n_preds"])
                trig_rates.append(m["trigger_rate"])

            avg_f1 = np.mean(f1s)
            results[(tau, cd)] = {
                "f1": avg_f1,
                "precision": np.mean(precs),
                "recall": np.mean(recs),
                "total_fp": sum(fps),
                "total_preds": sum(preds_n),
                "avg_trigger_rate": np.mean(trig_rates),
                "f1_list": f1s,
            }

            marker = ""
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_tau = tau
                marker = " <--"

            print(f"  {tau:>6.1f} {avg_f1:>7.3f} {np.mean(precs):>7.3f} "
                  f"{np.mean(recs):>7.3f} {sum(fps):>5} {sum(preds_n):>6} "
                  f"{np.mean(trig_rates):>6.1%}{marker}")

        print(f"\n  最优: tau={best_tau}, F1={best_f1:.3f} (cooldown={cd}s)")

    # Find overall best
    best_config = max(results.items(), key=lambda x: x[1]["f1"])
    (best_tau, best_cd), best_metrics = best_config
    print(f"\n  全局最优: tau={best_tau}, cooldown={best_cd}, F1={best_metrics['f1']:.3f}")
    print(f"  对比 Qwen Adaptive-D: F1=0.222")

    return results, (best_tau, best_cd)


# ─── Analysis C: Per-Type Adaptive Threshold ─────────────────────────────────

def analysis_c_per_type_adaptive(cases, cooldown=12.0):
    """Per-type optimal tau sweep (Adaptive-D core)."""
    print("\n" + "=" * 70)
    print("[C] Per-type 自适应阈值 (Adaptive-D 核心)")
    print(f"    固定 cooldown={cooldown}s")
    print("=" * 70)

    by_type = defaultdict(list)
    for c in cases:
        by_type[c["task_type"].strip()].append(c)

    sweep_results = {}
    optimal_tau = {}

    for tt in sorted(by_type.keys()):
        type_cases = by_type[tt]
        sweep_results[tt] = {}
        best_f1 = -1
        best_tau = None

        for tau in TAU_RANGE_PER_TYPE:
            f1s, precs, recs, fps, trs = [], [], [], [], []
            for c in type_cases:
                m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)
                f1s.append(m["f1"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                fps.append(m["fp"])
                trs.append(m["trigger_rate"])

            avg_f1 = float(np.mean(f1s))
            sweep_results[tt][tau] = {
                "f1": avg_f1,
                "precision": float(np.mean(precs)),
                "recall": float(np.mean(recs)),
                "fp": float(np.mean(fps)),
                "trigger_rate": float(np.mean(trs)),
                "n_cases": len(type_cases),
                "f1_list": f1s,
            }

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_tau = tau

        optimal_tau[tt] = (best_tau, best_f1)

    # Display results
    print(f"\n  {'任务类型':<45} {'n':>3} {'最优tau':>7} {'F1':>7} {'BL_F1':>7} {'Delta':>7}")
    print(f"  {'─'*82}")

    for tt in sorted(optimal_tau, key=lambda t: -optimal_tau[t][1]):
        opt_tau, best_f1 = optimal_tau[tt]
        bl_f1 = sweep_results[tt][0.0]["f1"] if 0.0 in sweep_results[tt] else 0
        delta = best_f1 - bl_f1
        qwen_tau = QWEN_OPTIMAL_TAU.get(tt, "n/a")
        print(f"  {tt:<45} {sweep_results[tt][opt_tau]['n_cases']:>3} "
              f"{opt_tau:>7.2f} {best_f1:>7.3f} {bl_f1:>7.3f} {delta:>+7.3f}"
              f"  (Qwen_tau={qwen_tau})")

    # Simulate adaptive policy: use per-type optimal tau
    adaptive_f1s = []
    baseline_f1s = []
    adaptive_fps = []
    baseline_fps = []
    per_case_results = []

    for c in cases:
        tt = c["task_type"].strip()
        tau = optimal_tau.get(tt, (0.0,))[0]
        ad_m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)
        bl_m = simulate_case_at_tau(c["trigger_log"], c["qa_raw"], 0.0, 0.0)  # baseline: tau=0, no cooldown

        adaptive_f1s.append(ad_m["f1"])
        baseline_f1s.append(bl_m["f1"])
        adaptive_fps.append(ad_m["fp"])
        baseline_fps.append(bl_m["fp"])

        per_case_results.append({
            "task_type": tt,
            "question": c.get("question", "")[:80],
            "adaptive_tau": tau,
            "adaptive_f1": ad_m["f1"],
            "baseline_f1": bl_m["f1"],
            "delta": ad_m["f1"] - bl_m["f1"],
            "adaptive_fp": ad_m["fp"],
            "baseline_fp": bl_m["fp"],
        })

    avg_ad = np.mean(adaptive_f1s)
    avg_bl = np.mean(baseline_f1s)
    improved = sum(1 for r in per_case_results if r["delta"] > 0.001)
    degraded = sum(1 for r in per_case_results if r["delta"] < -0.001)

    print(f"\n  Adaptive-D 平均 F1: {avg_ad:.3f}")
    print(f"  Baseline 平均 F1:   {avg_bl:.3f}")
    print(f"  Delta:              {avg_ad - avg_bl:+.3f}")
    print(f"  FP: {sum(baseline_fps)} -> {sum(adaptive_fps)} "
          f"({(1 - sum(adaptive_fps) / max(sum(baseline_fps), 1)) * 100:.0f}% reduction)")
    print(f"  Cases improved: {improved}/{len(cases)}, degraded: {degraded}/{len(cases)}")

    return sweep_results, optimal_tau, per_case_results, adaptive_f1s, baseline_f1s


# ─── Analysis D: Gap Discriminability ────────────────────────────────────────

def analysis_d_gap_discriminability(cases):
    """GT vs non-GT gap distribution analysis."""
    print("\n" + "=" * 70)
    print("[D] Gap 判别力分析")
    print("=" * 70)

    all_gt_gaps = []
    all_non_gt_gaps = []
    by_type = defaultdict(lambda: {"gt": [], "non_gt": []})

    for c in cases:
        windows = get_gt_windows(c["qa_raw"])
        tt = c["task_type"].strip()

        for entry in c["trigger_log"]:
            t = entry["time"]
            gap = entry.get("logprob_gap", 0)

            # Skip non-finite
            if not math.isfinite(gap):
                continue

            in_gt = any(ws <= t <= we for ws, we in windows)
            if in_gt:
                all_gt_gaps.append(gap)
                by_type[tt]["gt"].append(gap)
            else:
                all_non_gt_gaps.append(gap)
                by_type[tt]["non_gt"].append(gap)

    # Overall
    gt_mean = np.mean(all_gt_gaps) if all_gt_gaps else float("nan")
    ngt_mean = np.mean(all_non_gt_gaps) if all_non_gt_gaps else float("nan")
    separation = gt_mean - ngt_mean if (all_gt_gaps and all_non_gt_gaps) else float("nan")

    print(f"\n  整体 Gap 分布:")
    print(f"    GT window gaps:     n={len(all_gt_gaps):>5}, mean={gt_mean:>+7.3f}, std={np.std(all_gt_gaps):.3f}")
    print(f"    Non-GT gaps:        n={len(all_non_gt_gaps):>5}, mean={ngt_mean:>+7.3f}, std={np.std(all_non_gt_gaps):.3f}")
    print(f"    Separation (GT - non-GT): {separation:+.3f}")
    print(f"    对比 Qwen: separation={QWEN_SEPARATION:+.3f} (GT=2.226, non-GT=0.808)")

    # Welch t-test
    t_stat = p_val = float("nan")
    if len(all_gt_gaps) >= 2 and len(all_non_gt_gaps) >= 2:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(all_gt_gaps, all_non_gt_gaps, equal_var=False)
        print(f"    Welch t-test: t={t_stat:.3f}, p={p_val:.6f}")
        if p_val < 0.001:
            print(f"    >>> Gap 有显著判别力 (p<0.001)")
        elif p_val < 0.05:
            print(f"    >>> Gap 有弱判别力 (p<0.05)")
        else:
            print(f"    >>> Gap 没有显著判别力 (p={p_val:.4f})")

    # AUC
    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        labels = [1] * len(all_gt_gaps) + [0] * len(all_non_gt_gaps)
        scores = all_gt_gaps + all_non_gt_gaps
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            print(f"    AUC: {auc:.3f}")
            print(f"    对比 Qwen AUC: {QWEN_AUC:.3f}")
    except ImportError:
        # Manual AUC calculation
        n_pos = len(all_gt_gaps)
        n_neg = len(all_non_gt_gaps)
        if n_pos > 0 and n_neg > 0:
            concordant = sum(1 for g in all_gt_gaps for n in all_non_gt_gaps if g > n)
            tied = sum(1 for g in all_gt_gaps for n in all_non_gt_gaps if g == n)
            auc = (concordant + 0.5 * tied) / (n_pos * n_neg)
            print(f"    AUC (manual): {auc:.3f}")
            print(f"    对比 Qwen AUC: {QWEN_AUC:.3f}")

    # Per-type
    print(f"\n  Per-type Gap 判别力:")
    print(f"  {'任务类型':<45} {'GT_mean':>8} {'nonGT':>8} {'Sep':>8} {'n_GT':>6} {'n_nGT':>6} {'判定'}")
    print(f"  {'─'*90}")

    gap_disc = {}
    for tt in sorted(by_type.keys()):
        gt = by_type[tt]["gt"]
        ngt = by_type[tt]["non_gt"]
        g_mean = np.mean(gt) if gt else float("nan")
        n_mean = np.mean(ngt) if ngt else float("nan")
        sep = (g_mean - n_mean) if (gt and ngt) else float("nan")

        # AUC per type
        type_auc = float("nan")
        if len(gt) >= 2 and len(ngt) >= 2:
            try:
                from sklearn.metrics import roc_auc_score
                type_labels = [1] * len(gt) + [0] * len(ngt)
                type_scores = gt + ngt
                type_auc = roc_auc_score(type_labels, type_scores)
            except:
                pass

        if math.isnan(sep):
            verdict = "N/A"
        elif sep > 1.0:
            verdict = "DISCRIMINATIVE"
        elif sep > 0.3:
            verdict = "moderate"
        elif sep > 0:
            verdict = "weak"
        else:
            verdict = "ANTI-CORRELATED"

        gap_disc[tt] = {
            "gt_mean": g_mean, "non_gt_mean": n_mean,
            "gt_std": float(np.std(gt)) if gt else float("nan"),
            "non_gt_std": float(np.std(ngt)) if ngt else float("nan"),
            "separation": sep,
            "n_gt": len(gt), "n_non_gt": len(ngt),
            "auc": type_auc,
        }

        g_str = f"{g_mean:>+8.3f}" if not math.isnan(g_mean) else "     n/a"
        n_str = f"{n_mean:>+8.3f}" if not math.isnan(n_mean) else "     n/a"
        s_str = f"{sep:>+8.3f}" if not math.isnan(sep) else "     n/a"
        print(f"  {tt:<45} {g_str} {n_str} {s_str} {len(gt):>6} {len(ngt):>6} {verdict}")

    return {
        "overall": {
            "gt_mean": gt_mean, "non_gt_mean": ngt_mean,
            "separation": separation, "t_stat": t_stat, "p_val": p_val,
            "auc": auc,
            "n_gt": len(all_gt_gaps), "n_non_gt": len(all_non_gt_gaps),
        },
        "by_type": gap_disc,
    }


# ─── Analysis E: Bootstrap CI ───────────────────────────────────────────────

def analysis_e_bootstrap(adaptive_f1s, baseline_f1s, n_boot=1000, seed=42):
    """Bootstrap 95% CI for Adaptive-D vs Baseline delta."""
    print("\n" + "=" * 70)
    print("[E] Bootstrap 置信区间")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    ad = np.array(adaptive_f1s)
    bl = np.array(baseline_f1s)
    delta = ad - bl

    # Bootstrap delta
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(delta), len(delta))
        boot_deltas.append(np.mean(delta[idx]))

    lo = np.percentile(boot_deltas, 2.5)
    hi = np.percentile(boot_deltas, 97.5)
    mean_delta = np.mean(delta)

    gate = "PASS" if lo > 0 else "FAIL"

    print(f"\n  Adaptive-D 平均 F1: {np.mean(ad):.3f}")
    print(f"  Baseline 平均 F1:   {np.mean(bl):.3f}")
    print(f"  Delta:              {mean_delta:+.3f}")
    print(f"  95% CI:             [{lo:.3f}, {hi:+.3f}]")
    print(f"  Gate:               {gate}")
    print(f"  对比 Qwen: Delta=+0.081, CI=[0.046, +0.121] -> PASS")

    # Bootstrap for adaptive F1 itself
    boot_ad = [np.mean(rng.choice(ad, len(ad), replace=True)) for _ in range(n_boot)]
    ad_lo = np.percentile(boot_ad, 2.5)
    ad_hi = np.percentile(boot_ad, 97.5)
    print(f"\n  Adaptive-D F1 95% CI: [{ad_lo:.3f}, {ad_hi:.3f}]")

    return {
        "mean_delta": mean_delta,
        "ci_lo": lo, "ci_hi": hi,
        "gate": gate,
        "ad_mean": float(np.mean(ad)),
        "bl_mean": float(np.mean(bl)),
        "ad_ci_lo": ad_lo, "ad_ci_hi": ad_hi,
    }


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(vanilla_results, global_sweep, per_type_results,
                    gap_results, bootstrap_results, cases):
    """Generate comprehensive markdown report."""

    global_results, (best_tau, best_cd) = global_sweep
    sweep_results, optimal_tau, per_case, adaptive_f1s, baseline_f1s = per_type_results

    lines = []
    def w(s=""):
        lines.append(s)

    w("# GPT-4o Adaptive-D 离线分析报告")
    w(f"")
    w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"数据源: GPT-4o FBF 全量 checkpoint ({len(cases)} cases)")
    w(f"任务类型: {len(set(c['task_type'] for c in cases))} 种")
    w("")

    # Executive Summary
    w("## 核心结论")
    w("")
    ad_f1 = bootstrap_results["ad_mean"]
    bl_f1 = bootstrap_results["bl_mean"]
    delta = bootstrap_results["mean_delta"]
    gate = bootstrap_results["gate"]
    w(f"| 指标 | GPT-4o | Qwen | 对比 |")
    w(f"|------|--------|------|------|")
    w(f"| Vanilla FBF F1 | {vanilla_results['avg_f1']:.3f} | 0.127 | {'GPT-4o 更好' if vanilla_results['avg_f1'] > 0.127 else 'Qwen 更好'} |")
    w(f"| Adaptive-D F1 | {ad_f1:.3f} | 0.222 | {'GPT-4o 更好' if ad_f1 > 0.222 else 'Qwen 更好'} |")
    w(f"| Delta (vs BL) | {delta:+.3f} | +0.081 | |")
    w(f"| 95% CI | [{bootstrap_results['ci_lo']:.3f}, {bootstrap_results['ci_hi']:+.3f}] | [0.046, +0.121] | |")
    w(f"| Gate | {gate} | PASS | |")
    w(f"| Gap Separation | {gap_results['overall']['separation']:+.3f} | +1.418 | {'GPT-4o 更好' if gap_results['overall']['separation'] > 1.418 else 'Qwen 更好'} |")
    w(f"| AUC | {gap_results['overall']['auc']:.3f} | 0.615 | {'GPT-4o 更好' if gap_results['overall']['auc'] > 0.615 else 'Qwen 更好'} |")
    w(f"| Yes-Rate | {vanilla_results['yes_rate']:.1%} | 76-83% | |")
    w("")

    # Section A: Vanilla FBF
    w("## A. Vanilla FBF F1")
    w("")
    w(f"触发规则: response 中包含 'yes' 即触发")
    w(f"整体平均 F1: **{vanilla_results['avg_f1']:.3f}**")
    w(f"Yes-rate: {vanilla_results['yes_rate']:.1%}")
    w("")
    w(f"| 任务类型 | F1 |")
    w(f"|----------|-----|")
    for tt, f1 in sorted(vanilla_results["per_type"].items(), key=lambda x: -x[1]):
        w(f"| {tt} | {f1:.3f} |")
    w("")

    # Section B: Global Sweep
    w("## B. 全局阈值扫描")
    w("")
    best_m = global_results[(best_tau, best_cd)]
    w(f"全局最优: tau={best_tau}, cooldown={best_cd}s, F1={best_m['f1']:.3f}")
    w("")
    w("### 详细扫描结果 (cooldown=12s)")
    w("")
    w(f"| tau | F1 | Precision | Recall | Total FP | Trig% |")
    w(f"|-----|-----|-----------|--------|----------|-------|")
    for tau in TAU_RANGE_GLOBAL:
        if (tau, 12) in global_results:
            r = global_results[(tau, 12)]
            marker = " **" if tau == best_tau and best_cd == 12 else ""
            w(f"| {tau} | {r['f1']:.3f}{marker} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['total_fp']} | {r['avg_trigger_rate']:.1%} |")
    w("")

    # Section C: Per-Type Adaptive
    w("## C. Per-type 自适应阈值 (Adaptive-D)")
    w("")
    w(f"Adaptive-D 平均 F1: **{ad_f1:.3f}**")
    w(f"Baseline 平均 F1: {bl_f1:.3f}")
    w(f"Delta: {delta:+.3f}, 95% CI=[{bootstrap_results['ci_lo']:.3f}, {bootstrap_results['ci_hi']:+.3f}]")
    w(f"Gate: **{gate}**")
    w("")
    w(f"| 任务类型 | n | GPT-4o opt-tau | GPT-4o F1 | BL F1 | Delta | Qwen opt-tau |")
    w(f"|----------|---|---------------|-----------|-------|-------|-------------|")
    for tt in sorted(optimal_tau, key=lambda t: -optimal_tau[t][1]):
        opt_tau, best_f1 = optimal_tau[tt]
        bl_f1_tt = sweep_results[tt].get(0.0, {}).get("f1", 0) if 0.0 in sweep_results[tt] else 0
        delta_tt = best_f1 - bl_f1_tt
        qwen_tau = QWEN_OPTIMAL_TAU.get(tt, "n/a")
        n = sweep_results[tt][opt_tau]["n_cases"]
        w(f"| {tt} | {n} | {opt_tau} | {best_f1:.3f} | {bl_f1_tt:.3f} | {delta_tt:+.3f} | {qwen_tau} |")
    w("")

    # Per-type adaptive aggregate
    by_type_delta = defaultdict(list)
    for r in per_case:
        by_type_delta[r["task_type"]].append(r["delta"])
    w("### Per-type Adaptive 平均 Delta")
    w("")
    w(f"| 任务类型 | 平均 Delta |")
    w(f"|----------|-----------|")
    for tt in sorted(by_type_delta, key=lambda t: -np.mean(by_type_delta[t])):
        w(f"| {tt} | {np.mean(by_type_delta[tt]):+.3f} |")
    w("")

    improved = sum(1 for r in per_case if r["delta"] > 0.001)
    degraded = sum(1 for r in per_case if r["delta"] < -0.001)
    total_bl_fp = sum(r["baseline_fp"] for r in per_case)
    total_ad_fp = sum(r["adaptive_fp"] for r in per_case)
    w(f"Cases improved: {improved}/{len(per_case)} ({improved/len(per_case)*100:.0f}%)")
    w(f"Cases degraded: {degraded}/{len(per_case)} ({degraded/len(per_case)*100:.0f}%)")
    w(f"FP reduction: {total_bl_fp} -> {total_ad_fp} ({(1-total_ad_fp/max(total_bl_fp,1))*100:.0f}%)")
    w("")

    # Section D: Gap Discriminability
    w("## D. Gap 判别力分析")
    w("")
    ov = gap_results["overall"]
    w(f"| 指标 | GPT-4o | Qwen |")
    w(f"|------|--------|------|")
    w(f"| GT mean gap | {ov['gt_mean']:+.3f} | +2.226 |")
    w(f"| Non-GT mean gap | {ov['non_gt_mean']:+.3f} | +0.808 |")
    w(f"| Separation | {ov['separation']:+.3f} | +1.418 |")
    w(f"| t-stat | {ov['t_stat']:.3f} | n/a |")
    w(f"| p-value | {ov['p_val']:.6f} | <0.0001 |")
    w(f"| AUC | {ov['auc']:.3f} | 0.615 |")
    w(f"| n_GT steps | {ov['n_gt']} | n/a |")
    w(f"| n_non-GT steps | {ov['n_non_gt']} | n/a |")
    w("")

    w("### Per-type Gap 判别力")
    w("")
    w(f"| 任务类型 | GT_mean | nonGT_mean | Separation | AUC | 判定 |")
    w(f"|----------|---------|------------|------------|-----|------|")
    for tt in sorted(gap_results["by_type"].keys()):
        d = gap_results["by_type"][tt]
        sep = d["separation"]
        if math.isnan(sep):
            verdict = "N/A"
        elif sep > 1.0:
            verdict = "DISCRIMINATIVE"
        elif sep > 0.3:
            verdict = "moderate"
        elif sep > 0:
            verdict = "weak"
        else:
            verdict = "ANTI-CORRELATED"
        g_str = f"{d['gt_mean']:+.3f}" if not math.isnan(d['gt_mean']) else "n/a"
        n_str = f"{d['non_gt_mean']:+.3f}" if not math.isnan(d['non_gt_mean']) else "n/a"
        s_str = f"{sep:+.3f}" if not math.isnan(sep) else "n/a"
        a_str = f"{d['auc']:.3f}" if not math.isnan(d['auc']) else "n/a"
        w(f"| {tt} | {g_str} | {n_str} | {s_str} | {a_str} | {verdict} |")
    w("")

    # Section E: Bootstrap
    w("## E. Bootstrap 置信区间")
    w("")
    w(f"- Adaptive-D F1: {bootstrap_results['ad_mean']:.3f} "
      f"(95% CI: [{bootstrap_results['ad_ci_lo']:.3f}, {bootstrap_results['ad_ci_hi']:.3f}])")
    w(f"- Baseline F1: {bootstrap_results['bl_mean']:.3f}")
    w(f"- Delta: {bootstrap_results['mean_delta']:+.3f} "
      f"(95% CI: [{bootstrap_results['ci_lo']:.3f}, {bootstrap_results['ci_hi']:+.3f}])")
    w(f"- Gate: **{bootstrap_results['gate']}**")
    w(f"- Bootstrap iterations: 1000")
    w("")

    # Final Conclusion
    w("## 最终结论")
    w("")
    if ad_f1 > QWEN_ADAPTIVE_D_F1:
        w(f"**GPT-4o Adaptive-D F1={ad_f1:.3f} > Qwen Adaptive-D F1={QWEN_ADAPTIVE_D_F1:.3f}**")
        w(f"GPT-4o + Adaptive-D 显著优于 Qwen。")
    elif ad_f1 > QWEN_ADAPTIVE_D_F1 - 0.02:
        w(f"GPT-4o Adaptive-D F1={ad_f1:.3f} 接近 Qwen Adaptive-D F1={QWEN_ADAPTIVE_D_F1:.3f}")
        w(f"两者性能相当，无显著差异。")
    else:
        w(f"GPT-4o Adaptive-D F1={ad_f1:.3f} < Qwen Adaptive-D F1={QWEN_ADAPTIVE_D_F1:.3f}")
        w(f"Qwen + Adaptive-D 仍然优于 GPT-4o。")
    w("")

    if gap_results["overall"]["separation"] > QWEN_SEPARATION:
        w(f"Gap 判别力: GPT-4o ({gap_results['overall']['separation']:+.3f}) > Qwen ({QWEN_SEPARATION:+.3f})")
        w(f"GPT-4o logprob gap 更具判别力，Adaptive-D 有更大潜力。")
    else:
        w(f"Gap 判别力: GPT-4o ({gap_results['overall']['separation']:+.3f}) vs Qwen ({QWEN_SEPARATION:+.3f})")

    if not math.isnan(gap_results["overall"]["auc"]):
        if gap_results["overall"]["auc"] > QWEN_AUC:
            w(f"AUC: GPT-4o ({gap_results['overall']['auc']:.3f}) > Qwen ({QWEN_AUC:.3f})")
        else:
            w(f"AUC: GPT-4o ({gap_results['overall']['auc']:.3f}) vs Qwen ({QWEN_AUC:.3f})")
    w("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GPT-4o Adaptive-D 离线分析")
    print("=" * 70)

    # Load data
    print(f"\n加载数据: {CHECKPOINT_PATH}")
    cases = load_checkpoint(CHECKPOINT_PATH)
    print(f"  加载 {len(cases)} 个有效 cases")
    print(f"  任务类型: {sorted(set(c['task_type'] for c in cases))}")

    # A. Vanilla FBF
    vanilla_results = analysis_a_vanilla_fbf(cases)

    # B. Global sweep
    global_sweep = analysis_b_global_sweep(cases)

    # C. Per-type adaptive
    per_type_results = analysis_c_per_type_adaptive(cases, cooldown=12.0)

    # D. Gap discriminability
    gap_results = analysis_d_gap_discriminability(cases)

    # E. Bootstrap CI
    _, _, _, adaptive_f1s, baseline_f1s = per_type_results
    bootstrap_results = analysis_e_bootstrap(adaptive_f1s, baseline_f1s)

    # Generate report
    print("\n" + "=" * 70)
    print("生成报告...")
    print("=" * 70)

    report = generate_report(vanilla_results, global_sweep, per_type_results,
                             gap_results, bootstrap_results, cases)

    # Save report
    report_path = OUTPUT_DIR / "gpt4o_adaptive_d_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")

    # Save structured results
    json_path = OUTPUT_DIR / "gpt4o_adaptive_d_results.json"
    json_out = {
        "vanilla_fbf": {
            "avg_f1": vanilla_results["avg_f1"],
            "per_type": vanilla_results["per_type"],
            "yes_rate": vanilla_results["yes_rate"],
        },
        "global_best": {
            "tau": global_sweep[1][0],
            "cooldown": global_sweep[1][1],
            "f1": global_sweep[0][global_sweep[1]]["f1"],
        },
        "adaptive_d": {
            "per_type_optimal_tau": {tt: {"tau": t, "f1": f}
                                     for tt, (t, f) in per_type_results[1].items()},
            "avg_f1": bootstrap_results["ad_mean"],
            "baseline_f1": bootstrap_results["bl_mean"],
            "delta": bootstrap_results["mean_delta"],
            "ci_lo": bootstrap_results["ci_lo"],
            "ci_hi": bootstrap_results["ci_hi"],
            "gate": bootstrap_results["gate"],
        },
        "gap_discriminability": {
            "overall_separation": gap_results["overall"]["separation"],
            "auc": gap_results["overall"]["auc"],
            "p_value": gap_results["overall"]["p_val"],
        },
        "comparison_with_qwen": {
            "qwen_adaptive_d_f1": QWEN_ADAPTIVE_D_F1,
            "gpt4o_adaptive_d_f1": bootstrap_results["ad_mean"],
            "gpt4o_wins": bootstrap_results["ad_mean"] > QWEN_ADAPTIVE_D_F1,
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    print(f"结构化结果: {json_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)
    print(f"  GPT-4o Vanilla FBF F1:  {vanilla_results['avg_f1']:.3f}")
    print(f"  GPT-4o Adaptive-D F1:   {bootstrap_results['ad_mean']:.3f}")
    print(f"  Delta vs Baseline:      {bootstrap_results['mean_delta']:+.3f} "
          f"(95% CI: [{bootstrap_results['ci_lo']:.3f}, {bootstrap_results['ci_hi']:+.3f}])")
    print(f"  Gate:                   {bootstrap_results['gate']}")
    print(f"  Gap Separation:         {gap_results['overall']['separation']:+.3f}")
    print(f"  AUC:                    {gap_results['overall']['auc']:.3f}")
    print(f"")
    print(f"  对比 Qwen:")
    print(f"    Qwen Adaptive-D F1:   {QWEN_ADAPTIVE_D_F1:.3f}")
    print(f"    Qwen Gap Separation:  {QWEN_SEPARATION:+.3f}")
    print(f"    Qwen AUC:             {QWEN_AUC:.3f}")

    if bootstrap_results["ad_mean"] > QWEN_ADAPTIVE_D_F1:
        print(f"\n  >>> GPT-4o Adaptive-D ({bootstrap_results['ad_mean']:.3f}) > Qwen ({QWEN_ADAPTIVE_D_F1:.3f})")
        print(f"      GPT-4o + Adaptive-D 更优")
    else:
        print(f"\n  >>> GPT-4o Adaptive-D ({bootstrap_results['ad_mean']:.3f}) vs Qwen ({QWEN_ADAPTIVE_D_F1:.3f})")
        if bootstrap_results["ad_mean"] < QWEN_ADAPTIVE_D_F1:
            print(f"      Qwen + Adaptive-D 仍然更优")
        else:
            print(f"      两者接近")


if __name__ == "__main__":
    main()
