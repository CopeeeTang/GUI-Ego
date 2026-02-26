#!/usr/bin/env python3
"""
Phase I: Full-Scale Hypothesis D Validation on 5-cases (60 QA).

Strategy: Single GPU pass at tau=2.80 (lower threshold), collecting full
logprob data at every polling step. Then simulate tau=3.0 and baseline
offline from the saved logprob data.

Key optimizations vs hypothesis_runner.py:
  - run_logprobs_only(): single forward pass (no generate) for non-trigger steps
  - Checkpoint/resume via JSONL for long runs
  - Bootstrap CI for statistical rigor

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate

    # Validate first (3 cases):
    python3 proactive-project/experiments/estp_phase3/fullscale_d_runner.py \\
        --tau 2.80 --cooldown 12.0 --limit 3 --verbose

    # Full run (60 cases, ~2h):
    python3 proactive-project/experiments/estp_phase3/fullscale_d_runner.py \\
        --tau 2.80 --cooldown 12.0 --verbose

    # Resume interrupted run:
    python3 proactive-project/experiments/estp_phase3/fullscale_d_runner.py \\
        --tau 2.80 --cooldown 12.0 --resume

    # Generate report from existing checkpoint:
    python3 proactive-project/experiments/estp_phase3/fullscale_d_runner.py \\
        --report_only
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ─── Path setup ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
ESTP_DIR = PROJECT_ROOT / "data" / "ESTP-Bench" / "estp_dataset"
VIDEO_ROOT = PROJECT_ROOT / "data" / "ESTP-Bench" / "full_scale_2fps_max384"
DATA_5CASES = ESTP_DIR / "estp_bench_sq_5_cases.json"
OUTPUT_DIR = SCRIPT_DIR / "results" / "fullscale_d"

ANTICIPATION = 1
LATENCY = 2

# ─── Import core functions from hypothesis_runner ───
# Sharing the model singleton (_model, _processor) avoids double-loading 16GB.
sys.path.insert(0, str(SCRIPT_DIR))

from hypothesis_runner import (
    load_model,
    run_inference,
    _extract_frames_upto,
    find_video_file,
    compute_case_metrics,
    ceil_time_by_fps,
    PROMPT_TEMPLATE,
)
from prompts import get_trigger_prompt, get_answer_prompt


# ═══════════════════════════════════════════════════════════
#  Data Handling
# ═══════════════════════════════════════════════════════════

def flatten_dataset_to_cases(data):
    """Convert nested 5-cases JSON to a flat list of case dicts.

    Handles two data formats present in estp_bench_sq_5_cases.json:
      1. Standard (50 QAs):  clip_start_time, clip_end_time, question, qa_id
      2. Goalstep (10 QAs):  start_time, end_time, conversation[0] = user question

    Both formats preserve 'qa_raw' for use with compute_case_metrics().
    'Task Type' is stripped of leading/trailing whitespace.
    """
    cases = []
    for vid_uid, clips in data.items():
        for clip_uid, qas in clips.items():
            if not isinstance(qas, list):
                continue
            for idx, qa in enumerate(qas):
                task_type = qa.get("Task Type", "").strip()

                if qa.get("clip_start_time") is not None:
                    # Standard format
                    cases.append({
                        "video_uid": vid_uid,
                        "clip_uid": clip_uid,
                        "qa_id": qa.get("qa_id", idx),
                        "task_type": task_type,
                        "question": qa.get("question", ""),
                        "start_time": float(qa["clip_start_time"]),
                        "end_time": float(qa["clip_end_time"]),
                        "qa_raw": qa,
                    })
                elif qa.get("start_time") is not None:
                    # Goalstep format: question is the first user turn content
                    conv = qa.get("conversation", [])
                    user_msgs = [c for c in conv if c.get("role", "").lower() == "user"]
                    question = user_msgs[0]["content"] if user_msgs else ""
                    cases.append({
                        "video_uid": qa.get("video_uid", vid_uid),
                        "clip_uid": clip_uid,
                        "qa_id": idx,
                        "task_type": task_type,
                        "question": question,
                        "start_time": float(qa["start_time"]),
                        "end_time": float(qa["end_time"]),
                        "qa_raw": qa,
                    })
    return cases


# ═══════════════════════════════════════════════════════════
#  Optimized Inference
# ═══════════════════════════════════════════════════════════

def run_logprobs_only(frames, prompt):
    """Get first-token yes/no logprobs with a SINGLE forward pass.

    Unlike run_inference_with_logprobs(), this does NOT call model.generate().
    Saves ~1.5s per non-trigger polling step (~60 min over 60 QA runs).

    Returns: (yes_logp, no_logp)
    """
    model, processor = load_model()

    if not frames:
        return float("-inf"), float("-inf")

    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input], images=frames, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    yes_tokens = processor.tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = processor.tokenizer.encode("no", add_special_tokens=False)
    Yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
    No_tokens = processor.tokenizer.encode("No", add_special_tokens=False)

    yes_logp = max(
        log_probs[yes_tokens[0]].item() if yes_tokens else float("-inf"),
        log_probs[Yes_tokens[0]].item() if Yes_tokens else float("-inf"),
    )
    no_logp = max(
        log_probs[no_tokens[0]].item() if no_tokens else float("-inf"),
        log_probs[No_tokens[0]].item() if No_tokens else float("-inf"),
    )

    return yes_logp, no_logp


def run_d_with_full_logprobs(video_path, question, start_time, end_time,
                              tau=2.80, cooldown_sec=12.0, frame_fps=0.175,
                              max_frames=64, verbose=False):
    """Run Hypothesis D collecting full logprob data at EVERY polling step.

    Core loop per polling step:
      1. Extract frames up to current_time
      2. run_logprobs_only() → single forward pass, no generate
      3. Three-segment rule + cooldown decision
      4. If triggered: run_inference() for answer generation only

    Returns:
        dialog_history: [{role, content, time}] for ESTP-F1 scoring
        trigger_log: [{time, yes_logp, no_logp, logprob_gap, triggered, ...}]
    """
    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)
    last_trigger_time = -float("inf")

    while current_time <= end_time:
        frames = _extract_frames_upto(video_path, start_time, current_time, max_frames)
        if not frames:
            current_time += 1 / frame_fps
            continue

        t0 = time.time()
        yes_logp, no_logp = run_logprobs_only(frames, trigger_prompt)
        cost_logprob = time.time() - t0

        logprob_gap = yes_logp - no_logp

        # Three-segment rule: gap > tau → trigger candidate
        gap_decision = "trigger" if logprob_gap > tau else "reject"
        in_cooldown = (current_time - last_trigger_time) < cooldown_sec
        triggered = (gap_decision == "trigger") and not in_cooldown

        step_record = {
            "time": current_time,
            "yes_logp": yes_logp,
            "no_logp": no_logp,
            "logprob_gap": logprob_gap,
            "gap_decision": gap_decision,
            "in_cooldown": in_cooldown,
            "triggered": triggered,
            "n_frames": len(frames),
            "cost_logprob": cost_logprob,
        }

        if verbose:
            if triggered:
                status = "TRIGGERED"
            elif in_cooldown:
                status = f"cooldown({current_time - last_trigger_time:.1f}s)"
            else:
                status = f"reject(gap={logprob_gap:.2f})"
            print(f"    [D] t={current_time:.1f}s gap={logprob_gap:.2f} {status}")

        if triggered:
            last_trigger_time = current_time
            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost_answer = time.time() - t0
            step_record["cost_answer"] = cost_answer

            dialog_history.append({"role": "user", "content": "trigger",
                                    "time": current_time})
            dialog_history.append({"role": "assistant", "content": answer,
                                    "time": current_time,
                                    "cost": cost_logprob + cost_answer})

        trigger_log.append(step_record)
        current_time += 1 / frame_fps

    return dialog_history, trigger_log


# ═══════════════════════════════════════════════════════════
#  Offline Simulation (no GPU needed)
# ═══════════════════════════════════════════════════════════

def simulate_config_offline(trigger_log, qa_raw, tau, cooldown_sec):
    """Re-simulate (tau, cooldown) from saved logprob data.

    Logprob values are independent of threshold — the forward pass is
    identical regardless of what tau we choose. So we can replay any
    configuration offline without GPU for timing-only ESTP-F1.

    Returns: (trigger_times, metrics_dict)
    """
    trigger_times = []
    last_trigger = -float("inf")

    for entry in trigger_log:
        t = entry["time"]
        gap = entry["logprob_gap"]
        if gap > tau and (t - last_trigger) >= cooldown_sec:
            trigger_times.append(t)
            last_trigger = t

    dialog = [{"role": "assistant", "content": "sim", "time": t}
               for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    return trigger_times, metrics


def reconstruct_baseline(trigger_log, qa_raw):
    """Reconstruct vanilla FBF baseline metrics from logprob data.

    Vanilla FBF triggers when greedy decode outputs 'yes'.
    Greedy decode outputs 'yes' iff logP(yes) >= logP(no) iff gap >= 0.
    No cooldown applied.

    Returns: (trigger_times, metrics_dict)
    """
    trigger_times = [e["time"] for e in trigger_log if e["logprob_gap"] > 0]
    dialog = [{"role": "assistant", "content": "sim", "time": t}
               for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    return trigger_times, metrics


# ═══════════════════════════════════════════════════════════
#  Statistical Analysis
# ═══════════════════════════════════════════════════════════

def bootstrap_ci(values, n_resamples=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns: (lower, upper, mean)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
    n = len(values)
    means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), \
           float(np.percentile(means, (1 - alpha) * 100)), \
           float(np.mean(values))


def bootstrap_delta_ci(values_a, values_b, n_resamples=1000, ci=0.95, seed=42):
    """Bootstrap CI for the paired difference (a - b).

    Uses paired resampling: each resample draws matched (a_i, b_i) pairs,
    computing mean(a_i - b_i). More powerful than independent CIs when
    data is correlated (same video = same confounders).

    Gate condition: lower > 0 → improvement statistically significant.
    Returns: (lower, upper, mean_delta)
    """
    rng = np.random.RandomState(seed)
    a = np.array(values_a, dtype=float)
    b = np.array(values_b, dtype=float)
    deltas = a - b
    n = len(deltas)
    means = np.array([
        np.mean(rng.choice(deltas, size=n, replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), \
           float(np.percentile(means, (1 - alpha) * 100)), \
           float(np.mean(deltas))


# ═══════════════════════════════════════════════════════════
#  Checkpoint I/O (JSONL for atomic appends)
# ═══════════════════════════════════════════════════════════

def get_checkpoint_path():
    return OUTPUT_DIR / "checkpoint.jsonl"


def load_checkpoint():
    """Load completed cases from JSONL checkpoint.

    Returns: (completed_keys: set, results: list)
    """
    cp = get_checkpoint_path()
    if not cp.exists():
        return set(), []

    completed = set()
    results = []
    with open(cp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = f"{r['video_uid']}_{r['clip_uid']}_{r['qa_id']}"
                completed.add(key)
                results.append(r)
            except json.JSONDecodeError:
                pass
    return completed, results


def save_checkpoint(result):
    """Atomically append one case result to JSONL checkpoint."""
    cp = get_checkpoint_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def case_key(case):
    return f"{case['video_uid']}_{case['clip_uid']}_{case['qa_id']}"


# ═══════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════

def generate_fullscale_report(results, tau_primary, tau_compare, cooldown_sec,
                               n_bootstrap=1000):
    """Comprehensive report with bootstrap CI and task-type stratification."""
    valid = [r for r in results if r.get("status") == "ok"]
    n = len(valid)

    lines = []
    lines.append("=" * 80)
    lines.append("Phase I: Full-Scale Hypothesis D Validation (5-cases subset)")
    lines.append("=" * 80)
    lines.append(f"\nConfig: tau_primary={tau_primary} (GPU online), "
                 f"tau_compare={tau_compare} (offline), "
                 f"cooldown={cooldown_sec}s, fps=0.175")
    lines.append(f"Cases: {n} valid (of {len(results)} total)")
    lines.append(f"Bootstrap: n_resamples={n_bootstrap}, seed=42\n")

    if n == 0:
        lines.append("No valid results. Check checkpoint file.")
        return "\n".join(lines)

    # ── Collect per-case metrics ──
    bl_f1s  = [r["baseline"]["metrics"]["f1"] for r in valid]
    pr_f1s  = [r["primary"]["metrics"]["f1"] for r in valid]
    cm_f1s  = [r["compare"]["metrics"]["f1"] for r in valid]

    bl_fps  = [r["baseline"]["metrics"]["false_positives"] for r in valid]
    pr_fps  = [r["primary"]["metrics"]["false_positives"] for r in valid]
    cm_fps  = [r["compare"]["metrics"]["false_positives"] for r in valid]

    bl_rec  = [r["baseline"]["metrics"]["recall"] for r in valid]
    pr_rec  = [r["primary"]["metrics"]["recall"] for r in valid]
    cm_rec  = [r["compare"]["metrics"]["recall"] for r in valid]

    bl_prec = [r["baseline"]["metrics"]["precision"] for r in valid]
    pr_prec = [r["primary"]["metrics"]["precision"] for r in valid]

    total_polls = sum(r["n_polls"] for r in valid)
    bl_trig = sum(r["baseline"]["n_triggers"] for r in valid)
    pr_trig = sum(r["primary"]["n_triggers"] for r in valid)
    cm_trig = sum(r["compare"]["n_triggers"] for r in valid)

    # ── Overall table ──
    lines.append("─" * 80)
    lines.append("OVERALL METRICS")
    lines.append("─" * 80)
    col_bl  = "Baseline"
    col_pr  = f"D@{tau_primary} (GPU)"
    col_cm  = f"D@{tau_compare} (offline)"
    lines.append(f"  {'Metric':<20} {col_bl:>14} {col_pr:>18} {col_cm:>20}")
    lines.append("  " + "-" * 72)

    def row(name, bl, pr, cm, fmt=".3f"):
        bl_s = f"{bl:{fmt}}" if not math.isnan(bl) else "  n/a"
        pr_s = f"{pr:{fmt}}" if not math.isnan(pr) else "  n/a"
        cm_s = f"{cm:{fmt}}" if not math.isnan(cm) else "  n/a"
        return f"  {name:<20} {bl_s:>14} {pr_s:>18} {cm_s:>20}"

    lines.append(row("Avg F1",       np.mean(bl_f1s),  np.mean(pr_f1s),  np.mean(cm_f1s)))
    lines.append(row("  std",        np.std(bl_f1s),   np.std(pr_f1s),   np.std(cm_f1s)))
    lines.append(row("  min",        np.min(bl_f1s),   np.min(pr_f1s),   np.min(cm_f1s)))
    lines.append(row("  max",        np.max(bl_f1s),   np.max(pr_f1s),   np.max(cm_f1s)))
    lines.append(row("Avg FP",       np.mean(bl_fps),  np.mean(pr_fps),  np.mean(cm_fps), ".1f"))
    lines.append(row("Avg Recall",   np.mean(bl_rec),  np.mean(pr_rec),  np.mean(cm_rec)))
    lines.append(row("Avg Precision",np.mean(bl_prec), np.mean(pr_prec), float("nan")))
    lines.append(f"  {'Trigger Rate':<20} {bl_trig/total_polls:>14.1%} "
                 f"{pr_trig/total_polls:>18.1%} {cm_trig/total_polls:>20.1%}")

    # ── Bootstrap CIs ──
    lines.append("\n" + "─" * 80)
    lines.append("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    lines.append("─" * 80)

    def ci_line(name, values):
        lo, hi, mean = bootstrap_ci(values, n_resamples=n_bootstrap)
        return f"  {name:<32} mean={mean:.3f}   95% CI=[{lo:.3f}, {hi:.3f}]"

    lines.append(ci_line("Baseline F1",              bl_f1s))
    lines.append(ci_line(f"D@{tau_primary} F1 (GPU)", pr_f1s))
    lines.append(ci_line(f"D@{tau_compare} F1 (offline)", cm_f1s))
    lines.append(ci_line("Baseline FP",              [float(x) for x in bl_fps]))
    lines.append(ci_line(f"D@{tau_primary} FP",      [float(x) for x in pr_fps]))
    lines.append(ci_line("Baseline Recall",           bl_rec))
    lines.append(ci_line(f"D@{tau_primary} Recall",  pr_rec))

    # ── Gate: paired delta CI ──
    lines.append("\n" + "─" * 80)
    lines.append(f"GATE: Paired Delta CI  (D@{tau_primary} vs Baseline)")
    lines.append("─" * 80)

    lo_f1, hi_f1, mean_f1 = bootstrap_delta_ci(pr_f1s, bl_f1s, n_resamples=n_bootstrap)
    lo_fp, hi_fp, mean_fp = bootstrap_delta_ci(
        [float(x) for x in pr_fps], [float(x) for x in bl_fps], n_resamples=n_bootstrap)
    lo_re, hi_re, mean_re = bootstrap_delta_ci(pr_rec, bl_rec, n_resamples=n_bootstrap)

    lines.append(f"  Delta F1:     mean={mean_f1:+.3f}  95% CI=[{lo_f1:+.3f}, {hi_f1:+.3f}]")
    lines.append(f"  Delta FP:     mean={mean_fp:+.1f}   95% CI=[{lo_fp:+.1f}, {hi_fp:+.1f}]")
    lines.append(f"  Delta Recall: mean={mean_re:+.3f}  95% CI=[{lo_re:+.3f}, {hi_re:+.3f}]")

    gate_pass = lo_f1 > 0
    lines.append(f"\n  >>> GATE: {'PASS' if gate_pass else 'FAIL'} <<<")
    if gate_pass:
        lines.append("      CI lower bound > 0. D improvement statistically significant.")
        lines.append("      Proceed to Phase II: mechanism analysis + task-type stratification.")
    else:
        lines.append("      CI crosses 0. Improvement not significant at 95% confidence.")
        lines.append("      Consider: adjust tau, larger sample, or revisit design.")

    # ── Task-type stratification ──
    lines.append("\n" + "─" * 80)
    lines.append("TASK-TYPE STRATIFIED RESULTS")
    lines.append("─" * 80)
    by_type = defaultdict(list)
    for r in valid:
        by_type[r["task_type"]].append(r)

    lines.append(f"  {'Task Type':<40} {'n':>3} {'BL F1':>7} {'D F1':>7} "
                 f"{'ΔF1':>7} {'BL FP':>7} {'D FP':>7}")
    lines.append("  " + "-" * 78)

    type_deltas = {}
    for tt in sorted(by_type.keys()):
        cases_tt = by_type[tt]
        n_tt = len(cases_tt)
        bl_f1_tt = np.mean([r["baseline"]["metrics"]["f1"] for r in cases_tt])
        pr_f1_tt = np.mean([r["primary"]["metrics"]["f1"] for r in cases_tt])
        delta_tt = pr_f1_tt - bl_f1_tt
        bl_fp_tt = np.mean([r["baseline"]["metrics"]["false_positives"] for r in cases_tt])
        pr_fp_tt = np.mean([r["primary"]["metrics"]["false_positives"] for r in cases_tt])
        type_deltas[tt] = delta_tt
        lines.append(
            f"  {tt:<40} {n_tt:>3} {bl_f1_tt:>7.3f} {pr_f1_tt:>7.3f} "
            f"{delta_tt:>+7.3f} {bl_fp_tt:>7.1f} {pr_fp_tt:>7.1f}"
        )

    best_type = max(type_deltas, key=type_deltas.get)
    worst_type = min(type_deltas, key=type_deltas.get)
    lines.append(f"\n  D helps most:  {best_type} (Δ={type_deltas[best_type]:+.3f})")
    lines.append(f"  D helps least: {worst_type} (Δ={type_deltas[worst_type]:+.3f})")

    # ── Logprob gap analysis ──
    lines.append("\n" + "─" * 80)
    lines.append("LOGPROB GAP DISTRIBUTION")
    lines.append("─" * 80)

    all_gaps, gt_gaps, non_gt_gaps = [], [], []
    for r in valid:
        qa_raw = r["qa_raw"]
        clip_end = qa_raw.get("clip_end_time", qa_raw.get("end_time", 0))
        gold = [c for c in qa_raw["conversation"] if c["role"].lower() == "assistant"]
        gt_spans = [
            (ceil_time_by_fps(c["start_time"], 2, 0, clip_end),
             ceil_time_by_fps(c["end_time"], 2, 0, clip_end))
            for c in gold
        ]
        for entry in r["trigger_log"]:
            t, gap = entry["time"], entry["logprob_gap"]
            all_gaps.append(gap)
            if any(gs - ANTICIPATION <= t <= ge + LATENCY for gs, ge in gt_spans):
                gt_gaps.append(gap)
            else:
                non_gt_gaps.append(gap)

    if all_gaps:
        lines.append(f"\n  All gaps (n={len(all_gaps)}): "
                     f"mean={np.mean(all_gaps):.3f}, std={np.std(all_gaps):.3f}, "
                     f"median={np.median(all_gaps):.3f}")
        if gt_gaps:
            lines.append(f"  GT-window  gaps (n={len(gt_gaps):4d}): "
                         f"mean={np.mean(gt_gaps):.3f}, std={np.std(gt_gaps):.3f}")
        if non_gt_gaps:
            lines.append(f"  Non-GT     gaps (n={len(non_gt_gaps):4d}): "
                         f"mean={np.mean(non_gt_gaps):.3f}, std={np.std(non_gt_gaps):.3f}")
        if gt_gaps and non_gt_gaps:
            sep = np.mean(gt_gaps) - np.mean(non_gt_gaps)
            lines.append(f"  Separation (GT − non-GT mean): {sep:+.3f}")
        lines.append("\n  Threshold coverage:")
        for thr in [0.0, 1.0, 2.0, 2.8, 3.0, 4.0, 5.0]:
            pct = sum(1 for g in all_gaps if g > thr) / len(all_gaps)
            lines.append(f"    Gap > {thr:.1f}: {pct:.1%}")

    # ── Case-level summary ──
    lines.append("\n" + "─" * 80)
    lines.append("CASE-LEVEL SUMMARY")
    lines.append("─" * 80)
    improved  = sum(1 for r in valid if r["primary"]["metrics"]["f1"] > r["baseline"]["metrics"]["f1"] + 0.001)
    degraded  = sum(1 for r in valid if r["primary"]["metrics"]["f1"] < r["baseline"]["metrics"]["f1"] - 0.001)
    unchanged = n - improved - degraded
    lines.append(f"  Improved:  {improved}/{n} ({improved/n:.0%})")
    lines.append(f"  Degraded:  {degraded}/{n} ({degraded/n:.0%})")
    lines.append(f"  Unchanged: {unchanged}/{n} ({unchanged/n:.0%})")

    # Top degraded cases for Phase II analysis
    degraded_list = sorted(
        [r for r in valid if r["primary"]["metrics"]["f1"] < r["baseline"]["metrics"]["f1"] - 0.001],
        key=lambda r: r["primary"]["metrics"]["f1"] - r["baseline"]["metrics"]["f1"]
    )
    if degraded_list:
        lines.append(f"\n  Top degraded cases (for Phase II analysis):")
        for r in degraded_list[:5]:
            delta = r["primary"]["metrics"]["f1"] - r["baseline"]["metrics"]["f1"]
            lines.append(f"    [{r['task_type']}] delta={delta:+.3f} | {r['question'][:65]}")

    lines.append("\n" + "=" * 80)
    lines.append(f"Report generated: {n} valid / {len(results)} total cases")
    lines.append("=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase I: Full-Scale Hypothesis D on 5-cases (60 QA)"
    )
    parser.add_argument("--tau", type=float, default=2.80,
                        help="Primary logprob threshold (GPU run)")
    parser.add_argument("--compare_tau", type=float, default=3.0,
                        help="Secondary tau for offline comparison")
    parser.add_argument("--cooldown", type=float, default=12.0,
                        help="Cooldown seconds between triggers")
    parser.add_argument("--frame_fps", type=float, default=0.175,
                        help="Polling rate (Hz)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit cases for testing (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--report_only", action="store_true",
                        help="Generate report from checkpoint without GPU")
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print(f"Loading: {DATA_5CASES}")
    with open(DATA_5CASES) as f:
        data = json.load(f)

    cases = flatten_dataset_to_cases(data)
    task_types = sorted(set(c["task_type"] for c in cases))
    print(f"Found {len(cases)} cases, "
          f"{len(set(c['video_uid'] for c in cases))} videos, "
          f"{len(task_types)} task types")

    # Sort by video for OS page-cache efficiency
    cases.sort(key=lambda c: (c["video_uid"], c["start_time"]))

    if args.limit > 0:
        cases = cases[:args.limit]
        print(f"Limited to {len(cases)} cases")

    # ── Report-only mode ──
    if args.report_only:
        _, results = load_checkpoint()
        if not results:
            print("ERROR: No checkpoint. Run without --report_only first.")
            return
        print(f"Generating report from {len(results)} checkpoint entries...")
        report = generate_fullscale_report(
            results, args.tau, args.compare_tau, args.cooldown, args.bootstrap_n
        )
        rp = OUTPUT_DIR / "fullscale_d_report.txt"
        rp.write_text(report)
        print(report)
        print(f"\nSaved: {rp}")
        return

    # ── Checkpoint / resume ──
    completed_keys, prev_results = (load_checkpoint() if args.resume
                                     else (set(), []))
    if args.resume:
        print(f"Resuming: {len(completed_keys)} already done")

    # ── Pre-load model ──
    load_model()

    # ── Main loop ──
    all_results = list(prev_results)
    t_run_start = time.time()

    for i, case in enumerate(cases):
        ck = case_key(case)
        if ck in completed_keys:
            print(f"[{i+1:02d}/{len(cases)}] SKIP (done): {case['task_type']}")
            continue

        print(f"\n[{i+1:02d}/{len(cases)}] {case['task_type']}: "
              f"{case['question'][:60]}...")
        print(f"  video={case['video_uid'][:12]}... "
              f"t=[{case['start_time']:.0f}s, {case['end_time']:.0f}s]")

        # Find video
        video_path = find_video_file(case["video_uid"], case["clip_uid"])
        if not video_path:
            print(f"  SKIP: video not found")
            r = {k: case[k] for k in ("video_uid", "clip_uid", "qa_id",
                                       "task_type", "question")}
            r["status"] = "video_not_found"
            save_checkpoint(r); all_results.append(r); completed_keys.add(ck)
            continue

        # Check GT exists
        qa_raw = case["qa_raw"]
        gold = [c for c in qa_raw.get("conversation", [])
                if c.get("role", "").lower() == "assistant"]
        if not gold:
            print(f"  SKIP: no GT")
            r = {k: case[k] for k in ("video_uid", "clip_uid", "qa_id",
                                       "task_type", "question")}
            r["status"] = "no_gt"
            save_checkpoint(r); all_results.append(r); completed_keys.add(ck)
            continue

        duration = case["end_time"] - case["start_time"]
        clip_end = qa_raw.get("clip_end_time", qa_raw.get("end_time", 0))
        gt_spans = [
            (ceil_time_by_fps(c["start_time"], 2, 0, clip_end),
             ceil_time_by_fps(c["end_time"], 2, 0, clip_end))
            for c in gold
        ]
        print(f"  dur={duration:.0f}s, GT_windows={len(gt_spans)}, "
              f"spans={[(f'{s:.0f}-{e:.0f}s') for s,e in gt_spans[:3]]}")

        # ── GPU run ──
        t0 = time.time()
        dialog, trigger_log = run_d_with_full_logprobs(
            video_path, case["question"],
            case["start_time"], case["end_time"],
            tau=args.tau, cooldown_sec=args.cooldown,
            frame_fps=args.frame_fps, verbose=args.verbose,
        )
        gpu_time = time.time() - t0
        primary_metrics = compute_case_metrics(dialog, qa_raw)
        n_polls = len(trigger_log)
        n_trig_primary = sum(1 for e in trigger_log if e["triggered"])
        print(f"  D@{args.tau}: F1={primary_metrics['f1']:.3f}, "
              f"preds={primary_metrics['n_preds']}, "
              f"hits={len(primary_metrics['hits'])}/{len(gt_spans)}, "
              f"FP={primary_metrics['false_positives']}, "
              f"trig={n_trig_primary}/{n_polls} ({gpu_time:.1f}s)")

        # ── Offline: compare_tau ──
        compare_times, compare_metrics = simulate_config_offline(
            trigger_log, qa_raw, tau=args.compare_tau, cooldown_sec=args.cooldown
        )
        print(f"  D@{args.compare_tau}: F1={compare_metrics['f1']:.3f}, "
              f"FP={compare_metrics['false_positives']}")

        # ── Offline: baseline ──
        baseline_times, baseline_metrics = reconstruct_baseline(trigger_log, qa_raw)
        delta_f1 = primary_metrics["f1"] - baseline_metrics["f1"]
        verdict = ("IMPROVED" if delta_f1 > 0.001 else
                   "DEGRADED" if delta_f1 < -0.001 else "UNCHANGED")
        print(f"  Baseline: F1={baseline_metrics['f1']:.3f}, "
              f"FP={baseline_metrics['false_positives']}, "
              f"trig={len(baseline_times)}/{n_polls}")
        print(f"  VERDICT: {verdict} (delta={delta_f1:+.3f})")

        # ── Save ──
        result = {
            "video_uid": case["video_uid"],
            "clip_uid": case["clip_uid"],
            "qa_id": case["qa_id"],
            "task_type": case["task_type"],
            "question": case["question"],
            "status": "ok",
            "duration": duration,
            "n_gt": len(gt_spans),
            "gt_spans": [(float(s), float(e)) for s, e in gt_spans],
            "n_polls": n_polls,
            "qa_raw": qa_raw,
            "primary": {
                "tau": args.tau, "cooldown": args.cooldown,
                "metrics": primary_metrics,
                "n_triggers": n_trig_primary,
                "gpu_time": gpu_time,
            },
            "compare": {
                "tau": args.compare_tau, "cooldown": args.cooldown,
                "trigger_times": compare_times,
                "metrics": compare_metrics,
                "n_triggers": len(compare_times),
            },
            "baseline": {
                "trigger_times": baseline_times,
                "metrics": baseline_metrics,
                "n_triggers": len(baseline_times),
            },
            "trigger_log": trigger_log,
        }
        save_checkpoint(result)
        all_results.append(result)
        completed_keys.add(ck)

        # ETA
        done_ok = sum(1 for r in all_results if r.get("status") == "ok")
        remaining = sum(1 for c in cases if case_key(c) not in completed_keys)
        elapsed = time.time() - t_run_start
        eta = (elapsed / max(done_ok, 1)) * remaining
        print(f"  [{done_ok} done, {remaining} left, ETA ~{eta/60:.0f}min]")

    # ── Final report ──
    print(f"\n{'='*60}")
    print("COMPLETE — Generating report")
    print(f"{'='*60}")

    valid = [r for r in all_results if r.get("status") == "ok"]
    report = generate_fullscale_report(
        valid, args.tau, args.compare_tau, args.cooldown, args.bootstrap_n
    )

    rp = OUTPUT_DIR / "fullscale_d_report.txt"
    rp.write_text(report)

    raw_p = OUTPUT_DIR / "fullscale_d_raw.json"
    with open(raw_p, "w") as f:
        json.dump(valid, f, indent=2, default=str)

    print(report)
    print(f"\nReport:    {rp}")
    print(f"Raw data:  {raw_p}")
    print(f"Checkpoint:{get_checkpoint_path()}")
    print(f"\n===== DONE =====")


if __name__ == "__main__":
    main()
