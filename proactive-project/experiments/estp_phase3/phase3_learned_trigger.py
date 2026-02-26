#!/usr/bin/env python3
"""
Phase III: Lightweight Learned Trigger (Stage A / B ablation)
=============================================================
Single-head MLP with progressive feature addition.

Stage A  (6-dim):  D-features + memory basics             [offline, no GPU]
Stage B-goal (9-dim): + goal_state one-hot (on/unc/off)   [needs goal_state_cache.json]
Stage B-clip (8-dim): + clip_change + clip_relevance       [needs clip_cache.json]
Stage B-full (11-dim): + both caches

Evaluation: Task-Type Leave-One-Out (TTLOO) + video-level split check
Target: F1 > Adaptive-D TTLOO (fair out-of-sample baseline)

Usage:
    python3 proactive-project/experiments/estp_phase3/phase3_learned_trigger.py
    python3 proactive-project/experiments/estp_phase3/phase3_learned_trigger.py \\
        --checkpoint results/fullscale_d/checkpoint.jsonl \\
        --dataset ../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json \\
        --goal-state-cache results/phase3_learned/goal_state_cache.json \\
        --clip-cache results/phase3_learned/clip_cache.json \\
        --output results/phase3_learned/ \\
        --verbose
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Try PyTorch; fall back to sklearn ──────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Constants (mirrors hypothesis_runner.py / phase2) ─────────────────────
ANTICIPATION     = 1.0    # s: can trigger up to 1s before GT window
LATENCY          = 2.0    # s: can trigger up to 2s after GT window start
FRAME_FPS        = 2.0    # dataset fps
COOLDOWN_DEFAULT = 12.0   # s: cooldown between triggers

SLOPE_WINDOW        = 5    # steps for slope calculation
VOLATILITY_WINDOW   = 10   # steps for volatility std
STAGNATION_THRESH   = 0.3  # gap-volatility threshold to reset stagnation timer

FEATURE_DIM_A    = 6    # Stage A: D + memory basics
FEATURE_DIM_GOAL = 3    # goal_state one-hot: on_track / uncertain / off_track
FEATURE_DIM_CLIP = 2    # clip_change + clip_relevance
GOAL_STATE_INTERVAL = 10.0  # seconds between goal_state buckets

HIDDEN_DIMS  = (32, 16)
FEATURE_DIM  = FEATURE_DIM_A   # default; overridden per stage run
EPOCHS       = 150
LR           = 1e-3
THRESHOLD_SWEEP = np.arange(0.1, 0.95, 0.05)

DEFAULT_CHECKPOINT = "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET    = "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_OUTPUT     = "results/phase3_learned"

# Phase II Adaptive-D best-case F1 (in-sample) — target to beat
ADAPTIVE_D_F1 = 0.222


# ── ESTP-F1 utilities (inlined from hypothesis_runner.py) ─────────────────

def ceil_time_by_fps(t, fps, min_t, max_t):
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
                "tp": 0, "fp": 0, "fn": len(gt_spans)}

    hits = set()
    for gi, (gs, ge) in enumerate(gt_spans):
        for pred in preds:
            pt = pred.get("time", -1)
            if gs - anticipation <= pt <= ge + latency:
                hits.add(gi)
                break

    fp = sum(
        1 for pred in preds
        if not any(gs - anticipation <= pred.get("time", -1) <= ge + latency
                   for gs, ge in gt_spans)
    )
    tp, fn = len(hits), len(gt_spans) - len(hits)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "n_preds": len(preds), "n_gt": len(gt_spans),
            "tp": tp, "fp": fp, "fn": fn}


# ── Data Loading (pattern from phase2_stratified_analysis.py) ─────────────

def _build_dataset_index(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
    index = {}
    for video_uid, clips in data.items():
        for clip_uid, qa_list in clips.items():
            for qa in qa_list:
                q = qa.get("question", "")
                if not q:
                    for c in qa.get("conversation", []):
                        if c.get("role", "").lower() == "user":
                            q = c.get("content", "")
                            break
                index[(video_uid, clip_uid, q.strip())] = qa
    return index


def load_cases(checkpoint_path, dataset_path, verbose=False):
    """Load old-schema cases and attach qa_raw. Returns list of dicts."""
    ds_index = _build_dataset_index(dataset_path)
    if verbose:
        print(f"  Dataset index: {len(ds_index)} entries")

    cases, skipped = [], 0
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if "baseline_metrics" not in c:   # skip new-schema entries
                continue
            key = (c["video_uid"], c["clip_uid"], c.get("question", "").strip())
            qa_raw = ds_index.get(key)
            if qa_raw is None:
                skipped += 1
                if verbose:
                    print(f"  [SKIP] no match: {c.get('task_type')} | {c.get('question','')[:60]}")
                continue
            c["qa_raw"] = qa_raw
            cases.append(c)

    if verbose:
        print(f"  Loaded: {len(cases)} cases  |  skipped: {skipped}")
    return cases


# ── Feature Engineering ────────────────────────────────────────────────────

def extract_features(trigger_log, n_steps, video_duration):
    """
    Extract 6-dim feature matrix from trigger_log.

    Features:
      0  gap              logprob_gap at this step
      1  slope            gap change rate over last SLOPE_WINDOW steps (per sec)
      2  volatility       std of gap over last VOLATILITY_WINDOW steps
      3  since_last_trig  time since last triggered step, normalized to [0,1] (clip 60s)
      4  step_progress    step / n_steps
      5  stagnation       log-normalized time since last high-volatility event
    """
    n = len(trigger_log)
    if n == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    gaps = np.array(
        [e.get("logprob_gap", e.get("gap", 0.0)) for e in trigger_log],
        dtype=np.float32
    )
    times = np.array([e["time"] for e in trigger_log], dtype=np.float32)
    steps = np.array([e.get("step", i + 1) for i, e in enumerate(trigger_log)], dtype=np.float32)
    triggered = np.array([bool(e.get("triggered", False)) for e in trigger_log])

    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    last_trig_time = 0.0
    last_high_vol_time = float(times[0])

    for i in range(n):
        t = float(times[i])

        # 0: gap (raw, not clamped — keep sign information)
        feats[i, 0] = gaps[i]

        # 1: slope
        lo = max(0, i - SLOPE_WINDOW)
        dt = times[i] - times[lo]
        feats[i, 1] = (gaps[i] - gaps[lo]) / dt if dt > 1e-6 else 0.0

        # 2: volatility
        lo2 = max(0, i - VOLATILITY_WINDOW + 1)
        w = gaps[lo2: i + 1]
        feats[i, 2] = float(np.std(w)) if len(w) > 1 else 0.0

        # 3: since_last_trigger (normalized, clipped at 60s)
        feats[i, 3] = min(t - last_trig_time, 60.0) / 60.0
        if triggered[i]:
            last_trig_time = t

        # 4: step_progress
        feats[i, 4] = float(steps[i]) / max(float(n_steps), 1.0)

        # 5: stagnation (log-normalized by 5 min)
        if feats[i, 2] > STAGNATION_THRESH:
            last_high_vol_time = t
        stag = t - last_high_vol_time
        feats[i, 5] = math.log1p(stag) / math.log1p(300.0)

    return feats


def extract_labels(trigger_log, qa_raw):
    """Label each step 1 if within any GT window, 0 otherwise."""
    n = len(trigger_log)
    labels = np.zeros(n, dtype=np.float32)

    gt_windows = []
    if qa_raw and "conversation" in qa_raw:
        for turn in qa_raw["conversation"]:
            if turn.get("role", "").lower() == "assistant":
                gs = turn.get("start_time")
                ge = turn.get("end_time")
                if gs is not None and ge is not None:
                    gt_windows.append((gs - ANTICIPATION, ge + LATENCY))

    for i, entry in enumerate(trigger_log):
        t = entry["time"]
        if any(ws <= t <= we for ws, we in gt_windows):
            labels[i] = 1.0

    return labels


# ── Goal-State Cache Loading ───────────────────────────────────────────────

GOAL_STATE_ENCODING = {
    "on_track":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "uncertain": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "off_track": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def load_goal_state_cache(cache_path):
    """Load goal_state_cache.json. Returns {} if file not found."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_clip_cache(cache_path):
    """Load clip_cache.json. Returns {} if file not found."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def get_goal_state_encoding(goal_state_cache, case_id, time_sec, interval=GOAL_STATE_INTERVAL):
    """
    Look up goal_state for a given time in the cache.
    Uses most recent bucket: floor(time_sec / interval) * interval.
    Returns 3-dim one-hot encoding.
    """
    case_data = goal_state_cache.get(case_id, {})
    if not case_data:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)  # default: uncertain

    bucket = math.floor(time_sec / interval) * interval
    # Try exact bucket first, then scan backwards for most-recent available
    for t in [bucket, bucket - interval, bucket - 2 * interval]:
        label = case_data.get(str(float(t))) or case_data.get(str(int(t)))
        if label in GOAL_STATE_ENCODING:
            return GOAL_STATE_ENCODING[label]
    return GOAL_STATE_ENCODING["uncertain"]


def build_dataset(cases, goal_state_cache=None, clip_cache=None, verbose=False):
    """
    Build per-case feature matrices and labels.

    Stage A (default):   features = D-features (6-dim)
    Stage B-goal:        features = D-features + goal_state one-hot (9-dim)
    Stage B-clip:        features = D-features + clip dynamics (8-dim)
    Stage B-full:        features = D-features + goal_state + clip (11-dim)

    goal_state_cache and clip_cache are dicts loaded from JSON cache files.
    If None or empty, corresponding features are omitted (graceful fallback).
    """
    use_goal = bool(goal_state_cache)
    use_clip = bool(clip_cache)

    n_total_pos = 0
    for c in cases:
        tlog     = c.get("trigger_log", [])
        n_steps  = c.get("n_steps", len(tlog))
        duration = c.get("duration", 300.0)
        cid      = c["case_id"]

        # Base D-features (6-dim)
        base_feats = extract_features(tlog, n_steps, duration)

        # Optional: goal_state features (3-dim one-hot per step)
        if use_goal:
            goal_feats = np.array([
                get_goal_state_encoding(goal_state_cache, cid, e["time"])
                for e in tlog
            ], dtype=np.float32)
        else:
            goal_feats = None

        # Optional: CLIP dynamics features (2-dim per step)
        if use_clip:
            case_clip = clip_cache.get(cid, [])
            # Build lookup by time (rounded to 3 decimals for matching)
            clip_by_time = {round(e["time"], 2): e for e in case_clip}
            clip_feats = np.array([
                [clip_by_time.get(round(e["time"], 2), {}).get("clip_change", 0.0),
                 clip_by_time.get(round(e["time"], 2), {}).get("clip_relevance", 0.0)]
                for e in tlog
            ], dtype=np.float32)
        else:
            clip_feats = None

        # Concatenate available features
        parts = [base_feats]
        if goal_feats is not None:
            parts.append(goal_feats)
        if clip_feats is not None:
            parts.append(clip_feats)
        c["features"] = np.hstack(parts) if len(parts) > 1 else base_feats

        c["labels"] = extract_labels(tlog, c["qa_raw"])
        c["times"]  = np.array([e["time"] for e in tlog], dtype=np.float32)
        n_total_pos += int(c["labels"].sum())

    feat_dim = cases[0]["features"].shape[1] if cases else 0
    stage_tag = (
        "B-full" if use_goal and use_clip else
        "B-goal" if use_goal else
        "B-clip" if use_clip else
        "A"
    )
    if verbose:
        total_steps = sum(len(c["features"]) for c in cases)
        print(f"  Stage {stage_tag}: feature_dim={feat_dim}  "
              f"steps={total_steps}  pos={n_total_pos}  "
              f"pos_rate={n_total_pos/total_steps:.1%}")
    return cases, stage_tag


# ── MLP Model (PyTorch) ────────────────────────────────────────────────────

class AssistMLP(nn.Module):
    def __init__(self, input_dim, hidden=(32, 16)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


SEED = 42


def _set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_torch(X_tr, y_tr, verbose=False):
    """Train AssistMLP with BCEWithLogitsLoss and pos_weight."""
    _set_seed()
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)])

    input_dim = X_tr.shape[1]
    model = AssistMLP(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X = torch.FloatTensor(X_tr)
    y = torch.FloatTensor(y_tr)

    model.train()
    best_loss, patience, wait = 1e9, 15, 0
    best_state = None
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_probs_torch(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        return torch.sigmoid(logits).numpy()


def train_sklearn(X_tr, y_tr):
    """Fallback: sklearn MLPClassifier with balanced class_weight."""
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_DIMS,
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    # sklearn doesn't support class_weight for MLP natively; use sample_weight
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    w_pos = (len(y_tr) / (2 * max(n_pos, 1)))
    w_neg = (len(y_tr) / (2 * max(n_neg, 1)))
    sw = np.where(y_tr == 1, w_pos, w_neg)
    clf.fit(X_tr, y_tr.astype(int), sample_weight=sw)
    return clf


def predict_probs_sklearn(clf, X):
    return clf.predict_proba(X)[:, 1]


# ── Trigger Decoding ───────────────────────────────────────────────────────

def decode_triggers(probs, times, threshold, cooldown=COOLDOWN_DEFAULT):
    """Convert step-level probabilities into trigger timestamps."""
    trigger_times = []
    last_trigger = -cooldown - 1.0
    for prob, t in zip(probs, times):
        if prob >= threshold and (t - last_trigger) >= cooldown:
            trigger_times.append(float(t))
            last_trigger = float(t)
    return trigger_times


def evaluate_case_from_triggers(trigger_times, qa_raw):
    dialog = [{"role": "assistant", "time": t} for t in trigger_times]
    return compute_case_metrics(dialog, qa_raw)


# ── Threshold Sweep (on training fold) ────────────────────────────────────

def find_best_threshold(cases_train, predict_fn, cooldown=COOLDOWN_DEFAULT):
    """Sweep thresholds on training cases; return threshold maximising macro-F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in THRESHOLD_SWEEP:
        f1s = []
        for c in cases_train:
            probs = predict_fn(c["features_scaled"])
            tt = decode_triggers(probs, c["times"], thr, cooldown)
            m = evaluate_case_from_triggers(tt, c["qa_raw"])
            f1s.append(m["f1"])
        avg = float(np.mean(f1s))
        if avg > best_f1:
            best_f1 = avg
            best_thr = float(thr)
    return best_thr, best_f1


# ── Baselines (re-simulate from trigger_log) ──────────────────────────────

def simulate_at_tau(trigger_log, qa_raw, tau, cooldown=COOLDOWN_DEFAULT):
    """Replay trigger_log at given tau+cooldown. Returns metrics dict."""
    dialog = []
    last_trig = -cooldown - 1.0
    for entry in trigger_log:
        gap = entry.get("logprob_gap", entry.get("gap", 0.0))
        t = entry["time"]
        if gap > tau and (t - last_trig) >= cooldown:
            dialog.append({"role": "assistant", "time": t})
            last_trig = t
    return compute_case_metrics(dialog, qa_raw)


TAU_RANGE = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0]

PHASE2_OPTIMAL_TAU = {
    "Object Localization":                5.0,
    "Information Function":               2.8,
    "Text-Rich Understanding":            2.0,
    "Object State Change Recognition":    2.5,
    "Action Reasoning":                   2.5,
    "Attribute Perception":               0.5,
    "Action Recognition":                 0.5,
    "Object Recognition":                 2.5,
    "Object Function":                    2.0,
    "Task Understanding":                 0.0,
    "Ego Object Localization":            2.0,
    "Ego Object State Change Recognition": 1.0,
}


def ttloo_adaptive_d(cases, tau_range=TAU_RANGE, cooldown=COOLDOWN_DEFAULT, verbose=False):
    """
    TTLOO for Adaptive-D: fair out-of-sample comparison.
    Each fold:
      - Find per-type optimal tau on TRAINING cases only.
      - For test type (unseen in training): use global best tau as fallback.
    Returns: {task_type: avg_f1}
    """
    task_types = sorted(set(c["task_type"] for c in cases))
    by_type = defaultdict(list)
    for c in cases:
        by_type[c["task_type"]].append(c)

    results = {}
    for test_type in task_types:
        train = [c for c in cases if c["task_type"] != test_type]
        test  = [c for c in cases if c["task_type"] == test_type]
        if not train or not test:
            continue

        # Per-type optimal tau on training fold
        train_by_type = defaultdict(list)
        for c in train:
            train_by_type[c["task_type"]].append(c)

        type_opt_tau = {}
        for tt, tc in train_by_type.items():
            best_tau, best_f1 = tau_range[0], -1.0
            for tau in tau_range:
                f1s = [simulate_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)["f1"]
                       for c in tc]
                avg = float(np.mean(f1s))
                if avg > best_f1:
                    best_f1 = avg
                    best_tau = tau
            type_opt_tau[tt] = best_tau

        # Fallback: global best tau on training fold
        best_global_tau, best_global_f1 = tau_range[0], -1.0
        for tau in tau_range:
            f1s = [simulate_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)["f1"]
                   for c in train]
            avg = float(np.mean(f1s))
            if avg > best_global_f1:
                best_global_f1 = avg
                best_global_tau = tau

        # Evaluate test fold
        fold_f1s = []
        for c in test:
            tau = type_opt_tau.get(c["task_type"], best_global_tau)
            m = simulate_at_tau(c["trigger_log"], c["qa_raw"], tau, cooldown)
            fold_f1s.append(m["f1"])

        results[test_type] = float(np.mean(fold_f1s))
        if verbose:
            print(f"  [AdaptiveD-TTLOO {test_type:<40}] F1={results[test_type]:.3f}  "
                  f"fallback_tau={best_global_tau}")

    return results


def compute_baselines(cases):
    """Compute baseline, D@2.8, and Adaptive-D F1 for each case."""
    results = {}
    for c in cases:
        tlog = c["trigger_log"]
        qa = c["qa_raw"]
        results[c["case_id"]] = {
            "baseline":   simulate_at_tau(tlog, qa, tau=0.0, cooldown=0.0),
            "d28":        simulate_at_tau(tlog, qa, tau=2.8, cooldown=COOLDOWN_DEFAULT),
            "adaptive_d": simulate_at_tau(
                tlog, qa,
                tau=PHASE2_OPTIMAL_TAU.get(c["task_type"], 2.8),
                cooldown=COOLDOWN_DEFAULT
            ),
        }
    return results


# ── TTLOO Evaluation ───────────────────────────────────────────────────────

def ttloo_evaluation(cases, use_torch=True, cooldown=COOLDOWN_DEFAULT, verbose=False):
    """
    Task-Type Leave-One-Out cross-validation.
    Each fold: train on 11 task types, test on held-out 1 type.
    StandardScaler fitted on train fold, applied to test (no leakage).
    Threshold selected on train fold.
    """
    task_types = sorted(set(c["task_type"] for c in cases))
    all_fold_results = {}

    for test_type in task_types:
        train = [c for c in cases if c["task_type"] != test_type]
        test  = [c for c in cases if c["task_type"] == test_type]

        if not train or not test:
            continue

        # Fit scaler on train features
        X_tr_all = np.vstack([c["features"] for c in train])
        y_tr_all = np.concatenate([c["labels"] for c in train])

        scaler = _fit_scaler(X_tr_all)
        for c in train:
            c["features_scaled"] = scaler.transform(c["features"])
        for c in test:
            c["features_scaled"] = scaler.transform(c["features"])

        # Train model
        X_tr_scaled = np.vstack([c["features_scaled"] for c in train])
        if use_torch and TORCH_AVAILABLE:
            model = train_torch(X_tr_scaled, y_tr_all, verbose=False)
            predict_fn = lambda X, m=model: predict_probs_torch(m, X)
        else:
            model = train_sklearn(X_tr_scaled, y_tr_all)
            predict_fn = lambda X, m=model: predict_probs_sklearn(m, X)

        # Tune threshold on train fold
        best_thr, train_f1 = find_best_threshold(train, predict_fn, cooldown)

        # Evaluate on test fold
        fold_results = []
        for c in test:
            probs = predict_fn(c["features_scaled"])
            tt = decode_triggers(probs, c["times"], best_thr, cooldown)
            m = evaluate_case_from_triggers(tt, c["qa_raw"])
            # Store per-step predictions for diagnostics
            triggered_mask = np.array(
                [1 if t in tt else 0 for t in c["times"]], dtype=np.float32
            )
            fold_results.append({
                "case_id":   c["case_id"],
                "task_type": c["task_type"],
                "video_uid": c["video_uid"],
                "f1":        m["f1"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "fp":        m["fp"],
                "n_preds":   m["n_preds"],
                "threshold": best_thr,
                "_probs":    probs,
                "_triggered": triggered_mask,
                "_labels":   c["labels"],
                "_times":    c["times"],
            })

        avg_f1 = float(np.mean([r["f1"] for r in fold_results]))
        all_fold_results[test_type] = {
            "cases": fold_results,
            "avg_f1": avg_f1,
            "threshold": best_thr,
            "n_train": len(train),
            "n_test": len(test),
        }

        if verbose:
            print(f"  [{test_type:45s}] test_F1={avg_f1:.3f}  thr={best_thr:.2f}  "
                  f"(n_train={len(train)}, n_test={len(test)})")

    return all_fold_results


def _fit_scaler(X):
    """Simple StandardScaler."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    class _Scaler:
        def transform(self, X):
            return (X - mean) / std
    return _Scaler()


# ── Video-Level Split Check ────────────────────────────────────────────────

def video_level_split_check(cases, use_torch=True, cooldown=COOLDOWN_DEFAULT, verbose=False):
    """
    Within each task type, split by video_uid (train=all other videos, test=one video).
    Checks if model learns scene biases rather than generalizable features.
    """
    by_type = defaultdict(list)
    for c in cases:
        by_type[c["task_type"]].append(c)

    results = {}
    for task_type, type_cases in by_type.items():
        video_uids = list(set(c["video_uid"] for c in type_cases))
        if len(video_uids) < 2:
            continue  # can't split with only 1 video

        fold_f1s = []
        for test_vid in video_uids:
            train = [c for c in type_cases if c["video_uid"] != test_vid]
            test  = [c for c in type_cases if c["video_uid"] == test_vid]
            if not train or not test:
                continue

            X_tr = np.vstack([c["features"] for c in train])
            y_tr = np.concatenate([c["labels"] for c in train])
            scaler = _fit_scaler(X_tr)
            for c in train:
                c["features_scaled_vl"] = scaler.transform(c["features"])
            for c in test:
                c["features_scaled_vl"] = scaler.transform(c["features"])

            X_tr_s = np.vstack([c["features_scaled_vl"] for c in train])
            if use_torch and TORCH_AVAILABLE:
                model = train_torch(X_tr_s, y_tr)
                predict_fn = lambda X, m=model: predict_probs_torch(m, X)
            else:
                model = train_sklearn(X_tr_s, y_tr)
                predict_fn = lambda X, m=model: predict_probs_sklearn(m, X)

            # Use fixed threshold 0.3 for video-level check (no tuning bias)
            for c in test:
                probs = predict_fn(c["features_scaled_vl"])
                tt = decode_triggers(probs, c["times"], threshold=0.3, cooldown=cooldown)
                m = evaluate_case_from_triggers(tt, c["qa_raw"])
                fold_f1s.append(m["f1"])

        if fold_f1s:
            results[task_type] = float(np.mean(fold_f1s))

    return results


# ── Bootstrap CI ──────────────────────────────────────────────────────────

def bootstrap_delta_ci(a, b, n=1000, seed=42):
    """Paired bootstrap 95% CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.array(a) - np.array(b)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


# ── Report Generation ──────────────────────────────────────────────────────

def generate_report(ttloo_results, baseline_results, video_check, output_dir,
                    ttloo_adp_d=None):
    lines = []
    W = 80
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def sep(char="─"):
        lines.append(char * W)

    def hdr(title):
        sep("═")
        lines.append(title)
        sep("═")

    hdr(f"Phase III Stage A: Learned Trigger (D-features MLP)")
    lines.append(f"Generated: {now}")
    lines.append(f"Evaluation: Task-Type Leave-One-Out (TTLOO)")
    lines.append(f"Features:   gap, slope, volatility, since_last_trig, step_progress, stagnation")
    lines.append(f"Model:      MLP {FEATURE_DIM}→32→16→1  |  Backend: {'PyTorch' if TORCH_AVAILABLE else 'sklearn'}")
    lines.append(f"Target:     F1 > {ADAPTIVE_D_F1} (Adaptive-D Phase II in-sample)")

    # ── Section 1: Aggregate comparison ──────────────────────────────────
    sep()
    lines.append("SECTION 1: AGGREGATE RESULTS (TTLOO)")
    sep()

    ttloo_f1s_by_case = {}   # case_id → mlp_f1
    for fold in ttloo_results.values():
        for r in fold["cases"]:
            ttloo_f1s_by_case[r["case_id"]] = r["f1"]

    mlp_f1s, bl_f1s, d28_f1s, adp_f1s = [], [], [], []
    for cid, bl in baseline_results.items():
        if cid not in ttloo_f1s_by_case:
            continue
        mlp_f1s.append(ttloo_f1s_by_case[cid])
        bl_f1s.append(bl["baseline"]["f1"])
        d28_f1s.append(bl["d28"]["f1"])
        adp_f1s.append(bl["adaptive_d"]["f1"])

    avg = lambda xs: float(np.mean(xs)) if xs else 0.0
    lines.append(f"  {'Method':<35}  {'Avg F1':>8}  {'N cases':>8}")
    lines.append(f"  {'─'*35}  {'─'*8}  {'─'*8}")
    lines.append(f"  {'Baseline (τ=0, no cd)':<35}  {avg(bl_f1s):>8.3f}  {len(bl_f1s):>8}")
    lines.append(f"  {'D@2.8 + cd=12':<35}  {avg(d28_f1s):>8.3f}  {len(d28_f1s):>8}")
    lines.append(f"  {'Adaptive-D (Phase II in-sample ref)':<35}  {ADAPTIVE_D_F1:>8.3f}  {'—':>8}")

    # TTLOO Adaptive-D (fair comparison)
    ttloo_adp_f1s = []
    if ttloo_adp_d:
        for fold in ttloo_results.values():
            for r in fold["cases"]:
                tt = r["task_type"]
                if tt in ttloo_adp_d:
                    ttloo_adp_f1s.append(ttloo_adp_d[tt])
                # use per-case approximation: just the fold average for this type
        # Simpler: case-level from type averages
        ttloo_adp_f1s = []
        for fold in ttloo_results.values():
            for r in fold["cases"]:
                tt = r["task_type"]
                ttloo_adp_f1s.append(ttloo_adp_d.get(tt, 0.0))

    if ttloo_adp_f1s:
        lines.append(f"  {'Adaptive-D (TTLOO, fair baseline)':<35}  {avg(ttloo_adp_f1s):>8.3f}  {len(ttloo_adp_f1s):>8}")
    lines.append(f"  {'MLP Stage A (TTLOO)':<35}  {avg(mlp_f1s):>8.3f}  {len(mlp_f1s):>8}")

    # Gate 1: MLP vs TTLOO Adaptive-D (fair)
    lines.append("")
    if ttloo_adp_f1s and len(ttloo_adp_f1s) == len(mlp_f1s):
        delta_mean, ci_lo, ci_hi = bootstrap_delta_ci(mlp_f1s, ttloo_adp_f1s)
        lines.append(f"  Gate 1 — MLP vs Adaptive-D TTLOO (fair):")
        lines.append(f"    Delta F1:  mean={delta_mean:+.3f}  95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]")
        gate = "PASS" if ci_lo > 0 else ("BORDERLINE" if delta_mean > 0 else "FAIL")
        lines.append(f"    >>> {gate} <<<")
    else:
        delta_mean, ci_lo, ci_hi = bootstrap_delta_ci(mlp_f1s, adp_f1s)
        lines.append(f"  Gate 1 — MLP vs Adaptive-D in-sample (ref only):")
        lines.append(f"    Delta F1:  mean={delta_mean:+.3f}  95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]")
        lines.append(f"    Note: Adaptive-D is in-sample; MLP is out-of-sample. Comparison is informative only.")

    # MLP vs D@2.8
    delta2, lo2, hi2 = bootstrap_delta_ci(mlp_f1s, d28_f1s)
    lines.append(f"  Gate 2 — MLP vs D@2.8: mean={delta2:+.3f}  95% CI=[{lo2:+.3f}, {hi2:+.3f}]")

    # ── Section 2: Per-type results ───────────────────────────────────────
    sep()
    lines.append("SECTION 2: PER TASK-TYPE RESULTS (TTLOO)")
    sep()

    bl_by_type = defaultdict(list)
    adp_by_type = defaultdict(list)
    for c_id, bl in baseline_results.items():
        # find task_type from ttloo results
        for fold in ttloo_results.values():
            for r in fold["cases"]:
                if r["case_id"] == c_id:
                    bl_by_type[r["task_type"]].append(bl["baseline"]["f1"])
                    adp_by_type[r["task_type"]].append(bl["adaptive_d"]["f1"])

    hdr_row = f"  {'Task Type':<45}  {'n':>3}  {'BL F1':>6}  {'Adp-D F1':>8}  {'MLP F1':>7}  {'Δ(MLP-BL)':>10}  {'thr':>5}"
    lines.append(hdr_row)
    lines.append(f"  {'─'*45}  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*5}")

    sorted_types = sorted(ttloo_results.keys(),
                          key=lambda t: -ttloo_results[t]["avg_f1"])
    for tt in sorted_types:
        fold = ttloo_results[tt]
        mlp_f1 = fold["avg_f1"]
        bl_f1  = avg(bl_by_type.get(tt, [0]))
        adp_f1 = avg(adp_by_type.get(tt, [0]))
        delta  = mlp_f1 - bl_f1
        n      = fold["n_test"]
        thr    = fold["threshold"]
        lines.append(f"  {tt:<45}  {n:>3}  {bl_f1:>6.3f}  {adp_f1:>8.3f}  "
                     f"{mlp_f1:>7.3f}  {delta:>+10.3f}  {thr:>5.2f}")

    # ── Section 3: Case-level details ────────────────────────────────────
    sep()
    lines.append("SECTION 3: CASE-LEVEL RESULTS")
    sep()

    all_case_rows = []
    for fold in ttloo_results.values():
        for r in fold["cases"]:
            cid = r["case_id"]
            bl_f1  = baseline_results.get(cid, {}).get("baseline", {}).get("f1", 0)
            adp_f1 = baseline_results.get(cid, {}).get("adaptive_d", {}).get("f1", 0)
            all_case_rows.append({
                "cid": cid,
                "task_type": r["task_type"],
                "mlp_f1": r["f1"],
                "bl_f1": bl_f1,
                "adp_f1": adp_f1,
                "delta": r["f1"] - bl_f1,
            })

    improved  = sum(1 for r in all_case_rows if r["delta"] > 0.01)
    degraded  = sum(1 for r in all_case_rows if r["delta"] < -0.01)
    unchanged = len(all_case_rows) - improved - degraded
    lines.append(f"  Improved:  {improved}/{len(all_case_rows)} ({improved/max(len(all_case_rows),1):.0%})")
    lines.append(f"  Degraded:  {degraded}/{len(all_case_rows)} ({degraded/max(len(all_case_rows),1):.0%})")
    lines.append(f"  Unchanged: {unchanged}/{len(all_case_rows)}")

    lines.append("")
    lines.append("  Top 5 improved cases (MLP vs Baseline):")
    for r in sorted(all_case_rows, key=lambda x: -x["delta"])[:5]:
        lines.append(f"    [{r['task_type']:<42}] Δ={r['delta']:+.3f}  "
                     f"BL={r['bl_f1']:.3f} → MLP={r['mlp_f1']:.3f}")
    lines.append("")
    lines.append("  Top 5 degraded cases (MLP vs Baseline):")
    for r in sorted(all_case_rows, key=lambda x: x["delta"])[:5]:
        lines.append(f"    [{r['task_type']:<42}] Δ={r['delta']:+.3f}  "
                     f"BL={r['bl_f1']:.3f} → MLP={r['mlp_f1']:.3f}")

    # ── Section 4: Video-level split check ───────────────────────────────
    sep()
    lines.append("SECTION 4: VIDEO-LEVEL SPLIT CHECK")
    sep()
    lines.append("  Purpose: detect if model learned scene-specific biases.")
    lines.append("  Method:  within-type train/test split by video_uid (threshold=0.3 fixed).")
    lines.append("")
    if video_check:
        lines.append(f"  {'Task Type':<45}  {'Vid-split F1':>12}")
        lines.append(f"  {'─'*45}  {'─'*12}")
        for tt, f1 in sorted(video_check.items(), key=lambda x: -x[1]):
            ttloo_f1 = ttloo_results.get(tt, {}).get("avg_f1", 0.0)
            gap_flag = "⚠ large drop" if (ttloo_f1 - f1) > 0.05 else ""
            lines.append(f"  {tt:<45}  {f1:>12.3f}  (TTLOO={ttloo_f1:.3f}) {gap_flag}")
    else:
        lines.append("  [skipped — insufficient data for video-level split]")

    # ── Section 5: Feature summary ────────────────────────────────────────
    sep()
    lines.append("SECTION 5: FEATURE SUMMARY")
    sep()
    lines.append("  Stage A features (6-dim):")
    lines.append("    0: gap              — raw logprob_gap (primary D signal)")
    lines.append("    1: slope            — gap change rate over last 5 steps")
    lines.append("    2: volatility       — gap std over last 10 steps")
    lines.append("    3: since_last_trig  — time since last trigger (norm 60s)")
    lines.append("    4: step_progress    — step / n_steps")
    lines.append("    5: stagnation       — log-norm time since last high-vol event")
    lines.append("")
    lines.append("  Stage B (next): + CLIP ViT-B/32 embedding (64-dim)")
    lines.append("  Stage C (next): + goal_state_gap_score (3-dim one-hot, VLM 10s)")

    sep("═")
    lines.append(f"Report generated: {len(all_case_rows)} cases  |  {len(ttloo_results)} TTLOO folds")
    sep("═")

    return "\n".join(lines)


# ── Diagnostics ───────────────────────────────────────────────────────────

def diagnose_goal_state_trigger(cases, ttloo_results, goal_state_cache):
    """
    Diagnostic 1: Cross-tab goal_state label × MLP trigger.
    Reports P(trigger | label) and P(hit | label, triggered).
    """
    if not goal_state_cache:
        return ""

    label_counts    = {"on_track": 0, "uncertain": 0, "off_track": 0}
    label_triggered = {"on_track": 0, "uncertain": 0, "off_track": 0}
    label_hit       = {"on_track": 0, "uncertain": 0, "off_track": 0}

    for fold_data in ttloo_results.values():
        for r in fold_data["cases"]:
            cid = r["case_id"]
            if "_triggered" not in r or "_labels" not in r or "_times" not in r:
                continue
            for j in range(len(r["_times"])):
                t = float(r["_times"][j])
                gs_enc = get_goal_state_encoding(goal_state_cache, cid, t)
                if gs_enc[0] > 0.5:
                    gs_label = "on_track"
                elif gs_enc[2] > 0.5:
                    gs_label = "off_track"
                else:
                    gs_label = "uncertain"
                label_counts[gs_label] += 1
                if r["_triggered"][j] > 0.5:
                    label_triggered[gs_label] += 1
                    if r["_labels"][j] > 0.5:
                        label_hit[gs_label] += 1

    lines = []
    lines.append("\n┌─────────────────────────────────────────────────┐")
    lines.append("│  DIAGNOSTIC 1: Goal-State × Trigger Cross-Tab   │")
    lines.append("├────────────┬───────┬──────────────┬──────────────┤")
    lines.append("│ label      │ count │ P(trig|label)│ P(hit|trig,l)│")
    lines.append("├────────────┼───────┼──────────────┼──────────────┤")
    for lab in ["on_track", "uncertain", "off_track"]:
        n = label_counts[lab]
        tr = label_triggered[lab]
        h  = label_hit[lab]
        p_trig = tr / max(n, 1)
        p_hit  = h / max(tr, 1)
        lines.append(f"│ {lab:<10s} │ {n:>5d} │    {p_trig:>6.1%}    │    {p_hit:>6.1%}    │")
    lines.append("└────────────┴───────┴──────────────┴──────────────┘")

    total_tr = sum(label_triggered.values())
    total_hit = sum(label_hit.values())
    lines.append(f"  Total triggered: {total_tr}  |  Total hits: {total_hit}  "
                 f"|  Overall precision: {total_hit/max(total_tr,1):.1%}")
    return "\n".join(lines)


def diagnose_per_task_type_delta(ttloo_results, ttloo_adp_d, goal_state_cache=None, cases=None):
    """
    Diagnostic 2: Per-task-type F1 delta (MLP vs Adaptive-D) + goal_state informative rate.
    """
    lines = []
    lines.append("\n┌──────────────────────────────────────────────────────────────────────┐")
    lines.append("│  DIAGNOSTIC 2: Per-Task-Type F1 Delta + Goal-State Separation        │")
    lines.append("├─────────────────────────────────┬────────┬────────┬────────┬──────────┤")
    lines.append("│ Task Type                       │ MLP F1 │ AdpD F1│  Δ F1  │ GS info% │")
    lines.append("├─────────────────────────────────┼────────┼────────┼────────┼──────────┤")

    gs_info_by_type = {}
    if goal_state_cache and cases:
        from collections import defaultdict
        type_labels = defaultdict(list)
        for c in cases:
            cid = c["case_id"]
            tt = c.get("task_type", "?")
            case_gs = goal_state_cache.get(cid, {})
            for label in case_gs.values():
                type_labels[tt].append(label)
        for tt, labels in type_labels.items():
            informative = sum(1 for l in labels if l in ("on_track", "off_track"))
            gs_info_by_type[tt] = informative / max(len(labels), 1)

    wins, losses = 0, 0
    for test_type in sorted(ttloo_results.keys()):
        fold = ttloo_results[test_type]
        mlp_f1 = fold["avg_f1"]
        adp_f1 = ttloo_adp_d.get(test_type, 0.0)
        delta = mlp_f1 - adp_f1
        gs_info = gs_info_by_type.get(test_type, float("nan"))
        tag = "+" if delta > 0.005 else ("-" if delta < -0.005 else "=")
        if delta > 0.005:
            wins += 1
        elif delta < -0.005:
            losses += 1
        gs_str = f"{gs_info:>6.1%}" if not math.isnan(gs_info) else "   N/A"
        lines.append(f"│ {test_type:<31s} │ {mlp_f1:>6.3f} │ {adp_f1:>6.3f} │ {delta:>+6.3f}{tag}│ {gs_str}   │")

    lines.append("├─────────────────────────────────┴────────┴────────┴────────┴──────────┤")
    n_tie = len(ttloo_results) - wins - losses
    lines.append(f"│  Wins: {wins}  Losses: {losses}  Ties: {n_tie}  "
                 f"(win = MLP > AdpD by > 0.005)                     │")
    lines.append("└────────────────────────────────────────────────────────────────────────┘")
    return "\n".join(lines)


def bootstrap_paired_ci(mlp_f1_by_case, adp_f1_by_case, n_boot=5000, alpha=0.05):
    """Bootstrap paired delta CI: MLP_F1 - AdaptiveD_F1 per case."""
    rng = np.random.RandomState(42)
    deltas = np.array(mlp_f1_by_case) - np.array(adp_f1_by_case)
    n = len(deltas)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(deltas[idx])))
    boot_means = sorted(boot_means)
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2))]
    return float(np.mean(deltas)), lo, hi


# ── Main ──────────────────────────────────────────────────────────────────

def run_one_stage(cases, stage_tag, use_torch, cooldown, output_dir,
                  baseline_results, ttloo_adp_d, no_vl_check, verbose):
    """Run TTLOO + report for a single feature configuration."""
    print(f"\n{'─'*50}")
    print(f"Running Stage {stage_tag} (feature_dim={cases[0]['features'].shape[1]})...")

    ttloo_results = ttloo_evaluation(cases, use_torch=use_torch,
                                     cooldown=cooldown, verbose=verbose)
    mlp_f1s = [r["f1"] for fold in ttloo_results.values() for r in fold["cases"]]
    print(f"  MLP TTLOO [{stage_tag}]:  F1={np.mean(mlp_f1s):.3f}  (n={len(mlp_f1s)})")

    video_check = {}
    if not no_vl_check:
        video_check = video_level_split_check(cases, use_torch=use_torch,
                                              cooldown=cooldown, verbose=verbose)

    report = generate_report(ttloo_results, baseline_results, video_check, output_dir,
                             ttloo_adp_d=ttloo_adp_d)

    fname = f"phase3_stage_{stage_tag.lower().replace('-', '_')}_report.txt"
    report_path = output_dir / fname
    report_path.write_text(report)
    print(f"  Report: {report_path}")

    return ttloo_results, float(np.mean(mlp_f1s))


def main():
    parser = argparse.ArgumentParser(description="Phase III: Learned Trigger (Stage A/B ablation)")
    parser.add_argument("--checkpoint",        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",           default=DEFAULT_DATASET)
    parser.add_argument("--goal-state-cache",  default=None,
                        help="Path to goal_state_cache.json (enables Stage B-goal)")
    parser.add_argument("--clip-cache",        default=None,
                        help="Path to clip_cache.json (enables Stage B-clip)")
    parser.add_argument("--output",            default=DEFAULT_OUTPUT)
    parser.add_argument("--cooldown",          type=float, default=COOLDOWN_DEFAULT)
    parser.add_argument("--no-vl-check",       action="store_true",
                        help="Skip video-level split check (faster)")
    parser.add_argument("--stage-a-only",      action="store_true",
                        help="Run only Stage A (offline, no caches needed)")
    parser.add_argument("--verbose", "-v",     action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    checkpoint = script_dir / args.checkpoint
    dataset    = script_dir / args.dataset
    output_dir = script_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not TORCH_AVAILABLE and not SKLEARN_AVAILABLE:
        raise RuntimeError("Neither PyTorch nor sklearn is available.")

    backend = "PyTorch" if TORCH_AVAILABLE else "sklearn"
    print(f"[Phase III Learned Trigger]  backend={backend}  cooldown={args.cooldown}s")

    # ── Load caches ───────────────────────────────────────────────────────
    goal_state_cache = {}
    clip_cache       = {}

    if args.goal_state_cache:
        p = script_dir / args.goal_state_cache
        goal_state_cache = load_goal_state_cache(str(p))
        print(f"Goal-state cache: {len(goal_state_cache)} cases  ({p})")
    if args.clip_cache:
        p = script_dir / args.clip_cache
        clip_cache = load_clip_cache(str(p))
        print(f"CLIP cache:       {len(clip_cache)} cases  ({p})")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading cases...")
    cases = load_cases(str(checkpoint), str(dataset), verbose=args.verbose)
    if not cases:
        raise RuntimeError("No cases loaded. Check checkpoint and dataset paths.")

    # ── Compute baselines (once, shared across stages) ────────────────────
    print("Computing baselines...")
    baseline_results = compute_baselines(cases)
    bl_f1s  = [baseline_results[c["case_id"]]["baseline"]["f1"] for c in cases]
    d28_f1s = [baseline_results[c["case_id"]]["d28"]["f1"]       for c in cases]
    adp_f1s = [baseline_results[c["case_id"]]["adaptive_d"]["f1"] for c in cases]
    print(f"  Baseline:   F1={np.mean(bl_f1s):.3f}")
    print(f"  D@2.8:      F1={np.mean(d28_f1s):.3f}")
    print(f"  Adaptive-D (in-sample): F1={np.mean(adp_f1s):.3f}")

    # ── TTLOO Adaptive-D once ─────────────────────────────────────────────
    print("Running TTLOO Adaptive-D (fair baseline)...")
    ttloo_adp_d = ttloo_adaptive_d(cases, cooldown=args.cooldown, verbose=False)
    print(f"  Adaptive-D TTLOO: F1={np.mean(list(ttloo_adp_d.values())):.3f}")

    # ── Stage A: D-features only ──────────────────────────────────────────
    print("\nExtracting Stage A features...")
    cases, stage_a_tag = build_dataset(cases, verbose=args.verbose)
    stage_a_results, stage_a_f1 = run_one_stage(
        cases, "A", TORCH_AVAILABLE, args.cooldown, output_dir,
        baseline_results, ttloo_adp_d, args.no_vl_check, args.verbose
    )

    if args.stage_a_only:
        print(f"\nStage A done. F1={stage_a_f1:.3f}")
        return

    # ── Stage B-goal: + goal_state ────────────────────────────────────────
    stage_b_goal_f1 = None
    b_goal_ttloo_results = None
    if goal_state_cache:
        print("\nExtracting Stage B-goal features (D + goal_state)...")
        cases, stage_bg_tag = build_dataset(
            cases, goal_state_cache=goal_state_cache, verbose=args.verbose
        )
        b_goal_ttloo_results, stage_b_goal_f1 = run_one_stage(
            cases, "B-goal", TORCH_AVAILABLE, args.cooldown, output_dir,
            baseline_results, ttloo_adp_d, args.no_vl_check, args.verbose
        )

        # ── Diagnostics (cheap, no extra GPU) ─────────────────────────────
        print(diagnose_goal_state_trigger(cases, b_goal_ttloo_results, goal_state_cache))
        print(diagnose_per_task_type_delta(
            b_goal_ttloo_results, ttloo_adp_d,
            goal_state_cache=goal_state_cache, cases=cases,
        ))

        # ── Gate: Bootstrap paired CI vs Adaptive-D ───────────────────────
        # Collect per-case F1s (MLP B-goal vs Adaptive-D TTLOO)
        mlp_f1_by_case = []
        adp_f1_by_case = []
        for fold_data in b_goal_ttloo_results.values():
            adp_type_f1 = ttloo_adp_d.get(fold_data["cases"][0]["task_type"], 0.0) \
                if fold_data["cases"] else 0.0
            for r in fold_data["cases"]:
                mlp_f1_by_case.append(r["f1"])
                adp_f1_by_case.append(adp_type_f1)
        mean_delta, ci_lo, ci_hi = bootstrap_paired_ci(mlp_f1_by_case, adp_f1_by_case)
        gate_pass = mean_delta > 0 and ci_lo > 0
        print(f"\n{'─'*60}")
        print(f"GATE: MLP B-goal vs Adaptive-D TTLOO")
        print(f"  ΔF1 = {mean_delta:+.4f}  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  ΔF1>0: {'✓' if mean_delta > 0 else '✗'}  |  CI above 0: {'✓' if ci_lo > 0 else '✗'}")
        print(f"  → {'PASS — Gemini goal_state enters main path' if gate_pass else 'FAIL — Gemini goal_state does NOT beat Adaptive-D'}")
        print(f"{'─'*60}")
    else:
        print("\n[Stage B-goal skipped] — no goal_state_cache provided.")
        print("  Run: python3 phase3_goal_state_extractor.py  (needs GPU, ~2hr)")
        print("  Then: python3 phase3_learned_trigger.py --goal-state-cache results/phase3_learned/goal_state_cache.json")

    # ── Stage B-clip: + CLIP dynamics ─────────────────────────────────────
    stage_b_clip_f1 = None
    if clip_cache:
        print("\nExtracting Stage B-clip features (D + clip dynamics)...")
        cases, stage_bc_tag = build_dataset(
            cases, clip_cache=clip_cache, verbose=args.verbose
        )
        _, stage_b_clip_f1 = run_one_stage(
            cases, "B-clip", TORCH_AVAILABLE, args.cooldown, output_dir,
            baseline_results, ttloo_adp_d, args.no_vl_check, args.verbose
        )
    else:
        print("\n[Stage B-clip skipped] — no clip_cache provided.")

    # ── Stage B-full: + both ──────────────────────────────────────────────
    stage_b_full_f1 = None
    if goal_state_cache and clip_cache:
        print("\nExtracting Stage B-full features (D + goal_state + clip)...")
        cases, stage_bf_tag = build_dataset(
            cases,
            goal_state_cache=goal_state_cache,
            clip_cache=clip_cache,
            verbose=args.verbose,
        )
        _, stage_b_full_f1 = run_one_stage(
            cases, "B-full", TORCH_AVAILABLE, args.cooldown, output_dir,
            baseline_results, ttloo_adp_d, args.no_vl_check, args.verbose
        )

    # ── Summary ───────────────────────────────────────────────────────────
    adp_ttloo_f1 = np.mean(list(ttloo_adp_d.values()))
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY (TTLOO, out-of-sample)")
    print(f"{'='*60}")
    print(f"  Baseline (τ=0):           F1={np.mean(bl_f1s):.3f}")
    print(f"  D@2.8:                    F1={np.mean(d28_f1s):.3f}")
    print(f"  Adaptive-D TTLOO:         F1={adp_ttloo_f1:.3f}  ← fair target")
    print(f"  MLP Stage A (TTLOO):      F1={stage_a_f1:.3f}")
    if stage_b_goal_f1 is not None:
        delta_a = stage_b_goal_f1 - stage_a_f1
        delta_adp = stage_b_goal_f1 - adp_ttloo_f1
        print(f"  MLP Stage B-goal (TTLOO): F1={stage_b_goal_f1:.3f}  "
              f"(Δ={delta_a:+.3f} vs A, Δ={delta_adp:+.3f} vs AdpD)")
    if stage_b_clip_f1 is not None:
        delta = stage_b_clip_f1 - stage_a_f1
        print(f"  MLP Stage B-clip (TTLOO): F1={stage_b_clip_f1:.3f}  (Δ={delta:+.3f} vs A)")
    if stage_b_full_f1 is not None:
        delta_a = stage_b_full_f1 - stage_a_f1
        delta_adp = stage_b_full_f1 - adp_ttloo_f1
        print(f"  MLP Stage B-full (TTLOO): F1={stage_b_full_f1:.3f}  "
              f"(Δ={delta_a:+.3f} vs A, Δ={delta_adp:+.3f} vs AdpD)")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
