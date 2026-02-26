#!/usr/bin/env python3
"""
eval_trigger_v6.py - Evaluate v6 model with per-type adaptive thresholds

Strategy:
1. Find optimal per-type thresholds on TRAIN set (cross-validated)
2. Apply those thresholds on VAL set for final ESTP-F1
3. Also report global threshold baseline for comparison
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_trigger_v6 import TriggerClassifierV6, TYPE2IDX, TASK_TYPES

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
SAVE_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/checkpoints_v6"
RESULTS_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/results_v6"


def compute_temporal_features(img_feats, metadata):
    n = len(metadata)
    change_scores = np.zeros((n, 3), dtype=np.float32)
    type_ids = np.zeros(n, dtype=np.int64)

    case_start = 0
    current_key = (metadata[0]["video_uid"], metadata[0]["question"])

    for i in range(1, n + 1):
        if i < n:
            key = (metadata[i]["video_uid"], metadata[i]["question"])
        else:
            key = None
        if key != current_key or i == n:
            case_feats = img_feats[case_start:i]
            for j in range(i - case_start):
                global_idx = case_start + j
                type_ids[global_idx] = TYPE2IDX.get(metadata[global_idx]["task_type"], 0)
                current = case_feats[j]
                c_norm = np.linalg.norm(current)
                if c_norm > 1e-8:
                    for di, delta in enumerate([1, 3, 5]):
                        past_idx = j - delta
                        if past_idx >= 0:
                            past = case_feats[past_idx]
                            p_norm = np.linalg.norm(past)
                            if p_norm > 1e-8:
                                change_scores[global_idx, di] = 1.0 - np.dot(current, past) / (c_norm * p_norm)
            if i < n:
                case_start = i
                current_key = key

    return change_scores, type_ids


def get_predictions(model, img_feats, change_scores, txt_feats, type_ids, device, batch_size=1024):
    model.eval()
    all_probs = []
    n = len(img_feats)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            img_b = torch.from_numpy(img_feats[i:i+batch_size]).float().to(device)
            cs_b = torch.from_numpy(change_scores[i:i+batch_size]).float().to(device)
            txt_b = torch.from_numpy(txt_feats[i:i+batch_size]).float().to(device)
            tid_b = torch.from_numpy(type_ids[i:i+batch_size]).long().to(device)
            logits = model(img_b, cs_b, txt_b, tid_b)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())

    return np.array(all_probs)


def simulate_trigger(probs, timestamps, threshold, cooldown):
    trigger_times = []
    last = -float("inf")
    for p, t in zip(probs, timestamps):
        if t - last < cooldown:
            continue
        if p >= threshold:
            trigger_times.append(t)
            last = t
    return trigger_times


def compute_estp_f1(pred_times, gt_windows):
    if not gt_windows:
        if not pred_times:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 0, "fp": 0, "fn": 0}
        return {"f1": 0.0, "precision": 0.0, "recall": 1.0, "tp": 0, "fp": len(pred_times), "fn": 0}

    hits = []
    for gi, (gs, ge) in enumerate(gt_windows):
        for pt in pred_times:
            if gs - 1.0 <= pt <= ge + 2.0:
                hits.append(gi)
                break

    fp = sum(1 for pt in pred_times
             if not any(gs - 1.0 <= pt <= ge + 2.0 for gs, ge in gt_windows))

    tp = len(hits)
    fn = len(gt_windows) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def find_optimal_thresholds_on_train(model, device):
    """Find per-type optimal thresholds on training set."""
    print("Finding optimal thresholds on training set...")

    img_feats = np.load(os.path.join(DATA_DIR, "train_img_feats.npy"))
    txt_feats = np.load(os.path.join(DATA_DIR, "train_txt_feats.npy"))
    with open(os.path.join(DATA_DIR, "train_meta.json")) as f:
        metadata = json.load(f)

    change_scores, type_ids = compute_temporal_features(img_feats, metadata)
    probs = get_predictions(model, img_feats, change_scores, txt_feats, type_ids, device)

    # Group by case
    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    # Group cases by type
    type_cases = defaultdict(list)
    for (vid, q), indices in cases.items():
        tt = metadata[indices[0]]["task_type"]
        type_cases[tt].append((vid, q, indices))

    # Find optimal threshold + cooldown per type
    optimal_thresholds = {}

    for tt in TASK_TYPES:
        if tt not in type_cases:
            optimal_thresholds[tt] = {"threshold": 0.5, "cooldown": 12}
            continue

        best_f1 = 0
        best_th = 0.5
        best_cd = 12

        for th in np.arange(0.05, 1.0, 0.05):
            for cd in [3, 6, 9, 12]:
                f1s = []
                for vid, q, indices in type_cases[tt]:
                    case_probs = probs[indices]
                    case_ts = [metadata[i]["timestamp"] for i in indices]
                    case_gt = metadata[indices[0]]["gt_windows"]
                    sorted_pairs = sorted(zip(case_ts, case_probs))
                    sorted_ts = [p[0] for p in sorted_pairs]
                    sorted_probs = [p[1] for p in sorted_pairs]
                    trigger_times = simulate_trigger(sorted_probs, sorted_ts, th, cd)
                    m = compute_estp_f1(trigger_times, case_gt)
                    f1s.append(m["f1"])
                avg = np.mean(f1s)
                if avg > best_f1:
                    best_f1 = avg
                    best_th = th
                    best_cd = cd

        optimal_thresholds[tt] = {
            "threshold": round(float(best_th), 3),
            "cooldown": best_cd,
            "train_f1": round(best_f1, 4),
            "n_cases": len(type_cases[tt]),
        }
        print(f"  {tt:40s}  th={best_th:.2f}  cd={best_cd}  train_F1={best_f1:.3f}  (n={len(type_cases[tt])})")

    # Also find global optimal
    best_global_f1 = 0
    best_global_th = 0.5
    best_global_cd = 12
    for th in np.arange(0.1, 1.0, 0.05):
        for cd in [0, 3, 6, 9, 12]:
            f1s = []
            for (vid, q), indices in cases.items():
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]
                sorted_pairs = sorted(zip(case_ts, case_probs))
                sorted_ts = [p[0] for p in sorted_pairs]
                sorted_probs = [p[1] for p in sorted_pairs]
                trigger_times = simulate_trigger(sorted_probs, sorted_ts, th, cd)
                m = compute_estp_f1(trigger_times, case_gt)
                f1s.append(m["f1"])
            avg = np.mean(f1s)
            if avg > best_global_f1:
                best_global_f1 = avg
                best_global_th = th
                best_global_cd = cd

    global_config = {
        "threshold": round(float(best_global_th), 3),
        "cooldown": best_global_cd,
        "train_f1": round(best_global_f1, 4),
    }
    print(f"\n  Global optimal: th={best_global_th:.2f}, cd={best_global_cd}, train_F1={best_global_f1:.3f}")

    return optimal_thresholds, global_config


def evaluate_on_val(model, device, optimal_thresholds, global_config):
    """Evaluate on val set using both global and per-type thresholds."""
    print("\nEvaluating on validation set...")

    img_feats = np.load(os.path.join(DATA_DIR, "val_img_feats.npy"))
    txt_feats = np.load(os.path.join(DATA_DIR, "val_txt_feats.npy"))
    with open(os.path.join(DATA_DIR, "val_meta.json")) as f:
        metadata = json.load(f)

    change_scores, type_ids = compute_temporal_features(img_feats, metadata)
    probs = get_predictions(model, img_feats, change_scores, txt_feats, type_ids, device)

    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    # 1. Global threshold evaluation
    global_th = global_config["threshold"]
    global_cd = global_config["cooldown"]
    global_f1s = []
    global_type_metrics = defaultdict(list)

    for (vid, q), indices in cases.items():
        case_probs = probs[indices]
        case_ts = [metadata[i]["timestamp"] for i in indices]
        case_gt = metadata[indices[0]]["gt_windows"]
        task_type = metadata[indices[0]]["task_type"]

        sorted_pairs = sorted(zip(case_ts, case_probs))
        sorted_ts = [p[0] for p in sorted_pairs]
        sorted_probs = [p[1] for p in sorted_pairs]

        trigger_times = simulate_trigger(sorted_probs, sorted_ts, global_th, global_cd)
        m = compute_estp_f1(trigger_times, case_gt)
        global_f1s.append(m["f1"])
        global_type_metrics[task_type].append(m)

    global_estp_f1 = np.mean(global_f1s)

    # 2. Per-type adaptive threshold evaluation
    adaptive_f1s = []
    adaptive_type_metrics = defaultdict(list)

    for (vid, q), indices in cases.items():
        case_probs = probs[indices]
        case_ts = [metadata[i]["timestamp"] for i in indices]
        case_gt = metadata[indices[0]]["gt_windows"]
        task_type = metadata[indices[0]]["task_type"]

        tt_config = optimal_thresholds.get(task_type, {"threshold": 0.5, "cooldown": 12})
        th = tt_config["threshold"]
        cd = tt_config["cooldown"]

        sorted_pairs = sorted(zip(case_ts, case_probs))
        sorted_ts = [p[0] for p in sorted_pairs]
        sorted_probs = [p[1] for p in sorted_pairs]

        trigger_times = simulate_trigger(sorted_probs, sorted_ts, th, cd)
        m = compute_estp_f1(trigger_times, case_gt)
        adaptive_f1s.append(m["f1"])
        adaptive_type_metrics[task_type].append(m)

    adaptive_estp_f1 = np.mean(adaptive_f1s)

    # 3. Also try full val-set sweep (oracle) for reference
    best_val_f1 = 0
    best_val_config = {}
    for threshold in np.arange(0.1, 1.0, 0.05):
        for cooldown in [0, 3, 6, 9, 12]:
            f1s = []
            for (vid, q), indices in cases.items():
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]
                sorted_pairs = sorted(zip(case_ts, case_probs))
                sorted_ts = [p[0] for p in sorted_pairs]
                sorted_probs = [p[1] for p in sorted_pairs]
                trigger_times = simulate_trigger(sorted_probs, sorted_ts, threshold, cooldown)
                m = compute_estp_f1(trigger_times, case_gt)
                f1s.append(m["f1"])
            avg = np.mean(f1s)
            if avg > best_val_f1:
                best_val_f1 = avg
                best_val_config = {"threshold": round(float(threshold), 3), "cooldown": cooldown}

    return {
        "global": {
            "f1": round(global_estp_f1, 4),
            "config": global_config,
            "per_type": {tt: {
                "f1": round(np.mean([m["f1"] for m in ms]), 4),
                "precision": round(np.mean([m["precision"] for m in ms]), 4),
                "recall": round(np.mean([m["recall"] for m in ms]), 4),
                "n_cases": len(ms),
            } for tt, ms in sorted(global_type_metrics.items())},
        },
        "adaptive": {
            "f1": round(adaptive_estp_f1, 4),
            "thresholds": optimal_thresholds,
            "per_type": {tt: {
                "f1": round(np.mean([m["f1"] for m in ms]), 4),
                "precision": round(np.mean([m["precision"] for m in ms]), 4),
                "recall": round(np.mean([m["recall"] for m in ms]), 4),
                "n_cases": len(ms),
            } for tt, ms in sorted(adaptive_type_metrics.items())},
        },
        "oracle_val": {
            "f1": round(best_val_f1, 4),
            "config": best_val_config,
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TriggerClassifierV6()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {ckpt_path}, epoch={checkpoint.get('epoch', '?')}")

    # Step 1: Find optimal thresholds on train
    optimal_thresholds, global_config = find_optimal_thresholds_on_train(model, device)

    # Step 2: Evaluate on val
    results = evaluate_on_val(model, device, optimal_thresholds, global_config)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Global threshold (train-tuned):   ESTP-F1 = {results['global']['f1']:.4f}")
    print(f"    Config: th={results['global']['config']['threshold']}, cd={results['global']['config']['cooldown']}")
    print(f"  Adaptive per-type (train-tuned):  ESTP-F1 = {results['adaptive']['f1']:.4f}")
    print(f"  Oracle val sweep:                 ESTP-F1 = {results['oracle_val']['f1']:.4f}")
    print(f"    Config: th={results['oracle_val']['config']['threshold']}, cd={results['oracle_val']['config']['cooldown']}")

    print(f"\n--- Comparison ---")
    print(f"  Simple Baseline F1:    0.141")
    print(f"  Adaptive-D F1:         0.222")
    print(f"  V1 (global):           0.304")
    print(f"  V6 (global):           {results['global']['f1']:.3f}")
    print(f"  V6 (adaptive):         {results['adaptive']['f1']:.3f}")

    print(f"\n--- Per Type (adaptive) ---")
    for tt, m in sorted(results["adaptive"]["per_type"].items()):
        th_info = optimal_thresholds.get(tt, {})
        th = th_info.get("threshold", "?")
        cd = th_info.get("cooldown", "?")
        print(f"  {tt:40s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  (n={m['n_cases']}, th={th}, cd={cd})")

    print(f"{'='*60}")

    with open(os.path.join(RESULTS_DIR, "eval_report.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
