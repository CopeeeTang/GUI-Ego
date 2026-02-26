#!/usr/bin/env python3
"""
train_trigger_v6.py - Task-type conditioned CLIP+MLP

Key insight: Different task types have very different optimal thresholds (0.05-0.95),
indicating the model needs type-specific decision boundaries.

Changes:
1. Task type embedding (12 types -> 32d) concatenated with features
2. Type-specific classification heads via FiLM conditioning
3. Change score features for temporal context
4. Weighted BCE + per-type adaptive threshold at eval time
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
SAVE_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/checkpoints_v6"

TASK_TYPES = [
    "Action Reasoning",
    "Action Recognition",
    "Attribute Perception",
    "Ego Object Localization",
    "Ego Object State Change Recognition",
    "Information Function",
    "Object Function",
    "Object Localization",
    "Object Recognition",
    "Object State Change Recognition",
    "Task Understanding",
    "Text-Rich Understanding",
]
TYPE2IDX = {t: i for i, t in enumerate(TASK_TYPES)}


class TriggerClassifierV6(nn.Module):
    """Task-type conditioned MLP classifier.

    Uses FiLM (Feature-wise Linear Modulation) to condition the hidden
    representation on task type.

    Input: img_feat(512) + change_scores(3) + txt_feat(512) = 1027d
    Conditioning: task_type_id -> embedding(32) -> FiLM parameters
    """

    def __init__(self, img_dim=512, txt_dim=512, n_types=12,
                 type_embed_dim=32, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.type_embedding = nn.Embedding(n_types, type_embed_dim)

        input_dim = img_dim + 3 + txt_dim  # 1027

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM: type embedding -> scale and shift for hidden features
        self.film_gamma = nn.Linear(type_embed_dim, hidden_dim)
        self.film_beta = nn.Linear(type_embed_dim, hidden_dim)

        # Classifier after FiLM modulation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 2),
        )

    def forward(self, img_feat, change_scores, txt_feat, type_ids):
        combined = torch.cat([img_feat, change_scores, txt_feat], dim=-1)
        h = self.encoder(combined)

        # FiLM conditioning
        type_emb = self.type_embedding(type_ids)
        gamma = self.film_gamma(type_emb)
        beta = self.film_beta(type_emb)
        h = gamma * h + beta

        return self.classifier(h)


class TypeConditionedDataset(Dataset):
    """Dataset with task type IDs and change scores."""

    def __init__(self, split="train"):
        self.img_feats = np.load(os.path.join(DATA_DIR, f"{split}_img_feats.npy"))
        self.txt_feats = np.load(os.path.join(DATA_DIR, f"{split}_txt_feats.npy"))
        self.labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))

        with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
            metadata = json.load(f)

        n = len(self.labels)
        self.change_scores = np.zeros((n, 3), dtype=np.float32)
        self.type_ids = np.zeros(n, dtype=np.int64)

        # Set type IDs
        for i, m in enumerate(metadata):
            tt = m["task_type"]
            self.type_ids[i] = TYPE2IDX.get(tt, 0)

        # Compute change scores
        case_start = 0
        current_key = (metadata[0]["video_uid"], metadata[0]["question"])

        for i in range(1, n + 1):
            if i < n:
                key = (metadata[i]["video_uid"], metadata[i]["question"])
            else:
                key = None

            if key != current_key or i == n:
                case_feats = self.img_feats[case_start:i]
                for j in range(i - case_start):
                    global_idx = case_start + j
                    current = case_feats[j]
                    c_norm = np.linalg.norm(current)
                    if c_norm < 1e-8:
                        continue
                    for di, delta in enumerate([1, 3, 5]):
                        past_idx = j - delta
                        if past_idx >= 0:
                            past = case_feats[past_idx]
                            p_norm = np.linalg.norm(past)
                            if p_norm > 1e-8:
                                cos_sim = np.dot(current, past) / (c_norm * p_norm)
                                self.change_scores[global_idx, di] = 1.0 - cos_sim

                if i < n:
                    case_start = i
                    current_key = key

        print(f"Loaded {split}: {n} samples, "
              f"pos={self.labels.sum()}, neg={n-self.labels.sum()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.img_feats[idx]).float(),
            torch.from_numpy(self.change_scores[idx]).float(),
            torch.from_numpy(self.txt_feats[idx]).float(),
            torch.tensor(self.type_ids[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def get_class_weights(labels):
    n_total = len(labels)
    n_pos = labels.sum()
    n_neg = n_total - n_pos
    w_neg = n_total / (2 * n_neg)
    w_pos = n_total / (2 * n_pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for img_feat, cs, txt_feat, type_ids, labels in loader:
            img_feat = img_feat.to(device)
            cs = cs.to(device)
            txt_feat = txt_feat.to(device)
            type_ids = type_ids.to(device)
            labels = labels.to(device)

            logits = model(img_feat, cs, txt_feat, type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= threshold).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / max(n_batches, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate_estp(model, device, split="val"):
    """Full ESTP-F1 evaluation with per-type adaptive thresholds."""
    from collections import defaultdict

    img_feats = np.load(os.path.join(DATA_DIR, f"{split}_img_feats.npy"))
    txt_feats = np.load(os.path.join(DATA_DIR, f"{split}_txt_feats.npy"))
    labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))
    with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
        metadata = json.load(f)

    n = len(labels)
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

    # Get probabilities
    model.eval()
    all_probs = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, n, batch_size):
            img_b = torch.from_numpy(img_feats[i:i+batch_size]).float().to(device)
            cs_b = torch.from_numpy(change_scores[i:i+batch_size]).float().to(device)
            txt_b = torch.from_numpy(txt_feats[i:i+batch_size]).float().to(device)
            tid_b = torch.from_numpy(type_ids[i:i+batch_size]).long().to(device)
            logits = model(img_b, cs_b, txt_b, tid_b)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
    probs = np.array(all_probs)

    # Group by case
    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    # Global threshold sweep
    best_global_f1 = 0
    best_global_config = {}

    for threshold in np.arange(0.1, 1.0, 0.05):
        for cooldown in [0, 3, 6, 9, 12]:
            case_f1s = []
            for (vid, q), indices in cases.items():
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]

                sorted_pairs = sorted(zip(case_ts, case_probs))
                sorted_ts = [p[0] for p in sorted_pairs]
                sorted_probs = [p[1] for p in sorted_pairs]

                trigger_times = []
                last = -float("inf")
                for p, t in zip(sorted_probs, sorted_ts):
                    if t - last < cooldown:
                        continue
                    if p >= threshold:
                        trigger_times.append(t)
                        last = t

                if not case_gt:
                    case_f1s.append(1.0 if not trigger_times else 0.0)
                    continue

                hits = sum(1 for gs, ge in case_gt
                           if any(gs-1.0 <= pt <= ge+2.0 for pt in trigger_times))
                fp = sum(1 for pt in trigger_times
                         if not any(gs-1.0 <= pt <= ge+2.0 for gs, ge in case_gt))
                tp = hits
                fn = len(case_gt) - tp
                p = tp/(tp+fp) if (tp+fp) > 0 else 0
                r = tp/(tp+fn) if (tp+fn) > 0 else 0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                case_f1s.append(f1)

            avg_f1 = np.mean(case_f1s)
            if avg_f1 > best_global_f1:
                best_global_f1 = avg_f1
                best_global_config = {
                    "threshold": round(float(threshold), 3),
                    "cooldown": cooldown,
                    "avg_f1": round(avg_f1, 4),
                }

    # Per-type adaptive threshold sweep
    type_cases = defaultdict(list)
    for (vid, q), indices in cases.items():
        tt = metadata[indices[0]]["task_type"]
        type_cases[tt].append((vid, q, indices))

    per_type_f1s = []
    for tt in TASK_TYPES:
        if tt not in type_cases:
            continue
        best_tt_f1 = 0
        for th in np.arange(0.05, 1.0, 0.05):
            f1s = []
            for vid, q, indices in type_cases[tt]:
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]
                sorted_pairs = sorted(zip(case_ts, case_probs))
                sorted_ts = [p[0] for p in sorted_pairs]
                sorted_probs = [p[1] for p in sorted_pairs]
                trigger_times = []
                last = -float("inf")
                for p, t in zip(sorted_probs, sorted_ts):
                    if t - last < 12: continue
                    if p >= th:
                        trigger_times.append(t)
                        last = t
                if not case_gt:
                    f1s.append(1.0 if not trigger_times else 0.0)
                    continue
                hits = sum(1 for gs, ge in case_gt if any(gs-1.0 <= pt <= ge+2.0 for pt in trigger_times))
                fp = sum(1 for pt in trigger_times if not any(gs-1.0 <= pt <= ge+2.0 for gs, ge in case_gt))
                tp = hits
                fn = len(case_gt) - tp
                p = tp/(tp+fp) if (tp+fp) > 0 else 0
                r = tp/(tp+fn) if (tp+fn) > 0 else 0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                f1s.append(f1)
            avg = np.mean(f1s)
            if avg > best_tt_f1:
                best_tt_f1 = avg
        for vid, q, indices in type_cases[tt]:
            per_type_f1s.append(best_tt_f1)

    adaptive_f1 = np.mean(per_type_f1s) if per_type_f1s else 0

    return best_global_f1, best_global_config, adaptive_f1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = TypeConditionedDataset("train")
    val_dataset = TypeConditionedDataset("val")

    labels = train_dataset.labels
    class_counts = np.bincount(labels.astype(int))
    sample_weights = np.where(labels == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = TriggerClassifierV6(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    class_weights = get_class_weights(labels)
    print(f"Class weights: neg={class_weights[0]:.3f}, pos={class_weights[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_estp_f1 = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img_feat, cs, txt_feat, type_ids, labels_batch in pbar:
            img_feat = img_feat.to(device)
            cs = cs.to(device)
            txt_feat = txt_feat.to(device)
            type_ids = type_ids.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(img_feat, cs, txt_feat, type_ids)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        # ESTP-F1 every 3 epochs
        estp_f1 = 0
        estp_config = {}
        adaptive_f1 = 0
        if epoch % 3 == 0 or epoch == args.epochs or epoch == 1:
            estp_f1, estp_config, adaptive_f1 = evaluate_estp(model, device)

        elapsed = time.time() - start_time
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_f1": val_metrics["f1"],
            "estp_f1": estp_f1,
            "adaptive_estp_f1": adaptive_f1,
            "estp_config": estp_config,
        }
        history.append(record)

        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, "
              f"val_F1={val_metrics['f1']:.3f}, "
              f"ESTP-F1={estp_f1:.4f}, "
              f"Adaptive-ESTP-F1={adaptive_f1:.4f}, "
              f"elapsed={elapsed:.0f}s")
        if estp_config:
            print(f"  Config: th={estp_config.get('threshold', '?')}, cd={estp_config.get('cooldown', '?')}")

        if estp_f1 > best_estp_f1:
            best_estp_f1 = estp_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "estp_f1": estp_f1,
                "estp_config": estp_config,
                "adaptive_f1": adaptive_f1,
                "args": vars(args),
                "model_class": "TriggerClassifierV6",
                "type2idx": TYPE2IDX,
            }, os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  -> New best (ESTP-F1={best_estp_f1:.4f})")

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! Best ESTP-F1: {best_estp_f1:.4f}, Time: {total_time:.0f}s")

    return best_estp_f1


def main():
    parser = argparse.ArgumentParser(description="Train trigger model v6")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
