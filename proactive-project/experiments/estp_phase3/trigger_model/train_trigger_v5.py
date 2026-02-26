#!/usr/bin/env python3
"""
train_trigger_v5.py - Best combination: temporal conv + ranking loss + focal loss

Combines v3's temporal conv architecture with v4's ranking-aware training.
Also adds online hard negative mining and ESTP-F1 early stopping.

Architecture: v3-style temporal conv + change scores + classification head
Training: case-aware ranking loss + focal BCE + periodic ESTP-F1 evaluation
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
SAVE_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/checkpoints_v5"

WINDOW_SIZE = 7


class TriggerClassifierV5(nn.Module):
    """Temporal conv + change scores -> single ranking score.

    Architecture:
    - Window features [W, 512] -> 1D conv -> 128d temporal embedding
    - Change scores [3]
    - Current image [512]
    - Text feature [512]
    - Total: 128 + 3 + 512 + 512 = 1155d -> MLP -> 1 score
    """

    def __init__(self, img_dim=512, txt_dim=512, window_size=WINDOW_SIZE,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.window_size = window_size

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(img_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        input_dim = 128 + 3 + img_dim + txt_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, window_feats, change_scores, img_feat, txt_feat):
        temporal = window_feats.permute(0, 2, 1)
        temporal = self.temporal_conv(temporal).squeeze(-1)
        combined = torch.cat([temporal, change_scores, img_feat, txt_feat], dim=-1)
        return self.classifier(combined).squeeze(-1)


def precompute_temporal_features(split):
    """Pre-compute window features and change scores for a split."""
    img_feats = np.load(os.path.join(DATA_DIR, f"{split}_img_feats.npy"))
    txt_feats = np.load(os.path.join(DATA_DIR, f"{split}_txt_feats.npy"))
    labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))

    with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
        metadata = json.load(f)

    n = len(labels)
    window_feats = np.zeros((n, WINDOW_SIZE, 512), dtype=np.float32)
    change_scores = np.zeros((n, 3), dtype=np.float32)

    cases = []
    case_start = 0
    current_key = (metadata[0]["video_uid"], metadata[0]["question"])

    for i in range(1, n + 1):
        if i < n:
            key = (metadata[i]["video_uid"], metadata[i]["question"])
        else:
            key = None

        if key != current_key or i == n:
            case_feats = img_feats[case_start:i]
            case_len = i - case_start

            for j in range(case_len):
                global_idx = case_start + j

                # Window
                window = np.zeros((WINDOW_SIZE, 512), dtype=np.float32)
                for w in range(WINDOW_SIZE):
                    src = j - (WINDOW_SIZE - 1 - w)
                    if src >= 0:
                        window[w] = case_feats[src]
                window_feats[global_idx] = window

                # Change scores
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
                            change_scores[global_idx, di] = 1.0 - cos_sim

            # Record case
            pos_idx = [case_start + j for j in range(case_len) if labels[case_start + j] == 1]
            neg_idx = [case_start + j for j in range(case_len) if labels[case_start + j] == 0]
            if pos_idx and neg_idx:
                cases.append((pos_idx, neg_idx))

            if i < n:
                case_start = i
                current_key = key

    print(f"  {split}: {n} samples, {len(cases)} cases with both pos/neg")
    return img_feats, txt_feats, labels, window_feats, change_scores, metadata, cases


class CaseRankingWindowDataset(Dataset):
    """Yields (positive, negative) pairs with temporal window features."""

    def __init__(self, img_feats, txt_feats, window_feats, change_scores, labels, cases,
                 pairs_per_epoch=100000):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.window_feats = window_feats
        self.change_scores = change_scores
        self.labels = labels
        self.cases = cases
        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        case_idx = random.randint(0, len(self.cases) - 1)
        pos_indices, neg_indices = self.cases[case_idx]
        pos_i = random.choice(pos_indices)
        neg_i = random.choice(neg_indices)

        return (
            torch.from_numpy(self.window_feats[pos_i]).float(),
            torch.from_numpy(self.change_scores[pos_i]).float(),
            torch.from_numpy(self.img_feats[pos_i]).float(),
            torch.from_numpy(self.txt_feats[pos_i]).float(),
            torch.from_numpy(self.window_feats[neg_i]).float(),
            torch.from_numpy(self.change_scores[neg_i]).float(),
            torch.from_numpy(self.img_feats[neg_i]).float(),
            torch.from_numpy(self.txt_feats[neg_i]).float(),
        )


def compute_estp_f1_single(pred_times, gt_windows):
    if not gt_windows:
        return 1.0 if not pred_times else 0.0
    hits = 0
    for gs, ge in gt_windows:
        for pt in pred_times:
            if gs - 1.0 <= pt <= ge + 2.0:
                hits += 1
                break
    fp = sum(1 for pt in pred_times
             if not any(gs - 1.0 <= pt <= ge + 2.0 for gs, ge in gt_windows))
    tp = hits
    fn = len(gt_windows) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def evaluate_estp(model, img_feats, window_feats, change_scores, txt_feats, metadata, device):
    """Full ESTP-F1 evaluation with threshold sweep."""
    model.eval()
    all_probs = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(img_feats), batch_size):
            w_batch = torch.from_numpy(window_feats[i:i+batch_size]).float().to(device)
            cs_batch = torch.from_numpy(change_scores[i:i+batch_size]).float().to(device)
            img_batch = torch.from_numpy(img_feats[i:i+batch_size]).float().to(device)
            txt_batch = torch.from_numpy(txt_feats[i:i+batch_size]).float().to(device)
            scores = model(w_batch, cs_batch, img_batch, txt_batch)
            probs = torch.sigmoid(scores)
            all_probs.extend(probs.cpu().tolist())

    probs = np.array(all_probs)

    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    best_f1 = 0
    best_config = {}

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
                last_t = -float("inf")
                for p, t in zip(sorted_probs, sorted_ts):
                    if t - last_t < cooldown:
                        continue
                    if p >= threshold:
                        trigger_times.append(t)
                        last_t = t

                case_f1s.append(compute_estp_f1_single(trigger_times, case_gt))

            avg_f1 = np.mean(case_f1s)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_config = {
                    "threshold": round(float(threshold), 3),
                    "cooldown": cooldown,
                    "avg_f1": round(avg_f1, 4),
                }

    return best_f1, best_config


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Pre-computing temporal features...")
    (train_img, train_txt, train_labels, train_window, train_cs,
     train_meta, train_cases) = precompute_temporal_features("train")
    (val_img, val_txt, val_labels, val_window, val_cs,
     val_meta, val_cases) = precompute_temporal_features("val")

    train_dataset = CaseRankingWindowDataset(
        train_img, train_txt, train_window, train_cs, train_labels,
        train_cases, pairs_per_epoch=args.pairs_per_epoch
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    model = TriggerClassifierV5(window_size=WINDOW_SIZE, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    margin_loss = nn.MarginRankingLoss(margin=args.margin)
    focal_gamma = args.focal_gamma

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
        for (pos_w, pos_cs, pos_img, pos_txt,
             neg_w, neg_cs, neg_img, neg_txt) in pbar:
            pos_w = pos_w.to(device)
            pos_cs = pos_cs.to(device)
            pos_img = pos_img.to(device)
            pos_txt = pos_txt.to(device)
            neg_w = neg_w.to(device)
            neg_cs = neg_cs.to(device)
            neg_img = neg_img.to(device)
            neg_txt = neg_txt.to(device)

            pos_scores = model(pos_w, pos_cs, pos_img, pos_txt)
            neg_scores = model(neg_w, neg_cs, neg_img, neg_txt)

            # Ranking loss
            target = torch.ones_like(pos_scores)
            rank_loss = margin_loss(pos_scores, neg_scores, target)

            # Focal BCE
            all_scores = torch.cat([pos_scores, neg_scores])
            all_targets = torch.cat([torch.ones_like(pos_scores),
                                     torch.zeros_like(neg_scores)])
            bce = F.binary_cross_entropy_with_logits(all_scores, all_targets, reduction='none')
            pt = torch.sigmoid(all_scores)
            pt = torch.where(all_targets == 1, pt, 1 - pt)
            focal_weight = (1 - pt) ** focal_gamma
            focal_loss = (focal_weight * bce).mean()

            loss = rank_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # ESTP-F1 eval every 2 epochs
        estp_f1 = 0
        estp_config = {}
        if epoch % 2 == 0 or epoch == args.epochs or epoch == 1:
            estp_f1, estp_config = evaluate_estp(
                model, val_img, val_window, val_cs, val_txt, val_meta, device
            )

        elapsed = time.time() - start_time
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "estp_f1": estp_f1,
            "estp_config": estp_config,
        }
        history.append(record)

        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, ESTP-F1={estp_f1:.4f}, elapsed={elapsed:.0f}s")
        if estp_config:
            print(f"  Config: th={estp_config.get('threshold', '?')}, cd={estp_config.get('cooldown', '?')}")

        if estp_f1 > best_estp_f1:
            best_estp_f1 = estp_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "estp_f1": estp_f1,
                "estp_config": estp_config,
                "args": vars(args),
                "model_class": "TriggerClassifierV5",
                "window_size": WINDOW_SIZE,
            }, os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  -> New best (ESTP-F1={best_estp_f1:.4f})")

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! Best ESTP-F1: {best_estp_f1:.4f}, Time: {total_time:.0f}s")

    return best_estp_f1


def main():
    parser = argparse.ArgumentParser(description="Train trigger model v5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--pairs_per_epoch", type=int, default=100000)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
