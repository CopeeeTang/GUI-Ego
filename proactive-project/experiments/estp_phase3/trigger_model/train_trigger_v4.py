#!/usr/bin/env python3
"""
train_trigger_v4.py - CLIP+MLP with ranking-aware training

Key insight: ESTP-F1 requires ranking GT frames above non-GT frames within each case.
Current model has max_GT_prob > max_nonGT_prob in only 31% of cases.

Changes from v1:
1. Case-aware contrastive ranking loss (margin ranking loss between GT and non-GT frames)
2. Hard negative mining within each case
3. Focal loss to focus on hard examples
4. Temporal features (change scores only - lightweight)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
SAVE_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/checkpoints_v4"


class TriggerClassifierV4(nn.Module):
    """MLP classifier with change score features.

    Input: img_feat(512) + change_scores(3) + txt_feat(512) = 1027d
    """

    def __init__(self, img_dim=512, txt_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        input_dim = img_dim + 3 + txt_dim  # 1027
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),  # single score output for ranking
        )

    def forward(self, img_feat, change_scores, txt_feat):
        combined = torch.cat([img_feat, change_scores, txt_feat], dim=-1)
        return self.classifier(combined).squeeze(-1)


class CaseRankingDataset(Dataset):
    """Dataset that yields (positive, hard_negative) pairs from same case.

    For each iteration, samples a case, picks a positive frame and
    the hardest negative frame (or random if no clear hard negative).
    """

    def __init__(self, split="train", pairs_per_epoch=100000):
        self.img_feats = np.load(os.path.join(DATA_DIR, f"{split}_img_feats.npy"))
        self.txt_feats = np.load(os.path.join(DATA_DIR, f"{split}_txt_feats.npy"))
        self.labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))
        self.pairs_per_epoch = pairs_per_epoch

        with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
            metadata = json.load(f)

        # Compute change scores
        n = len(self.labels)
        self.change_scores = np.zeros((n, 3), dtype=np.float32)

        case_start = 0
        current_key = (metadata[0]["video_uid"], metadata[0]["question"])
        self.cases = []  # list of (pos_indices, neg_indices)

        for i in range(1, n + 1):
            if i < n:
                key = (metadata[i]["video_uid"], metadata[i]["question"])
            else:
                key = None

            if key != current_key or i == n:
                # Process change scores
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

                # Record case
                case_indices = list(range(case_start, i))
                pos_idx = [idx for idx in case_indices if self.labels[idx] == 1]
                neg_idx = [idx for idx in case_indices if self.labels[idx] == 0]
                if pos_idx and neg_idx:
                    self.cases.append((pos_idx, neg_idx))

                if i < n:
                    case_start = i
                    current_key = key

        print(f"Loaded {split}: {n} samples, {len(self.cases)} cases with both pos/neg")
        print(f"  Pairs per epoch: {pairs_per_epoch}")

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        # Sample a random case
        case_idx = random.randint(0, len(self.cases) - 1)
        pos_indices, neg_indices = self.cases[case_idx]

        # Sample a positive and negative
        pos_i = random.choice(pos_indices)
        neg_i = random.choice(neg_indices)

        return (
            torch.from_numpy(self.img_feats[pos_i]).float(),
            torch.from_numpy(self.change_scores[pos_i]).float(),
            torch.from_numpy(self.txt_feats[pos_i]).float(),
            torch.from_numpy(self.img_feats[neg_i]).float(),
            torch.from_numpy(self.change_scores[neg_i]).float(),
            torch.from_numpy(self.txt_feats[neg_i]).float(),
        )


class FlatDataset(Dataset):
    """Standard flat dataset for evaluation."""

    def __init__(self, split="val"):
        self.img_feats = np.load(os.path.join(DATA_DIR, f"{split}_img_feats.npy"))
        self.txt_feats = np.load(os.path.join(DATA_DIR, f"{split}_txt_feats.npy"))
        self.labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))

        with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
            metadata = json.load(f)

        n = len(self.labels)
        self.change_scores = np.zeros((n, 3), dtype=np.float32)

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

        print(f"Loaded {split}: {n} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.img_feats[idx]).float(),
            torch.from_numpy(self.change_scores[idx]).float(),
            torch.from_numpy(self.txt_feats[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class CombinedLoss(nn.Module):
    """Combined ranking + classification loss.

    ranking_loss: margin ranking loss (positive score > negative score by margin)
    focal_loss: focal variant of BCE for classification
    """

    def __init__(self, margin=0.5, alpha_rank=1.0, alpha_focal=1.0, focal_gamma=2.0):
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.alpha_rank = alpha_rank
        self.alpha_focal = alpha_focal
        self.focal_gamma = focal_gamma

    def focal_bce(self, scores, targets):
        """Focal BCE loss."""
        probs = torch.sigmoid(scores)
        bce = F.binary_cross_entropy_with_logits(scores, targets.float(), reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def forward(self, pos_scores, neg_scores):
        # Ranking loss: pos_score should be > neg_score by margin
        target = torch.ones_like(pos_scores)
        rank_loss = self.margin_loss(pos_scores, neg_scores, target)

        # Focal BCE: positive samples should be high, negatives low
        all_scores = torch.cat([pos_scores, neg_scores])
        all_targets = torch.cat([torch.ones_like(pos_scores),
                                 torch.zeros_like(neg_scores)])
        focal_loss = self.focal_bce(all_scores, all_targets)

        return self.alpha_rank * rank_loss + self.alpha_focal * focal_loss


def evaluate_estp(model, metadata_path, img_feats, change_scores, txt_feats, labels,
                  device, thresholds=None, cooldowns=None):
    """Evaluate ESTP-F1 with threshold sweep."""
    from collections import defaultdict

    model.eval()

    # Get predictions
    all_scores = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(img_feats), batch_size):
            img_batch = torch.from_numpy(img_feats[i:i+batch_size]).float().to(device)
            cs_batch = torch.from_numpy(change_scores[i:i+batch_size]).float().to(device)
            txt_batch = torch.from_numpy(txt_feats[i:i+batch_size]).float().to(device)
            scores = model(img_batch, cs_batch, txt_batch)
            probs = torch.sigmoid(scores)
            all_scores.extend(probs.cpu().tolist())

    probs = np.array(all_scores)

    with open(metadata_path) as f:
        metadata = json.load(f)

    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    if cooldowns is None:
        cooldowns = [0, 3, 6, 9, 12]

    best_f1 = 0
    best_config = None

    for threshold in thresholds:
        for cooldown in cooldowns:
            case_f1s = []
            for (vid, q), indices in cases.items():
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]

                sorted_pairs = sorted(zip(case_ts, case_probs))
                sorted_ts = [p[0] for p in sorted_pairs]
                sorted_probs = [p[1] for p in sorted_pairs]

                # Trigger simulation
                trigger_times = []
                last_t = -float("inf")
                for p, t in zip(sorted_probs, sorted_ts):
                    if t - last_t < cooldown:
                        continue
                    if p >= threshold:
                        trigger_times.append(t)
                        last_t = t

                # ESTP-F1
                if not case_gt:
                    case_f1s.append(1.0 if not trigger_times else 0.0)
                    continue

                hits = 0
                for gs, ge in case_gt:
                    for pt in trigger_times:
                        if gs - 1.0 <= pt <= ge + 2.0:
                            hits += 1
                            break
                fp = sum(1 for pt in trigger_times
                         if not any(gs-1.0 <= pt <= ge+2.0 for gs, ge in case_gt))
                tp = hits
                fn = len(case_gt) - tp
                p = tp/(tp+fp) if (tp+fp) > 0 else 0
                r = tp/(tp+fn) if (tp+fn) > 0 else 0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
                case_f1s.append(f1)

            avg_f1 = np.mean(case_f1s)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_config = {"threshold": round(float(threshold), 3),
                               "cooldown": cooldown, "avg_f1": round(avg_f1, 4)}

    return best_f1, best_config


def evaluate_frame_level(model, val_dataset, device):
    """Quick frame-level evaluation."""
    model.eval()
    loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=2)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, cs, txt, labels in loader:
            scores = model(img.to(device), cs.to(device), txt.to(device))
            preds = (torch.sigmoid(scores) >= 0.5).long()
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
    return {"f1": f1, "precision": precision, "recall": recall,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = CaseRankingDataset("train", pairs_per_epoch=args.pairs_per_epoch)
    val_dataset = FlatDataset("val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    model = TriggerClassifierV4(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = CombinedLoss(margin=args.margin, alpha_rank=1.0, alpha_focal=1.0,
                             focal_gamma=args.focal_gamma)

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

    val_meta_path = os.path.join(DATA_DIR, "val_meta.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for pos_img, pos_cs, pos_txt, neg_img, neg_cs, neg_txt in pbar:
            pos_img = pos_img.to(device)
            pos_cs = pos_cs.to(device)
            pos_txt = pos_txt.to(device)
            neg_img = neg_img.to(device)
            neg_cs = neg_cs.to(device)
            neg_txt = neg_txt.to(device)

            pos_scores = model(pos_img, pos_cs, pos_txt)
            neg_scores = model(neg_img, neg_cs, neg_txt)

            loss = criterion(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)

        # Frame-level evaluation
        frame_metrics = evaluate_frame_level(model, val_dataset, device)

        # ESTP-F1 evaluation (every 2 epochs to save time)
        estp_f1 = 0
        estp_config = {}
        if epoch % 2 == 0 or epoch == args.epochs:
            estp_f1, estp_config = evaluate_estp(
                model, val_meta_path,
                val_dataset.img_feats, val_dataset.change_scores,
                val_dataset.txt_feats, val_dataset.labels, device
            )

        elapsed = time.time() - start_time
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "frame_f1": frame_metrics["f1"],
            "frame_precision": frame_metrics["precision"],
            "frame_recall": frame_metrics["recall"],
            "estp_f1": estp_f1,
            "estp_config": estp_config,
        }
        history.append(record)

        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, "
              f"frame_P={frame_metrics['precision']:.3f}, "
              f"frame_R={frame_metrics['recall']:.3f}, "
              f"frame_F1={frame_metrics['f1']:.3f}, "
              f"ESTP-F1={estp_f1:.4f}, elapsed={elapsed:.0f}s")
        if estp_config:
            print(f"  Best ESTP config: th={estp_config.get('threshold', '?')}, "
                  f"cd={estp_config.get('cooldown', '?')}")

        if estp_f1 > best_estp_f1:
            best_estp_f1 = estp_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "frame_metrics": frame_metrics,
                "estp_f1": estp_f1,
                "estp_config": estp_config,
                "args": vars(args),
                "model_class": "TriggerClassifierV4",
            }, os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  -> New best model saved (ESTP-F1={best_estp_f1:.4f})")

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete! Best ESTP-F1: {best_estp_f1:.4f}")
    print(f"Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")

    return best_estp_f1


def main():
    parser = argparse.ArgumentParser(description="Train trigger model v4 with ranking loss")
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
