#!/usr/bin/env python3
"""
train_trigger_vitl.py - CLIP ViT-L/14 features + MLP trigger model

Uses larger CLIP model (768d features vs 512d) for better visual discrimination.
Re-extracts features from cached data, then trains MLP.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
VITL_DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data_vitl"
SAVE_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/checkpoints_vitl"
VIDEO_DIR = "/home/v-tangxin/GUI/data/ESTP-Bench/full_scale_2fps_max384"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


class TriggerClassifierViTL(nn.Module):
    """MLP classifier for ViT-L/14 features (768d)."""

    def __init__(self, img_dim=768, txt_dim=768, hidden_dim=384, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(96, 2),
        )

    def forward(self, img_feat, txt_feat):
        combined = torch.cat([img_feat, txt_feat], dim=-1)
        return self.classifier(combined)


class FeatureDataset(Dataset):
    def __init__(self, split="train", data_dir=VITL_DATA_DIR):
        self.img_feats = np.load(os.path.join(data_dir, f"{split}_img_feats.npy"))
        self.txt_feats = np.load(os.path.join(data_dir, f"{split}_txt_feats.npy"))
        self.labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"))
        print(f"Loaded {split}: {len(self.labels)} samples, "
              f"pos={self.labels.sum()}, neg={len(self.labels)-self.labels.sum()}, "
              f"feat_dim={self.img_feats.shape[1]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.img_feats[idx]).float(),
            torch.from_numpy(self.txt_feats[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


@torch.no_grad()
def extract_vitl_features(split, batch_size=128):
    """Extract ViT-L/14 features for a data split."""
    print(f"\n{'='*60}")
    print(f"Extracting ViT-L/14 features for {split}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    # Load existing metadata (same samples, just new features)
    with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
        metadata = json.load(f)
    labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))

    n = len(metadata)
    feat_dim = 768
    img_feats = np.zeros((n, feat_dim), dtype=np.float32)
    txt_feats = np.zeros((n, feat_dim), dtype=np.float32)

    # 1. Extract text features (unique questions)
    unique_questions = list(set(m["question"] for m in metadata))
    print(f"Extracting text features for {len(unique_questions)} unique questions...")
    q2feat = {}
    for i in range(0, len(unique_questions), batch_size):
        batch_qs = unique_questions[i:i+batch_size]
        txt_inputs = processor(text=batch_qs, return_tensors="pt", padding=True,
                              truncation=True, max_length=77).to(device)
        text_outputs = model.text_model(**txt_inputs)
        feat = model.text_projection(text_outputs.pooler_output)
        feat = F.normalize(feat, dim=-1)
        for j, q in enumerate(batch_qs):
            q2feat[q] = feat[j].cpu().numpy()

    for i, m in enumerate(metadata):
        txt_feats[i] = q2feat[m["question"]]

    # 2. Extract image features (by video, sequential)
    video_groups = defaultdict(list)
    for i, m in enumerate(metadata):
        video_groups[m["video_uid"]].append((i, m["frame_idx"]))

    print(f"Extracting image features from {len(video_groups)} videos...")

    for vid_uid in tqdm(sorted(video_groups.keys()), desc="Videos"):
        video_path = os.path.join(VIDEO_DIR, f"{vid_uid}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open {video_path}")
            continue

        frame_requests = video_groups[vid_uid]
        unique_frames = sorted(set(fi for _, fi in frame_requests))
        frame_to_feat = {}

        frame_batch = []
        frame_idx_batch = []

        for target_frame in unique_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_batch.append(frame)
            frame_idx_batch.append(target_frame)

            if len(frame_batch) >= batch_size:
                _process_batch(frame_batch, frame_idx_batch, frame_to_feat,
                              model, processor, device)
                frame_batch = []
                frame_idx_batch = []

        if frame_batch:
            _process_batch(frame_batch, frame_idx_batch, frame_to_feat,
                          model, processor, device)

        cap.release()

        for orig_idx, fi in frame_requests:
            if fi in frame_to_feat:
                img_feats[orig_idx] = frame_to_feat[fi]

    # Save
    os.makedirs(VITL_DATA_DIR, exist_ok=True)
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_img_feats.npy"), img_feats)
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_txt_feats.npy"), txt_feats)
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_labels.npy"), labels)

    print(f"Saved {split} features: img={img_feats.shape}, txt={txt_feats.shape}")
    return img_feats, txt_feats, labels


def _process_batch(frames, frame_indices, frame_to_feat, model, processor, device):
    img_inputs = processor(images=frames, return_tensors="pt", padding=True)
    pixel_values = img_inputs["pixel_values"].to(device)
    vis_outputs = model.vision_model(pixel_values=pixel_values)
    img_feat = model.visual_projection(vis_outputs.pooler_output)
    img_feat = F.normalize(img_feat, dim=-1).cpu().numpy()
    for i, fi in enumerate(frame_indices):
        frame_to_feat[fi] = img_feat[i]


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
        for img_feat, txt_feat, labels in loader:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            labels = labels.to(device)
            logits = model(img_feat, txt_feat)
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
        "precision": precision, "recall": recall, "f1": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate_estp(model, device):
    """Quick ESTP-F1 evaluation."""
    img_feats = np.load(os.path.join(VITL_DATA_DIR, "val_img_feats.npy"))
    txt_feats = np.load(os.path.join(VITL_DATA_DIR, "val_txt_feats.npy"))
    with open(os.path.join(DATA_DIR, "val_meta.json")) as f:
        metadata = json.load(f)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(img_feats), 1024):
            img_b = torch.from_numpy(img_feats[i:i+1024]).float().to(device)
            txt_b = torch.from_numpy(txt_feats[i:i+1024]).float().to(device)
            logits = model(img_b, txt_b)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
    probs = np.array(all_probs)

    cases = defaultdict(list)
    for i, m in enumerate(metadata):
        cases[(m["video_uid"], m["question"])].append(i)

    best_f1 = 0
    best_config = {}
    for threshold in np.arange(0.1, 1.0, 0.05):
        for cooldown in [0, 3, 6, 9, 12]:
            f1s = []
            for (vid, q), indices in cases.items():
                case_probs = probs[indices]
                case_ts = [metadata[i]["timestamp"] for i in indices]
                case_gt = metadata[indices[0]]["gt_windows"]
                sorted_pairs = sorted(zip(case_ts, case_probs))
                st = [p[0] for p in sorted_pairs]
                sp = [p[1] for p in sorted_pairs]
                trigger_times = []
                last = -float("inf")
                for p, t in zip(sp, st):
                    if t - last < cooldown: continue
                    if p >= threshold:
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
                f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0)
            avg = np.mean(f1s)
            if avg > best_f1:
                best_f1 = avg
                best_config = {"threshold": round(float(threshold), 3),
                               "cooldown": cooldown, "avg_f1": round(avg, 4)}

    return best_f1, best_config


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if features exist, otherwise extract
    if not os.path.exists(os.path.join(VITL_DATA_DIR, "val_img_feats.npy")):
        print("ViT-L/14 features not found, extracting...")
        extract_vitl_features("train", batch_size=args.extract_batch_size)
        extract_vitl_features("val", batch_size=args.extract_batch_size)
    else:
        print("ViT-L/14 features found, skipping extraction.")

    train_dataset = FeatureDataset("train", VITL_DATA_DIR)
    val_dataset = FeatureDataset("val", VITL_DATA_DIR)

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

    model = TriggerClassifierViTL(dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    class_weights = get_class_weights(labels)
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
        for img_feat, txt_feat, labels_batch in pbar:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(img_feat, txt_feat)
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
        val_metrics = evaluate(model, val_loader, device)

        # ESTP-F1 every 3 epochs
        estp_f1 = 0
        estp_config = {}
        if epoch % 3 == 0 or epoch == args.epochs or epoch == 1:
            estp_f1, estp_config = evaluate_estp(model, device)

        elapsed = time.time() - start_time
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "estp_f1": estp_f1,
            "estp_config": estp_config,
        }
        history.append(record)

        print(f"\nEpoch {epoch}: loss={avg_loss:.4f}, "
              f"val_P={val_metrics['precision']:.3f}, "
              f"val_R={val_metrics['recall']:.3f}, "
              f"val_F1={val_metrics['f1']:.3f}, "
              f"ESTP-F1={estp_f1:.4f}, elapsed={elapsed:.0f}s")

        if estp_f1 > best_estp_f1:
            best_estp_f1 = estp_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "estp_f1": estp_f1,
                "estp_config": estp_config,
                "args": vars(args),
                "model_class": "TriggerClassifierViTL",
            }, os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  -> New best (ESTP-F1={best_estp_f1:.4f})")

    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDone! Best ESTP-F1: {best_estp_f1:.4f}, Time: {total_time:.0f}s ({total_time/60:.1f}min)")

    return best_estp_f1


def main():
    parser = argparse.ArgumentParser(description="Train trigger model with ViT-L/14")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--extract_batch_size", type=int, default=128,
                        help="Batch size for CLIP feature extraction")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
