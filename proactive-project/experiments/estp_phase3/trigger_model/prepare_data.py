#!/usr/bin/env python3
"""
prepare_data.py - 为触发模型准备训练数据

功能:
1. 从 estp_bench_sq.json 加载所有 1212 QAs
2. 对每个 QA 提取帧级标注（正样本=在 GT window 附近，负样本=其余）
3. 提取 CLIP 视觉+文本特征并缓存到 .npy
4. 按视频 80/20 划分 train/val

优化策略:
- 按视频顺序读取帧（避免反复 seek）
- 文本特征按唯一 question 提取（~1212 而非 654K）
- 批量 GPU 推理
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ──────── 常量 ────────
ANTICIPATION = 1.0
LATENCY = 2.0
FPS = 2
TRAIN_RATIO = 0.8

ESTP_JSON = "/home/v-tangxin/GUI/data/ESTP-Bench/estp_dataset/estp_bench_sq.json"
VIDEO_DIR = "/home/v-tangxin/GUI/data/ESTP-Bench/full_scale_2fps_max384"
OUTPUT_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def ceil_time_by_fps(t, fps, min_t, max_t):
    return min(max(math.ceil(t * fps) / fps, min_t), max_t)


def load_estp_data():
    """加载 ESTP-Bench 数据，返回平坦化的 QA 列表."""
    with open(ESTP_JSON) as f:
        data = json.load(f)

    qas = []
    skipped_goalstep = 0
    for video_uid, clips in data.items():
        for clip_uid, qa_list in clips.items():
            for qa in qa_list:
                if clip_uid == "goalstep":
                    user_entries = [c for c in qa["conversation"] if c["role"] == "user"]
                    if not user_entries:
                        skipped_goalstep += 1
                        continue
                    question = user_entries[0]["content"]
                    clip_start = qa.get("start_time", 0)
                    clip_end = qa.get("end_time", qa.get("duration", 300))
                    qas.append({
                        "video_uid": qa.get("video_uid", video_uid),
                        "clip_uid": clip_uid,
                        "clip_start_time": clip_start,
                        "clip_end_time": clip_end,
                        "question": question,
                        "task_type": qa.get("Task Type", "Task Understanding").strip(),
                        "conversation": qa["conversation"],
                    })
                else:
                    qas.append({
                        "video_uid": video_uid,
                        "clip_uid": clip_uid,
                        "clip_start_time": qa["clip_start_time"],
                        "clip_end_time": qa["clip_end_time"],
                        "question": qa["question"],
                        "task_type": qa.get("Task Type", "unknown").strip(),
                        "conversation": qa["conversation"],
                    })
    if skipped_goalstep:
        print(f"Skipped {skipped_goalstep} goalstep entries without user question")
    return qas


def get_gt_windows(qa):
    """提取 GT windows."""
    gold = [c for c in qa["conversation"] if c["role"].lower() == "assistant"]
    clip_end = qa["clip_end_time"]
    windows = []
    for c in gold:
        s = ceil_time_by_fps(c["start_time"], FPS, 0, clip_end)
        e = ceil_time_by_fps(c["end_time"], FPS, 0, clip_end)
        windows.append((s, e))
    return windows


def frame_in_gt(frame_time, gt_windows, anticipation=ANTICIPATION, latency=LATENCY):
    for gs, ge in gt_windows:
        if gs - anticipation <= frame_time <= ge + latency:
            return True
    return False


def build_frame_labels(qas):
    """为所有 QA 构建帧级标注."""
    samples = []
    skipped = 0

    for qa in tqdm(qas, desc="Building frame labels"):
        video_uid = qa["video_uid"]
        video_path = os.path.join(VIDEO_DIR, f"{video_uid}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        clip_start = qa["clip_start_time"]
        clip_end = qa["clip_end_time"]
        gt_windows = get_gt_windows(qa)
        question = qa["question"]
        task_type = qa["task_type"]

        start_frame = int(clip_start * FPS)
        end_frame = int(clip_end * FPS)

        for frame_idx in range(start_frame, end_frame + 1):
            frame_time = frame_idx / FPS
            label = 1 if frame_in_gt(frame_time, gt_windows) else 0
            samples.append({
                "video_uid": video_uid,
                "frame_idx": frame_idx,
                "timestamp": frame_time,
                "question": question,
                "task_type": task_type,
                "label": label,
                "gt_windows": gt_windows,
            })

    print(f"Total samples: {len(samples)}, skipped videos: {skipped}")
    pos = sum(1 for s in samples if s["label"] == 1)
    neg = len(samples) - pos
    print(f"Positive: {pos} ({pos/len(samples)*100:.1f}%), Negative: {neg} ({neg/len(samples)*100:.1f}%)")
    return samples


def split_by_video(samples, train_ratio=TRAIN_RATIO):
    """按视频划分 train/val."""
    video_uids = sorted(set(s["video_uid"] for s in samples))
    random.seed(42)
    random.shuffle(video_uids)
    n_train = int(len(video_uids) * train_ratio)
    train_vids = set(video_uids[:n_train])
    val_vids = set(video_uids[n_train:])

    train_samples = [s for s in samples if s["video_uid"] in train_vids]
    val_samples = [s for s in samples if s["video_uid"] in val_vids]

    print(f"Train: {len(train_samples)} samples from {len(train_vids)} videos")
    print(f"Val: {len(val_samples)} samples from {len(val_vids)} videos")
    return train_samples, val_samples


@torch.no_grad()
def extract_text_features(samples, model, processor, device):
    """提取所有唯一 question 的文本特征（只需 ~1212 次而非 654K 次）."""
    unique_questions = list(set(s["question"] for s in samples))
    print(f"Extracting text features for {len(unique_questions)} unique questions...")

    q2feat = {}
    batch_size = 128
    for i in range(0, len(unique_questions), batch_size):
        batch_qs = unique_questions[i:i+batch_size]
        txt_inputs = processor(text=batch_qs, return_tensors="pt", padding=True,
                              truncation=True, max_length=77).to(device)
        # Get text embeddings via text model + projection
        text_outputs = model.text_model(**txt_inputs)
        txt_feat = model.text_projection(text_outputs.pooler_output)
        txt_feat = F.normalize(txt_feat, dim=-1)
        for j, q in enumerate(batch_qs):
            q2feat[q] = txt_feat[j].cpu().numpy()

    # Map back to sample order
    txt_feats = np.stack([q2feat[s["question"]] for s in samples])
    return txt_feats


@torch.no_grad()
def extract_image_features(samples, model, processor, device, batch_size=256):
    """按视频顺序读取帧并提取 CLIP 图像特征.

    优化: 按 (video_uid, frame_idx) 排序，顺序读取避免 seek。
    同一视频的帧连续处理。
    """
    # 按视频和帧排序
    indexed = list(enumerate(samples))
    indexed.sort(key=lambda x: (x[1]["video_uid"], x[1]["frame_idx"]))

    img_feats = np.zeros((len(samples), 512), dtype=np.float32)
    current_video = None
    cap = None
    current_frame_pos = -1

    # 按视频分组
    video_groups = defaultdict(list)
    for orig_idx, s in indexed:
        video_groups[s["video_uid"]].append((orig_idx, s["frame_idx"]))

    print(f"Processing {len(video_groups)} videos for image features...")

    for vid_uid in tqdm(sorted(video_groups.keys()), desc="Videos"):
        video_path = os.path.join(VIDEO_DIR, f"{vid_uid}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open {video_path}")
            continue

        # 获取该视频需要的所有帧（已排序）
        frame_requests = video_groups[vid_uid]
        # 去重帧号（同一帧可能被多个 QA 引用）
        unique_frames = sorted(set(fi for _, fi in frame_requests))
        frame_to_feat = {}

        # 批量读取并提取特征
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
                _process_image_batch(frame_batch, frame_idx_batch, frame_to_feat,
                                    model, processor, device)
                frame_batch = []
                frame_idx_batch = []

        if frame_batch:
            _process_image_batch(frame_batch, frame_idx_batch, frame_to_feat,
                                model, processor, device)

        cap.release()

        # 映射回原始索引
        for orig_idx, fi in frame_requests:
            if fi in frame_to_feat:
                img_feats[orig_idx] = frame_to_feat[fi]

    return img_feats


def _process_image_batch(frames, frame_indices, frame_to_feat, model, processor, device):
    """处理一批帧的 CLIP 图像特征."""
    img_inputs = processor(images=frames, return_tensors="pt", padding=True)
    pixel_values = img_inputs["pixel_values"].to(device)
    vis_outputs = model.vision_model(pixel_values=pixel_values)
    img_feat = model.visual_projection(vis_outputs.pooler_output)
    img_feat = F.normalize(img_feat, dim=-1).cpu().numpy()

    for i, fi in enumerate(frame_indices):
        frame_to_feat[fi] = img_feat[i]


@torch.no_grad()
def extract_clip_features(samples, split_name, batch_size=256):
    """提取 CLIP 特征并保存."""
    print(f"\n{'='*60}")
    print(f"Extracting CLIP features for {split_name} ({len(samples)} samples)")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    # 1. 提取文本特征（快，只需处理 ~1212 unique questions）
    txt_feats = extract_text_features(samples, model, processor, device)

    # 2. 提取图像特征（按视频顺序读取）
    img_feats = extract_image_features(samples, model, processor, device, batch_size)

    labels = np.array([s["label"] for s in samples])

    print(f"Image features shape: {img_feats.shape}")
    print(f"Text features shape: {txt_feats.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Positive ratio: {labels.sum()}/{len(labels)} = {labels.mean()*100:.1f}%")

    return img_feats, txt_feats, labels


def save_metadata(samples, path):
    """保存样本元数据（不含特征）."""
    meta = []
    for s in samples:
        meta.append({
            "video_uid": s["video_uid"],
            "frame_idx": s["frame_idx"],
            "timestamp": s["timestamp"],
            "question": s["question"],
            "task_type": s["task_type"],
            "label": s["label"],
            "gt_windows": s["gt_windows"],
        })
    with open(path, "w") as f:
        json.dump(meta, f)
    print(f"Saved metadata to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare trigger model training data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for CLIP feature extraction")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature extraction, only build labels")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("Loading ESTP-Bench data...")
    qas = load_estp_data()
    print(f"Loaded {len(qas)} QAs")

    # Step 2: 构建帧级标注
    print("\nBuilding frame-level labels...")
    samples = build_frame_labels(qas)

    # Step 3: 划分 train/val
    train_samples, val_samples = split_by_video(samples)

    # Step 4: 保存元数据
    save_metadata(train_samples, os.path.join(OUTPUT_DIR, "train_meta.json"))
    save_metadata(val_samples, os.path.join(OUTPUT_DIR, "val_meta.json"))

    if args.skip_features:
        print("\nSkipping feature extraction (--skip_features)")
        return

    # Step 5: 提取 CLIP 特征
    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        img_feats, txt_feats, labels = extract_clip_features(
            split_samples, split_name, batch_size=args.batch_size
        )
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_img_feats.npy"), img_feats)
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_txt_feats.npy"), txt_feats)
        np.save(os.path.join(OUTPUT_DIR, f"{split_name}_labels.npy"), labels)
        print(f"Saved {split_name} features to {OUTPUT_DIR}/")

    print("\nDone! Data preparation complete.")


if __name__ == "__main__":
    main()
