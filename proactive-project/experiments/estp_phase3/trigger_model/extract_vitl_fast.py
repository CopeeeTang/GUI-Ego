#!/usr/bin/env python3
"""
extract_vitl_fast.py - Fast ViT-L/14 feature extraction

Optimized:
- Pre-read all frames per video before GPU inference
- Large batch size (256)
- Only process unique (video, frame_idx) pairs
"""

import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data"
VITL_DATA_DIR = "/home/v-tangxin/GUI/proactive-project/experiments/estp_phase3/trigger_model/data_vitl"
VIDEO_DIR = "/home/v-tangxin/GUI/data/ESTP-Bench/full_scale_2fps_max384"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 256


@torch.no_grad()
def extract_split(split, model, processor, device):
    print(f"\n{'='*60}")
    print(f"Extracting ViT-L/14 features for {split}")

    with open(os.path.join(DATA_DIR, f"{split}_meta.json")) as f:
        metadata = json.load(f)
    labels = np.load(os.path.join(DATA_DIR, f"{split}_labels.npy"))

    n = len(metadata)
    feat_dim = 768
    img_feats = np.zeros((n, feat_dim), dtype=np.float32)
    txt_feats = np.zeros((n, feat_dim), dtype=np.float32)

    # Text features
    unique_qs = list(set(m["question"] for m in metadata))
    print(f"Extracting text features for {len(unique_qs)} unique questions...")
    q2feat = {}
    for i in range(0, len(unique_qs), 128):
        batch = unique_qs[i:i+128]
        inputs = processor(text=batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(device)
        out = model.text_model(**inputs)
        feat = model.text_projection(out.pooler_output)
        feat = F.normalize(feat, dim=-1)
        for j, q in enumerate(batch):
            q2feat[q] = feat[j].cpu().numpy()

    for i, m in enumerate(metadata):
        txt_feats[i] = q2feat[m["question"]]

    # Image features - group by video
    video_groups = defaultdict(list)
    for i, m in enumerate(metadata):
        video_groups[m["video_uid"]].append((i, m["frame_idx"]))

    # Count unique frames
    total_unique = sum(len(set(fi for _, fi in reqs)) for reqs in video_groups.values())
    print(f"Extracting image features: {len(video_groups)} videos, {total_unique} unique frames")

    start = time.time()
    frames_done = 0

    for vid_uid in tqdm(sorted(video_groups.keys()), desc=f"{split} videos"):
        video_path = os.path.join(VIDEO_DIR, f"{vid_uid}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        frame_requests = video_groups[vid_uid]
        unique_frames = sorted(set(fi for _, fi in frame_requests))
        frame_to_feat = {}

        # Read all needed frames first
        frame_images = {}
        for target_frame in unique_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_images[target_frame] = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame_images[target_frame] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

        # Process in batches
        batch_frames = []
        batch_idxs = []

        for fi in unique_frames:
            batch_frames.append(frame_images[fi])
            batch_idxs.append(fi)

            if len(batch_frames) >= BATCH_SIZE:
                inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
                vis_out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
                feat = model.visual_projection(vis_out.pooler_output)
                feat = F.normalize(feat, dim=-1).cpu().numpy()
                for j, fi in enumerate(batch_idxs):
                    frame_to_feat[fi] = feat[j]
                frames_done += len(batch_frames)
                batch_frames = []
                batch_idxs = []

        if batch_frames:
            inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
            vis_out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
            feat = model.visual_projection(vis_out.pooler_output)
            feat = F.normalize(feat, dim=-1).cpu().numpy()
            for j, fi in enumerate(batch_idxs):
                frame_to_feat[fi] = feat[j]
            frames_done += len(batch_frames)

        # Map back
        for orig_idx, fi in frame_requests:
            if fi in frame_to_feat:
                img_feats[orig_idx] = frame_to_feat[fi]

    elapsed = time.time() - start
    fps = frames_done / elapsed
    print(f"Done! {frames_done} frames in {elapsed:.0f}s = {fps:.0f} FPS")

    # Save
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_img_feats.npy"), img_feats)
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_txt_feats.npy"), txt_feats)
    np.save(os.path.join(VITL_DATA_DIR, f"{split}_labels.npy"), labels)
    print(f"Saved: img={img_feats.shape}, txt={txt_feats.shape}")


def main():
    os.makedirs(VITL_DATA_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ViT-L/14...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    # Only extract train if not already done
    if not os.path.exists(os.path.join(VITL_DATA_DIR, "train_img_feats.npy")):
        extract_split("train", model, processor, device)
    else:
        print("Train features already exist, skipping.")

    # Val already extracted, but redo if needed for consistency
    if not os.path.exists(os.path.join(VITL_DATA_DIR, "val_img_feats.npy")):
        extract_split("val", model, processor, device)
    else:
        print("Val features already exist, skipping.")


if __name__ == "__main__":
    main()
