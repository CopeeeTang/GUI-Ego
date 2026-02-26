#!/usr/bin/env python3
"""
Phase III Stage B: Goal-State Memory Extractor
===============================================
Precomputes goal_state labels for each case at 10-second intervals.
Uses a 3-way classification prompt (on_track / uncertain / off_track)
to AVOID yes-bias — no binary yes/no question asked.

The "goal" is the case question (user's task intent).
The "current state" is the last GOAL_FRAMES frames before each time bucket.

Output: results/phase3_learned/goal_state_cache.json
  {case_id: {str(time_bucket): "on_track"|"uncertain"|"off_track"}}

Resumable: already-computed entries are preserved on restart.

Usage:
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/phase3_goal_state_extractor.py
    python3 proactive-project/experiments/estp_phase3/phase3_goal_state_extractor.py \\
        --checkpoint results/fullscale_d/checkpoint.jsonl \\
        --dataset ../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json \\
        --video-root ../../../data/ESTP-Bench/full_scale_2fps_max384 \\
        --output results/phase3_learned/goal_state_cache.json \\
        --interval 10 \\
        --goal-frames 5 \\
        --verbose
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
GOAL_STATE_INTERVAL = 10    # seconds between goal_state inferences
GOAL_FRAMES         = 5     # number of most-recent frames for context
MAX_FRAMES_VLM      = 8     # max frames passed to VLM (for VRAM safety)
VALID_LABELS        = {"on_track", "uncertain", "off_track"}

DEFAULT_CHECKPOINT  = "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET     = "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT  = "../../../data/ESTP-Bench/full_scale_2fps_max384"
DEFAULT_OUTPUT      = "results/phase3_learned/goal_state_cache.json"

# ── Goal-State Prompt ──────────────────────────────────────────────────────
# Deliberately avoids yes/no framing to bypass yes-bias.
# Uses 3-way classification with semantically distinct labels.
GOAL_STATE_SYSTEM = (
    "You are a smart glasses assistant. You observe the user's activities through "
    "first-person video frames. Your task is to assess their current progress "
    "toward a goal."
)

GOAL_STATE_USER = """The user's goal / what they need help with:
"{goal}"

The following frames show what the user is currently doing (in temporal order, most recent last).

Based on these frames, classify the user's CURRENT state into EXACTLY ONE category:

on_track   — The user is clearly making progress toward their goal. Relevant objects
             or actions are present and proceeding normally.

uncertain  — The situation is ambiguous. The user may be transitioning between steps,
             or their current action is unclear or not directly related to the goal.

off_track  — The user appears stuck, confused, lost, or is doing something that
             does not align with their goal. The goal-relevant object or action is
             absent when it should be present.

Respond with EXACTLY ONE word (on_track, uncertain, or off_track):"""


# ── Data Loading (same pattern as phase2 / phase3_learned_trigger) ─────────

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
    ds_index = _build_dataset_index(dataset_path)
    cases = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if "baseline_metrics" not in c:
                continue
            key = (c["video_uid"], c["clip_uid"], c.get("question", "").strip())
            qa_raw = ds_index.get(key)
            if qa_raw is None:
                continue
            c["qa_raw"] = qa_raw
            cases.append(c)
    if verbose:
        print(f"  Loaded {len(cases)} matched cases")
    return cases


# ── Video Frame Extraction ─────────────────────────────────────────────────

def find_video(video_uid, video_root):
    """Locate MP4 file for a given video_uid."""
    root = Path(video_root)
    candidates = [
        root / f"{video_uid}.mp4",
        root / video_uid / f"{video_uid}.mp4",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: glob
    matches = list(root.glob(f"{video_uid[:8]}*.mp4"))
    return matches[0] if matches else None


def extract_frames_at_times(video_path, times, max_frames=MAX_FRAMES_VLM):
    """
    Extract frames from video at the given timestamps (seconds).
    Returns list of PIL Images (at most max_frames, uniformly subsampled).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Subsample if too many times requested
    if len(times) > max_frames:
        idxs = np.linspace(0, len(times) - 1, max_frames, dtype=int)
        times = [times[i] for i in idxs]

    for t in times:
        frame_idx = min(int(t * fps), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


# ── VLM Model (lazy-loaded) ────────────────────────────────────────────────

_qwen_model = None
_qwen_processor = None


def _load_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print("  Loading Qwen3-VL-8B-Instruct...")
    _qwen_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    _qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _qwen_model.eval()
    print("  Model loaded.")
    return _qwen_model, _qwen_processor


def _infer_qwen(frames, goal, verbose=False):
    """
    Run Qwen3-VL with goal-state prompt.
    Returns: "on_track" | "uncertain" | "off_track"
    """
    import torch
    model, processor = _load_qwen()

    # Build message with image content
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    prompt_text = GOAL_STATE_USER.format(goal=goal)
    content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": GOAL_STATE_SYSTEM},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=frames if frames else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only new tokens
    input_len = inputs["input_ids"].shape[1]
    out_text = processor.decode(
        out_ids[0][input_len:], skip_special_tokens=True
    ).strip().lower()

    # Parse label
    for label in ("on_track", "off_track", "uncertain"):
        if label in out_text:
            return label
    if verbose:
        print(f"    [warn] unexpected output: '{out_text}' → defaulting to 'uncertain'")
    return "uncertain"


# ── Goal-State Computation ─────────────────────────────────────────────────

def compute_goal_state_for_case(case, video_root, interval=GOAL_STATE_INTERVAL,
                                goal_frames=GOAL_FRAMES, verbose=False):
    """
    Compute goal_state at each time bucket (floor(t/interval)*interval) for one case.

    Strategy:
      - Buckets: 0, interval, 2*interval, ... up to case duration
      - For each bucket t_b: take the last `goal_frames` trigger_log times <= t_b
      - Extract those frames from video
      - Run VLM with goal-state prompt
      - Cache result at bucket t_b

    Returns: dict {str(t_bucket): label}
    """
    video_uid = case["video_uid"]
    duration  = case.get("duration", 300.0)
    goal      = case.get("question", "")
    tlog      = case.get("trigger_log", [])

    # FIX: Get clip_start_time from qa_raw to align bucket times with trigger_log
    qa_raw = case.get("qa_raw", {})
    clip_start = qa_raw.get("clip_start_time", 0.0)

    video_path = find_video(video_uid, video_root)
    if video_path is None:
        if verbose:
            print(f"    [warn] video not found: {video_uid}")
        return {}

    if verbose and clip_start > 0:
        print(f"    clip_start_time={clip_start:.1f}s (absolute offset)")

    # All trigger_log timestamps (absolute video time)
    all_times = [e["time"] for e in tlog]

    # Generate time buckets (relative to clip start)
    n_buckets = int(math.ceil(duration / interval)) + 1
    buckets = [i * interval for i in range(n_buckets)]

    result = {}
    for t_b in buckets:
        if t_b > duration + interval:
            break
        # FIX: Convert relative bucket time to absolute for trigger_log matching
        t_abs = clip_start + t_b
        # Find last `goal_frames` trigger_log steps at or before t_abs
        recent_times = [t for t in all_times if t <= t_abs]
        if not recent_times:
            result[str(t_b)] = "uncertain"
            continue
        # Take last goal_frames
        frame_times = recent_times[-goal_frames:]
        frames = extract_frames_at_times(video_path, frame_times, max_frames=goal_frames)
        if not frames:
            result[str(t_b)] = "uncertain"
            continue

        label = _infer_qwen(frames, goal, verbose=verbose)
        result[str(t_b)] = label

        if verbose:
            print(f"    t={t_b:6.1f}s (abs={t_abs:.1f}s) → {label}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase III: Goal-State Extractor")
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",     default=DEFAULT_DATASET)
    parser.add_argument("--video-root",  default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--interval",    type=float, default=GOAL_STATE_INTERVAL)
    parser.add_argument("--goal-frames", type=int,   default=GOAL_FRAMES)
    parser.add_argument("--limit",       type=int,   default=None,
                        help="Process only first N cases (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    checkpoint  = script_dir / args.checkpoint
    dataset     = script_dir / args.dataset
    video_root  = script_dir / args.video_root
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache (for resumability)
    cache = {}
    if output_path.exists():
        with open(output_path) as f:
            cache = json.load(f)
        print(f"Loaded existing cache: {len(cache)} cases already done")

    # Load cases
    print("Loading cases...")
    cases = load_cases(str(checkpoint), str(dataset), verbose=args.verbose)
    if args.limit:
        cases = cases[:args.limit]
    print(f"Processing {len(cases)} cases (interval={args.interval}s, "
          f"goal_frames={args.goal_frames})")

    t0_total = time.time()
    for i, case in enumerate(cases):
        cid = case["case_id"]
        if cid in cache:
            print(f"[{i+1:3d}/{len(cases)}] {cid} — already cached, skipping")
            continue

        t0 = time.time()
        task_type = case.get("task_type", "?")
        goal_short = case.get("question", "")[:50]
        print(f"[{i+1:3d}/{len(cases)}] {cid}  [{task_type}]  '{goal_short}...'")

        try:
            result = compute_goal_state_for_case(
                case, str(video_root),
                interval=args.interval,
                goal_frames=args.goal_frames,
                verbose=args.verbose,
            )
            cache[cid] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            cache[cid] = {}

        elapsed = time.time() - t0
        total_elapsed = time.time() - t0_total
        avg_per_case = total_elapsed / (i + 1)
        remaining = avg_per_case * (len(cases) - i - 1)
        print(f"  Done in {elapsed:.1f}s  |  ETA: {remaining/60:.1f} min")

        # Save after each case (resumable)
        with open(output_path, "w") as f:
            json.dump(cache, f)

    # Final save
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    # Summary
    total_buckets = sum(len(v) for v in cache.values())
    label_counts = {"on_track": 0, "uncertain": 0, "off_track": 0}
    for case_data in cache.values():
        for label in case_data.values():
            if label in label_counts:
                label_counts[label] += 1

    print(f"\n{'='*50}")
    print(f"Cache saved: {output_path}")
    print(f"  Cases:    {len(cache)}")
    print(f"  Buckets:  {total_buckets}")
    print(f"  on_track:  {label_counts['on_track']} "
          f"({label_counts['on_track']/max(total_buckets,1):.1%})")
    print(f"  uncertain: {label_counts['uncertain']} "
          f"({label_counts['uncertain']/max(total_buckets,1):.1%})")
    print(f"  off_track: {label_counts['off_track']} "
          f"({label_counts['off_track']/max(total_buckets,1):.1%})")
    print(f"  Total time: {(time.time()-t0_total)/60:.1f} min")


if __name__ == "__main__":
    main()
