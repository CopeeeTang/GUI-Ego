#!/usr/bin/env python3
"""
Phase III: Goal-State Extractor (Gemini via Vertex AI)
=====================================================
Same prompt and output format as phase3_goal_state_extractor.py (Qwen),
but uses Gemini via Vertex AI for inference.

Output: results/phase3_learned/goal_state_cache_gemini.json
  {case_id: {str(time_bucket): "on_track"|"uncertain"|"off_track"}}

Resumable: already-computed cases are preserved.

Usage:
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/phase3_goal_state_gemini.py --verbose
"""

import argparse
import base64
import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load .env
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))

# ── Constants ──────────────────────────────────────────────────────────────
GOAL_STATE_INTERVAL = 10
GOAL_FRAMES = 5
VALID_LABELS = {"on_track", "uncertain", "off_track"}

DEFAULT_CHECKPOINT = "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET    = "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = "../../../data/ESTP-Bench/full_scale_2fps_max384"
DEFAULT_OUTPUT     = "results/phase3_learned/goal_state_cache_gemini.json"

# Same prompt as Qwen extractor for fair comparison
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


# ── Data Loading ──────────────────────────────────────────────────────────

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
    root = Path(video_root)
    for p in [root / f"{video_uid}.mp4", root / video_uid / f"{video_uid}.mp4"]:
        if p.exists():
            return p
    matches = list(root.glob(f"{video_uid[:8]}*.mp4"))
    return matches[0] if matches else None


def extract_frames_at_time(video_path, t_bucket, n_frames=GOAL_FRAMES, fps_hint=2.0):
    """Extract n_frames leading up to t_bucket from video at 2fps spacing."""
    step = 1.0 / fps_hint
    times = [max(0.0, t_bucket - step * (n_frames - 1 - i)) for i in range(n_frames)]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in times:
        idx = min(int(t * fps), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def pil_to_b64_bytes(img, quality=85):
    """Convert PIL Image to base64-encoded JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ── Gemini Inference (Vertex AI) ──────────────────────────────────────────

_gemini_model = None


_gemini_model_name = "gemini-2.0-flash"


def _load_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model
    import vertexai
    from vertexai.generative_models import GenerativeModel
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    region = os.getenv("VERTEX_REGION", "global")
    vertexai.init(project=project, location=region)
    _gemini_model = GenerativeModel(_gemini_model_name)
    print(f"  Gemini initialized: {_gemini_model_name} (Vertex AI, project={project}, region={region})")
    return _gemini_model


def parse_label(text):
    text = text.strip().lower()
    for label in ("on_track", "off_track", "uncertain"):
        if label in text:
            return label
    # Handle truncated responses from thinking models (e.g. "off" → "off_track")
    if text.startswith("off"):
        return "off_track"
    if text.startswith("on"):
        return "on_track"
    if text.startswith("unc"):
        return "uncertain"
    return "uncertain"


def infer_gemini(frames, goal, verbose=False):
    """Run goal-state prompt via Gemini. Returns label string."""
    from vertexai.generative_models import Part
    model = _load_gemini()

    content = []
    for img in frames:
        img_bytes = pil_to_b64_bytes(img)
        content.append(Part.from_data(data=img_bytes, mime_type="image/jpeg"))
    content.append(GOAL_STATE_SYSTEM + "\n\n" + GOAL_STATE_USER.format(goal=goal))

    try:
        resp = model.generate_content(
            content,
            generation_config={"max_output_tokens": 512, "temperature": 0.0},
        )
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            if hasattr(resp, "candidates") and resp.candidates:
                for p in resp.candidates[0].content.parts:
                    try:
                        text += p.text
                    except Exception:
                        pass
        return parse_label(text)
    except Exception as e:
        if verbose:
            print(f"    [error] {e}")
        return "uncertain"


# ── Per-Case Goal-State ──────────────────────────────────────────────────

def compute_goal_state_for_case(case, video_root, interval=GOAL_STATE_INTERVAL,
                                goal_frames=GOAL_FRAMES, verbose=False):
    """Compute goal_state at each time bucket for one case."""
    video_uid = case["video_uid"]
    duration  = case.get("duration", 300.0)
    goal      = case.get("question", "")

    # FIX: Get clip_start_time from qa_raw to extract frames from correct video segment
    qa_raw = case.get("qa_raw", {})
    clip_start = qa_raw.get("clip_start_time", 0.0)

    video_path = find_video(video_uid, video_root)
    if video_path is None:
        if verbose:
            print(f"    [warn] video not found: {video_uid}")
        return {}

    if verbose and clip_start > 0:
        print(f"    clip_start_time={clip_start:.1f}s (absolute offset)")

    n_buckets = int(math.ceil(duration / interval)) + 1
    buckets = [i * interval for i in range(n_buckets)]

    result = {}
    for t_b in buckets:
        if t_b > duration + interval:
            break
        # FIX: Convert relative bucket time to absolute video time
        t_abs = clip_start + t_b
        frames = extract_frames_at_time(video_path, float(t_abs), n_frames=goal_frames)
        if not frames:
            result[str(float(t_b))] = "uncertain"
            continue

        label = infer_gemini(frames, goal, verbose=verbose)
        result[str(float(t_b))] = label

        if verbose:
            print(f"    t={t_b:6.1f}s (abs={t_abs:.1f}s) → {label}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase III: Goal-State Extractor (Gemini)")
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",     default=DEFAULT_DATASET)
    parser.add_argument("--video-root",  default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--interval",    type=float, default=GOAL_STATE_INTERVAL)
    parser.add_argument("--goal-frames", type=int,   default=GOAL_FRAMES)
    parser.add_argument("--limit",       type=int,   default=None)
    parser.add_argument("--model",       default="gemini-2.0-flash",
                        help="Vertex AI model name (e.g. gemini-3-flash-preview)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Set model name globally before loading
    global _gemini_model_name, _gemini_model
    _gemini_model_name = args.model
    _gemini_model = None  # Reset in case of re-run

    script_dir  = Path(__file__).parent
    checkpoint  = script_dir / args.checkpoint
    dataset     = script_dir / args.dataset
    video_root  = script_dir / args.video_root
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
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
    print(f"Processing {len(cases)} cases (interval={args.interval}s)")

    t0_total = time.time()
    for i, case in enumerate(cases):
        cid = case["case_id"]
        if cid in cache and cache[cid]:  # skip if already done AND non-empty
            print(f"[{i+1:3d}/{len(cases)}] {cid} — cached, skipping")
            continue

        t0 = time.time()
        task_type = case.get("task_type", "?")
        n_buckets = int(math.ceil(case.get("duration", 300.0) / args.interval)) + 1
        print(f"[{i+1:3d}/{len(cases)}] {cid}  [{task_type}]  ~{n_buckets} buckets")

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
        done_count = sum(1 for c in cases[:i+1] if c["case_id"] in cache and cache[c["case_id"]])
        if done_count > 0:
            total_elapsed = time.time() - t0_total
            avg_per = total_elapsed / done_count
            remaining = avg_per * (len(cases) - done_count)
        else:
            remaining = 0
        print(f"  Done in {elapsed:.1f}s ({n_buckets} buckets)  |  ETA: {remaining/60:.1f} min")

        # Save after each case
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
    print(f"Gemini goal_state cache saved: {output_path}")
    print(f"  Cases:     {len(cache)}")
    print(f"  Buckets:   {total_buckets}")
    print(f"  on_track:  {label_counts['on_track']} "
          f"({label_counts['on_track']/max(total_buckets,1):.1%})")
    print(f"  uncertain: {label_counts['uncertain']} "
          f"({label_counts['uncertain']/max(total_buckets,1):.1%})")
    print(f"  off_track: {label_counts['off_track']} "
          f"({label_counts['off_track']/max(total_buckets,1):.1%})")
    print(f"  Total:     {(time.time()-t0_total)/60:.1f} min")


if __name__ == "__main__":
    main()
