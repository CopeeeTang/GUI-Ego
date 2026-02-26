#!/usr/bin/env python3
"""
Phase III: Goal-State 3-Way Pilot Comparison
=============================================
Compares goal_state reasoning quality across 3 VLM backends:
  1. Qwen3-VL-8B (local, already cached)
  2. GPT-4o (Azure OpenAI)
  3. Gemini Flash (Google)

On 10-15 pilot cases covering 4-6 task types.

Metrics:
  - informative_rate = P(on_track | off_track)  — target: >= 45%
  - GT separation = mean(label==on_track | in GT window) vs out of GT window
  - latency per call
  - cost estimate

Gate: ΔF1 >= +0.01 AND CI does not cross 0 → proceed to full-scale.

Usage:
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/phase3_goal_state_pilot.py \\
        --n-per-type 3 --verbose
"""

import argparse
import base64
import io
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))

# ── Constants ──────────────────────────────────────────────────────────────
ANTICIPATION = 1.0
LATENCY_MARGIN = 2.0
GOAL_STATE_INTERVAL = 10.0
GOAL_FRAMES = 5

DEFAULT_CHECKPOINT = "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET    = "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = "../../../data/ESTP-Bench/full_scale_2fps_max384"
DEFAULT_OUTPUT     = "results/phase3_learned/pilot_comparison.json"
DEFAULT_QWEN_CACHE = "results/phase3_learned/goal_state_cache.json"

VALID_LABELS = {"on_track", "uncertain", "off_track"}

# Same prompt across all backends for fair comparison
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


def load_cases(checkpoint_path, dataset_path):
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
    return cases


def select_pilot_cases(cases, n_per_type=3, target_types=None):
    """Select n_per_type cases from each target task type."""
    if target_types is None:
        target_types = [
            "Task Understanding",
            "Information Function",
            "Object Recognition",
            "Action Recognition",
            "Attribute Perception",
        ]
    by_type = defaultdict(list)
    for c in cases:
        if c["task_type"] in target_types:
            by_type[c["task_type"]].append(c)
    selected = []
    for tt in target_types:
        selected.extend(by_type[tt][:n_per_type])
    return selected


# ── Video Frame Utils ─────────────────────────────────────────────────────

def find_video(video_uid, video_root):
    root = Path(video_root)
    for p in [root / f"{video_uid}.mp4", root / video_uid / f"{video_uid}.mp4"]:
        if p.exists():
            return p
    matches = list(root.glob(f"{video_uid[:8]}*.mp4"))
    return matches[0] if matches else None


def extract_frames_at_time(video_path, t_bucket, n_frames=GOAL_FRAMES, fps_hint=2.0):
    """
    Extract n_frames leading up to t_bucket from video.
    Frames are spaced at 1/fps_hint intervals ending at t_bucket.
    E.g. n_frames=5, fps=2 → times = [t-2.0, t-1.5, t-1.0, t-0.5, t]
    """
    # Generate frame times: evenly spaced ending at t_bucket
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


def pil_to_base64(img, quality=85):
    """Convert PIL Image to base64 JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_label(text):
    """Extract label from model response."""
    text = text.strip().lower()
    for label in ("on_track", "off_track", "uncertain"):
        if label in text:
            return label
    return "uncertain"


# ── Backend: GPT-4o via Azure OpenAI ─────────────────────────────────────

def _init_gpt4o():
    from openai import AzureOpenAI
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "http://52.151.57.21:9999")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-09-01-preview",
    )


def infer_gpt4o(client, frames, goal):
    """Run goal-state prompt via GPT-4o. Returns (label, latency_ms)."""
    content = []
    for img in frames:
        b64 = pil_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
        })
    content.append({"type": "text", "text": GOAL_STATE_USER.format(goal=goal)})

    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GOAL_STATE_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_tokens=8,
        temperature=0.0,
    )
    latency = (time.time() - t0) * 1000
    text = resp.choices[0].message.content or ""
    return parse_label(text), latency


# ── Backend: Gemini via Vertex AI ─────────────────────────────────────────

def _init_gemini():
    import vertexai
    from vertexai.generative_models import GenerativeModel
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    region = os.getenv("VERTEX_REGION", "global")
    vertexai.init(project=project, location=region)
    # gemini-2.0-flash: fast, reliable, no thinking overhead
    return GenerativeModel("gemini-2.0-flash")


def infer_gemini(model, frames, goal):
    """Run goal-state prompt via Gemini (Vertex AI). Returns (label, latency_ms)."""
    from vertexai.generative_models import Part
    content = []
    for img in frames:
        b64 = pil_to_base64(img)
        img_bytes = base64.b64decode(b64)
        content.append(Part.from_data(data=img_bytes, mime_type="image/jpeg"))
    content.append(GOAL_STATE_SYSTEM + "\n\n" + GOAL_STATE_USER.format(goal=goal))

    t0 = time.time()
    resp = model.generate_content(content, generation_config={"max_output_tokens": 32, "temperature": 0.0})
    latency = (time.time() - t0) * 1000
    # Extract text safely
    text = ""
    try:
        text = resp.text or ""
    except Exception:
        if hasattr(resp, "candidates") and resp.candidates:
            parts = resp.candidates[0].content.parts
            for p in parts:
                try:
                    text += p.text
                except Exception:
                    pass
    return parse_label(text), latency


# ── Backend: Qwen3-VL (from cache) ───────────────────────────────────────

def load_qwen_cache(cache_path):
    with open(cache_path) as f:
        return json.load(f)


def infer_qwen_cached(cache, case_id, t_bucket):
    """Look up Qwen result from pre-computed cache."""
    case_data = cache.get(case_id, {})
    for key in [str(float(t_bucket)), str(int(t_bucket))]:
        if key in case_data:
            return case_data[key], 0.0  # 0 latency (cached)
    return "uncertain", 0.0


# ── GT Window Analysis ────────────────────────────────────────────────────

def is_in_gt_window(t, qa_raw, anticipation=ANTICIPATION, latency=LATENCY_MARGIN):
    """Check if time t falls within any GT window."""
    for turn in qa_raw.get("conversation", []):
        if turn.get("role", "").lower() == "assistant":
            gs = turn.get("start_time")
            ge = turn.get("end_time")
            if gs is not None and ge is not None:
                if (gs - anticipation) <= t <= (ge + latency):
                    return True
    return False


# ── Main Pilot ────────────────────────────────────────────────────────────

def run_pilot(cases, video_root, qwen_cache, n_buckets_per_case=5, verbose=False):
    """
    For each pilot case, sample n_buckets_per_case time buckets,
    run all 3 backends, collect labels + latency.
    """
    # Init API clients
    print("Initializing API clients...")
    gpt4o_client = _init_gpt4o()
    gemini_model = _init_gemini()
    print("  GPT-4o: Azure OpenAI ready")
    print("  Gemini: gemini-3-flash-preview ready")
    print("  Qwen:   using cached results")

    all_results = []

    for i, case in enumerate(cases):
        cid       = case["case_id"]
        goal      = case.get("question", "")
        task_type = case.get("task_type", "?")
        duration  = case.get("duration", 300.0)
        qa_raw    = case["qa_raw"]

        video_path = find_video(case["video_uid"], video_root)
        if not video_path:
            print(f"  [{i+1}] SKIP — video not found: {case['video_uid']}")
            continue

        # Sample time buckets (evenly spaced from 10s to end)
        max_bucket = int(duration / GOAL_STATE_INTERVAL) * GOAL_STATE_INTERVAL
        if max_bucket < GOAL_STATE_INTERVAL:
            max_bucket = GOAL_STATE_INTERVAL
        buckets = np.linspace(
            GOAL_STATE_INTERVAL,
            min(max_bucket, duration),
            n_buckets_per_case,
        ).tolist()

        print(f"[{i+1:2d}/{len(cases)}] {cid}  [{task_type}]  {len(buckets)} buckets")

        for t_b in buckets:
            t_b = float(t_b)
            in_gt = is_in_gt_window(t_b, qa_raw)

            # Extract frames directly from video at bucket time
            frames = extract_frames_at_time(video_path, t_b)
            if not frames:
                if verbose:
                    print(f"    t={t_b:6.1f}s — no frames extracted, skipping")
                continue

            entry = {
                "case_id": cid,
                "task_type": task_type,
                "t_bucket": t_b,
                "in_gt": in_gt,
                "goal": goal[:80],
                "qwen": {},
                "gpt4o": {},
                "gemini": {},
            }

            # 1. Qwen (cached)
            label_q, lat_q = infer_qwen_cached(qwen_cache, cid, t_b)
            entry["qwen"] = {"label": label_q, "latency_ms": lat_q}

            # 2. GPT-4o
            try:
                label_g, lat_g = infer_gpt4o(gpt4o_client, frames, goal)
                entry["gpt4o"] = {"label": label_g, "latency_ms": lat_g}
            except Exception as e:
                entry["gpt4o"] = {"label": "error", "latency_ms": 0, "error": str(e)}

            # 3. Gemini
            try:
                label_m, lat_m = infer_gemini(gemini_model, frames, goal)
                entry["gemini"] = {"label": label_m, "latency_ms": lat_m}
            except Exception as e:
                entry["gemini"] = {"label": "error", "latency_ms": 0, "error": str(e)}

            all_results.append(entry)

            if verbose:
                gt_tag = "GT" if in_gt else "  "
                print(f"    t={t_b:6.1f}s [{gt_tag}]  "
                      f"Qwen={label_q:10s}  "
                      f"GPT4o={entry['gpt4o']['label']:10s} ({entry['gpt4o']['latency_ms']:.0f}ms)  "
                      f"Gemini={entry['gemini']['label']:10s} ({entry['gemini']['latency_ms']:.0f}ms)")

    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def analyze_results(results):
    """Compute comparison metrics across 3 backends."""
    backends = ["qwen", "gpt4o", "gemini"]
    analysis = {}

    for be in backends:
        labels  = [r[be]["label"] for r in results if r[be].get("label") not in ("error", None)]
        lats    = [r[be]["latency_ms"] for r in results if r[be].get("latency_ms", 0) > 0]
        in_gt   = [r[be]["label"] for r in results if r["in_gt"] and r[be].get("label") not in ("error",)]
        out_gt  = [r[be]["label"] for r in results if not r["in_gt"] and r[be].get("label") not in ("error",)]

        total = len(labels)
        n_on  = sum(1 for l in labels if l == "on_track")
        n_off = sum(1 for l in labels if l == "off_track")
        n_unc = sum(1 for l in labels if l == "uncertain")
        n_err = sum(1 for r in results if r[be].get("label") == "error")
        informative = n_on + n_off

        # GT separation: P(on_track | in_GT) vs P(on_track | out_GT)
        p_on_in_gt  = sum(1 for l in in_gt if l == "on_track") / max(len(in_gt), 1)
        p_on_out_gt = sum(1 for l in out_gt if l == "on_track") / max(len(out_gt), 1)
        gt_sep = p_on_in_gt - p_on_out_gt

        # P(off_track | out_GT) vs P(off_track | in_GT)
        p_off_out_gt = sum(1 for l in out_gt if l == "off_track") / max(len(out_gt), 1)
        p_off_in_gt  = sum(1 for l in in_gt if l == "off_track") / max(len(in_gt), 1)

        analysis[be] = {
            "total": total,
            "on_track": n_on,
            "uncertain": n_unc,
            "off_track": n_off,
            "errors": n_err,
            "informative_rate": informative / max(total, 1),
            "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
            "p90_latency_ms": float(np.percentile(lats, 90)) if lats else 0.0,
            "gt_separation": {
                "p_on_in_gt": p_on_in_gt,
                "p_on_out_gt": p_on_out_gt,
                "on_track_sep": gt_sep,
                "p_off_out_gt": p_off_out_gt,
                "p_off_in_gt": p_off_in_gt,
                "n_in_gt": len(in_gt),
                "n_out_gt": len(out_gt),
            },
        }

    return analysis


def print_report(analysis, results):
    """Print comparison table."""
    W = 70
    print(f"\n{'='*W}")
    print(f"GOAL-STATE 3-WAY PILOT COMPARISON")
    print(f"{'='*W}")
    print(f"  Samples: {len(results)}  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Label distribution
    print(f"\n{'─'*W}")
    print(f"LABEL DISTRIBUTION")
    print(f"{'─'*W}")
    print(f"  {'Backend':<12}  {'on_track':>8}  {'uncertain':>9}  {'off_track':>9}  {'errors':>6}  {'info_rate':>9}")
    for be in ["qwen", "gpt4o", "gemini"]:
        a = analysis[be]
        print(f"  {be:<12}  {a['on_track']:>8}  {a['uncertain']:>9}  {a['off_track']:>9}  "
              f"{a['errors']:>6}  {a['informative_rate']:>9.1%}")

    # GT Separation
    print(f"\n{'─'*W}")
    print(f"GT SEPARATION (higher = more discriminative)")
    print(f"{'─'*W}")
    print(f"  {'Backend':<12}  {'P(on|GT)':>8}  {'P(on|¬GT)':>9}  {'sep':>6}  "
          f"{'P(off|¬GT)':>10}  {'P(off|GT)':>9}")
    for be in ["qwen", "gpt4o", "gemini"]:
        g = analysis[be]["gt_separation"]
        print(f"  {be:<12}  {g['p_on_in_gt']:>8.3f}  {g['p_on_out_gt']:>9.3f}  "
              f"{g['on_track_sep']:>+6.3f}  "
              f"{g['p_off_out_gt']:>10.3f}  {g['p_off_in_gt']:>9.3f}")

    # Latency
    print(f"\n{'─'*W}")
    print(f"LATENCY (ms)")
    print(f"{'─'*W}")
    print(f"  {'Backend':<12}  {'avg':>8}  {'p90':>8}")
    for be in ["qwen", "gpt4o", "gemini"]:
        a = analysis[be]
        q_tag = " (cached)" if be == "qwen" else ""
        print(f"  {be:<12}  {a['avg_latency_ms']:>8.0f}  {a['p90_latency_ms']:>8.0f}{q_tag}")

    # Gate decision
    print(f"\n{'─'*W}")
    print(f"GATE DECISION")
    print(f"{'─'*W}")
    for be in ["gpt4o", "gemini"]:
        a = analysis[be]
        ir = a["informative_rate"]
        sep = a["gt_separation"]["on_track_sep"]
        pass_ir = ir >= 0.45
        pass_sep = sep > 0.0
        verdict = "PASS" if (pass_ir and pass_sep) else "FAIL"
        print(f"  {be:<12}  info_rate={ir:.1%} ({'✓' if pass_ir else '✗'} ≥45%)  "
              f"GT_sep={sep:+.3f} ({'✓' if pass_sep else '✗'} >0)  "
              f"→ {verdict}")
    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase III: Goal-State 3-Way Pilot")
    parser.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",      default=DEFAULT_DATASET)
    parser.add_argument("--video-root",   default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--qwen-cache",   default=DEFAULT_QWEN_CACHE)
    parser.add_argument("--output",       default=DEFAULT_OUTPUT)
    parser.add_argument("--n-per-type",   type=int, default=3)
    parser.add_argument("--n-buckets",    type=int, default=5,
                        help="Time buckets per case to sample")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    checkpoint = script_dir / args.checkpoint
    dataset    = script_dir / args.dataset
    video_root = script_dir / args.video_root
    qwen_cache_path = script_dir / args.qwen_cache
    output_path = script_dir / args.output

    # Load
    print("Loading cases...")
    cases = load_cases(str(checkpoint), str(dataset))
    print(f"  Total: {len(cases)} cases")

    pilot = select_pilot_cases(cases, n_per_type=args.n_per_type)
    print(f"  Pilot: {len(pilot)} cases across "
          f"{len(set(c['task_type'] for c in pilot))} task types")

    qwen_cache = load_qwen_cache(str(qwen_cache_path))
    print(f"  Qwen cache: {len(qwen_cache)} cases")

    # Run
    print(f"\nRunning pilot ({len(pilot)} cases × {args.n_buckets} buckets = "
          f"~{len(pilot) * args.n_buckets} API calls per backend)...")
    results = run_pilot(
        pilot, str(video_root), qwen_cache,
        n_buckets_per_case=args.n_buckets,
        verbose=args.verbose,
    )

    # Analyze
    analysis = analyze_results(results)
    print_report(analysis, results)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "results": results,
            "analysis": analysis,
            "meta": {
                "n_cases": len(pilot),
                "n_buckets": args.n_buckets,
                "generated": datetime.now().isoformat(),
            },
        }, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
