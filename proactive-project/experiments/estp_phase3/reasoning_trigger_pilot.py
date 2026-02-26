#!/usr/bin/env python3
"""
Reasoning Trigger Pilot: Direct VLM-as-Judge for Proactive Assistance
=====================================================================
Instead of classifying goal_state (on_track/off_track), the VLM directly
reasons about WHETHER to proactively provide information NOW.

Key difference from goal_state:
- goal_state asks "what is the user's state?" → anti-correlated with trigger need
- reasoning trigger asks "should I help now?" → directly aligned with trigger decision

Uses Gemini-3-Flash via google.genai SDK (Vertex AI backend, thinking enabled).

Usage:
    python3 reasoning_trigger_pilot.py --limit 5 --verbose
"""

import argparse
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))

# ── Constants ──────────────────────────────────────────────────────────────
ANTICIPATION = 1.0
LATENCY = 2.0
COOLDOWN = 12.0
BUCKET_INTERVAL = 5   # seconds between VLM calls (finer than 10s for GT alignment)
N_FRAMES = 5          # frames per VLM call

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET = SCRIPT_DIR / "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = SCRIPT_DIR / "../../../data/ESTP-Bench/full_scale_2fps_max384"

# ── Reasoning Trigger Prompt ──────────────────────────────────────────────
# This prompt asks the VLM to REASON about whether to help,
# not just classify the user's state.

SYSTEM_PROMPT = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to decide: is RIGHT NOW the best moment to answer their question?"
)

USER_PROMPT_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

Is the user RIGHT NOW at the moment where the answer to their question is most relevant and useful?

Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene that the question asks about
2. The user appears to be actively engaging with or looking at it
3. Providing the answer NOW would be more useful than waiting

Say NO if the relevant object/action is not visible, or the user is doing something unrelated.

Respond with EXACTLY one word: YES or NO"""


# ── Data Loading ──────────────────────────────────────────────────────────

def build_dataset_index(dataset_path):
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
    ds_index = build_dataset_index(str(dataset_path))
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


def get_gt_windows(qa_raw):
    windows = []
    for turn in qa_raw.get("conversation", []):
        if turn.get("role", "").lower() == "assistant":
            st = turn.get("start_time")
            et = turn.get("end_time")
            if st is not None and et is not None:
                windows.append((float(st), float(et)))
    return windows


# ── Video Frame Extraction ────────────────────────────────────────────────

def find_video(video_uid, video_root):
    root = Path(video_root)
    for p in [root / f"{video_uid}.mp4", root / video_uid / f"{video_uid}.mp4"]:
        if p.exists():
            return p
    matches = list(root.glob(f"{video_uid[:8]}*.mp4"))
    return matches[0] if matches else None


def extract_frames(video_path, t_bucket, n_frames=N_FRAMES, fps_hint=2.0):
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
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames.append(buf.getvalue())
    cap.release()
    return frames


# ── VLM Inference ─────────────────────────────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    from google import genai
    _client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("VERTEX_REGION", "global"),
    )
    print(f"  google.genai client initialized (Vertex AI)")
    return _client


def infer_reasoning_trigger(frame_bytes_list, goal, model_name="gemini-3-flash-preview",
                            verbose=False):
    """
    Ask VLM: "Should I proactively help the user RIGHT NOW?"
    Returns: (decision: bool, raw_text: str)
    """
    from google.genai.types import GenerateContentConfig, Part

    client = _get_client()

    content_parts = [Part.from_bytes(data=fb, mime_type="image/jpeg") for fb in frame_bytes_list]
    prompt = USER_PROMPT_TEMPLATE.format(goal=goal, n_frames=len(frame_bytes_list))
    content_parts.append(Part.from_text(text=SYSTEM_PROMPT + "\n\n" + prompt))

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.0,
            ),
        )
        raw = (resp.text or "").strip().lower()
        decision = raw.startswith("yes")
        return decision, raw
    except Exception as e:
        if verbose:
            print(f"    [error] {e}")
        return False, f"error: {e}"


# ── ESTP-F1 Metric ───────────────────────────────────────────────────────

def compute_estp_f1(trigger_times, gt_windows, duration):
    """Compute ESTP-F1 for a single case."""
    if not gt_windows:
        # No GT → FP-only
        fp = len(trigger_times)
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fp": fp, "tp": 0, "fn": 0}

    # Match triggers to GT windows
    tp = 0
    matched_gt = set()
    fp = 0

    for tt in sorted(trigger_times):
        hit = False
        for i, (gs, ge) in enumerate(gt_windows):
            win_start = gs - ANTICIPATION
            win_end = gs + LATENCY
            if win_start <= tt <= win_end and i not in matched_gt:
                tp += 1
                matched_gt.add(i)
                hit = True
                break
        if not hit:
            fp += 1

    fn = len(gt_windows) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp, "tp": tp, "fn": fn}


# ── Per-Case Evaluation ──────────────────────────────────────────────────

def evaluate_case(case, video_root, model_name, interval=BUCKET_INTERVAL,
                  cooldown=COOLDOWN, verbose=False):
    """Run reasoning trigger on one case, return trigger_times and metrics."""
    video_uid = case["video_uid"]
    duration = case.get("duration", 300.0)
    goal = case.get("question", "")
    gt_windows = get_gt_windows(case["qa_raw"])

    # FIX: Get clip_start_time from qa_raw to extract frames from correct video segment
    qa_raw = case.get("qa_raw", {})
    clip_start = qa_raw.get("clip_start_time", 0.0)

    video_path = find_video(video_uid, str(video_root))
    if video_path is None:
        return None

    if verbose and clip_start > 0:
        print(f"    clip_start_time={clip_start:.1f}s (absolute offset)")

    n_buckets = int(math.ceil(duration / interval)) + 1
    buckets = [i * interval for i in range(n_buckets) if i * interval <= duration + interval]

    trigger_times = []
    last_trigger = -cooldown - 1.0
    decisions = []

    for t_b in buckets:
        # FIX: Convert relative bucket time to absolute video time for frame extraction
        t_abs = clip_start + t_b
        frames = extract_frames(str(video_path), float(t_abs))
        if not frames:
            decisions.append((t_b, False, "no_frames"))
            continue

        decision, raw = infer_reasoning_trigger(frames, goal, model_name, verbose)
        decisions.append((t_b, decision, raw))
        time.sleep(0.5)  # rate limit

        # FIX: Use absolute time for trigger tracking and GT comparison
        if decision and (t_abs - last_trigger) >= cooldown:
            trigger_times.append(float(t_abs))
            last_trigger = float(t_abs)

        if verbose:
            marker = " ◀ TRIGGER" if (decision and float(t_abs) in trigger_times) else ""
            gt_mark = ""
            for gs, ge in gt_windows:
                if gs - ANTICIPATION <= t_abs <= gs + LATENCY:
                    gt_mark = " [GT]"
                    break
            print(f"    t={t_b:6.1f}s (abs={t_abs:.1f}s)  {'YES' if decision else 'no ':3s}  "
                  f"raw={raw[:20]:<20s}{gt_mark}{marker}")

    metrics = compute_estp_f1(trigger_times, gt_windows, duration)
    metrics["trigger_times"] = trigger_times
    metrics["n_triggers"] = len(trigger_times)
    metrics["n_buckets"] = len(buckets)
    metrics["n_yes"] = sum(1 for _, d, _ in decisions if d)
    metrics["decisions"] = decisions

    return metrics


# ── Baseline Comparison ──────────────────────────────────────────────────

def compute_baseline_f1(case, cooldown=COOLDOWN):
    """Compute baseline (gap>0, no D) F1 from trigger_log."""
    trigger_log = case.get("trigger_log", [])
    if not trigger_log:
        bm = case.get("baseline_metrics", {})
        return bm.get("f1", 0.0)

    # Replay with gap>0
    gt_windows = get_gt_windows(case["qa_raw"])
    duration = case.get("duration", 300.0)

    trigger_times = []
    last_trigger = -cooldown - 1.0
    for e in trigger_log:
        gap = e.get("logprob_gap", e.get("gap", 0))
        t = e["time"]
        if gap > 0 and (t - last_trigger) >= cooldown:
            trigger_times.append(float(t))
            last_trigger = float(t)

    return compute_estp_f1(trigger_times, gt_windows, duration)["f1"]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reasoning Trigger Pilot")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--interval", type=int, default=BUCKET_INTERVAL)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start from this case index (skip earlier ones)")
    parser.add_argument("--min-bl-f1", type=float, default=0.0,
                        help="Only include cases with baseline F1 >= this value")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Reasoning Trigger Pilot: VLM-as-Judge for Proactive Assistance")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Interval: {args.interval}s  |  Cooldown: {COOLDOWN}s")
    print()

    all_cases = load_cases(args.checkpoint, args.dataset)

    # Filter by start index and min baseline F1
    cases = []
    for i, c in enumerate(all_cases):
        if i < args.start_idx:
            continue
        bl_f1 = c.get("baseline_metrics", {}).get("f1", 0)
        if bl_f1 >= args.min_bl_f1:
            cases.append(c)

    if args.limit:
        cases = cases[:args.limit]
    print(f"Evaluating {len(cases)} cases (start_idx={args.start_idx}, min_bl_f1={args.min_bl_f1})\n")

    results = []
    t0_total = time.time()

    for i, case in enumerate(cases):
        cid = case["case_id"]
        task_type = case.get("task_type", "?")
        gt_windows = get_gt_windows(case["qa_raw"])
        baseline_f1 = compute_baseline_f1(case)

        print(f"[{i+1}/{len(cases)}] {cid}  [{task_type}]  GT={len(gt_windows)} windows")

        t0 = time.time()
        metrics = evaluate_case(
            case, args.video_root, args.model,
            interval=args.interval, verbose=args.verbose,
        )
        elapsed = time.time() - t0

        if metrics is None:
            print(f"  [skip] video not found")
            continue

        metrics["case_id"] = cid
        metrics["task_type"] = task_type
        metrics["baseline_f1"] = baseline_f1
        metrics["delta_f1"] = metrics["f1"] - baseline_f1
        results.append(metrics)

        print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FP={metrics['fp']}  triggers={metrics['n_triggers']}  "
              f"yes_rate={metrics['n_yes']}/{metrics['n_buckets']}  "
              f"({elapsed:.1f}s)")
        print(f"  Baseline F1={baseline_f1:.3f}  Δ={metrics['delta_f1']:+.3f}")
        print()

    total_time = time.time() - t0_total

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not results:
        print("No results!")
        return

    avg_f1 = np.mean([r["f1"] for r in results])
    avg_bl = np.mean([r["baseline_f1"] for r in results])
    avg_delta = np.mean([r["delta_f1"] for r in results])
    total_triggers = sum(r["n_triggers"] for r in results)
    total_yes = sum(r["n_yes"] for r in results)
    total_buckets = sum(r["n_buckets"] for r in results)

    print(f"  Reasoning Trigger F1:  {avg_f1:.3f}")
    print(f"  Baseline F1:           {avg_bl:.3f}")
    print(f"  Delta F1:              {avg_delta:+.3f}")
    print(f"  Total triggers:        {total_triggers}  (yes_rate={total_yes}/{total_buckets}="
          f"{total_yes/max(total_buckets,1):.1%})")
    print(f"  Time: {total_time/60:.1f} min  ({total_time/len(results):.1f}s/case)")

    # Per-type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)

    print(f"\n  {'Task Type':<40} {'n':>3} {'Reason F1':>10} {'BL F1':>8} {'Delta':>8}")
    print(f"  {'─'*38}  {'─'*2} {'─'*9} {'─'*7} {'─'*7}")
    for tt in sorted(by_type):
        rr = by_type[tt]
        tf1 = np.mean([r["f1"] for r in rr])
        bf1 = np.mean([r["baseline_f1"] for r in rr])
        print(f"  {tt:<40} {len(rr):>3} {tf1:>10.3f} {bf1:>8.3f} {tf1-bf1:>+8.3f}")

    # Case-level detail
    print(f"\n  {'Case':<15} {'Type':<30} {'Reason F1':>10} {'BL F1':>8} {'Delta':>8} {'Triggers':>9}")
    print(f"  {'─'*13}  {'─'*28}  {'─'*9} {'─'*7} {'─'*7} {'─'*8}")
    for r in results:
        print(f"  {r['case_id']:<15} {r['task_type']:<30} {r['f1']:>10.3f} "
              f"{r['baseline_f1']:>8.3f} {r['delta_f1']:>+8.3f} {r['n_triggers']:>9}")

    print(f"\n{'='*70}")

    # Save results
    output = SCRIPT_DIR / "results/phase3_learned/reasoning_trigger_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model": args.model,
        "interval": args.interval,
        "n_cases": len(results),
        "avg_f1": avg_f1,
        "avg_baseline_f1": avg_bl,
        "avg_delta_f1": avg_delta,
        "results": [{k: v for k, v in r.items() if k != "decisions"} for r in results],
    }
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
