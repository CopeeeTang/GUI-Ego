#!/usr/bin/env python3
"""
Change Detection Trigger Pilot: Two-Frame Comparison for Proactive Assistance
==============================================================================
Instead of asking "should I help now?" (single frame), we ask "what changed
between these two frames?" — a more concrete visual question that better
aligns with GT windows where specific events occur.

Hypothesis: VLM is better at detecting CHANGES (concrete, visual) than judging
TIMING (abstract, contextual). Change detection should reduce yes-bias and
improve precision on state-change / appearance events.

Uses Gemini-3-Flash via google.genai SDK (Vertex AI backend).

Usage:
    python3 change_detection_trigger_pilot.py --limit 5 --verbose
    python3 change_detection_trigger_pilot.py --limit 15 --verbose
"""

import argparse
import io
import json
import math
import os
import signal
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    load_dotenv(str(_env_path))

# -- Constants ----------------------------------------------------------------
ANTICIPATION = 1.0
LATENCY = 2.0
COOLDOWN = 12.0
POLL_INTERVAL = 10    # seconds between VLM calls
DELTA_SECONDS = 10    # gap between frame1 (previous) and frame2 (current)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET = SCRIPT_DIR / "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = SCRIPT_DIR / "../../../data/ESTP-Bench/full_scale_2fps_max384"
DEFAULT_RT_RESULTS = SCRIPT_DIR / "results/phase3_learned/reasoning_trigger_pilot.json"

# -- Change Detection Prompt --------------------------------------------------

CHANGE_DETECTION_PROMPT = """You are an AI assistant for smart glasses. The user has asked: "{question}"

I'm showing you two frames from the user's first-person view:
- Frame 1: from {delta}s ago
- Frame 2: the current view

Compare these two frames carefully. Has anything SIGNIFICANT changed between them that is directly relevant to the user's question?

Rules:
- Say YES only if you can identify a SPECIFIC, CONCRETE change (new object appeared, text became visible, state changed, action completed)
- Say NO if the scene is stable, changes are minor, or changes are unrelated to the question
- Do NOT say YES just because the scene is generally related to the question

Respond with ONLY "YES" or "NO"."""


# -- Data Loading -------------------------------------------------------------

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


def load_rt_case_ids(rt_results_path):
    """Load case IDs from reasoning trigger pilot results."""
    with open(rt_results_path) as f:
        data = json.load(f)
    return [r["case_id"] for r in data.get("results", [])]


# -- Video Frame Extraction ---------------------------------------------------

def find_video(video_uid, video_root):
    root = Path(video_root)
    for p in [root / f"{video_uid}.mp4", root / video_uid / f"{video_uid}.mp4"]:
        if p.exists():
            return p
    matches = list(root.glob(f"{video_uid[:8]}*.mp4"))
    return matches[0] if matches else None


def extract_single_frame(video_path, t_abs, fps_hint=2.0):
    """Extract a single frame at absolute time t_abs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(int(t_abs * fps), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# -- VLM Inference ------------------------------------------------------------

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
    print("  google.genai client initialized (Vertex AI)")
    return _client


def infer_change_detection(frame1_bytes, frame2_bytes, question, delta,
                           model_name="gemini-3-flash-preview", verbose=False):
    """
    Ask VLM: "Has anything significant changed between these two frames
    that is relevant to the user's question?"
    Returns: (decision: bool, raw_text: str)
    """
    from google.genai.types import GenerateContentConfig, Part

    client = _get_client()

    content_parts = [
        Part.from_bytes(data=frame1_bytes, mime_type="image/jpeg"),
        Part.from_bytes(data=frame2_bytes, mime_type="image/jpeg"),
        Part.from_text(text=CHANGE_DETECTION_PROMPT.format(
            question=question, delta=delta)),
    ]

    def _timeout_handler(signum, frame):
        raise TimeoutError("API call timed out after 60s")

    for attempt in range(3):
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(60)  # 60s timeout
            resp = client.models.generate_content(
                model=model_name,
                contents=content_parts,
                config=GenerateContentConfig(
                    max_output_tokens=64,
                    temperature=0.0,
                ),
            )
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            raw = (resp.text or "").strip().lower()
            decision = raw.startswith("yes")
            return decision, raw
        except TimeoutError:
            signal.alarm(0)
            if verbose:
                print(f"    [timeout] attempt {attempt+1}/3")
            time.sleep(5)
            continue
        except Exception as e:
            signal.alarm(0)
            if verbose:
                print(f"    [error] {e}")
            if "429" in str(e) and attempt < 2:
                time.sleep(10)
                continue
            return False, f"error: {e}"
    return False, "error: timeout after 3 retries"


# -- ESTP-F1 Metric ----------------------------------------------------------

def compute_estp_f1(trigger_times, gt_windows, duration):
    """Compute ESTP-F1 for a single case."""
    if not gt_windows:
        fp = len(trigger_times)
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fp": fp, "tp": 0, "fn": 0}

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


# -- Per-Case Evaluation ------------------------------------------------------

def evaluate_case(case, video_root, model_name, interval=POLL_INTERVAL,
                  delta=DELTA_SECONDS, cooldown=COOLDOWN, verbose=False):
    """Run change detection trigger on one case."""
    video_uid = case["video_uid"]
    duration = case.get("duration", 300.0)
    question = case.get("question", "")
    gt_windows = get_gt_windows(case["qa_raw"])

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
        t_abs_current = clip_start + t_b
        t_abs_prev = clip_start + max(0.0, t_b - delta)

        # Extract two frames: previous and current
        frame_prev = extract_single_frame(str(video_path), float(t_abs_prev))
        frame_curr = extract_single_frame(str(video_path), float(t_abs_current))

        if frame_prev is None or frame_curr is None:
            decisions.append((t_b, False, "no_frames"))
            continue

        actual_delta = t_b - max(0.0, t_b - delta)
        decision, raw = infer_change_detection(
            frame_prev, frame_curr, question, int(actual_delta),
            model_name, verbose)
        decisions.append((t_b, decision, raw))
        time.sleep(1.0)  # rate limit

        if decision and (t_abs_current - last_trigger) >= cooldown:
            trigger_times.append(float(t_abs_current))
            last_trigger = float(t_abs_current)

        if verbose:
            is_trigger = decision and float(t_abs_current) in trigger_times
            marker = " << TRIGGER" if is_trigger else ""
            gt_mark = ""
            for gs, ge in gt_windows:
                if gs - ANTICIPATION <= t_abs_current <= gs + LATENCY:
                    gt_mark = " [GT]"
                    break
            print(f"    t={t_b:6.1f}s (abs={t_abs_current:.1f}s)  "
                  f"{'YES' if decision else 'no ':3s}  "
                  f"raw={raw[:30]:<30s}{gt_mark}{marker}")

    metrics = compute_estp_f1(trigger_times, gt_windows, duration)
    metrics["trigger_times"] = trigger_times
    metrics["n_triggers"] = len(trigger_times)
    metrics["n_buckets"] = len(buckets)
    metrics["n_yes"] = sum(1 for _, d, _ in decisions if d)
    metrics["decisions"] = decisions

    return metrics


# -- Baseline Comparison ------------------------------------------------------

def compute_baseline_f1(case, cooldown=COOLDOWN):
    """Compute baseline (gap>0, no D) F1 from trigger_log."""
    trigger_log = case.get("trigger_log", [])
    if not trigger_log:
        bm = case.get("baseline_metrics", {})
        return bm.get("f1", 0.0)

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


# -- Checkpoint ---------------------------------------------------------------

def load_checkpoint(path):
    """Load completed case IDs from checkpoint file."""
    if not path.exists():
        return {}
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            results[r["case_id"]] = r
    return results


def save_checkpoint(path, result):
    """Append a single result to checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Change Detection Trigger Pilot")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--rt-results", default=str(DEFAULT_RT_RESULTS))
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL)
    parser.add_argument("--delta", type=int, default=DELTA_SECONDS)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Change Detection Trigger Pilot: Two-Frame Comparison")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Interval: {args.interval}s  |  Delta: {args.delta}s  |  Cooldown: {COOLDOWN}s")
    print()

    # Load the same cases as reasoning trigger pilot
    rt_case_ids = load_rt_case_ids(str(args.rt_results))
    print(f"Reasoning trigger case IDs: {len(rt_case_ids)} cases")

    all_cases = load_cases(args.checkpoint, args.dataset)
    case_by_id = {c["case_id"]: c for c in all_cases}

    cases = []
    for cid in rt_case_ids:
        if cid in case_by_id:
            cases.append(case_by_id[cid])

    if args.limit:
        cases = cases[:args.limit]
    print(f"Evaluating {len(cases)} cases\n")

    # Checkpoint support
    ckpt_path = SCRIPT_DIR / "results/phase3_learned/change_detection_checkpoint.jsonl"
    completed = load_checkpoint(ckpt_path)
    print(f"Checkpoint: {len(completed)} cases already completed")

    results = []
    t0_total = time.time()

    for i, case in enumerate(cases):
        cid = case["case_id"]
        task_type = case.get("task_type", "?")
        gt_windows = get_gt_windows(case["qa_raw"])
        baseline_f1 = compute_baseline_f1(case)

        print(f"[{i+1}/{len(cases)}] {cid}  [{task_type}]  GT={len(gt_windows)} windows")

        # Check if already in checkpoint
        if cid in completed:
            print(f"  [cached] F1={completed[cid]['f1']:.3f}")
            r = completed[cid]
            r["baseline_f1"] = baseline_f1
            r["delta_f1"] = r["f1"] - baseline_f1
            results.append(r)
            continue

        t0 = time.time()
        metrics = evaluate_case(
            case, args.video_root, args.model,
            interval=args.interval, delta=args.delta, verbose=args.verbose,
        )
        elapsed = time.time() - t0

        if metrics is None:
            print(f"  [skip] video not found")
            continue

        metrics["case_id"] = cid
        metrics["task_type"] = task_type
        metrics["baseline_f1"] = baseline_f1
        metrics["delta_f1"] = metrics["f1"] - baseline_f1

        # Save checkpoint (without decisions to keep file small)
        ckpt_entry = {k: v for k, v in metrics.items() if k != "decisions"}
        save_checkpoint(ckpt_path, ckpt_entry)

        results.append(metrics)

        print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FP={metrics['fp']}  triggers={metrics['n_triggers']}  "
              f"yes_rate={metrics['n_yes']}/{metrics['n_buckets']}  "
              f"({elapsed:.1f}s)")
        print(f"  Baseline F1={baseline_f1:.3f}  Delta={metrics['delta_f1']:+.3f}")
        print()

    total_time = time.time() - t0_total

    # -- Summary ---------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY: Change Detection Trigger")
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

    print(f"  Change Detection F1:   {avg_f1:.3f}")
    print(f"  Baseline F1:           {avg_bl:.3f}")
    print(f"  Delta F1:              {avg_delta:+.3f}")
    print(f"  Total triggers:        {total_triggers}  (yes_rate={total_yes}/{total_buckets}="
          f"{total_yes/max(total_buckets,1):.1%})")
    print(f"  Time: {total_time/60:.1f} min  ({total_time/max(len(results),1):.1f}s/case)")

    # Load RT results for comparison
    try:
        with open(args.rt_results) as f:
            rt_data = json.load(f)
        rt_by_id = {r["case_id"]: r for r in rt_data.get("results", [])}
        print(f"\n  Comparison with Reasoning Trigger (RT F1={rt_data['avg_f1']:.3f}):")

        # Only compare cases we actually ran
        our_ids = {r["case_id"] for r in results}
        matched_rt = [rt_by_id[cid] for cid in our_ids if cid in rt_by_id]
        matched_cd = [r for r in results if r["case_id"] in rt_by_id]

        if matched_rt and matched_cd:
            rt_f1 = np.mean([r["f1"] for r in matched_rt])
            cd_f1 = np.mean([r["f1"] for r in matched_cd])
            rt_yes = sum(r.get("n_yes", 0) for r in matched_rt)
            rt_bkt = sum(r.get("n_buckets", 0) for r in matched_rt)
            cd_yes = sum(r["n_yes"] for r in matched_cd)
            cd_bkt = sum(r["n_buckets"] for r in matched_cd)
            print(f"  [matched {len(matched_cd)} cases]")
            print(f"    Change Detection F1: {cd_f1:.3f}  yes_rate={cd_yes}/{cd_bkt}={cd_yes/max(cd_bkt,1):.1%}")
            print(f"    Reasoning Trigger F1:{rt_f1:.3f}  yes_rate={rt_yes}/{rt_bkt}={rt_yes/max(rt_bkt,1):.1%}")
            print(f"    Delta (CD - RT):     {cd_f1 - rt_f1:+.3f}")
    except Exception:
        pass

    # Per-type breakdown
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)

    print(f"\n  {'Task Type':<40} {'n':>3} {'CD F1':>8} {'BL F1':>8} {'Delta':>8}")
    print(f"  {'---'*20}")
    for tt in sorted(by_type):
        rr = by_type[tt]
        tf1 = np.mean([r["f1"] for r in rr])
        bf1 = np.mean([r["baseline_f1"] for r in rr])
        print(f"  {tt:<40} {len(rr):>3} {tf1:>8.3f} {bf1:>8.3f} {tf1-bf1:>+8.3f}")

    # Case-level detail
    print(f"\n  {'Case':<15} {'Type':<35} {'CD F1':>8} {'BL F1':>8} {'Delta':>8} {'Trig':>5} {'yes%':>6}")
    print(f"  {'---'*25}")
    for r in results:
        yr = r["n_yes"] / max(r["n_buckets"], 1) * 100
        print(f"  {r['case_id']:<15} {r['task_type']:<35} {r['f1']:>8.3f} "
              f"{r['baseline_f1']:>8.3f} {r['delta_f1']:>+8.3f} {r['n_triggers']:>5} {yr:>5.1f}%")

    print(f"\n{'='*70}")

    # Save full results
    output = SCRIPT_DIR / "results/phase3_learned/change_detection_trigger_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model": args.model,
        "interval": args.interval,
        "delta": args.delta,
        "cooldown": COOLDOWN,
        "n_cases": len(results),
        "avg_f1": float(avg_f1),
        "avg_baseline_f1": float(avg_bl),
        "avg_delta_f1": float(avg_delta),
        "total_yes_rate": total_yes / max(total_buckets, 1),
        "results": [{k: v for k, v in r.items() if k != "decisions"} for r in results],
    }
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
