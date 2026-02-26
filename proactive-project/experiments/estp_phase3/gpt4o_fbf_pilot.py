#!/usr/bin/env python3
"""
GPT-4o FBF (Frame-by-Frame) Yes-Bias Pilot Test.

Purpose:
    Verify whether GPT-4o exhibits the same yes-bias observed in Qwen3-VL
    (76-83% yes-rate) when asked "Is it the right time to answer?" on
    individual video frames. Uses Azure OpenAI API with logprobs to
    compute logP(yes) - logP(no) gap at each polling step.

Background:
    Qwen3-VL shows strong yes-bias in FBF trigger mode: when asked whether
    it's the right time to answer a question, it responds "yes" on 76-83%
    of all frames regardless of ground truth timing. This bias collapses
    precision and inflates false positives. We need to verify if this is
    a VLM-universal phenomenon or Qwen-specific.

Architecture:
    Reuses data loading & frame extraction from fullscale_d_runner.py.
    Replaces Qwen3-VL local inference with Azure OpenAI GPT-4o API calls.
    Uses the same prompt format and evaluation pipeline for direct comparison.

API:
    GPT-4o via Azure OpenAI endpoint: http://52.151.57.21:9999
    Supports: logprobs=True, top_logprobs=20 for first-token probability.

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate

    # Pilot: 5 cases (default)
    python3 proactive-project/experiments/estp_phase3/gpt4o_fbf_pilot.py \
        --limit 5 --verbose

    # Resume interrupted run
    python3 proactive-project/experiments/estp_phase3/gpt4o_fbf_pilot.py \
        --resume --verbose

    # Report from existing checkpoint (no API calls)
    python3 proactive-project/experiments/estp_phase3/gpt4o_fbf_pilot.py \
        --report_only
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ─── Path setup ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # /home/v-tangxin/GUI
ESTP_DIR = PROJECT_ROOT / "data" / "ESTP-Bench" / "estp_dataset"
VIDEO_ROOT = PROJECT_ROOT / "data" / "ESTP-Bench" / "full_scale_2fps_max384"
DATA_5CASES = ESTP_DIR / "estp_bench_sq_5_cases.json"
OUTPUT_DIR = SCRIPT_DIR / "results" / "gpt4o_fbf"

# Azure OpenAI API endpoint (Azure-style path, not /v1/)
_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT", "http://52.151.57.21:9999").rstrip("/")
_API_VERSION = os.getenv("api_version", "2024-09-01-preview")
_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("api_key", ""))
API_URL = f"{_API_BASE}/openai/deployments/gpt-4o/chat/completions?api-version={_API_VERSION}"
API_HEADERS = {
    "Content-Type": "application/json",
    "api-key": _API_KEY,
}

# Evaluation constants (same as hypothesis_runner.py)
ANTICIPATION = 1
LATENCY = 2

# Import shared utilities
sys.path.insert(0, str(SCRIPT_DIR))
from prompts import PROMPT_TEMPLATE, get_trigger_prompt, get_answer_prompt
from hypothesis_runner import ceil_time_by_fps, compute_case_metrics


# ═══════════════════════════════════════════════════════════
#  Data Loading (reuses fullscale_d_runner logic)
# ═══════════════════════════════════════════════════════════

def flatten_dataset_to_cases(data):
    """Convert nested 5-cases JSON to flat list of case dicts.

    Handles both standard format (clip_start_time) and goalstep format
    (start_time + conversation). See fullscale_d_runner.py for details.
    """
    cases = []
    for vid_uid, clips in data.items():
        for clip_uid, qas in clips.items():
            if not isinstance(qas, list):
                continue
            for idx, qa in enumerate(qas):
                task_type = qa.get("Task Type", "").strip()

                if qa.get("clip_start_time") is not None:
                    cases.append({
                        "video_uid": vid_uid,
                        "clip_uid": clip_uid,
                        "qa_id": qa.get("qa_id", idx),
                        "task_type": task_type,
                        "question": qa.get("question", ""),
                        "start_time": float(qa["clip_start_time"]),
                        "end_time": float(qa["clip_end_time"]),
                        "qa_raw": qa,
                    })
                elif qa.get("start_time") is not None:
                    conv = qa.get("conversation", [])
                    user_msgs = [c for c in conv if c.get("role", "").lower() == "user"]
                    question = user_msgs[0]["content"] if user_msgs else ""
                    cases.append({
                        "video_uid": qa.get("video_uid", vid_uid),
                        "clip_uid": clip_uid,
                        "qa_id": idx,
                        "task_type": task_type,
                        "question": question,
                        "start_time": float(qa["start_time"]),
                        "end_time": float(qa["end_time"]),
                        "qa_raw": qa,
                    })
    return cases


def find_video_file(video_uid, clip_uid):
    """Find the video file path by trying common naming patterns."""
    for ext in [".mp4", ".mkv", ".avi"]:
        for pattern in [
            VIDEO_ROOT / f"{video_uid}{ext}",
            VIDEO_ROOT / video_uid / f"{clip_uid}{ext}",
            VIDEO_ROOT / f"{clip_uid}{ext}",
        ]:
            if pattern.exists():
                return str(pattern)
    # Fallback: recursive search by prefix
    for f in VIDEO_ROOT.rglob(f"*{video_uid[:8]}*"):
        if f.suffix in [".mp4", ".mkv", ".avi"]:
            return str(f)
    return None


# ═══════════════════════════════════════════════════════════
#  Frame Extraction (reuses _extract_frames_upto logic)
# ═══════════════════════════════════════════════════════════

def extract_single_frame(video_path, timestamp):
    """Extract a single frame at the given timestamp.

    For GPT-4o FBF, we send ONE frame per polling step (not accumulated
    frames) to match the single-image-per-call API pattern and reduce
    API cost. This also provides a cleaner yes-bias measurement since
    the model sees exactly one moment in time.

    Returns: PIL.Image or None
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(int(timestamp * video_fps), total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_base64(img, quality=80):
    """Convert PIL Image to base64-encoded JPEG string.

    Uses quality=80 (not 95) to reduce payload size for API calls.
    Typical frame ~50-80KB at 384p with q=80.
    """
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════
#  GPT-4o API Inference
# ═══════════════════════════════════════════════════════════

def gpt4o_trigger_with_logprobs(img_b64, prompt, max_retries=6, timeout=30):
    """Call GPT-4o API with a single image and prompt, return logprobs.

    Sends a vision request with logprobs=True, top_logprobs=20 to get
    the first-token probability distribution. Extracts logP(yes) and
    logP(no) from the top_logprobs list.

    Args:
        img_b64: Base64-encoded JPEG image
        prompt: The trigger prompt text
        max_retries: Number of retries on transient failures
        timeout: Request timeout in seconds

    Returns:
        (response_text, yes_logp, no_logp, raw_top_logprobs)
        On failure: ("ERROR", -inf, -inf, [])
    """
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": 5,  # Azure OpenAI max is 5
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, json=payload, headers=API_HEADERS, timeout=timeout
            )

            # Handle rate limiting explicitly
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                wait = max(retry_after, 1)
                if attempt < max_retries - 1:
                    print(f"      [API] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"      [API] Rate limited after {max_retries} attempts")
                    return "ERROR:rate_limit", float("-inf"), float("-inf"), []

            resp.raise_for_status()
            data = resp.json()

            # Extract response text
            choice = data["choices"][0]
            response_text = choice["message"]["content"].strip()

            # Extract first-token logprobs
            logprobs_data = choice.get("logprobs", {})
            content_logprobs = logprobs_data.get("content", [])

            if not content_logprobs:
                # API returned no logprobs (unexpected)
                return response_text, float("-inf"), float("-inf"), []

            first_token = content_logprobs[0]
            top_logprobs = first_token.get("top_logprobs", [])

            # Search for yes/no variants in top_logprobs
            yes_logp = float("-inf")
            no_logp = float("-inf")

            for entry in top_logprobs:
                token = entry["token"].strip().lower()
                logp = entry["logprob"]

                if token in ("yes", "y"):
                    yes_logp = max(yes_logp, logp)
                elif token in ("no", "n"):
                    no_logp = max(no_logp, logp)

            # Floor -inf to -20.0 to avoid NaN/inf gap when yes/no
            # tokens are missing from top-5 logprobs
            if yes_logp == float("-inf"):
                yes_logp = -20.0
            if no_logp == float("-inf"):
                no_logp = -20.0

            return response_text, yes_logp, no_logp, top_logprobs

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"      [API] Timeout, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      [API] Timeout after {max_retries} attempts")
                return "ERROR:timeout", float("-inf"), float("-inf"), []

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = max(2 ** attempt, 2)
                print(f"      [API] Error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"      [API] Failed after {max_retries} attempts: {e}")
                return f"ERROR:{e}", float("-inf"), float("-inf"), []

        except (KeyError, IndexError) as e:
            # Malformed response
            print(f"      [API] Malformed response: {e}")
            return "ERROR:malformed", float("-inf"), float("-inf"), []


def gpt4o_answer(img_b64, prompt, max_retries=6, timeout=60):
    """Call GPT-4o API to generate an answer (no logprobs needed).

    Used when a trigger fires — generates the actual answer content.
    Longer max_tokens (256) and timeout for full generation.
    """
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 256,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, json=payload, headers=API_HEADERS, timeout=timeout
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                if attempt < max_retries - 1:
                    time.sleep(max(retry_after, 1))
                    continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(max(2 ** attempt, 2))
            else:
                return f"ERROR:{e}"


# ═══════════════════════════════════════════════════════════
#  FBF Loop (GPT-4o version)
# ═══════════════════════════════════════════════════════════

def run_gpt4o_fbf(video_path, question, start_time, end_time,
                   frame_fps=0.175, verbose=False):
    """Run FBF polling loop with GPT-4o, collecting logprob data.

    At each polling step:
      1. Extract single frame at current_time
      2. Encode as base64 JPEG
      3. Call GPT-4o with trigger prompt + logprobs=True
      4. Extract yes/no logprobs from top_logprobs
      5. Record gap = logP(yes) - logP(no)
      6. If "yes" in response (vanilla FBF rule): generate answer

    This collects full logprob data for offline threshold sweeps, same
    as fullscale_d_runner.py does for Qwen3-VL.

    Returns:
        dialog_history: [{role, content, time}] for ESTP-F1 scoring
        trigger_log: [{time, yes_logp, no_logp, logprob_gap, ...}]
    """
    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)

    step = 0
    while current_time <= end_time:
        step += 1

        # Extract single frame
        frame = extract_single_frame(video_path, current_time)
        if frame is None:
            current_time += 1 / frame_fps
            continue

        img_b64 = pil_to_base64(frame)

        # API call: trigger check with logprobs
        t0 = time.time()
        response, yes_logp, no_logp, top_lp = gpt4o_trigger_with_logprobs(
            img_b64, trigger_prompt
        )
        cost_trigger = time.time() - t0

        logprob_gap = yes_logp - no_logp
        triggered = "yes" in response.lower()

        entry = {
            "time": round(current_time, 3),
            "response": response[:100],
            "triggered": triggered,
            "yes_logp": yes_logp,
            "no_logp": no_logp,
            "logprob_gap": logprob_gap,
            "cost_trigger": round(cost_trigger, 3),
            "top_logprobs": [
                {"token": t["token"], "logprob": t["logprob"]}
                for t in (top_lp or [])[:5]  # keep top-5 for analysis
            ],
        }

        if verbose:
            status = "TRIGGERED" if triggered else "suppress"
            print(f"    [GPT4o] t={current_time:.1f}s "
                  f"gap={logprob_gap:+.2f} "
                  f"yes={yes_logp:.2f} no={no_logp:.2f} "
                  f"{status} ({cost_trigger:.1f}s)")

        # If triggered, generate answer
        if triggered:
            t0 = time.time()
            answer = gpt4o_answer(img_b64, answer_prompt)
            cost_answer = time.time() - t0
            entry["cost_answer"] = round(cost_answer, 3)
            entry["answer"] = answer[:200]

            dialog_history.append({
                "role": "user",
                "content": "trigger",
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost_trigger + cost_answer,
            })

        trigger_log.append(entry)
        current_time += 1 / frame_fps

    return dialog_history, trigger_log


# ═══════════════════════════════════════════════════════════
#  Offline Simulation (no API needed)
# ═══════════════════════════════════════════════════════════

def simulate_threshold_offline(trigger_log, qa_raw, tau, cooldown_sec=0.0):
    """Re-simulate a (tau, cooldown) config from saved logprob data.

    Same logic as fullscale_d_runner.simulate_config_offline().
    Returns: (trigger_times, metrics_dict)
    """
    trigger_times = []
    last_trigger = -float("inf")

    for entry in trigger_log:
        t = entry["time"]
        gap = entry["logprob_gap"]
        if gap > tau and (t - last_trigger) >= cooldown_sec:
            trigger_times.append(t)
            last_trigger = t

    dialog = [{"role": "assistant", "content": "sim", "time": t}
              for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    return trigger_times, metrics


def reconstruct_baseline(trigger_log, qa_raw):
    """Reconstruct vanilla FBF baseline: trigger when gap > 0 (text says yes)."""
    trigger_times = [e["time"] for e in trigger_log if e["logprob_gap"] > 0]
    dialog = [{"role": "assistant", "content": "sim", "time": t}
              for t in trigger_times]
    metrics = compute_case_metrics(dialog, qa_raw)
    return trigger_times, metrics


# ═══════════════════════════════════════════════════════════
#  Checkpoint I/O (JSONL, same pattern as fullscale_d_runner)
# ═══════════════════════════════════════════════════════════

def get_checkpoint_path():
    return OUTPUT_DIR / "checkpoint.jsonl"


def load_checkpoint():
    """Load completed cases from JSONL checkpoint.

    Returns: (completed_keys: set, results: list)
    """
    cp = get_checkpoint_path()
    if not cp.exists():
        return set(), []

    completed = set()
    results = []
    with open(cp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = f"{r['video_uid']}_{r['clip_uid']}_{r['qa_id']}"
                completed.add(key)
                results.append(r)
            except json.JSONDecodeError:
                pass
    return completed, results


def save_checkpoint(result):
    """Atomically append one case result to JSONL checkpoint."""
    cp = get_checkpoint_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def case_key(case):
    return f"{case['video_uid']}_{case['clip_uid']}_{case['qa_id']}"


# ═══════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════

def generate_report(results):
    """Generate comprehensive yes-bias analysis report.

    Outputs:
      - Overall yes-rate and gap distribution for GPT-4o
      - GT vs non-GT gap separation (discriminative power)
      - Per-task-type breakdown
      - Comparison reference to Qwen3-VL known values
    """
    valid = [r for r in results if r.get("status") == "ok"]
    n = len(valid)

    lines = []
    lines.append("=" * 80)
    lines.append("GPT-4o FBF Yes-Bias Pilot Test Report")
    lines.append("=" * 80)
    lines.append(f"\nCases: {n} valid (of {len(results)} total)")
    lines.append(f"Polling rate: 0.175 Hz (~5.7s per step)")
    lines.append(f"API: Azure OpenAI GPT-4o @ {API_URL}")

    if n == 0:
        lines.append("\nNo valid results found.")
        return "\n".join(lines)

    # ── Collect all logprob data ──
    all_gaps = []
    gt_gaps = []
    non_gt_gaps = []
    total_steps = 0
    total_yes = 0
    total_api_time = 0

    by_type = defaultdict(lambda: {
        "gaps": [], "gt_gaps": [], "non_gt_gaps": [],
        "steps": 0, "yes": 0, "n_cases": 0,
    })

    for r in valid:
        qa_raw = r["qa_raw"]
        clip_end = qa_raw.get("clip_end_time", qa_raw.get("end_time", 0))
        gold = [c for c in qa_raw["conversation"]
                if c.get("role", "").lower() == "assistant"]
        gt_spans = [
            (ceil_time_by_fps(c["start_time"], 2, 0, clip_end),
             ceil_time_by_fps(c["end_time"], 2, 0, clip_end))
            for c in gold
        ]

        task_type = r["task_type"]
        by_type[task_type]["n_cases"] += 1

        for entry in r["trigger_log"]:
            t = entry["time"]
            gap = entry["logprob_gap"]
            triggered = entry.get("triggered", False)

            total_steps += 1
            total_api_time += entry.get("cost_trigger", 0)
            if triggered:
                total_yes += 1
            by_type[task_type]["steps"] += 1
            if triggered:
                by_type[task_type]["yes"] += 1

            # Skip non-finite gaps (from API errors or missing tokens)
            if not math.isfinite(gap):
                continue

            all_gaps.append(gap)
            by_type[task_type]["gaps"].append(gap)

            in_gt = any(gs - ANTICIPATION <= t <= ge + LATENCY
                        for gs, ge in gt_spans)
            if in_gt:
                gt_gaps.append(gap)
                by_type[task_type]["gt_gaps"].append(gap)
            else:
                non_gt_gaps.append(gap)
                by_type[task_type]["non_gt_gaps"].append(gap)

    # ── Overall Yes-Bias ──
    yes_rate = total_yes / total_steps if total_steps > 0 else 0
    lines.append(f"\n{'─' * 80}")
    lines.append("OVERALL YES-BIAS ANALYSIS")
    lines.append(f"{'─' * 80}")
    lines.append(f"  Total polling steps: {total_steps}")
    lines.append(f"  Total 'yes' triggers: {total_yes}")
    lines.append(f"  Yes-rate: {yes_rate:.1%}")
    lines.append(f"  Total API time: {total_api_time:.0f}s "
                 f"(avg {total_api_time/total_steps:.1f}s/step)")
    lines.append(f"")
    lines.append(f"  Reference: Qwen3-VL yes-rate = 76-83%")
    if yes_rate > 0.60:
        lines.append(f"  >>> GPT-4o ALSO shows yes-bias ({yes_rate:.0%} > 60%)")
    elif yes_rate > 0.40:
        lines.append(f"  >>> GPT-4o shows moderate yes-bias ({yes_rate:.0%})")
    else:
        lines.append(f"  >>> GPT-4o does NOT show strong yes-bias ({yes_rate:.0%})")

    # ── Gap Distribution ──
    lines.append(f"\n{'─' * 80}")
    lines.append("LOGPROB GAP DISTRIBUTION")
    lines.append(f"{'─' * 80}")
    lines.append(f"  gap = logP(yes) - logP(no)")
    lines.append(f"  gap > 0 means model favors 'yes'")
    lines.append(f"")
    lines.append(f"  All gaps (n={len(all_gaps)}):")
    lines.append(f"    mean={np.mean(all_gaps):.3f}, "
                 f"std={np.std(all_gaps):.3f}, "
                 f"median={np.median(all_gaps):.3f}")
    lines.append(f"    min={np.min(all_gaps):.3f}, max={np.max(all_gaps):.3f}")

    if gt_gaps:
        lines.append(f"  GT-window gaps (n={len(gt_gaps)}):")
        lines.append(f"    mean={np.mean(gt_gaps):.3f}, "
                     f"std={np.std(gt_gaps):.3f}")
    if non_gt_gaps:
        lines.append(f"  Non-GT gaps (n={len(non_gt_gaps)}):")
        lines.append(f"    mean={np.mean(non_gt_gaps):.3f}, "
                     f"std={np.std(non_gt_gaps):.3f}")

    if gt_gaps and non_gt_gaps:
        sep = np.mean(gt_gaps) - np.mean(non_gt_gaps)
        lines.append(f"  Separation (GT - non-GT mean): {sep:+.3f}")
        lines.append(f"  Reference: Qwen3-VL separation = +1.418 (GT=2.226, non-GT=0.808)")

        # Welch's t-test for significance
        n_gt, n_ngt = len(gt_gaps), len(non_gt_gaps)
        if n_gt >= 2 and n_ngt >= 2:
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(gt_gaps, non_gt_gaps, equal_var=False)
            lines.append(f"  Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.001:
                lines.append(f"  >>> Gap is discriminative (p<0.001)")
            elif p_val < 0.05:
                lines.append(f"  >>> Gap is weakly discriminative (p<0.05)")
            else:
                lines.append(f"  >>> Gap is NOT discriminative (p>={p_val:.3f})")

    # Threshold coverage
    lines.append(f"\n  Threshold coverage:")
    for thr in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        pct = sum(1 for g in all_gaps if g > thr) / len(all_gaps)
        lines.append(f"    gap > {thr:.1f}: {pct:.1%} of frames")

    # ── Per-Task-Type ──
    lines.append(f"\n{'─' * 80}")
    lines.append("PER-TASK-TYPE ANALYSIS")
    lines.append(f"{'─' * 80}")
    lines.append(f"  {'Task Type':<40} {'n':>3} {'steps':>6} "
                 f"{'yes%':>6} {'gap_mean':>9} {'gt_gap':>8} {'ngt_gap':>8}")
    lines.append("  " + "-" * 81)

    for tt in sorted(by_type.keys()):
        td = by_type[tt]
        n_c = td["n_cases"]
        n_s = td["steps"]
        yr = td["yes"] / n_s if n_s > 0 else 0
        gm = np.mean(td["gaps"]) if td["gaps"] else float("nan")
        gt_m = np.mean(td["gt_gaps"]) if td["gt_gaps"] else float("nan")
        ngt_m = np.mean(td["non_gt_gaps"]) if td["non_gt_gaps"] else float("nan")
        lines.append(
            f"  {tt:<40} {n_c:>3} {n_s:>6} {yr:>5.0%} "
            f"{gm:>9.3f} {gt_m:>8.3f} {ngt_m:>8.3f}"
        )

    # ── ESTP-F1 Comparison ──
    lines.append(f"\n{'─' * 80}")
    lines.append("ESTP-F1 (VANILLA FBF BASELINE)")
    lines.append(f"{'─' * 80}")

    bl_f1s = []
    bl_fps_list = []
    for r in valid:
        qa_raw = r["qa_raw"]
        bl_times, bl_m = reconstruct_baseline(r["trigger_log"], qa_raw)
        bl_f1s.append(bl_m["f1"])
        bl_fps_list.append(bl_m["false_positives"])
        lines.append(f"  {r['task_type']:<30} F1={bl_m['f1']:.3f} "
                     f"P={bl_m['precision']:.3f} R={bl_m['recall']:.3f} "
                     f"FP={bl_m['false_positives']}")

    if bl_f1s:
        lines.append(f"\n  Avg F1: {np.mean(bl_f1s):.3f}, "
                     f"Avg FP: {np.mean(bl_fps_list):.1f}")
        lines.append(f"  Reference: Qwen3-VL FBF F1=0.127 (60 cases)")

    # ── Threshold Sweep (offline) ──
    lines.append(f"\n{'─' * 80}")
    lines.append("OFFLINE THRESHOLD SWEEP (GPT-4o)")
    lines.append(f"{'─' * 80}")
    lines.append(f"  {'tau':>5} {'cd':>4} {'F1':>7} {'Prec':>7} {'Rec':>7} "
                 f"{'FP':>5} {'Trig':>5}")
    lines.append("  " + "-" * 46)

    for tau in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for cd in [0.0, 12.0]:
            sweep_f1s = []
            sweep_fps = []
            sweep_rec = []
            sweep_prec = []
            sweep_trig = []
            for r in valid:
                _, m = simulate_threshold_offline(
                    r["trigger_log"], r["qa_raw"], tau, cd
                )
                sweep_f1s.append(m["f1"])
                sweep_fps.append(m["false_positives"])
                sweep_rec.append(m["recall"])
                sweep_prec.append(m["precision"])
                sweep_trig.append(m["n_preds"])

            lines.append(
                f"  {tau:>5.1f} {cd:>4.0f} "
                f"{np.mean(sweep_f1s):>7.3f} "
                f"{np.mean(sweep_prec):>7.3f} "
                f"{np.mean(sweep_rec):>7.3f} "
                f"{np.sum(sweep_fps):>5.0f} "
                f"{np.sum(sweep_trig):>5.0f}"
            )

    lines.append(f"\n{'=' * 80}")
    lines.append(f"Report generated: {n} cases")
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GPT-4o FBF Yes-Bias Pilot Test"
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of cases to test (default: 5 pilot)")
    parser.add_argument("--frame_fps", type=float, default=0.175,
                        help="Polling rate in Hz (default: 0.175, ~5.7s/step)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--report_only", action="store_true",
                        help="Generate report from existing checkpoint (no API calls)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──
    print(f"Loading: {DATA_5CASES}")
    with open(DATA_5CASES) as f:
        data = json.load(f)

    cases = flatten_dataset_to_cases(data)
    print(f"Found {len(cases)} total cases")

    # Sort by video for consistency
    cases.sort(key=lambda c: (c["video_uid"], c["start_time"]))

    if args.limit > 0:
        cases = cases[:args.limit]
        print(f"Pilot: using {len(cases)} cases")

    # ── Report-only mode ──
    if args.report_only:
        _, results = load_checkpoint()
        if not results:
            print("ERROR: No checkpoint found. Run without --report_only first.")
            return
        print(f"Generating report from {len(results)} checkpoint entries...")
        report = generate_report(results)
        rp = OUTPUT_DIR / "gpt4o_fbf_report.txt"
        rp.write_text(report)
        print(report)
        print(f"\nSaved: {rp}")
        return

    # ── Check API connectivity ──
    print(f"\nTesting API connectivity: {API_URL[:80]}...")
    if not _API_KEY:
        print("  ERROR: No API key found. Set AZURE_OPENAI_API_KEY in .env")
        return
    try:
        test_payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        resp = requests.post(API_URL, json=test_payload, headers=API_HEADERS, timeout=15)
        if resp.status_code == 200:
            print(f"  API OK (model={resp.json().get('model', '?')})")
        elif resp.status_code == 429:
            print(f"  API reachable (rate-limited, will retry)")
        else:
            print(f"  WARNING: status={resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  WARNING: API check failed: {e}")
        print(f"  Proceeding anyway (may fail on first call)")

    # ── Checkpoint / resume ──
    completed_keys, prev_results = (
        load_checkpoint() if args.resume else (set(), [])
    )
    if args.resume:
        print(f"Resuming: {len(completed_keys)} already done")

    # ── Main loop ──
    all_results = list(prev_results)
    t_run_start = time.time()

    for i, case in enumerate(cases):
        ck = case_key(case)
        if ck in completed_keys:
            print(f"[{i+1:02d}/{len(cases)}] SKIP (done): {case['task_type']}")
            continue

        print(f"\n[{i+1:02d}/{len(cases)}] {case['task_type']}: "
              f"{case['question'][:60]}...")
        print(f"  video={case['video_uid'][:12]}... "
              f"t=[{case['start_time']:.0f}s, {case['end_time']:.0f}s]")

        # Find video
        video_path = find_video_file(case["video_uid"], case["clip_uid"])
        if not video_path:
            print(f"  SKIP: video not found")
            r = {k: case[k] for k in ("video_uid", "clip_uid", "qa_id",
                                       "task_type", "question")}
            r["status"] = "video_not_found"
            save_checkpoint(r)
            all_results.append(r)
            completed_keys.add(ck)
            continue

        qa_raw = case["qa_raw"]
        gold = [c for c in qa_raw.get("conversation", [])
                if c.get("role", "").lower() == "assistant"]
        if not gold:
            print(f"  SKIP: no GT")
            r = {k: case[k] for k in ("video_uid", "clip_uid", "qa_id",
                                       "task_type", "question")}
            r["status"] = "no_gt"
            save_checkpoint(r)
            all_results.append(r)
            completed_keys.add(ck)
            continue

        duration = case["end_time"] - case["start_time"]
        n_steps_est = int(duration * args.frame_fps) + 1
        print(f"  dur={duration:.0f}s, GT_windows={len(gold)}, "
              f"~{n_steps_est} polling steps")

        # ── Run FBF ──
        t0 = time.time()
        dialog, trigger_log = run_gpt4o_fbf(
            video_path, case["question"],
            case["start_time"], case["end_time"],
            frame_fps=args.frame_fps, verbose=args.verbose,
        )
        wall_time = time.time() - t0

        # Compute metrics
        metrics = compute_case_metrics(dialog, qa_raw)
        n_polls = len(trigger_log)
        n_yes = sum(1 for e in trigger_log if e.get("triggered", False))
        yes_rate = n_yes / n_polls if n_polls > 0 else 0

        # Gap stats
        gaps = [e["logprob_gap"] for e in trigger_log
                if e["logprob_gap"] != float("-inf") and e["logprob_gap"] != float("inf")]
        gap_mean = np.mean(gaps) if gaps else float("nan")

        print(f"  Result: F1={metrics['f1']:.3f}, "
              f"preds={metrics['n_preds']}, "
              f"FP={metrics['false_positives']}, "
              f"yes_rate={yes_rate:.0%}, "
              f"gap_mean={gap_mean:.2f}")
        print(f"  Time: {wall_time:.0f}s ({wall_time/n_polls:.1f}s/step)")

        # ── Save checkpoint ──
        result = {
            "video_uid": case["video_uid"],
            "clip_uid": case["clip_uid"],
            "qa_id": case["qa_id"],
            "task_type": case["task_type"],
            "question": case["question"],
            "status": "ok",
            "duration": duration,
            "n_gt": len(gold),
            "n_polls": n_polls,
            "n_yes": n_yes,
            "yes_rate": yes_rate,
            "gap_mean": gap_mean,
            "wall_time": wall_time,
            "metrics": metrics,
            "qa_raw": qa_raw,
            "trigger_log": trigger_log,
        }
        save_checkpoint(result)
        all_results.append(result)
        completed_keys.add(ck)

        # ETA
        done_ok = sum(1 for r in all_results if r.get("status") == "ok")
        remaining = sum(1 for c in cases if case_key(c) not in completed_keys)
        elapsed = time.time() - t_run_start
        eta = (elapsed / max(done_ok, 1)) * remaining
        print(f"  [{done_ok} done, {remaining} left, "
              f"ETA ~{eta/60:.0f}min]")

    # ── Final report ──
    print(f"\n{'='*60}")
    print("COMPLETE - Generating report")
    print(f"{'='*60}")

    report = generate_report(all_results)

    rp = OUTPUT_DIR / "gpt4o_fbf_report.txt"
    rp.write_text(report)
    print(report)
    print(f"\nReport: {rp}")
    print(f"Checkpoint: {get_checkpoint_path()}")


if __name__ == "__main__":
    main()
