#!/usr/bin/env python3
"""
Reasoning Trigger Validation on ProAssist WTAG Data.

Purpose:
    Use GPT-4o vision API to perform "reasoning trigger" evaluation on WTAG
    cooking/assembly videos. For sampled frames, GPT-4o judges whether the user
    needs proactive help. Results are compared against:
    1. Ground truth talk labels (from WTAG dialogues)
    2. ProAssist w2t_prob (learned when-to-talk probability)

Background:
    On ESTP-Bench, reasoning trigger (F1=0.173) performed worse than baseline
    (0.225) and Adaptive-D (0.298), except on text-rich tasks (0.667 vs 0.154).
    WTAG is a task-oriented dialogue dataset (cooking, assembly) where reasoning
    about task state may be more valuable.

API:
    GPT-4o via Azure OpenAI: http://52.151.57.21:9999
    Supports vision (base64 images) and logprobs.

Usage:
    cd /home/v-tangxin/GUI
    source proassist_env/bin/activate

    # Run full experiment (5 samples, ~100 API calls)
    python3 proassist_experiments/reasoning_trigger_proassist.py

    # Report only from existing results
    python3 proassist_experiments/reasoning_trigger_proassist.py --report_only

    # Custom settings
    python3 proassist_experiments/reasoning_trigger_proassist.py --num_samples 3 --frame_step 15
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import requests
try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback: manual .env loading
    def load_dotenv(path):
        if Path(path).exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        val = val.strip().strip('"').strip("'")
                        os.environ.setdefault(key.strip(), val)

# ─── Path Setup ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # /home/v-tangxin/GUI
PROASSIST_ROOT = PROJECT_ROOT / "temp" / "ProAssist"
DATA_ROOT = PROJECT_ROOT / "data" / "ProAssist"
WTAG_FRAMES_DIR = DATA_ROOT / "processed_data" / "wtag" / "frames"
WTAG_PREPARED_DIR = DATA_ROOT / "processed_data" / "wtag" / "prepared"
EVAL_DIR = DATA_ROOT / "ProAssist-Model-L4096-I1" / "eval" / "wtag-dialog-klg-sum_val_L4096_I1" / "stream"
OUTPUT_DIR = SCRIPT_DIR / "results" / "reasoning_trigger"

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# ─── Azure OpenAI API Config ───
_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT", "http://52.151.57.21:9999").rstrip("/")
_API_VERSION = os.getenv("api_version", "2024-09-01-preview")
_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("api_key", ""))
API_URL = f"{_API_BASE}/openai/deployments/gpt-4o/chat/completions?api-version={_API_VERSION}"
API_HEADERS = {
    "Content-Type": "application/json",
    "api-key": _API_KEY,
}

# ─── Sample Selection ───
# Pick one sample per unique video, covering different tasks
# Sample 0: T48 (pour-over coffee), Sample 6: T54 (pour-over coffee, different video)
# Sample 12: T6 (mug cake), Sample 15: T47 (pinwheels), Sample 18: T7 (pinwheels, different video)
# Also include Sample 21: T9 (mug cake, different video)
SELECTED_SAMPLES = [0, 6, 12, 15, 18, 21]

# ─── Reasoning Prompt ───
REASONING_PROMPT = """You are observing a first-person video of someone performing a cooking or assembly task. An AI assistant should proactively speak to help them when appropriate.

Task context: {task_goal}

Current task knowledge:
{knowledge}

Based on this video frame, determine whether the AI assistant should speak NOW to help the user.

Consider:
- Is the user at a transition between steps? (likely needs guidance)
- Is the user struggling or hesitating? (likely needs encouragement or correction)
- Is the user doing something wrong? (needs a warning)
- Is the user actively performing a step correctly? (no interruption needed)
- Has the user just completed a step? (may need next step instruction)

Respond in JSON format ONLY:
{{"action": "<what the user is currently doing>", "needs_help": "<yes/no/uncertain>", "help_type": "<instruction/warning/encouragement/confirmation/none>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

REASONING_PROMPT_WITH_HISTORY = """You are observing a first-person video of someone performing a cooking or assembly task. An AI assistant should proactively speak to help them when appropriate.

Task context: {task_goal}

Current task knowledge:
{knowledge}

Recent conversation context:
{recent_dialog}

Based on this video frame and the conversation history, determine whether the AI assistant should speak NOW to help the user.

Consider:
- Is the user at a transition between steps? (likely needs guidance)
- Is the user struggling or hesitating? (likely needs encouragement or correction)
- Is the user doing something wrong? (needs a warning)
- Is the user actively performing a step correctly? (no interruption needed)
- Has the user just completed a step? (may need next step instruction)
- Has the assistant recently spoken? (avoid repeating too soon)

Respond in JSON format ONLY:
{{"action": "<what the user is currently doing>", "needs_help": "<yes/no/uncertain>", "help_type": "<instruction/warning/encouragement/confirmation/none>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


# ═══════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════

def load_wtag_sample(sample_idx: int) -> dict:
    """Load a WTAG validation sample with metadata and conversation."""
    data_file = WTAG_PREPARED_DIR / "dialog-klg-sum_val_L4096_I1.jsonl"
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    raise ValueError(f"Sample {sample_idx} not found")


def load_frames(video_uid: str) -> list:
    """Load all frames for a video from arrow file. Returns list of base64 strings."""
    arrow_path = WTAG_FRAMES_DIR / f"{video_uid}.arrow"
    reader = pa.ipc.open_stream(str(arrow_path))
    table = reader.read_all()
    return [table.column(0)[i].as_py() for i in range(table.num_rows)]


def load_eval_result(sample_idx: int, threshold: float = 0.2) -> dict:
    """Load ProAssist stream evaluation result (contains w2t_prob per frame)."""
    eval_file = EVAL_DIR / f"notalk{threshold}-maxlen_4k" / "results" / f"{sample_idx}.json"
    if not eval_file.exists():
        # Try 0.3 threshold as fallback
        eval_file = EVAL_DIR / "notalk0.3-maxlen_4k" / "results" / f"{sample_idx}.json"
    with open(eval_file) as f:
        return json.load(f)


def extract_ground_truth(sample: dict, eval_result: dict) -> dict:
    """Extract ground truth talk frames and dialogue context.

    Returns dict with:
    - talk_frames: set of frame indices where assistant speaks
    - talk_content: dict mapping frame_idx -> assistant text
    - dialog_before: dict mapping frame_idx -> list of recent dialog items
    """
    predictions = eval_result["predictions"]

    talk_frames = set()
    talk_content = {}

    for i, pred in enumerate(predictions):
        if pred["ref"]:  # Has reference text = ground truth talk
            talk_frames.add(i)
            talk_content[i] = pred["ref"]

    return {
        "talk_frames": talk_frames,
        "talk_content": talk_content,
        "total_frames": len(predictions),
    }


def get_dialog_context(sample: dict, frame_idx: int, fps: int = 2) -> str:
    """Extract recent dialog before the given frame index."""
    timestamp = frame_idx / fps
    recent = []

    for conv in sample["conversation"]:
        role = conv.get("role", "")
        if role in ("assistant", "user"):
            t = conv.get("time", 0)
            if t < timestamp:
                content = conv.get("content", "")[:100]
                recent.append(f"[{role} @ {t:.1f}s]: {content}")

    # Keep last 3 turns
    return "\n".join(recent[-3:]) if recent else "(no previous dialog)"


# ═══════════════════════════════════════════════════════════
#  GPT-4o Vision API
# ═══════════════════════════════════════════════════════════

def call_gpt4o_vision(frame_b64: str, prompt: str, max_retries: int = 3) -> dict:
    """Call GPT-4o vision API with a single frame.

    Returns parsed JSON response or error dict.
    """
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "low",  # Save tokens, sufficient for scene understanding
                        },
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1,  # Low temperature for consistent judgments
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                API_URL, headers=API_HEADERS, json=payload, timeout=30
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle cases where model wraps in markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())
            parsed["_raw"] = result["choices"][0]["message"]["content"]
            parsed["_tokens"] = result.get("usage", {})
            return parsed

        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract key fields
            raw = result["choices"][0]["message"]["content"]
            return {
                "action": "unknown",
                "needs_help": "uncertain",
                "help_type": "none",
                "confidence": 0.5,
                "reasoning": f"JSON parse error: {raw[:200]}",
                "_raw": raw,
                "_parse_error": True,
            }
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"  Request error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {
                    "action": "error",
                    "needs_help": "uncertain",
                    "help_type": "none",
                    "confidence": 0.0,
                    "reasoning": f"API error after {max_retries} attempts: {str(e)}",
                    "_error": True,
                }
        except Exception as e:
            return {
                "action": "error",
                "needs_help": "uncertain",
                "help_type": "none",
                "confidence": 0.0,
                "reasoning": f"Unexpected error: {str(e)}",
                "_error": True,
            }

    return {
        "action": "error",
        "needs_help": "uncertain",
        "help_type": "none",
        "confidence": 0.0,
        "reasoning": "Max retries exceeded",
        "_error": True,
    }


# ═══════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════

def run_sample(sample_idx: int, frame_step: int = 10, verbose: bool = False,
               stratified: bool = True, max_frames: int = 20) -> dict:
    """Run reasoning trigger on a single WTAG sample.

    Args:
        sample_idx: Index in val JSONL file
        frame_step: Sample every N frames for uniform mode
        verbose: Print per-frame details
        stratified: Use stratified sampling (half talk, half notalk)
        max_frames: Max frames to evaluate per sample

    Returns:
        Dict with per-frame results, ground truth, and w2t_prob
    """
    print(f"\n{'='*60}")
    print(f"Processing sample {sample_idx}...")

    # Load data
    sample = load_wtag_sample(sample_idx)
    eval_result = load_eval_result(sample_idx)

    video_uid = sample["video_uid"]
    task_goal = sample["metadata"]["task_goal"]
    knowledge = sample["metadata"]["knowledge"]
    fps = sample["fps"]
    start_idx = sample["start_frame_idx"]
    end_idx = sample["end_frame_idx"]

    print(f"  Video: {video_uid}, Task: {task_goal}")
    print(f"  Frames: {start_idx}-{end_idx} ({end_idx - start_idx} total), FPS: {fps}")

    # Load frames
    all_frames = load_frames(video_uid)

    # Extract ground truth
    gt = extract_ground_truth(sample, eval_result)
    print(f"  Ground truth talk frames: {len(gt['talk_frames'])}/{gt['total_frames']}")

    # Get w2t_prob for all frames
    predictions = eval_result["predictions"]
    total_frames = end_idx - start_idx

    if stratified:
        # STRATIFIED SAMPLING: half from talk frames, half from notalk frames
        # This ensures we actually test on both classes
        import random
        random.seed(42 + sample_idx)

        talk_indices = sorted(gt["talk_frames"])
        notalk_indices = sorted(set(range(total_frames)) - gt["talk_frames"])

        n_talk = min(max_frames // 2, len(talk_indices))
        n_notalk = min(max_frames - n_talk, len(notalk_indices))

        # Sample talk frames (spread evenly)
        if n_talk < len(talk_indices):
            step_t = len(talk_indices) // n_talk
            sampled_talk = [talk_indices[i * step_t] for i in range(n_talk)]
        else:
            sampled_talk = talk_indices

        # Sample notalk frames (spread evenly)
        if n_notalk < len(notalk_indices):
            step_nt = len(notalk_indices) // n_notalk
            sampled_notalk = [notalk_indices[i * step_nt] for i in range(n_notalk)]
        else:
            sampled_notalk = notalk_indices[:n_notalk]

        eval_frame_indices = sorted(sampled_talk + sampled_notalk)
        print(f"  Stratified sampling: {n_talk} talk + {n_notalk} notalk = {len(eval_frame_indices)} frames")
    else:
        # Uniform sampling
        eval_frame_indices = list(range(0, total_frames, frame_step))
        if len(eval_frame_indices) > max_frames:
            step = len(eval_frame_indices) // max_frames
            eval_frame_indices = eval_frame_indices[::step][:max_frames]
        print(f"  Evaluating {len(eval_frame_indices)} frames (step={frame_step})")

    results = []
    api_calls = 0

    for frame_idx in eval_frame_indices:
        actual_frame_idx = start_idx + frame_idx

        if actual_frame_idx >= len(all_frames):
            continue

        # Get frame
        frame_b64 = all_frames[actual_frame_idx]

        # Get dialog context
        dialog_context = get_dialog_context(sample, frame_idx, fps)

        # Build prompt
        if dialog_context != "(no previous dialog)":
            prompt = REASONING_PROMPT_WITH_HISTORY.format(
                task_goal=task_goal,
                knowledge=knowledge[:500],  # Truncate long knowledge
                recent_dialog=dialog_context,
            )
        else:
            prompt = REASONING_PROMPT.format(
                task_goal=task_goal,
                knowledge=knowledge[:500],
            )

        # Call GPT-4o
        gpt_result = call_gpt4o_vision(frame_b64, prompt)
        api_calls += 1

        # Get ground truth and w2t_prob for this frame
        is_talk_gt = frame_idx in gt["talk_frames"]
        w2t_prob = predictions[frame_idx]["w2t_prob"] if frame_idx < len(predictions) else None

        result = {
            "frame_idx": frame_idx,
            "actual_frame_idx": actual_frame_idx,
            "timestamp": frame_idx / fps,
            "is_talk_gt": is_talk_gt,
            "gt_content": gt["talk_content"].get(frame_idx, ""),
            "w2t_prob": w2t_prob,
            "gpt_needs_help": gpt_result.get("needs_help", "uncertain"),
            "gpt_confidence": gpt_result.get("confidence", 0.5),
            "gpt_help_type": gpt_result.get("help_type", "none"),
            "gpt_action": gpt_result.get("action", "unknown"),
            "gpt_reasoning": gpt_result.get("reasoning", ""),
            "_parse_error": gpt_result.get("_parse_error", False),
            "_error": gpt_result.get("_error", False),
        }
        results.append(result)

        if verbose:
            talk_marker = "TALK" if is_talk_gt else "    "
            help_marker = gpt_result.get("needs_help", "?")
            conf = gpt_result.get("confidence", 0)
            w2t = w2t_prob if w2t_prob is not None else -1
            print(f"  [{talk_marker}] Frame {frame_idx:4d} ({frame_idx/fps:6.1f}s): "
                  f"GPT={help_marker:11s} conf={conf:.2f} | w2t={w2t:.4f} | "
                  f"action={gpt_result.get('action', '?')[:40]}")

        # Rate limit: ~2 calls per second
        time.sleep(0.5)

    return {
        "sample_idx": sample_idx,
        "video_uid": video_uid,
        "task_goal": task_goal,
        "total_frames": total_frames,
        "gt_talk_count": len(gt["talk_frames"]),
        "eval_frames": len(results),
        "api_calls": api_calls,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════

def compute_metrics(all_results: list, threshold_conf: float = 0.5) -> dict:
    """Compute precision, recall, F1 for GPT-4o reasoning trigger.

    A frame is predicted as "talk" if needs_help == "yes" AND confidence >= threshold.
    """
    tp, fp, fn, tn = 0, 0, 0, 0

    for sample in all_results:
        for r in sample["results"]:
            if r["_error"]:
                continue

            is_gt_talk = r["is_talk_gt"]
            is_pred_talk = (
                r["gpt_needs_help"] == "yes" and
                r["gpt_confidence"] >= threshold_conf
            )

            if is_gt_talk and is_pred_talk:
                tp += 1
            elif not is_gt_talk and is_pred_talk:
                fp += 1
            elif is_gt_talk and not is_pred_talk:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold_conf,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": tp + fp + fn + tn,
        "accuracy": (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0,
    }


def compute_w2t_comparison(all_results: list) -> dict:
    """Compare GPT-4o confidence with w2t_prob."""
    pairs = []

    for sample in all_results:
        for r in sample["results"]:
            if r["_error"] or r["w2t_prob"] is None:
                continue

            # w2t_prob is probability of NOT talking (high = don't talk)
            # So 1 - w2t_prob = probability of talking
            talk_prob = 1 - r["w2t_prob"]
            gpt_conf = r["gpt_confidence"] if r["gpt_needs_help"] == "yes" else (1 - r["gpt_confidence"])

            pairs.append({
                "talk_prob": talk_prob,
                "gpt_talk_conf": gpt_conf,
                "is_talk_gt": r["is_talk_gt"],
                "w2t_prob": r["w2t_prob"],
                "gpt_needs_help": r["gpt_needs_help"],
                "gpt_confidence": r["gpt_confidence"],
            })

    if not pairs:
        return {"correlation": 0, "n": 0}

    # Correlation between talk_prob and gpt_talk_conf
    talk_probs = np.array([p["talk_prob"] for p in pairs])
    gpt_confs = np.array([p["gpt_talk_conf"] for p in pairs])

    if len(talk_probs) > 1 and np.std(talk_probs) > 0 and np.std(gpt_confs) > 0:
        correlation = float(np.corrcoef(talk_probs, gpt_confs)[0, 1])
    else:
        correlation = 0.0

    # Agreement analysis: when both agree vs disagree
    agree_talk = sum(1 for p in pairs if p["talk_prob"] > 0.5 and p["gpt_talk_conf"] > 0.5)
    agree_notalk = sum(1 for p in pairs if p["talk_prob"] <= 0.5 and p["gpt_talk_conf"] <= 0.5)
    disagree = len(pairs) - agree_talk - agree_notalk

    # w2t_prob uncertain zone analysis
    uncertain = [p for p in pairs if 0.1 < p["w2t_prob"] < 0.9]
    if uncertain:
        gpt_correct_uncertain = sum(
            1 for p in uncertain
            if (p["gpt_needs_help"] == "yes") == p["is_talk_gt"]
        )
        uncertain_accuracy = gpt_correct_uncertain / len(uncertain)
    else:
        uncertain_accuracy = 0.0

    return {
        "correlation": correlation,
        "n": len(pairs),
        "agree_talk": agree_talk,
        "agree_notalk": agree_notalk,
        "disagree": disagree,
        "agreement_rate": (agree_talk + agree_notalk) / len(pairs) if pairs else 0,
        "uncertain_zone_n": len(uncertain),
        "uncertain_zone_gpt_accuracy": uncertain_accuracy,
    }


def compute_per_task_metrics(all_results: list) -> dict:
    """Compute metrics broken down by task type."""
    task_groups = defaultdict(list)

    for sample in all_results:
        task = sample["task_goal"]
        task_groups[task].append(sample)

    per_task = {}
    for task, samples in task_groups.items():
        metrics = compute_metrics(samples, threshold_conf=0.5)
        per_task[task] = metrics

    return per_task


def analyze_yes_bias(all_results: list) -> dict:
    """Analyze whether GPT-4o shows yes-bias (like Qwen3-VL on ESTP-Bench)."""
    total = 0
    yes_count = 0
    no_count = 0
    uncertain_count = 0

    for sample in all_results:
        for r in sample["results"]:
            if r["_error"]:
                continue
            total += 1
            if r["gpt_needs_help"] == "yes":
                yes_count += 1
            elif r["gpt_needs_help"] == "no":
                no_count += 1
            else:
                uncertain_count += 1

    return {
        "total": total,
        "yes": yes_count,
        "no": no_count,
        "uncertain": uncertain_count,
        "yes_rate": yes_count / total if total > 0 else 0,
        "no_rate": no_count / total if total > 0 else 0,
        "uncertain_rate": uncertain_count / total if total > 0 else 0,
    }


def generate_report(all_results: list) -> str:
    """Generate comprehensive analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("REASONING TRIGGER ON PROASSIST WTAG - ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Date: 2026-02-25")
    lines.append(f"Model: GPT-4o (Azure OpenAI)")
    lines.append(f"Dataset: WTAG validation (dialog-klg-sum_val_L4096_I1)")
    lines.append(f"Samples: {len(all_results)}")
    total_api = sum(s["api_calls"] for s in all_results)
    lines.append(f"Total API calls: {total_api}")
    lines.append("")

    # Per-sample summary
    lines.append("-" * 70)
    lines.append("PER-SAMPLE SUMMARY")
    lines.append("-" * 70)
    for s in all_results:
        lines.append(f"  Sample {s['sample_idx']} ({s['video_uid']}): {s['task_goal']}")
        lines.append(f"    Frames evaluated: {s['eval_frames']}/{s['total_frames']}, "
                      f"GT talk: {s['gt_talk_count']}")
    lines.append("")

    # Yes-bias analysis
    lines.append("-" * 70)
    lines.append("YES-BIAS ANALYSIS")
    lines.append("-" * 70)
    bias = analyze_yes_bias(all_results)
    lines.append(f"  Total frames evaluated: {bias['total']}")
    lines.append(f"  GPT-4o 'yes' (needs help): {bias['yes']} ({bias['yes_rate']:.1%})")
    lines.append(f"  GPT-4o 'no':               {bias['no']} ({bias['no_rate']:.1%})")
    lines.append(f"  GPT-4o 'uncertain':         {bias['uncertain']} ({bias['uncertain_rate']:.1%})")
    lines.append(f"  [Comparison: Qwen3-VL on ESTP-Bench showed 76-83% yes-bias]")
    lines.append("")

    # F1 at different thresholds
    lines.append("-" * 70)
    lines.append("F1 AT DIFFERENT CONFIDENCE THRESHOLDS")
    lines.append("-" * 70)
    lines.append(f"  {'Threshold':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}")

    best_f1 = 0
    best_threshold = 0
    for thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = compute_metrics(all_results, threshold_conf=thresh)
        lines.append(f"  {thresh:10.1f} {m['precision']:10.3f} {m['recall']:10.3f} {m['f1']:8.3f} "
                      f"{m['tp']:5d} {m['fp']:5d} {m['fn']:5d} {m['tn']:5d}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = thresh

    lines.append(f"\n  Best F1: {best_f1:.3f} at threshold={best_threshold:.1f}")
    lines.append(f"  [Comparison: ESTP-Bench Reasoning Trigger F1=0.173, Baseline=0.225]")
    lines.append(f"  [Comparison: ProAssist w2t_prob best F1=0.373 at threshold=0.20]")
    lines.append("")

    # w2t_prob comparison
    lines.append("-" * 70)
    lines.append("GPT-4o vs W2T_PROB COMPARISON")
    lines.append("-" * 70)
    w2t_cmp = compute_w2t_comparison(all_results)
    lines.append(f"  Correlation (talk_prob vs gpt_confidence): {w2t_cmp['correlation']:.3f}")
    lines.append(f"  Agreement rate: {w2t_cmp['agreement_rate']:.1%} "
                  f"(agree_talk={w2t_cmp['agree_talk']}, agree_notalk={w2t_cmp['agree_notalk']}, "
                  f"disagree={w2t_cmp['disagree']})")
    lines.append(f"  w2t_prob uncertain zone [0.1, 0.9]:")
    lines.append(f"    N={w2t_cmp['uncertain_zone_n']}, GPT-4o accuracy={w2t_cmp['uncertain_zone_gpt_accuracy']:.1%}")
    lines.append("")

    # Per-task analysis
    lines.append("-" * 70)
    lines.append("PER-TASK ANALYSIS")
    lines.append("-" * 70)
    per_task = compute_per_task_metrics(all_results)
    for task, m in per_task.items():
        lines.append(f"  {task}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                      f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
    lines.append("")

    # Detailed frame-level analysis
    lines.append("-" * 70)
    lines.append("FRAME-LEVEL DETAIL (first 5 per sample)")
    lines.append("-" * 70)
    for s in all_results:
        lines.append(f"\n  Sample {s['sample_idx']} ({s['video_uid']}): {s['task_goal']}")
        for r in s["results"][:5]:
            talk_mark = "GT:TALK" if r["is_talk_gt"] else "GT:----"
            gpt_mark = f"GPT:{r['gpt_needs_help']}"
            w2t = r["w2t_prob"] if r["w2t_prob"] is not None else -1
            lines.append(f"    Frame {r['frame_idx']:4d} ({r['timestamp']:6.1f}s): "
                          f"{talk_mark} | {gpt_mark:14s} conf={r['gpt_confidence']:.2f} | "
                          f"w2t_prob={w2t:.4f}")
            lines.append(f"      Action: {r['gpt_action'][:70]}")
            lines.append(f"      Reasoning: {r['gpt_reasoning'][:70]}")
            if r["is_talk_gt"]:
                lines.append(f"      GT content: {r['gt_content'][:70]}")
    lines.append("")

    # Conclusions
    lines.append("-" * 70)
    lines.append("CONCLUSIONS")
    lines.append("-" * 70)

    # Compare with ESTP-Bench
    m_05 = compute_metrics(all_results, threshold_conf=0.5)
    lines.append(f"  1. GPT-4o Reasoning Trigger on WTAG:")
    lines.append(f"     F1={best_f1:.3f} (best, at threshold={best_threshold:.1f})")
    lines.append(f"     F1={m_05['f1']:.3f} (at default threshold=0.5)")
    lines.append(f"")
    lines.append(f"  2. Yes-bias: {bias['yes_rate']:.1%} "
                  f"({'SEVERE' if bias['yes_rate'] > 0.7 else 'MODERATE' if bias['yes_rate'] > 0.5 else 'LOW'})")
    lines.append(f"     [ESTP-Bench Qwen3-VL: 76-83%]")
    lines.append(f"")
    lines.append(f"  3. Correlation with w2t_prob: {w2t_cmp['correlation']:.3f}")
    lines.append(f"     {'Complementary (low correlation)' if abs(w2t_cmp['correlation']) < 0.3 else 'Redundant (high correlation)' if abs(w2t_cmp['correlation']) > 0.6 else 'Moderate overlap'}")
    lines.append(f"")

    if best_f1 > 0.373:
        lines.append(f"  4. RESULT: Reasoning trigger OUTPERFORMS w2t_prob on WTAG")
        lines.append(f"     ({best_f1:.3f} vs 0.373)")
    elif best_f1 > 0.2:
        lines.append(f"  4. RESULT: Reasoning trigger shows MODERATE effectiveness on WTAG")
        lines.append(f"     ({best_f1:.3f} vs w2t_prob 0.373)")
        if abs(w2t_cmp["correlation"]) < 0.3:
            lines.append(f"     Low correlation with w2t_prob suggests COMPLEMENTARY value")
    else:
        lines.append(f"  4. RESULT: Reasoning trigger shows POOR effectiveness on WTAG")
        lines.append(f"     ({best_f1:.3f} vs w2t_prob 0.373)")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Reasoning Trigger on ProAssist WTAG")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to evaluate (max 6)")
    parser.add_argument("--frame_step", type=int, default=10,
                        help="Evaluate every N frames (10 = every 5s at 2fps)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-frame details")
    parser.add_argument("--report_only", action="store_true",
                        help="Generate report from existing results (no API calls)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = OUTPUT_DIR / "checkpoint.jsonl"
    results_file = OUTPUT_DIR / "reasoning_trigger_results.json"
    report_file = OUTPUT_DIR / "reasoning_trigger_report.txt"

    if args.report_only:
        if results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)
            report = generate_report(all_results)
            print(report)
            with open(report_file, "w") as f:
                f.write(report)
            print(f"\nReport saved to {report_file}")
        else:
            print(f"No results file found at {results_file}")
        return

    # Test API connectivity
    print("Testing API connectivity...")
    try:
        test_resp = requests.post(
            API_URL, headers=API_HEADERS,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5},
            timeout=15,
        )
        if test_resp.status_code != 200:
            print(f"API test failed: {test_resp.status_code} {test_resp.text[:200]}")
            return
        print(f"  API OK: {test_resp.json()['model']}")
    except Exception as e:
        print(f"API connection failed: {e}")
        return

    # Select samples
    samples_to_run = SELECTED_SAMPLES[:args.num_samples]
    print(f"\nWill evaluate {len(samples_to_run)} samples: {samples_to_run}")

    # Load checkpoint if resuming
    completed = set()
    all_results = []
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                result = json.loads(line)
                all_results.append(result)
                completed.add(result["sample_idx"])
        print(f"Resumed {len(completed)} completed samples from checkpoint")

    # Run experiment
    start_time = time.time()

    for sample_idx in samples_to_run:
        if sample_idx in completed:
            print(f"Skipping sample {sample_idx} (already completed)")
            continue

        result = run_sample(sample_idx, frame_step=args.frame_step, verbose=args.verbose)
        all_results.append(result)

        # Save checkpoint
        with open(checkpoint_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  Done: {result['eval_frames']} frames, {result['api_calls']} API calls")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Total API calls: {sum(s['api_calls'] for s in all_results)}")

    # Save full results
    # Convert sets to lists for JSON serialization
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {results_file}")

    # Generate and save report
    report = generate_report(all_results)
    print(report)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
