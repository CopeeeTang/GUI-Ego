#!/usr/bin/env python3
"""
Phase 3: Innovation Hypothesis Testing on ESTP-Bench.

Runs modified FBF inference on selected failure cases for each hypothesis,
then evaluates with ESTP-F1 and produces per-case qualitative analysis.

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate
    python3 -m proactive-project.experiments.estp_phase3.hypothesis_runner \
        --hypothesis all --verbose

    # Or run a single hypothesis:
    python3 -m proactive-project.experiments.estp_phase3.hypothesis_runner \
        --hypothesis A --verbose
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# ─── Path setup ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # /home/v-tangxin/GUI
ESTP_DIR = PROJECT_ROOT / "data" / "ESTP-Bench" / "estp_dataset"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ESTP_DIR))

from prompts import (
    CONFIDENCE_CHECK_PROMPT,
    CONFIDENCE_THRESHOLD,
    PROMPT_TEMPLATE,
    get_answer_prompt,
    get_trigger_prompt,
)
from frame_selector import AdaptiveFrameSelector, VanillaFrameSelector


# ─── Constants ───
VIDEO_ROOT = PROJECT_ROOT / "data" / "ESTP-Bench" / "full_scale_2fps_max384"
DATA_FILE = ESTP_DIR / "estp_bench_sq.json"
PHASE3_CASES = ESTP_DIR / "analysis_full" / "phase3_test_cases.json"
OUTPUT_DIR = SCRIPT_DIR / "results"

ANTICIPATION = 1
LATENCY = 2


def ceil_time_by_fps(t, fps, min_t, max_t):
    return min(max(math.ceil(t * fps) / fps, min_t), max_t)


# ═══════════════════════════════════════════════════════════
#  Model Wrapper (reuses the global Qwen3VL singleton)
# ═══════════════════════════════════════════════════════════

_model = None
_processor = None


def load_model():
    """Load Qwen3-VL-8B-Instruct (singleton)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"[Phase3] Loading {model_id}...")
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    print(f"[Phase3] Model loaded.")
    return _model, _processor


def run_inference(frames, prompt):
    """Run VLM inference on a list of PIL frames with a text prompt.

    Returns: response text (str)
    """
    model, processor = load_model()

    if not frames:
        return ""

    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input], images=frames, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True).strip()
    return response


def run_inference_with_logprobs(frames, prompt):
    """Run VLM inference and return first-token logprobs for yes/no.

    Returns: (response_text, yes_logprob, no_logprob)
    """
    model, processor = load_model()

    if not frames:
        return "", float("-inf"), float("-inf")

    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_input], images=frames, padding=True, return_tensors="pt"
    ).to(model.device)

    # Get logits for the first generated token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Find token IDs for "yes", "Yes", "no", "No"
    yes_tokens = processor.tokenizer.encode("yes", add_special_tokens=False)
    no_tokens = processor.tokenizer.encode("no", add_special_tokens=False)
    Yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
    No_tokens = processor.tokenizer.encode("No", add_special_tokens=False)

    # Take the first subword token for each
    yes_logp = max(
        log_probs[yes_tokens[0]].item() if yes_tokens else float("-inf"),
        log_probs[Yes_tokens[0]].item() if Yes_tokens else float("-inf"),
    )
    no_logp = max(
        log_probs[no_tokens[0]].item() if no_tokens else float("-inf"),
        log_probs[No_tokens[0]].item() if No_tokens else float("-inf"),
    )

    # Also generate the full response for logging
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True).strip()

    return response, yes_logp, no_logp


# ═══════════════════════════════════════════════════════════
#  FBF Loop Variants
# ═══════════════════════════════════════════════════════════

def run_fbf_vanilla(video_path, question, start_time, end_time, frame_fps=0.175,
                    max_frames=64, verbose=False):
    """Vanilla FBF baseline (identical to ESTP-Bench eval_fbf)."""
    from frame_selector import VanillaFrameSelector
    import cv2
    from PIL import Image

    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)

    # Use cv2-based frame extraction matching ESTP's model.Run()
    while current_time <= end_time:
        frames = _extract_frames_upto(video_path, start_time, current_time, max_frames)
        if not frames:
            current_time += 1 / frame_fps
            continue

        t0 = time.time()
        response = run_inference(frames, trigger_prompt)
        cost = time.time() - t0

        triggered = "yes" in response.lower()
        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "response": response[:100],
            "n_frames": len(frames),
            "cost": cost,
        })

        if verbose and triggered:
            print(f"    [VANILLA] t={current_time:.1f}s TRIGGERED: {response[:60]}")

        if triggered:
            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": get_trigger_prompt(question, None, "vanilla"),
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


def run_fbf_hypothesis_a(video_path, question, task_type, start_time, end_time,
                         frame_fps=0.175, max_frames=64, verbose=False):
    """Hypothesis A: Task-Semantic Enriched Trigger."""
    semantic_trigger = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, task_type, "semantic")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)

    while current_time <= end_time:
        frames = _extract_frames_upto(video_path, start_time, current_time, max_frames)
        if not frames:
            current_time += 1 / frame_fps
            continue

        t0 = time.time()
        response = run_inference(frames, semantic_trigger)
        cost = time.time() - t0

        triggered = "yes" in response.lower()
        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "response": response[:100],
            "n_frames": len(frames),
            "cost": cost,
        })

        if verbose and triggered:
            print(f"    [HYPO-A] t={current_time:.1f}s TRIGGERED: {response[:60]}")

        if triggered:
            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": get_trigger_prompt(question, task_type, "semantic"),
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


def run_fbf_hypothesis_b(video_path, question, start_time, end_time,
                         frame_fps=0.175, verbose=False):
    """Hypothesis B: Adaptive Temporal Attention Window.

    Uses video-native fps (2fps) for frame sampling, NOT the polling rate.
    Recent window = 20s at ~2fps = up to 20 frames of recent context.
    History = keyframes selected by visual change detection.
    """
    selector = AdaptiveFrameSelector(
        recent_window_sec=20.0,
        max_total_frames=32,
        max_recent_frames=20,
        change_threshold=0.90,
        sample_fps=2.0,   # Use video native fps, NOT polling rate
    )

    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)

    while current_time <= end_time:
        # Key difference: use adaptive frame selection
        frames, timestamps = selector.select_frames(
            str(video_path), start_time, current_time, frame_fps
        )
        if not frames:
            current_time += 1 / frame_fps
            continue

        # Add temporal context to prompt
        time_context = (
            f"You are at timestamp {current_time:.1f}s in a video "
            f"(total: {end_time - start_time:.0f}s). "
            f"You see {len(frames)} frames spanning "
            f"{timestamps[0]:.1f}s to {timestamps[-1]:.1f}s.\n\n"
        )
        full_trigger = time_context + trigger_prompt

        t0 = time.time()
        response = run_inference(frames, full_trigger)
        cost = time.time() - t0

        triggered = "yes" in response.lower()
        recent_boundary = current_time - selector.recent_window_sec
        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "response": response[:100],
            "n_frames": len(frames),
            "n_recent": sum(1 for ts in timestamps if ts >= recent_boundary),
            "n_history": sum(1 for ts in timestamps if ts < recent_boundary),
            "cost": cost,
        })

        if verbose:
            status = "TRIGGERED" if triggered else "suppressed"
            print(f"    [HYPO-B] t={current_time:.1f}s {status} "
                  f"({len(frames)} frames: {sum(1 for ts in timestamps if ts >= recent_boundary)}R+"
                  f"{sum(1 for ts in timestamps if ts < recent_boundary)}H)")

        if triggered:
            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": "trigger",
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


def run_fbf_hypothesis_c(video_path, question, start_time, end_time,
                         frame_fps=0.175, max_frames=64, verbose=False):
    """Hypothesis C: Confidence-Calibrated Two-Stage Filter."""
    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)

    while current_time <= end_time:
        frames = _extract_frames_upto(video_path, start_time, current_time, max_frames)
        if not frames:
            current_time += 1 / frame_fps
            continue

        t0 = time.time()
        response = run_inference(frames, trigger_prompt)
        cost = time.time() - t0

        triggered = "yes" in response.lower()
        confidence = 0
        passed_filter = False

        if triggered:
            # Stage 2: confidence check
            conf_prompt = CONFIDENCE_CHECK_PROMPT.format(question=question)
            t1 = time.time()
            conf_response = run_inference(frames, conf_prompt)
            cost += time.time() - t1

            # Parse confidence score
            for ch in conf_response:
                if ch in "12345":
                    confidence = int(ch)
                    break

            passed_filter = confidence >= CONFIDENCE_THRESHOLD

            if verbose:
                status = "PASS" if passed_filter else "FILTERED"
                print(f"    [HYPO-C] t={current_time:.1f}s conf={confidence} {status}: {response[:40]}")

        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "confidence": confidence,
            "passed_filter": passed_filter,
            "response": response[:100],
            "n_frames": len(frames),
            "cost": cost,
        })

        if passed_filter:
            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": "trigger",
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


def run_fbf_hypothesis_d(video_path, question, start_time, end_time,
                         frame_fps=0.175, max_frames=64, logprob_threshold=3.0,
                         cooldown_sec=0.0, verbose=False):
    """Hypothesis D: Logprob-Based Trigger with three-segment rule + cooldown.

    Three-segment rule for logprob gap = logP(yes) - logP(no):
      - gap < 0:   REJECT (model favors "no" — high confidence negative)
      - 0 <= gap <= threshold: REJECT (model leans "yes" but uncertain)
      - gap > threshold: TRIGGER (model is confidently "yes")

    Cooldown: after a trigger, suppress subsequent triggers for cooldown_sec
    seconds to prevent burst-triggering on consecutive frames.
    """
    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)
    last_trigger_time = -float("inf")  # cooldown tracking

    while current_time <= end_time:
        frames = _extract_frames_upto(video_path, start_time, current_time, max_frames)
        if not frames:
            current_time += 1 / frame_fps
            continue

        t0 = time.time()
        response, yes_logp, no_logp = run_inference_with_logprobs(frames, trigger_prompt)
        cost = time.time() - t0

        logprob_gap = yes_logp - no_logp

        # Three-segment rule
        gap_decision = "trigger" if logprob_gap > logprob_threshold else "reject"

        # Cooldown check
        in_cooldown = (current_time - last_trigger_time) < cooldown_sec
        triggered = (gap_decision == "trigger") and not in_cooldown

        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "gap_decision": gap_decision,
            "in_cooldown": in_cooldown,
            "response": response[:100],
            "yes_logp": yes_logp,
            "no_logp": no_logp,
            "logprob_gap": logprob_gap,
            "n_frames": len(frames),
            "cost": cost,
        })

        if verbose:
            if triggered:
                status = "TRIGGERED"
            elif in_cooldown:
                status = f"cooldown({current_time - last_trigger_time:.1f}s)"
            else:
                status = f"reject(gap={logprob_gap:.2f}<=τ={logprob_threshold})"
            print(f"    [HYPO-D] t={current_time:.1f}s gap={logprob_gap:.2f} {status}")

        if triggered:
            last_trigger_time = current_time

            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": "trigger",
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


def run_fbf_hypothesis_e(video_path, question, start_time, end_time,
                         frame_fps=0.175, logprob_threshold=3.0,
                         cooldown_sec=12.0, verbose=False):
    """Hypothesis E: Combined B+D+cooldown pipeline.

    Combines the best elements of Hypothesis B and D:
    - B (fixed): Adaptive frame selection with video-native 2fps sampling
      and 20s recent window → better temporal context for the VLM
    - D (fixed): Logprob-based trigger with three-segment rule → bypass
      text-level yes-bias using model's internal uncertainty
    - Cooldown: minimum interval between triggers → prevent burst-firing

    This is the "full stack" optimization that addresses both the input
    quality (B) and the decision mechanism (D) simultaneously.
    """
    selector = AdaptiveFrameSelector(
        recent_window_sec=20.0,
        max_total_frames=32,
        max_recent_frames=20,
        change_threshold=0.90,
        sample_fps=2.0,
    )

    trigger_prompt = PROMPT_TEMPLATE.format(
        query=get_trigger_prompt(question, None, "vanilla")
    )
    answer_prompt = PROMPT_TEMPLATE.format(query=get_answer_prompt(question))

    dialog_history = []
    trigger_log = []
    current_time = min(start_time + 1 / frame_fps, end_time)
    last_trigger_time = -float("inf")

    while current_time <= end_time:
        # B: Adaptive frame selection
        frames, timestamps = selector.select_frames(
            str(video_path), start_time, current_time, frame_fps
        )
        if not frames:
            current_time += 1 / frame_fps
            continue

        # B: Add temporal context
        time_context = (
            f"You are at timestamp {current_time:.1f}s in a video "
            f"(total: {end_time - start_time:.0f}s). "
            f"You see {len(frames)} frames spanning "
            f"{timestamps[0]:.1f}s to {timestamps[-1]:.1f}s.\n\n"
        )
        full_trigger = time_context + trigger_prompt

        # D: Logprob-based trigger
        t0 = time.time()
        response, yes_logp, no_logp = run_inference_with_logprobs(frames, full_trigger)
        cost = time.time() - t0

        logprob_gap = yes_logp - no_logp

        # D: Three-segment rule
        gap_decision = "trigger" if logprob_gap > logprob_threshold else "reject"

        # Cooldown check
        in_cooldown = (current_time - last_trigger_time) < cooldown_sec
        triggered = (gap_decision == "trigger") and not in_cooldown

        recent_boundary = current_time - selector.recent_window_sec
        trigger_log.append({
            "time": current_time,
            "triggered": triggered,
            "gap_decision": gap_decision,
            "in_cooldown": in_cooldown,
            "response": response[:100],
            "yes_logp": yes_logp,
            "no_logp": no_logp,
            "logprob_gap": logprob_gap,
            "n_frames": len(frames),
            "n_recent": sum(1 for ts in timestamps if ts >= recent_boundary),
            "n_history": sum(1 for ts in timestamps if ts < recent_boundary),
            "cost": cost,
        })

        if verbose:
            if triggered:
                status = "TRIGGERED"
            elif in_cooldown:
                status = f"cooldown({current_time - last_trigger_time:.1f}s)"
            else:
                status = f"reject(gap={logprob_gap:.2f})"
            n_r = sum(1 for ts in timestamps if ts >= recent_boundary)
            n_h = sum(1 for ts in timestamps if ts < recent_boundary)
            print(f"    [HYPO-E] t={current_time:.1f}s gap={logprob_gap:.2f} "
                  f"{status} ({len(frames)}f: {n_r}R+{n_h}H)")

        if triggered:
            last_trigger_time = current_time

            t0 = time.time()
            answer = run_inference(frames, answer_prompt)
            cost2 = time.time() - t0

            dialog_history.append({
                "role": "user",
                "content": "trigger",
                "time": current_time,
            })
            dialog_history.append({
                "role": "assistant",
                "content": answer,
                "time": current_time,
                "cost": cost + cost2,
            })

        current_time += 1 / frame_fps

    return dialog_history, trigger_log


# ═══════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════

def _extract_frames_upto(video_path, start_time, end_time, max_frames=64):
    """Extract frames from video (same logic as Qwen3VL adapter)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * video_fps)
    end_frame = min(int(end_time * video_fps), total_frames - 1)
    total_seg = end_frame - start_frame + 1

    if total_seg <= 0:
        cap.release()
        return []

    step = 1 if total_seg <= max_frames else total_seg / max_frames
    frame_indices = []
    i = 0.0
    while i < total_seg and len(frame_indices) < max_frames:
        frame_indices.append(start_frame + int(i))
        i += step

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def lookup_qa_in_dataset(dataset, video_uid, clip_uid, qa_id):
    """Find a specific QA instance in the dataset.

    Dataset structure: dataset[video_uid][clip_uid] = [qa1, qa2, ...]
    Each qa has 'qa_id' field.
    """
    if video_uid not in dataset:
        return None
    video = dataset[video_uid]

    # Primary path: look up by clip_uid
    if clip_uid in video:
        qas = video[clip_uid]
        if isinstance(qas, list):
            for qa in qas:
                if qa.get("qa_id", qa.get("id")) == qa_id:
                    return qa

    # Fallback: search all clips in this video
    for key, qas in video.items():
        if isinstance(qas, list):
            for qa in qas:
                if qa.get("qa_id", qa.get("id")) == qa_id:
                    return qa
    return None


def find_video_file(video_uid, clip_uid):
    """Find the video file path."""
    # Try common patterns
    for ext in [".mp4", ".mkv", ".avi"]:
        for pattern in [
            VIDEO_ROOT / f"{video_uid}{ext}",
            VIDEO_ROOT / video_uid / f"{clip_uid}{ext}",
            VIDEO_ROOT / f"{clip_uid}{ext}",
        ]:
            if pattern.exists():
                return str(pattern)
    # Search recursively
    for f in VIDEO_ROOT.rglob(f"*{video_uid[:8]}*"):
        if f.suffix in [".mp4", ".mkv", ".avi"]:
            return str(f)
    return None


# ═══════════════════════════════════════════════════════════
#  Per-Case ESTP-F1 Computation
# ═══════════════════════════════════════════════════════════

def compute_case_metrics(dialog_history, qa, anticipation=ANTICIPATION, latency=LATENCY):
    """Compute ESTP-F1 for a single QA case.

    Returns dict with f1, precision, recall, timing details.
    """
    gold_list = [c for c in qa["conversation"] if c["role"].lower() == "assistant"]
    clip_end = qa.get("clip_end_time", qa.get("end_time", 0))
    gt_spans = [
        (ceil_time_by_fps(c["start_time"], 2, 0, clip_end),
         ceil_time_by_fps(c["end_time"], 2, 0, clip_end))
        for c in gold_list
    ]

    preds = [d for d in dialog_history if d.get("role") == "assistant"]

    if not preds:
        return {
            "f1": 0, "precision": 0, "recall": 0,
            "n_preds": 0, "n_gt": len(gt_spans),
            "hits": [], "misses": list(range(len(gt_spans))),
            "false_positives": 0,
        }

    # Check which GT windows are hit
    hits = []
    for gi, (gs, ge) in enumerate(gt_spans):
        hit = False
        for pred in preds:
            pt = pred.get("time", -1)
            if gs - anticipation <= pt <= ge + latency:
                hit = True
                break
        if hit:
            hits.append(gi)
    misses = [gi for gi in range(len(gt_spans)) if gi not in hits]

    # Check false positives
    fp = 0
    for pred in preds:
        pt = pred.get("time", -1)
        in_any = False
        for gs, ge in gt_spans:
            if gs - anticipation <= pt <= ge + latency:
                in_any = True
                break
        if not in_any:
            fp += 1

    tp = len(hits)
    fn = len(misses)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "n_preds": len(preds),
        "n_gt": len(gt_spans),
        "hits": hits,
        "misses": misses,
        "false_positives": fp,
        "pred_times": [p.get("time", -1) for p in preds],
        "gt_spans": gt_spans,
    }


# ═══════════════════════════════════════════════════════════
#  Main Experiment Runner
# ═══════════════════════════════════════════════════════════

def run_hypothesis_experiment(hypothesis_id, cases, dataset, frame_fps=0.175,
                              verbose=True, logprob_threshold=0.5,
                              cooldown_sec=0.0):
    """Run a single hypothesis on its test cases.

    Returns list of per-case results with baseline comparison.
    """
    results = []

    for i, case in enumerate(cases):
        video_uid = case["video_uid"]
        clip_uid = case["clip_uid"]
        qa_id = case["qa_id"]
        task_type = case["task_type"]
        question = case["question"]

        print(f"\n  [{i+1}/{len(cases)}] {task_type}: {question[:60]}...")
        print(f"    video={video_uid[:12]}... clip={clip_uid[:12]}... qa_id={qa_id}")

        # Find video
        video_path = find_video_file(video_uid, clip_uid)
        if not video_path:
            print(f"    SKIP: video not found")
            results.append({"case": case, "status": "video_not_found"})
            continue

        # Find QA data
        qa = lookup_qa_in_dataset(dataset, video_uid, clip_uid, qa_id)
        if qa is None:
            # Try broader search
            if video_uid in dataset:
                for key, qas in dataset[video_uid].items():
                    if isinstance(qas, list) and len(qas) > qa_id:
                        qa = qas[qa_id]
                        break

        if qa is None:
            print(f"    SKIP: QA not found in dataset")
            results.append({"case": case, "status": "qa_not_found"})
            continue

        start_time = qa.get("clip_start_time", qa.get("start_time", 0))
        end_time = qa.get("clip_end_time", qa.get("end_time", 0))
        duration = end_time - start_time

        # GT info
        gold = [c for c in qa["conversation"] if c["role"].lower() == "assistant"]
        gt_spans = [
            (ceil_time_by_fps(c["start_time"], 2, 0, end_time),
             ceil_time_by_fps(c["end_time"], 2, 0, end_time))
            for c in gold
        ]
        print(f"    Duration={duration:.0f}s, GT_windows={len(gt_spans)}, "
              f"GT_spans={[(f'{s:.0f}-{e:.0f}') for s,e in gt_spans[:3]]}")

        # ─── Run baseline ───
        print(f"    Running BASELINE (vanilla FBF)...")
        t0 = time.time()
        baseline_dialog, baseline_log = run_fbf_vanilla(
            video_path, question, start_time, end_time, frame_fps, verbose=verbose
        )
        baseline_time = time.time() - t0
        baseline_metrics = compute_case_metrics(baseline_dialog, qa)
        print(f"    BASELINE: F1={baseline_metrics['f1']:.3f}, "
              f"preds={baseline_metrics['n_preds']}, "
              f"hits={len(baseline_metrics['hits'])}/{baseline_metrics['n_gt']}, "
              f"FP={baseline_metrics['false_positives']} ({baseline_time:.1f}s)")

        # ─── Run hypothesis ───
        print(f"    Running HYPOTHESIS {hypothesis_id}...")
        t0 = time.time()

        if hypothesis_id == "A":
            hyp_dialog, hyp_log = run_fbf_hypothesis_a(
                video_path, question, task_type, start_time, end_time,
                frame_fps, verbose=verbose,
            )
        elif hypothesis_id == "B":
            hyp_dialog, hyp_log = run_fbf_hypothesis_b(
                video_path, question, start_time, end_time,
                frame_fps, verbose=verbose,
            )
        elif hypothesis_id == "C":
            hyp_dialog, hyp_log = run_fbf_hypothesis_c(
                video_path, question, start_time, end_time,
                frame_fps, verbose=verbose,
            )
        elif hypothesis_id == "D":
            hyp_dialog, hyp_log = run_fbf_hypothesis_d(
                video_path, question, start_time, end_time,
                frame_fps, logprob_threshold=logprob_threshold,
                cooldown_sec=cooldown_sec, verbose=verbose,
            )
        elif hypothesis_id == "E":
            hyp_dialog, hyp_log = run_fbf_hypothesis_e(
                video_path, question, start_time, end_time,
                frame_fps, logprob_threshold=logprob_threshold,
                cooldown_sec=cooldown_sec, verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        hyp_time = time.time() - t0
        hyp_metrics = compute_case_metrics(hyp_dialog, qa)
        print(f"    HYPO-{hypothesis_id}: F1={hyp_metrics['f1']:.3f}, "
              f"preds={hyp_metrics['n_preds']}, "
              f"hits={len(hyp_metrics['hits'])}/{hyp_metrics['n_gt']}, "
              f"FP={hyp_metrics['false_positives']} ({hyp_time:.1f}s)")

        # ─── Delta ───
        delta_f1 = hyp_metrics["f1"] - baseline_metrics["f1"]
        indicator = "+" if delta_f1 > 0 else ""
        print(f"    DELTA F1: {indicator}{delta_f1:.3f}")

        results.append({
            "case": case,
            "status": "ok",
            "duration": duration,
            "n_gt": len(gt_spans),
            "gt_spans": gt_spans,
            "gt_answers": [c["content"][:80] for c in gold],
            "baseline": {
                "dialog": baseline_dialog,
                "metrics": baseline_metrics,
                "trigger_log": baseline_log,
                "wall_time": baseline_time,
            },
            "hypothesis": {
                "dialog": hyp_dialog,
                "metrics": hyp_metrics,
                "trigger_log": hyp_log,
                "wall_time": hyp_time,
            },
            "delta_f1": delta_f1,
        })

    return results


def generate_report(hypothesis_id, hypothesis_name, results, output_path):
    """Generate a detailed report for one hypothesis."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Phase 3 Hypothesis {hypothesis_id}: {hypothesis_name}")
    lines.append("=" * 80)

    valid = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]

    lines.append(f"\nCases tested: {len(valid)}, Skipped: {len(skipped)}")

    if not valid:
        lines.append("No valid cases to analyze.")
        report = "\n".join(lines)
        with open(output_path, "w") as f:
            f.write(report)
        return report

    # Aggregate metrics
    baseline_f1s = [r["baseline"]["metrics"]["f1"] for r in valid]
    hyp_f1s = [r["hypothesis"]["metrics"]["f1"] for r in valid]
    delta_f1s = [r["delta_f1"] for r in valid]

    baseline_fps = [r["baseline"]["metrics"]["false_positives"] for r in valid]
    hyp_fps = [r["hypothesis"]["metrics"]["false_positives"] for r in valid]

    lines.append(f"\n--- Aggregate Metrics ---")
    lines.append(f"  Baseline avg F1:    {np.mean(baseline_f1s):.3f} (std={np.std(baseline_f1s):.3f})")
    lines.append(f"  Hypothesis avg F1:  {np.mean(hyp_f1s):.3f} (std={np.std(hyp_f1s):.3f})")
    lines.append(f"  Delta F1:           {np.mean(delta_f1s):+.3f}")
    lines.append(f"  Cases improved:     {sum(1 for d in delta_f1s if d > 0)}/{len(valid)}")
    lines.append(f"  Cases degraded:     {sum(1 for d in delta_f1s if d < 0)}/{len(valid)}")
    lines.append(f"  Cases unchanged:    {sum(1 for d in delta_f1s if d == 0)}/{len(valid)}")
    lines.append(f"  Baseline avg FP:    {np.mean(baseline_fps):.1f}")
    lines.append(f"  Hypothesis avg FP:  {np.mean(hyp_fps):.1f}")

    # Per-case detailed analysis
    lines.append(f"\n--- Per-Case Analysis ---")
    for i, r in enumerate(valid):
        c = r["case"]
        bm = r["baseline"]["metrics"]
        hm = r["hypothesis"]["metrics"]

        lines.append(f"\n  Case {i+1}: {c['task_type']}")
        lines.append(f"  Q: {c['question']}")
        lines.append(f"  Duration: {r['duration']:.0f}s, GT windows: {r['n_gt']}")
        lines.append(f"  GT spans: {[(f'{s:.0f}-{e:.0f}s') for s,e in r['gt_spans']]}")
        lines.append(f"  GT answers: {r['gt_answers']}")

        # Baseline detail
        lines.append(f"  BASELINE: F1={bm['f1']:.3f}, Preds={bm['n_preds']}, "
                      f"Hits={len(bm['hits'])}/{bm['n_gt']}, FP={bm['false_positives']}")
        if bm.get("pred_times"):
            lines.append(f"    Pred times: {[f'{t:.0f}s' for t in bm['pred_times']]}")

        # Hypothesis detail
        lines.append(f"  HYPO-{hypothesis_id}: F1={hm['f1']:.3f}, Preds={hm['n_preds']}, "
                      f"Hits={len(hm['hits'])}/{hm['n_gt']}, FP={hm['false_positives']}")
        if hm.get("pred_times"):
            lines.append(f"    Pred times: {[f'{t:.0f}s' for t in hm['pred_times']]}")

        # Delta
        delta = r["delta_f1"]
        indicator = "IMPROVED" if delta > 0 else ("DEGRADED" if delta < 0 else "UNCHANGED")
        lines.append(f"  VERDICT: {indicator} (delta={delta:+.3f})")

        # Qualitative: show answers for hard cases
        if bm["f1"] == 0 or hm["f1"] > bm["f1"]:
            lines.append(f"  [HARD CASE DETAIL]")
            for d in r["baseline"]["dialog"]:
                if d["role"] == "assistant":
                    lines.append(f"    Baseline answer (t={d['time']:.0f}s): {d['content'][:120]}")
            for d in r["hypothesis"]["dialog"]:
                if d["role"] == "assistant":
                    lines.append(f"    Hypo answer (t={d['time']:.0f}s): {d['content'][:120]}")

    # Trigger pattern analysis
    lines.append(f"\n--- Trigger Pattern Analysis ---")
    baseline_triggers = sum(
        sum(1 for t in r["baseline"]["trigger_log"] if t["triggered"])
        for r in valid
    )
    hyp_triggers = sum(
        sum(1 for t in r["hypothesis"]["trigger_log"] if t["triggered"])
        for r in valid
    )
    total_polls = sum(len(r["baseline"]["trigger_log"]) for r in valid)

    lines.append(f"  Total polling steps: {total_polls}")
    lines.append(f"  Baseline triggers:   {baseline_triggers} ({baseline_triggers/total_polls*100:.1f}%)")
    lines.append(f"  Hypothesis triggers: {hyp_triggers} ({hyp_triggers/total_polls*100:.1f}%)")

    if hypothesis_id == "C":
        filtered = sum(
            sum(1 for t in r["hypothesis"]["trigger_log"]
                if t.get("triggered") and not t.get("passed_filter"))
            for r in valid
        )
        lines.append(f"  Filtered by confidence: {filtered}")

    if hypothesis_id == "D":
        # Logprob gap analysis
        all_gaps = []
        for r in valid:
            for t in r["hypothesis"]["trigger_log"]:
                gap = t.get("logprob_gap", 0)
                all_gaps.append(gap)
        if all_gaps:
            lines.append(f"  Logprob gap statistics:")
            lines.append(f"    Mean: {np.mean(all_gaps):.3f}")
            lines.append(f"    Std:  {np.std(all_gaps):.3f}")
            lines.append(f"    Min:  {np.min(all_gaps):.3f}")
            lines.append(f"    Max:  {np.max(all_gaps):.3f}")
            lines.append(f"    Median: {np.median(all_gaps):.3f}")
            # Distribution
            for thresh in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
                pct = sum(1 for g in all_gaps if g > thresh) / len(all_gaps) * 100
                lines.append(f"    Gap > {thresh}: {pct:.1f}%")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")
    return report


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Hypothesis Testing")
    parser.add_argument("--hypothesis", type=str, default="all",
                        choices=["A", "B", "C", "D", "E", "all"],
                        help="Which hypothesis to test (E = combined B+D+cooldown)")
    parser.add_argument("--logprob_threshold", type=float, default=3.0,
                        help="Logprob gap threshold for hypothesis D (three-segment rule)")
    parser.add_argument("--cooldown_sec", type=float, default=0.0,
                        help="Cooldown seconds between triggers for hypothesis D")
    parser.add_argument("--max_cases", type=int, default=5,
                        help="Max cases per hypothesis (for quick testing)")
    parser.add_argument("--frame_fps", type=float, default=0.175)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from: {DATA_FILE}")
    with open(DATA_FILE) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} videos")

    print(f"Loading test cases from: {PHASE3_CASES}")
    with open(PHASE3_CASES) as f:
        test_cases = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load model
    load_model()

    hypotheses = {
        "A": ("Task-Semantic Enriched Trigger", test_cases["hypothesis_a"]["cases"]),
        "B": ("Adaptive Temporal Attention Window (fixed)", test_cases["hypothesis_b"]["cases"]),
        "C": ("Confidence-Calibrated Two-Stage Filter", test_cases["hypothesis_c"]["cases"]),
        "D": ("Logprob-Based Trigger + Cooldown", test_cases["hypothesis_c"]["cases"]),
        # Hypothesis E combines B+D+cooldown, uses D's test cases (high-FP cases)
        "E": ("Combined B+D+Cooldown Pipeline", test_cases["hypothesis_c"]["cases"]),
    }

    to_run = ["A", "B", "C", "D", "E"] if args.hypothesis == "all" else [args.hypothesis]

    all_reports = {}
    for hyp_id in to_run:
        name, cases = hypotheses[hyp_id]
        cases = cases[:args.max_cases]

        print(f"\n{'='*80}")
        print(f"HYPOTHESIS {hyp_id}: {name}")
        print(f"Testing on {len(cases)} cases")
        print(f"{'='*80}")

        results = run_hypothesis_experiment(
            hyp_id, cases, dataset, args.frame_fps, verbose=args.verbose,
            logprob_threshold=args.logprob_threshold,
            cooldown_sec=args.cooldown_sec,
        )

        # Save raw results
        raw_path = OUTPUT_DIR / f"hypothesis_{hyp_id}_raw.json"
        # Convert non-serializable items
        serializable = json.loads(json.dumps(results, default=str))
        with open(raw_path, "w") as f:
            json.dump(serializable, f, indent=2)

        # Generate report
        report_path = OUTPUT_DIR / f"hypothesis_{hyp_id}_report.txt"
        report = generate_report(hyp_id, name, results, report_path)
        all_reports[hyp_id] = report
        print(report)

    # Summary
    print(f"\n{'='*80}")
    print("PHASE 3 SUMMARY")
    print(f"{'='*80}")
    for hyp_id in to_run:
        name, _ = hypotheses[hyp_id]
        raw_path = OUTPUT_DIR / f"hypothesis_{hyp_id}_raw.json"
        if raw_path.exists():
            with open(raw_path) as f:
                results = json.load(f)
            valid = [r for r in results if r.get("status") == "ok"]
            if valid:
                deltas = [r["delta_f1"] for r in valid]
                improved = sum(1 for d in deltas if d > 0)
                print(f"  Hypo {hyp_id} ({name}):")
                print(f"    Avg delta F1: {np.mean(deltas):+.3f}")
                print(f"    Improved: {improved}/{len(valid)}")


if __name__ == "__main__":
    main()
