#!/usr/bin/env python3
"""
Zero-shot evaluation of MMDuet2 on ESTP-Bench.

This script:
1. Converts ESTP-Bench data to MMDuet2 frame_input_format
2. Runs MMDuet2 inference in proactive mode
3. Computes ESTP-F1 score

Usage:
    # Step 1: Prepare data (extract frames, generate input JSON)
    python3 mmduet2_estp_eval.py --prepare --limit 5

    # Step 2: Run inference (needs GPU)
    python3 mmduet2_estp_eval.py --infer --limit 5

    # Step 3: Evaluate
    python3 mmduet2_estp_eval.py --eval

    # All steps:
    python3 mmduet2_estp_eval.py --all --limit 5

Environment:
    source /home/v-tangxin/GUI/proactive-project/mmduet2_env/bin/activate
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/v-tangxin/GUI")
ESTP_DIR = PROJECT_ROOT / "data" / "ESTP-Bench" / "estp_dataset"
VIDEO_ROOT = PROJECT_ROOT / "data" / "ESTP-Bench" / "full_scale_2fps_max384"
DATA_FILE = ESTP_DIR / "estp_bench_sq.json"
FRAME_DIR = PROJECT_ROOT / "data" / "ESTP-Bench" / "frames_2fps_384"
MMDUET2_DIR = PROJECT_ROOT / "proactive-project" / "mmduet2"
CKPT_DIR = MMDUET2_DIR / "checkpoints" / "MMDuet2"
OUTPUT_DIR = PROJECT_ROOT / "proactive-project" / "experiments" / "estp_phase3" / "results" / "mmduet2_zeroshot"

SYSTEM_PROMPT = (
    'You are a helpful assistant. Your task is to answer questions based on '
    'continuously incoming video frames. Your responses should include information '
    'from the video since your last reply (if any). If the information in this '
    'segment of the video cannot answer the question, output "NO REPLY".'
)

FPS = 2
SEC_PER_FRAME = 2  # 2 seconds per user turn (matches MMDuet2 training)
NO_REPLY = "NO REPLY"


def load_estp_data():
    with open(DATA_FILE) as f:
        return json.load(f)


def extract_frames_for_video(video_id: str):
    """Extract JPG frames from a single video at 2fps."""
    video_path = VIDEO_ROOT / f"{video_id}.mp4"
    frame_dir = FRAME_DIR / video_id

    if not video_path.exists():
        print(f"  WARNING: Video not found: {video_path}")
        return 0

    if frame_dir.exists() and len(list(frame_dir.glob("*.jpg"))) > 0:
        return len(list(frame_dir.glob("*.jpg")))

    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "%06d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={FPS}",
        "-q:v", "2",
        pattern,
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    return len(list(frame_dir.glob("*.jpg")))


def get_frame_path(video_id: str, frame_idx: int) -> str:
    """Get absolute path for frame. Frame indices are 1-based."""
    return str(FRAME_DIR / video_id / f"{frame_idx:06d}.jpg")


def convert_estp_to_frame_input(data: dict, limit: int = None):
    """Convert ESTP-Bench data to MMDuet2 frame_input_format.

    Returns:
        input_data: list of dicts for inference
        gt_data: dict for evaluation {question_id: {question, gt_windows}}
    """
    input_data = []
    gt_data = {}

    video_ids = sorted(data.keys())
    processed_count = 0

    for video_id in video_ids:
        if limit and processed_count >= limit:
            break

        # Extract frames
        n_frames = extract_frames_for_video(video_id)
        if n_frames == 0:
            continue

        qa_dict = data[video_id]
        for qa_id, qa_entries in sorted(qa_dict.items()):
            for entry_idx, qa_entry in enumerate(qa_entries):
                if limit and processed_count >= limit:
                    break

                # Handle two data formats in ESTP-Bench
                if "clip_start_time" in qa_entry:
                    clip_start = qa_entry["clip_start_time"]
                    clip_end = qa_entry["clip_end_time"]
                    question = qa_entry["question"]
                else:
                    clip_start = qa_entry["start_time"]
                    clip_end = qa_entry["end_time"]
                    # Question is in the first user turn of conversation
                    user_turns = [c for c in qa_entry["conversation"] if c["role"] == "user"]
                    question = user_turns[0]["content"] if user_turns else "Describe what is happening."

                conversations = qa_entry["conversation"]
                duration = clip_end - clip_start
                if duration < 4:
                    continue

                # Build conversation in Qwen VL format
                conv = []

                # Each user turn: 2 seconds of frames (at 2fps = 4 frames, but we use 2 per turn like MMDuet2)
                # MMDuet2 training: 2 frames per turn at 1fps effective
                n_turns = int(duration // SEC_PER_FRAME)
                if n_turns < 2:
                    continue

                for turn_idx in range(n_turns):
                    t_start = clip_start + turn_idx * SEC_PER_FRAME
                    t_end = clip_start + (turn_idx + 1) * SEC_PER_FRAME

                    # Frame indices at 2fps: frame_idx = time * fps + 1 (1-based)
                    frame_start = int(math.ceil(t_start * FPS)) + 1
                    frame_end = int(math.floor(t_end * FPS)) + 1

                    # Collect frame paths for this turn
                    frame_content = []
                    for fi in range(frame_start, min(frame_end + 1, n_frames + 1)):
                        fp = get_frame_path(video_id, fi)
                        if os.path.exists(fp):
                            frame_content.append({
                                "type": "image",
                                "image": f"file://{fp}"
                            })

                    if not frame_content:
                        # Use at least 1 frame
                        fi = min(frame_start, n_frames)
                        fp = get_frame_path(video_id, fi)
                        if os.path.exists(fp):
                            frame_content.append({
                                "type": "image",
                                "image": f"file://{fp}"
                            })

                    if not frame_content:
                        continue

                    # First turn includes the question
                    if turn_idx == 0:
                        frame_content.append({
                            "type": "text",
                            "text": question
                        })

                    conv.append({
                        "role": "user",
                        "content": frame_content
                    })

                if len(conv) < 2:
                    continue

                question_id = f"estp_{video_id}_{qa_id}_{entry_idx}"

                input_data.append({
                    "question_id": question_id,
                    "conversation": conv
                })

                # Build GT for evaluation
                gt_windows = []
                for c in conversations:
                    if c["role"] != "assistant":
                        continue
                    c_start = c.get("start_time", c.get("time", clip_start))
                    c_end = c.get("end_time", c.get("time", clip_start))
                    gt_windows.append({
                        "content": c["content"],
                        "start_time": c_start - clip_start,  # Relative
                        "end_time": c_end - clip_start,
                        "question": question,
                    })

                gt_data[question_id] = {
                    "question": question,
                    "task_type": qa_entry.get("Task Type", "Unknown").strip(),
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "duration": duration,
                    "gt_windows": gt_windows,
                    "n_turns": n_turns,
                }

                processed_count += 1

    return input_data, gt_data


def run_inference(input_file: str, output_file: str, limit: int = None):
    """Run MMDuet2 inference using ProactiveInferenceClient."""
    import torch
    import copy

    # Add proactive_eval to path for custom model and qwen_vl_utils
    eval_dir = str(MMDUET2_DIR / "proactive_eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    data_list = json.load(open(input_file))
    if limit:
        data_list = data_list[:limit]

    # Check existing results
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                existing_ids.add(json.loads(line)["question_id"])
        print(f"Found {len(existing_ids)} existing results")

    # Load model
    print(f"Loading model from {CKPT_DIR}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(CKPT_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(str(CKPT_DIR))
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    f_out = open(output_file, "a")

    for i, example in enumerate(data_list):
        qid = example["question_id"]
        if qid in existing_ids:
            print(f"  [{i+1}/{len(data_list)}] Skip {qid} (exists)")
            continue

        print(f"  [{i+1}/{len(data_list)}] Processing {qid} ({len(example['conversation'])} turns)...")
        t0 = time.time()

        # Turn-by-turn inference with KV cache
        conversation = example["conversation"]
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        past_key_values = None
        video_time = 0.0
        prev_image_inputs = []
        if hasattr(model, 'reset_status'):
            model.reset_status()
        else:
            model.model.all_keep_masks = list()
            model.model.all_drop_ratios = list()
            if hasattr(model.model, 'all_last_frames'):
                model.model.all_last_frames = None

        for turn_idx, turn in enumerate(conversation):
            history.append(turn)

            # Process new images
            new_image_inputs, new_video_inputs = process_vision_info([turn])
            if new_image_inputs:
                prev_image_inputs.extend(new_image_inputs)
                video_time += len(new_image_inputs) * (1.0 / FPS)

            text = processor.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )

            image_inputs = copy.deepcopy(prev_image_inputs) if prev_image_inputs else None

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to("cuda:0")

            # Handle DTD keep masks
            if model.model.all_keep_masks:
                keep_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
                old_keep_mask = torch.cat(model.model.all_keep_masks, dim=1)
                keep_mask[:, :old_keep_mask.size(1)] = old_keep_mask
                inputs["input_ids"] = inputs.input_ids[keep_mask].unsqueeze(0)
                inputs["attention_mask"] = inputs.attention_mask[keep_mask].unsqueeze(0)

            model_output = model.generate(
                **inputs,
                max_new_tokens=512,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
                drop_method="none", drop_threshold=1.0, drop_absolute=True,
                do_sample=False,
            )

            past_key_values = model_output.past_key_values
            output_ids = model_output.sequences[:, inputs.input_ids.size(1):]
            reply = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

            history.append({"role": "assistant", "content": reply, "time": video_time})

        # Extract results
        response_list = []
        for h in history:
            if h["role"] == "assistant":
                response_list.append(h)
            elif h["role"] == "user":
                # Simplify user turns (remove image objects)
                content = h.get("content", "")
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    content = " ".join(text_parts).strip()
                if content:
                    response_list.append({"role": "user", "content": content, "time": h.get("time", 0)})

        res = {
            "question_id": qid,
            "model_response_list": response_list,
        }

        f_out.write(json.dumps(res) + "\n")
        f_out.flush()

        elapsed = time.time() - t0
        n_replies = sum(1 for h in history
                        if h["role"] == "assistant" and h.get("content", "") != NO_REPLY)
        print(f"    Done in {elapsed:.1f}s, {n_replies}/{len(conversation)} replies, "
              f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

        # Reset for next example
        past_key_values = None
        if hasattr(model, 'reset_status'):
            model.reset_status()
        else:
            model.model.all_keep_masks = list()
            model.model.all_drop_ratios = list()
            if hasattr(model.model, 'all_last_frames'):
                model.model.all_last_frames = None
        torch.cuda.empty_cache()

    f_out.close()
    print(f"\nInference complete. Results saved to {output_file}")

    del model
    torch.cuda.empty_cache()


def evaluate(pred_file: str, gt_file: str):
    """Evaluate predictions using ESTP-F1 metric.

    ESTP-F1 measures whether the model responds within GT time windows
    with correct content.
    """
    with open(pred_file) as f:
        predictions = {json.loads(line)["question_id"]: json.loads(line)
                       for line in f}

    with open(gt_file) as f:
        gt_data = json.load(f)

    # Metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_content_scores = []
    results_by_type = {}

    for qid, gt in gt_data.items():
        if qid not in predictions:
            # All GT windows are missed
            total_fn += len(gt["gt_windows"])
            continue

        pred = predictions[qid]
        responses = pred["model_response_list"]

        # Extract model responses with timing
        model_replies = []
        for resp in responses:
            if resp["role"] == "assistant" and resp.get("content", "") != NO_REPLY:
                t = resp.get("time", 0)
                model_replies.append({
                    "time": t,
                    "content": resp["content"],
                })

        # Match replies to GT windows
        gt_windows = gt["gt_windows"]
        matched_gt = set()
        matched_pred = set()

        for gi, gw in enumerate(gt_windows):
            gt_start = gw["start_time"]
            gt_end = gw["end_time"]

            for pi, pr in enumerate(model_replies):
                if pi in matched_pred:
                    continue
                # Check if reply falls within GT window (with 2s tolerance)
                if gt_start - 2 <= pr["time"] <= gt_end + 2:
                    matched_gt.add(gi)
                    matched_pred.add(pi)

                    # Simple content similarity (word overlap)
                    gt_words = set(gw["content"].lower().split())
                    pred_words = set(pr["content"].lower().split())
                    if gt_words and pred_words:
                        overlap = len(gt_words & pred_words)
                        precision = overlap / len(pred_words) if pred_words else 0
                        recall = overlap / len(gt_words) if gt_words else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        total_content_scores.append(f1)
                    break

        tp = len(matched_gt)
        fp = len(model_replies) - len(matched_pred)
        fn = len(gt_windows) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-type tracking
        task_type = gt.get("task_type", "Unknown")
        if task_type not in results_by_type:
            results_by_type[task_type] = {"tp": 0, "fp": 0, "fn": 0}
        results_by_type[task_type]["tp"] += tp
        results_by_type[task_type]["fp"] += fp
        results_by_type[task_type]["fn"] += fn

    # Compute overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_content = sum(total_content_scores) / len(total_content_scores) if total_content_scores else 0

    print("\n" + "=" * 60)
    print("MMDuet2 Zero-Shot Evaluation on ESTP-Bench")
    print("=" * 60)
    print(f"Total QA evaluated: {len(gt_data)}")
    print(f"Total predictions:  {len(predictions)}")
    print(f"\nESTP-F1: {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Avg content F1: {avg_content:.3f}")

    print(f"\nPer Task Type:")
    for task_type, counts in sorted(results_by_type.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        type_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {task_type:30s}: F1={type_f1:.3f} (P={p:.3f} R={r:.3f} TP={tp} FP={fp} FN={fn})")

    # Save results
    results = {
        "overall": {
            "f1": f1, "precision": precision, "recall": recall,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "avg_content_f1": avg_content,
            "n_evaluated": len(gt_data),
            "n_predicted": len(predictions),
        },
        "per_type": {
            t: {"f1": 2*c["tp"]/(2*c["tp"]+c["fp"]+c["fn"]) if (2*c["tp"]+c["fp"]+c["fn"])>0 else 0, **c}
            for t, c in results_by_type.items()
        }
    }

    eval_result_file = OUTPUT_DIR / "eval_results.json"
    with open(eval_result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MMDuet2 zero-shot eval on ESTP-Bench")
    parser.add_argument("--prepare", action="store_true", help="Prepare input data (extract frames + convert)")
    parser.add_argument("--infer", action="store_true", help="Run inference (needs GPU)")
    parser.add_argument("--eval", action="store_true", help="Evaluate predictions")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA entries")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = str(OUTPUT_DIR / "estp_bench-2_sec_per_frame-frame_input_format.json")
    gt_file = str(OUTPUT_DIR / "estp_bench-gt.json")
    pred_file = str(OUTPUT_DIR / "pred.jsonl")

    if args.prepare or args.all:
        print("=" * 60)
        print("Phase 1: Preparing data")
        print("=" * 60)

        data = load_estp_data()
        input_data, gt_data = convert_estp_to_frame_input(data, limit=args.limit)

        with open(input_file, "w") as f:
            json.dump(input_data, f, indent=2)
        print(f"Saved {len(input_data)} entries to {input_file}")

        with open(gt_file, "w") as f:
            json.dump(gt_data, f, indent=2)
        print(f"Saved {len(gt_data)} GT entries to {gt_file}")

    if args.infer or args.all:
        print("\n" + "=" * 60)
        print("Phase 2: Running inference")
        print("=" * 60)
        run_inference(input_file, pred_file, limit=args.limit)

    if args.eval or args.all:
        print("\n" + "=" * 60)
        print("Phase 3: Evaluating")
        print("=" * 60)
        if not os.path.exists(pred_file):
            print(f"ERROR: Prediction file not found: {pred_file}")
            print("Run --infer first.")
            return
        evaluate(pred_file, gt_file)


if __name__ == "__main__":
    main()
