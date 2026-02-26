#!/usr/bin/env python3
"""
Context-Enhanced Reasoning Trigger (CERT) Experiment
=====================================================
Extends the vanilla Reasoning Trigger with temporal context to reduce yes-bias.

Three methods:
  - baseline: Original RT (no context)
  - A: Judgment history injection (lightweight)
  - B: Scene description + judgment history (medium)
  - C: Multi-frame comparison + judgment history (rich)

Uses Gemini-3-Flash via google.genai SDK (Vertex AI backend).

Usage:
    cd /home/v-tangxin/GUI
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/context_enhanced_rt.py --method baseline --verbose
    python3 proactive-project/experiments/estp_phase3/context_enhanced_rt.py --method A --verbose
    python3 proactive-project/experiments/estp_phase3/context_enhanced_rt.py --method B --verbose
    python3 proactive-project/experiments/estp_phase3/context_enhanced_rt.py --method C --verbose
"""

import argparse
import io
import json
import math
import os
import re
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

# ── Constants ──────────────────────────────────────────────────────────────
ANTICIPATION = 1.0
LATENCY = 2.0
COOLDOWN = 12.0
BUCKET_INTERVAL = 10   # seconds between VLM calls
N_FRAMES = 5           # frames per VLM call (baseline, A, B)
N_FRAMES_C = 7         # frames for method C (history + current)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET = SCRIPT_DIR / "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = SCRIPT_DIR / "../../../data/ESTP-Bench/full_scale_2fps_max384"
RESULTS_DIR = SCRIPT_DIR / "results" / "context_enhanced_rt"

# 5 representative case IDs
TARGET_CASE_IDS = [
    "0eb97247e799",  # Object Recognition, extreme yes-bias (97%)
    "ec8c429db0d8",  # Text-Rich Understanding, RT best (F1=0.667)
    "4a51f9ebaa22",  # Object Function, medium yes-bias (47%)
    "85a99403e624",  # Object State Change Recognition, precise but low recall
    "2098c3e8d904",  # Action Recognition, RT worse than BL
]


# ── Prompt Templates ──────────────────────────────────────────────────────

# Baseline (original RT)
BASELINE_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to decide: is RIGHT NOW the best moment to answer their question?"
)

BASELINE_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

Is the user RIGHT NOW at the moment where the answer to their question is most relevant and useful?

Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene that the question asks about
2. The user appears to be actively engaging with or looking at it
3. Providing the answer NOW would be more useful than waiting

Say NO if the relevant object/action is not visible, or the user is doing something unrelated.

Respond with EXACTLY one word: YES or NO"""


# Method A: Judgment history injection
CONTEXT_A_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to decide: is RIGHT NOW the best moment to answer their question?\n\n"
    "IMPORTANT: You are making this judgment as part of a CONTINUOUS monitoring stream. "
    "You should NOT trigger repeatedly for the same unchanged situation."
)

CONTEXT_A_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

=== YOUR RECENT JUDGMENT HISTORY ===
{judgment_history}
===

Is the user RIGHT NOW at the moment where the answer to their question is most relevant and useful?

Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene that the question asks about
2. The user appears to be actively engaging with or looking at it
3. Providing the answer NOW would be more useful than waiting
4. This is a GENUINELY NEW opportunity compared to your recent judgments — something has changed

Say NO if:
- The relevant object/action is not visible or the user is doing something unrelated
- The scene looks essentially the same as when you last said YES
- You already triggered recently and nothing has meaningfully changed

Respond with EXACTLY one word: YES or NO"""


# Method B: Scene description + judgment history
CONTEXT_B_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to: (1) describe what you see, and (2) decide if NOW is the right time to help.\n\n"
    "You are monitoring CONTINUOUSLY. Use the scene history to understand temporal context."
)

CONTEXT_B_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

=== SCENE HISTORY (what happened recently) ===
{scene_history}
===

=== YOUR JUDGMENT HISTORY ===
{judgment_history}
===

Based on the frames AND the scene history above, answer TWO questions:

1. SCENE DESCRIPTION: In one sentence, describe what the user is currently doing/seeing that is DIFFERENT from the previous observation. If nothing changed, say "No significant change."

2. TRIGGER DECISION: Should you provide the answer NOW?
   - YES only if there is a NEW, SPECIFIC visual event relevant to the question (not a continuation of the same scene)
   - NO if the scene is unchanged, or the relevant content is not visible

Format your response as:
SCENE: <one sentence>
DECISION: YES or NO"""


# Method C: Multi-frame comparison + judgment history
CONTEXT_C_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view over time. "
    "I'm showing you frames from MULTIPLE time points so you can see how the scene has evolved. "
    "Your job is to decide if RIGHT NOW is the best moment to answer the user's question."
)

CONTEXT_C_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you frames from multiple time points:
{frame_description}

=== YOUR JUDGMENT HISTORY ===
{judgment_history}
===

Compare the CURRENT frames with the EARLIER frames. Has a MEANINGFUL CHANGE occurred that makes NOW the right time to answer?

Say YES only if:
1. Something NEW and RELEVANT has appeared or changed compared to the earlier frames
2. The user is actively engaging with the relevant object/action/scene
3. This is not just a continuation of the same activity shown in earlier frames

Say NO if:
- The scene is essentially the same across all frames
- No new relevant information has appeared
- The change is trivial (slight camera movement, lighting change)

Respond with EXACTLY one word: YES or NO"""


# ── ContextState Class ────────────────────────────────────────────────────

class ContextState:
    """Manages temporal context across polling steps."""

    def __init__(self, max_history=5, max_scenes=4):
        self.judgments = []           # [(t, decision_bool, triggered_bool)]
        self.scene_descriptions = []  # [(t, description_str)]
        self.last_trigger_time = -999.0
        self.max_history = max_history
        self.max_scenes = max_scenes

    def add_judgment(self, t, decision, triggered):
        self.judgments.append((t, decision, triggered))
        if len(self.judgments) > self.max_history:
            self.judgments.pop(0)
        if triggered:
            self.last_trigger_time = t

    def add_scene(self, t, description):
        self.scene_descriptions.append((t, description))
        if len(self.scene_descriptions) > self.max_scenes:
            self.scene_descriptions.pop(0)

    def format_judgment_history(self, current_t):
        if not self.judgments:
            return "No previous judgments yet. This is your first observation."
        lines = []
        for t, dec, trig in self.judgments:
            ago = current_t - t
            if trig:
                lines.append(f"- {ago:.0f}s ago: YES -> TRIGGERED (provided answer)")
            elif dec:
                lines.append(f"- {ago:.0f}s ago: YES (but in cooldown, not triggered)")
            else:
                lines.append(f"- {ago:.0f}s ago: NO")
        if self.last_trigger_time > 0:
            since_trigger = current_t - self.last_trigger_time
            lines.append(f"[You last triggered {since_trigger:.0f} seconds ago]")
        else:
            lines.append("[You have not triggered yet in this session]")
        return "\n".join(lines)

    def format_scene_history(self, current_t):
        if not self.scene_descriptions:
            return "No scene observations yet."
        lines = []
        for t, desc in self.scene_descriptions:
            ago = current_t - t
            lines.append(f'- {ago:.0f}s ago: "{desc}"')
        return "\n".join(lines)


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


def load_cases(checkpoint_path, dataset_path, case_ids=None):
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
            if case_ids and c.get("case_id") not in case_ids:
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


def extract_frames(video_path, t_end, n_frames=N_FRAMES, fps_hint=2.0):
    """Extract n_frames ending at t_end."""
    step = 1.0 / fps_hint
    times = [max(0.0, t_end - step * (n_frames - 1 - i)) for i in range(n_frames)]
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


def extract_frames_multi_window(video_path, t_current, clip_start,
                                 last_trigger_time, fps_hint=2.0):
    """Extract frames from multiple time windows for method C.
    Returns (frame_bytes_list, frame_description_str)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], ""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # History frames: ~30s ago (2 frames) + ~15s ago (2 frames)
    history_times = []
    for delta in [30.0, 28.0, 15.0, 13.0]:
        t = max(clip_start, t_current - delta)
        history_times.append(t)

    # If last trigger was within 30s, use trigger moment as reference
    if last_trigger_time > 0 and (t_current - last_trigger_time) < 30:
        history_times[0] = last_trigger_time
        history_times[1] = min(last_trigger_time + 0.5, t_current - 1.0)

    # Current frames: most recent 3 frames
    step = 1.0 / fps_hint
    current_times = [max(clip_start, t_current - step * (3 - 1 - i)) for i in range(3)]

    all_times = history_times + current_times
    frames = []
    for t in all_times:
        idx = min(int(t * fps), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames.append(buf.getvalue())
    cap.release()

    # Build description
    t_ref_ago_1 = t_current - history_times[0]
    t_ref_ago_2 = t_current - history_times[2]
    desc = (
        f"- Frames 1-2: from ~{t_ref_ago_1:.0f} seconds ago (reference point)\n"
        f"- Frames 3-4: from ~{t_ref_ago_2:.0f} seconds ago\n"
        f"- Frames 5-7: current view (most recent)"
    )
    return frames, desc


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


def call_vlm(frame_bytes_list, system_prompt, user_prompt,
             model_name="gemini-2.0-flash", verbose=False):
    """Send frames + prompt to VLM, return raw text."""
    from google.genai.types import GenerateContentConfig, Part

    client = _get_client()

    content_parts = [Part.from_bytes(data=fb, mime_type="image/jpeg")
                     for fb in frame_bytes_list]
    content_parts.append(Part.from_text(text=system_prompt + "\n\n" + user_prompt))

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=GenerateContentConfig(
                max_output_tokens=512,
                temperature=0.0,
            ),
        )
        raw = (resp.text or "").strip()
        return raw
    except Exception as e:
        if verbose:
            print(f"    [error] {e}")
        return f"error: {e}"


# ── Output Parsing ────────────────────────────────────────────────────────

def parse_yes_no(raw_text):
    """Parse YES/NO from raw VLM output."""
    return raw_text.strip().lower().startswith("yes")


def parse_scene_and_decision(raw_text):
    """Parse method B's two-part output: SCENE + DECISION."""
    scene = ""
    decision = False

    for line in raw_text.split("\n"):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("SCENE:"):
            scene = line_stripped[6:].strip()
        elif line_stripped.upper().startswith("DECISION:"):
            dec_text = line_stripped[9:].strip().lower()
            decision = dec_text.startswith("yes")

    # Fallback: if no structured output, check for YES anywhere
    if not scene and "SCENE:" not in raw_text.upper():
        scene = raw_text[:100]
        decision = "yes" in raw_text.lower().split("\n")[0] if raw_text else False

    return scene, decision


# ── ESTP-F1 Metric ───────────────────────────────────────────────────────

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

    return {"f1": f1, "precision": precision, "recall": recall,
            "fp": fp, "tp": tp, "fn": fn}


# ── Per-Case Evaluation ──────────────────────────────────────────────────

def evaluate_case(case, video_root, model_name, method="baseline",
                  interval=BUCKET_INTERVAL, cooldown=COOLDOWN, verbose=False):
    """Run one method on one case."""
    video_uid = case["video_uid"]
    duration = case.get("duration", 300.0)
    goal = case.get("question", "")
    gt_windows = get_gt_windows(case["qa_raw"])
    qa_raw = case.get("qa_raw", {})
    clip_start = qa_raw.get("clip_start_time", 0.0)

    video_path = find_video(video_uid, str(video_root))
    if video_path is None:
        return None

    if verbose and clip_start > 0:
        print(f"    clip_start_time={clip_start:.1f}s")

    n_buckets = int(math.ceil(duration / interval)) + 1
    buckets = [i * interval for i in range(n_buckets) if i * interval <= duration + interval]

    trigger_times = []
    last_trigger = -cooldown - 1.0
    decisions = []
    ctx = ContextState()

    for t_b in buckets:
        t_abs = clip_start + t_b

        # ── Get frames ──
        if method == "C":
            frame_bytes, frame_desc = extract_frames_multi_window(
                str(video_path), t_abs, clip_start, ctx.last_trigger_time
            )
        else:
            frame_bytes = extract_frames(str(video_path), float(t_abs))
            frame_desc = ""

        if not frame_bytes:
            decisions.append((t_b, False, False, "no_frames"))
            continue

        # ── Build prompt ──
        if method == "baseline":
            sys_prompt = BASELINE_SYSTEM
            user_prompt = BASELINE_USER_TEMPLATE.format(
                goal=goal, n_frames=len(frame_bytes)
            )
        elif method == "A":
            sys_prompt = CONTEXT_A_SYSTEM
            user_prompt = CONTEXT_A_USER_TEMPLATE.format(
                goal=goal,
                n_frames=len(frame_bytes),
                judgment_history=ctx.format_judgment_history(t_abs),
            )
        elif method == "B":
            sys_prompt = CONTEXT_B_SYSTEM
            user_prompt = CONTEXT_B_USER_TEMPLATE.format(
                goal=goal,
                n_frames=len(frame_bytes),
                scene_history=ctx.format_scene_history(t_abs),
                judgment_history=ctx.format_judgment_history(t_abs),
            )
        elif method == "C":
            sys_prompt = CONTEXT_C_SYSTEM
            user_prompt = CONTEXT_C_USER_TEMPLATE.format(
                goal=goal,
                frame_description=frame_desc,
                judgment_history=ctx.format_judgment_history(t_abs),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Call VLM ──
        raw = call_vlm(frame_bytes, sys_prompt, user_prompt, model_name, verbose)
        time.sleep(0.3)  # rate limit

        # ── Parse output ──
        if method == "B":
            scene, decision = parse_scene_and_decision(raw)
            ctx.add_scene(t_abs, scene)
        else:
            decision = parse_yes_no(raw)

        # ── Cooldown + trigger ──
        triggered = decision and (t_abs - last_trigger) >= cooldown
        if triggered:
            trigger_times.append(float(t_abs))
            last_trigger = float(t_abs)

        # ── Update context ──
        ctx.add_judgment(t_abs, decision, triggered)
        decisions.append((t_b, decision, triggered, raw[:60]))

        # ── Verbose output ──
        if verbose:
            marker = " << TRIGGER" if triggered else ""
            gt_mark = ""
            for gs, ge in gt_windows:
                if gs - ANTICIPATION <= t_abs <= gs + LATENCY:
                    gt_mark = " [GT]"
                    break
            dec_str = "YES" if decision else "no "
            extra = ""
            if method == "B" and "SCENE:" in raw.upper():
                extra = f" scene={raw[:40]}"
            print(f"    t={t_b:6.1f}s (abs={t_abs:.1f}s)  {dec_str}  "
                  f"raw={raw[:25]:<25s}{gt_mark}{marker}{extra}")

    metrics = compute_estp_f1(trigger_times, gt_windows, duration)
    metrics["trigger_times"] = trigger_times
    metrics["n_triggers"] = len(trigger_times)
    metrics["n_buckets"] = len(buckets)
    metrics["n_yes"] = sum(1 for _, d, _, _ in decisions if d)
    metrics["yes_rate"] = metrics["n_yes"] / max(metrics["n_buckets"], 1)
    metrics["decisions"] = decisions

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Context-Enhanced Reasoning Trigger")
    parser.add_argument("--method", default="baseline",
                        choices=["baseline", "A", "B", "C"],
                        help="Method to run")
    parser.add_argument("--case_ids", nargs="*", default=None,
                        help="Specific case IDs (default: 5 representative cases)")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--interval", type=int, default=BUCKET_INTERVAL)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    case_ids = args.case_ids or TARGET_CASE_IDS

    print("=" * 70)
    print(f"Context-Enhanced Reasoning Trigger: Method {args.method}")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Interval: {args.interval}s  |  Cooldown: {COOLDOWN}s")
    print(f"Target cases: {case_ids}")
    print()

    cases = load_cases(args.checkpoint, args.dataset, set(case_ids))
    # Sort to match TARGET_CASE_IDS order
    case_order = {cid: i for i, cid in enumerate(case_ids)}
    cases.sort(key=lambda c: case_order.get(c.get("case_id", ""), 999))

    print(f"Loaded {len(cases)} cases\n")

    results = []
    t0_total = time.time()

    for i, case in enumerate(cases):
        cid = case["case_id"]
        task_type = case.get("task_type", "?")
        gt_windows = get_gt_windows(case["qa_raw"])

        print(f"[{i+1}/{len(cases)}] {cid}  [{task_type}]  GT={len(gt_windows)} windows")
        print(f"  Q: {case.get('question', '')[:80]}")

        t0 = time.time()
        metrics = evaluate_case(
            case, args.video_root, args.model, method=args.method,
            interval=args.interval, verbose=args.verbose,
        )
        elapsed = time.time() - t0

        if metrics is None:
            print(f"  [skip] video not found")
            continue

        metrics["case_id"] = cid
        metrics["task_type"] = task_type
        metrics["method"] = args.method
        results.append(metrics)

        print(f"  F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"FP={metrics['fp']}  triggers={metrics['n_triggers']}  "
              f"yes_rate={metrics['n_yes']}/{metrics['n_buckets']} ({metrics['yes_rate']:.1%})  "
              f"({elapsed:.1f}s)")
        print()

    total_time = time.time() - t0_total

    # ── Summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"SUMMARY: Method {args.method}")
    print("=" * 70)

    if not results:
        print("No results!")
        return

    avg_f1 = np.mean([r["f1"] for r in results])
    total_triggers = sum(r["n_triggers"] for r in results)
    total_yes = sum(r["n_yes"] for r in results)
    total_buckets = sum(r["n_buckets"] for r in results)
    total_fp = sum(r["fp"] for r in results)

    print(f"  Avg F1:           {avg_f1:.3f}")
    print(f"  Total triggers:   {total_triggers}")
    print(f"  Total FP:         {total_fp}")
    print(f"  Global yes-rate:  {total_yes}/{total_buckets} ({total_yes/max(total_buckets,1):.1%})")
    print(f"  Time: {total_time/60:.1f} min  ({total_time/len(results):.1f}s/case)")

    # Per-case detail
    print(f"\n  {'Case':<15} {'Type':<35} {'F1':>6} {'P':>6} {'R':>6} {'FP':>4} "
          f"{'Trigs':>6} {'YR':>8}")
    print(f"  {'─'*13}  {'─'*33}  {'─'*5} {'─'*5} {'─'*5} {'─'*3} {'─'*5} {'─'*7}")
    for r in results:
        yr = f"{r['n_yes']}/{r['n_buckets']}"
        print(f"  {r['case_id']:<15} {r['task_type']:<35} {r['f1']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['fp']:>4} "
              f"{r['n_triggers']:>6} {yr:>8}")

    print(f"\n{'='*70}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"method_{args.method}.json"
    save_data = {
        "method": args.method,
        "model": args.model,
        "interval": args.interval,
        "cooldown": COOLDOWN,
        "n_cases": len(results),
        "avg_f1": avg_f1,
        "total_fp": total_fp,
        "global_yes_rate": total_yes / max(total_buckets, 1),
        "total_time_min": total_time / 60,
        "results": [{k: v for k, v in r.items() if k != "decisions"}
                    for r in results],
    }
    with open(output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
