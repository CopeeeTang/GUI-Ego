#!/usr/bin/env python3
"""
Phase III Stage B: CLIP Dynamics Extractor
===========================================
Precomputes two independent visual signals for each trigger_log step:

  clip_change:     1 - cosine_sim(frame_t, frame_{t-1})
                   Captures: "how much has the scene changed?"
                   → High = scene transition / new object appeared
                   → Low  = stable/repetitive action

  clip_relevance:  cosine_sim(frame_t, text_embed(question))
                   Captures: "how visually aligned is the scene with the task goal?"
                   → High = task-relevant objects/actions visible
                   → Low  = off-task / irrelevant scene

Both signals are INDEPENDENT of logprob_gap (D-features).
CLIP can run on CPU (slow ~30min) or GPU (fast ~5min).

Output: results/phase3_learned/clip_cache.json
  {case_id: [{time, clip_change, clip_relevance}, ...]}

Resumable: already-computed cases are preserved.

Usage:
    source ml_env/bin/activate
    python3 proactive-project/experiments/estp_phase3/phase3_clip_extractor.py
    python3 proactive-project/experiments/estp_phase3/phase3_clip_extractor.py \\
        --device cuda  \\
        --batch-size 64 \\
        --verbose
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CHECKPOINT = "results/fullscale_d/checkpoint.jsonl"
DEFAULT_DATASET    = "../../../data/ESTP-Bench/estp_dataset/estp_bench_sq_5_cases.json"
DEFAULT_VIDEO_ROOT = "../../../data/ESTP-Bench/full_scale_2fps_max384"
DEFAULT_OUTPUT     = "results/phase3_learned/clip_cache.json"
DEFAULT_MODEL      = "ViT-B/32"   # fast and well-tested; ViT-L/14 for more capacity


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


def extract_all_frames(video_path, times):
    """
    Extract frames at given timestamps from MP4.
    Returns list of PIL Images (None for failed frames).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [None] * len(times)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for t in times:
        idx = min(int(t * fps), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(None)
    cap.release()
    return frames


# ── CLIP Inference ─────────────────────────────────────────────────────────

def load_clip_model(model_name, device):
    """Load CLIP model. Tries openai/clip, then transformers as fallback."""
    try:
        import clip  # pip install clip / openai-clip
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        print(f"  Loaded CLIP {model_name} via openai/clip")
        return "openai_clip", model, preprocess
    except ImportError:
        pass

    # Fallback: transformers CLIP
    from transformers import CLIPModel, CLIPProcessor
    hf_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(hf_name)
    model = CLIPModel.from_pretrained(hf_name).to(device)
    model.eval()
    print(f"  Loaded CLIP via transformers ({hf_name})")
    return "transformers", model, processor


def embed_images_batch(backend, model, preprocess_or_processor, images, device,
                       batch_size=64):
    """
    Compute image embeddings for a list of PIL Images.
    Returns: np.array of shape (N, D), L2-normalized.
    """
    import torch
    valid = [(i, img) for i, img in enumerate(images) if img is not None]
    out = np.zeros((len(images), 512), dtype=np.float32)  # 512 for ViT-B/32

    for start in range(0, len(valid), batch_size):
        batch = valid[start: start + batch_size]
        idxs  = [b[0] for b in batch]
        imgs  = [b[1] for b in batch]

        if backend == "openai_clip":
            imgs_tensor = torch.stack([preprocess_or_processor(img) for img in imgs]).to(device)
            with torch.no_grad():
                embs = model.encode_image(imgs_tensor)
        else:
            inputs = preprocess_or_processor(images=imgs, return_tensors="pt",
                                              padding=True).to(device)
            with torch.no_grad():
                clip_out = model.get_image_features(**inputs)
                # transformers >=5.x returns BaseModelOutputWithPooling
                embs = clip_out.pooler_output if hasattr(clip_out, "pooler_output") else clip_out

        embs = embs.float().cpu().numpy()
        # L2-normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-8)
        for j, idx in enumerate(idxs):
            out[idx] = embs[j]

    return out


def embed_texts_batch(backend, model, preprocess_or_processor, texts, device):
    """
    Compute text embeddings for a list of strings.
    Returns: np.array of shape (N, D), L2-normalized.
    """
    import torch
    if backend == "openai_clip":
        import clip
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
    else:
        inputs = preprocess_or_processor(text=texts, return_tensors="pt",
                                          padding=True, truncation=True).to(device)
        with torch.no_grad():
            clip_out = model.get_text_features(**inputs)
            # transformers >=5.x returns BaseModelOutputWithPooling
            embs = clip_out.pooler_output if hasattr(clip_out, "pooler_output") else clip_out

    embs = embs.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


# ── Per-Case CLIP Feature Computation ─────────────────────────────────────

def compute_clip_features_for_case(case, video_root, backend, model, preprocess,
                                   device, batch_size, verbose=False):
    """
    Compute clip_change + clip_relevance for every trigger_log step.

    clip_change[i]     = 1 - cosine_sim(emb[i], emb[i-1])   (0.0 for first step)
    clip_relevance[i]  = cosine_sim(emb[i], goal_text_emb)

    Returns: list of dicts [{time, clip_change, clip_relevance}, ...]
    """
    tlog      = case.get("trigger_log", [])
    video_uid = case["video_uid"]
    goal      = case.get("question", "")

    if not tlog:
        return []

    video_path = find_video(video_uid, video_root)
    if video_path is None:
        if verbose:
            print(f"    [warn] video not found: {video_uid}")
        # Return zeros so MLP can still train (feature will be uninformative)
        return [{"time": e["time"], "clip_change": 0.0, "clip_relevance": 0.0}
                for e in tlog]

    times = [e["time"] for e in tlog]

    # Extract all frames in one video pass
    if verbose:
        print(f"    Extracting {len(times)} frames from {video_path.name}...")
    frames = extract_all_frames(video_path, times)

    # Batch-compute image embeddings
    embs = embed_images_batch(backend, model, preprocess, frames, device, batch_size)

    # Compute goal text embedding
    goal_emb = embed_texts_batch(backend, model, preprocess, [goal], device)[0]

    # Compute per-step features
    results = []
    for i, entry in enumerate(tlog):
        emb_cur = embs[i]
        # clip_change: cosine distance from previous step
        if i == 0 or np.all(embs[i - 1] == 0):
            clip_change = 0.0
        else:
            cos_sim = float(np.dot(emb_cur, embs[i - 1]))  # already L2-normalized
            clip_change = float(1.0 - max(-1.0, min(1.0, cos_sim)))  # clamp to [0, 2]
            clip_change = clip_change / 2.0  # normalize to [0, 1]

        # clip_relevance: cosine similarity with goal text
        if np.all(emb_cur == 0):
            clip_relevance = 0.0
        else:
            clip_relevance = float(np.dot(emb_cur, goal_emb))
            clip_relevance = max(0.0, clip_relevance)   # clip negative to 0

        results.append({
            "time":            float(entry["time"]),
            "clip_change":     round(clip_change, 5),
            "clip_relevance":  round(clip_relevance, 5),
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase III: CLIP Dynamics Extractor")
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset",     default=DEFAULT_DATASET)
    parser.add_argument("--video-root",  default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--model",       default=DEFAULT_MODEL, help="CLIP model variant")
    parser.add_argument("--device",      default="cuda",
                        help="Device: cuda / cpu")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--limit",       type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    checkpoint  = script_dir / args.checkpoint
    dataset     = script_dir / args.dataset
    video_root  = script_dir / args.video_root
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache (resumable)
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

    # Check device
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [warn] CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"Device: {device}")

    # Load CLIP model
    print(f"Loading CLIP {args.model}...")
    backend, model, preprocess = load_clip_model(args.model, device)

    print(f"Processing {len(cases)} cases...")
    t0_total = time.time()

    for i, case in enumerate(cases):
        cid = case["case_id"]
        if cid in cache:
            if args.verbose:
                print(f"[{i+1:3d}/{len(cases)}] {cid} — already cached")
            continue

        t0 = time.time()
        task_type = case.get("task_type", "?")
        n_steps   = len(case.get("trigger_log", []))
        print(f"[{i+1:3d}/{len(cases)}] {cid}  [{task_type}]  n_steps={n_steps}")

        try:
            result = compute_clip_features_for_case(
                case, str(video_root), backend, model, preprocess,
                device, args.batch_size, verbose=args.verbose
            )
            cache[cid] = result
        except Exception as e:
            print(f"  [ERROR] {e}")
            cache[cid] = []

        elapsed     = time.time() - t0
        total_ela   = time.time() - t0_total
        avg_per     = total_ela / (i + 1)
        remaining   = avg_per * (len(cases) - i - 1)
        print(f"  Done in {elapsed:.1f}s  |  ETA: {remaining/60:.1f} min")

        # Save after each case
        with open(output_path, "w") as f:
            json.dump(cache, f)

    # Final save + summary
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    total_steps = sum(len(v) for v in cache.values())
    print(f"\n{'='*50}")
    print(f"Cache saved: {output_path}")
    print(f"  Cases:  {len(cache)}")
    print(f"  Steps:  {total_steps}")
    print(f"  Total:  {(time.time()-t0_total)/60:.1f} min")

    # Quick stats on clip_change distribution
    changes = [e["clip_change"] for v in cache.values() for e in v if "clip_change" in e]
    if changes:
        changes = np.array(changes)
        print(f"  clip_change: mean={changes.mean():.3f}  std={changes.std():.3f}  "
              f"p90={np.percentile(changes, 90):.3f}")


if __name__ == "__main__":
    main()
