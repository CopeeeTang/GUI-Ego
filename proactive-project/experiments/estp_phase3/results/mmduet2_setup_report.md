# MMDuet2 RL Fine-tuning Setup Report

## 1. Overview

**MMDuet2** (arXiv:2512.06810) is a Video MLLM for proactive interaction, built on Qwen2.5-VL-3B-Instruct.
It uses multi-turn RL (GRPO) to learn when to respond during streaming video, without requiring precise response time annotations.

- Paper: https://arxiv.org/abs/2512.06810
- Code: https://github.com/yellow-binary-tree/mmduet2
- Checkpoints: https://huggingface.co/wangyueqian/MMDuet2
- Data: https://huggingface.co/datasets/wangyueqian/MMDuet2-data

## 2. Architecture and Training Pipeline

### 2.1 Two-Phase Training
1. **SFT Phase**: Uses `ms-swift` (v3.2.0) with Qwen2.5-VL-3B-Instruct base
   - lr=1e-5, batch=1, grad_accum=2, 1 epoch, full fine-tune (ViT frozen)
   - DeepSpeed ZeRO-3, bfloat16
   - Data: ~52K videos from Live-WhisperX, Ego-Exo4D, EgoExoLearn + offline video data

2. **RL Phase**: Uses `verl` (forked from commit 83ebd00) + `sglang` (forked from commit dcae1f)
   - Algorithm: GRPO (Group Relative Policy Optimization)
   - lr=1e-6, temperature=1.2, top_k=10, n=4 rollouts
   - FSDP with param/optimizer offload, gradient checkpointing
   - Data: ~8.9K samples (7.97K train, 883 val), 20s segments

### 2.2 Key Innovation: Multi-Turn RL with PAUC Reward
- Model outputs "NO REPLY" or actual answer at each timestep
- Reward = `3*PAUC + 2*repetition_penalty + 0.5*outspan_penalty + 2*common_prefix_penalty`
  - **PAUC** (Proactive Area Under Curve): Rewards correct answers that come early
  - **Repetition**: LLM-judged penalty for repeating previous answers
  - **Out-of-span**: Penalty for responding outside GT reply windows
  - **Common prefix**: Penalty for copy-pasting previous responses
- Reward function requires an **external LLM evaluator** (Doubao-Seed-1.6 or Qwen3-32B) for similarity scoring

### 2.3 Multi-Turn Tool-Call Architecture
- RL rollout uses a custom `OnlineVideoProvider` tool that feeds video frames turn-by-turn
- Each turn: user sends `<image><image>` tokens -> model responds -> next frames provided
- `sglang` engine handles multi-turn generation with tool calls

## 3. Dependency Analysis

### 3.1 Required Packages (RL)
| Package | MMDuet2 Required | Current ml_env | Compatible? |
|---------|-----------------|----------------|-------------|
| torch | 2.6.0 | 2.10.0+cu128 | CONFLICT - major version diff |
| transformers | 4.49.0 | 5.1.0 | CONFLICT |
| accelerate | 1.7.0 | 1.12.0 | Likely OK |
| flash-attn | 2.7.4 | not installed | NEED INSTALL |
| vllm | 0.8.5 | not installed | NEED INSTALL (heavy) |
| ray | 2.48.0 | not installed | NEED INSTALL (heavy) |
| sglang (custom fork) | custom | not installed | NEED INSTALL |
| verl (custom fork) | custom | not installed | NEED INSTALL |
| apex | custom | not installed | NEED INSTALL (build from source) |
| decord | 0.6.0 | not installed | NEED INSTALL |
| peft | 0.15.2 | 0.18.1 | Likely OK |
| deepspeed | 0.16.3 (SFT) | not installed | NEED INSTALL (SFT only) |

### 3.2 Compatibility Assessment: SEVERE CONFLICTS

**Critical Issues:**
1. **PyTorch version mismatch**: ml_env has torch 2.10.0, MMDuet2 needs 2.6.0. The custom sglang/verl forks are pinned to specific CUDA kernels for torch 2.6.
2. **Transformers version mismatch**: ml_env has 5.1.0, MMDuet2 needs 4.49.0. The codebase includes custom `modeling_qwen2_5_vl_DTD.py` that patches specific internals.
3. **Custom forks**: Both `sglang` and `verl` are modified forks that need `pip install -e .`
4. **apex**: NVIDIA apex needs compilation from source with CUDA extensions.

**Recommendation: Create a NEW conda environment** (cannot share ml_env).

### 3.3 GPU Requirements
- Original training: Multi-node, multi-GPU (script uses NNODES, GPUS_PER_NODE)
- Memory: With FSDP param/optimizer offload + gradient checkpointing:
  - Per-frame token count: ~190 tokens/frame (128 image tokens + overhead)
  - Max prompt length: 90 frames * 190 = 17,100 tokens
  - Max response: 45 frames * 190 = 8,550 tokens
  - With rollout n=4 and 3B model: **~40-60GB estimated** for single-GPU
  - **A100 80GB: Likely feasible for single-GPU with offloading**, but tight

## 4. ESTP-Bench Data Adaptation

### 4.1 ESTP-Bench Format
```json
{
  "video_id": {
    "qa_id": [
      {
        "clip_start_time": 1073.5,
        "clip_end_time": 1382.0,
        "Task Type": " Object Recognition",
        "question": "Can you remind me what the white plastic object being handled is?",
        "conversation": [
          {"role": "assistant", "content": "...", "start_time": 1073.5, "end_time": 1076.5},
          ...
        ]
      }
    ]
  }
}
```

### 4.2 MMDuet2 RL Format
```json
{
  "data_source": "estp_bench",
  "prompt": [
    {"content": "system prompt...", "last_image_time": 0, "role": "system"},
    {"content": "<image><image>", "last_image_time": 2, "role": "user"},
    ...
  ],
  "images": [{"image": "path/to/frame.jpg", "type": "image"}],
  "ability": "proactive",
  "reward_model": {
    "ground_truth": {
      "answer": [{"content": "...", "question": "...", "reply_timespan": [s, e]}],
      "model_pred_timestamps": [t1, t2, ...],
      "question_id": "unique_id"
    },
    "prev_assistant_replies": [],
    "style": "rule"
  },
  "extra_info": {"num_images": N, "tools_kwargs": {...}}
}
```

### 4.3 Key Mapping
| ESTP-Bench | MMDuet2 |
|-----------|---------|
| video mp4 (2fps) | Extract frames as JPG |
| conversation[i].start_time/end_time | reply_timespan |
| conversation[i].content | answer[i].content |
| question | answer[i].question (same for all spans) |
| clip_start_time..clip_end_time | Generate timestamps at 2s intervals |

### 4.4 Frame Extraction
ESTP-Bench videos are pre-processed at 2fps/384p. MMDuet2 also uses 2fps.
Need to extract JPG frames from MP4 files.
- 164 videos, avg ~1400 frames each = ~230K frames total
- At 384x384 JPG quality, ~50-100KB/frame = ~10-20GB disk space

## 5. Feasibility Assessment

### 5.1 What's Feasible on Single A100 80GB

| Phase | Feasibility | Notes |
|-------|------------|-------|
| SFT (full) | MODERATE | Needs DeepSpeed ZeRO-3 with offload. 3B model, should fit. |
| RL (GRPO) | CHALLENGING | Needs actor + ref model + sglang rollout engine. Very tight with 80GB. |
| Inference only | EASY | 3B model, ~8GB. No issues. |

### 5.2 Critical Blockers

1. **LLM Evaluator for RL Reward**: The reward function calls an external LLM (Doubao-Seed-1.6) at training time. We need a substitute:
   - Option A: Use GPT-4o via Azure (http://52.151.57.21:9999) - available but costly per-call
   - Option B: Use a local LLM (Qwen3-8B on same GPU) - but GPU already busy with training
   - Option C: Pre-compute rewards offline with a simpler rule-based function
   - Option D: Skip RL, do SFT only

2. **Environment**: Must create new conda env. ~30min setup + compilation.

3. **Data**: Need to extract frames from ESTP mp4 files and create conversion pipeline.

### 5.3 Recommended Approach

**Priority 1 - Inference/Evaluation (Low Risk):**
- Download MMDuet2 checkpoint from HuggingFace
- Set up inference with proactive_eval code
- Evaluate on ESTP-Bench directly (zero-shot)
- This alone would be valuable for comparison

**Priority 2 - SFT on ESTP-Bench (Medium Risk):**
- Convert ESTP-Bench data to SFT format
- Fine-tune on ESTP-Bench data
- More straightforward than RL, no external LLM needed

**Priority 3 - RL Fine-tuning (High Risk):**
- Full GRPO training requires external LLM evaluator
- Single-GPU may be insufficient for full pipeline
- Consider simplified reward (rule-based F1 instead of LLM-as-judge)

## 6. Data Conversion Script

See `convert_estp_to_mmduet2.py` in the same directory.

## 7. Environment Setup Script

```bash
# Create new conda environment for MMDuet2
conda create -n mmduet2_rl python=3.10
conda activate mmduet2_rl

# Install from MMDuet2 requirements
cd /home/v-tangxin/GUI/proactive-project/mmduet2/rl
pip install -r requirements.txt

# Install apex (build from source)
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..

# Install custom sglang and verl
cd sglang && pip install -e . && cd ..
cd verl && pip install -e . && cd ..
```

## 8. Zero-Shot Pilot Results (5 cases)

Environment: `mmduet2_env` (Python 3.10, torch 2.4.0+cu124, transformers 4.49.0, flash-attn 2.6.3)
Checkpoint: `wangyueqian/MMDuet2` (Qwen2.5-VL-3B, 7.6GB)
Script: `experiments/estp_phase3/mmduet2_estp_eval.py`

### Overall
| Metric | Value |
|--------|-------|
| ESTP-F1 | 0.154 |
| Precision | 1.000 |
| Recall | 0.083 |
| TP=1, FP=0, FN=11 | |
| Content F1 (when replied) | 0.800 |

### Per Task Type
| Task Type | F1 | TP | FP | FN |
|-----------|----|----|----|----|
| Action Reasoning | 0.000 | 0 | 0 | 4 |
| Task Understanding | 0.000 | 0 | 0 | 7 |
| Object Localization | 1.000 | 1 | 0 | 0 |

### Key Observations
- **Model is extremely conservative**: 5 videos, 347 total turns, only 1 reply
- GPU memory: 10-16 GB depending on video length
- Inference speed: ~1.5s/turn for short videos, slower for long ones (710s for 151 turns)
- Long videos (>60s) exceed 128K token limit (warning)
- The one successful reply (Object Localization) had content F1=0.8 -- quality is high when it does reply
- Action Reasoning and Task Understanding: complete silence (F1=0)

### Interpretation
MMDuet2 was trained on web/egocentric videos with different question types. On ESTP-Bench:
- The model's "proactive" training teaches it to wait for clear visual evidence before responding
- ESTP-Bench's egocentric + task-oriented format differs significantly from training distribution
- SFT fine-tuning on ESTP data would likely improve recall dramatically

## 9. Summary

| Aspect | Status |
|--------|--------|
| Code cloned | Done |
| Data format understood | Done |
| Dependency analysis | Done - SEVERE conflicts, need new env |
| Data conversion script | Done |
| Environment setup | Done (uv venv, Python 3.10) |
| MMDuet2 checkpoint | Done (7.6GB downloaded) |
| Zero-shot eval script | Done (end-to-end working) |
| Zero-shot pilot (5 cases) | Done - F1=0.154 |
| SFT training | NOT STARTED |
| RL training | NOT STARTED - high risk for single GPU |
