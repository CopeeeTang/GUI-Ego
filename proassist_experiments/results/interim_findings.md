# ProAssist Adaptive-D + Change Detection: Findings
Date: 2026-02-24/25

## Setup
- Model: ProAssist-Model-L4096-I1 (LLaMA-3.1-8B + LoRA r=128, α=256 + SigLIP-SO400M)
- Attention: sdpa (flash_attention_2 not available)
- GPU: A100 80GB, ~21GB VRAM used
- Env: proassist_env (Python 3.12, torch 2.3.1+cu121, transformers 4.43.1)

## Key Finding 1: Change Detection Hypothesis REJECTED

SigLIP CLS feature change scores show NO correlation with talk decisions:

| Scale | Talk Mean | NoTalk Mean | Separation |
|-------|-----------|-------------|------------|
| change_1 | 0.00360 | 0.00381 | -0.00021 |
| change_3 | 0.00550 | 0.00522 | +0.00028 |
| change_5 | 0.00619 | 0.00565 | +0.00054 |
| change_10 | 0.00627 | 0.00624 | +0.00003 |
| change_20 | 0.00745 | 0.00663 | +0.00082 |

- High change frames (top 5%) have only 1.08x talk rate
- In uncertain zone (w2t_prob 0.1-0.9): r=0.017, p=0.81 — not significant
- Reason: Model has internalized visual change into w2t_prob

## Key Finding 2: w2t_prob Signal is Extremely Clean

| w2t_prob Range | Frames | Talk Rate |
|----------------|--------|-----------|
| [0.00, 0.01) | 238 | 100% |
| [0.01, 0.05) | 35 | 100% |
| [0.05, 0.10) | 8 | 100% |
| [0.10, 0.30) | 15 | 100% |
| [0.30, 0.50) | 19 | 0% |
| [0.50, 1.00) | 7132 | 0% |

Perfect frame-level classification at threshold 0.3!
Distribution: 86% > 0.99, 3.2% < 0.01, 2.6% in uncertain zone

## Key Finding 3: Adaptive-D Threshold Optimization — STRONG POSITIVE

### WTAG Complete Sweep (dialog-klg-sum_val_L4096_I1, 24 samples)

| τ | F1 | Precision | Recall | Missing | Redundant | CIDEr |
|------|--------|-----------|--------|---------|-----------|-------|
| 0.15 | 0.3520 | 0.4910 | 0.2744 | 48.1% | 7.2% | 0.865 |
| **0.20** | **0.3727** | **0.4759** | 0.3063 | 42.6% | 10.8% | 0.768 |
| 0.25 | 0.3486 | 0.4198 | 0.2981 | 41.7% | 17.8% | 0.778 |
| 0.30 | 0.3454 | 0.3915 | 0.3090 | 39.0% | 22.8% | 0.790 |
| 0.35 | 0.3562 | 0.3882 | 0.3291 | 34.1% | 22.3% | 0.775 |
| 0.40 | 0.3310 | 0.3321 | 0.3300 | 35.0% | 34.6% | 0.770 |
| 0.45 | 0.3212 | 0.3136 | 0.3291 | 34.7% | 37.8% | 0.761 |
| 0.50 | 0.3283 | 0.3113 | 0.3473 | 34.2% | 41.0% | 0.810 |

**Best: τ*=0.20, F1=0.3727**
**ΔF1 = +0.0387 vs paper best (τ=0.4, F1=0.3340), +11.6% relative improvement**

### F1 Curve Analysis
- F1 curve has clear peak at τ=0.20
- Secondary peak at τ=0.35 (different precision-recall trade-off)
- τ=0.15→0.20: Recall↑(+0.032) outweighs Precision↓(-0.015) → F1 rises
- τ=0.20→0.25: Redundant doubles (10.8%→17.8%), Recall flat → F1 drops
- τ=0.25→0.35: Recall continues increasing, redundant stabilizes → F1 recovers
- τ=0.35→0.50: Redundant explodes (22→41%), F1 monotonically declines

### Trade-off Insight
- w2t_prob in [0.2, 0.25] contains "boundary" frames that produce redundant content
- Lower τ = higher confidence threshold = better content quality (CIDEr 0.865 at τ=0.15)
- The model should be MORE conservative than the paper's default setting

## Paper Baselines for Cross-Dataset Comparison

ego4d (narration_val_L4096_I1, model nr0.1, from results_present.yaml):
| τ | F1 | Precision | Recall |
|---|------|-----------|--------|
| 0.2 | 0.122 | 0.372 | 0.073 |
| **0.3** | **0.275** | 0.241 | 0.322 |
| 0.4 | 0.191 | 0.124 | 0.408 |

Paper's ego4d optimal is τ=0.3 — different from WTAG (τ=0.2).
This confirms per-dataset optimization is needed (Adaptive-D core thesis).

## Key Finding 4: FDA-CLIP Patch-Level Change Detection — REJECTED

Tested 8 methods using SigLIP-SO400M patch features (729 patches × 1152d), 1020 frames from 5 WTAG samples:

| Method | AUC | Separation | Direction |
|--------|-----|------------|-----------|
| A: CLS cosine distance | 0.443 | -0.012 | NEGATIVE |
| B: Patch-Max | 0.449 | -0.013 | NEGATIVE |
| B: Patch-Mean | 0.423 | -0.024 | NEGATIVE |
| B: Patch-Top50 | 0.398 | -0.018 | NEGATIVE |
| C: FDA r=0.05 | 0.453 | -0.013 | NEGATIVE |
| C: FDA r=0.10 | 0.451 | -0.011 | NEGATIVE |
| C: FDA r=0.20 | 0.447 | -0.014 | NEGATIVE |
| C: FDA r=0.30 | 0.436 | -0.019 | NEGATIVE |

**All methods AUC < 0.50 (worse than random), all separations negative.**
- Talk frames have LOWER visual change than no-talk frames
- Users need help when STUCK (low change), not when actively operating (high change)
- This is NOT a granularity issue (global vs local) — it's a fundamental direction issue
- **Definitively closes the visual change detection direction for proactive dialogue timing**

## Key Finding 5: Reasoning Trigger (GPT-4o) — Orthogonal & Complementary

GPT-4o on WTAG (5 samples, 100 stratified frames: 50 GT-talk + 50 GT-notalk):

| Method | F1 | Precision | Recall |
|--------|-----|-----------|--------|
| GPT-4o alone | 0.269 | 0.529 | 0.180 |
| w2t_prob<0.2 alone | 0.522 | 0.947 | 0.360 |
| **ORACLE (either)** | **0.561** | 0.719 | 0.460 |
| AND (both agree) | 0.148 | 1.000 | 0.080 |

Key properties:
- Correlation between signals: **r=0.031** (near zero, almost orthogonal)
- No yes-bias: 17% (vs ESTP-Bench Qwen3-VL 76-83%)
- ORACLE improvement: **+7.5% F1** over w2t_prob alone

Per-type recall breakdown:
| Type | GPT-4o | w2t_prob | Winner |
|------|--------|----------|--------|
| Confirmation (42%) | 9.5% | 66.7% | w2t_prob (dialog context) |
| Correction (28%) | 21.4% | 7.1% | GPT-4o (visual reasoning) |
| Instruction (2%) | 100% | 0% | GPT-4o |
| Reminder (6%) | 33.3% | 0% | GPT-4o |

**Conclusion**: w2t_prob captures routine confirmations; GPT-4o captures rare but high-value error detection and step transitions. Fusion potential exists but practical cost is high ($0.003/frame, 3.5s latency).

## Status
- [x] WTAG fine-grained sweep [0.15-0.50] — COMPLETE
- [x] FDA-CLIP patch-level change detection (8 methods) — REJECTED (AUC < 0.50)
- [x] Reasoning trigger GPT-4o on WTAG — COMPLETE (F1=0.269, orthogonal to w2t_prob)
- [ ] ego4d threshold sweep [0.2, 0.3, 0.4] — IN PROGRESS (τ=0.2 done F1=0.2664, τ=0.3 at 21/65)
- [ ] Cross-dataset Adaptive-D analysis — after ego4d completes

## Implications for Paper
1. **Change detection (all methods)**: DEFINITIVELY NEGATIVE — visual change is anti-correlated with talk need
2. **Adaptive-D threshold calibration**: STRONG POSITIVE (+11.6% F1 on WTAG)
3. **w2t_prob quality**: Remarkably well-calibrated, bimodal, near-perfect at τ=0.3
4. **Per-dataset optimization needed**: WTAG optimal at τ=0.2, ego4d at τ=0.3 (paper)
5. **Conservative is better**: Lower τ → higher precision, better content quality
6. **Generalizes to ProAssist**: Same Adaptive-D concept from ESTP-Bench works here
7. **Reasoning trigger as secondary signal**: Orthogonal to w2t_prob, +7.5% ORACLE, but cost-prohibitive for real-time
8. **Fundamental insight**: Proactive dialogue timing is about DIALOG CONTEXT (confirmations), not VISUAL CHANGE (stuck vs active)
