# Proactive Streaming Video Understanding

Streaming video understanding system for cooking scenarios with proactive intervention and three-layer memory.

## Architecture

```
EGTEA Gaze+ Video Clips
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stream Simulator (2 FPS)                               │
│  Reconstructs session timeline from action clips        │
└──────────┬──────────────────────────────────────────────┘
           │ StreamFrame (timestamp, frame, action, ...)
           ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  RQ1: Proactive      │    │  RQ2: Memory Manager     │
│  ┌─────────────────┐ │    │  ┌──────────────────────┐│
│  │ Trigger Detector │ │    │  │ L1: Task Memory      ││
│  │ (periodic /      │ │    │  │ (recipe + progress)  ││
│  │  oracle /        │ │    │  ├──────────────────────┤│
│  │  vlm_delta)      │ │    │  │ L2: Event Memory     ││
│  └────────┬────────┘ │    │  │ (embedding retrieval) ││
│           │trigger    │    │  ├──────────────────────┤│
│  ┌────────▼────────┐ │    │  │ L3: Visual Memory     ││
│  │ Content         │ │    │  │ (sliding window +     ││
│  │ Generator       │ │    │  │  compression)         ││
│  │ (VLM / template)│ │    │  └──────────────────────┘│
│  └─────────────────┘ │    └──────────────────────────┘
└──────────────────────┘
           │                            │
           ▼                            ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Proactive Metrics   │    │  Memory Metrics          │
│  - Trigger P/R/F1    │    │  - Step Detection Acc    │
│  - Timing MAE        │    │  - Event Recall@K, MRR   │
│  - Content Similarity│    │  - Frame Retrieval P@K   │
│  - LLM-as-Judge      │    │  - End-to-End QA Acc     │
│  - FP Classification │    │  - LLM-as-Judge Score    │
└──────────────────────┘    └──────────────────────────┘
```

## Quick Start

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate
cd proactive-project

# 1. Verify dataset
python3 -m data.egtea_loader

# 2. Generate & inspect ground truth
python3 -m scripts.generate_gt --sessions 3

# 3. Run proactive eval (baseline, no VLM)
python3 -m scripts.run_proactive --trigger periodic --sessions 5

# 4. Run proactive eval (oracle upper bound)
python3 -m scripts.run_proactive --trigger action_boundary --sessions 5

# 5. Run proactive eval (VLM-based, requires model)
python3 -m scripts.run_proactive --trigger vlm_delta --model gpt4o --sessions 3

# 6. Run memory eval (Layer 1-3 only)
python3 -m scripts.run_memory --sessions 5

# 7. Run memory eval (with VLM for QA)
python3 -m scripts.run_memory --model gpt4o --sessions 3

# 8. Download Qwen3-VL for local inference
bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct
```

## Evaluation Metrics

### RQ1: Proactive Intervention

| Layer | Metric | Description |
|-------|--------|-------------|
| Trigger Timing | Precision / Recall / F1 | Soft-window matching (±3s) |
| Trigger Timing | Timing MAE | Mean absolute error to GT trigger |
| Content Quality | Semantic Similarity | sentence-transformer cosine sim |
| Content Quality | LLM-as-Judge | Relevance + Helpfulness (0-5) |
| System Quality | FP Classification | benign FP vs harmful FP |
| System Quality | Triggers/min | Intervention density |

### RQ2: Memory System

| Layer | Metric | Description |
|-------|--------|-------------|
| L1: Task Memory | Step Detection Acc | Correctly identified completed steps |
| L1: Task Memory | Entity Tracking F1 | Precision/Recall on tracked objects |
| L1: Task Memory | Progress MAE | Error in progress % estimation |
| L2: Event Memory | Recall@K (1,3,5) | Relevant event in top-K results |
| L2: Event Memory | MRR | Mean Reciprocal Rank |
| L2: Event Memory | Temporal IoU | Timestamp localization accuracy |
| L3: Visual Memory | Frame P@K | Correct frame retrieval |
| Cross-Layer | QA Accuracy | End-to-end question answering |
| Cross-Layer | QA Score | LLM-as-Judge quality (0-5) |

## Project Structure

```
proactive-project/
├── config/default.yaml          # All configuration
├── data/egtea_loader.py         # EGTEA Gaze+ dataset loader
├── src/
│   ├── streaming/
│   │   ├── frame_processor.py   # Frame extraction & encoding
│   │   └── stream_simulator.py  # Session-level streaming
│   ├── proactive/               # RQ1
│   │   ├── gt_generator.py      # Ground truth from annotations
│   │   ├── trigger.py           # Trigger detection strategies
│   │   └── generator.py         # Content generation
│   ├── memory/                  # RQ2
│   │   ├── task_memory.py       # Layer 1: task progress
│   │   ├── event_memory.py      # Layer 2: semantic events
│   │   ├── visual_memory.py     # Layer 3: frame buffer
│   │   └── manager.py           # Orchestration
│   ├── models/
│   │   ├── base.py              # Abstract VLM interface
│   │   ├── qwen_vl.py           # Qwen3-VL local model
│   │   └── gpt4o.py             # GPT-4o via Azure API
│   └── eval/
│       ├── proactive_metrics.py # RQ1 evaluation
│       ├── memory_metrics.py    # RQ2 evaluation
│       └── benchmark.py         # End-to-end runner
├── scripts/
│   ├── run_proactive.py         # RQ1 experiment entry point
│   ├── run_memory.py            # RQ2 experiment entry point
│   ├── generate_gt.py           # GT inspection
│   └── download_model.sh        # Model download
├── experiments/                 # Results output
└── requirements.txt
```

## Models

| Model | VRAM | Use Case |
|-------|------|----------|
| GPT-4o (Azure API) | 0 (API) | LLM-as-Judge, content generation |
| Qwen3-VL-8B-Instruct | ~17 GB | Local trigger detection + generation |
| Qwen3-VL-32B-Instruct | ~64 GB | Higher quality (fits A100 80GB) |
| all-MiniLM-L6-v2 | ~100 MB | Event memory embeddings |
