# Project Iteration Plan: ESTP-Bench Baseline + Innovation

> **日期**: 2026-02-15
> **项目**: Streaming Video Proactive Agent for Smart Glasses
> **前置**: [Round 1 讨论](./2025-02-14-rq-discussion-round1.md) | [Round 2 讨论](./2026-02-15-rq-discussion-round2.md) | [原型实验报告](../proactive-memory-eval-report-20260215.md)
> **硬件**: 1× A100 80GB (当前), 可扩展至 2-4×
> **决策**: 不使用 KV-Cache (当前阶段), Agent-based 架构

---

## Phase 1: ESTP-Bench Baseline (当前阶段)

### 目标
在 ESTP-Bench 上跑出可复现的 baseline 数字，理解评估协议，为后续创新提供对比基准。

### 1.1 理解评估协议

**任务**: 完整理解 ESTP-Bench 的评估流程

**ESTP-Bench 关键概念**:
```
输入:
  - 视频片段 [clip_start_time, clip_end_time]
  - 问题 question (在 query_time 提出)

GT:
  - 多个有效回答窗口 [{start_time, end_time, content}, ...]
  - 每个窗口代表"在这段时间内，视觉证据足以回答问题"

模型任务:
  - Streaming 处理视频帧
  - 在合适的时间点给出回答（时机 + 内容都要对）

评估:
  - ESTP-F1 = f(answer_quality, response_timing, temporal_precision)
  - answer_quality: LLM-as-Judge (text_score + response_score)
  - timing: 回答是否落在有效窗口内
  - precision: 无效回答 (FP) 的惩罚
```

**14 种任务类型 (3 类)**:
```
Explicit Proactive (8 types):
  视觉线索直接可见 → 较易
  Object Recognition, Attribute Perception, Text-Rich Understanding,
  Object Localization, Object State Change, Ego Object Localization,
  Ego Object State Change, Action Recognition

Implicit Proactive (4 types):
  需要推理 → 较难
  Object Function Reasoning, Information Function Reasoning,
  Next Action Reasoning, Task Understanding

Contextual Proactive (2 types):
  需要跨时间上下文 → 最难
  Object Relative Context, Task Relative Context
```

**交付物**: 评估流程文档 + 可运行的 eval 脚本

### 1.2 适配 Qwen3-VL 模型接口

**任务**: 实现符合 ESTP-Bench 评估框架的 Qwen3-VL 模型包装器

**需实现的接口** (参考 `estp_dataset/benchmark/estp.py`):
```python
class QwenModel:
    def name(self) -> str:
        return "Qwen3VL"

    def Run(self, video_path, question, start_time, end_time, query_time=0):
        """
        frame_by_frame 模式:
          逐帧处理 [start_time, end_time] 的视频
          每帧判断 "Is it the right time to answer?"
          如果 yes → 生成答案
          返回 dialog_history

        passive_inference 模式:
          看完所有帧后一次性回答
          返回 [frame_idx] answer 格式
        """
        ...
```

**关键参数**:
- `frame_fps`: 帧率 (ESTP 预处理为 2fps)
- `video_root`: `/home/v-tangxin/GUI/data/ESTP-Bench/full_scale_2fps_max384/`
- 视频格式: `{video_id}.mp4`

**交付物**: `model/qwen3vl.py` 包装器

### 1.3 运行 Baseline

**需要跑的配置**:

| # | 模型 | 评估模式 | 预计耗时 | 优先级 |
|---|------|---------|---------|:------:|
| 1 | Qwen3-VL-8B | frame_by_frame (singleQA) | ~4-8h (164 videos) | P0 |
| 2 | Qwen3-VL-8B | passive_inference (singleQA) | ~1-2h | P0 |
| 3 | Qwen3-VL-8B | frame_by_frame (contextualQA) | ~2-4h | P1 |
| 4 | GPT-4o | frame_by_frame (singleQA, 5 cases) | ~30min | P2 |

**收集已有 baseline 数据** (从论文/结果文件):
- EyeWO: 34.7% ESTP-F1 (3-stage trained, 论文最佳)
- MiniCPMV: 22.9% (Polling Strategy, 论文 best baseline)
- VideollmOnline: 存在预测结果文件 (`estp_bench_sq_VideollmOnline*.json`)
- MMDuet: 存在预测结果文件

**交付物**: Qwen3-VL baseline 数字 + 现有 baseline 汇总表

### 1.4 Failure Case 分析

**分析维度**:
1. **Per-task-type 性能分布**: 哪些任务类型最难?
2. **Timing 错误分析**: 模型是倾向于"说太早"还是"说太晚"?
3. **Content 错误分析**: 时机对了但内容错的 case
4. **Silent 错误分析**: 该说话但完全没说的 case
5. **与 EyeWO 的 gap 分析**: EyeWO 在哪些任务上显著优于我们?

**交付物**: failure case 分析报告

---

## Phase 2: Analysis & Innovation Selection

### 目标
基于 Phase 1 的 failure analysis，确定最有价值的创新方向。

### 2.1 创新候选

| Idea | 核心 | 解决什么问题 | 适用条件 |
|------|------|------------|---------|
| **A: Task-Graph Guided** | 任务图谱状态 → 触发决策 | 纯视觉判断遗漏的 implicit proactive | 如果 implicit 类型 failure 多 |
| **B: Context-Aware Δ** | Δ(context) → trigger | 场景变化时触发 | 如果 timing 错误是主要原因 |
| **C: Hierarchical Urgency** | urgency score (0-1) | 区分轻重缓急 | 如果 FP (说太多) 是主要问题 |

### 2.2 选择标准

1. **Impact**: 能在 ESTP-F1 上带来可测量的提升 (>3%)
2. **Novelty**: 在 streaming proactive 领域尚未被做过
3. **Feasibility**: 在 1× A100 上可实现 (training-free 或轻量训练)
4. **Story**: 能讲一个清晰的 "从 X 到 Y" 的创新故事

**交付物**: 创新方向决定 + 技术方案设计

---

## Phase 3: Innovation Implementation

### 目标
在 baseline 上叠加创新模块，验证效果。

### 3.1 实施框架
```
baseline_system/
├── model/qwen3vl.py          # Phase 1 产出
├── trigger/
│   ├── base.py               # trigger 接口
│   ├── periodic.py            # baseline: 固定间隔
│   └── innovation.py          # Phase 3: 创新 trigger
├── memory/                    # 如果需要
├── eval/
│   └── estp_evaluator.py     # ESTP-F1 计算
└── scripts/
    └── run_estp.py            # 一键评估脚本
```

### 3.2 对比实验设计

| 实验 | 配置 | 目的 |
|------|------|------|
| Exp1 | Qwen3-VL + frame_by_frame (baseline) | 基准 |
| Exp2 | Qwen3-VL + innovation trigger | 验证创新 |
| Exp3 | Exp2 + Memory system | 验证 memory 增益 |
| Exp4 | Ablation (各组件独立贡献) | 论文 table |

**交付物**: Exp1-4 结果 + 分析

---

## Phase 4: Generalization & Paper

### 目标
扩展评估、写论文。

### 4.1 评估矩阵

| Benchmark | 任务 | 交付物 |
|-----------|------|--------|
| ESTP-Bench (全量) | 14 task types × singleQA + contextualQA | 主表 |
| ESTP-Bench (ablation) | 各组件独立贡献 | ablation 表 |
| OmniMMI (PA + PT) | Proactive Alerting + Turn-taking | 泛化表 |
| EGTEA Gaze+ (cooking) | 定性分析 | case study figure |

### 4.2 论文结构草案

```
1. Introduction: Streaming Proactive Agent 的挑战
2. Related Work: Streaming VLM + Proactive Intervention + Memory
3. Method:
   3.1 Problem Formulation
   3.2 [创新模块名称]
   3.3 Three-Layer Memory (supporting)
   3.4 System Architecture
4. Experiments:
   4.1 Setup (ESTP-Bench, OmniMMI, metrics)
   4.2 Main Results (Table 1: ESTP-F1 comparison)
   4.3 Ablation Study
   4.4 Analysis (per-task, timing, qualitative)
5. Conclusion
```

---

## 时间线预估

| 阶段 | 内容 | 预计时间 |
|------|------|---------|
| Phase 1.1-1.2 | 理解协议 + 适配接口 | 2-3 天 |
| Phase 1.3 | 跑 baseline | 1-2 天 (含 GPU 等待) |
| Phase 1.4 | Failure analysis | 1-2 天 |
| Phase 2 | Analysis + innovation selection | 2-3 天 |
| Phase 3 | Innovation implementation + experiments | 1-2 周 |
| Phase 4 | Generalization + paper | 2-3 周 |

---

## 资源清单

### 数据
- [x] ESTP-Bench: `/home/v-tangxin/GUI/data/ESTP-Bench/` (21GB, 164 videos)
- [x] EGTEA Gaze+: `/home/v-tangxin/GUI/data/EGTEA_Gaze_Plus/`
- [ ] OmniMMI: 需下载 (已克隆代码至 `/temp/OmniMMI`)

### 模型
- [x] Qwen3-VL-8B-Instruct: `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`
- [x] GPT-4o: Azure OpenAI API
- [x] all-MiniLM-L6-v2: sentence-transformers

### 代码
- [x] 原型系统: `/home/v-tangxin/GUI/proactive-project/`
- [x] ESTP-Bench 评估代码: `/home/v-tangxin/GUI/data/ESTP-Bench/estp_dataset/benchmark/`
- [ ] Baseline 适配代码: 待开发 (Phase 1.2)

### 硬件
- [x] 1× A100 80GB
- [ ] 额外 GPU (如需训练)
