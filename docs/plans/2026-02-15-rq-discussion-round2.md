# RQ Discussion Round 2: 实验验证 + 方向确认

> **日期**: 2026-02-15
> **前置文档**: [Round 1](./2025-02-14-rq-discussion-round1.md)
> **本轮主题**: 基于 EGTEA 原型实验结果，确认 Benchmark 迁移方案 + 架构决策

---

## 一、原型实验总结 (EGTEA Gaze+ Prototype)

### 1.1 实验概况

在 Round 1 讨论后，基于 EGTEA Gaze+ 数据集搭建了完整的 Proactive + Memory 原型系统，完成了两轮迭代 (v1 → v2) 和 8 项优化。

**系统架构**:
```
Video Stream (2 FPS) → Trigger Detection → Memory Update → Content Generation → QA
```

**模型配置**: GPT-4o (Azure API) + Qwen3-VL-8B (本地 A100 80GB, bf16)

### 1.2 关键实验数据

#### RQ1: Proactive Intervention

| 指标 | Periodic (15s) | Oracle (GT) | VLM Delta (Qwen) |
|------|:-:|:-:|:-:|
| F1 | 0.403 | 0.662 | 0.198 |
| Precision | 0.652 | 0.898 | 0.617 |
| Recall | 0.292 | 0.524 | 0.118 |
| Timing MAE | 1.35s | 1.20s | **1.03s** |
| Content Sim | 0.626 | 0.636 | 0.574 |
| LLM Helpfulness | 0 | 0 | **3.30/5** |
| Harmful FP | 20 | 6.7 | **0** |
| Latency | 0ms | 0ms | 1194ms |

#### RQ2: Memory System

| 指标 | v1 (GPT-4o) | v2 (GPT-4o) | v2 (Qwen) |
|------|:-:|:-:|:-:|
| Entity F1 | 0.447 | 1.000 | 1.000 |
| Event R@1 | 0.221 | 0.503 | 0.562 |
| QA Accuracy | 0.061 | 0.624 | **0.812** |
| QA Score | 0.87/5 | 2.74/5 | **3.44/5** |
| Latency | 3177ms | 3071ms | **1872ms** |

### 1.3 原型实验的核心发现

1. **Ground Truth 不合理**: EGTEA action boundary ≠ 用户需要帮助的时刻。Oracle F1 天花板仅 0.662，不是系统差而是 GT 本身有 gap
2. **VLM Delta 是最安全的 trigger**: 零有害 FP，但召回率太低 (11.8%)。问题在于逐帧独立决策 + 保守阈值
3. **Qwen3-VL-8B > GPT-4o**: 本地模型在 QA 上全面超越 API (准确率 +30%, 延迟 -39%)
4. **Content quality vs Trigger quality 解耦**: 模板的 semantic similarity 高但 helpfulness 为零；VLM 生成内容 similarity 低但 helpfulness 3.30/5

### 1.4 原型实验的价值

虽然 GT 不理想，但积累了关键工程资产：
- 完整的 streaming video → trigger → memory → generation pipeline
- Qwen3-VL-8B 部署和推理验证 (17.5GB VRAM, ~5s/frame)
- 三层 Memory 系统的可行性验证
- 评估框架和指标体系的初步设计

**结论: 原型实验作为系统工程验证完成，不作为论文数据。正式实验迁移到 ESTP-Bench。**

---

## 二、Benchmark 迁移决策

### 2.1 为什么弃用 EGTEA 自定义 GT，转向 ESTP-Bench

| 维度 | EGTEA (当前) | ESTP-Bench |
|------|---|---|
| GT 来源 | 自己定义 (action boundary) | 人工标注 (2264 QA, 双标注验证) |
| Proactive 定义 | 无 (我们假设) | "relevant visual evidence emerges" |
| 评估指标 | 自定义 F1 + Content Sim | **ESTP-F1** (answer quality + timing + precision) |
| 任务覆盖 | 单一 (cooking 步骤) | 14 种任务类型 × 3 类 proactive |
| 学术认可 | 无 | NeurIPS 2025 |
| 视频数量 | 2-3 sessions | 164 videos (890 clips from Ego4D) |

### 2.2 Benchmark 角色分配

| Benchmark | 角色 | 理由 |
|-----------|------|------|
| **ESTP-Bench** | 主评估 (Proactive) | 专为 ego proactive streaming 设计, ESTP-F1 是最合理的指标 |
| **OmniMMI** | 泛化验证 | PA + PT 两个 proactive subtask 验证非 ego 场景 |
| **EGTEA Gaze+** | 保留做 case study | 深度分析 cooking 场景, ablation, memory 评估 |

### 2.3 ESTP-Bench 数据现状

**已下载至**: `/home/v-tangxin/GUI/data/ESTP-Bench/` (21GB)

**数据结构**:
```
ESTP-Bench/
├── estp_dataset/
│   ├── estp_bench_sq.json        # 主评估: single-QA (164 videos, 12 task types)
│   ├── estp_bench_cq_v3.json     # 上下文评估: contextual-QA (2 task types)
│   ├── benchmark/
│   │   ├── estp.py               # 评估框架 (ESTP_singleQ_benchmark, ESTP_contextualQ_benchmark)
│   │   └── eval.py               # ESTP-F1 计算 (validScoreF1)
│   ├── model/                    # 模型包装器接口
│   ├── estpSqa_baseline/         # baseline 预测结果
│   └── estpSqa_ours/             # EyeWO 预测结果
└── full_scale_2fps_max384/       # 预处理视频 (171 个 mp4, 2fps, max384p)
```

**数据格式** (单条 QA):
```json
{
  "clip_start_time": 1073.5,
  "clip_end_time": 1382.0,
  "Task Type": "Object Recognition",
  "question": "Can you remind me what the white plastic object being handled is?",
  "conversation": [
    {"role": "assistant", "content": "...", "start_time": 1073.5, "end_time": 1076.5},
    {"role": "assistant", "content": "...", "start_time": 1080.5, "end_time": 1082.5},
    ...  // 多个有效回答时间窗口
  ]
}
```

**关键理解**: 每个 question 有多个 `[start_time, end_time]` 有效回答窗口。模型需要在 streaming 中判断"现在是不是合适的时机回答这个问题"。

**评估模式**:
- `frame_by_frame`: 逐帧问模型 "Is it the right time to answer?"，模型回答 yes/no + 生成答案
- `passive_inference`: 离线看完视频后一次性回答，标注 `[frame_idx] answer`

---

## 三、关键架构决策

### 3.1 KV-Cache: 当前不需要，后续按需引入

**结论: 当前阶段不做 KV-Cache。**

| 层面 | Agent-based (当前) | 端到端 Streaming VLM |
|------|---|---|
| 调用方式 | 每次触发 → 独立 VLM call | 视频帧持续输入模型 |
| Context 增长 | 不增长 (固定长度输入) | 线性增长 |
| 是否需要 KV-Cache | **否** | 是 |
| 代表系统 | 我们的系统 | StreamAgent, ProAssist |

**KV-Cache 的引入时机**:
- 当切换到端到端 streaming VLM 时
- 当需要处理 30min+ 长视频时
- 当需要跨帧注意力 (temporal attention) 时

### 3.2 StreamAgent 的借鉴与区别

| 维度 | StreamAgent | 我们的系统 |
|------|---|---|
| 核心范式 | Query-conditioned anticipation (先有问题，预测何时能回答) | Query-free proactive (无问题，主动判断何时介入) |
| 触发决策 | 轻量 Anticipatory Agent | VLM Delta / Task-Graph Trigger |
| 内容生成 | Main Responder | ContentGenerator (VLM) |
| 记忆结构 | Streaming KV-Cache (GPU短期 + CPU长期) | 三层外置 Memory (Task + Event + Visual) |
| 训练需求 | 需训练 Anticipatory Agent | Training-free (prompt-based) → 后续可训练 |

**可借鉴的设计**:
1. **决策与生成分离**: 轻量模型做 trigger，重量模型做生成 — 我们已在这样做
2. **分层记忆**: GPU 近期 + CPU 远期 — 概念类似我们的 L3 Visual + L2 Event
3. **Anticipation 思想**: 不是预测 "下一帧有什么"，而是预测 "当前状态离需要介入还有多远" — 可融入 Task-Graph Guided

### 3.3 当前系统的定位

**我们不是在做 StreamAgent 的竞品，而是在做互补方向。**

StreamAgent = "给定 query，何时能找到证据"（被动）
我们 = "无 query 时，何时该主动说话"（主动）

这在 ESTP-Bench 上有清晰对应:
- StreamAgent 更擅长: `frame_by_frame` 模式（已知 question，判断时机）
- 我们更擅长（目标）: 真正的 proactive 场景（模型自主判断）

---

## 四、Baseline → Innovation 执行路线

### 4.1 流程确认

```
Phase 1: Baseline
  ├── 在 ESTP-Bench 上复现/跑现有 baseline
  │   ├── Polling Strategy (类似 Periodic)
  │   ├── Qwen3-VL frame-by-frame (逐帧判断)
  │   └── 收集已有 baseline 结果 (论文中的数据)
  ├── 记录各方法的 failure cases
  └── 产出: baseline 数字 + failure case 分析

Phase 2: Failure Analysis
  ├── 归类失败原因: 时机不对? 内容不对? 该说没说? 不该说却说了?
  ├── 分析 ESTP 14 种任务的难度分布
  └── 确定最有价值的创新切入点

Phase 3: Innovation Module
  ├── 基于 Phase 2 选择创新方向:
  │   ├── Idea 1: Context-Aware Trigger (Δ-based)
  │   ├── Idea 2: Hierarchical Proactive (urgency score)
  │   └── Idea 3: Task-Graph Guided ★
  ├── 在 baseline 代码上叠加创新模块
  └── 产出: baseline vs baseline+innovation 对比

Phase 4: Generalization + Paper
  ├── ESTP-Bench 主实验 (全量 + ablation)
  ├── OmniMMI PA/PT 泛化验证
  ├── EGTEA cooking 场景 case study
  └── 写论文
```

### 4.2 Phase 1 的具体任务

1. **理解 ESTP-Bench 评估协议**: 读懂 `estp.py` + `eval.py`，理解 ESTP-F1 的计算方式
2. **适配模型接口**: 实现 Qwen3-VL 的 `model.Run(video_path, question, start_time, end_time, query_time)` 接口
3. **跑 frame-by-frame baseline**: Qwen3-VL 逐帧判断 "Is it the right time?"
4. **收集已有数据**: ESTP-Bench 论文中报告的 baseline 数据 (MiniCPMV, Qwen2VL, etc.)
5. **分析 failure cases**: 哪些任务类型最难? 什么样的 proactive 场景当前方法失败?

---

## 五、待定决策更新

### 从 Round 1 继承

| 决策项 | Round 1 状态 | Round 2 状态 |
|--------|---|---|
| 核心创新选择 | 待定 (A/B/C) | **Phase 2 后决定**，倾向 A (Task-Graph) |
| 训练 vs Training-free | 建议训练 | **Phase 1 先 training-free，Phase 3 考虑训练** |
| 评估 benchmark | 待定 | **ESTP-Bench (主) + OmniMMI (泛化)** |
| Memory 存储后端 | 待定 | **Phase 1 用 Python dict，后续按需升级** |

### Round 2 新增决策

| 决策项 | 状态 | 备注 |
|--------|------|------|
| KV-Cache | **当前不做** | Agent-based 架构不需要，后续按需引入 |
| 主 baseline 模型 | **Qwen3-VL-8B** | 已验证可用，本地推理 |
| EGTEA 原型实验 | **不进入论文** | 作为工程验证和代码资产保留 |
| 与 StreamAgent 的关系 | **互补而非竞品** | 我们做 query-free proactive |

---

## 六、关键相关工作索引 (更新)

| 论文 | 发表 | 核心贡献 | 与我们的关系 | 优先级 |
|------|------|---------|-------------|:------:|
| **EyeWO** | NeurIPS 2025 | ESTP-Bench + ESTP-F1 + 3-stage training | **主评估 benchmark + 直接对比目标** | ★★★ |
| **StreamAgent** | arXiv 2025 | Anticipatory agent + streaming KV-cache | 架构借鉴，互补方向 | ★★☆ |
| **OmniMMI** | CVPR 2025 | 6-task streaming benchmark (PA + PT) | 泛化验证 | ★★☆ |
| ProAssist | EMNLP 2025 | W2T + streaming dialogue | Trigger 参考 | ★★☆ |
| StreamBridge | NeurIPS 2025 | 解耦 activation + compression | 解耦架构参考 | ★★☆ |
| MMDuet2 | EMNLP 2025 | RL 训练 respond/silence | RL trigger 参考 | ★☆☆ |
| Dispider | CVPR 2025 | 感知-决策-反应解耦 | 异步架构参考 | ★☆☆ |
