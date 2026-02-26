# RQ Discussion Round 1: Streaming Video Understanding

> **日期**: 2026-02-14
> **讨论主题**: Agentic Streaming Video Understanding 的 Proactive + Memory + Encoder + 执行方案
> **参与者**: 研究者 + AI Research Assistant

---

## 一、Proactive Intervention

### Q1: Focus cooking 还是按 benchmark 场景改系统？

**结论: Focus cooking 做系统，在通用 benchmark 上验证泛化能力。**

- Cooking 是"主实验场景" (Primary Setting) — 深挖系统设计、case study、ablation
- OmniMMI / ESTP-Bench / OVO-Bench 是"泛化验证" (Generalization) — 不需要 SOTA，只需 competitive
- 论文策略: "核心场景深挖 + 通用 benchmark 验证"（ProAssist 也是这样做的）

**可用的 benchmark 对比：**

| Benchmark | 评估什么 | 关键特点 | 发表 |
|-----------|---------|---------|------|
| **OVO-Bench** | Online video understanding | 3 场景: Backward/Real-time/Forward Active | CVPR 2025 |
| **OmniMMI** | Streaming interaction | 6 subtasks 含 PA + PT | CVPR 2025 |
| **ESTP-Bench** | Ego proactive streaming | ESTP-F1 整合 quality + timing + precision | NeurIPS 2025 |

**OmniMMI 代码分析关键发现**（已克隆至 `/temp/OmniMMI`）：
- Proactive Alerting (PA): answer 是时间区间 [start, end]，指标为 Accuracy + Precision + IoU
- Proactive Turn-taking (PT): Binary accuracy（模型是否正确判断"该不该说话"）
- 大部分模型在 proactive 任务上 < 50%，说明这是一个有价值的研究方向

### Q2: Proactive 的主要技术方案 + 创新 idea

**现有技术方案：**

| 方法 | 核心机制 | Trigger 方式 | 局限 |
|------|---------|------------|------|
| ProAssist (W2T) | 单模型, 显式二分类头 | `not_talk_prob > threshold` | 耦合 trigger+generation |
| StreamBridge | 解耦, 0.5B activation model | `score_head → prob ≥ threshold` | 逐帧独立决策 |
| VideoLLM-Online (LIVE) | 隐式 token 分布 | `P(frame_interval_token) < threshold` | 不可控 |
| Dispider | 三模块异步 | perception 模块触发 | 本质还是 threshold-based |
| MMDuet2 | 强化学习 | RL reward: 惩罚 FN+FP | 需要大量 RL 训练 |

**三个创新 idea：**

#### Idea 1: Context-Aware Trigger（基于情境变化的触发）
- 现有: `frame[t] → trigger?` (逐帧独立决策)
- 提出: `Δ(context[t-k:t]) → trigger?` (基于变化而非内容)
- 触发条件: 步骤转换 / 异常检测 / 空闲检测 / 注意力分散
- 技术: frame-level embedding diff + threshold 或 temporal change detection model

#### Idea 2: Hierarchical Proactive（分层干预强度）
- Level 0: 静默观察
- Level 1: 轻量提示（更新进度条）
- Level 2: 中等干预（弹出 Badge）
- Level 3: 强干预（Alert Card）
- 输出 urgency score (0-1) 而非 binary decision

#### Idea 3: Task-Graph Guided Proactive（任务图谱指导触发）★推荐
- 用 recipe/task graph 指导 trigger 时机
- 将 proactive 从纯感知问题变为 planning + perception 问题
- 创新点: 第一个将 structured task knowledge 融入 streaming trigger decision
- 技术: 任务图谱状态 + 帧特征 + 历史事件 → 多信号融合触发

**执行建议**: 先用简单方案跑 baseline → 收集 hard case → 分析失败案例 → 确定哪个 idea 最有价值

### 评估框架建议

**三层评估体系：**

```
Layer 1: Trigger Accuracy (时机)
  - Trigger Precision / Recall / F1
  - Soft matching: valid window [t-δ, t+δ]
  - Timing MAE (平均时间偏差)

Layer 2: Content Quality (内容, 仅对 matched triggers)
  - Semantic Score (sentence-transformer cosine sim)
  - LLM-as-Judge (relevance, helpfulness, correctness)

Layer 3: Redundant Analysis (多余 trigger)
  - 将 FP 分为 benign_FP (合理但 GT 没标) 和 harmful_FP (纯噪音)
  - LLM 判断: "Given context, is this intervention helpful?"
```

---

## 二、Memory

### Q1: Streaming vs Offline Memory 系统

**大部分 memory 工作是离线的，只有少数是真 streaming：**

| 系统 | Online/Offline | 能否用于 Streaming |
|------|---------------|-------------------|
| MA-LMM (CVPR'24) | **Online** — 逐帧 + memory bank | ✅ |
| Flash-VStream (ICCV'25) | **Online** — STAR memory | ✅ |
| Infinity-Video (ICML'25) | **Online** — sticky memories | ✅ |
| VideoAgent (ECCV'24) | **Offline** — 全视频建 memory | ❌ |
| GCAgent (CVPR'25) | **Offline** — 全视频 episodic | ❌ |
| DVD (MSRA'25) | **Offline** — 先分 chunk 建索引 | ❌ |

**Streaming vs Long Video 核心区别：**
- Streaming: 只知道 [f1...f_t]，不知道 f_{t+1}。必须基于已知信息实时决策。
- Long Video: 有完整视频 [f1...fN]，可以全局统计、前后对比。

**检索 vs 推理的问题：**
- 简单 embedding similarity 有语义鸿沟（"钥匙"vs"金属物体放在台面"）和多跳推理需求
- 正确做法: Retrieve + Reason two-stage pipeline（类似 DVD 的 agentic search）

### Q2: 三层 Memory 架构的创新性和实施方案

**创新定位：** 三层分层本身不是创新（认知科学基本概念），但"在 streaming video agent 中在线实现 + 各层独立评估协议"是创新。

**现有工作的不足：**
- MA-LMM: 单层 memory bank，不区分类型
- VideoAgent: 双 memory 但离线构建
- GCAgent: 有 schematic + narrative，但也离线
- **没有人在 streaming 场景中实现过分层 memory 并给出层级评估**

**三层 Memory 实施方案：**

```
Layer 1: Task Memory (文本, 始终在 context 中)
  - 内容: 任务目标 + 步骤列表 + 当前进度 + 已操作 entities
  - 更新: VLM 检测到步骤完成时触发
  - 上下文占用: ~200-500 tokens (固定)
  - 评估: Step Detection Accuracy, Entity Tracking F1

Layer 2: Semantic Event Memory (结构化存储, 按需检索)
  - 内容: 关键事件的时间/地点/内容的结构化记录
  - 更新: 检测到"有意义的事件"时追加
  - 上下文占用: 0 (不在 context 中, 检索时注入)
  - 检索: CLIP embedding similarity → LLM reranking
  - 评估: Event Recall@K, Temporal Localization IoU

Layer 3: Visual Working Memory (帧特征, 滑动窗口 + 压缩)
  - 内容: 最近 N 帧完整特征 + 历史帧压缩特征
  - 更新: 每帧滑动窗口
  - 压缩: StreamBridge-style mean-pooling
  - 评估: Retrieval Precision@K, Frame Localization Accuracy

三层交互示例 (找钥匙):
  1. Layer 1: 提供任务上下文 ("调味步骤会用到盐")
  2. Layer 2: retrieve("盐") → 事件记录列表
  3. Layer 3: 检索对应时间的帧特征做细粒度分析
```

### Q3: KV Cache vs Context 的区别

| | Context (Embedding 层面) | KV Cache (Attention 层面) |
|---|---|---|
| 是什么 | 模型的输入 — token embeddings | 模型的中间计算结果 — 每层 K,V |
| 在哪里 | 模型外面（输入侧） | 模型里面（每层） |
| 大小 | [seq_len, hidden_dim] | [n_layers × 2 × seq_len × hidden_dim] |
| 可操作性 | 可自由拼接、压缩、替换 | 修改后必须重新计算 |
| ProAssist | ❌ | ✅ DROP_MIDDLE 删 KV Cache |
| StreamBridge | ✅ mean-pool embeddings | ❌ |

**Agent 系统建议：**
- Task Memory (Layer 1): 在 Context 中 → 每次作为 system prompt
- Event Memory (Layer 2): 按需检索后注入 Context
- Visual Working Memory (Layer 3): 近期帧用 KV Cache (增量), 远期帧用 embedding (按需)

**简记**: Context = 给模型的"输入材料", KV Cache = 模型"记住的思考过程"。Agent 切换材料 → Context 更适合。Streaming 持续高效 → KV Cache 更适合。

---

## 三、Online Visual Encoder

| 范式 | 做法 | 优势 | 劣势 |
|------|------|------|------|
| 逐帧编码 + 缓存 | 每帧独立过 ViT → cache | 简单、延迟可预测 | 帧间无交互 |
| Disentangled Perception (Dispider) | 感知/决策/反应三模块异步 | 生成时不会错过新帧 | 工程复杂度高 |
| Anticipatory Encoding (StreamAgent) | 编码时预测未来时空位置 | 主动感知、减少无关帧开销 | 需额外训练、预测可能出错 |

**建议路径：**
- Phase 1: 逐帧编码 + 缓存 (frozen SigLIP/CLIP)
- Phase 2: 加入 Dispider 异步感知
- Phase 3: 考虑 temporal aggregation (Flash-VStream STAR)

---

## 四、执行方案与创新性

**避免"拼凑"的关键：围绕一个核心创新展开**

**三个可选方向：**

| Option | 核心 idea | 创新点 | 故事 |
|--------|----------|--------|------|
| **A: Task-Graph Guided Proactive** ★ | 用任务图谱指导 trigger | 第一个将 structured task knowledge 融入 streaming trigger | "从 perception-only 到 planning-guided" |
| B: Hierarchical Memory for Streaming | 三层 memory 的 online 构建 + 层级评估 | 第一个在 streaming agent 中分层 memory + 评估协议 | "现有 memory 要么离线要么单层" |
| C: System Paper | 完整 streaming agent + GUI downstream | 第一个连接 streaming video agent 到 GUI | "端到端从视频到 UI" |

**GPU 资源分配 (2-4 张 A100)：**
- 1 张: VLM / API 推理
- 1 张: Trigger model (0.5B-3B) 训练/推理
- 2 张 (可选): 大模型微调

---

## 关键相关工作索引

| 论文 | 发表 | 核心贡献 | 与我们的关系 |
|------|------|---------|-------------|
| StreamAgent | ICLR 2026 | Anticipatory agents + streaming KV-cache | 最直接竞品 |
| GCAgent | CVPR 2025 | Schematic + Narrative Episodic Memory | Memory 设计参考 |
| MMDuet2 | EMNLP 2025 | RL 训练 respond/silence | Trigger 训练参考 |
| EyeWO | NeurIPS 2025 | ESTP-F1 评估指标 | 评估参考 |
| Infinity-Video | ICML 2025 | Sticky memories | Memory 概念参考 |
| DVD | MSRA 2025 | Multi-granularity agentic search | 检索框架参考 |
| OmniMMI | CVPR 2025 | 6-task streaming benchmark | 泛化验证 |
| OVO-Bench | CVPR 2025 | Online video 3-scenario benchmark | 泛化验证 |
| Dispider | CVPR 2025 | 感知-决策-反应解耦 | 架构参考 |
| ProAssist | EMNLP 2025 | W2T + streaming dialogue | 核心参考 |
| StreamBridge | NeurIPS 2025 | 解耦 activation + round-decayed compression | 核心参考 |

---

## 待定决策

1. **核心创新选择**: Option A (Task-Graph Guided) vs B (Hierarchical Memory) vs C (System)
2. **训练 vs Training-free**: GPU 充足 (2-4×A100)，建议训练 trigger model
3. **评估 benchmark 优先级**: ESTP-Bench (更偏 ego/proactive) vs OVO-Bench (更通用)
4. **Memory 存储后端**: Python dict (简单) vs FAISS/ChromaDB (长视频扩展性)
