# Change-Driven Segment Memory for Streaming Video Agents

> 日期: 2026-02-26
> 状态: 设计阶段（brainstorming 完成，待实验验证）
> 来源: Claude Code Memory System 跨领域迁移 + ProAssist 实验 negative result 转化
> 关联: `docs/survey/2026-02-25-change-detection-as-memory-mechanism.md`, ProAssist Phase I/II 实验

---

## 1. 核心定位

**Agent 架构中 Memory 模块的创新**，不是完整系统论文。

核心思想：从 Prompt Engineering 到 Context Engineering 的迁移 — 不优化"如何问 VLM 问题"，而是优化"VLM 在推理时看到什么信息"。

灵感来源：
- Anthropic Context Engineering 实验：Context Editing 性能+29% token-84%，加 Memory Tool 后+39%
- Claude Code Memory System：分层作用域、200行索引约束、按需加载、Auto Memory 被动积累
- Serena 代码智能：符号级检索（find_symbol）、渐进式信息获取、不读整个文件只读需要的符号

### 与现有工作的差异化

| 维度 | StreamAgent (ICLR'26) | StreamBridge (NeurIPS'25) | Ours |
|---|---|---|---|
| 核心理念 | Anticipatory Planning | 嵌入空间压缩 | Context Engineering |
| 记忆形成 | 被动 KV-cache 积累 | 轮回衰减 mean-pooling | **变化驱动的主动切片** |
| 记忆粒度 | token-level KV-cache | embedding-level | **段落级语义索引 + 关键帧** |
| 检索方式 | 层自适应 KV 召回 | 无检索 | **summary 索引 → 按需加载关键帧** |
| 工具集 | Zoom（单一） | 无 | 多工具（replay, compare, zoom 等） |
| 自适应性 | 固定策略 | 固定压缩率 | 事件驱动的动态参数调整 |

---

## 2. 架构设计

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Task Context (Pre-Video)                          │
│  Agent 理解任务 + 用户状态 → 全局长期记忆                       │
│  (类比 CLAUDE.md: 项目级指令，启动时全量加载)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  Layer 1: Change-Driven Segmentation (Streaming, 自动运行)   │
│                                                             │
│  帧流 → CLIP Δ → ┬─ Δ < τ_low  → 跳过/稀疏采样 (~90%帧)   │
│                   ├─ τ_low<Δ<τ_high → 局部patch更新          │
│                   └─ Δ ≥ τ_high → 切片触发！                 │
│                         │                                   │
│                         ▼                                   │
│               Segment Memory Store                          │
│               {seg_id, t_start, t_end, summary, keyframes}  │
│               不断追加 → 语义记忆库                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  Layer 2: Tool-Augmented Reasoning (On-Demand)              │
│                                                             │
│  VLM 处理近几帧关键帧 + 按需调用工具执行任务：                   │
│  · replay(query)    — 按 summary 检索历史段落的关键帧          │
│  · compare(a, b)    — 跨段场景对比                            │
│  · zoom(region)     — 空间细节放大                            │
│  · summarize(range) — 压缩多段为概括                          │
│  · timeline()       — 获取全部段落的 summary 概览              │
│                                                             │
│  工具是为了执行任务，记忆检索是工具箱中的一项能力                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Layer 间交互：事件驱动混合模式

```
默认: Layer 1 以固定 τ 自动运行（低开销）
事件触发 VLM 介入调整:
  · 用户提问          → 可临时降低 τ（更细粒度监控）
  · 任务阶段切换       → 重置段落上下文
  · 长时间无切片       → 强制生成一次 summary（防遗忘）
  · 异常检测          → 提高关键帧保留密度
```

选择事件驱动而非全程 Agent-in-the-Loop 的理由：
- A（纯 Pipeline）太被动 — Phase I 实验已证明固定阈值的局限性
- B（全程 Agent）太昂贵 — 每帧让 VLM 参与决策不现实
- C（事件驱动）平衡效率与灵活性，与上下文工程的"按需调整"理念一致

### 2.3 VLM 推理时的上下文组成

```
┌──────────────────────────────────────┐
│ Task Context (全局，固定)              │
│ "用户正在做意大利面，当前步骤：煮面"     │
├──────────────────────────────────────┤
│ Recent Keyframes (近几帧，滑动窗口)     │
│ [frame_t-2, frame_t-1, frame_t]      │
├──────────────────────────────────────┤
│ Retrieved Segments (按需，来自记忆库)   │
│ seg_3: "切洋葱" [keyframe]            │
│ seg_7: "准备酱料" [keyframe]           │
└──────────────────────────────────────┘
```

核心类比：
| Claude Code | 视频 Memory 模块 | 功能 |
|---|---|---|
| CLAUDE.md (项目指令) | Task Context + User State | 全局不变上下文，启动时注入 |
| Auto Memory (MEMORY.md) | Segment Memory Store | 渐进积累的语义索引 |
| 子目录按需加载 | 按 summary 检索历史段落 | 不全量加载，需要时才拉取 |
| Serena find_symbol | replay(query) | 按语义定位具体内容 |
| 200行索引约束 | Context window 预算 | 强制精简，防止膨胀 |

---

## 3. 实验证据基础

来自 ProAssist 验证实验（2026-02-24/25）的已有数据：

| 统计量 | 值 | Memory 含义 |
|--------|-----|------------|
| w2t_prob > 0.99 占比 | 86% | 86% 的帧是稳态，可安全跳过 |
| w2t_prob < 0.01 占比 | 3.2% | 仅 3.2% 帧需要完整处理 |
| 高变化帧(top 5%) talk rate | 1.08× | 变化帧和说话帧几乎不重叠 → 可独立用于 memory |

关键发现：变化检测和对话触发是**两个独立维度**。
- 变化检测不预测"什么时候该说话"（AUC < 0.50，8种方法全部失败）
- 但它精准预测"什么时候画面有新信息值得存储"

这是一个优雅的 negative-to-positive 转化：同一个信号，换一个用途，从失败变成了核心机制。

---

## 4. 待验证的实验变量

| 变量 | 选项 | 意义 |
|---|---|---|
| 检索粒度 | A: 两级（先扫 summary 索引 → 再加载关键帧）vs B: 单级（summary 全量注入，Agent 自选） | 索引开销 vs 上下文效率 |
| τ 选择策略 | 固定 vs 自适应 vs per-task | Phase II 自适应τ经验可复用 |
| Summary 生成方式 | VLM 生成 vs 轻量模型（SigLIP + 文本解码器） | 计算预算分配 |
| 关键帧数量 | 1 vs 3 per segment | 记忆精度 vs 存储开销 |
| 切片机制 | 变化驱动 vs 固定间隔（baseline） | 核心消融：内容驱动 vs 时间驱动 |

---

## 5. 理论支撑

### 信息论视角

连续帧之间互信息 I(f_t; f_{t-1}) 极高。真正的新信息是条件熵：
```
H(f_t | f_{t-1}) = H(f_t) - I(f_t; f_{t-1})
```
CLIP change score 是 H(f_t | f_{t-1}) 的近似。只有条件熵大的帧才值得编码存储。

### 人类注意力类比

1. **漫游模式** (idle scanning): 画面稳定 → 不形成新记忆 → 跳过
2. **聚焦模式** (focused attention): 局部变化 → 注意力集中到变化区域 → 局部更新
3. **摘要模式** (gist extraction): 场景切换 → 对前段形成概括性记忆 → 切片 + summary

### 上下文工程视角

| Context Engineering (代码) | Context Engineering (视频) |
|---|---|
| 只保留相关代码片段 | 只保留关键帧 + 变化区域 |
| 项目结构/调试经验存入记忆 | 段落摘要/场景转换点存入记忆 |
| 文件路径 + 符号相关性排序 | 时间邻近性 + 语义相关性排序 |

---

## 6. 评估与训练策略

### 6.1 Benchmark（评估跑分）与 Dataset（训练微调）是两件事

```
Benchmark: 跑分验证系统是否有效，不训练
Dataset:   微调模型学会工具调用，需要构造训练数据
```

### 6.2 Benchmark 选择

**按评估维度映射：**

| 评估维度 | Benchmark | 匹配理由 |
|---------|-----------|---------|
| 检索有效性 | **Ego4D NLQ** (27K queries, 227h) | "给定查询→定位时间窗口" 直接对应 replay(query) |
| 跨段证据融合 | **HERBench** (26K QA) | 每题需≥3个不重叠段落证据，SOTA仅31-42%，改进空间巨大 |
| 长期记忆保持 | **TeleEgo** (3,291 QA, >14h/人) | 独有 MPT(Memory Persistence Time) 指标 |
| Streaming 协议 | **OVO-Bench** (CVPR'25) / **StreamingBench** (ICLR'25) | 与 StreamAgent 用同一 benchmark 直接可比 |

**Prototype 验证**：EGTEA Gaze+（本地已有，21GB，有动作段标注做 GT 对比）

### 6.3 递进式训练路线

```
Phase 0: Training-Free 验证（先做这个）
  │  · 用现有 VLM (Qwen3-VL / GPT-4o) + Prompt Engineering
  │  · 验证 memory 架构本身的价值
  │  · 不需要训练数据
  │
  │  核心问题：架构有没有用？
  │
  ├─ 有效 ──→ Phase 1: 微调工具调用决策
  │            · Memory 模块（CLIP 切片 + summary）保持不训练
  │            · 只微调 VLM 的工具调用部分
  │            · 构造 tool-calling SFT 数据（几千~几万条）
  │            · 数据来源：从 Ego4D NLQ 等标注反推工具调用序列
  │
  └─ 非常有效 ──→ Phase 2: 端到端联合优化
                   · Memory 模块 + VLM 联合训练
                   · 需要 100K+ 规模训练数据
                   · 可参考 StreamBridge 的 Stream-IT 120K 构造方式
```

### 6.4 本地已有数据资源

| 数据集 | 大小 | 路径 | 可复用于 |
|--------|------|------|---------|
| EGTEA Gaze+ | 21GB | `/data/EGTEA_Gaze_Plus` | Phase 0 prototype，动作段 GT |
| ESTP-Bench | 31GB | `/data/ESTP-Bench` | 已有完整实验框架，2fps/384p 预处理帧 |
| ProAssist (含 ego4d/epic-kitchens/wtag) | 240GB | `/data/ProAssist` | Arrow 格式帧，Phase 1 训练数据源 |
| ego-dataset | 336GB | `/data/ego-dataset` | 第一人称标注数据 |

---

## 7. 开放问题

1. **视频 workspace 与 coding workspace 的根本差异**：代码有天然符号结构（文件、类、函数），视频没有。段落切片 + summary 是否足以充当"视频的符号"？还是需要更丰富的结构（事件图、因果链）？

2. **Segment summary 的质量瓶颈**：整个检索链依赖 summary 的准确性。如果 summary 遗漏了关键信息，后续检索就会失效。如何保证 summary 质量？需要什么级别的模型？

3. **实时性约束**：切片检测（CLIP Δ）需要多快？Summary 生成的延迟预算是多少？这决定了用 VLM 还是轻量模型生成 summary。

4. **Training-Free 的天花板在哪**：Prompt engineering 能让 VLM 学会多复杂的工具调用？如果 Phase 0 效果不佳，是架构问题还是工具调用能力问题？

---

## 8. 下一步行动（Phase 0 优先）

- [ ] 在 EGTEA 上实现 CLIP 变化驱动切片 prototype
- [ ] 对比实验：变化驱动切片 vs 固定间隔切片的段落边界质量
- [ ] 用 Qwen3-VL + prompt 实现 tool-calling（replay, compare）
- [ ] 检索粒度消融：两级检索 vs 单级检索
- [ ] 在 Ego4D NLQ 或 OVO-Bench 上跑首轮 benchmark
- [ ] 根据 Phase 0 结果决定是否进入 Phase 1 微调
