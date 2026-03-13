# Streaming Video Memory Agent - Round 1

> 日期: 2026-03-03
> 会话时长: 约 20+ 轮深度对话
> 主要方向: 从 Claude Code Memory 系统迁移到 Streaming Video Memory Agent 的全景调研与架构设计

## 会话目标

探索如何将 Code Agent（Claude Code）的 Memory Management 原则系统性迁移到 Streaming Video 领域，建立研究方向、选择框架、确定 benchmark，并设计核心架构。

## 行动路线

### 1. 全景调研（4 个并行 Agent）

**Prompt**: "思考一下一个顶尖的科研工作者会如何解决跨领域迁移的问题...你也可以用一个agent teams调研目前该领域的主流进展"

**并行调研**:
- Agent 1: 探索 proactive-project 架构 → 三层内存（Task/Event/Visual Memory）
- Agent 2: 深入分析 StreamBridge 代码 → 双模型架构（0.5B activation + 7B streaming），圆形衰减压缩
- Agent 3: 调研 StreamAgent (ICLR'26) 及领域进展 → training-free 流式 SOTA
- Agent 4: 获取 Claude Code memory 文档 → 六层分层记忆、Auto-compact、200行索引约束

**结果**: 建立了完整的领域全景图，识别出三个派系（KV-Cache工程派、帧级压缩派、Agent规划派）

### 2. 跨领域迁移映射

**Prompt**: 用户选择 D（探索阶段），认为 A（分层持久化）+ B（Tool-augmented）+ C（Workspace 结构化）都有迁移价值

**关键映射**:
| Claude Code 概念 | 视频记忆对应 |
|-----------------|------------|
| CLAUDE.md (项目指令) | Task Context (任务目标) |
| Auto Memory (MEMORY.md) | Segment Memory Store (段落摘要索引) |
| 子目录按需加载 | 按 summary 检索历史段落关键帧 |
| Serena find_symbol | replay(query) 工具 |
| 200行索引约束 | Context window 预算 |

### 3. 视频 Workspace 概念调研

**Prompt**: "如何在video领域定义这个workspace呢？是否已有工作定义了"

**调研结果（Opus agent）**:
- **没有人完整定义过 "Video Workspace"**
- 最接近的工作: VideoTree（树结构）、AVA（事件知识图谱）、AVI（实体图+工具集）
- **关键 Gap**: 没有任何工作同时具备——结构化 + 工具可访问 + 流式增量更新 + Proactive 触发
- 代码 vs 视频的根本差异: 代码结构是人工创建的（文件名、类名），视频结构必须自动发现

### 4. Benchmark 与训练需求调研

**Prompt**: "进行一下目前benchmark和dataset的调研，以及我们的方法是否需要进行训练"

**Benchmark 推荐（3 个梯队）**:
- 第一梯队: StreamBench（StreamChat, 显式测试记忆检索）、OVO-Bench（Backward Tracing）、ESTP-Bench（已有）
- 第二梯队: SVBench（多轮对话链, 50K QA）、StreamingBench、Ego4D-NLQ
- 第三梯队: Video-MME / MLVU（离线长视频标准对比）

**训练需求分析**:
- 7 个系统中 5 个是 Training-Free（StreamAgent, VideoAgent, AVA, AVI, VideoTree）
- Memory 模块不需要训练
- 如需训练, 优先级: Activation Model > Tool-calling SFT > Summary 生成

**本地数据映射**:
| 数据集 | 大小 | 用途 |
|--------|------|------|
| EGTEA Gaze+ | 21GB | Phase 0 prototype 开发 |
| ESTP-Bench | 31GB | 主评估（已有框架） |
| ProAssist | 240GB | Phase 1 训练数据源 |
| ego-dataset | 336GB | 大规模验证 |

### 5. Idea 定位与差异化

**Prompt**: "我的idea只是增强了memory的retrieve么？CLIP的设计只是为了流式视频减少需要处理的帧数么？"

**定位结论**:
- **不只是 retrieve**。三层贡献: Better Retrieval(表面) → Structured Workspace(中间) → Context Engineering Paradigm(最深)
- **CLIP Δ 三重角色**: 帧过滤器（效率）+ 事件边界检测器（结构）+ 信息密度量化器（自适应处理）
- **一句话定位**: "Context Engineering for Streaming Video Agents"
- **独特位置**: 唯一在 streaming 场景下提供显式结构化记忆 + 主动处理的方案

**关键竞争者发现: EventMemAgent (arXiv:2602.15329, 2026-02)**:
- 设计高度相似: 在线事件分段 + 事件级结构化记忆 + 多粒度工具集
- 差异化点: (1) 你的 CLIP Δ 是语义级 vs 像素级; (2) 你有 Proactive 能力; (3) 你有 Context Engineering 理论框架

### 6. 框架选型

**Prompt**: "我认为如果这是一个Agent框架，我们应该先搭一个Agent框架起来"

**调研了 7 个框架**:
| 框架 | 开源 | Agent Loop | 推荐度 |
|------|:----:|-----------|:------:|
| DVD (Microsoft) | 完整 | ReAct + Function Calling | 首选(training-free) |
| VITAL (ByteDance) | 完整 | XML tag + SFT+RL | **首选(可训练)** |
| VideoAgent | 完整 | LangChain ReAct | 参考 Memory |
| StreamBridge | 完整 | 无(不是Agent) | 参考流式前端 |
| AVI/StreamAgent/VideoARM | 未开源 | - | 不可用 |

**用户倾向 VITAL**: GPU 充足可训练。VITAL 优势: 训练内化工具调用(≤2轮精准出手), DGRPO 难度自适应, Qwen2.5-VL-7B 基座

### 7. Streaming 工具设计与快慢推理系统

**Prompt**: "streaming场景下的工具定义...streaming场景下的快慢推理系统问题"

**最小有效工具集（5个）**:
1. `recall(query)` — 记忆检索 (~50ms, embedding 检索)
2. `observe(target)` — 精细感知 (~300-500ms, VLM 推理)
3. `diff(seg_a, seg_b)` — 状态对比 (~500ms, **streaming 最独特的工具**)
4. `timeline(filter?)` — 全局概览 (~10ms, 读索引)
5. `track(entity)` — 实体追踪 (~10ms, 查字典)

如果只保留 3 个: **recall + observe + diff**

**三级快慢推理系统**:
```
Tier 0 (5ms/帧): CLIP Δ → skip 86% / store 11% / trigger 3%
Tier 1 (200ms/段): 轻量 VLM 生成 summary, 判断是否触发 Tier 2
Tier 2 (1-2s/按需): 完整 Agent + ≤2轮工具调用
```

核心: Tier 0 永不停止, Tier 1/2 异步后台运行。推理不阻塞帧流。

### 8. Streaming vs Long Video 的六维差距

**Gap 分析 Agent 结论（Opus model, 78K tokens）**:

| 维度 | 长视频 | 流式视频 | 设计影响 |
|------|--------|---------|---------|
| 时间访问 | 随机跳转 | 因果顺序 | 工具只能回溯 |
| 信息完整性 | 全部可用 | 持续增长 | 需要信念修正机制 |
| 处理预算 | 可重试 | 实时约束 | 快/慢路径分离 |
| 任务性质 | 回顾QA | Reactive + Proactive | 双模式决策框架 |
| 上下文管理 | 全局最优采样 | 在线增量 | 变化驱动 > 时间驱动 |
| 评估 | 准确率 | 准确率+时效+主动性 | 多 benchmark 多维度评估 |

**重要发现**: 需要增加 `revise(segment_id, new_info)` 工具支持信念修正

## 关键决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 研究定位 | Context Engineering for Streaming Video Agents | 不只是 retrieve 增强，而是范式迁移 |
| Memory 模块训练 | Training-Free | 5/7 系统证明可行，Memory 不需训练 |
| Agent 框架 | VITAL (ThinkingWithVideos) | 用户 GPU 充足，训练内化工具调用更可靠 |
| 工具核心 | recall + observe + diff | diff 是 streaming 独有核心工具 |
| 推理架构 | 三级快慢系统 (Tier 0/1/2) | 解耦感知与推理，86% 帧不触发推理 |
| Proactive 触发 | 放在 Tier 1（段落切换时评估） | 平衡速度与准确性 |

## 关键数据/指标

- ProAssist 验证: 86% 帧 w2t_prob > 0.99（可跳过），3.2% 帧需完整处理
- CLIP 变化检测: 与对话触发正交 (r=0.031)，适合做 memory 管理而非 trigger
- EventMemAgent (竞争者): OVO-Bench 60.75% (32帧输入超越 GPT-4o 59.54%)
- VITAL 性能: Video-MME 63.1%, MLVU 71.3%, Charades-STA mIoU 59.9%

## 当前状态

1. 全景调研 **完成** — 领域全貌、gap、竞争者都已明确
2. 架构设计 **基本完成** — 三层架构 + 三级快慢系统 + 5 个工具
3. 框架选型 **倾向 VITAL** — 待确认 clone 和评估代码结构
4. 设计文档已更新 — `docs/memory/2026-02-26-change-driven-memory-design.md` 含 benchmark 和训练策略
5. **未开始实现** — 还在设计阶段

## 下一步

- [ ] Clone VITAL 代码 (`git clone https://github.com/zhang9302002/ThinkingWithVideos`)，评估推理路径可分离性
- [ ] 读 EventMemAgent 原文 (arXiv:2602.15329)，确认竞争关系细节
- [ ] 确定 Tier 1 摘要模型选择（Qwen2.5-VL-3B vs 7B）
- [ ] 确定 Proactive 触发逻辑的层级（Tier 0 vs Tier 1 vs Tier 2）
- [ ] 更新设计文档: 加入三级快慢系统、工具集定义、EventMemAgent 差异化策略
- [ ] Phase 0 实现: 在 EGTEA Gaze+ 上搭建 CLIP 变化驱动切片 prototype
- [ ] 选择一个 Benchmark（OVO-Bench 或 StreamBench）准备评估环境

## 相关文件

| 文件 | 内容 |
|------|------|
| `docs/memory/2026-02-26-change-driven-memory-design.md` | 核心设计文档（已更新含 benchmark 和训练策略） |
| `docs/survey/2026-02-25-change-detection-as-memory-mechanism.md` | 变化检测作为记忆管理机制 |
| `docs/LongVideoAgent.md` | VITAL / AVI / VideoARM 框架调研笔记 |
| `proactive-project/` | 已有的三层内存系统代码 |
| `temp/ml-streambridge/` | StreamBridge 代码（已 clone） |
