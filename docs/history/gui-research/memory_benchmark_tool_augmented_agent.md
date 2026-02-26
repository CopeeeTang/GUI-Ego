# Session: Memory Management for Streaming Video — Tool-Augmented Agent 讨论

Session ID: `c1770285-1b82-40c2-acf5-2fc9cbc6aa21`
日期: 2026-02 (未命名 session)
状态: **进行中** — 停在 novelty 方向选择 (A/B/C)

---

## 用户 (Round 1)

思考一下一个顶尖的科研工作者会如何解决跨领域迁移的问题，研究的方向是 Memory Management For Streaming Video。我从 Agent Memory 的管理之中有了一些灵感，主要是看了 Claude 团队搭建 Claude Code memory system 的 blog，以及操作文档。

我认为几个值得研究的方向可能是：
- **Tool-augmented agent** 来解决问题，但问题是有什么适合流式视频的工具，以及如何利用工具
- 会不会像人看视频会生成每段的摘要和时间点来划分？就像人看视频的时候也可以看回放，但只有一个大概的印象(总结)，不可能每帧都记得
- Serena 这类检索工具迁移到 streaming video？
- 需要思考 streaming video 的 workspace 与 coding 的 workspace 的最大差异是什么？

参考文献：
- StreamBridge (arXiv:2505.05467, NeurIPS'25) — 已克隆至 `/home/v-tangxin/GUI/temp/ml-streambridge`
- StreamAgent (ICLR'26)

---

## 助手 — 三路调研综合分析

### 当前领域主流进展（三条技术路线）

| 路线 | 代表工作 | 核心思想 |
|------|---------|---------|
| **KV-Cache 分层记忆** | HERMES, StreamMem, MemStream, StreamChat | 将 KV-cache 映射为感觉/工作/长期记忆，分层压缩与检索 |
| **训练式流式适配** | StreamBridge (NeurIPS'25), StreamingVLM | 训练轻量激活模型 + 嵌入空间内压缩，无外部工具 |
| **工具增强 Agent** | StreamAgent (ICLR'26), DVD, VideoMind, VideoAgent | LLM 作为规划器，调用外部工具完成理解 |

### StreamBridge 深度分析

- **三层结构**：past_embeds（长期历史）→ cache（当前帧缓冲）→ modality_indicators（模态追踪）
- **轮回衰减压缩**：旧帧 mean-pooling 压缩，新帧保留细节，文本段不压缩
- **分离架构**：轻量 SigLIP 分类器决定"何时响应"（~5ms），重量级 VLM 决定"响应什么"
- **关键局限**：没有工具增强，所有操作都在 tensor 级别，无法"回看"或"检索"

### StreamAgent (ICLR'26) — 最接近用户设想

- **预期性规划**：三种模式（反应/前瞻/推测），预测未来关键时间区间和空间区域
- **分层 KV-cache**：GPU 驻留短期记忆 + CPU 卸载长期记忆 + 层自适应选择性召回
- **工具集**：空间放大（Zoom）等感知工具，主动决定 when/where/how 获取信息
- 比 Dispider 在 Forward Active Responding 上提升 **10.7%**

### Claude Code 记忆系统核心设计

- **六级层叠作用域**：组织 → 用户 → 项目 → 本地 → 子目录 → Auto Memory
- **200行索引约束**：MEMORY.md 强制精简，详细内容分散到 topic 文件按需加载
- **`@import` 模块化**：支持 5 级递归导入
- **路径条件激活**：`.claude/rules/*.md` 绑定文件路径 glob
- **渐进发现**：父目录全量加载，子目录按需加载

### 跨领域迁移核心张力

> Claude Code 的记忆系统有效是因为代码工作空间有天然的**符号结构**（文件、类、函数、变量）和**引用关系**（import、调用图）。流式视频是**连续的、非符号的、时间单向的** — 没有 `file_path:line_number`，没有 `Class/method`，没有 `@import`。迁移不能是 1:1 映射，需要重新定义视频工作空间的"符号"和"引用"。

---

## 用户选择: 倾向 B — Tool-Augmented Agent 框架

---

## 助手 — B 路线展开

### StreamAgent 的工具设计缺口

StreamAgent 的核心贡献在于"预期性规划"，它证明了 agent 式的 when/where/how 决策优于被动触发。但它的工具集非常有限 — 基本只有 Zoom（空间放大）。Claude Code 生态中光 Serena 就有十几个工具，每个解决信息获取的不同粒度和维度。这个"工具丰富度"的差距是机会所在。

### 待决策：核心新颖性（novelty）方向

- **A）视频工具集的设计**：定义一套完整的、受 coding agent 启发的视频理解工具（摘要生成、时间段检索、关键帧定位、场景对比、细节放大等），核心贡献是"什么工具对流式视频有用"以及"工具之间如何协作"

- **B）记忆架构的迁移**：将 Claude Code 的分层记忆思想（层叠作用域、200行索引约束、按需加载、路径条件激活）系统性迁移到视频领域，核心贡献是"如何为流式视频设计分层记忆"

- **C）两者融合 — 记忆驱动的工具调用**：记忆系统与工具集协同设计的框架，记忆不只是被动存储，而是驱动 agent 决定调用什么工具、何时调用（就像 MEMORY.md 影响后续所有工具调用行为）

**助手倾向 C**：单独的工具设计或记忆架构容易被质疑增量性不足，"记忆如何驱动工具使用"恰好是 StreamAgent 没有深入探讨的 — 它的 KV-cache 记忆和工具调用是相对独立的。

---

## 下一步

用户尚未回复 novelty 方向选择。
