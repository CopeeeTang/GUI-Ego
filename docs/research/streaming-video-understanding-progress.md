# Streaming Video Understanding — Research Progress Update

> Date: 2026-02-26
> Author: Tangxin
> Status: In Progress
> Project: Smart Glasses GUI System

---

## 1. Research Gap

现有 Streaming Video LLM 在"何时主动说话"上统一采用 **单信号阈值范式**：

```
每帧 → 提取信号 S(t) → S(t) > τ ? → 说话 / 沉默
```

| 系统 | 发表 | 触发机制 |
|------|------|---------|
| VideoLLM-Online | CVPR'24 | 固定周期触发 |
| MMDuet / MMDuet2 | EMNLP'25 | Duet 格式 / RL 学习触发 |
| ProAssist | EMNLP'25 | w2t_prob > τ (trained binary head) |
| StreamBridge | NeurIPS'25 | 解耦 activation model (0.5B) |
| Dispider | CVPR'25 | 解耦 decision module |
| EyeWO | arXiv'25 | 端到端三阶段数据引擎 |

**Gap 1 — 缺乏系统性验证**: 没有工作跨 benchmark 验证这些触发信号在什么条件下有效、什么条件下失败。

**Gap 2 — 范式局限**: Proactive timing 可能不是一维阈值问题，而是需要多因素推理（信息缺口 × 信息相关性 × 用户接受度 × 时序合理性）。

**Gap 3 — Memory 被动管理**: 现有 memory 策略（KV-cache drop, round-decayed compression）都是被动/时间驱动的，没有根据内容变化主动管理。

---

## 2. Research Questions

### RQ1: Proactive Intervention Timing

> 在流式第一人称视频中，单信号阈值触发范式是否足以解决主动干预时机问题？如果不够，需要什么样的替代架构？

### RQ2: Streaming Context Memory

> 在长时间流式视频理解中，如何设计内容驱动（而非时间驱动）的记忆管理机制？

**两者关系**: RQ1 的决策质量依赖 RQ2 提供的上下文质量；RQ2 的"什么值得记"可复用 RQ1 中发现的信号特性（如变化检测从失败的 trigger 重定位为 memory 机制）。

---

## 3. Existing Benchmarks & Related Work

### 3.1 使用的 Benchmark

| Benchmark | 来源 | 规模 | 用途 |
|-----------|------|------|------|
| **ESTP-Bench** | EyeWO (arXiv'25) | 164 视频, 1212 QA, 12 任务类型 | RQ1 信号探索 (Phase I-III) |
| **ProAssist** | Zhang et al. (EMNLP'25) | 30K 对话, WTAG + ego4d | RQ1 变化检测 + 推理触发验证 |
| **EGTEA Gaze+** | Georgia Tech | 28h 烹饪, 眼动 + 动作标注 | 早期原型验证 (三层记忆) |

### 3.2 Related Work 分类

**A. 端到端训练方法** (学习 when + what)
- VideoLLM-Online (CVPR'24): LIVE 框架，开创性工作
- MMDuet/MMDuet2 (EMNLP'25 / arXiv'25): Duet 交互 + RL 阈值替代
- EyeWO (arXiv'25): 三阶段数据引擎 + 动态压缩，ESTP-Bench SOTA
- LiveCC (CVPR'25): ASR 时间戳驱动大规模训练

**B. 解耦触发方法** (分离 when 和 what)
- StreamBridge (NeurIPS'25): 0.5B activation + 7B generation，Round-Decayed Compression
- Dispider (CVPR'25): Perception-Decision-Reaction 三模块异步

**C. Memory/Compression 方法**
- ProAssist: KV-cache 级 (DROP_ALL / DROP_MIDDLE / SUMMARIZE_AND_DROP)
- StreamBridge: Embedding 级 round-decayed mean-pooling
- StreamAgent (ICLR'26): 层自适应 KV 召回 + Zoom 工具

### 3.3 我们的定位

不是提出新模型，而是通过系统性实验揭示现有范式的根本局限，并提出替代的 Agent 架构方向。

---

## 4. Our Approach

### 4.1 系统性信号探索 (RQ1)

在两个 benchmark 上穷举测试 7 类触发信号：

| 信号类型 | 具体方法 | 检测的维度 |
|---------|---------|----------|
| 模型不确定性 | logprob gap, w2t_prob | 模型自身置信度 |
| 视觉状态判断 | Goal-state (Qwen/Gemini/GPT-4o) | 用户是否 on-track |
| 视觉变化 | SigLIP CLS + FDA-CLIP (8 种) | 帧间变化量 |
| 语义推理 | GPT-4o reasoning trigger | "用户是否需要帮助" |
| 特征融合 | MLP learned trigger (v1-v6) | 多特征组合 |
| 时间规则 | Cooldown, 周期触发 | 时序间隔 |
| 自适应阈值 | Adaptive-D (per task type τ) | 超参数优化 |

### 4.2 提出的新架构: Information Gap Agent

基于实验结论提出范式转移——从 Signal Engineering 到 Agent Architecture：

触发条件不再是 S(t) > τ，而是：

> 当前存在一个"信息缺口"——有与当前场景相关的、用户尚未获得的信息，且当前时机适合传递。

```
Video Stream → Change-Driven Memory → Reasoning Agent (VLM)
                                            ↑
                                    Task Knowledge + Dialog History
                                            ↓
                                    说/不说 + 说什么 + 为什么
```

各模块与实验证据对应：

| 模块 | 实验证据 |
|------|---------|
| Change-Driven Memory | 86% 帧可跳过; 变化检测精准区分稳态/变化 |
| Reasoning Agent | GPT-4o 检测 error + step transition (ORACLE +7.5%) |
| Temporal Context | w2t_prob(有时序) >> GPT-4o(无时序)，Recall 差 2× |
| Dialog State | ProAssist 42% 是 confirmation，需对话历史 |
| Task Knowledge | Task Understanding 需 bypass D (sep=-2.072) |

### 4.3 Change-Driven Segment Memory (RQ2, idea stage)

变化检测在 trigger 上失败，但重定位为 memory 管理机制：

| 变化程度 | 操作 | 效果 |
|---------|------|------|
| 低变化 (Δ < τ_low) | 跳过/稀疏采样 | ~90% 帧节省 |
| 中变化 | 局部 patch 更新 | 复用不变区域 KV-cache |
| 高变化 (Δ ≥ τ_high) | 全帧重编码 + 段落摘要 | 触发上下文重建 |

与现有方法差异化：

| 维度 | StreamAgent (ICLR'26) | StreamBridge (NeurIPS'25) | Ours |
|------|----------------------|--------------------------|------|
| 记忆形成 | 被动 KV-cache 积累 | 轮回衰减 mean-pooling | 变化驱动的主动切片 |
| 记忆粒度 | token-level | embedding-level | 段落级语义索引 + 关键帧 |
| 检索方式 | 层自适应 KV 召回 | 无检索 | summary 索引 → 按需加载 |

**状态**: 设计阶段，待实验验证。

---

## 5. Evaluation Design

### 5.1 ESTP-Bench 实验

| Phase | 规模 | 目的 | 状态 |
|-------|------|------|------|
| Phase I | 60 QA 全量 | Baseline vs 统一 D@τ=2.8 | ✅ 完成 |
| Phase II | 53 matched cases | Per-task-type Adaptive-D | ✅ 完成 |
| Phase III | 各方法 pilot | Goal-state / Reasoning / Learned | ✅ 完成 |

评估指标: **ESTP-F1** (anticipation=1s, latency=2s 的匹配 F1)

### 5.2 ProAssist 实验

| 实验 | 规模 | 目的 | 状态 |
|------|------|------|------|
| 阈值扫描 | WTAG 51×8τ, ego4d 38×3τ | w2t_prob 最优 τ | ✅ 完成 |
| 变化检测 | 5 样本, 8 种方法 | SigLIP/FDA-CLIP trigger | ✅ 完成 (REJECTED) |
| 推理触发 | 5 样本, 100 帧 | GPT-4o reasoning | ✅ 完成 |

评估指标: **ProAssist F1** (时间窗口匹配 Precision × Recall)

---

## 6. Results

### 6.1 信号 × 平台 × 结果矩阵

| 信号 | ESTP-Bench | ProAssist | 结论 |
|------|-----------|-----------|------|
| **Logprob Gap** | Adaptive F1=0.222 (+57%) | N/A | ✅ 唯一 PASS |
| **w2t_prob** | N/A | F1=0.373 (+11.6%) | ✅ 有效但域特异 |
| **Goal-State** | 反向 (sep=-0.636, 3 VLM 一致) | 未测 | ❌ 结构性矛盾 |
| **Change Detection CLS** | 弱正, F1=0.121 | 反向 (sep<0.001) | ❌ 无效 |
| **FDA-CLIP (8种)** | 未测 | AUC 全<0.50, sep 全负 | ❌ 彻底关闭 |
| **Reasoning Trigger** | 仅 Text-Rich 正 (F1=0.667) | 弱正交 (r=0.031) | ⚠️ 高度任务依赖 |
| **MLP Learned** | 弱正但过拟合 (F1≈0.219) | 未测 | ⚠️ 不稳定 |

### 6.2 数值结果

**ESTP-Bench:**
- Baseline (periodic): F1=0.127, FP=1516
- D@τ=2.8 (Phase I): F1=0.114, FP=287 → Gate FAIL (95% CI crosses zero)
- Adaptive-D (Phase II): F1=0.222, Δ=+0.081, CI=[0.046, 0.121] → Gate PASS
- 按任务类型最优 τ: Info Fn=2.8, Text-Rich=2.0, Object Recog=2.5, Action Recog=0.5, Task Understanding=0.0 (bypass)

**ProAssist:**
- WTAG best: τ*=0.20, F1=0.3727 (论文 0.3340, +11.6%)
- ego4d best: τ*=0.30, F1=0.3056 (论文 0.2750, +11.1%)
- Change detection: 8 种方法 AUC 全<0.50 → REJECTED
- GPT-4o reasoning: F1=0.269, 与 w2t_prob 正交 (r=0.031), ORACLE +7.5%

### 6.3 三个核心发现

**发现 1: 所有外部观察信号失败 — Need vs Opportunity 错配**

| 信号 | 检测的是 | GT 标注的是 | 结果 |
|------|---------|-----------|------|
| Goal-state | 用户是否困难 (need) | 信息传递最佳时机 (opportunity) | 反向 |
| Change detection | 操作流畅度 | 信息需求 | 反向 |
| Reasoning trigger | "需要帮助吗" | "现在说最有效" | 部分有效 |

根因：好的教师在学生做对时确认 (on_track → 该说话)，不是等困惑才说 (off_track)。ESTP 中 42%+ GT 是 confirmation 类型。

**发现 2: 只有模型自身不确定性信号有效，但不可泛化**

w2t_prob 和 logprob gap 有效，但本质是超参调优 (argmax_τ F1(S>τ, GT))，需要 GT 搜索最优 τ，不同数据集/任务类型需不同 τ。

**发现 3: Proactive timing 是四维度推理问题**

| 维度 | 需要的信息 | 哪个信号能覆盖 |
|------|----------|--------------|
| 信息缺口 | Dialog history | 无 |
| 信息相关性 | Visual understanding | Reasoning (部分) |
| 用户接受度 | User state | Goal-state (但反向) |
| 时机合理性 | Temporal context | Cooldown (简单规则) |

没有任何单信号能同时编码四个因素 → 需要 Agent 架构做多因素推理。

---

## 7. Next Steps

### 近期 (RQ1 → Agent 验证)

1. **Information Gap Agent 原型**: 在 ESTP-Bench 上实现 Memory + Reasoning Agent，对比 Adaptive-D baseline
2. **信息缺口形式化**: 定义 gap(t) = relevance(context, unsaid_info) × urgency(task_state)

### 中期 (RQ2 → Memory 实验)

3. **Change-Driven Memory 验证**: 在 ProAssist SigLIP 特征上实现变化驱动帧选择，对比均匀采样
4. **Memory 质量评估**: 相同 token budget 下，变化驱动 vs 时间驱动的下游对话质量

### 开放问题

- 信息缺口能否有效量化？还是只能端到端？
- Agent 推理延迟如何与实时性平衡？
- 该架构是 training-free 的吗？哪些模块需要训练？

---

## 文件索引

| 内容 | 路径 |
|------|------|
| 核心 RQ 定义 | `docs/survey/streaming-gui-research-questions.md` |
| ProAssist 验证结论 | `docs/survey/2026-02-26-ProAssist-Validation-Conclusions.md` |
| ProAssist 实验报告 | `proassist_experiments/results/final_report.md` |
| ESTP Phase I 报告 | `proactive-project/.../fullscale_d/fullscale_d_report.txt` |
| ESTP Phase II 报告 | `proactive-project/.../phase2_stratified/phase2_report.txt` |
| 文献调研 | `docs/survey/2026-02-24-ProAssist-Literature-Survey.md` |
| Memory 设计 | `docs/memory/2026-02-26-change-driven-memory-design.md` |
| EGTEA 原型报告 | `docs/proactive-memory-eval-report-20260215.md` |
