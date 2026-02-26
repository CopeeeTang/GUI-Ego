# 从信号工程到 Agent 架构：Proactive Timing 的范式转移

> 来源: ProAssist (EMNLP 2025) 验证实验 + ESTP-Bench Phase I-III
> 日期: 2026-02-24 ~ 2026-02-26
> 结论性质: 实验驱动的方向转向

---

## 一、问题定义

**Proactive Timing**: 在流式视频辅助场景中，系统何时应该主动开口说话？

现有方法统一范式:
```
每帧 → 提取某种信号 S(t) → S(t) > τ ? → 说话 / 沉默
```

本文档基于在 ESTP-Bench（12 任务类型, 164 视频）和 ProAssist（WTAG + ego4d, 89 样本）上的系统性实验，论证**这个范式的根本局限**，并提出替代方向。

---

## 二、全部信号实验汇总

### 2.1 信号 × 平台 × 结果矩阵

| 信号 | 原理 | ESTP-Bench | ProAssist | 结论 |
|------|------|-----------|-----------|------|
| **Logprob Gap** | P(yes)-P(no) from VLM | 正相关, F1=0.222 (Adaptive-D) | N/A | 唯一 PASS |
| **w2t_prob** | Trained binary head | N/A | 正相关, F1=0.373 | 双峰分离好 |
| **Goal-State** | VLM 判断 on/off track | **反向** (sep=-0.636) | 未测 | 结构性矛盾 |
| **Change Detection (CLS)** | SigLIP 帧间余弦距离 | 弱正, F1=0.121 | **反向** (sep<0.001) | 无效 |
| **Change Detection (FDA-CLIP)** | Patch-level 8种方法 | 未测 | **反向** (AUC全<0.50) | 彻底关闭 |
| **Reasoning Trigger** | VLM 推理"需要帮助?" | 仅 Text-Rich 正 (F1=0.667) | 弱正交 (r=0.031) | 高度任务依赖 |
| **MLP Learned** | 多特征融合分类器 | 弱正但过拟合 (F1≈0.219) | 未测 | 不稳定 |

### 2.2 Adaptive-D 跨数据集验证

| 数据集 | 信号 | 我们 τ* | 我们 F1 | 论文 F1 | 提升 |
|--------|------|---------|---------|---------|------|
| ESTP (Phase II) | logprob gap | per-type | 0.222 | 0.141 | +57% |
| ProAssist WTAG | w2t_prob | 0.20 | 0.3727 | 0.3340 | +11.6% |
| ProAssist ego4d | w2t_prob | 0.30 | 0.3056 | 0.2750 | +11.1% |

**Adaptive-D 的定位**: 有效但只是超参调优。不提出新信号，不泛化到新数据集（需 GT 做搜索），本质是 `argmax_τ F1(S>τ, GT)`。价值在于揭示现有信号 calibration 差的现象。

---

## 三、为什么所有外部信号失败？

### 3.1 Need-Based vs Opportunity-Based 的根本错配

所有外部观察信号（goal-state、change detection、reasoning trigger）都在回答：

> **"用户现在遇到困难了吗？"** (need-based detection)

但 GT 标注的实际问题是：

> **"这个时刻提供信息最有效？"** (opportunity-based timing)

这是两个不同的问题。

### 3.2 具体的错配模式

| 场景 | 用户视觉状态 | 外部信号判断 | GT 标注 | 匹配？ |
|------|------------|------------|---------|--------|
| 用户盯着目标物体 | on_track | "不需要帮助" | **该说话**（识别/描述信息） | **否** |
| 用户完成了一步 | on_track | "不需要帮助" | **该说话**（确认 + 下一步） | **否** |
| 用户用错了工具 | off_track | "需要帮助" | 该说话（纠错） | 是 |
| 用户在等水开 | idle/低变化 | "可能需要帮助" | **不该说话** | **否** |
| 用户在快速切菜 | active/高变化 | "不需要帮助" | **不该说话** | 是（偶然） |

### 3.3 Goal-State 反相关的深层原因

ESTP 中 goal-state sep = -0.636（三个模型一致），因为：
- GT 中 42%+ 是 **confirmation** 类型（"Great!", "Yes, that's right"）
- Confirmation 发生在用户**做对了**的时候 → goal-state = on_track
- 好的教师不是等学生困惑才教，而是在学生**注意力集中在正确位置**时给出信息

### 3.4 Change Detection 反向的深层原因

ProAssist 中所有 8 种变化检测方法 separation 为负，因为：
- 用户**需要帮助时往往是卡住了**（盯着材料发呆、手停在半空） → 低变化
- 用户**不需要帮助时在流畅操作**（切菜、翻炒、移动） → 高变化
- 视觉变化反映的是**操作流畅度**，不是**信息需求**

### 3.5 为什么模型自身信号有效？

w2t_prob 和 logprob gap 之所以有效，不是因为它们检测了"用户需要帮助"，而是因为：
- **w2t_prob**: 从对话历史的时序模式中学到了"confirmation 通常在什么时机出现"
- **logprob gap**: 反映"模型对当前帧的回答有多确定"，间接编码了信息相关性

但它们的局限也很明显：
- w2t_prob 在纯视觉场景（ego4d）犹豫不决 → 不同数据集需不同阈值
- logprob gap 在 Task Understanding 类型上反向 → 有些任务信号完全无效
- **都不是 well-calibrated 的**: "0.3" 在 WTAG 和 ego4d 上含义不同

---

## 四、核心结论：单信号不可能跨场景泛化

这不是 calibration 可以解决的问题，而是**建模层面的根本局限**。

Proactive timing 的决策取决于至少四个独立维度，任何单一信号最多覆盖其中一个：

| 维度 | 含义 | 所需信息 | 哪个信号能覆盖？ |
|------|------|---------|----------------|
| **信息缺口** | 有什么信息用户还没收到？ | Dialog history | 无（w2t_prob 部分隐式编码） |
| **信息相关性** | 当前场景与哪条信息相关？ | Visual understanding | Reasoning trigger（部分） |
| **用户接受度** | 用户现在是否适合接收信息？ | User state | Goal-state（但方向反了） |
| **时机合理性** | 距上次交互是否足够远？ | Temporal context | Cooldown（简单规则） |

**没有任何单一信号能同时编码这四个因素。** 这就是为什么：
- 每个信号只在某些任务类型有效（恰好该维度是瓶颈的场景）
- 其余场景要么无效、要么反向
- 融合多信号（MLP）也不稳定（过拟合）

---

## 五、范式转移：从 Signal → Agent

### 5.1 旧范式的本质

```
Signal Engineering: 找一个更好的 S(t)，使 S(t) > τ 更接近 GT
```

问题：这把一个多因素推理问题降维成了一维分类，信息损失是根本性的。

### 5.2 新范式：Information Gap Agent

```
Agent Architecture: 构建一个能综合多源信息做推理决策的系统
```

核心思想：触发条件不是某个信号超过阈值，而是：

> **当前存在一个"信息缺口"——有与当前场景相关的、用户尚未获得的信息，且当前时机适合传递。**

### 5.3 架构设计

```
                    ┌──────────────────────┐
                    │   Task Knowledge     │
                    │   (步骤/知识/FAQ)     │
                    └──────────┬───────────┘
                               │
┌───────────┐     ┌────────────▼───────────┐     ┌──────────────┐
│           │     │                        │     │              │
│  Video    │────▶│   Change-Driven       │────▶│  Reasoning   │
│  Stream   │     │   Memory Module       │     │  Agent (VLM) │
│           │     │                        │     │              │
└───────────┘     │  ・低变化→跳过(~86%)    │     │  输入:        │
                  │  ・中变化→局部patch更新  │     │  - 当前帧     │
                  │  ・高变化→全帧+段落摘要  │     │  - 记忆摘要   │
                  └────────────────────────┘     │  - 任务知识   │
                                                 │  - 对话历史   │
┌───────────┐                                    │              │
│  Dialog   │───────────────────────────────────▶│  输出:        │
│  History  │                                    │  - 说/不说    │
│  (已说/未说)│                                    │  - 说什么     │
└───────────┘                                    │  - 为什么     │
                                                 └──────────────┘
```

### 5.4 各模块与实验证据的对应

| 模块 | 实验证据 | 贡献 |
|------|---------|------|
| **Change-Driven Memory** | 86% 帧可跳过; 变化检测精准区分稳态/变化 | 10× 效率提升 |
| **Reasoning Agent (VLM)** | GPT-4o 检测 error + step transition (ORACLE +7.5%) | 语义推理 |
| **Temporal Context** | w2t_prob(有时序) >> GPT-4o(无时序)，Recall 差 2× | 时序必要性 |
| **Dialog State** | ProAssist 42% 是 confirmation，需要对话历史判断 | 信息缺口 |
| **Task Knowledge** | Task Understanding 需 bypass D (sep=-2.072) | 任务特异性 |

### 5.5 为什么这个架构能解决之前所有信号的问题

| 之前失败原因 | Agent 如何解决 |
|------------|--------------|
| Goal-state on_track 但 GT 说该说话 | Agent 知道"用户正看着目标物" + "还没告诉他功能" → **信息缺口存在** → 触发 |
| Change 低(卡住)但不知道是等待还是困惑 | Memory 记录"上次指令 30s 前" + Agent 推理"该步骤应该已完成" → 触发 |
| Reasoning trigger 无时序上下文 | Memory 提供压缩的"过去发生了什么"，Agent 有完整上下文 |
| Confirmation 需要对话历史 | Dialog State 记录"刚完成步骤 3" → Agent 判断"该确认 + 给步骤 4" |
| Text-Rich 需语义匹配 | Agent 看到文字 + 知道任务 → 直接推理信息相关性 |

### 5.6 与现有工作的区别

| 系统 | Memory 策略 | 触发机制 | 推理能力 |
|------|-----------|---------|---------|
| ProAssist | SummarizeAndDrop (时间驱动) | w2t_prob > τ (单信号) | 无（trained head） |
| VideoLLM-Online | 固定间隔采样 | 周期性 | 无 |
| StreamBridge | 相似度过滤 | Token 触发 | 无 |
| ESTP-Bench (ours) | 无 | logprob gap > τ | 无 |
| **Proposed Agent** | **变化驱动（内容驱动）** | **信息缺口推理** | **VLM 多因素推理** |

---

## 六、变化检测重定位：Memory 管理机制

### 6.1 为什么不是 Trigger，而是 Memory

实验明确证明：
- 视觉变化 ≠ 对话需求（反向相关）
- 视觉变化 = 信息新颖度（正向相关：变化大 = 画面有新内容）

因此变化检测的正确用途不是"什么时候说话"，而是"什么时候有新内容值得记住"。

### 6.2 三层分层机制

| 层级 | 变化程度 | 操作 | 资源节省 |
|------|---------|------|---------|
| 层1 | 低变化 (Δ < τ_low) | 跳过/稀疏采样 | ~86% 帧（基于 w2t_prob > 0.99 占比） |
| 层2 | 中变化 (τ_low < Δ < τ_high) | 只更新变化 patch tokens | 复用 ~70% 不变区域的 KV-cache |
| 层3 | 高变化 (Δ > τ_high) | 全帧重编码 + 段落摘要 | 触发上下文重建 |

### 6.3 与人类注意力的类比

- 低变化 → 漫游模式（不形成新记忆）
- 中变化 → 聚焦模式（注意力集中到变化区域）
- 高变化 → 摘要模式（对前一段形成概括）

详细文档: `.claude/worktrees/memory/docs/survey/2026-02-25-change-detection-as-memory-mechanism.md`

---

## 七、下一步

### 7.1 Agent 架构验证 (新 worktree)

1. **Memory Module 原型**: 在 ProAssist SigLIP 特征上实现变化驱动的帧选择
   - 验证: 相同 token budget 下，变化驱动 vs 均匀采样的对话质量对比
2. **信息缺口定义**: 形式化 `gap(t) = relevance(context, unsaid_info) × urgency(task_state)`
3. **Reasoning Agent**: 带完整上下文（memory + dialog + task）的 VLM 推理
   - 对比: 有 memory context vs 无 memory context 的触发准确率

### 7.2 关键科学问题

1. 信息缺口能否被有效量化？还是只能做端到端？
2. Memory module 的变化阈值 (τ_low, τ_high) 是否也需要 per-dataset 调？还是可以用固定百分位？
3. Agent 的推理延迟如何与实时性需求平衡？
4. 这个架构是 training-free 的吗？哪些模块需要训练？

### 7.3 Benchmark 适配

- ESTP-Bench: 已有完整 FBF 推理框架，可直接嵌入 Agent
- ProAssist: 需要修改 StreamInferenceRunner 支持 Agent 决策
- 新的评估维度: 不只评 F1(timing)，还要评 "为什么现在说"（explainability）

---

## 附录: 关键数据速查

### A. WTAG F1 曲线 (ProAssist, 8 点)
```
τ=0.15 F1=0.352  |  τ=0.30 F1=0.345  |  τ=0.45 F1=0.321
τ=0.20 F1=0.373* |  τ=0.35 F1=0.356  |  τ=0.50 F1=0.328
τ=0.25 F1=0.349  |  τ=0.40 F1=0.331  |
```

### B. ESTP Phase II Adaptive-D 按任务类型 τ
```
Info Function=2.8  Text-Rich=2.0  Object Recog=2.5
Action Recog=0.5   Attribute=0.5  Object Function=0.5
Task Understanding=0.0 (BYPASS, sep=-2.072)
```

### C. 推理触发 GPT-4o on WTAG
```
F1=0.269  P=0.529  R=0.180  Yes-rate=17%
与 w2t_prob 相关性: r=0.031
ORACLE F1=0.561 (+7.5%)
```

### D. 文件索引
```
实验数据:  /home/v-tangxin/GUI/proassist_experiments/results/
最终报告:  proassist_experiments/results/final_report.md
实验发现:  proassist_experiments/results/interim_findings.md
文献调研:  docs/survey/2026-02-24-ProAssist-Literature-Survey.md
Memory文档: .claude/worktrees/memory/docs/survey/2026-02-25-change-detection-as-memory-mechanism.md
会话历史:  docs/history/proassist-validation/round_{1,2,3}.md
```
