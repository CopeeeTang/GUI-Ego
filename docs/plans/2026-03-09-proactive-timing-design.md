---
title: "Proactive Timing Design: From Single-Signal Threshold to Information-Gap Decision"
date: 2026-03-09
status: draft
author: Xin Tang
---

# Proactive Timing Design

## TL;DR

流式第一人称视频中的 proactive timing 问题不能用"单信号阈值"解决。跨 ESTP-Bench 和 ProAssist 两个数据集的系统性实验表明：(1) 所有外部观察信号（goal-state、change detection、reasoning trigger）因 need vs opportunity 错配而失败；(2) 唯一有效的模型内部不确定性信号（logprob gap、w2t_prob）本质是超参调优，无法泛化。核心洞察是 **timing 应建模为 memory 状态的函数**——当存在信息缺口（系统知道但用户不知道的相关信息）且时机合适时触发。这将 timing 从独立信号问题转化为 context engineering 框架下的多因素推理问题，与 streaming memory 架构自然统一。

---

## 1. 问题定义

### 1.1 核心问题

在流式第一人称视频中，系统何时应当主动介入（proactive speaking）？

这是 streaming video agent 从"被动问答"走向"主动助手"的关键瓶颈。

### 1.2 现有范式及其局限

所有 Streaming Video LLM 统一采用 **单信号阈值范式**：

```
每帧 → 提取信号 S(t) → S(t) > τ ? → 说话 / 沉默
```

这个范式有两个根本问题：
- **单信号无法编码多维决策**：proactive timing 至少涉及四个维度（见 §5.3）
- **阈值 τ 需要 GT 搜索**：本质是超参调优，无法泛化到新场景

### 1.3 已验证信号汇总

| 信号 | 数据集 | 最佳 F1 | Gate 结论 | 备注 |
|------|--------|---------|-----------|------|
| Logprob Gap (Adaptive-D) | ESTP-Bench | 0.222 (+57%) | PASS | 唯一通过，但本质是逐任务类型调 τ |
| w2t_prob | ProAssist | 0.373 (+11.6%) | PASS | 有效但域特异，与 change detection 正交 |
| Goal-State Classification | ESTP-Bench | — | FAIL | 反向 (sep=-0.636)，3 个 VLM 一致 |
| Change Detection (8种) | ProAssist | — | FAIL | AUC 全 < 0.50，彻底关闭 |
| Reasoning Trigger (Gemini) | ESTP-Bench | 0.173 | FAIL | 仅 Text-Rich 正 (F1=0.667)，高度任务依赖 |

---

## 2. Information-Gap Decision 范式

### 2.1 核心思想

触发条件不再是 `S(t) > τ`，而是：

> **当前存在信息缺口——有与当前场景相关的、用户尚未获得的信息，且当前时机适合传递。**

这将 timing 从"检测某个信号超过阈值"转变为"推理当前是否存在值得传递的信息"。

### 2.2 与 Streaming 的兼容性：多速率决策

Information-gap reasoning 不意味着每帧做完整推理，而是采用多速率架构：

**Fast Path（每帧或每 few frames）**——只做廉价更新：
- CLIP delta（场景变化检测）
- Step transition 候选（动作切换检测）
- Safety cue 候选（危险信号检测）
- Uncertainty cue（模型不确定性）
- 输出：`candidate_trigger_score`

**Slow Path（仅候选分数过线时）**——调用 reasoning：
- 输入：当前 working memory + 最新 progress summary + top-k relevant segments + 必要时 top-3 keyframes
- 输出：是否触发 + 内容生成

**关键效率数字**：
- ~90% 帧被 CLIP delta 过滤（不进入后续流程）
- Segment 平均长度 ~13s
- VLM reasoning 平均每 ~13s 才可能被调用一次（而非每帧）

---

## 3. Timing 作为 Memory 状态的函数

### 3.1 核心洞察

Timing 不需要自己的独立信号，它可以直接从 memory 状态中推导。

### 3.2 形式化

```python
def should_speak(memory, current_frame):
    # 维度 1: 信息缺口 (Gap)
    unsaid_segments = [
        s for s in memory.recent_segments
        if s.communicated == False and s.relevance > threshold
    ]
    has_gap = len(unsaid_segments) > 0

    # 维度 2: 用户接受度 (Opportunity)
    user_idle = current_frame.activity_change_rate < idle_threshold

    # 维度 3: 时序合理性 (Non-disruption)
    cooldown_ok = time_since_last_speak > min_cooldown
    not_in_critical_action = current_frame.action_density < critical_threshold

    return has_gap and user_idle and cooldown_ok and not_in_critical_action
```

### 3.3 三个决策维度

| 维度 | 含义 | 近似方式 |
|------|------|----------|
| **Opportunity(now)** | 当前是否适合说话 | 步骤转换、停顿、确认时机 |
| **Gap(now)** | 是否有未传达且相关的信息 | memory 中 `communicated=False` 且 `relevance > θ` 的 segments |
| **Disruption(now)** | 说话是否会打断用户 | cooldown 时间、当前动作密集度、安全风险 |

---

## 4. 与 Memory 的连接

### 4.1 统一框架

Memory 和 Timing 不是两个独立模块，而是同一个 **Context Engineering 框架**的两面：

- **Memory** 回答："系统当前知道什么"
- **Timing** 回答："系统当前知道的 vs 用户需要知道的 之间有没有 gap"

因此，**timing quality depends on memory quality**——如果 memory 不准确，timing 决策也必然失败。

### 4.2 信号正交性的解释

ProAssist 实验发现 w2t_prob 和 change detection 正交 (`r=0.031`)，说明"何时说话"和"画面变了"是两个独立维度。

但如果将 timing 建模为 `gap(memory_state, user_state)`，这两个维度就自然统一：
- Change detection 更新 `memory_state`（画面变了 → memory 需要更新）
- w2t_prob 近似 `gap` 的大小（模型不确定性高 → 可能存在信息缺口）
- 两者正交是因为它们分别编码了 gap 函数的不同输入，而非 gap 本身

---

## 5. 实验发现总结

### 5.1 发现 1: 所有外部观察信号失败——Need vs Opportunity 错配

所有基于"检测用户困难/画面变化"的信号都失败了，根本原因是 **GT 标注的 timing 是 opportunity-based 而非 need-based**：

- 好的教师在学生**做对时确认** (`on_track → 该说话`)，而不是等学生困惑才说
- ESTP-Bench 中 42%+ GT 是 confirmation 类型
- Goal-state 反相关是结构性的：用户 on_track 时恰好是教学机会

### 5.2 发现 2: 模型内部不确定性信号有效但不可泛化

Logprob gap 和 w2t_prob 是唯二通过 gate 的信号，但：
- 需要逐任务类型搜索最优 τ（本质是超参调优）
- Adaptive-D 的价值不在于方法贡献，而在于揭示信号 calibration 差异
- 无法直接迁移到新任务类型

### 5.3 发现 3: Proactive Timing 是四维度推理问题

| 维度 | 含义 | 为什么单信号不够 |
|------|------|------------------|
| 信息缺口 | 系统有用户不知道的相关信息 | 需要 memory 状态 |
| 信息相关性 | 该信息与当前场景相关 | 需要 context matching |
| 用户接受度 | 用户当前可以接收信息 | 需要 activity 分析 |
| 时序合理性 | 此刻说话不会打断关键动作 | 需要 temporal reasoning |

没有任何单信号能同时编码四个因素 → 需要 Agent 架构做多因素推理。

---

## 6. 下一步计划

| 阶段 | 目标 | 方法 |
|------|------|------|
| **短期** | 验证 timing 作为 memory 状态函数的可行性 | Training-free 框架，基于 streaming memory agent 的 memory 状态做 rule-based timing |
| **中期** | 如果有 clear signal，训练专用模型 | Information-gap scorer 或 DGRPO timing decision |
| **长期** | 端到端联合训练 | SFT+RL 联合训练 timing + content generation |

---

## 7. 相关文献

| 文献 | 会议 | 核心方法 | 与本工作关系 |
|------|------|----------|-------------|
| VideoLLM-Online | CVPR'24 | LIVE 框架，EOS Token 触发 | 最早的 streaming timing 方案，但 EOS 只编码"说完了" |
| ProAssist | EMNLP'25 | w2t_prob 触发 | 提供了 w2t_prob 基线和 ProAssist 数据集 |
| MMDuet2 | arXiv'25 | GRPO RL 替代手动阈值 | RL 自动学 timing，但仍是单信号 |
| StreamBridge | NeurIPS'25 | 0.5B activation model | 轻量级 timing 模型，但 training-dependent |
| EyeWO | NeurIPS'25 | ESTP-Bench, ESTP-F1=34.7% | 提供了 ESTP-Bench 评测框架 |
| VITAL | CVPR'26 | DGRPO 训练 timing + content | 最接近的方法，但 timing 仍是单维度 |
| EventMemAgent | arXiv'26.02 | 双层事件记忆 + GRPO | Memory-aware timing 的早期探索 |
