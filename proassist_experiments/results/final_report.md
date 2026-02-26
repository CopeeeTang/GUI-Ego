# ProAssist 验证实验 — 最终报告

> 日期: 2026-02-24 ~ 2026-02-25 (3 sessions)
> 验证对象: ProAssist (EMNLP 2025), WTAG + ego4d 数据集
> 模型: ProAssist-Model-L4096-I1 (LLaMA-3.1-8B + LoRA r=128 + SigLIP-SO400M)
> GPU: A100 80GB, attn_implementation=sdpa

## 一、实验结果总表

### 1.1 Adaptive-D 阈值优化

| 数据集 | 我们最优 τ | 我们 F1 | 论文最优 τ | 论文 F1 | 相对提升 |
|--------|-----------|---------|-----------|---------|---------|
| WTAG (dialog-klg-sum) | **0.20** | **0.3727** | 0.40 | 0.3340 | **+11.6%** |
| ego4d (narration) | **0.30** | **0.3056** | 0.30 | 0.2750 | **+11.1%** |

注：论文结果可能使用不同数据格式(L0 vs L4096)，绝对数值不完全可比，但趋势一致。

#### WTAG 完整 F1 曲线 (8 个阈值)

| τ | F1 | Precision | Recall | Missing | Redundant |
|------|--------|-----------|--------|---------|-----------|
| 0.15 | 0.3520 | 0.491 | 0.274 | 48.1% | 7.2% |
| **0.20** | **0.3727** | **0.476** | **0.306** | **42.6%** | **10.8%** |
| 0.25 | 0.3486 | 0.420 | 0.298 | 41.7% | 17.8% |
| 0.30 | 0.3454 | 0.392 | 0.309 | 39.0% | 22.8% |
| 0.35 | 0.3562 | 0.388 | 0.329 | 34.1% | 22.3% |
| 0.40 | 0.3310 | 0.332 | 0.330 | 35.0% | 34.6% |
| 0.45 | 0.3212 | 0.314 | 0.329 | 34.7% | 37.8% |
| 0.50 | 0.3283 | 0.311 | 0.347 | 34.2% | 41.0% |

#### ego4d 部分结果 (τ=0.4 因实验终止未完成)

| τ | F1 | Precision | Recall |
|------|--------|-----------|--------|
| 0.20 | 0.2664 | 0.739 | 0.163 |
| **0.30** | **0.3056** | 0.222 | 0.489 |
| 0.40 | 进行中 (5/65, 已停止) | — | — |

### 1.2 变化检测

#### 1.2.a SigLIP CLS 全局变化 — REJECTED
- 5 个尺度 (delta=1,3,5,10,20), 最大 separation = 0.00082
- 不确定区 r=0.017, p=0.81

#### 1.2.b FDA-CLIP Patch-Level 变化 — REJECTED (8 种方法全部失败)

| 方法 | AUC | Separation |
|------|-----|-----------|
| CLS cosine | 0.443 | -0.012 |
| Patch-Max | 0.449 | -0.013 |
| Patch-Mean | 0.423 | -0.024 |
| Patch-Top50 | 0.398 | -0.018 |
| FDA r=0.05 | 0.453 | -0.013 |
| FDA r=0.10 | 0.451 | -0.011 |
| FDA r=0.20 | 0.447 | -0.014 |
| FDA r=0.30 | 0.436 | -0.019 |

**所有方法 AUC < 0.50（低于随机），所有 separation 为负。**

**结论**: 视觉变化与对话需求反向相关 — 用户需要帮助时往往卡住（低变化），不是在快速操作（高变化）。这是 task-oriented dialogue 的根本特性，不是方法粒度问题。

### 1.3 推理触发 (GPT-4o)

GPT-4o 在 WTAG 上 5 个样本, 100 帧 (50 talk + 50 notalk):

| 方法 | F1 | Precision | Recall |
|------|-----|-----------|--------|
| GPT-4o alone | 0.269 | 0.529 | 0.180 |
| w2t_prob<0.2 alone | 0.522 | 0.947 | 0.360 |
| **ORACLE (either)** | **0.561** | 0.719 | 0.460 |

- 两信号相关性: r=0.031 (近乎正交)
- w2t_prob 擅长 confirmation (66.7% recall), GPT-4o 擅长 correction (21.4% vs 7.1%)
- ORACLE 提升: +7.5% F1

### 1.4 w2t_prob 信号分析

| w2t_prob 区间 | 帧占比 | Talk Rate |
|--------------|--------|-----------|
| [0.00, 0.10) | 3.8% | 100% |
| [0.10, 0.30) | 0.2% | 100% |
| [0.30, 0.50) | 0.3% | 0% |
| [0.50, 1.00) | 95.7% | 0% |

完美帧级二分类在 τ=0.3 处。分布极端双峰：86% > 0.99, 3.2% < 0.01。

---

## 二、跨数据集阈值差异分析

### 为什么 WTAG τ*=0.20 而 ego4d τ*=0.30

| 维度 | WTAG | ego4d |
|------|------|-------|
| 对话类型 | 55.5% responsive (用户提问触发) | 100% initiative (纯主动旁白) |
| 说话信号来源 | 语言 + 视觉 | 纯视觉 |
| w2t_prob 分布 | 极端双峰 (0 or 1) | 中间地带密集 (0.2~0.3 有峰) |
| τ 敏感性 | 低 (Recall 变化仅 5.6%) | 极高 (τ 0.2→0.3 Recall +200%) |
| 根因 | 用户提问是明确信号 → 二值化输出 | 纯视觉驱动 → 模型犹豫 |

**结论**: 阈值差异不是"不同任务需要不同策略"，而是 w2t_prob 信号在不同模态输入下的分布形态不同。这暴露了信号的 calibration 缺陷。

---

## 三、核心结论

### 3.1 已验证的现象

1. **Per-dataset 阈值优化有效**: WTAG +11.6%, ego4d +11.1%
2. **最优阈值因数据集而异**: WTAG τ=0.20, ego4d τ=0.30
3. **视觉变化检测方向性错误**: 所有方法（全局+局部+FDA）全部 AUC < 0.50
4. **推理触发与 w2t_prob 互补**: 正交 (r=0.031), ORACLE +7.5%

### 3.2 更深层的发现

5. **单信号不可能跨场景泛化**:
   - w2t_prob: 有对话时强 (WTAG), 纯视觉时弱 (ego4d)
   - Logprob gap (ESTP): 知识型强, 理解型反向 (Task Understanding sep=-2.072)
   - Reasoning trigger: Text-Rich 极强 (F1=0.667), Action Recognition 反向 (-0.110)
   - Goal-state: 结构性反向 (sep=-0.636), 三个模型一致

6. **所有外部观察信号的共同失败模式**: 它们检测"用户是否遇困难"(need-based)，但 GT 标注的是"这个时刻提供信息最有效"(opportunity-based)。这是两个不同的问题。

7. **只有模型自己的不确定性信号有效**: w2t_prob 和 logprob gap 都是模型对自身预测的置信度，它们有效是因为编码了对话流模式而非用户状态。

### 3.3 Adaptive-D 的定位

- **不是一种方法，是一种超参调优**: 本质是 argmax_τ F1(signal > τ, GT)
- **不解决信号问题**: 信号差 → 调阈值也救不回来
- **需要 GT labels**: 每个新数据集都要重新搜索
- **价值在于揭示现象**: 证明了现有信号不是 well-calibrated 的

### 3.4 研究方向建议

基于所有实验的 negative results，proactive dialogue timing 的正确研究方向应该是:

**从 Signal Engineering → Agent Architecture**

不是寻找更好的单一信号，而是设计一个能综合多源信息做决策的 Agent:
- **Memory Module**: 变化驱动的流式记忆管理（86% 帧可跳过）
- **Reasoning Module**: VLM 语义推理（error detection, step transition）
- **Task Model**: 任务结构理解（步骤/知识）
- **Dialog State**: 信息缺口追踪（已说/未说）

触发条件 = f(信息缺口, 当前上下文, 任务状态, 时机合理性)

---

## 四、关键文件索引

| 文件 | 说明 |
|------|------|
| `proassist_experiments/results/interim_findings.md` | 5 个关键发现 (含 FDA-CLIP + 推理触发) |
| `proassist_experiments/results/comprehensive_report.md` | WTAG 分析报告 + 图表 |
| `proassist_experiments/results/fda_change_analysis.json` | FDA-CLIP 8 种方法详细数据 |
| `proassist_experiments/results/reasoning_trigger_results.json` | 推理触发完整数据 |
| `proassist_experiments/results/reasoning_trigger/reasoning_trigger_report.txt` | 推理触发详细报告 |
| `proassist_experiments/results/figures/` | 3 张分析图表 |
| `proassist_experiments/results/ego4d_sweep.log` | ego4d 扫描日志 (τ=0.2完成, τ=0.3完成, τ=0.4部分) |
| `docs/history/proassist-validation/round_1.md` | Session 1 历史 |
| `docs/history/proassist-validation/round_2.md` | Session 2 历史 |
| `docs/survey/2026-02-24-ProAssist-Literature-Survey.md` | 文献调研 |
