# Phase 3 Round 2 实验综合总结

> 日期：2026-02-21
> 工作目录：`proactive-project/experiments/estp_phase3/`
> 作者：estp-final 团队

---

## 1. 研究背景与问题

### 1.1 ESTP-Bench 评估体系

ESTP-Bench 是面向第一人称视频的主动式智能助手评估基准：

| 维度 | 规格 |
|------|------|
| 视频数量 | 164 个第一人称视频 |
| QA 总数 | 1,212 个问答对 |
| 任务类型 | 12 种（Action Recognition, Text-Rich Understanding 等） |
| 评估指标 | ESTP-F1（ANTICIPATION=1s, LATENCY=2s） |
| 采样率 | 0.175 Hz（约 5.7s/步） |

ESTP-F1 在严格的时序匹配窗口下评估触发质量：系统必须在 GT 事件开始前 1s 至结束后 2s 的窗口内发出触发，否则计为误报（FP）或漏报（FN）。

### 1.2 前期实验结果

**Phase I** — 全局阈值 Adaptive-D（60 cases）：

| 配置 | Avg F1 | Recall | Precision | FP | 结论 |
|------|--------|--------|-----------|-----|------|
| Baseline（gap>0, cd=12） | 0.127 | 0.668 | 0.086 | 1,516 | 高 recall 但 FP 泛滥 |
| D@tau=2.8+cd=12 | 0.114 | 0.237 | 0.137 | 287 | FP 减少 81%，但 recall 崩塌 65% |
| Delta F1 | -0.013 | — | — | — | **Gate: FAIL**，95% CI=[-0.059, +0.031] |

全局阈值 tau=2.8 过于严格，在 Action Recognition（recall=0）和 Attribute Perception 上造成灾难性损失。

**Phase II** — 自适应阈值 Adaptive-D（53 matched cases）：

| 配置 | Avg F1 | FP | Delta vs BL | 95% CI |
|------|--------|-----|-------------|--------|
| Baseline | 0.141 | 1,466 | — | — |
| D@2.8（全局） | 0.109 | 416 | -0.032 | [-0.071, +0.013] |
| **Adaptive-D（per-type tau）** | **0.222** | **416** | **+0.081** | **[0.046, +0.121]** |

自适应阈值通过为每种任务类型优化 tau 值，**Gate 2: PASS**。最优 tau 分布：

| 任务类型 | 最优 tau | 含义 |
|---------|---------|------|
| Information Function | 2.8 | 高阈值有效 |
| Text-Rich Understanding | 2.0 | 中高阈值 |
| Object Recognition | 2.5 | 中高阈值 |
| Ego Object Localization | 1.5 | 中等阈值 |
| Action Recognition | 0.5 | 极低阈值（需高频触发） |
| Attribute Perception | 0.5 | 极低阈值 |
| Object Function | 0.5 | 极低阈值 |
| Ego Object State Change | 0.5 | 极低阈值 |
| Task Understanding | 0.0 | 完全旁路（gap 反相关） |
| Object State Change | 0.0 | 完全旁路 |
| Action Reasoning | 0.0 | 完全旁路 |

**Reasoning Trigger（RT）**（15 cases，Gemini-3-Flash）：

| 方法 | avg F1 |
|------|--------|
| Reasoning Trigger | 0.173 |
| Adaptive-D (tau=2.8 全局) | 0.165 |
| Baseline | 0.107 |

RT 整体低于 Phase II Adaptive-D（0.222），但在 Text-Rich Understanding 上表现突出（F1=0.667 vs AD=0.154）。

### 1.3 本轮研究问题

基于上述结果，Phase 3 Round 2 聚焦以下五个核心问题：

1. **Yes-bias 跨模型验证**：Qwen3-VL 的 yes-rate=76-83%，GPT-4o 是否存在同样问题？
2. **CLIP 作为触发候选检测**：能否用 CLIP 视觉变化作为预过滤器降低 VLM 调用频率？
3. **Logprob gap 时序建模**：gap 是否存在可利用的时序模式（趋势、加速度等）？
4. **混合路由策略**：RT 与 AD 的按任务类型路由能带来多大收益？
5. **变化检测触发**：VLM-based 变化检测是否是比 yes-bias 更好的触发方式？

---

## 2. 实验设计与结果

### 2.1 CLIP Filter 可行性分析（53 cases）

**目标**：评估 CLIP 帧间相似度变化能否作为 VLM 调用的预过滤器。不是用 CLIP 作特征，而是在 VLM 推理前先用轻量级 CLIP 检测视觉变化，仅在变化显著时调用 VLM。

**方法**：对 53 个 case 的每帧计算 CLIP embedding 余弦距离变化量（clip_change），分析 GT 窗口与非 GT 帧的分布差异。

**核心结果**：

| 指标 | 值 |
|------|-----|
| 全帧平均 clip_change | 0.0546 +/- 0.0241 |
| GT 窗口平均 | 0.0600 +/- 0.0316 |
| 非 GT 帧平均 | 0.0550 +/- 0.0244 |
| GT 前 2s 平均 | 0.0635 +/- 0.0315 |
| **GT/non-GT ratio** | **1.09x** |

**阈值扫描**：

| 阈值 | GT 窗口 Recall | 过滤率 | FP 帧 | TP 帧 |
|------|---------------|--------|-------|-------|
| 0.010 | 93.7% | 11.3% | 26,181 | 1,889 |
| 0.030 | 86.0% | 34.4% | 19,392 | 1,392 |
| **0.050** | **81.7%** | **55.6%** | **13,072** | **982** |
| 0.100 | 47.1% | 85.9% | 4,170 | 303 |
| 0.200 | 7.8% | 99.2% | 248 | 16 |

**按任务类型分析**（theta=0.05）：

- CLIP 最有效：Object Localization（recall=90%, filter=67%）、Text-Rich（recall=92.5%, filter=70.1%）
- CLIP 最弱：Object Function（recall=100%, filter=33.7% — 过滤率低）、Action Recognition（recall=83.9%, filter=52.2%）
- Task Understanding：recall=0%，因为该类型的 GT 窗口几乎没有视觉变化

**结论**：

CLIP 作为判别器极弱（GT/non-GT ratio 仅 1.09x），无法替代 logprob gap。但作为粗筛有限用途：在 theta=0.05 下可过滤 55.6% 的帧（减少一半的 VLM 调用），代价是损失 18.3% 的 GT recall。对于计算受限场景是一个可选的工程优化。

### 2.2 GPT-4o FBF Yes-Bias 测试（5 cases）

**目标**：验证 yes-bias 是否为 Qwen3-VL 特有问题，还是 VLM 在 FBF（逐帧判断）触发范式中的共性问题。

**方法**：使用 Azure OpenAI GPT-4o（`http://52.151.57.21:9999`）对 5 个 case 执行与 Qwen3-VL 相同的 FBF 触发流程。

**核心结果**：

| 指标 | GPT-4o | Qwen3-VL（参考） |
|------|--------|----------------|
| **Yes-rate** | **77.6%** | **76-83%** |
| 总 polling 步数 | 228 | — |
| 总 yes 触发 | 177 | — |
| 平均 API 时间 | 3.3s/步 | ~5s/步 |

GPT-4o 同样存在严重的 yes-bias（77.6% > 60% 阈值），确认 yes-bias 是 FBF 范式的结构性问题，而非特定模型的缺陷。

**Logprob Gap 分析（修正后）**：

原始报告中 gap 统计量为 NaN，经调试发现仅 3/228 步异常（1.3%）：1 步 API rate_limit 错误导致 gap=NaN，2 步因 yes/no token 不在 top-5 logprobs 导致 gap=±inf。过滤这 3 个异常值后：

| 指标 | GPT-4o（修正） | Qwen3-VL（参考） |
|------|--------------|----------------|
| GT 窗口 gap mean | **8.705** | 2.226 |
| 非 GT 窗口 gap mean | **1.902** | 0.808 |
| **Separation** | **+6.802** | +1.418 |
| Welch t-test p值 | <0.0001 | <0.0001 |
| GPT-4o / Qwen 比值 | **4.8x** | — |

GPT-4o 的 gap 判别力是 Qwen3-VL 的 4.8 倍，且 gap 绝对值更大（median=5.750 vs Qwen ~1.5）。这意味着 GPT-4o + Adaptive-D 可能显著超越当前最佳结果。

详细修正报告：`results/gpt4o_fbf/gpt4o_gap_reanalysis_report.md`

**ESTP-F1 性能**（GPT-4o vanilla FBF）：

| 任务类型 | F1 | FP |
|---------|-----|-----|
| Text-Rich Understanding | **0.778** | 3 |
| Task Understanding | 0.235 | 12 |
| Object Function | 0.242 | 24 |
| Information Function | 0.158 | 32 |
| Action Reasoning | 0.000 | 12 |
| **平均** | **0.283** | 16.6 |

GPT-4o 的 vanilla FBF F1=0.283 远高于 Qwen3-VL 的 0.127，尤其在 Text-Rich Understanding 上（0.778 vs ~0.265）。这表明更强的 VLM 可以部分缓解 yes-bias 的影响——即使都说 YES，说 YES 的时机更准确。

**离线阈值扫描（GPT-4o）**最佳配置为 tau=0.0+cd=12，F1=0.357。

### 2.3 Gap 时序建模分析（60 cases）

**目标**：探索 logprob gap 是否存在可利用的时序模式。如果 gap 在 GT 事件前有系统性的上升趋势或突变，时序特征可能提供比原始单帧 gap 更好的触发信号。

**方法**：在 60 个 fullscale_d checkpoint case 上评估 17 种时序特征：

- 基线：raw_gap（单帧 logprob gap）
- 滑动窗口：gap_mean_w3/w5, gap_max_w3/w5, gap_std_w3/w5
- 时序导数：gap_slope_w3/w5, gap_acceleration, gap_jump, gap_jump_abs
- 统计变换：gap_zscore, gap_percentile, gap_cusum
- 交叉特征：gap_x_slope, gap_x_jump

**核心排名**：

| 排名 | 特征 | AUC | Best F1 | Separation |
|------|------|-----|---------|------------|
| 1 | **raw_gap** | **0.615** | **0.252** | **1.418** |
| 2 | gap_max_w3 | 0.599 | 0.246 | 1.240 |
| 3 | gap_max_w5 | 0.592 | 0.243 | 1.167 |
| 4 | gap_percentile | 0.525 | 0.238 | 0.021 |
| 5 | gap_mean_w5 | 0.595 | 0.233 | 1.183 |

**按任务类型对比**（raw_gap vs 最佳时序特征 gap_max_w3）：

- 改善：Information Function (+0.044), Object State Change (+0.026), Ego Object State Change (+0.010)
- 下降：Object Function (-0.047), Attribute Perception (-0.028), Ego Object Localization (-0.021)
- 不变：Task Understanding (0.000), Text-Rich Understanding (0.000)

**结论**：

**没有任何时序特征能显著超越原始单帧 gap**。raw_gap 在 AUC (0.615) 和 Best F1 (0.252) 上均为最优。这说明：

1. **Gap 是瞬时信号**：logprob gap 反映的是 VLM 在当前帧上的即时判断置信度，不存在可预测的上升趋势或时序模式
2. **时序平滑有害**：滑动窗口特征（mean, max）引入了延迟，模糊了触发时机
3. **变化检测类特征无效**：gap_slope, gap_jump 等时序导数的 separation 极低（<0.2），说明 gap 在 GT 前后没有系统性的突变

### 2.4 混合路由策略评估（60 cases）

**目标**：将 Reasoning Trigger (RT) 和 Adaptive-D (AD) 按任务类型路由，利用各方法的互补性。

**方法**：

- **Strategy A（Type-Optimal AD）**：根据 Phase II 每种任务类型的最优 tau，使用 Adaptive-D 触发
- **Strategy B（Mixed Routing）**：Task Understanding 用 BL（旁路），Text-Rich Understanding 用 RT，其余用 AD with type-optimal tau
- **Strategy C（Oracle per-case）**：每个 case 选择 RT/AD/BL 中最优者（上界参考）

**整体结果（60 cases）**：

| 策略 | avg F1 | 描述 |
|------|--------|------|
| Baseline | 0.203 | gap>0, cd=12 |
| **A: Type-Optimal AD** | **0.237** | Per-type best tau |
| **B: Mixed Routing** | **0.245** | RT/BL/AD 路由 |

Delta B vs A: **+0.009** | Delta B vs BL: **+0.042**

**RT 子集对比（15 cases）**：

| 策略 | avg F1 |
|------|--------|
| Baseline | 0.225 |
| A: Type-Optimal AD | 0.298 |
| B: Mixed Routing | 0.332 |
| C: Oracle per-case | 0.361 |

**按任务类型分析**：

Mixed Routing (B) 相比 Type-Optimal AD (A) 的改进**几乎完全来自一个任务类型**：

| 任务类型 | A avg | B avg | B-A delta | B method |
|---------|-------|-------|-----------|----------|
| Text-Rich Understanding | 0.288 | **0.390** | **+0.103** | RT |
| 其余 11 种 | — | — | +0.000 | AD/BL |

实际上只有 **1 个 case** 因路由改善了（ec8c429db0d8, Text-Rich, A F1=0.154 → B F1=0.667），其余 59 个 case 不变。

**Oracle 分析**：15 个 RT case 中 AD 赢 12 个，RT 仅赢 3 个（均为 Text-Rich 或 Attribute 类型），证实 RT 的优势范围极窄。

**结论**：

混合路由的收益有限（+0.009 F1），受限于 RT 仅在 Text-Rich Understanding 上有明确优势。Oracle 上界（0.361）与 Mixed Routing（0.332）的差距（0.028）表明，即使完美路由也只能带来有限的额外收益。核心瓶颈不在路由策略，而在于 RT 自身在大多数任务类型上不如 AD。

### 2.5 变化检测触发实验（15 cases，已完成）

**目标**：使用 VLM-based 变化检测（比较当前帧与 10s 前的帧的视觉差异）替代简单的 yes/no 判断，从根本上解决 yes-bias 问题。

**方法**：向 Gemini-3-Flash 发送两帧图像（当前帧 + 10s 前的帧），提问"Has anything SIGNIFICANT changed between these frames that is directly relevant to the user's question?"，仅在检测到相关变化时触发。使用 10s polling 间隔和 12s cooldown。

**整体结果**：

| 指标 | 变化检测 (CD) | Baseline (BL) | Reasoning Trigger (RT) |
|------|-------------|--------------|----------------------|
| **Avg F1** | **0.121** | 0.106 | 0.173 |
| Yes-rate | **21.0%** | — | 29.6% |
| 总 Triggers | 71 | — | — |
| Delta vs BL | +0.014 | — | — |
| Delta vs RT | **-0.052** | — | — |

**按任务类型分析**：

| 任务类型 | n | CD F1 | BL F1 | Delta |
|---------|---|-------|-------|-------|
| **Text-Rich Understanding** | 1 | **0.500** | 0.000 | **+0.500** |
| **Object Recognition** | 3 | **0.210** | 0.059 | **+0.151** |
| Object Function | 3 | 0.092 | 0.074 | +0.018 |
| Action Recognition | 1 | 0.154 | 0.235 | -0.081 |
| Attribute Perception | 2 | 0.125 | 0.250 | -0.125 |
| Object State Change | 2 | 0.000 | 0.184 | -0.184 |
| Ego Object State Change | 2 | 0.000 | 0.000 | 0.000 |
| Object Localization | 1 | 0.000 | 0.095 | -0.095 |

**关键发现**：

1. **Yes-bias 有效抑制**：yes-rate 从 FBF 的 76-83% 大幅降至 21.0%，变化检测范式确实解决了 yes-bias 的结构性问题
2. **但整体 F1 低于 RT**：CD F1=0.121 < RT F1=0.173，过度抑制导致漏报增多
3. **Text-Rich 和 Object Recognition 是优势领域**：视觉变化明确的任务类型受益显著（+0.500, +0.151）
4. **State Change 类型完全失效**：Object State Change CD F1=0.000（BL F1=0.184），VLM 无法通过两帧对比检测状态变化
5. **变化检测与 yes-bias 的权衡**：降低 yes-rate 减少了 FP，但也大幅降低了 recall，net 效果取决于 GT 窗口的视觉显著性

### 2.6 失败案例深度分析（15 cases）

**目标**：深入分析 Reasoning Trigger、Adaptive-D 和 Baseline 在 15 个 case 上的逐案例表现，识别失败模式。

**方法**：对 15 个 RT case 进行逐 case 的触发行为分析，包括触发时间、yes-rate、gap separation、GT 窗口特征等。

#### 2.6.1 失败模式分类

| 分类 | 涉及任务类型 | RT avg | AD avg | BL avg | 根因 |
|------|------------|--------|--------|--------|------|
| **A: Logprob 有效, RT 失败** | Ego Object State Change, Object Localization | 0.000 | 0.172 | 0.048 | RT 完全沉默或触发延迟 |
| **B: RT 有效, Logprob 失败** | Attribute Perception (部分), Text-Rich | 0.482 | 0.177 | 0.125 | Gap 弱但 VLM 视觉判断好 |
| **C: 两者都失败** | Action Recognition | 0.125 | 0.000 | 0.235 | 极短 GT + 瞬间动作 |
| **D: 两者都有效** | Object Recognition, Object State Change, Object Function | 0.162 | 0.162 | 0.072 | 场景持续相关 |

Winner 分布：AD=5, BL=4, RT=3, TIE=1, ALL_FAIL=2

#### 2.6.2 关键发现

**发现 1：RT yes_rate 与性能的 U 形关系**

| yes_rate 范围 | Cases | avg RT F1 | 特征 |
|--------------|-------|-----------|------|
| 0-5% | 5 | 0.233 | 精准但低覆盖 |
| 5-35% | 4 | 0.072 | 中等，多为 FP |
| 35-60% | 4 | 0.121 | FP 主导 |
| >60% | 2 | 0.222 | 极端但场景高度相关 |

RT 在极低 yes_rate (<5%) 时最精准（VLM 只在最确定时触发），但覆盖率受限。

**发现 2：GT 窗口时长是决定性因素**

| GT 窗口平均时长 | RT 表现 | AD 表现 |
|---------------|---------|---------|
| < 2s（瞬间动作） | 差 | 差 |
| 2-10s（短事件） | 中等 | 中等 |
| > 10s（持续场景） | 好 | 好 |

典型：State Change Case 8（主窗口 35.5s）→ 两者 F1=0.500；Action Case 11（8 个 0.5s 窗口）→ 两者 F1<=0.125。

**发现 3：Logprob gap 绝对值 vs 分离度的脱节**

| Case | Gap separation | GT avg gap | tau=2.8 能否触发 | AD F1 |
|------|---------------|-----------|-----------------|-------|
| a14e (Object Func) | +3.515 | 0.312 | 否 | 0.000 |
| 2098 (Action Recog) | +2.693 | 1.831 | 否 | 0.000 |
| f732 (Object Recog) | +0.671 | 5.154 | 是 | 0.211 |

**分离度高不等于可触发**。分离度衡量 GT 与非 GT 的相对差异，但如果绝对值都低于阈值，高分离度也无用。这正是 Phase II 自适应 tau 的核心价值。

**发现 4：RT 最典型成功案例**

Case ec8c429db0d8（Text-Rich Understanding）：

- 问题："What specific text is written on the sticky note when I look at it?"
- RT：F1=**0.667**，1 次精准触发（yes_rate=3.1%），命中 GT 窗口
- AD：F1=0.154，11 次触发，大量 FP
- 原因：文字可见性是最直接的视觉信号，VLM 精确判断"此刻能看到文字内容"

**发现 5：RT FP 来源**

15 个 case 共 80 个 FP：

| FP 原因 | 数量（估计） | 典型 case |
|---------|------------|----------|
| 场景相关但时间偏差 | ~35 | Case 1, 5, 7 |
| VLM 过度触发（yes-bias） | ~25 | Case 10, 14 |
| 抽象场景误判 | ~12 | Case 15 |
| GT 后持续触发 | ~8 | Case 4 (AD) |

---

## 3. 关键发现总结

### 3.1 Gap 是瞬时信号，时序建模无益

17 种时序特征无一能超越原始单帧 gap（raw_gap AUC=0.615, Best F1=0.252）。滑动窗口、时序导数、统计变换等均未能提供额外信息。logprob gap 反映的是 VLM 在当前帧上的即时判断，不存在可预测的时序模式。

**启示**：不应继续投入时序特征工程，应聚焦于改善单帧信号质量或引入外部信号。

### 3.2 CLIP 作为判别器极弱，作为粗筛有限用途

CLIP 帧间变化的 GT/non-GT ratio 仅 1.09x，几乎没有判别力。但作为工程优化，theta=0.05 可过滤 55.6% 的帧（减半 VLM 调用），代价是 18.3% 的 GT recall 损失。

**启示**：CLIP 不能替代 logprob gap 作为触发信号，但在推理预算受限时可作为预过滤器使用。

### 3.3 GPT-4o 同样存在 yes-bias，但 gap 判别力远超 Qwen

GPT-4o yes-rate=77.6%（与 Qwen3-VL 76-83% 持平），确认 yes-bias 是 FBF 范式的结构性问题。但 GPT-4o 的 vanilla FBF F1=0.283 远高于 Qwen3-VL 的 0.127，特别是 Text-Rich（0.778 vs ~0.265）。

**关键发现：GPT-4o logprob gap 修正后 separation=+6.802，是 Qwen 的 4.8 倍**。原 NaN 问题仅涉及 3/228 步（1.3%），由 API rate limit 和极端确信度导致。修正后 GPT-4o 在所有有效 case 上都展现出强判别力（GT gap mean=8.705 vs non-GT=1.902）。

**启示**：
- Yes-bias 是 VLM + FBF 的范式瓶颈，更换模型不能解决根本问题
- 更强的 VLM 在两个维度都更好：（1）触发时机更准（F1=0.283 vs 0.127）；（2）gap 分离度更大（+6.802 vs +1.418）
- **GPT-4o + Adaptive-D 是最有前景的短期方向**，预期可大幅超越当前最佳 F1=0.222

### 3.4 混合路由策略收益有限（+0.009 F1）

Mixed Routing 相比 Type-Optimal AD 仅提升 0.009 F1，且改进完全来自 Text-Rich Understanding 的 1 个 case。Oracle 分析显示 15 个 case 中 AD 赢 12 个，RT 仅赢 3 个。

**启示**：RT 的优势范围极窄（仅 Text-Rich），混合路由的投入产出比低。除非 RT 在更多任务类型上被证明有效，否则 Adaptive-D 是更可靠的主策略。

### 3.5 失败模式分类揭示了方法互补性的边界

四类失败模式（A-D）清晰界定了各方法的适用范围：

- **A 类（Logprob 有效, RT 失败）**：视觉信号不显著的隐式变化（Ego Object State Change, Object Localization） → AD 依赖隐式信号（logprob 波动），RT 需要显式视觉判据
- **B 类（RT 有效, Logprob 失败）**：明确视觉事件（Text-Rich, 部分 Attribute） → VLM 可直接判断，无需 logprob 中间信号
- **C 类（两者都失败）**：极短 GT + 瞬间动作（Action Recognition） → 结构性问题，需要高频采样或专门动作检测
- **D 类（两者都有效）**：持续可见场景 → 两种方法都能利用

### 3.6 变化检测有效抑制 yes-bias 但整体不如 RT

变化检测（15 cases）将 yes-rate 从 76-83% 降至 21.0%，成功解决了 yes-bias 的结构性问题。但 avg F1=0.121 低于 RT（0.173）和 Adaptive-D（0.222），原因是过度抑制导致 recall 大幅下降。变化检测在 Text-Rich（+0.500）和 Object Recognition（+0.151）上有显著优势，但在 State Change（-0.184）和 Attribute Perception（-0.125）上反而退步。

**启示**：变化检测方向是正确的（证明了 yes-bias 可以被解决），但当前 prompt 对"变化"的定义过于严格。后续可以尝试：(1) 放松变化检测标准；(2) 将变化检测与 logprob gap 结合（变化检测做粗筛，gap 做精排）。

---

## 4. 各实验间的联系与整体洞察

### 4.1 信号层级

本轮实验揭示了三个层级的触发信号及其特点：

| 层级 | 信号 | 优势 | 劣势 | 代表实验 |
|------|------|------|------|---------|
| **底层** | CLIP 视觉变化 | 快速、无需 VLM | 判别力极弱 (1.09x) | 2.1 |
| **中层** | Logprob gap | 隐式捕获 VLM 不确定性 | 瞬时信号，受阈值敏感 | 2.3 |
| **顶层** | VLM 显式推理 | 可利用语义理解 | yes-bias 严重 (77-83%) | 2.2, 2.5 |

Adaptive-D 在中层信号上做到了最优（Phase II F1=0.222），但受限于 gap 作为触发信号的固有上限。顶层的 VLM 推理（RT 和变化检测）在特定场景下可突破这一上限（Text-Rich F1=0.667），但泛化性不足。

### 4.2 瓶颈分析

| 瓶颈 | 影响 | 当前最佳对策 | 上限 |
|------|------|------------|------|
| Yes-bias | FBF 范式下所有 VLM 都趋向说 YES | Logprob gap 过滤 | 受限于 gap 判别力 |
| 短 GT 窗口 | <2s 的瞬间动作难以命中 | 低阈值 (tau=0.5) | 采样率瓶颈 (5.7s/步) |
| 任务类型异质性 | 12 种类型的最优策略差异极大 | 自适应 tau 路由 | 需要任务类型先验 |
| 视觉信号模糊性 | 抽象/功能性问题缺乏明确视觉线索 | — | 可能需要多模态上下文 |

---

## 5. 下一步建议

### 5.1 短期可行方向

**5.1.1 GPT-4o Adaptive-D 全量评估**（优先级最高）

GPT-4o logprob gap NaN 问题已修复（仅 3/228 步异常），修正后 separation=+6.802（Qwen 的 4.8 倍）。结合 FBF F1=0.283 的基础性能优势，GPT-4o + Adaptive-D 极可能大幅超越当前最佳 F1=0.222。建议在 60 cases 全量上评估 GPT-4o FBF + Adaptive-D。

**5.1.2 完成变化检测实验**

当前 8/15 cases（avg F1=0.141），等待完整 15 case 结果。若最终 F1 超过 0.173（RT 基线），则变化检测方向值得深入。

**5.1.3 CLIP 预过滤工程化**

在计算受限场景下，CLIP theta=0.05 可减少约 50% VLM 调用，代价为 18% recall 损失。可作为部署优化方案，与 Adaptive-D 组合使用。

### 5.2 中期探索方向

**5.2.1 提升采样率应对短 GT 窗口**

当前 0.175Hz (~5.7s/步) 对 <2s 的瞬间动作类 GT 窗口命中率极低。考虑：
- 对 Action Recognition 等瞬间动作类型使用 2-3s 采样间隔 + tau=0.5
- 基于 CLIP 变化的自适应采样率（变化大时增加频率）

**5.2.2 改进 RT Prompt 设计**

当前 RT 的 yes-bias 主要来自过于宽泛的判断标准。改进方向：
- 针对动作类型：增加"与上一帧相比用户完成了什么新动作"的时序对比
- 增加显式否定条件：降低 VLM 说 YES 的倾向
- 使用变化检测范式替代直接 yes/no 判断

**5.2.3 多模型集成**

GPT-4o FBF F1=0.283 > Qwen3-VL FBF F1=0.127，差距显著。考虑：
- 用 Qwen3-VL 做高频低成本的 logprob gap 预筛
- 用 GPT-4o 做低频高精度的最终确认
- 两阶段级联可平衡成本和精度

### 5.3 长期研究方向

**5.3.1 任务自适应触发框架**

将 Phase II 的自适应 tau 扩展为完整的任务自适应框架：
- 任务类型自动识别（基于问题文本的分类器）
- 触发策略自适应选择（AD vs RT vs BL）
- 采样率自适应调整
- 预期 F1 上界：Oracle per-case = 0.361（15 cases）

**5.3.2 超越 FBF 范式**

FBF（逐帧二元判断）的 yes-bias 是结构性问题。替代范式：
- **事件驱动触发**：检测视觉事件（物体出现/消失、动作完成、文字出现）而非逐帧投票
- **连续得分触发**：VLM 输出连续相关度分数而非二元判断，用时序阈值检测峰值
- **上下文窗口触发**：给 VLM 多帧历史上下文，利用跨帧对比而非单帧判断

---

## 附录：实验配置与数据来源

| 实验 | 数据源 | Case 数 | GPU 需求 | 报告路径 |
|------|--------|--------|---------|---------|
| CLIP Filter | fullscale_d checkpoint | 53 | 无 | `results/clip_filter_analysis_report.txt` |
| GPT-4o FBF | 5 case pilot | 5 | 无（API） | `results/gpt4o_fbf/gpt4o_fbf_report.txt` |
| Gap 时序建模 | fullscale_d checkpoint | 60 | 无 | `results/temporal_gap_analysis_report.md` |
| 混合策略 | fullscale_d + RT checkpoint | 60 | 无 | `results/mixed_strategy_report.md` |
| 变化检测 | RT 15 cases subset | 15 | Gemini API | `results/phase3_learned/change_detection_trigger_pilot.json` |
| GPT-4o Gap 重分析 | GPT-4o checkpoint | 5 | 无 | `results/gpt4o_fbf/gpt4o_gap_reanalysis_report.md` |
| 失败案例分析 | RT 15 cases + AD 60 cases | 15 | 无 | `results/failure_case_deep_analysis.md` |
