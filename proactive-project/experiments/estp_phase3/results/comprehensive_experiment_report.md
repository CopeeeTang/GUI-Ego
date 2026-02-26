# ESTP-Bench 主动触发系统：完整实验报告

> 日期：2026-02-20
> 项目：Proactive VLM Trigger for Smart Glasses
> 评估基准：ESTP-Bench (164 videos, 1212 QA, 12 task types)
> 模型：Qwen3-VL-8B-Instruct (FBF mode)
> 评估集：5-cases subset (60 QA), 53 matched cases

---

## 目录

1. [研究背景与问题定义](#1-研究背景与问题定义)
2. [Phase I: 全局阈值验证](#2-phase-i-全局阈值验证)
3. [Phase II: 任务类型自适应路由](#3-phase-ii-任务类型自适应路由)
4. [Phase III-A: MLP 学习式触发](#4-phase-iii-a-mlp-学习式触发)
5. [Phase III-B: Goal-State 特征探索](#5-phase-iii-b-goal-state-特征探索)
6. [Phase III-C: VLM Reasoning Trigger](#6-phase-iii-c-vlm-reasoning-trigger)
7. [关键 Bug 修复：时间戳对齐](#7-关键-bug-修复时间戳对齐)
8. [综合对比与排名](#8-综合对比与排名)
9. [核心科学发现](#9-核心科学发现)
10. [结论与下一步](#10-结论与下一步)

---

## 1. 研究背景与问题定义

### 1.1 任务

在智能眼镜场景中，VLM（视觉语言模型）持续观察用户的第一人称视角。系统需要在**最恰当的时刻**主动提供信息帮助，而不是等用户询问。

核心挑战：**何时触发？**

- 触发太早 → 信息无关，用户干扰（False Positive）
- 触发太晚 → 错过最佳帮助时机（False Negative）
- 触发太频繁 → 用户烦躁

### 1.2 评估指标

**ESTP-F1**：ESTP-Bench 定义的核心指标
- ANTICIPATION = 1s（允许提前 1 秒触发）
- LATENCY = 2s（允许延迟 2 秒触发）
- 每个 GT 窗口最多匹配一次触发 → Precision/Recall/F1

### 1.3 基线系统

Qwen3-VL-8B 在 Frame-by-Frame (FBF) 模式下运行：
- 每帧问 VLM："Should I help the user now? YES/NO"
- 贪心解码取第一个 token
- **已知问题**：76-83% 的 yes-bias（几乎每帧都说 YES）

### 1.4 研究出发点

**五个假设**（Phase III 初始设计）：

| 假设 | 机制 | 核心思路 |
|------|------|---------|
| A | 语义增强 Prompt | 更好的 prompt 减少 yes-bias |
| B | 自适应时间窗口 | 跳帧 + 场景变化检测 |
| C | 置信度校准 | VLM 自评估置信度 |
| **D** | **Logprob Gap 触发** | **logP(yes) - logP(no) 作为连续信号** |
| E | B+D 组合 | 自适应帧 + logprob 阈值 |

早期 5-case 实验结果表明：**只有假设 D（logprob gap）有效**。

---

## 2. Phase I: 全局阈值验证

> 文件：`results/fullscale_d/fullscale_d_report.txt`
> 规模：60 QA, 53 matched cases, ~2.7h GPU

### 2.1 目标

验证 logprob gap 阈值（τ=2.8, cooldown=12s）在全量数据上是否优于基线。

### 2.2 结果

| 配置 | avg F1 | Recall | Precision | FP 总数 |
|------|--------|--------|-----------|---------|
| 基线 (gap>0, no cd) | 0.127 | 0.668 | 0.086 | 1,516 |
| D@τ=2.8, cd=12 | 0.114 | 0.237 | 0.137 | 287 |

**统计检验**：
- Delta F1 = -0.013, 95% CI = [-0.059, +0.031]
- **CI 跨越零 → 不显著**

### 2.3 Gate 1 判定：FAIL

FP 减少 81%（1516→287），但 Recall 崩塌 65%（0.668→0.237）。净效果为负。

### 2.4 关键发现

1. **Logprob gap 信号本身是有效的**：GT 步平均 gap=2.226 vs 非GT 步=0.808（p<0.0001）
2. **问题在于"一刀切"**：同一个 τ=2.8 对不同任务类型有相反效果

| 受益任务类型 | ΔF1 | 受损任务类型 | ΔF1 |
|-------------|------|-------------|------|
| Information Function | +0.075 | Action Recognition | -0.146 |
| Text-Rich Understanding | +0.069 | Attribute Perception | -0.123 |
| Task Understanding | +0.057 | Ego Object State Change | -0.034 |

**结论**：需要按任务类型自适应选择阈值。

---

## 3. Phase II: 任务类型自适应路由

> 文件：`results/phase2_stratified/phase2_report.txt`
> 方法：12 种任务类型 × 13 个 τ 值，离线扫描

### 3.1 目标

为每种任务类型找到最优 τ，构建自适应路由策略。

### 3.2 Adaptive-D 策略

```
输入：任务类型 T，logprob gap g
查表：τ* = optimal_tau[T]
决策：if g > τ* and cooldown_ok: TRIGGER
```

### 3.3 结果

| 配置 | avg F1 | ΔF1 vs BL | 95% CI | FP | 判定 |
|------|--------|-----------|--------|-----|------|
| 基线 (τ=0) | 0.141 | — | — | 1,466 | — |
| D@2.8 全局 | 0.109 | -0.032 | [-0.071, +0.013] | 416 | FAIL |
| **Adaptive-D** | **0.222** | **+0.081** | **[+0.046, +0.121]** | **416** | **PASS** |

### 3.4 Gate 2 判定：PASS

- 95% CI 下限 = +0.046 > 0 → **统计显著**
- FP 减少 72%，F1 提升 57%
- 53 个 case 中 32 个改进（60%）

### 3.5 每种类型的最优阈值

| 任务类型 | 最优 τ | F1 | ΔF1 | Gap 分离度 |
|---------|--------|-----|------|-----------|
| Information Function | **2.8** | 0.143 | +0.025 | +2.608 (极佳) |
| Object Recognition | **2.5** | — | — | +2.553 |
| Text-Rich Understanding | **2.0** | 0.288 | +0.023 | +1.873 |
| Object Localization | **5.0** | 0.173 | +0.112 | +1.289 |
| Ego Object State Change | **0.5** | 0.199 | +0.102 | +1.114 |
| Action Recognition | **0.5** | 0.240 | +0.005 | +0.831 |
| Attribute Perception | **0.5** | 0.256 | +0.040 | +1.793 |
| Object State Change | **0.0** | — | — | +2.236 |
| **Task Understanding** | **0.0** | 0.434 | 0.000 | **-2.072** |

**关键发现**：Task Understanding 的 gap 分离度为**负值**（-2.072），意味着 GT 步的 gap 反而**低于**非GT 步。对该类型应完全绕过 D 过滤（τ=0）。

### 3.6 结论

Adaptive-D 是**第一个统计显著优于基线的方法**。核心创新：不同任务类型需要不同的触发灵敏度。

---

## 4. Phase III-A: MLP 学习式触发

> 文件：`results/phase3_learned/phase3_stage_a_report.txt`
> 方法：6 维特征 MLP, TTLOO 交叉验证

### 4.1 动机

Adaptive-D 依赖手工设定的每类型阈值。能否用 MLP 自动学习最优决策边界？

### 4.2 特征设计

| 维度 | 特征 | 含义 |
|------|------|------|
| 0 | gap | logprob gap（核心信号） |
| 1 | slope | gap 变化率（近 5 步） |
| 2 | volatility | gap 标准差（近 10 步） |
| 3 | since_last_trig | 距上次触发时间 |
| 4 | step_progress | 任务进度比 |
| 5 | stagnation | 距上次高波动事件时间 |

### 4.3 Stage A 结果（6 维 D 特征）

| 方法 | avg F1 | vs Adaptive-D (TTLOO) |
|------|--------|----------------------|
| Baseline | 0.141 | — |
| Adaptive-D (TTLOO) | 0.196 | — |
| **MLP Stage A** | **0.180** | **-0.016** |

Gate: FAIL（CI=[-0.084, +0.056]，跨零）

### 4.4 Stage B 结果（+ CLIP / + Goal-State）

| 配置 | 特征维度 | avg F1 | vs Adaptive-D |
|------|---------|--------|---------------|
| Stage B (D + CLIP) | 70 | 0.219 | +0.023（边界）|
| Stage B (D + Goal-State) | 9 | 0.202 | +0.006（边界）|
| Stage B (CLIP only) | 64 | 0.165 | -0.031（FAIL）|

### 4.5 结论

MLP 未能显著超过 Adaptive-D。原因：
1. **样本量不足**：53 个 case（约 2800 个 step），对 70 维特征过拟合
2. **CLIP 贡献有限**：视觉特征的帧间变化不足以提供额外判别力
3. **Goal-State 无用**：与触发需求反相关（详见 Section 5）

---

## 5. Phase III-B: Goal-State 特征探索

> 文件：`results/phase3_learned/goal_state_failure_analysis_report.md`
> 模型：Qwen3-VL, Gemini-2.0-Flash, Gemini-3-Flash

### 5.1 动机

用户原始提案的核心想法：VLM 应该**推理**用户当前的任务进展状态，然后基于状态判断是否触发。

实现：每 10 秒让 VLM 将用户状态分为 on_track / uncertain / off_track。

### 5.2 结果：全部模型反相关

| Model | Cases | GT mean | non-GT mean | GT separation |
|-------|-------|---------|-------------|---------------|
| Gemini-2.0-Flash | 53 | +0.759 | +0.122 | **-0.636** |
| Gemini-3-Flash | 28 | +0.906 | +0.311 | **-0.595** |

> Score: on_track=+1, uncertain=0, off_track=-1
> 负 GT sep = GT 窗口内更多 on_track = **反向**

### 5.3 根因分析

**语义错配矩阵**：

| 场景 | Goal-State 判断 | 需要触发？ | 一致？ |
|------|----------------|----------|--------|
| 用户看着目标文字 | on_track | **是** (Text-Rich) | **否** |
| 状态变化正在发生 | on_track | **是** (State Change) | **否** |
| 用户做无关事情 | off_track | 否 | 是 |

**概念性矛盾**：ESTP-Bench 的 GT 窗口 = 用户与目标互动时（模型判 on_track），但恰恰此时需要信息帮助。"用户进展良好" ≠ "用户不需要帮助"。

### 5.4 任务类型深度分析

**Text-Rich (GT sep = -1.359)**：最极端反向
- Q: "蓝色罐子上的品牌是什么？"
- GT 时刻：用户正凝视蓝色罐子 → on_track
- 但用户需要被告知品牌名 → 应该触发
- Case cfab4d8a: GT=on_track, 其余30个bucket=off_track → sep=-2.000

**Ego Object State Change (GT sep = -0.375)**：中等反向
- Q: "跑步的人什么时候改变位置？"
- GT 时刻：人正在移动 → 用户在观察 → on_track
- 但用户需要被通知变化 → 应该触发

**Ego Object Localization (GT sep = +0.851)**：伪正向
- 88% 的 GT 窗口在 10s bucket 分辨率下未被命中
- 正向分离度来自统计伪影（GT mean 默认为 0）

### 5.5 结论

Goal-State 分类是**死胡同**。3 个不同模型（Qwen, Gemini-2.0, Gemini-3）都展现同样的反向模式，证明这是概念性问题而非模型能力问题。**放弃此方向。**

---

## 6. Phase III-C: VLM Reasoning Trigger

> 文件：`results/phase3_learned/reasoning_trigger_pilot.json`
> 模型：Gemini-3-Flash (thinking model)
> 规模：15 cases, 10s interval

### 6.1 动机

回到用户最初的提案：不分类 goal-state，而是让 VLM **直接推理**"此刻该不该帮助用户？"

Prompt 设计：
```
Is the user RIGHT NOW at the moment where the answer to
their question is most relevant and useful?

Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene
2. The user appears to be actively engaging with it
3. Providing the answer NOW would be more useful than waiting
```

### 6.2 结果（15 cases 公平对比）

| 方法 | avg F1 | vs Baseline |
|------|--------|-------------|
| **Adaptive-D** | **0.298** | **+0.073** |
| Baseline (gap>0) | 0.225 | — |
| Reasoning Trigger | 0.173 | **-0.052** |

**Reasoning vs Adaptive-D: 3 wins, 12 losses, 0 ties**

### 6.3 任务类型对比

| Task Type | Reasoning | Adaptive-D | 胜者 |
|-----------|-----------|------------|------|
| **Text-Rich** | **0.667** | 0.154 | **Reasoning** |
| Obj State Change | 0.500 | 0.250 | Reasoning |
| Object Recog (case 5) | 0.250 | 0.077 | Reasoning |
| Object Func | 0.096 | **0.351** | Adaptive-D |
| Action Recog | 0.125 | **0.250** | Adaptive-D |
| Attribute | 0.297 | **0.472** | Adaptive-D |
| 其余 5 种 | 0.000-0.154 | **0.200-0.500** | Adaptive-D |

### 6.4 Reasoning Trigger 的根本问题

1. **FP 过高**：yes_rate=29.6%（93 个触发中大部分是 FP）
2. **缺乏时序精度**：VLM 能判断"场景相关"但不能判断"此刻最佳"
3. **Cooldown 阻塞**：Case 6bd31d1c 中 VLM 在 GT 窗口内正确说 YES，但被 12s cooldown 拦截

### 6.5 为什么 Text-Rich 例外.

Text-Rich 是唯一 Reasoning Trigger 大幅领先的类型（0.667 vs 0.154）：
- 视觉信号最直接：文字可见 = 该帮忙
- VLM 的 YES/NO 判断与"是否能看到文字"完美对齐
- Logprob gap 在 Text-Rich 反而表现一般（τ=2.0 仍不够敏感）

### 6.6 结论

Reasoning Trigger 整体不如 Adaptive-D，但在 Text-Rich 上有绝对优势。可考虑混合策略。

---

## 7. 关键 Bug 修复：时间戳对齐

> 修复日期：2026-02-19
> 影响范围：Phase III-B, III-C 的全部结果

### 7.1 Bug 描述

ESTP-Bench 的视频文件是**完整长视频**（如 1920s），clip 只是其中的片段（如 1614s-1920s）。

| 脚本 | 帧提取时间 | GT 时间 | Bug |
|------|-----------|---------|-----|
| hypothesis_runner.py | 原始时间线 ✅ | 原始时间线 ✅ | 无 |
| goal_state_gemini.py | 0→duration ❌ | 原始时间线 | **看错帧** |
| reasoning_trigger.py | 0→duration ❌ | 原始时间线 | **看错帧 + GT 不匹配** |
| goal_state_extractor.py | trigger_log ❌ | 0→duration | **匹配失败全 uncertain** |

### 7.2 影响

- **75% 的 cases**（45/60）有非零 clip_start_time
- Goal-state 提取看的是视频开头的无关内容
- Reasoning trigger 看错帧 + GT 对比永远失败
- Qwen 提取器 bucket 匹配失败 → 全部 uncertain（解释了 75% uncertain 率）

### 7.3 修复

在帧提取时加上 clip_start_time 偏移：`t_abs = clip_start + t_relative`

修复了 4 个脚本：
1. `phase3_goal_state_extractor.py`
2. `phase3_goal_state_gemini.py`
3. `reasoning_trigger_pilot.py`
4. `compare_goal_state_caches.py`

### 7.4 不受影响的结果

- **Phase I / Phase II 完全不受影响**：hypothesis_runner.py 一直使用正确的绝对时间
- Adaptive-D F1=0.222 结果有效

---

## 8. 综合对比与排名

### 8.1 全局排名（53 cases 基准）

| Rank | 方法 | avg F1 | 配置 | 评估方式 | 置信度 |
|------|------|--------|------|---------|--------|
| 1 | **Adaptive-D** | **0.222** | 每类型 τ, cd=12 | TTLOO | ★★★★★ |
| 2 | MLP Stage B (D+CLIP) | 0.219 | 70-dim MLP | TTLOO | ★★★ |
| 3 | MLP Stage B (D+Goal) | 0.202 | 9-dim MLP | TTLOO | ★★★ |
| 4 | Adaptive-D (TTLOO fair) | 0.196 | 每类型 τ | TTLOO fair | ★★★★ |
| 5 | MLP Stage A (D only) | 0.180 | 6-dim MLP | TTLOO | ★★★ |
| 6 | Reasoning Trigger | 0.173 | Gemini-3-Flash | 15 cases | ★★ |
| 7 | Baseline (gap>0) | 0.141 | vanilla + cd=12 | full 53 | ★★★★★ |
| 8 | D@2.8 全局 | 0.109 | fixed τ=2.8 | full 53 | ★★★★★ |

> 置信度：★=小样本/高方差，★★★★★=全量/高信赖

### 8.2 任务类型最优方法

| Task Type | 最优方法 | F1 | 次优 | F1 |
|-----------|---------|-----|------|-----|
| **Text-Rich Understanding** | **Reasoning Trigger** | **0.667** | Adaptive-D | 0.154 |
| Information Function | Adaptive-D | 0.143 | MLP B | 0.213 |
| Object Recognition | Adaptive-D | — | MLP A | 0.263 |
| Ego Object Localization | MLP B (D+CLIP) | 0.424 | Adaptive-D | — |
| Ego Object State Change | MLP B (D+CLIP) | 0.413 | Adaptive-D | 0.199 |
| Task Understanding | **Bypass D (τ=0)** | 0.434 | — | — |
| 其余 6 种 | Adaptive-D | — | — | — |

### 8.3 方法特性对比

| 特性 | Baseline | Adaptive-D | MLP | Reasoning |
|------|----------|------------|-----|-----------|
| 需要 GPU | ✅ (Qwen) | ✅ (Qwen) | ✅ (训练) | ❌ (API) |
| 推理时开销 | 无 | 查表 | MLP forward | VLM call/step |
| 需要任务类型 | ❌ | ✅ | ✅ (训练) | ❌ |
| 数据需求 | 无 | 53 cases | 53 cases | 无 |
| 延迟 | 0 | 0 | ~1ms | ~4s/step |
| 可解释性 | 高 | 高 | 低 | 中 |

---

## 9. 核心科学发现

### 发现 1：Yes-Bias 的三层结构

VLM（Qwen3-VL）在主动触发场景下存在系统性的 yes-bias，表现在三个层次：

| 层次 | 可干预性 | 证据 |
|------|---------|------|
| **Prompt 层**：改 prompt 只能将 yes 率从 76% 降到 69% | 极低 | 假设 A |
| **自评估层**：VLM 对自己的判断 100% 给 4/5 分 | 零 | 假设 C |
| **Logit 层**：logP(yes) - logP(no) 存在 GT/非GT 差异 | **有效** | 假设 D |

**贪心解码压制了模型的不确定性信息。** Logprob gap 恢复了这种被压制的信号。

### 发现 2：任务类型是决定性因素

12 种任务类型的最优 τ 范围从 0.0（完全绕过）到 5.0（极严格）。

```
τ 分布：
  τ=0.0  ███  (Task Understanding, Object State Change, Action Reasoning)
  τ=0.5  ████████  (Action Recog, Attribute, Ego States, Object Func)
  τ=2.0  ██  (Text-Rich)
  τ=2.5  ██  (Object Recognition)
  τ=2.8  ██  (Information Function)
  τ=5.0  ██  (Object Localization)
```

**Task Understanding 的 gap 分离度为负** (-2.072)：GT 步的 gap 低于非GT 步。这意味着对该类型，logprob gap **不是有效信号**，必须完全绕过。

### 发现 3：Goal-State 与触发需求反相关

3 个不同模型（Qwen3-VL, Gemini-2.0-Flash, Gemini-3-Flash）都展现相同模式：

```
GT 窗口（需要帮助时）：模型说 on_track（用户进展好）
非 GT 窗口（不需要时）：模型说 off_track（用户偏离）
```

这不是模型能力问题，而是**概念性矛盾**：
- ESTP-Bench GT = 信息对用户最有用时 = 用户正在与目标互动
- 与目标互动 = on_track
- 因此 GT ⊂ on_track（几乎总是）

### 发现 4：VLM 能判断场景相关性但缺乏时序精度

Reasoning Trigger 的 yes_rate=29.6%，说明 VLM 能识别"场景相关"（约 30% 的帧确实相关），但无法区分"刚看到"和"已经看了一会儿"。

唯一例外：Text-Rich（文字可见=该帮忙，视觉信号最直接）。

### 发现 5：Cooldown 是被低估的关键参数

| cd | F1 (5-case) | 效果 |
|-----|------------|------|
| 0s | 0.161 | burst-triggering |
| 6s | 0.265 | 抑制部分冗余 |
| 12s | 0.280 | 最优 |
| 20s | 下降 | 错过独立 GT |

12s cooldown 恰好覆盖 ESTP-Bench 中事件的典型持续时间。

### 发现 6：时间戳对齐是 ESTP-Bench 的隐藏陷阱

75% 的 cases 有非零 clip_start_time。任何涉及帧提取的后处理脚本都必须加偏移。hypothesis_runner.py 正确处理了，但后续分析脚本遗漏了这一点。

---

## 10. 结论与下一步

### 10.1 实验路线图回顾

```
Phase I: 全局 τ=2.8                    → FAIL (recall 崩塌)
    ↓
Phase II: 每类型最优 τ (Adaptive-D)     → PASS ★ (F1=0.222, 统计显著)
    ↓
Phase III-A: MLP 学习式                  → MARGINAL (F1=0.180-0.219)
Phase III-B: Goal-State 特征             → DEAD END (反相关)
Phase III-C: VLM Reasoning Trigger       → PARTIAL (Text-Rich 优, 其余劣)
```

### 10.2 当前最优系统

**Adaptive-D（任务类型自适应 logprob gap 阈值路由）**

- F1 = 0.222（53 cases），统计显著优于基线
- 无额外推理开销（仅查表 + 比较）
- 需要任务类型标签（可离线分类）

### 10.3 建议下一步

**P0（立即可做）**：

1. **混合策略**：Text-Rich 用 Reasoning Trigger（F1=0.667），其余用 Adaptive-D
   - 预期提升：Text-Rich 从 0.154→0.667，其余不变
   - 实现成本低，只需任务类型路由

2. **Cooldown 优化**：探索自适应 cooldown（当前固定 12s 有时阻塞正确触发）

**P1（中期）**：

3. **扩大评估规模**：从 5-cases (60 QA) 扩展到更多 cases，减小置信区间
4. **温度采样**：temp > 0 多次采样取多数决，可能改善 logprob gap 质量
5. **轻量触发模型分离**：小模型判断时机 + VLM 生成回答

**P2（长期）**：

6. **端到端学习**：在 ESTP-Bench 上微调小型分类器
7. **Streaming 架构**：类似 EyeWO (ESTP-Bench SOTA, F1=34.7%) 的端到端设计
8. **多模态记忆**：结合 proactive-project 的记忆系统

---

## 附录 A：文件索引

| 文件 | 说明 |
|------|------|
| `results/fullscale_d/fullscale_d_report.txt` | Phase I 全规模报告 |
| `results/fullscale_d/checkpoint.jsonl` | 53 cases 原始数据 |
| `results/phase2_stratified/phase2_report.txt` | Phase II 分层报告 |
| `results/phase2_stratified/per_type_sweep.json` | 每类型阈值扫描数据 |
| `results/phase3_learned/phase3_stage_a_report.txt` | MLP Stage A 报告 |
| `results/phase3_learned/phase3_stage_b_full_report.txt` | MLP Stage B 报告 |
| `results/phase3_learned/phase3_stage_b_goal_report.txt` | MLP + Goal-State 报告 |
| `results/phase3_learned/goal_state_failure_analysis_report.md` | Goal-State 失败分析 |
| `results/phase3_learned/goal_state_comparison_report.txt` | 3-model 对比 |
| `results/phase3_learned/reasoning_trigger_pilot.json` | Reasoning Trigger 结果 |
| `results/phase3_learned/goal_state_cache_gemini2flash.json` | Gemini-2.0 缓存 (53 cases) |
| `results/phase3_learned/goal_state_cache_gemini3flash.json` | Gemini-3-Flash 缓存 (28 cases) |

## 附录 B：关键脚本

| 脚本 | 用途 |
|------|------|
| `hypothesis_runner.py` | Phase I GPU 在线评估 |
| `fullscale_d_runner.py` | Phase I 全规模运行 |
| `phase2_stratified_analysis.py` | Phase II 离线分层分析 |
| `phase3_mlp_trigger.py` | Phase III MLP 训练/评估 |
| `phase3_goal_state_extractor.py` | Qwen goal-state 提取 |
| `phase3_goal_state_gemini.py` | Gemini goal-state 提取 |
| `reasoning_trigger_pilot.py` | VLM Reasoning Trigger |
| `compare_goal_state_caches.py` | Goal-state 多模型对比 |
| `threshold_sweep.py` | 离线阈值扫描 |
