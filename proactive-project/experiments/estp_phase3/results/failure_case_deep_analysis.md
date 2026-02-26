# 失败 Case 深层分析报告

> 日期：2026-02-20
> 分析范围：15 个 Reasoning Trigger case + 60 个 Adaptive-D checkpoint case
> 方法对比：Reasoning Trigger (Gemini-3-Flash) vs Adaptive-D (Qwen3-VL logprob gap) vs Baseline (gap>0)

---

## 1. 数据概览

### 1.1 Reasoning Trigger 15 Case 汇总

| 指标 | 值 |
|------|-----|
| 平均 F1 | 0.173 |
| 平均 Baseline F1 | 0.107 |
| 平均 Delta F1 | +0.066 |
| 总 TP / FP / FN | 13 / 80 / 53 |
| 总 yes_rate | 9.4 yes/case (29.6%) |
| 覆盖任务类型 | 8/12 种 |

### 1.2 同 15 Case 下 Adaptive-D 对比

| 方法 | avg F1 | 配置 |
|------|--------|------|
| **Adaptive-D** | **0.165** | tau=2.8, cd=12 (全局阈值) |
| Reasoning Trigger | 0.173 | Gemini-3-Flash, 10s interval |
| Baseline | 0.107 | gap>0, cd=12 |

注意：这里的 Adaptive-D 是全局 tau=2.8（Phase I），而非 Phase II 的自适应 tau。Phase II 自适应 Adaptive-D 在 53 case 上 F1=0.222。

### 1.3 15 Case 任务类型分布

| 任务类型 | Case 数 | RT avg F1 | AD avg F1 | BL avg F1 |
|---------|---------|-----------|-----------|-----------|
| Object Recognition | 3 | 0.139 | 0.124 | 0.059 |
| Object Function | 3 | 0.096 | 0.111 | 0.074 |
| Object State Change Recognition | 2 | 0.250 | 0.250 | 0.184 |
| Ego Object State Change Recognition | 2 | 0.000 | 0.143 | 0.000 |
| Attribute Perception | 2 | 0.297 | 0.200 | 0.250 |
| Text-Rich Understanding | 1 | **0.667** | 0.154 | 0.000 |
| Action Recognition | 1 | 0.125 | 0.000 | 0.235 |
| Object Localization | 1 | 0.000 | 0.200 | 0.095 |

---

## 2. 逐 Case 分析表

### Case 1: f7328dc0ae72 — Object Recognition

- **Question**: "Can you remind me what the yellow object near the cabinet doors is?"
- **GT windows**: [2430-2436.5], [2444.5-2496], [2619-2620.5] (3 windows)
- **RT**: F1=0.167, TP=1, FP=8, FN=2, 9 triggers, yes_rate=45.2%
- **AD**: F1=0.211, TP=2, FP=14, FN=1, 18 triggers
- **BL**: F1=0.000
- **Gap separation**: +0.671 (弱正向)
- **Winner**: ADAPTIVE-D
- **失败模式**: RT FP-dominant。RT 在 2430-2710 持续触发，但大部分在 GT 之外。VLM 识别到"黄色物体"出现在视野但无法精确定位最佳时刻。AD 高频触发（gap 很高，>5），但至少命中了更多 GT 窗口。

### Case 2: 8563cf568b39 — Object State Change Recognition

- **Question**: "Can you remind me when the state of the refrigerator changes?"
- **GT windows**: [1022-1023.5], [1060.5-1063.5], [1072-1074.5] (3 windows, 每个仅 1.5-3s)
- **RT**: F1=0.000, TP=0, FP=2, FN=3, 2 triggers at t=1000,1030
- **AD**: F1=0.000, TP=0, FP=0, FN=3, 0 triggers
- **BL**: F1=0.222 (较好)
- **Gap separation**: +2.804
- **Winner**: BASELINE
- **失败模式**: **两者都失败，BL 反而最好**。GT 窗口极短（1.5-3s），RT 触发太早（t=1000, GT 在 1022 才开始）。AD 在 tau=2.8 下 0 次触发，GT 步的 avg gap 仅 -0.499。这类"状态瞬间变化"的 GT 窗口太短，任何有 cooldown 的系统都难以命中多个窗口。BL 的宽触发（trigger_rate=28.8%）反而能偶然覆盖。

### Case 3: 1b83f0628411 — Ego Object State Change Recognition

- **Question**: "When does the cat change its position relative to me?"
- **GT windows**: [1631.5-1636] (1 window, 仅 4.5s)
- **RT**: F1=0.000, TP=0, FP=0, FN=1, 0 triggers, yes_rate=0%
- **AD**: F1=0.286, TP=1, FP=5, FN=0
- **BL**: F1=0.05z
- **Gap separation**: -2.293 (反向!)
- **Winner**: ADAPTIVE-D
- **失败模式**: RT **完全沉默** (0% yes_rate)。VLM 从未认为"此刻应告知用户猫的位置变化"。可能因为"猫移动"在第一人称视频中不够显著。AD 在 GT 附近恰好有一次高 gap 触发成功命中。反向 gap separation 说明 GT 步的 logprob gap 反而低，但 tau=2.8 下仍有触发（因为 gap 绝对值高于阈值）。

### Case 4: ec8c429db0d8 — Text-Rich Understanding **[RT 最佳表现]**

- **Question**: "What specific text is written on the sticky note when I look at it?"
- **GT windows**: [3079.5-3083], [3094-3099] (2 windows)
- **RT**: F1=**0.667**, TP=1, FP=0, FN=1, 1 trigger at t=3080, yes_rate=3.1%
- **AD**: F1=0.154, TP=1, FP=10, FN=1, 11 triggers
- **BL**: F1=0.114
- **Gap separation**: +0.372 (弱)
- **Winner**: **REASONING TRIGGER**
- **失败模式**: RT **精准单次触发**，完美命中 GT 窗口。VLM 看到便利贴文字 → 判断"用户看到了，该帮忙了"→ YES。而 AD 因为 gap 长期 > 2.8，在 GT 后持续触发导致大量 FP。**Text-Rich 的核心优势**：文字可见性是最直接的视觉信号，VLM 可以精确判断"我现在能看到文字内容了"。

### Case 5: 0eb97247e799 — Object Recognition

- **Question**: "What is the category of the object I draw on when I draw on the book?"
- **GT windows**: 8 个窗口，分布在 1614-1896 范围
- **RT**: F1=0.250, TP=3, FP=13, FN=5, 16 triggers, yes_rate=**96.9%**
- **AD**: F1=0.160, TP=2, FP=15, FN=6
- **BL**: F1=0.077 (BL trigger_rate=100%)
- **Gap separation**: +0.058 (几乎无分离)
- **Winner**: REASONING
- **失败模式**: 这是一个极端案例：几乎**所有帧**都显示画书的场景，因此 RT 的 96.9% yes rate 虽然极端，但确实反映了场景的持续相关性。Gap separation 接近 0，说明 logprob gap 在此 case 中**完全无判别力**。RT 凭借更多触发 (16>17) 与不同的触发时间分布略微领先。

### Case 6: 022907170077 — Object Localization

- **Question**: "Where is the smartphone located relative to the person lying on the couch?"
- **GT windows**: [1756.5-1776], [1891.5-1896.5], [1917-1920] (3 windows)
- **RT**: F1=0.000, TP=0, FP=1, FN=3, 1 trigger at t=1924, yes_rate=3.1%
- **AD**: F1=0.200, TP=2, FP=15, FN=1, 18 triggers
- **BL**: F1=0.107
- **Gap separation**: -0.704 (反向)
- **Winner**: ADAPTIVE-D
- **失败模式**: RT 唯一的触发在 t=1924，恰好在最后一个 GT 窗口 (1917-1920) **之后** 4 秒。VLM 判断太晚。此外，localization 任务需要识别"手机的空间关系"，这对 VLM 推理来说较难。AD 高频触发反而通过概率覆盖命中了 2/3 GT。

### Case 7: eb1a417d26c9 — Object Recognition

- **Question**: "Can you remind me what the electronic device on the desk is when it comes into view?"
- **GT windows**: [1111-1141.5], [1147.5-1192.5], [1210.5-1211.5], [1230.5-1233.5] (4 windows)
- **RT**: F1=0.000, TP=0, FP=7, FN=4, 7 triggers at 1120-1390, yes_rate=28.1%
- **AD**: F1=0.000, TP=0, FP=2, FN=4, 2 triggers
- **BL**: F1=0.157
- **Gap separation**: +0.943
- **Winner**: BASELINE
- **失败模式**: **两者都 F1=0**，BL 反而最好。RT 在 GT 范围内有触发 (t=1120 在 [1111-1141.5] 内)，但评估可能因 cooldown/距离超出 ANTICIPATION/LATENCY 窗口而未匹配。GT 窗口 [1210.5-1211.5] 仅 1 秒，极难命中。AD 的 2 次触发都错过了 GT。这是"多个短 GT 窗口"的典型失败场景。

### Case 8: 85a99403e624 — Object State Change Recognition **[双赢]**

- **Question**: "Can you remind me when the state of the phone changes?"
- **GT windows**: [1145.5-1147], [1148.5-1184], [1185-1185.5] (3 windows)
- **RT**: F1=0.500, TP=1, FP=0, FN=2, 1 trigger at t=1150, yes_rate=3.1%
- **AD**: F1=0.500, TP=1, FP=0, FN=2, 2 triggers
- **BL**: F1=0.091
- **Gap separation**: +1.845
- **Winner**: TIE (RT=AD=0.500)
- **失败模式**: 两者都成功命中了主要 GT 窗口 [1148.5-1184]（35.5s 长窗口）。RT 在 t=1150 精准触发（完美！），AD 在 t=1154 也成功。但两者都错过了极短窗口 [1145.5-1147](1.5s) 和 [1185-1185.5](0.5s)。**长 GT 窗口 + 明确状态变化** = 两种方法都有效。

### Case 9: 4a51f9ebaa22 — Object Function

- **Question**: "I need to prepare herbs. Where should I put my herbs?"
- **GT windows**: [175-176.5], [284-296.5], [64.5-116], [149.5-152], [274-276] (5 windows)
- **RT**: F1=0.133, TP=1, FP=9, FN=4, 10 triggers at 30.5-310.5, yes_rate=46.9%
- **AD**: F1=0.333, TP=2, FP=5, FN=3, 9 triggers
- **BL**: F1=0.213
- **Gap separation**: +2.763
- **Winner**: ADAPTIVE-D
- **失败模式**: RT FP-dominant。VLM 对整个烹饪场景持续说 YES（yes_rate=47%），无法区分"用户正在拿草药"和"用户在做其他事"。AD 利用 gap 的高峰精准性（gap_sep=2.763）更准确地定位了草药相关帧。

### Case 10: a14e569060e8 — Object Function

- **Question**: "How do I ensure the dough is shaped properly before baking?"
- **GT windows**: [189-194], [199.5-209.5] (2 windows)
- **RT**: F1=0.154, TP=1, FP=10, FN=1, 11 triggers, yes_rate=58.1%
- **AD**: F1=0.000, 0 triggers (tau=2.8 全部过滤)
- **BL**: F1=0.190
- **Gap separation**: +3.515 (很高!)
- **Winner**: BASELINE
- **失败模式**: **矛盾的一个 case**。Gap separation 很高 (+3.515)，但 AD 的 0 次触发说明在 tau=2.8 下所有 gap 都低于阈值（GT 步 avg_gap=0.312）。gap 的绝对值太低，即使分离度好，阈值也太高了。RT 的 58% yes rate 导致大量 FP 但至少命中了一个 GT。**这正是 Phase II Adaptive-D (tau=0.5 for Object Function) 会大幅改善的 case**。

### Case 11: 2098c3e8d904 — Action Recognition

- **Question**: "I want to know when I sprinkle flour on the mound of dough. Could you let me know?"
- **GT windows**: 8 个极短窗口 (每个仅 0.5-1s)，分布在 8-249.5 范围
- **RT**: F1=0.125, TP=1, FP=7, FN=7, 8 triggers, yes_rate=32.3%
- **AD**: F1=0.000, 0 triggers (tau=2.8 全部过滤)
- **BL**: F1=0.200
- **Gap separation**: +2.693
- **Winner**: BASELINE
- **失败模式**: **Action Recognition 的典型失败**。GT 窗口极短（"撒面粉"是瞬间动作，每次仅 0.5s），AD 在 tau=2.8 下完全静默。RT 虽然有触发但大部分错位。Phase II 的 tau=0.5 可以拯救 AD（Action Recognition 的最优 tau 就是 0.5）。这个 case 证明了瞬间动作需要极低阈值或完全不同的检测机制。

### Case 12: 958a57945bfc — Ego Object State Change Recognition

- **Question**: "Can you remind me how the dough changes position when I throw it onto the countertop?"
- **GT windows**: [3202-3203.5], [3205.5-3206.5], [3224-3225] (3 windows, 每个 1-1.5s)
- **RT**: F1=0.000, TP=0, FP=1, FN=3, 1 trigger at t=3199.5
- **AD**: F1=0.000, 0 triggers
- **BL**: F1=0.333
- **Gap separation**: +2.337
- **Winner**: **ALL FAIL** (RT=AD=0, BL=0.333 来自 trigger_rate=5.8%)
- **失败模式**: GT 窗口极短 + 需要精确识别"面团被扔到台面上的瞬间"。RT 在 t=3199.5 触发（GT 在 3202 才开始），提前了 2.5 秒，恰好在 ANTICIPATION=1s 之外。AD 的 avg_gap=-2.740 为负，完全不触发。BL 靠 5.8% 的低触发率偶然命中。**瞬时物理动作** 是两种方法的共同盲点。

### Case 13: 5f334102e386 — Attribute Perception **[RT 良好表现]**

- **Question**: "Can you remind me of the color of the mixer when I touch the flour inside it?"
- **GT windows**: [2989.5-2990.5] (单窗口, 仅 1s)
- **RT**: F1=0.400, TP=1, FP=3, FN=0, 4 triggers, yes_rate=15.6%
- **AD**: F1=0.000, 0 triggers
- **BL**: F1=0.182
- **Gap separation**: +2.059
- **Winner**: **REASONING**
- **失败模式**: AD 在 tau=2.8 下完全沉默（GT 步 gap=0.750，远低于阈值）。RT 在 t=2989.5 精准命中了唯一的 GT 窗口。VLM 成功识别了"用户手接触搅拌器中的面粉"这一视觉事件。**Attribute Perception 在 Phase II 的最优 tau=0.5**，如果使用自适应阈值，AD 可能也能触发。

### Case 14: 48c5b16addef — Attribute Perception

- **Question**: "Can you remind me of the color of the dough when I roll it on the table?"
- **GT windows**: 18 个窗口，分布在 2025-2242 范围，平均间隔约 10s
- **RT**: F1=0.194, TP=3, FP=10, FN=15, 13 triggers, yes_rate=65.6%
- **AD**: F1=0.400, TP=7, FP=10, FN=11, 17 triggers
- **BL**: F1=0.514 (BL 最好!)
- **Gap separation**: +0.373 (弱)
- **Winner**: ADAPTIVE-D (over RT; BL actually best)
- **失败模式**: 18 个 GT 窗口（用户持续揉面），场景高度重复。BL 的 100% trigger_rate 反而最优（因为几乎每一帧都是 GT）。RT 的 13 次触发无法覆盖 18 个 GT 窗口，且 cooldown 阻止了密集触发。**当 GT 密度极高时，任何限制触发频率的方法都不如 BL**。

### Case 15: e58c97407823 — Object Function

- **Question**: "I need to handle an emergency. Where should I look?"
- **GT windows**: [1080.5-1182], [1187-1192.5] (2 windows, 第一个长达 101.5s!)
- **RT**: F1=0.000, TP=0, FP=9, FN=2, 9 triggers at 1090.5-1360.5, yes_rate=37.5%
- **AD**: F1=0.000, 0 triggers
- **BL**: F1=0.069
- **Gap separation**: -0.115 (无分离)
- **Winner**: **ALL FAIL**
- **失败模式**: 这是一个抽象的功能性问题（"紧急情况该看哪里"），视觉线索不明确。RT 在 GT 时间范围内有触发（1090.5 在 [1080.5-1182] 内），但评估为 FP，可能是因为触发的帧不在 GT bucket 内或距离最佳时刻太远。Gap 完全无分离，AD 也失败。**抽象功能性问题** 的视觉触发信号最弱。

---

## 3. 失败模式分类

### 3.1 分类矩阵

| 分类 | 任务类型 | RT avg | AD avg | BL avg | 根因 |
|------|---------|--------|--------|--------|------|
| **A: Logprob 有效, RT 失败** | Ego Object State Change | 0.000 | 0.143 | 0.000 | RT 完全沉默 |
| | Object Localization | 0.000 | 0.200 | 0.095 | RT 触发延迟 |
| **B: RT 有效, Logprob 失败** | Attribute Perception | 0.297 | 0.200 | 0.250 | Gap 弱, RT 视觉判断好 |
| **C: 两者都失败** | Action Recognition | 0.125 | 0.000 | 0.235 | 极短 GT + 瞬间动作 |
| **D: 两者都有效** | Object Recognition | 0.139 | 0.124 | 0.059 | 场景持续相关 |
| | Object State Change | 0.250 | 0.250 | 0.184 | 状态变化明显 |
| | Text-Rich Understanding | **0.667** | 0.154 | 0.000 | 文字可见性信号 |
| | Object Function | 0.096 | 0.111 | 0.074 | 两者 F1 都低 |

### 3.2 A 类：Logprob 有效但 Reasoning 失败

**涉及类型**: Ego Object State Change Recognition, Object Localization

**为什么 Reasoning 失败？**

1. **视觉信号不够显著**：Ego 视角下的微小状态变化（"猫移动了位置"）或空间关系变化，在单帧图像上不足以让 VLM 判断"此刻需要提醒用户"。RT 的 yes_rate 极低（0%-3.1%），说明 VLM 对这类场景**几乎从不说 YES**。

2. **抽象推理需求**：Object Localization 需要理解"手机相对于沙发上的人的位置"，这是空间推理而非简单视觉识别。

**为什么 Logprob 有效？**

Logprob gap 捕获的是 Qwen3-VL 内部的"不确定性变化"，即使 VLM 最终输出 YES，gap 的波动也能反映场景变化。这种隐式信号不需要 VLM 做显式推理。

### 3.3 B 类：Reasoning 有效但 Logprob 失败

**涉及类型**: Attribute Perception (部分), Text-Rich Understanding

**为什么 Reasoning 有效？**

1. **Text-Rich (F1=0.667 vs 0.154)**：文字可见性是最直接的视觉判据。Gemini-3-Flash 看到便利贴上有文字 → 精准单次触发 → 完美命中。无需时序推理，只需回答"我现在能看到答案了吗？"

2. **Attribute Perception (Case 13, F1=0.400 vs 0.000)**：VLM 识别到"手接触搅拌器中的面粉" → YES。这是一个具体可见的交互事件。而 logprob gap 在此 case 的 GT 步仅 0.750，远低于 tau=2.8。

**为什么 Logprob 失败？**

这两类的 gap separation 较低（Text-Rich=0.372, Attribute Case 13=2.059）。Gap 信号在这些场景中**绝对值太低**，即使有正向分离，在 tau=2.8 下也被过滤掉了。Phase II 的自适应 tau (Text-Rich=2.0, Attribute=0.5) 会大幅改善。

### 3.4 C 类：两者都失败

**涉及类型**: Action Recognition

**根本原因**：

1. **极短 GT 窗口**：Action Recognition 的 GT 窗口平均仅 0.5-1s（"撒面粉"是瞬间动作）。在 10s 采样间隔（RT）或 ~5.7s 步长（AD）下，命中概率极低。

2. **动作的时间不确定性**：即使 VLM 看到"用户在揉面团"，也无法预测"下一次撒面粉"的精确时刻。这不是单帧能判断的。

3. **AD 在 tau=2.8 下完全沉默**：Action Recognition 在 Phase II 的最优 tau=0.5，但在 Phase I 的 2.8 下 recall=0。

**结构性结论**：瞬间动作类触发需要**高频采样 + 低阈值**或**专门的动作检测模型**。

### 3.5 D 类：两者都有效

**涉及类型**: Object Recognition, Object State Change, Object Function, Text-Rich

**什么特征让这些 case 容易？**

1. **GT 窗口较长**：State Change Case 8 的主窗口 35.5s，Object Function 的多个窗口合计 >50s。长窗口容忍时序偏差。

2. **视觉信号明确且持续**：Object Recognition 的目标物体在视野中持续可见，Object State Change 的状态变化明显（手机状态改变）。

3. **场景持续相关**：Object Recognition Case 5 的 yes_rate=96.9%（几乎整个视频都在画书），这意味着密集触发反而有利。

---

## 4. 关键发现

### 发现 1：Reasoning Trigger 的精度-覆盖率 trade-off

| yes_rate 范围 | Cases | avg RT F1 | 特征 |
|--------------|-------|-----------|------|
| 0-5% | 5 | 0.233 | 精准但低覆盖 (Text-Rich, State Change) |
| 5-35% | 4 | 0.072 | 中等，多为 FP |
| 35-60% | 4 | 0.121 | FP 主导 |
| >60% | 2 | 0.222 | 极端但场景高度相关 |

**核心矛盾**：RT 在 yes_rate 低 (<5%) 时精度最高，因为 VLM 只在最确定时才说 YES。但覆盖率低意味着很多 GT 被漏掉。

### 发现 2：GT 窗口时长是决定性因素

| GT 窗口平均时长 | RT 表现 | AD 表现 | 原因 |
|---------------|---------|---------|------|
| < 2s (瞬间动作) | 差 | 差 | 采样率不足以命中 |
| 2-10s (短事件) | 中等 | 中等 | 依赖触发时机精度 |
| > 10s (持续场景) | 好 | 好 | 容忍时序偏差 |

Case 8 (State Change, 主窗口 35.5s): 两者都 F1=0.500
Case 11 (Action, 8 个 0.5s 窗口): 两者都 F1<=0.125

### 发现 3：Logprob Gap 绝对值 vs 分离度

| Case | Gap sep | GT avg gap | tau=2.8 能否触发 | 结果 |
|------|---------|-----------|-----------------|------|
| Case 10 (a14e) | +3.515 | 0.312 | 否 (gap<<tau) | AD F1=0 |
| Case 11 (2098) | +2.693 | 1.831 | 否 | AD F1=0 |
| Case 1 (f732) | +0.671 | 5.154 | 是 (gap>>tau) | AD F1=0.211 |

**分离度高不等于可触发**。分离度衡量的是 GT 与非 GT 的相对差异，但如果绝对值都低于阈值，高分离度也无用。Phase II 的自适应 tau 正是解决了这个问题。

### 发现 4：Cooldown 对 RT 的双重影响

- **正面**: Case 4 (Text-Rich) — cooldown 防止了 RT 在精准触发后产生连续 FP
- **负面**: Case 14 (Attribute) — 18 个 GT 窗口，cooldown=12s 限制了最多触发 ~16 次 / 320s，无法覆盖所有 GT

### 发现 5：RT 的 FP 来源分析

15 个 case 共 80 个 FP，按原因分类：

| FP 原因 | 数量(估) | 典型 case |
|---------|---------|----------|
| **场景相关但时间偏差** | ~35 | Case 1, 5, 7: 看到目标物但不在 GT 窗口 |
| **VLM 过度触发 (yes-bias)** | ~25 | Case 10, 14: yes_rate>50% |
| **抽象场景误判** | ~12 | Case 15: 不明确的功能性问题 |
| **GT 后持续触发** | ~8 | Case 4 (AD): GT 结束后 gap 仍高 |

---

## 5. 对新方案的启示

### 5.1 混合策略设计

基于分析，建议如下路由：

```
if task_type == "Text-Rich Understanding":
    use Reasoning Trigger (Gemini-3-Flash)  # F1: 0.154 → 0.667
elif task_type == "Task Understanding":
    use Baseline (bypass D)                  # gap 反相关
elif task_type in ["Action Recognition", "Attribute Perception"]:
    use Adaptive-D with tau=0.5             # 需要低阈值
elif task_type in ["Information Function", "Object Recognition"]:
    use Adaptive-D with tau=2.5-2.8         # gap 信号强
else:
    use Adaptive-D with type-optimal tau
```

### 5.2 解决极短 GT 窗口问题

- 对 Action Recognition 等瞬间动作类型，10s 采样间隔太长
- 建议：对这些类型用 2-3s 间隔 + tau=0.5，或基于视觉变化检测的自适应采样
- 或引入"动作完成检测"而非"场景相关性检测"

### 5.3 RT prompt 改进方向

当前 RT prompt 要求 VLM 判断"此刻是否是最佳帮助时机"，这在非 Text-Rich 场景下太模糊。

改进方向：
- **针对动作类型**：加入"用户刚完成了什么动作？"的判断
- **针对状态变化**：加入"与上一帧相比有什么变化？"的时序对比
- **降低 yes-bias**：在 prompt 中增加具体否定条件（"如果你无法看到具体变化的内容，说 NO"）

### 5.4 评估指标反思

ESTP-Bench 的 ANTICIPATION=1s + LATENCY=2s 对瞬间动作极其严格。Case 12 中 RT 在 t=3199.5 触发，GT 在 3202.0 开始，仅差 2.5s 就被判为 FP。考虑是否需要按任务类型调整匹配窗口。

### 5.5 数据不均衡

15 case 中 Text-Rich 仅 1 个，但它贡献了 RT 最大的 F1 提升（+0.667）。需要在更大规模评估中验证 Text-Rich 的 RT 优势是否稳定。全量 5 case 数据中有 5 个 Text-Rich case，Phase II 的 sweep 数据可用于交叉验证。

---

## 附录：15 Case 完整对比表

| # | Case ID | Task Type | RT F1 | AD F1 | BL F1 | Winner | RT yes% | Gap Sep | FP/FN Pattern |
|---|---------|-----------|-------|-------|-------|--------|---------|---------|---------------|
| 1 | f7328dc0 | Object Recog | 0.167 | 0.211 | 0.000 | AD | 45.2% | +0.671 | FP-dom |
| 2 | 8563cf56 | Obj State Change | 0.000 | 0.000 | 0.250 | BL | 6.2% | +2.804 | FN-dom |
| 3 | 1b83f062 | Ego Obj State | 0.000 | 0.286 | 0.000 | AD | 0.0% | -2.293 | FN-dom |
| 4 | ec8c429d | **Text-Rich** | **0.667** | 0.154 | 0.000 | **RT** | 3.1% | +0.372 | FN-dom |
| 5 | 0eb97247 | Object Recog | 0.250 | 0.160 | 0.077 | RT | 96.9% | +0.058 | FP-dom |
| 6 | 02290717 | Object Loc | 0.000 | 0.200 | 0.095 | AD | 3.1% | -0.704 | FN-dom |
| 7 | eb1a417d | Object Recog | 0.000 | 0.000 | 0.100 | BL | 28.1% | +0.943 | FP-dom |
| 8 | 85a99403 | Obj State Change | 0.500 | 0.500 | 0.118 | TIE | 3.1% | +1.845 | FN-dom |
| 9 | 4a51f9eb | Object Func | 0.133 | 0.333 | 0.000 | AD | 46.9% | +2.763 | FP-dom |
| 10 | a14e5690 | Object Func | 0.154 | 0.000 | 0.222 | BL | 58.1% | +3.515 | FP-dom |
| 11 | 2098c3e8 | Action Recog | 0.125 | 0.000 | 0.235 | BL | 32.3% | +2.693 | Balanced |
| 12 | 958a5794 | Ego Obj State | 0.000 | 0.000 | 0.000 | ALL_FAIL | 3.1% | +2.337 | FN-dom |
| 13 | 5f334102 | Attribute | 0.400 | 0.000 | 0.333 | RT | 15.6% | +2.059 | FP-dom |
| 14 | 48c5b16a | Attribute | 0.194 | 0.400 | 0.167 | AD | 65.6% | +0.373 | FN-dom |
| 15 | e58c9740 | Object Func | 0.000 | 0.000 | 0.000 | ALL_FAIL | 37.5% | -0.115 | FP-dom |

**Winner 分布**: AD=5, BL=4, RT=3 (含 Case 5), TIE=1, ALL_FAIL=2
