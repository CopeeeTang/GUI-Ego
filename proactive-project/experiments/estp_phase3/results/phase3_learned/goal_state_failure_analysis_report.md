# Goal-State Anti-Correlation 失败分析报告

> 日期：2026-02-19
> 数据来源：ESTP-Bench 5-cases subset, 53 matched cases
> 模型：Gemini-2.0-Flash (53 cases), Gemini-3-Flash (28 cases)
> 帧提取：已修复 clip_start_time 对齐 (2026-02-19)

---

## 1. 核心发现

Goal-state 分类（on_track / uncertain / off_track）与触发需求**反相关**：
在 GT 窗口内（需要帮助时），模型更倾向于说 "on_track"。

| Model | GT mean | non-GT mean | GT separation |
|-------|---------|-------------|---------------|
| Gemini-2.0-Flash | **+0.759** | +0.122 | **-0.636** |
| Gemini-3-Flash | **+0.906** | +0.311 | **-0.595** |

> Score: on_track=+1, uncertain=0, off_track=-1
> GT separation = mean(non_GT) - mean(GT)，**正值**表示好的判别力

---

## 2. 任务类型详细分析

### 2.1 反向效应最强：Text-Rich Understanding (GT sep = -1.359)

**5 个 case 全部为负**，这是反向效应最极端的任务类型。

#### 失败机制

Text-Rich 任务的问题模式：
- Q: "What brand name is printed on the blue can?"
- Q: "What specific text is written on the sticky note?"
- Q: "What is the brand of the blue box?"

GT 窗口 = **用户正在凝视带有文字的物体**。此时：
- 视觉上：物体 + 文字清晰可见
- 模型判断："用户正在看目标物体" → **on_track**
- 实际需求：用户**需要被告知文字内容** → 应该触发

| Case | Question | GT label | non-GT分布 | sep |
|------|----------|----------|-----------|-----|
| cfab4d8a | "textual info on blue books?" | on_track | 0/0/30 (全off_track) | **-2.000** |
| ec8c429d | "text on sticky note?" | on_track | 5/4/22 | **-1.548** |
| d12a770d | "brand on blue can?" | on_track | 21/2/7 | -0.533 |
| 6bd31d1c | "characters on white object?" | on_track | 22/1/8 | -0.548 |
| c5e5cf04 | "brand of blue box?" | on_track | 24/1/5 | -0.367 |

**Case cfab4d8a 深度分析** (sep = -2.000，最极端)：
- Q: "Is there any textual information on the blue books when I stare at the room?"
- GT 窗口：[0.0s, 1.5s] — 视频最开始1.5秒
- GT 时刻 label = on_track（用户正在看蓝色书）
- 其余 30 个 bucket 全部 = off_track（用户在做其他事情）
- A: "No relative object. The description mentions blue books with handwritten notes..."

**根本矛盾**：Text-Rich 任务要求用户**看到文字才能获得帮助**。"看到文字" = on_track = GT 窗口。Goal-state 判断与触发需求的语义方向完全相反。

#### 为什么 Reasoning Trigger 在 Text-Rich 上反而表现好？

Reasoning Trigger 在 Text-Rich 的 F1=0.667 vs Adaptive-D 的 0.154：
- Reasoning prompt: "Can you SEE the specific object the question asks about?"
- 在 GT 窗口：VLM 能看到文字 → YES → 触发 ✓
- 在 non-GT 窗口：VLM 看不到文字 → NO → 不触发 ✓
- 与 goal-state 的"on_track"判断语义相同，但 YES/NO 的决策方向正确

---

### 2.2 中等反向：Ego Object State Change Recognition (GT sep = -0.375)

**4/5 case 为负**，1 case 无 GT bucket 命中。

#### 失败机制

问题模式：
- Q: "When does the man running change his position relative to me?"
- Q: "When does the car change position relative to me?"
- Q: "How does the dough change position when I throw it?"

GT 窗口 = **状态变化正在发生**。此时：
- 视觉上：目标对象正在移动/变化，用户与对象互动中
- 模型判断："用户正在与目标对象互动" → **on_track**
- 实际需求：用户**需要被告知变化正在发生** → 应该触发

| Case | Question | GT label | sep | 特殊情况 |
|------|----------|----------|-----|---------|
| 16a65a14 | "man running changes position?" | on_track | **-1.290** | — |
| 92a7fb33 | "man on bike changes position?" | on_track | **-1.290** | — |
| faf10555 | "car changes position?" | on_track | **-0.581** | — |
| 1b83f062 | "cat changes position?" | — | -0.625 | GT 未命中 |
| 958a5794 | "dough changes position?" | — | -0.125 | GT 未命中 |

**Case 16a65a14 深度分析** (sep = -1.290)：
- Q: "When does the man running changes his position relative to me?"
- GT 窗口：[1809.0s, 1813.5s]，A: "He moves towards you and become closer to you"
- t=190.0s (abs=1810.5s) → **on_track** ★GT★
- 周围 non-GT：11 个 on_track, 20 个 off_track（用户在做其他事时 off_track）
- 模型在人跑步靠近时说 "on_track"（用户确实在看这个人），但这恰恰是需要触发的时刻

**根本矛盾**：State Change 任务的触发时机 = 变化发生的瞬间。在变化发生时，用户必然在"观察"变化中的对象 → on_track。但 on_track ≠ "不需要帮助"。

---

### 2.3 为什么某些类型显示正向 GT sep？

#### Ego Object Localization (GT sep = +0.851) — 伪正向

**4/5 case 的 GT 窗口未被 10s bucket 命中**，正向 separation 来自统计伪影。

| Case | GT窗口宽度 | Bucket命中? | GT mean | non-GT mean |
|------|-----------|------------|---------|-------------|
| 00828d84 | 2.5s | ✗ | 0 (默认) | +0.875 |
| 042a58cd | 2.5s | ✗ | 0 (默认) | +0.800 |
| f0ba248f | 1.5s | ✗ | 0 (默认) | -0.355 |
| 39b8c54f | 36.0s | ✗ | 0 (默认) | -0.250 |
| 7f494611 | 3.5s | ✓ (1个) | +1.000 | +0.733 |

**机制**：
- 10s bucket 分辨率下，匹配窗口 = [GT_start - 1s, GT_start + 2s] = 仅 3s 宽
- 10s bucket 命中 3s 窗口的概率约 30%
- 未命中时 GT mean 默认为 0，而 non-GT 大多为 on_track (+0.8) → 伪正向

**证据**：全部 12 种任务类型的 GT 窗口命中率：

| Task Type | GT总数 | 命中 | 未命中率 | 平均GT宽度 |
|-----------|--------|------|---------|-----------|
| Task Understanding | 25 | 2 | **92%** | 37.3s |
| Ego Object Localization | 8 | 1 | **88%** | 7.0s |
| Attribute Perception | 28 | 9 | **68%** | 2.8s |
| Object Localization | 9 | 3 | **67%** | 4.4s |
| Object Function | 12 | 4 | **67%** | 16.3s |
| Action Recognition | 32 | 11 | **66%** | 1.7s |
| Text-Rich Understanding | 16 | 6 | 62% | 15.2s |
| Ego Object State Change | 7 | 3 | 57% | 5.5s |
| Information Function | 7 | 3 | 57% | 8.0s |
| Object State Change | 9 | 4 | 56% | 5.2s |
| Object Recognition | 20 | 13 | 35% | 9.2s |
| Action Reasoning | 3 | 0 | **100%** | 32.4s |

> 注意：即使 Task Understanding 的 GT 窗口平均 37.3s，也只有 8% 命中率，
> 这是因为 ESTP-F1 的匹配窗口仅看 [GT_start - 1s, GT_start + 2s]。

#### Information Function (GT sep = +0.312) — 混合型

5 个 case 中 3 个无 GT 命中，2 个有命中：

| Case | 命中? | GT label | sep | 原因 |
|------|-------|----------|-----|------|
| d2bf5beb | ✓ | **off_track** | +0.677 | Q:"What OS?" VLM 在看到屏幕时说 off_track |
| d6bca078 | ✓ | uncertain | -0.367 | Q:"建筑安全建议" — 抽象问题 |
| 2a05a129 | ✗ | — | -0.156 | GT 未命中 |
| 83e3558b | ✗ | — | +0.781 | GT 未命中 |
| eb9aa60b | ✗ | — | +0.500 | GT 未命中 |

**Case d2bf5beb** 是唯一真正"正确"的案例：
- Q: "What's operating system is running according to the monitor?"
- GT at 2631.5s: A: "It's a WiiU, according to the logo on screen"
- t=200.0s (abs=2630.5s) → **off_track** — 模型判断用户还没走到显示器前 → 正确！
- 但这是因为 GT 恰好在用户"接近目标"的转折点，模型的 off_track 反映了"还没到"

---

## 3. 根本原因模型

### 3.1 语义错配矩阵

| 用户状态 | Goal-State 判断 | 是否需要触发 | 匹配? |
|----------|----------------|-------------|-------|
| 看着目标物体 | on_track ✓ | **需要**（Text-Rich, Object Recog） | ✗ |
| 与目标互动中 | on_track ✓ | **需要**（State Change） | ✗ |
| 做无关事情 | off_track ✓ | 不需要 | ✓ |
| 走向目标 | uncertain/on_track | 不确定 | ~ |
| 离开目标 | off_track ✓ | 不需要 | ✓ |

Goal-state 在"不需要帮助"的情况下表现正确（off_track = 不需要），但在"需要帮助"的核心场景下系统性错误（on_track = 需要帮助）。

### 3.2 概念性矛盾的本质

ESTP-Bench 的 GT 定义：**当信息对用户最有用时触发**。
- 信息有用的前提 = 用户**正在与相关场景互动**
- 与相关场景互动 = goal-state 判断的 **on_track**
- 因此：GT ⊂ on_track（几乎总是成立）

这不是模型能力问题（3 个模型都展现同样的模式），而是**评估框架与特征定义之间的结构性矛盾**。

### 3.3 为什么 Logprob Gap 不受此影响？

Logprob gap = logP("yes") - logP("no") from VLM trigger question。

| 特征 | 信号含义 | 与 GT 的关系 |
|------|---------|-------------|
| goal_state | "用户进展如何？" | 反相关（on_track ↔ GT） |
| logprob gap | "此刻是否该提供信息？" | 正相关（高 gap ↔ GT） |

Logprob gap 直接编码了"此刻的信息相关性"，而不是"用户的任务进展"。这是 Adaptive-D（基于 logprob gap）优于所有 goal-state 方法的根本原因。

---

## 4. 对比总结：各方法在不同任务类型的表现

在 15 个公平对比 case 上：

| Task Type | Reasoning Trigger | Adaptive-D | Baseline | 最优方法 |
|-----------|-------------------|------------|----------|---------|
| **Text-Rich** | **0.667** | 0.154 | 0.154 | Reasoning |
| Obj State Change | 0.250 | **0.250** | 0.250 | 持平 |
| Attribute | 0.297 | **0.472** | 0.389 | Adaptive-D |
| Object Recog | 0.139 | **0.239** | 0.214 | Adaptive-D |
| Object Func | 0.096 | **0.351** | 0.241 | Adaptive-D |
| Action Recog | 0.125 | **0.250** | 0.235 | Adaptive-D |
| Ego Obj State | 0.000 | **0.327** | 0.072 | Adaptive-D |
| Object Loc | 0.000 | **0.200** | 0.200 | Adaptive-D |

---

## 5. 结论与建议

### 5.1 Goal-State 特征不可用

Goal-state 分类（on_track / off_track）与触发需求反相关，原因是概念性矛盾而非模型能力问题。3 个不同模型（Qwen3-VL, Gemini-2.0-Flash, Gemini-3-Flash）都展现同样的模式。**不建议继续在此方向投入**。

### 5.2 Reasoning Trigger 仅对 Text-Rich 有效

VLM-as-Judge 方法（直接问"该帮忙吗？"）在 Text-Rich 类型表现优异（F1=0.667），但在其他 11 种类型上均不如 Adaptive-D。原因：
- Text-Rich 的视觉信号最直接（文字可见 = 该帮忙）
- 其他类型需要更细粒度的时序判断能力

### 5.3 推荐策略

1. **主路线**：继续优化 Adaptive-D（logprob gap + 自适应阈值路由）
2. **Text-Rich 混合**：对 Text-Rich 类型单独使用 Reasoning Trigger 或更低的 τ
3. **放弃 Goal-State**：不作为 MLP 特征，不作为独立触发信号
4. **Cooldown 优化**：当前 12s cooldown 阻塞了部分正确触发（见 Case 6bd31d1c），可探索自适应 cooldown

---

## 附录：文件索引

| 文件 | 说明 |
|------|------|
| `goal_state_cache_gemini2flash.json` | Gemini-2.0-Flash, 53 cases, 修复后 |
| `goal_state_cache_gemini3flash.json` | Gemini-3-Flash, 28 cases, 修复后 |
| `goal_state_comparison_report.txt` | 自动生成的对比报告 |
| `reasoning_trigger_pilot.json` | Reasoning trigger, 15 cases 结果 |
| `stale_before_fix/` | 修复前的旧缓存（备份） |
