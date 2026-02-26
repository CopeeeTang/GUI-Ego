# 上下文增强推理触发 (Context-Enhanced Reasoning Trigger) 实验设计

## 1. 背景与问题分析

### 1.1 现有 Reasoning Trigger (RT) 的工作方式

当前 RT 实现在 `reasoning_trigger_pilot.py` 中：

- **模型**: Gemini-3-Flash (via Vertex AI)
- **输入**: 每次调用发送 5 帧 (最近 2.5s at 2fps) + 问题文本
- **间隔**: 每 5-10 秒调用一次 VLM (BUCKET_INTERVAL)
- **Cooldown**: 12 秒 (两次触发间最少间隔)
- **Prompt**: 系统 prompt + 用户 prompt，要求 VLM 回答 YES/NO

当前 prompt 模板：
```
System: "You are a proactive AI assistant embedded in smart glasses.
You observe the user's first-person view. The user has a question they need answered.
Your job is to decide: is RIGHT NOW the best moment to answer their question?"

User: 'The user needs to know: "{goal}"
I'm showing you {n_frames} recent frames from the user's perspective (most recent last).
Is the user RIGHT NOW at the moment where the answer to their question is most relevant and useful?
Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene that the question asks about
2. The user appears to be actively engaging with or looking at it
3. Providing the answer NOW would be more useful than waiting
Say NO if the relevant object/action is not visible, or the user is doing something unrelated.
Respond with EXACTLY one word: YES or NO'
```

### 1.2 核心问题

**15 个 RT case 的结果分析**:

| 指标 | RT | Change Detection (CD) | Baseline (gap>0) |
|------|----|-----------------------|-------------------|
| Avg F1 | 0.173 | 0.121 | 0.106 |
| Avg Delta F1 | +0.066 | +0.014 | - |

**RT 的 yes-rate 分布** (15 cases):
- 极端 yes-bias: case 0eb97247e799 → 31/32 (97%), case 48c5b16addef → 21/32 (66%)
- 正常: case ec8c429db0d8 → 1/32 (3%), case 1b83f0628411 → 0/32 (0%)
- 全局 yes-rate: ~141/478 = 29.5% (有 cooldown 限制否则更高)

**问题诊断**:
1. **无时序记忆**: 每帧独立判断，不知道"我刚才已经说了 YES"
2. **无场景变化感知**: 不知道过去 30 秒场景是否发生了有意义的变化
3. **缺乏抑制机制**: 对于持续可见的场景，VLM 会反复说 YES (如 Object Recognition 类型)
4. **特定类型过度触发**: Object Function, Object Recognition, Attribute Perception 类型 yes-rate 特别高

### 1.3 与 Change Detection 的对比

Change Detection 方法比较两帧差异，yes-rate 更低 (21% vs 29.5%)，但 F1 也更低。说明：
- 仅靠视觉变化检测不够，还需要语义理解
- 但 CD 的低 yes-rate 验证了：加入时序对比信息可以有效降低触发频率

---

## 2. 上下文增强方案设计

### 2.1 方案 A：轻量上下文 (判断历史注入)

**核心思路**: 告诉 VLM "你之前做过什么判断"，让它有抑制意识。

**Prompt 模板**:

```python
CONTEXT_A_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to decide: is RIGHT NOW the best moment to answer their question?\n\n"
    "IMPORTANT: You are making this judgment as part of a CONTINUOUS monitoring stream. "
    "You should NOT trigger repeatedly for the same unchanged situation."
)

CONTEXT_A_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

=== YOUR RECENT JUDGMENT HISTORY ===
{judgment_history}
===

Is the user RIGHT NOW at the moment where the answer to their question is most relevant and useful?

Say YES only if ALL of these are true:
1. You can actually SEE the specific object/action/scene that the question asks about
2. The user appears to be actively engaging with or looking at it
3. Providing the answer NOW would be more useful than waiting
4. This is a GENUINELY NEW opportunity compared to your recent judgments — something has changed

Say NO if:
- The relevant object/action is not visible or the user is doing something unrelated
- The scene looks essentially the same as when you last said YES
- You already triggered recently and nothing has meaningfully changed

Respond with EXACTLY one word: YES or NO"""
```

**judgment_history 格式** (最近 N=5 步):
```
- 30s ago: NO (scene was unrelated to the question)
- 20s ago: NO (relevant area not visible)
- 10s ago: YES → TRIGGERED (provided answer)
- 5s ago: NO (same scene as trigger, no new information)
[You last triggered 10 seconds ago]
```

**实现要点**:
- 维护一个 sliding window 记录最近 5 步的 (时间, YES/NO, 是否触发)
- 计算距离上次触发的时间差
- 如果从未触发过，显示 "You have not triggered yet in this session"

**预期效果**:
- 降低连续 YES 现象 (VLM 看到"我刚说了 YES"会更克制)
- 对高 yes-rate case 效果最明显 (如 0eb97247e799 的 97%)
- API 成本: 不变 (仅增加文本，不增加图像)

---

### 2.2 方案 B：中等上下文 (场景变化摘要)

**核心思路**: 在方案 A 基础上，VLM 每次判断后同时输出一句场景描述，作为下次判断的上下文。

**Prompt 模板**:

```python
CONTEXT_B_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view. The user has a question they need answered. "
    "Your job is to: (1) describe what you see, and (2) decide if NOW is the right time to help.\n\n"
    "You are monitoring CONTINUOUSLY. Use the scene history to understand temporal context."
)

CONTEXT_B_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you {n_frames} recent frames from the user's perspective (most recent last).

=== SCENE HISTORY (what happened recently) ===
{scene_history}
===

=== YOUR JUDGMENT HISTORY ===
{judgment_history}
===

Based on the frames AND the scene history above, answer TWO questions:

1. SCENE DESCRIPTION: In one sentence, describe what the user is currently doing/seeing that is DIFFERENT from the previous observation. If nothing changed, say "No significant change."

2. TRIGGER DECISION: Should you provide the answer NOW?
   - YES only if there is a NEW, SPECIFIC visual event relevant to the question (not a continuation of the same scene)
   - NO if the scene is unchanged, or the relevant content is not visible

Format your response as:
SCENE: <one sentence>
DECISION: YES or NO"""
```

**scene_history 格式** (最近 3-4 条):
```
- 40s ago: "User is walking through a kitchen, looking at countertop items"
- 30s ago: "User picked up a red cutting board and placed it on the counter"
- 20s ago: "User is chopping vegetables on the cutting board"
- 10s ago: "No significant change. User continues chopping."
```

**实现要点**:
- 第一步输出格式解析: 正则提取 `SCENE:` 和 `DECISION:` 字段
- scene_history 作为 sliding window，保留最近 4 条
- 如果 VLM 返回格式不符，fallback 到简单 YES/NO 检测

**预期效果**:
- 对"渐变场景"有更好的抑制 (VLM 看到"No significant change"会更倾向 NO)
- 对 Object Function、Object Recognition 效果好 (这些类型场景变化缓慢)
- API 成本: 增加约 50% (输出 token 增多，但无额外图像)

---

### 2.3 方案 C：丰富上下文 (多帧对比 + 历史)

**核心思路**: 发送当前帧 + 历史关键帧，让 VLM 直接看到时间变化。结合方案 A+B 的文本上下文。

**Prompt 模板**:

```python
CONTEXT_C_SYSTEM = (
    "You are a proactive AI assistant embedded in smart glasses. "
    "You observe the user's first-person view over time. "
    "I'm showing you frames from MULTIPLE time points so you can see how the scene has evolved. "
    "Your job is to decide if RIGHT NOW is the best moment to answer the user's question."
)

CONTEXT_C_USER_TEMPLATE = """The user needs to know:
"{goal}"

I'm showing you frames from multiple time points:
{frame_description}

=== YOUR JUDGMENT HISTORY ===
{judgment_history}
===

Compare the CURRENT frames with the EARLIER frames. Has a MEANINGFUL CHANGE occurred that makes NOW the right time to answer?

Say YES only if:
1. Something NEW and RELEVANT has appeared or changed compared to the earlier frames
2. The user is actively engaging with the relevant object/action/scene
3. This is not just a continuation of the same activity shown in earlier frames

Say NO if:
- The scene is essentially the same across all frames
- No new relevant information has appeared
- The change is trivial (slight camera movement, lighting change)

Respond with EXACTLY one word: YES or NO"""
```

**frame_description 格式**:
```
- Frame 1-2: from 30 seconds ago (reference point)
- Frame 3-4: from 15 seconds ago
- Frame 5-7: current view (most recent)
```

**帧选择策略**:
- 总共发送 7 帧 (或受限于 API 限制)
- 历史帧: 取 ~30s 前的 2 帧 + ~15s 前的 2 帧
- 当前帧: 最近 3 帧 (1.5s 窗口)
- 如果上次触发在 15s 内，用触发时刻的帧作为历史参考

**实现要点**:
- 需要缓存历史帧 (或从视频中重新提取)
- 帧数增加导致 API 输入 token 增多
- 可以复用 `change_detection_trigger_pilot.py` 的帧提取逻辑

**预期效果**:
- 最强的时序感知能力 (VLM 直接看到前后对比)
- 融合 Change Detection 的优点 (视觉对比) 和 Reasoning Trigger 的优点 (语义推理)
- API 成本: 增加约 40% (多 2-4 帧图像输入)

---

## 3. 实验计划

### 3.1 Case 选择

从 15 个 RT case 中选择 5 个代表性 case，覆盖不同失败模式：

| # | case_id | task_type | RT F1 | BL F1 | yes_rate | 选择理由 |
|---|---------|-----------|-------|-------|----------|----------|
| 1 | `0eb97247e799` | Object Recognition | 0.250 | 0.077 | 97% (31/32) | **极端 yes-bias**，RT 触发 16 次，FP=13。上下文应能大幅降低 |
| 2 | `ec8c429db0d8` | Text-Rich Understanding | 0.667 | 0.000 | 3% (1/32) | **RT 表现最好**的 case，作为正面参照 (验证不退化) |
| 3 | `4a51f9ebaa22` | Object Function | 0.133 | 0.000 | 47% (15/32) | **中等 yes-bias**，Object Function 是隐式推理类型，场景变化摘要可能有效 |
| 4 | `85a99403e624` | Object State Change Recognition | 0.500 | 0.118 | 3% (1/32) | RT 精准触发 1 次且命中 TP，但漏了 2 个 GT。测试上下文能否提高 recall |
| 5 | `2098c3e8d904` | Action Recognition | 0.125 | 0.235 | 31% (10/32) | **RT 比 baseline 差**，yes-bias 导致 FP=7。Action Recognition 对时序最敏感 |

**选择原则**:
- 覆盖 yes-bias 频谱: 极高 (97%), 中等 (47%, 31%), 低 (3%)
- 覆盖 RT vs BL 对比: RT 好 (ec8c, 85a9), RT 差 (2098), RT 中性 (0eb9, 4a51)
- 覆盖任务类型多样性: Object Recog, Text-Rich, Object Function, State Change, Action Recog

### 3.2 对比实验矩阵

| 方法 | 描述 | 图像输入 | 文本上下文 | 预估 API 调用 |
|------|------|---------|-----------|--------------|
| RT-Baseline | 原始 RT (无上下文) | 5帧/当前 | 无 | ~32/case |
| 方案 A | 判断历史注入 | 5帧/当前 | 判断历史 (5步) | ~32/case |
| 方案 B | 场景变化摘要 | 5帧/当前 | 判断历史 + 场景摘要 | ~32/case |
| 方案 C | 多帧对比 + 历史 | 7帧 (含历史帧) | 判断历史 | ~32/case |

### 3.3 评估指标

对每个 case + 方法组合，计算：
1. **ESTP-F1**: 主要指标 (anticipation=1s, latency=2s)
2. **yes-rate**: VLM 回答 YES 的比例 (越低越好，在 recall 不降的前提下)
3. **FP 数量**: 误触发次数
4. **Trigger 次数**: 实际触发次数 (经 cooldown 过滤后)
5. **Precision / Recall**: 分解 F1

### 3.4 预期假设

| 方案 | 假设 | 验证标准 |
|------|------|---------|
| A | yes-rate 下降 20-40%，F1 持平或提升 | yes-rate < 0.7 * RT_yes_rate |
| B | 对渐变场景的 FP 大幅降低 | Object Function 和 Object Recog 的 FP 减半 |
| C | 综合 F1 最高，但代价是多帧开销 | F1 > RT 且 F1 > A 且 F1 > B |
| 对照 | Text-Rich case 不退化 | ec8c429db0d8 的 F1 >= 0.5 |

### 3.5 运行时间与成本估算

**单 case 运行参数**:
- Duration: ~300s (5 min clip)
- Interval: 10s → ~32 buckets/case
- Gemini-3-Flash API 调用: ~32 次/case

**5 cases x 4 methods = 20 runs**:

| 方法 | API 调用/case | 总调用 | 输入 token/调用 | 输出 token/调用 | 预估时间/case |
|------|-------------|--------|----------------|----------------|--------------|
| RT-BL | 32 | 160 | ~2000 (5帧+prompt) | ~5 | ~2 min |
| A | 32 | 160 | ~2200 (5帧+prompt+history) | ~5 | ~2 min |
| B | 32 | 160 | ~2400 (5帧+prompt+history+scene) | ~50 | ~2.5 min |
| C | 32 | 160 | ~3000 (7帧+prompt+history) | ~5 | ~2.5 min |

**总计**:
- API 调用: ~640 次
- 预估时间: ~40-50 分钟 (含 rate-limit 等待)
- Gemini-3-Flash 成本: 极低 (~$0.05 总计，Flash 模型很便宜)

---

## 4. 实验流程

### 4.1 实现步骤

1. **创建 `context_enhanced_rt.py`**: 基于 `reasoning_trigger_pilot.py` 扩展
   - 添加 `ContextState` 类管理判断历史和场景摘要
   - 实现 A/B/C 三种 prompt 构建函数
   - 方案 B 需要额外解析逻辑 (提取 SCENE 和 DECISION)

2. **数据准备**: 使用已有 checkpoint.jsonl 的 case 元数据，匹配 ESTP dataset

3. **执行顺序**:
   - 先跑 RT-Baseline (复现，确认与已有结果一致)
   - 并行/串行跑 A, B, C
   - 每个 case 跑完后保存 checkpoint

4. **分析脚本**: 汇总结果，生成对比表格和可视化

### 4.2 实现关键类

```python
class ContextState:
    """管理上下文状态，跨步骤维持记忆"""

    def __init__(self, max_history=5, max_scenes=4):
        self.judgments = []       # [(t, decision_bool, triggered_bool)]
        self.scene_descriptions = []  # [(t, description_str)]
        self.last_trigger_time = -999.0
        self.max_history = max_history
        self.max_scenes = max_scenes

    def add_judgment(self, t, decision, triggered):
        self.judgments.append((t, decision, triggered))
        if len(self.judgments) > self.max_history:
            self.judgments.pop(0)
        if triggered:
            self.last_trigger_time = t

    def add_scene(self, t, description):
        self.scene_descriptions.append((t, description))
        if len(self.scene_descriptions) > self.max_scenes:
            self.scene_descriptions.pop(0)

    def format_judgment_history(self, current_t):
        if not self.judgments:
            return "No previous judgments yet. This is your first observation."
        lines = []
        for t, dec, trig in self.judgments:
            ago = current_t - t
            if trig:
                lines.append(f"- {ago:.0f}s ago: YES -> TRIGGERED (provided answer)")
            elif dec:
                lines.append(f"- {ago:.0f}s ago: YES (but in cooldown, not triggered)")
            else:
                lines.append(f"- {ago:.0f}s ago: NO")
        since_trigger = current_t - self.last_trigger_time
        if self.last_trigger_time > 0:
            lines.append(f"[You last triggered {since_trigger:.0f} seconds ago]")
        else:
            lines.append("[You have not triggered yet in this session]")
        return "\n".join(lines)

    def format_scene_history(self, current_t):
        if not self.scene_descriptions:
            return "No scene observations yet."
        lines = []
        for t, desc in self.scene_descriptions:
            ago = current_t - t
            lines.append(f"- {ago:.0f}s ago: \"{desc}\"")
        return "\n".join(lines)
```

### 4.3 方案 B 的输出解析

```python
def parse_scene_and_decision(raw_text):
    """解析方案 B 的两段式输出"""
    scene = ""
    decision = False

    for line in raw_text.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCENE:"):
            scene = line[6:].strip()
        elif line.upper().startswith("DECISION:"):
            dec_text = line[9:].strip().lower()
            decision = dec_text.startswith("yes")

    # Fallback: 如果没有结构化输出，用简单检测
    if not scene and not any(line.upper().startswith("SCENE:") for line in raw_text.split("\n")):
        scene = raw_text[:100]
        decision = "yes" in raw_text.lower().split("\n")[0] if raw_text else False

    return scene, decision
```

### 4.4 方案 C 的帧选择

```python
def select_context_frames(video_path, t_current, clip_start, last_trigger_time,
                          n_current=3, n_history=4, fps_hint=2.0):
    """为方案 C 选择历史帧 + 当前帧"""
    frames = []
    descriptions = []

    # 历史帧: 30s 前 2 帧 + 15s 前 2 帧
    history_times = []
    for delta in [30.0, 28.0, 15.0, 13.0]:
        t = max(clip_start, t_current - delta)
        history_times.append(t)

    # 如果上次触发在 30s 内，替换 30s 帧为触发时刻帧
    if last_trigger_time > 0 and (t_current - last_trigger_time) < 30:
        history_times[0] = last_trigger_time
        history_times[1] = last_trigger_time + 0.5

    # 当前帧: 最近 n_current 帧
    step = 1.0 / fps_hint
    current_times = [max(clip_start, t_current - step * (n_current - 1 - i))
                     for i in range(n_current)]

    all_times = history_times + current_times
    # 提取并返回帧 + 时间描述
    ...
```

---

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| VLM 忽略判断历史文本 | 方案 A 无效 | 对比 yes-rate 变化; 如果无变化则加粗/重复关键信息 |
| 方案 B 输出格式不稳定 | 解析失败 | Fallback 到简单 YES/NO 检测; 设 temperature=0 |
| 方案 C 多帧增加 latency | 不符合实时要求 | 监控每次调用耗时; 必要时减少帧数 |
| 所有方案 recall 下降 | 过度抑制 | 设定 recall 下限; 如果某方案 recall < BL 的 50%, 标记为失败 |
| 5 个 case 样本太小 | 统计意义不足 | 视初步结果决定是否扩展到 15 cases |

---

## 6. 成功标准

实验完成后，根据以下标准评估：

1. **PRIMARY**: 至少一种方案在 5 cases 上的 avg F1 > RT-Baseline 的 avg F1
2. **SECONDARY**: 高 yes-bias cases (0eb9, 4a51, 2098) 的 yes-rate 降低 > 30%
3. **GUARDRAIL**: Text-Rich case (ec8c) 的 F1 不退化超过 0.1
4. **BONUS**: 任一方案在所有 5 cases 上均不差于 RT-Baseline (无退化)

如果 PRIMARY 达成，进入扩展实验 (15 cases 全量评估)。

---

## 7. 与现有工作的关系

本实验与 Phase I-II 的 Adaptive-D (logprob 阈值) 是**互补**的：
- Adaptive-D 用 VLM 内部 logprob gap 做硬过滤 → 减少 FP
- 上下文增强 RT 用时序信息改善 VLM 的判断质量 → 从源头减少 YES-bias

**未来融合方向**: 如果上下文增强有效，可以将其与 Adaptive-D 组合:
- 先用上下文增强 RT 降低 yes-rate
- 再用 logprob gap 做第二层过滤
- 期望 F1 > 任何单一方法
