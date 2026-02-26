# Streaming Video → GUI 系统研究问题

> **项目**: Smart Glasses Adaptive UI System
> **日期**: 2026-02-11
> **状态**: 研究规划阶段

---

## 背景与动机

本项目旨在构建一个从第一人称视频流到智能 GUI 生成的端到端系统。当前系统为"标注驱动、单 UI 输出"的离线 pipeline，我们希望将其改造为能够处理复杂程序性任务（如烹饪）的流式视频理解与 UI 生成系统。

### 系统总体架构（两阶段）

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Stage 1: Streaming Video Understanding             │
│                                                                     │
│  视频输入 → 帧提取 → Multimodal Brain (VLM)                         │
│                         ├── 场景理解 (当前在做什么)                   │
│                         ├── 进度追踪 (进行到哪一步)                   │
│                         ├── 干预时机判断 (该不该说话)                  │
│                         └── 干预内容决策 (说什么)                     │
│                                                                     │
│  输出: StreamingContext (时间戳 + 触发类型 + 场景 + 内容需求)         │
├─────────────────────────────────────────────────────────────────────┤
│                  Stage 2: Generative UI                             │
│                                                                     │
│  StreamingContext → Code/Generation Model → A2UI JSON               │
│                         ├── 组件类型选择 (Card? Badge? Timeline?)    │
│                         ├── 内容填充 (标题、正文、操作按钮)           │
│                         └── 布局定位 (基于眼动 + 场景锚定)            │
│                                                                     │
│  输出: 带时间戳的 A2UI 组件序列 → 渲染到视频上                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 目标场景

**烹饪 (Cooking)** — 作为复杂程序性任务的代表场景：
- 多步骤、有明确顺序依赖（先切菜 → 再热锅 → 再下油）
- 需要长时间上下文理解（记住 20 分钟前放了什么调料）
- 存在错误纠正机会（火太大、忘放盐）
- 用户注意力频繁切换（看菜谱 → 看锅 → 拿调料 → 回头看锅）
- 物体在固定位置反复出现（锅、切菜板、调料架）

---

## 核心相关工作

### 1. ProAssist — 主动对话生成基准

**论文**: Proactive Assistant Dialogue Generation from Streaming Egocentric Videos (arXiv:2506.05904)
**代码**: https://github.com/pro-assist/ProAssist (已克隆至 `/home/v-tangxin/GUI/temp/ProAssist`)

**核心贡献**:
- 提出了从流式第一人称视频生成主动式助手对话的任务
- 构建了 ProAssist 数据集：30,135 个对话，覆盖 479 小时视频（烹饪、物体操作、组装、实验室）
- 设计了基于 LLaMA-3.1-8B 的 ProAct 模型，支持逐帧流式推理

**关键技术洞察**:

| 机制 | 实现方式 | 对我们的启发 |
|------|---------|-------------|
| **When to Talk (W2T)** | Binary Decision Head — 每帧预测 `not_talk_prob`，超过阈值则沉默 | 干预时机判断的核心机制 |
| **KV Cache 管理** | 三种策略：DROP_ALL / DROP_MIDDLE / SUMMARIZE_AND_DROP | 记忆管理的直接参考 |
| **Progress Summary** | 对话进行中生成任务进度摘要，用于上下文压缩后的恢复 | 全局任务追踪的实现方式 |
| **对话模拟** | LLM-based 分段生成，每段保留历史对话上下文 | 数据生成 pipeline 的参考 |

**ProAct 模型流式推理核心流程** (源自 `stream_inference.py`):
```
逐帧处理循环:
  for each frame:
    1. joint_embed(image + text tokens) → input_embeds
    2. fast_greedy_generate(input_embeds, past_key_values)
       ├── 计算 not_talk_prob (W2T)
       ├── 若 prob > threshold → 输出 silence token (不说话)
       └── 若 prob < threshold → 生成对话文本
    3. manage_kv_cache() → 上下文长度管理
       ├── DROP_ALL: 清空全部缓存
       ├── DROP_MIDDLE: 保留 init + 最近 512 tokens
       └── SUMMARIZE_AND_DROP: 生成进度摘要 → 替换上下文
```

### 2. StreamGaze — 眼动引导的流式视频理解基准

**论文**: StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos (arXiv:2512.01707)
**代码**: https://github.com/daeunni/StreamGaze (已克隆至 `/home/v-tangxin/GUI/temp/StreamGaze`)
**详细分析**: 见 `/home/v-tangxin/GUI/docs/streamgaze.md`

**关键技术洞察**:
- **60 秒滑动窗口**: `[t-60, t]` 窗口模拟流式处理
- **I-VT 注视点提取**: 从眼动数据中提取注视段 (Fixation)，用于确定用户关注点
- **Proactive Tasks**: 注视触发提醒 (GTA) + 物体出现提醒 (OAA)
- **多帧传递**: API 模型用 16 帧均匀采样，开源模型用动态 FPS

### 3. StreamBridge — 离线 Video-LLM 转流式 (NeurIPS 2025)

**论文**: StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant (arXiv:2505.05467)
**代码**: https://github.com/apple/ml-streambridge (已克隆至 `/home/v-tangxin/GUI/temp/ml-streambridge`)

**核心贡献**:
- **Memory Buffer + Round-Decayed Compression**: 早期帧 token 压缩，近期帧保留完整 — 解决长上下文问题
- **Decoupled Activation Model**: 轻量级独立模型判断"何时主动说话"，与主 Video-LLM 解耦
- **Stream-IT 数据集**: 交错式视频-文本序列，支持流式训练

#### 3.1 双模型解耦架构 (代码分析)

StreamBridge 的核心创新在于将"何时说话"与"说什么"拆分为两个独立模型：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    StreamBridge 推理流程                             │
│                                                                     │
│  Frame[i] ─┬─► Activate_VideoLLM (0.5B)                           │
│             │    ├── SiglipVisionModel → 729 tokens                │
│             │    ├── apply_pooling → 196 tokens                     │
│             │    ├── pooling_factor=4 → 16 tokens                  │
│             │    ├── + activation_embed → 17 tokens/frame           │
│             │    ├── 拼接 past_embeds → LLM forward                 │
│             │    └── score_head(hidden[-1]) → softmax → decision    │
│             │         if prob[1] >= threshold → TRIGGER             │
│             │                                                       │
│             └─► Streaming Model (7B, Qwen2-VL/LLaVA-OV/Oryx)      │
│                  ├── receive_one_frame() → encode + cache           │
│                  ├── [等待 activation decision]                     │
│                  │    if TRIGGER:                                    │
│                  │      response() → flush cache → generate text    │
│                  │      compressor() 如果 tokens > max_image_token  │
│                  └── 更新 past_embeds + modality_indicators         │
│                                                                     │
│  每 128 帧重置 Activate Model 的 past_embeds (避免累积过长)          │
└─────────────────────────────────────────────────────────────────────┘
```

源自 `demo.py` 的核心循环：
```python
for i in range(frame_pixel_values.shape[0]):
    # 1. 小模型判断是否触发
    probs, decision = activate_model.decide_response({"pixel_values": frame[i]})
    # 2. 大模型接收帧（缓存到 cache 中）
    streaming_model.receive_one_frame(frame[i])
    # 3. 如果触发，大模型生成回复
    if decision == 1:
        answer = streaming_model.response()
        # 将回复反馈给小模型（维持对话上下文）
        activate_model.decide_response({"text_inputs": [answer]})
    # 4. 定期重置小模型上下文
    if i - last_activate_window == 128:
        activate_model.past_embeds = None
        activate_model.decide_response({"text_inputs": [question]})
```

#### 3.2 Activation Model 详细机制 (源自 `activate_videollm.py`)

**模型组成**:
- **Vision Tower**: SiglipVisionModel (冻结) → 提取 729 维 patch tokens
- **MM Projector**: LlavaOnevisionMultiModalProjector (2-layer MLP with GeLU)
- **Language Model**: Qwen2Model 0.5B + LoRA (r=128, alpha=256)
- **Activation Embed**: `nn.Embedding(1, dim)` — 初始化为 `image_newline` 权重
- **Score Head**: `nn.Linear(dim, 2)` — 二分类：[不说话, 说话]

**推理逻辑** (`decide_response()`):
```python
# 输入图像帧时:
image_embeds = encode(pixel_values)       # [1, 16, dim]
activate_embeds = activation_embed(...)    # [1, 1, dim]
image_embeds = cat([image_embeds, activate_embeds])  # [1, 17, dim]
inputs_embeds = cat([past_embeds, image_embeds])     # 拼接历史

outputs = language_model(inputs_embeds=inputs_embeds)
logits = score_head(outputs.hidden_states[:, -1, :])  # 取最后 token
probs = softmax(logits)  # [prob_no_talk, prob_talk]
decision = 1 if probs[1] >= threshold else 0
```

**训练标签生成** (`forward()` 中的 `create_custom_tensor()`):
- 在每个 round 中，随机选择一个分割点 `split_idx`
- `split_idx` 之前的帧标记为 0 (不说话)，之后的帧标记为 1 (说话)
- 只在每帧的最后一个 token（activation_embed 位置）计算 loss
- `activate_ratio=0.5` 控制正负样本比例

#### 3.3 Round-Decayed Compression (源自 `compressor()` 方法)

StreamBridge 的记忆压缩策略是**基于交互轮次的渐进式视觉 token 压缩**：

```
压缩前 (tokens > max_image_token=32K):
┌───────┬──────────────────┬─────────┬──────────────────┬─────────┬──────────┐
│ sys   │ round1 img (N帧) │ round1  │ round2 img (M帧) │ round2  │ round3   │
│ prompt│ 50×N tokens      │ text    │ 50×M tokens      │ text    │ img+text │
└───────┴──────────────────┴─────────┴──────────────────┴─────────┴──────────┘

压缩后 (从最早的 round 开始):
┌───────┬────────────┬─────────┬──────────┬─────────┬──────────┐
│ sys   │ round1 img │ round1  │ round2   │ round2  │ round3   │
│ prompt│ 50 tokens  │ text    │ img(部分 │ text    │ img+text │
│       │ (mean池化) │ (完整)  │ 压缩)    │ (完整)  │ (完整)   │
└───────┴────────────┴─────────┴──────────┴─────────┴──────────┘
```

**关键实现细节**:

| 特性 | 实现方式 |
|------|---------|
| **Modality Indicator** | `list[int]`，0=文本 token，1=图像 token，精确追踪每个位置 |
| **压缩顺序** | 从 round 0 (最老) 开始向后，直到满足 token 预算 |
| **压缩方法** | Mean-pooling：将 N 帧的 token 在帧维度上取平均，压缩为 1 帧 |
| **最小压缩单位** | 2 帧以上才压缩 (`current_image_embeds.shape[1] >= token_per_frame*2`) |
| **部分压缩** | 如果一个 round 的帧太多，只压缩前面的帧，保留后面的 |
| **文本保护** | 文本 token 永远不压缩，只压缩图像 token |
| **Token 预算** | `max_image_token = 32*1024` (默认 32K tokens) |

**具体压缩算法**:
```python
# 1. 计算需要删除的 token 数
operation_num = (total_tokens - max_len) / token_per_frame + 2
compress_target_num = token_per_frame * operation_num

# 2. 分离文本和图像 token (按 modality_indicators)
text_groups, image_groups = ungroup_image_and_text(embeds, indicators)

# 3. 按 round 从老到新遍历
for round_idx in range(len(image_groups)):
    if compress_target_num > 0 and image_tokens >= 2 * token_per_frame:
        if image_tokens <= compress_target_num + token_per_frame:
            # 全部压缩为 1 帧: [N_frames, tokens_per_frame, dim] → mean → [1, tokens_per_frame, dim]
            compressed = image_embeds.reshape(N, tokens_per_frame, dim).mean(dim=0)
        else:
            # 部分压缩: 前面的帧压缩，后面的帧保留
            pre = compress_first_K_frames_to_1()
            post = keep_remaining_frames()
            compressed = cat([pre, post])
```

#### 3.4 与 ProAssist 的关键差异对比

| 维度 | ProAssist | StreamBridge |
|------|-----------|--------------|
| **模型架构** | 单模型 (LLaMA-3.1-8B + W2T head) | 双模型 (0.5B 激活 + 7B 生成) |
| **"何时说话"** | Binary Decision Head 在第一个 token 处判断 `not_talk_prob` | 独立 Activation Model，每帧 17 tokens → score_head 二分类 |
| **上下文管理** | KV Cache 级别 (DROP_ALL/MIDDLE/SUMMARIZE) | Embedding 级别 (mean-pooling 视觉 tokens) |
| **文本上下文** | DROP_MIDDLE 可能丢失中间对话 | 文本 token 永远保留，只压缩图像 token |
| **进度记忆** | SUMMARIZE_AND_DROP: LLM 生成进度摘要替换上下文 | 无显式摘要，通过保留文本 token 隐式保留 |
| **Vision Encoder** | LLaVA-style joint_embed (图像 token 插入文本序列) | SigLIP for Activation / 各模型原生 encoder for Generation |
| **帧表示** | 完整 patch tokens | 重度池化 (729 → 196 → 16 tokens/帧 for activation) |
| **训练** | 端到端训练 W2T + 对话生成 | Stage 1: 标准视频 QA; Stage 2: Stream-IT 微调 |
| **周期性重置** | 无（依赖 KV Cache 管理） | 每 128 帧重置 Activation Model 上下文 |

#### 3.5 对我们 GUI 系统的借鉴

**直接可借鉴**:

1. **双模型解耦架构 → 我们的 TriggerDecider + GenerativeUI**
   - StreamBridge 的 Activate Model (0.5B) 对应我们的 TriggerDecider (小模型判断是否出 UI)
   - StreamBridge 的 Streaming Model (7B) 对应我们的 A2UI Generator (生成 UI JSON)
   - 优势：小模型延迟低、资源占用少，可以每帧都运行；大模型只在触发时调用

2. **Activation Embed Token → 我们的 Trigger Probe**
   - 在每帧视觉 token 后附加一个 learnable 的 `activation_embed`
   - 用这个特殊 token 的 hidden state 做二分类决策
   - 比 ProAssist 的 W2T head 更优雅：不需要在生成模型上加 head，完全独立

3. **Modality-Aware Compression → 我们的 Memory Manager**
   - 只压缩视觉 token，保留文本 token — 确保语言上下文完整性
   - 对我们的 RQ2 (记忆管理) 非常有价值：可以保留所有 UI 决策历史 (文本)，只压缩旧的帧特征
   - 用 `modality_indicators` 追踪每个位置的类型 — 简单有效

4. **Round-Based 压缩粒度 → 我们的 UI Session 压缩**
   - StreamBridge 按"交互轮次"组织压缩
   - 我们可以按"UI Session"（一次完整的 UI 展示 → 隐藏周期）组织压缩
   - 老的 session 压缩得更狠，当前 session 保持完整

**需要改进/不同的地方**:

1. **StreamBridge 缺少进度摘要** — 纯 mean-pooling 压缩丢失了语义信息。ProAssist 的 SUMMARIZE_AND_DROP 在这一点上更优秀。我们应该结合两者：压缩视觉 token + 保留/更新进度摘要

2. **StreamBridge 的 128 帧硬性重置** — 粗暴的周期性重置可能丢失关键上下文。我们需要更智能的重置策略（如基于场景切换的自适应重置）

3. **StreamBridge 没有空间感知** — 压缩纯粹是时间维度的。对于 UI 定位 (RQ3)，我们还需要空间维度的信息保留（如物体位置、眼动区域）

4. **StreamBridge 的压缩是被动的** — 只在 token 超限时才压缩。我们应该设计主动压缩策略：每 N 帧评估一次，基于场景变化动态决定压缩时机和程度

---

## Research Question 1: 主动干预时机 (When to Intervene)

### 1.1 问题定义

> **在流式第一人称视频中，系统何时应当主动介入并生成 UI？如何区分 reactive（错误纠正）和 anticipatory（意图预判）两种干预模式？**

这是整个系统最核心的研究问题。当前系统依赖离线标注（`is_good_time_for_recommendation: true`）来确定干预时机，但真实场景中不存在这样的标注。

### 1.2 两种干预模式

#### Reactive 干预（错误纠正型）

**定义**: 检测到用户行为错误后，立即介入纠正。

**触发条件**:
- 用户执行了错误操作（如：应该切丁却切成了片）
- 用户遗漏了关键步骤（如：忘记预热烤箱）
- 用户使用了错误的工具/材料（如：用金属铲刮不粘锅）

**技术挑战**:
- 需要"知道正确答案"才能判断"什么是错误" → 依赖任务知识库（recipe / manual）
- 实时性要求高 — 错误发生后需要立即检测，延迟几秒就来不及了
- 误报的代价很高 — 如果用户做的是对的却被纠正，体验极差

**ProAssist 的做法**: ProAssist 在对话标签中明确区分了 `correction` 类型:
```
[time] Assistant: ... [initiative|correction]
```
同时在 Assembly101 数据集中利用了 mistake 标注来生成纠错对话。

#### Anticipatory 干预（意图预判型）

**定义**: 基于历史行为和任务理解，在用户需要之前主动提供信息。

**触发条件**:
- 用户即将进入下一步骤（如：切完菜了，该热锅了）
- 用户可能忘记某个步骤（如：长时间没有检查烤箱温度）
- 用户的注视模式暗示需要帮助（如：反复在几个物体间扫视 → 可能在找东西）

**技术挑战**:
- 需要理解"用户在做什么"以及"接下来应该做什么" → 任务进度追踪
- 干预的时机很微妙 — 太早会打断用户，太晚就失去价值
- 需要 long-context understanding — 理解 10 分钟甚至 30 分钟的操作历史

**ProAssist 的做法**: ProAssist 对话中大部分助手 turn 被标记为 `initiative|instruction`，模型通过 W2T (When to Talk) 机制学习在合适时机主动说话。

### 1.3 数据来源与采集策略

#### 现有数据

1. **本项目已有数据** — proactive 离线标注:
   ```json
   {
     "timestamp_start": 1000,
     "timestamp_end": 1100,
     "is_good_time_for_recommendation": true,
     "recommendations": [
       {"rank": 1, "intent": "task_assistance", "content": "需要我帮你记录这个商品信息吗？", ...}
     ]
   }
   ```
   → 包含"时机"和"内容"两部分标注，但仅限于日常活动场景。

2. **ProAssist 数据集** — 30K+ 对话:
   - 覆盖 cooking (EpicKitchens, EgoExoLearn), assembly (Assembly101), AR 辅助 (HoloAssist) 等场景
   - 每个 assistant turn 都有精确时间戳和意图标签 (`instruction / correction / info_sharing / feedback`)
   - 已有 `progress_summary` 字段记录任务进度

3. **StreamGaze 数据集** — 8.5K QA 对:
   - 10 种任务类型，包括 Proactive (GTA, OAA)
   - 有眼动数据，可以关联注视行为与干预时机

#### 数据扩展方案对比

| 方案 | 优势 | 劣势 | 适用阶段 |
|------|------|------|---------|
| **A. 使用 ProAssist 已有数据** | 数据量大 (30K)、标注质量经 LLM 验证、覆盖多场景 | 对话形式 (非 GUI)、缺少眼动数据、缺少 UI 组件标注 | 快速验证 |
| **B. Gemini 批量生成 + 人工筛选** | 灵活定制 GUI 场景、可包含 A2UI 组件信息、成本可控 | 需要设计 prompt 模板、生成质量不稳定、人工成本 | 中期构建 |
| **C. 真实用户标注** | 最真实、最权威 | 成本最高、规模有限 | 后期验证 |

**建议的渐进策略**:
1. **Phase 1 (快速验证)**: 直接使用 ProAssist 的 cooking 子集 (EpicKitchens + EgoExoLearn)，将对话时机转换为 UI 生成触发点
2. **Phase 2 (数据增强)**: 用 Gemini 基于 ProAssist 的时间戳 + 我们的 A2UI 组件规范，生成 UI-specific 的标注
3. **Phase 3 (真实验证)**: 在真实 cooking 场景中收集用户反馈，验证时机准确性

### 1.4 End-to-End System 设计

> 真实场景只有 video，没有标注数据。我们要做出什么样的系统？

#### 离线模式（预生成）

```
输入: 一段完整的烹饪视频 (30 min)
处理流程:
  1. 任务识别: VLM 观看全视频 → "这是一个做番茄炒蛋的视频"
  2. 步骤拆解: VLM 根据视频内容拆解步骤 → [打蛋, 切番茄, 热锅, 炒蛋, 炒番茄, 合炒, 调味, 出锅]
  3. 关键时间点检测: 对每一步标注 start_time / end_time
  4. GUI 生成: 对每个步骤生成对应的 A2UI 组件
     - 步骤过渡时: Card("下一步: 热锅倒油")
     - 关键提醒: Badge("记得加盐!")
     - 全局进度: ProgressBar(step=3, total=8)
输出: ui_timeline.json — 带时间戳的 UI 序列
```

#### 在线模式（边看边生成）

参考 StreamBridge 的双模型解耦架构，在线模式采用**轻量 Trigger Model + 重量 Generation Model**的分离设计：

```
输入: 实时视频流 (逐帧到达)
处理流程 (StreamBridge-style 双模型):

  ┌─ Trigger Model (轻量, 每帧都跑) ─────────────────────┐
  │  1. 每帧: encode_frame → visual_tokens + trigger_probe │
  │  2. 拼接历史 embeddings → LLM forward                  │
  │  3. score_head(last_token) → trigger_prob               │
  │  4. if trigger_prob >= threshold → TRIGGER              │
  │  5. 定期压缩/重置上下文 (避免累积过长)                  │
  └──────────────────────────────────────────────────────────┘
                     ↓ TRIGGER
  ┌─ Generation Model (重量, 仅触发时调用) ────────────────┐
  │  1. 聚合缓存帧 → 场景理解 + 任务进度                    │
  │  2. 根据 StreamingContext → A2UI JSON                    │
  │  3. 应用 compressor() 如果 token 超限                   │
  │  4. 将响应反馈给 Trigger Model 维持上下文               │
  └──────────────────────────────────────────────────────────┘
                     ↓
  前端渲染 → 推送到 AR 眼镜
```

**与 StreamBridge 的对应关系**:
- StreamBridge 的 `Activate_VideoLLM (0.5B)` ↔ 我们的 Trigger Model
- StreamBridge 的 `Streaming Model (7B)` ↔ 我们的 Generation Model
- StreamBridge 的 `activation_embed` token ↔ 我们的 `trigger_probe` token
- StreamBridge 的 `compressor()` ↔ 我们的 Memory Manager

#### Base Model 候选

| 模型 | 用途 | 优势 | 限制 |
|------|------|------|------|
| **Gemini 2.5 Pro** | Stage 1 Brain (场景理解 + 时机判断) | 原生长视频理解、多模态能力强 | API 成本、延迟、不适合每帧调用 |
| **GPT-4o** | Stage 1 Brain (替代方案) | 多图输入、指令跟随能力强 | 不支持原生视频输入 |
| **ProAct (LLaMA-3.1-8B)** | Stage 1 Brain (可微调) | 已有 W2T 机制、开源可控 | 单模型耦合 trigger+generation |
| **StreamBridge-style (0.5B+7B)** | Trigger (0.5B) + Generation (7B) | 解耦设计、trigger 延迟低 | 需要两个模型、训练流程更复杂 |
| **Claude Sonnet/Opus** | Stage 2 生成 (A2UI JSON) | 代码生成能力强、JSON 结构化输出好 | 不支持视频输入 |

**推荐方案**:
- **离线模式**: Gemini 2.5 Pro 作为 Brain (Stage 1)，Claude/GPT-4o 作为 Generator (Stage 2)
- **在线模式**: 采用 StreamBridge-style 双模型解耦 — 轻量 Trigger Model (0.5B~3B) 每帧判断时机，重量 Generation Model 仅在触发时生成 UI
- 两种模式的核心差异在于 trigger 机制：离线模式可以用 VLM 全视频分析找时机，在线模式必须用轻量模型逐帧判断

### 1.5 评估指标

参考 ProAssist 的评估体系:

| 指标 | 定义 | 衡量什么 |
|------|------|---------|
| **Missing Rate** | 该干预但没干预的比例 | 系统的召回能力 |
| **Redundant Rate** | 不该干预但干预了的比例 | 系统的精确度 |
| **Timing Accuracy** | 预测时机与 GT 时机的时间差 (秒) | 干预的及时性 |
| **Jaccard Index** | matched / (matched + missed + redundant) | 综合时机准确性 |
| **Content Semantic Score** | 生成内容与 GT 的语义相似度 (STS) | 内容质量 |
| **F1 (时机 × 内容)** | Precision × Recall 的调和 | 综合表现 |

### 1.6 开放问题

1. **W2T 机制能否迁移到 GUI 场景？** ProAssist 的 W2T 是在文本对话场景下训练的，GUI 的"干预"形式不同（弹出 Card vs 说话），需要重新定义什么算"干预"
2. **Reactive vs Anticipatory 的比例如何？** 在真实烹饪场景中，大部分干预应该是 anticipatory (主动提供下一步指导)，还是 reactive (纠错)？这决定了模型训练的重点
3. **眼动信号的价值有多大？** StreamGaze 证明了眼动对理解帮助很大，但我们的场景中眼动数据可能不可用（只有视频），如何用纯视觉替代？

---

## Research Question 2: 流式视频上下文记忆管理 (Streaming Context & Memory)

### 2.1 问题定义

> **在长时间流式视频理解中，系统如何管理上下文记忆？哪些信息需要持久保存（固有记忆），哪些需要动态更新（工作记忆）？如何评估记忆的有效性？**

这个问题的核心在于：LLM 的上下文窗口是有限的（4K-128K tokens），但一个 30 分钟的烹饪视频可能产生数百帧的视觉信息。如何在有限窗口内保留最重要的信息？

### 2.2 记忆分层架构

以烹饪场景为例，提出三层记忆模型:

```
┌─────────────────────────────────────────────────────┐
│          Layer 1: 固有记忆 (Persistent Memory)        │
│                                                       │
│  - 任务目标: "做一道番茄炒蛋"                          │
│  - 任务知识: recipe 步骤列表                            │
│  - 用户偏好: "不喜欢太咸"                              │
│  - 场景约束: 可用的厨具和食材清单                       │
│                                                       │
│  特点: 整个任务期间不变，始终保留在上下文中              │
│  对应 ProAssist: initial_sys_prompt + knowledge        │
├─────────────────────────────────────────────────────┤
│          Layer 2: 进度记忆 (Progress Memory)           │
│                                                       │
│  - 当前步骤: "第 3 步 — 热锅倒油"                     │
│  - 已完成步骤: [打蛋✓, 切番茄✓, 热锅→]                │
│  - 累计状态: "鸡蛋已打散，番茄已切块"                   │
│  - 关键事件: "2分钟前放了盐"                           │
│                                                       │
│  特点: 随任务进展单调更新，旧进度被新进度替换            │
│  对应 ProAssist: progress_summary 字段                 │
├─────────────────────────────────────────────────────┤
│          Layer 3: 工作记忆 (Working Memory)             │
│                                                       │
│  - 最近 N 帧的视觉特征                                │
│  - 当前场景中的物体及其位置                             │
│  - 最近生成的 UI 组件 (previous_ui)                    │
│  - 用户的最近注视轨迹                                   │
│                                                       │
│  特点: 容量有限，旧信息被新信息替换 (滑动窗口)          │
│  对应 ProAssist: KV Cache 的近期部分                   │
└─────────────────────────────────────────────────────┘
```

### 2.3 StreamingContext 数据结构设计

```python
@dataclass
class StreamingContext:
    """流式视频理解的上下文状态"""

    # ── 固有记忆 (Persistent) ──
    task_goal: str                        # "做番茄炒蛋"
    task_knowledge: str                   # recipe 步骤列表
    scene_constraints: dict               # 可用厨具、食材

    # ── 进度记忆 (Progress) ──
    current_step: int                     # 当前步骤编号
    total_steps: int                      # 总步骤数
    completed_steps: list[str]            # 已完成的步骤描述
    progress_summary: str                 # 自然语言进度摘要
    key_events: list[tuple[float, str]]   # (时间戳, 事件描述) 列表

    # ── 工作记忆 (Working) ──
    trigger_time: float                   # 当前触发时间
    trigger_type: str                     # "interval" | "fixation_change" | "scene_change"
    scene_description: str                # 当前场景描述
    detected_objects: list[str]           # 当前帧中检测到的物体
    object_positions: dict[str, tuple]    # 物体 → 位置 映射
    gaze_position: Optional[tuple[float, float]]
    previous_ui: Optional[dict]           # 上一个生成的 UI 组件
    previous_ui_anchor: Optional[tuple]   # 上一个 UI 的锚定位置

    # ── 方法 ──
    def update_progress(self, new_step: int, description: str): ...
    def add_key_event(self, time: float, event: str): ...
    def should_preserve_previous_ui(self) -> bool:
        """判断是否应该保留上一个 UI（位置一致性）"""
        ...
    def compress_to_summary(self) -> str:
        """压缩上下文为摘要文本（用于 KV cache 管理）"""
        ...
```

### 2.4 上下文压缩策略对比

现有两大类压缩策略，分别来自 ProAssist (KV Cache 级) 和 StreamBridge (Embedding 级):

#### A. ProAssist 的 KV Cache 管理

| 策略 | 实现方式 | GUI 场景适用性 | 问题 |
|------|---------|---------------|------|
| **DROP_ALL** | 清空全部 KV cache，从零开始 | ❌ 不适用 | 丢失所有任务进度，UI 会完全断裂 |
| **DROP_MIDDLE** | 保留 init (系统 prompt) + 最近 512 tokens | ⚠️ 部分适用 | 保留了任务定义和近期状态，但丢失中间步骤 |
| **SUMMARIZE_AND_DROP** | 生成 progress_summary → 替换全部上下文 | ✅ 较适用 | 保留任务进度的语义信息，允许 UI 连续性 |

#### B. StreamBridge 的 Embedding 级压缩 (Round-Decayed Compression)

| 特性 | 实现方式 | GUI 场景适用性 |
|------|---------|---------------|
| **Modality-Aware** | 用 `modality_indicators` 区分图像/文本 token，只压缩图像 | ✅ 保留所有 UI 决策历史文本 |
| **Round-Decayed** | 老 round 的视觉 token 压缩为 1 帧 (mean-pooling)，新 round 保持完整 | ✅ 近期画面精确，远期画面模糊但有 |
| **部分压缩** | 一个 round 中前面的帧可以压缩，后面的帧保留 | ✅ 灵活的粒度控制 |
| **文本保护** | 文本 token 永不压缩 | ✅ 对话/UI 历史完整保留 |

**StreamBridge 压缩的实质** (源自 `compressor()` 代码分析):
```
压缩操作: N 帧 → 1 帧
  image_embeds = [frame1_tokens, frame2_tokens, ..., frameN_tokens]
  image_embeds = image_embeds.reshape(N, tokens_per_frame, dim)
  compressed = image_embeds.mean(dim=0)  # 帧维度取平均
  → [1, tokens_per_frame, dim]
```
本质上是**多帧视觉特征的时间维度平均池化**，保留了空间结构 (每帧仍有 tokens_per_frame 个 token)，但时间粒度降低了。

#### C. 我们的混合策略 (Hybrid Compression)

结合两者的优势，提出 GUI 场景的混合压缩策略:

```
┌────────────────────────────────────────────────────────────────┐
│  混合压缩策略 (Hybrid Compression for GUI)                      │
│                                                                  │
│  1. 视觉 Token 压缩 (借鉴 StreamBridge):                       │
│     - 按 UI Session (一次 UI 展示→隐藏周期) 分组                │
│     - 老 session 的帧: mean-pooling 压缩为 1 帧                 │
│     - 当前 session 的帧: 保持完整                                │
│                                                                  │
│  2. 文本 Token 保护 (借鉴 StreamBridge):                        │
│     - 所有 UI 决策历史、触发记录: 永不压缩                       │
│     - 对话文本: 永不压缩                                         │
│                                                                  │
│  3. 进度摘要更新 (借鉴 ProAssist):                              │
│     - 每次压缩视觉 token 时，同时更新 progress_summary           │
│     - 摘要包含: 当前步骤、已完成步骤、关键物体位置               │
│     - 确保即使帧特征被压缩，语义信息仍可从摘要中恢复             │
│                                                                  │
│  4. 空间信息保留 (我们的扩展):                                   │
│     - 对于有 UI 锚点的帧，额外保留物体位置 embedding              │
│     - 这是 StreamBridge 和 ProAssist 都缺少的                    │
└────────────────────────────────────────────────────────────────┘
```

**GUI 场景的特殊需求**:
- 除了文本 summary，还需要保留**物体位置信息**（用于 UI 锚定的一致性）
- 需要保留**上一个 UI 组件的信息**（用于内容一致性判断 — 是更新还是替换？）
- 可能需要保留**关键帧的视觉特征** embedding（而不仅仅是文本描述）
- StreamBridge 的 modality-aware 压缩为此提供了很好的框架 — 可以扩展 modality_indicators 增加 `ui_anchor=2` 类型，对这些 token 施加不同的压缩策略

### 2.5 评估任务设计

#### 评估维度 1: 位置一致性（短期物体记忆）

**任务: UI 锚点持久性测试 (UI Anchor Persistence Test)**

> 当一个 UI Card 被锚定在某个物体上（如锅）后，用户短暂转身（几帧的跳过），再回头时，系统能否记住这个 Card 应该仍然显示在锅的位置，而不需要重新生成？

```
实验设计:
  1. 在 t=10s，系统在锅上生成一个 Card("正在热油")
  2. t=12s-18s，用户转身去拿调料（锅不在画面中）
  3. t=19s，用户转回看锅

  评估:
  - 系统是否识别出这仍然是同一个锅？
  - 是否保留了之前的 Card 而非重新生成？
  - Card 的位置是否准确（即使画面角度略有变化）？

  指标:
  - Anchor Match Rate: 重新出现的物体被正确匹配的比例
  - UI Persistence Rate: 应该保留的 UI 被成功保留的比例
  - Position Drift: 保留的 UI 在位置上的偏移量 (像素)
```

**这涉及的技术挑战**:
- **物体重识别 (Re-identification)**: 转身后画面完全不同，回来后需要重新检测锅并匹配为同一个物体
- **UI 状态缓存**: 需要在 StreamingContext 中记住 `{object_id: "pot_1", ui: Card(...), position: (0.5, 0.3)}`
- **时间窗口判断**: 如果离开超过 N 秒（如 60 秒），可能场景已经变化，应该重新生成

#### 评估维度 2: 全局任务进度追踪（长期记忆）

**任务: 任务进度一致性测试 (Task Progress Consistency Test)**

> 系统能否在整个烹饪过程中持续追踪"进行到第几步"，并在 UI 中准确反映？

```
实验设计:
  1. 给定一个完整的烹饪视频 (20 min, 8 步)
  2. 系统在视频的不同时间点被 query: "当前进行到第几步?"
  3. 对比系统回答与 GT 步骤标注

  评估:
  - 在 t=5min 时，系统是否知道"刚完成第 2 步，正在第 3 步"？
  - 在 t=15min 时，系统是否记住"第 1 步已完成"？
  - 如果用户中途做了无关操作（如接电话），恢复后系统能否正确接续？

  指标:
  - Step Detection Accuracy: 步骤识别准确率
  - Progress Tracking F1: 已完成步骤的 F1
  - Recovery Rate: 中断后能正确恢复的比例
```

**UI 层面的体现**:
- 左上角 / 右上角显示的 **任务进度条** (ProgressBar) 是否始终准确？
- **Timeline 视图**是否正确反映已完成和待完成的步骤？

#### 评估维度 3: 细节记忆回溯（问答记忆）

**任务: 操作细节回忆测试 (Action Detail Recall Test)**

> 用户询问之前操作的细节时，系统能否正确回忆？

```
实验设计:
  用户: "我刚才盐放在哪了？" (t=15min，盐在 t=8min 时使用过)
  系统应该能回答: "你把盐放在了左边灶台旁边的调料架上"

  评估:
  - 系统是否记得"放过盐"这个事件？
  - 系统是否记得盐的位置？
  - 如果记忆已被压缩（SUMMARIZE_AND_DROP），摘要中是否保留了这个关键细节？

  指标:
  - Event Recall Accuracy: 关键事件被记住的比例
  - Detail Retention Rate: 事件细节(物体、位置、时间)的保留率
  - Compression Loss: 压缩前后回答质量的差异
```

### 2.6 开放问题

1. **记忆压缩的信息损失有多大？** SUMMARIZE_AND_DROP 在 ProAssist 中效果最好，但将视觉细节压缩为文本必然有信息损失。对于 UI 锚定这种需要精确位置信息的任务，文本摘要够用吗？
2. **固有记忆从哪来？** 在离线模式下可以通过 VLM 分析全视频获得 task_knowledge，在线模式下如何获取？是否需要用户预先告知任务目标？
3. **哪种压缩策略最适合 GUI 场景？** 代码分析显示 StreamBridge 的 Round-Decayed Compression 本质上是 mean-pooling 多帧为 1 帧（在 embedding 空间），而 ProAssist 的 SUMMARIZE_AND_DROP 是将视觉信息转化为文本摘要。前者保留了空间结构但丢失时间细节，后者保留了语义但丢失了空间信息。GUI 场景可能需要混合两者：视觉压缩 + 语义摘要 + 关键位置缓存
4. **StreamBridge 的 modality_indicators 机制能否扩展？** 当前是二值 (0=text, 1=image)，我们可以扩展为多值 (0=text, 1=image, 2=ui_anchor, 3=gaze_info)，对不同类型的 token 施加不同的压缩策略

---

## Research Question 3: GUI 生成的形式与一致性 (GUI Form & Consistency)

### 3.1 问题定义

> **在流式场景中，如何保证生成的 GUI 在位置上和内容上的时序一致性？如何设计 UI 状态管理机制使得 GUI 在视觉上平滑连贯？**

这个问题与 RQ2 (记忆管理) 密切关联 — UI 一致性本质上是记忆问题的外在表现。但它有独立的研究价值，因为涉及到 GUI 特有的空间布局、组件生命周期和视觉连续性问题。

### 3.2 问题拆解

#### 3.2.1 Position Consistency (位置一致性)

**核心挑战**: 用户头部在持续运动，画面内容不断变化，UI 应该锚定在什么位置？

| 锚定策略 | 描述 | 适用场景 | 实现难度 |
|---------|------|---------|---------|
| **Screen-Fixed** | UI 固定在屏幕某个位置 (如右上角) | 状态信息、进度条 | ⭐ 简单 |
| **Object-Anchored** | UI 跟随特定物体 (如锚定在锅上) | 操作提示、温度/状态标注 | ⭐⭐⭐ 困难 |
| **Gaze-Relative** | UI 出现在注视点附近 | 上下文信息、操作建议 | ⭐⭐ 中等 |
| **Semantic-Region** | UI 出现在语义区域 (如"工作台区域") | 步骤指导、区域说明 | ⭐⭐ 中等 |

**Object-Anchored 的关键技术问题**:
- 物体检测 + 跟踪 (Object Detection & Tracking across frames)
- 遮挡处理 (物体被手遮挡时 UI 应该怎么办?)
- 视角变化 (同一个锅从正面看和侧面看，UI 位置应该如何调整?)

#### 3.2.2 Content Consistency (内容一致性)

**核心挑战**: 连续的 UI 生成之间应该保持怎样的关系？

```
时间轴上的 UI 更新:

  t=10s: Card("第1步: 打蛋", progress=1/8)
  t=30s: Card("第1步: 打蛋", progress=1/8)      ← 场景没变，应该保持不变
  t=45s: Card("第2步: 切番茄", progress=2/8)     ← 步骤转换，内容更新
  t=50s: Card("第2步: 切番茄", progress=2/8)     ← 场景没变，保持
  t=55s: Badge("⚠ 注意手指!")                    ← 检测到风险，叠加显示
  t=60s: Card("第2步: 切番茄", progress=2/8)     ← 风险消除，恢复原 UI
```

**需要解决的内容一致性问题**:
1. **不必要的重新生成**: 场景没变，不应该生成新 UI
2. **组件类型跳变**: 从 Card 突然变成 Badge 又变回 Card — 需要平滑过渡
3. **信息覆盖**: 新 UI 是否应该完全替换旧 UI，还是叠加显示？
4. **渐进式更新**: 能否只更新变化的部分（如进度从 2/8 → 3/8），而不重新生成整个组件？

### 3.3 UI State Machine 设计

这是将"一致性"问题形式化为研究问题的关键:

```python
class UIState(Enum):
    """UI 组件的状态机"""
    HIDDEN = "hidden"           # 没有 UI 显示
    ENTERING = "entering"       # UI 正在淡入
    ACTIVE = "active"           # UI 正常显示
    UPDATING = "updating"       # UI 内容正在更新 (渐变)
    ALERT = "alert"             # 紧急叠加 (如错误警告)
    EXITING = "exiting"         # UI 正在淡出
```

```
状态转移图:

  HIDDEN ──[触发条件满足]──→ ENTERING ──[动画完成]──→ ACTIVE
                                                       │
                                    ┌──[场景变化]───────┤
                                    │                   │
                                    ↓                   ↓
                                UPDATING            ──[风险检测]──→ ALERT
                                    │                               │
                                    ↓                               ↓
                                ACTIVE ←─────[风险消除]─────────── ACTIVE
                                    │
                        ┌──[超时/场景完全变化]
                        ↓
                    EXITING ──[动画完成]──→ HIDDEN
```

### 3.4 研究问题形式化

**RQ3 可以被形式化为以下优化问题**:

给定:
- 视频帧序列 $F = \{f_1, f_2, ..., f_T\}$
- 任务进度序列 $S = \{s_1, s_2, ..., s_K\}$ (K 个步骤)
- 用户注视序列 $G = \{g_1, g_2, ..., g_T\}$ (可选)

目标是生成 UI 序列 $U = \{u_1, u_2, ..., u_N\}$（N 个 UI 事件），使得:

1. **时序连贯性** (Temporal Coherence):
   $$\text{minimize} \sum_{i=1}^{N-1} \text{VisualJump}(u_i, u_{i+1})$$
   相邻 UI 之间的视觉跳变最小化

2. **信息增量性** (Information Gain):
   $$\text{maximize} \sum_{i=1}^{N} \text{InfoGain}(u_i | u_{1:i-1}, F, S)$$
   每个 UI 应该提供新信息，而非重复已知内容

3. **位置稳定性** (Position Stability):
   $$\text{minimize} \sum_{i=1}^{N-1} \|\text{pos}(u_i) - \text{pos}(u_{i+1})\| \cdot \mathbb{1}[\text{same\_context}(u_i, u_{i+1})]$$
   在相同上下文中，UI 位置不应大幅变化

4. **时机恰当性** (Timing Appropriateness):
   与 RQ1 关联 — 详见 RQ1 的评估指标

### 3.5 评估方案

#### 自动评估

| 指标 | 计算方式 | 衡量什么 |
|------|---------|---------|
| **UI Churn Rate** | 单位时间内 UI 变化次数 | 稳定性（越低越好） |
| **Unnecessary Regeneration Rate** | 场景未变但 UI 重新生成的比例 | 内容一致性 |
| **Position Variance** | 同一上下文中 UI 位置的方差 | 位置一致性 |
| **Component Type Consistency** | 相邻 UI 使用相同组件类型的比例 | 形式一致性 |
| **Information Redundancy** | 连续 UI 内容的语义相似度 (过高 = 重复) | 信息增量性 |

#### 用户研究

1. **A/B 测试**: 有 UIStateManager vs 无 UIStateManager，用户评价 UI 体验
2. **注视热力图对比**: 有稳定锚定的 UI 是否更容易被用户注意到？
3. **任务完成率**: 一致的 UI 是否帮助用户更高效地完成烹饪任务？

### 3.6 与两阶段架构的关系

UI 一致性问题主要在 **Stage 2 (Generative UI)** 解决，但需要 **Stage 1 (Video Understanding)** 提供:

- **场景变化检测**: Stage 1 告诉 Stage 2 "场景是否发生了实质变化"
- **物体跟踪信息**: Stage 1 提供 `object_id` 和 `position` 用于锚定
- **任务进度**: Stage 1 提供 `current_step` 用于判断内容是否需要更新

这意味着 **StreamingContext** (RQ2 的核心产出) 是连接 RQ1 和 RQ3 的桥梁:

```
RQ1 (When) → trigger_time, trigger_type
                    ↓
RQ2 (Memory) → StreamingContext (完整上下文状态)
                    ↓
RQ3 (What & Where) → A2UI Component + Position
```

### 3.7 开放问题

1. **UI 一致性能否用现有 VLM 实现？** 还是需要专门的轻量级模块（如 Object Tracker + UIStateManager）来保证实时性？
2. **用户对 UI 一致性的容忍度是多少？** 偶尔的位置跳变和内容跳变是否可以接受？需要用户研究数据
3. **渐进式更新 vs 完全重新生成**: 在 A2UI 组件规范下，"只更新 props" 是否比"重新生成整个组件" 更好？代码复杂度 vs 用户体验的权衡

---

## 三个 RQ 的关系

```
                        ┌─────────────────┐
                        │    RQ1: When     │
                        │  主动干预时机     │
                        │                  │
                        │  输出: 触发信号   │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   RQ2: Memory    │
                        │  上下文记忆管理   │
                        │                  │
                        │  输出:            │
                        │  StreamingContext │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  RQ3: What/Where │
                        │  GUI 形式与一致性 │
                        │                  │
                        │  输出:            │
                        │  A2UI + Position  │
                        └─────────────────┘
```

- **RQ1 → RQ2**: 干预时机决定了何时需要更新记忆状态
- **RQ2 → RQ3**: 记忆内容决定了生成什么样的 UI（内容一致性）
- **RQ3 → RQ1**: UI 的显示/更新规则可能反馈到触发策略（如"当前 UI 仍然有效，不需要新的触发"）
- **三者共享**: VLM Brain 的场景理解能力是所有 RQ 的基础

---

## 两阶段架构总结

```
┌────────────────────────────────────────────────────────────────┐
│  Stage 1: Multimodal Brain (VLM)                               │
│                                                                │
│  职责:                                                         │
│  1. [RQ1] 判断干预时机 (W2T 机制)                               │
│  2. [RQ2] 维护 StreamingContext (三层记忆)                      │
│  3. [RQ3] 提供场景理解 + 物体追踪信息                           │
│                                                                │
│  候选模型: Gemini 2.5 Pro / GPT-4o / ProAct (微调)             │
│  输入: 视频帧 + (可选) 眼动数据                                 │
│  输出: StreamingContext                                         │
├────────────────────────────────────────────────────────────────┤
│  Stage 2: Generation Model (Code/Agent)                        │
│                                                                │
│  职责:                                                         │
│  1. [RQ3] 根据 StreamingContext 生成 A2UI JSON                 │
│  2. [RQ3] UI 状态管理 (UIStateManager)                         │
│  3. [RQ3] 位置计算与锚定                                       │
│                                                                │
│  候选模型: Claude / GPT-4o / 同一 VLM (Agent 模式)             │
│  输入: StreamingContext                                         │
│  输出: A2UI Component JSON + Position                          │
│                                                                │
│  注: 也可以是同一个模型的两次调用 (Agent 架构)，                │
│  此时 Stage 1 = "思考" (reasoning)，Stage 2 = "行动" (acting)  │
└────────────────────────────────────────────────────────────────┘
```

---

## 下一步计划

1. **文献调研**: 补充 streaming video understanding 最新进展（StreamBridge, Flash-VStream, VideoLLM-Online, Kangaroo 等）
2. **数据评估**: 评估 ProAssist cooking 子集的数据量和质量，确定是否满足 RQ1 的训练需求
3. **原型搭建**: 先用 Gemini 2.5 Pro 搭建离线 pipeline 原型，验证两阶段架构的可行性
4. **评估基准构建**: 设计 RQ2 和 RQ3 的评估 benchmark（UI 锚点持久性测试 + 任务进度一致性测试）

---

## 参考文献

1. ProAssist: Zhang et al., "Proactive Assistant Dialogue Generation from Streaming Egocentric Videos", arXiv:2506.05904, 2025
2. StreamGaze: Lee et al., "StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos", arXiv:2512.01707, 2025
3. StreamBridge: "StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant", arXiv:2505.05467, NeurIPS 2025
4. EpicKitchens: Damen et al., "The EPIC-KITCHENS Dataset", ECCV 2018
5. Ego4D: Grauman et al., "Ego4D: Around the World in 3,000 Hours of Egocentric Video", CVPR 2022
6. HoloAssist: Huang et al., "HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World", ICCV 2023
7. EgoExoLearn: Song et al., "EgoExoLearn: A Dataset for Bridging Asynchronous Ego- and Exo-centric View of Procedural Activities", CVPR 2024
