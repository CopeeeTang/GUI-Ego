# Smart Glasses Proactive GUI System — Architecture Overview

> Date: 2026-02-26
> Status: Working System (EGTEA validated)
> Code: `/home/v-tangxin/GUI/agent/`

---

## 1. Problem

现有 Generative UI（如 Google 的 LLM-based UI Generator）只解决"给定需求生成 Web 页面"。但 AR 智能眼镜场景有四个根本不同的维度：

| 维度 | Web UI | AR Glasses UI |
|------|--------|---------------|
| **When** | 用户主动请求 | 系统判断时机 |
| **What** | 给定 query 生成页面 | 系统主动决定内容 |
| **Where** | 2D 页面布局 | 3D 空间锚定 |
| **How long** | 用户关闭 | 自动生命周期管理 |

我们需要一个从视频流到 GUI 生成的端到端系统，能主动判断何时、展示什么、在哪里展示。

---

## 2. System Architecture

系统分两个 Stage：视频理解（Stage 1）和 UI 生成（Stage 2），通过 StreamingContext 桥接。

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Streaming Video Understanding                             │
│                                                                     │
│  Video Stream (2FPS)                                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─ OBSERVE ──────────────────────────────┐                        │
│  │  SceneAnalyzer    → 场景描述/物体/活动  │                        │
│  │  ChangeDetector   → 帧间变化分数        │                        │
│  │  SignalReader      → 眼动/心率 (可选)    │                        │
│  └────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─ THINK ────────────────────────────────┐                        │
│  │  Memory System (三层)                   │                        │
│  │    L1 Persistent — 任务/步骤/约束       │                        │
│  │    L2 Progress   — 当前步骤/事件        │                        │
│  │    L3 Working    — 最近8帧窗口          │                        │
│  │           ↓                             │                        │
│  │  TaskTracker → 步骤转换检测             │                        │
│  │           ↓                             │                        │
│  │  StreamingContext ← 融合三层记忆         │                        │
│  │           ↓                             │                        │
│  │  TriggerDecider (三通道)                │                        │
│  │    Reactive     — 错误/危险 → HIGH      │                        │
│  │    Anticipatory — 步骤转换 → MEDIUM     │                        │
│  │    Signal       — 眼动固视 → 按类型     │                        │
│  └────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼ (触发时)                                                    │
│  ┌─ ACT ──────────────────────────────────┐                        │
│  │  InterventionEngine                     │                        │
│  │    LLM(StreamingContext) → Intervention  │                        │
│  │    {type, content, confidence, priority} │                        │
│  └────────────────────────────────────────┘                        │
│                                                                     │
│  输出: Intervention + StreamingContext                               │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2: Generative UI                                             │
│                                                                     │
│  Intervention                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─ UI Generation ────────────────────────┐                        │
│  │  Prompt Strategy (v2_smart_glasses)     │                        │
│  │    → AR HUD 约束: 最小遮挡/高对比度     │                        │
│  │    → 组件选择: Card/Badge/ProgressBar   │                        │
│  │  LLM → A2UI JSON                       │                        │
│  │  OutputValidator → 结构校验             │                        │
│  └────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─ Rendering ────────────────────────────┐                        │
│  │  Preview Server (port 8000)             │                        │
│  │  Video Overlay → UI 叠加到视频帧        │                        │
│  │  A2UI Lit Renderer → AR Glasses 样式    │                        │
│  └────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Streaming Video Understanding

### 3.1 Observe — 感知层

每帧（2FPS）执行：

**SceneAnalyzer** (`agent/ar_proactive/video/scene_analyzer.py`)
- 输入：视频帧 (base64 JPEG)
- 输出：环境描述、检测物体列表、当前活动、潜在危险
- 实现：VLM 单帧推理

**VisualChangeDetector** (`agent/ar_proactive/video/change_detector.py`)
- 输入：当前帧 + 前一帧
- 输出：变化分数 (0~1)
- 用途：辅助触发决策（场景是否发生实质变化）

**SignalReader** (`agent/ar_proactive/signals/reader.py`)
- 输入：眼动轨迹、心率、皮电反应（可选传感器数据）
- 输出：固视事件、异常信号
- 说明：当前评估中未使用生理信号通道

### 3.2 Think — 理解层

#### 三层记忆系统

```
┌───────────────────────────────────────────────────┐
│  L1: PersistentMemory (永久)                       │
│  persistent.py                                     │
│                                                    │
│  内容: 任务目标、步骤列表、用户偏好、场景约束       │
│  来源: 从视频前几帧推断 或 外部知识库               │
│  生命周期: 会话开始时初始化，永不删除                │
│                                                    │
│  示例 (PastaSalad):                                │
│    task_goal: "Prepare PastaSalad"                 │
│    steps: ["Gather ingredients",                   │
│            "Gather eating utensils",               │
│            "Turn on stove", ...]                   │
├───────────────────────────────────────────────────┤
│  L2: ProgressMemory (单调递增)                     │
│  progress.py                                       │
│                                                    │
│  内容: 当前步骤索引、已完成步骤、关键事件列表       │
│  容量: 最多50个关键事件，超出丢弃最早的             │
│  更新触发: 步骤转换、重要事件发生                   │
│  压缩: 超过30事件时自动生成摘要                     │
│                                                    │
│  示例:                                             │
│    current_step: 3 (Turn on stove)                 │
│    completed: 2/8                                  │
│    events: [(50.9s, "Recipe visible on microwave"),│
│             (61.8s, "Electrical cord near water")]  │
├───────────────────────────────────────────────────┤
│  L3: WorkingMemory (滑动窗口)                      │
│  working.py                                        │
│                                                    │
│  内容: 最近N帧的 FrameRecord                       │
│    - 帧数据 (base64)                               │
│    - 场景描述                                      │
│    - 检测物体、活动、危险                           │
│    - 前一次UI状态                                   │
│  容量: 8帧 (可配置)，FIFO                          │
│  用途: 提供即时的视觉上下文                         │
└───────────────────────────────────────────────────┘
```

#### StreamingContext (`agent/ar_proactive/context.py`)

每帧组装一次，桥接 Stage 1 和 Stage 2：

```python
@dataclass
class StreamingContext:
    # 当前帧
    timestamp: float
    trigger_type: str
    current_frame: FrameRecord

    # L1 持久化
    task_goal: str
    task_type: str
    total_steps: int
    step_descriptions: list[str]

    # L2 进度
    current_step_index: int
    completed_steps: int
    progress_text: str
    recent_events: list[str]

    # L3 工作
    recent_context_text: str
    recent_objects: set
    recent_activities: set

    # 信号与UI
    visual_change_score: float
    previous_ui: Optional[dict]
```

#### TaskTracker (`agent/ar_proactive/task/tracker.py`)

- 按帧调用 LLM 检测步骤转换
- 输入：当前帧 + 步骤列表 + 当前步骤
- 输出：是否发生步骤转换、新步骤索引
- 检查间隔：每3帧最多检查一次 + 视觉变化阈值过滤

#### TriggerDecider (`agent/ar_proactive/intervention/trigger.py`)

三通道评估，任一通道触发即生成干预：

| 通道 | 触发条件 | 优先级 | 示例 |
|------|---------|--------|------|
| **Reactive** | 检测到错误或危险 | HIGH | 电源线靠近水源 |
| **Anticipatory** | 步骤转换 / 场景大幅变化 | MEDIUM | 完成切菜，该热锅了 |
| **Signal** | 眼动固视>500ms / 心率尖峰 | 按类型 | 用户反复扫视找东西 |

冷却机制：两次触发间最短 5 秒（可配置），防止频繁干预。

### 3.3 Act — 干预层

**InterventionEngine** (`agent/ar_proactive/intervention/engine.py`)

输入：StreamingContext + TriggerDecision
过程：LLM 调用，生成干预内容
输出：

```python
@dataclass
class Intervention:
    timestamp: float
    intervention_type: InterventionType  # 8种类型
    intervention_mode: InterventionMode  # reactive/anticipatory/signal
    confidence: float                    # 0~1
    content: str                         # 用户看到的文本
    reasoning: str                       # 推理过程
    priority: str                        # high/medium/low
    related_step: Optional[int]
    ui_component: Optional[dict]         # A2UI JSON
```

8 种干预类型：

| 类型 | 说明 | 模式 |
|------|------|------|
| `safety_warning` | 安全警告 | Reactive |
| `error_correction` | 错误纠正 | Reactive |
| `step_instruction` | 步骤指导 | Anticipatory |
| `task_guidance` | 任务指导 | Anticipatory |
| `contextual_tip` | 上下文提示 | Anticipatory |
| `object_info` | 物体信息 | Signal |
| `navigation_help` | 导航辅助 | Signal |
| `social_cue` | 社交提示 | Signal |

---

## 4. Stage 2: Generative UI

### 4.1 Prompt Strategy

系统支持 4 种 Prompt 策略（`agent/src/prompts/`）：

| 策略 | 方法 | 视觉 | 适用场景 |
|------|------|------|---------|
| `v1_baseline` | 两步法：选组件→填属性 | ❌ | 开发调试 |
| `v2_google_gui` | 端到端生成 | ✅ | 通用 |
| `v2_smart_glasses` | AR HUD 优化 | ✅ | **生产用** |
| `v3_with_visual` | 视觉上下文增强 | ✅ | 多帧理解 |

**v2_smart_glasses 的设计约束**：
- 不遮挡视野中央（用户走路时安全）
- 高对比度（深色半透明底 + 亮色文字，适配动态视频背景）
- 最小文本（用户无法在行走时阅读长段落）
- 大触控目标（手势/注视交互友好）
- Peripheral 布局（行走时内容在视野边缘）/ Immersive 布局（静止时内容在中央）

### 4.2 Multi-LLM Support

| Provider | Model | 用途 |
|----------|-------|------|
| Azure OpenAI | GPT-4o | 默认，企业级稳定 |
| Google Gemini | 2.5-pro/flash | 长上下文，视觉能力强 |
| Anthropic Claude | Sonnet/Opus | 推理+代码生成 |
| Google Vertex AI | Gemini via Vertex | 企业级 Gemini |

切换方式：`--model gemini:gemini-2.5-flash`

### 4.3 A2UI 组件

输出格式为 A2UI JSON，支持的原子组件：

| 组件 | 变体 | 用途 |
|------|------|------|
| **Card** | glass / solid / outline / alert | 容器 |
| **Text** | hero / h1 / h2 / body / caption | 文本 |
| **Button** | primary / secondary / ghost | 交互 |
| **Badge** | info / success / warning / error | 状态 |
| **ProgressBar** | default / slim | 进度 |
| **Icon** | small / medium / large | Material Icons |
| **Row / Column** | — | 布局 |
| **List / Divider / Image** | — | 辅助 |

输出示例：
```json
{
  "type": "Card",
  "props": {"variant": "alert"},
  "children": [
    {"type": "Icon", "props": {"name": "warning", "size": "medium"}},
    {"type": "Text", "props": {"variant": "h2", "content": "WARNING"}},
    {"type": "Text", "props": {
      "variant": "body",
      "content": "Electrical cord near water source. Move cord away from sink."
    }}
  ]
}
```

### 4.4 Preview & Rendering

**Preview Server** (`agent/preview/server.py`, port 8000)
- 实时渲染 A2UI JSON 为 AR Glasses 样式
- 支持 Video Overlay（UI 叠加到原视频帧，基于眼动坐标定位）
- 组件属性编辑、截图导出

```bash
# 启动预览
python3 -m agent.preview.server --port 8000

# 运行 Pipeline 生成 + 叠加
python3 -m agent.src.pipeline --strategy v2_smart_glasses --overlay --limit 5
```

---

## 5. Development History

系统从"GT 到 GUI 的单步生成"演化为"端到端主动干预 + GUI 生成"，经历了 7 个关键阶段。

### 5.1 迭代路径总览

```
Phase 1: 项目规划 + 数据准备 (01-27~29)
    │
    ▼
Phase 2: A2UI Agent 首版实现 (01-29~30)
    │  发现: GT→GUI 单步生成太简单，无法评估
    ▼
Phase 3: V2 语义-渲染分离重构 (01-30~02-02)
    │  核心: Semantic Layer 与 Rendering Layer 解耦
    ▼
Phase 4: 功能扩展 (02-02~05)
    │  Video Overlay + Multi-LLM + Preview
    ▼
Phase 5: Proactive AR Agent 首版 (02-11)
    │  Observe→Think→Act 循环 + 三层记忆 + 双门控
    ▼
Phase 6: RQ 驱动重构 (02-13~16)
    │  按功能分层记忆 + 三通道触发 + TaskTracker + StreamingContext
    ▼
Phase 7: 评估框架 + GT 批量生成 (02-17~20)
    │  13,076 条标注，86/86 sessions 全覆盖
    ▼
当前: 用 GT 评估 Agent，迭代改进
```

### 5.2 Phase 1-2: 基础实现 (01-27 ~ 01-30)

**起点**：从 Ego-Dataset（20 名被试，60+ 小时视频，12 类场景）出发，验证最小闭环。

首版实现（V1）使用已标注的 Ground Truth 作为输入，直接生成 A2UI 组件代码。这是一个 **GT → GUI 的单步生成**，验证了 VLM + 代码生成的技术路线，但暴露了两个问题：
1. 仅基于一步生成太简单，研究贡献度不够
2. 生成的 GUI 组件难以自动化评估

### 5.3 Phase 3: 语义-渲染分离重构 (01-30 ~ 02-02)

将"理解"和"渲染"从耦合变为解耦：

```
V1: Video → [VLM: 理解+生成] → UI Component
V2: Video → [Semantic Layer: 理解] → Semantic JSON → [Rendering Layer: 渲染] → UI Component
```

分离带来三个好处：
1. 语义层可独立评估（不需要渲染就能判断理解是否正确）
2. 渲染层可复用不同组件库
3. 两层可用不同模型（语义层用强模型，渲染层用快模型）

同时完成了 `ExampleLoader` 统一数据加载、Output 目录标准化、数据质量验证。

### 5.4 Phase 4: 功能扩展 (02-02 ~ 02-05)

三个并行方向：

| 模块 | 实现 | Session 消息数 |
|------|------|:-----:|
| **Video UI Overlay** | UI 组件叠加到原始视频帧 | 103 |
| **Multi-LLM Backend** | Claude / GPT-4o / Gemini / Vertex AI | 112 |
| **Preview Server** | HTTP 预览 + AR Glasses UI 组件 | 105 |

> Multi-LLM 支持在后续 GT 生成中证明关键——Gemini-3-flash 全量 GT 成本约 $1，比 Claude 低两个数量级。

### 5.5 Phase 5: Proactive AR Agent 首版 (02-11)

从"被动生成 UI"到"主动感知并干预"的根本性转变。实现了 Observe → Think → Act 循环（详见 Section 3）。

初版三层记忆按 **重要性分数** 分层：短期（deque, 6帧）/ 中期（list, 20条）/ 长期（LLM 摘要），并通过双门控（cooldown + importance 阈值）减少 50% 的 LLM 调用。

**首次运行验证**（sample_001, P10_Ernesto/Task2.1）：
- 4 帧处理，0 次干预（正确判断社交用餐无需干预）
- 6 次 LLM 调用（4 次场景分析 + 2 次干预评估，另 2 次被门控跳过）

### 5.6 Phase 6: RQ 驱动重构 (02-13 ~ 02-16)

对照三个研究问题（RQ）诊断系统缺陷后，执行全面重构：

| 维度 | 重构前 | 重构后 | 驱动 RQ |
|------|--------|--------|---------|
| 记忆模型 | 按重要性分三层 | **按功能分三层**（Persistent/Progress/Working） | RQ2 |
| 任务理解 | 无 | `TaskKnowledgeExtractor` + `TaskTracker` | RQ1+RQ2 |
| 干预时机 | `score >= threshold` | **三通道**（Anticipatory/Reactive/Signal） | RQ1 |
| 上下文传递 | 7 个零散参数 | 统一 `StreamingContext` | RQ2 |
| UI 输出 | 纯文本字符串 | A2UI JSON + 状态机（计划中） | RQ3 |

同时引入 **EGTEA Gaze+** 数据集（86 sessions, 15,484 条动作标注, 25.6h 视频），发现其 action boundaries 可作为干预时机的 proxy GT。

**端到端测试结果**（PastaSalad, 3 clips）：

| 指标 | 数值 |
|------|------|
| 处理帧数 | 8 |
| 处理耗时 | 101.7s (~12.7s/帧) |
| 生成干预 | 3 条 (anticipatory: 2, reactive: 1) |
| GT 动作边界 | 3 个 |
| **边界覆盖率** | **100%** |

**发现的问题**：步骤检测跳跃（1→7→8→132），EGTEA 有 185 个 unique steps，LLM 倾向匹配视觉最相似的步骤而非顺序最近的。

### 5.7 Phase 7: 评估框架 + GT 批量生成 (02-17 ~ 02-20)

详见 Section 6（评估框架）和 Section 7（GT 数据系统）。

---

## 6. Ground Truth Data System

### 6.1 GT 生成框架

为 EGTEA Gaze+ 全部 86 个 session 的每个 action boundary 生成主动干预标注。

**输入**：每个 action boundary 处的视频帧 + 前后动作标签 + recipe 上下文

**输出**：

```json
{
  "should_intervene": true,
  "intervention_type": "safety_warning | step_instruction | contextual_tip | error_correction",
  "intervention_mode": "anticipatory | reactive",
  "content": "具体干预内容",
  "rationale": "推理过程",
  "visible_hazards": ["..."],
  "visible_objects": ["..."],
  "frame_description": "当前画面描述"
}
```

**生成模型**：Gemini-3-flash-preview（VLM，支持图像输入）

### 6.2 人工审查标准

采用"先生成 → 人工抽检 → 批量生成"的流程：

| 检查项 | 内容 |
|--------|------|
| Check 1 | `should_intervene` 决策是否合理 |
| Check 2 | `intervention_type` 和 `intervention_mode` 是否正确 |
| Check 3 | `content` 是否具体、可操作 |
| Check 4 | `visible_hazards` / `visible_objects` 场景识别准确性 |
| Check 5 | `rationale` 推理链是否成立（无幻觉） |

### 6.3 生成工程

#### API 策略演化

成本分析驱动了模型选择（全量 ~5000 boundaries）：

| 模型 | 估算费用 | 选择 |
|------|---------|------|
| Claude | ~$878 | ❌ 太贵 |
| **Gemini-3-flash** | **~$1.07** | ✅ 采用 |

API 策略经历三个阶段：
1. **Proxy gemini-3-flash** — 第一波（43 sessions，串行），代理每 1-2h 断连
2. **官方 gemini-3-flash-preview** — 利用免费额度并行加速
3. **Vertex AI gemini-3-flash-preview** — 最终方案（企业级配额，ADC 认证）

#### 工程挑战与解决

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| **故障传播** | 代理断连 → "5次失败后跳过"逻辑导致 42 session 空跑 | 区分"可跳过错误"（单 boundary 失败）和"需暂停错误"（基础设施断连），引入 `_wait_for_proxy_recovery()` |
| **代理周期性重启** | 每 1-2h 重启，5min 超时不够 | 30min 轮询等待 + checkpoint/resume |
| **速率限制** | 4 worker 并行触发 429 | 降至 3 worker，~930 boundaries/h |
| **Vertex AI 集成** | ADC token 刷新 bug, thinking tokens 占预算, `thought_signature` 字段 | `creds.refresh(auth_req)`, `max_output_tokens >= 1024`, 修复 `_extract_response_text` |
| **JSON 截断** | MAX_TOKENS 导致响应截断 | `_repair_truncated_json()` 自动修复 |

#### 并行协调

| | 串行 | 4 路并行 | 3 路并行（最终） |
|---|---|---|---|
| 速率 | ~310/h | 频繁 429 | **~930/h** |
| 预计耗时 | ~14h | 不稳定 | **~7.5h** |

两个进程共享输出目录、各自独立 checkpoint，"先完成者为准"的自然协调机制。

### 6.4 GT 最终结果

| 指标 | 数值 |
|------|------|
| Sessions | **86/86** (100%) |
| 标注 boundaries | **13,076** |
| 介入标注 | **1,781** (13.6%) |
| 缺失 boundaries | 23 (0.18%) |
| 数据问题率 | ~60/13,076 (0.46%) |

#### 干预类型分布

| 类型 | 数量 | 占比 | 说明 |
|------|------|------|------|
| `step_instruction` | 444 | 48.5% | 步骤指导 |
| `safety_warning` | 388 | 42.4% | 安全警告 |
| `contextual_tip` | 79 | 8.6% | 上下文提示 |
| `error_correction` | 5 | 0.5% | 纠错 |

#### 跨菜谱差异

| 菜谱 | 典型介入率 | 原因 |
|------|-----------|------|
| PastaSalad | 14-25% | 灶台操作，安全隐患多 |
| BaconAndEggs | 7-20% | 含高温操作 |
| TurkeySandwich | 1-4.5% | 最简单（切面包/放肉/放菜） |

#### 数据质量核查

核查发现 3 个高优先级问题（均集中在最早跑的 OP 系列 session）：

| Session | 问题 | 修复后 |
|---------|------|--------|
| OP01-R02-TurkeySandwich | boundary 74-87 重复乱序 | 88/88 连续, 4.5% |
| OP02-R04-ContinentalBreakfast | 前 5 条缺失 | 90/90 完整, 10.0% |
| OP02-R01-PastaSalad | 电源插座 10+ 次重复告警 | 25.4% (降低但仍偏高) |

#### GT 典型示例

| 类型 | Session | 内容 |
|------|---------|------|
| `safety_warning` | P23-BaconAndEggs | "电线靠近水槽，注意触电风险" |
| `safety_warning` | OP05-BaconAndEggs | "白布悬挂在灶台附近，注意火灾风险" |
| `step_instruction` | OP01-GreekSalad | "提示下一步加芝士和调料" |
| `contextual_tip` | P02-Cheeseburger | "肉饼烹饪时间和温度建议" |
| `error_correction` | P02-PastaSalad | "阻止用户把煮面水倒掉" |

---

## 7. Evaluation Framework

### 5.1 评估架构

```
agent/ar_proactive/eval/
├── gt_generator.py   # 从 EGTEA 标签生成 Ground Truth
├── runner.py         # 评估主循环：遍历视频→收集 Intervention
├── metrics.py        # 触发 F1 / 内容相似度 / 步骤准确率
├── judge.py          # LLM-as-Judge 5维评分
└── report.py         # 生成评估报告
```

### 7.2 评估指标

四维评估框架：

| 类别 | 指标 | 说明 |
|------|------|------|
| **Trigger Timing** | Precision, Recall, F1 | 基于 GT action boundaries 的时间窗口匹配 |
| | Harmful FP / Triggers per min | 有害假阳性 / 干预频率 |
| **Step Detection** | Change Detection R/P | 步骤转换检测准确率 |
| | Monotonicity | 步骤序号是否单调递增 |
| **System Health** | Avg Confidence / Processing Ratio | 系统置信度 / 实时比 |
| **Content Quality** | Gemini LLM-as-Judge (5维) | Relevance / Accuracy / Helpfulness / Timing / Conciseness |

### 7.3 早期小规模评估 (Phase 6, PastaSalad 2-3 sessions)

| 指标 | Periodic (15s) | Oracle (GT边界) | VLM Delta |
|------|:--------------:|:---------------:|:---------:|
| **Trigger F1** | 0.403 | **0.662** | 0.198 |
| **Content Sim** | 0.626 | **0.636** | 0.574 |
| **Harmful FP** | 20.0 | 6.7 | **0.0** |
| Precision | 0.652 | 0.898 | 0.617 |
| Recall | 0.292 | 0.524 | 0.118 |

关键发现：
- **Oracle** 达到生产级精度 (F1=0.662)，验证系统在正确触发时机下的能力
- **VLM Delta** 零有害假阳性，对安全敏感的 AR 场景至关重要
- **v1→v2 内容质量** 从 0.18 提升到 0.63 (3.5×)

### 7.4 详细评估结果 (Phase 7, PastaSalad 3 clips, 含 GT)

#### Trigger Timing

| 指标 | 数值 | 解读 |
|------|------|------|
| Precision | 66.7% | 3 次触发中 2 次在 GT 边界附近 |
| Recall | 66.7% | 3 个 GT 边界有 2 个被覆盖 |
| F1 | 66.7% | 基线表现 |
| False Trigger Rate | 33.3% | 1 次不该说话时说了 |

#### Step Detection

| 指标 | 数值 |
|------|------|
| Change Detection Recall | 50% |
| Change Detection Precision | 50% |
| Monotonicity | 100% |
| 步骤序列 | 1→7→8→132（跳跃过大） |

#### System Health

| 指标 | 数值 | 解读 |
|------|------|------|
| Interventions/min | 14.0 | 偏高（理想 < 5/min） |
| Avg Confidence | 0.88 | 系统过度自信 |
| Processing Ratio | 7.9× | 比实时慢（需优化） |

#### Content Quality (Gemini-3-flash LLM-as-Judge)

| 维度 | 分数 | 解读 |
|------|------|------|
| Relevance | 3.67/5 | 中上 |
| Accuracy | 4.67/5 | 事实性高 |
| **Helpfulness** | **3.00/5** | **最弱** — 冗余信息、优先级错误 |
| Timing | 3.67/5 | 1 条时机不当 |
| Conciseness | 5.00/5 | 适合 AR HUD |
| **Overall** | **4.00/5** | |
| Safety Precision | **100%** | 安全警告全部合理 |

> **核心发现**：系统最大弱点是 helpfulness (3.0/5)，不是内容不正确，而是不够有用——冗余信息、错误的步骤优先级。改进方向应聚焦于**上下文感知的内容过滤**，而非生成能力本身。

### 7.5 干预质量案例分析

PastaSalad 会话中 3 个干预的评估：

| # | 时间 | 类型 | 内容 | 评分 |
|---|------|------|------|:----:|
| 1 | 50.9s | step_instruction | "Recipe is visible. Ready to start Step 2: gather utensils." | 3.8/5 |
| 2 | 57.0s | step_instruction | "Turn on the stove. Set to medium-high heat." | 3.2/5 |
| 3 | 61.8s | **safety_warning** | "WARNING: Electrical cord near water source. Move cord away." | **5.0/5** |

BaconAndEggs 完整 session（约 13 分钟）的 5 次干预：

| 时间 | 类型 | 内容 |
|------|------|------|
| 1:42 | contextual_tip | "Add about 2 tablespoons of milk to the eggs for a fluffier texture." |
| 3:09 | step_instruction | "Turn the burner for the frying pan to medium heat to begin preheating." |
| 7:28 | safety_warning | "Move the laptop away from the stove to prevent heat damage or spills." |
| 8:39 | safety_warning | "Move the keyboard away from the cooking area to prevent contamination." |
| 12:19 | safety_warning | "Move the plastic bottle away from the hot burner to prevent melting." |

---

## 8. Key Technical Metrics

### 8.1 系统性能

| 指标 | 数值 | 说明 |
|------|------|------|
| 帧处理速度 | ~12.7s/帧 | A100 80GB, Claude VLM |
| GT 生成速度 | ~4-13s/boundary | Gemini-3-flash, 取决于 API 模式 |
| 实时比 | 7.9× | 比实时慢（需优化） |
| API 成本（全量 GT） | ~$1.07 | 86 sessions, 13,076 boundaries |

### 8.2 Agent 评估指标

| 指标 | 数值 | 目标 |
|------|------|------|
| Trigger Timing F1 | 66.7% | 持续提升 |
| Content Quality (Overall) | 4.00/5 | > 4.5/5 |
| Content Accuracy | 4.67/5 | 维持 |
| Content Helpfulness | 3.00/5 | **需要提升** |
| Safety Precision | 100% | 维持 |
| Boundary Coverage | 100% | 维持 |

### 8.3 GT 数据总览

| 指标 | 数值 |
|------|------|
| 总 sessions | 86 |
| 总 boundaries | 13,076 |
| 总介入标注 | 1,781 (13.6%) |
| 介入类型数 | 4 |
| 数据完整率 | 99.82% (仅 23/13,076 缺失) |

---

## 9. Configuration

核心可调参数（`agent/ar_proactive/config.py`）：

```python
# 记忆
working_memory_capacity: int = 8          # L3 滑动窗口大小
max_key_events: int = 50                  # L2 最大关键事件数
compress_events_threshold: int = 30       # 触发事件压缩阈值

# 任务追踪
task_identification_frames: int = 3       # 初始任务识别帧数
step_detection_interval: int = 3          # 步骤检测最小间隔(帧)
step_visual_change_threshold: float = 0.15

# 触发
min_confidence: float = 0.5               # LLM 置信度下限
cooldown_sec: float = 5.0                 # 干预最短间隔

# 模型
model_spec: str = "claude:claude-sonnet-4-5"
```

---

## 10. Usage

### 10.1 运行 Proactive Agent

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate

# 运行 AR Agent (EGTEA 数据)
python3 -m agent.ar_proactive \
  --dataset egtea \
  --session OP01-R01-PastaSalad \
  --trigger periodic \
  --model claude:claude-sonnet-4-5

# 运行评估
python3 -m agent.ar_proactive.eval \
  --dataset egtea \
  --sessions OP01-R01 OP02-R01 \
  --triggers periodic oracle vlm_delta
```

### 10.2 运行 UI Generation Pipeline

```bash
# 基础运行 (Azure GPT-4o)
python3 -m agent.src.pipeline --limit 10

# 切换模型和策略
python3 -m agent.src.pipeline \
  --model gemini:gemini-2.5-flash \
  --strategy v2_smart_glasses \
  --limit 5

# 视觉上下文 + Video Overlay
python3 -m agent.src.pipeline \
  --strategy v3_with_visual \
  --enable-visual \
  --overlay \
  --limit 5

# 多模型对比
python3 -m agent.src.pipeline --compare --scenes general --limit 1
```

### 10.3 启动预览

```bash
python3 -m agent.preview.server --port 8000
# 浏览器访问 http://localhost:8000
```

---

## 11. Project Structure

```
agent/
├── ar_proactive/                    # Stage 1: Streaming Understanding
│   ├── agent.py                     # Observe→Think→Act 主循环
│   ├── config.py                    # 全局配置
│   ├── context.py                   # StreamingContext 定义
│   ├── intervention/
│   │   ├── trigger.py               # TriggerDecider (三通道)
│   │   ├── engine.py                # InterventionEngine
│   │   ├── types.py                 # Intervention 数据结构
│   │   └── prompts.py              # LLM 提示词
│   ├── memory/
│   │   ├── persistent.py            # L1 任务知识
│   │   ├── progress.py              # L2 任务进度
│   │   ├── working.py               # L3 滑动窗口
│   │   ├── manager.py               # 统一协调器
│   │   └── types.py                # 数据结构
│   ├── task/
│   │   ├── tracker.py               # 步骤转换检测
│   │   └── knowledge.py             # 任务知识提取
│   ├── video/
│   │   ├── change_detector.py       # 帧间变化检测
│   │   └── scene_analyzer.py        # 场景理解
│   ├── signals/
│   │   ├── reader.py                # 传感器信号读取
│   │   └── analyzer.py              # 信号分析
│   └── eval/
│       ├── gt_generator.py          # GT 生成
│       ├── runner.py                # 评估循环
│       ├── metrics.py               # 评估指标
│       ├── judge.py                 # LLM-as-Judge
│       └── report.py               # 报告生成
│
├── src/                             # Stage 2: Generative UI
│   ├── pipeline.py                  # 主 Pipeline
│   ├── llm/                         # Multi-LLM 抽象
│   │   ├── base.py, config.py, factory.py
│   │   ├── azure_openai.py, gemini.py, claude.py, vertex.py
│   ├── prompts/                     # Prompt 策略
│   │   ├── v1_baseline.py
│   │   ├── v2_google_gui.py
│   │   ├── v2_smart_glasses.py      # AR 优化 ★
│   │   └── v3_with_visual.py
│   ├── output_validator.py          # A2UI 校验
│   ├── video/                       # 视频处理 + Overlay
│   └── sampling/                    # 数据采样
│
├── preview/                         # 渲染与预览
│   ├── server.py                    # HTTP 服务器
│   ├── templates/                   # HTML 模板
│   └── static/                      # AR UI 样式 + A2UI Core
│
└── example/                         # 100 标注样本
    ├── Task2.1/
    └── Task2.2/
```

---

## 12. Key Design Decisions

回顾整个开发历程中的关键设计决策及其验证结果：

| # | 决策 | 时间 | 验证结果 |
|---|------|------|---------|
| 1 | **语义-渲染分离** | Phase 3 | Video Understanding 和 GUI 生成可独立评估迭代 |
| 2 | **RQ 驱动重构** | Phase 6 | 精确诊断系统缺陷（记忆/触发/UI），避免盲目开发 |
| 3 | **EGTEA action boundaries 作为 proxy GT** | Phase 6 | 解决了"没有干预时机标注"的核心困境 |
| 4 | **Gemini 作为 GT 生成引擎** | Phase 7 | 成本低两个数量级（$1 vs $878），使大规模标注可行 |
| 5 | **区分"可跳过错误"和"需暂停错误"** | Phase 7 | 大规模批处理的关键工程经验 |
| 6 | **Vertex AI + 并行化** | Phase 7 | 从 14h 串行缩短到 7.5h |

---

## 13. Known Limitations & Next Steps

### 已完成

- [x] A2UI GUI 生成 Pipeline（4 种 Prompt 策略）
- [x] Multi-LLM 后端（Azure/Gemini/Claude/Vertex）
- [x] 视频帧提取 + VLM 视觉上下文
- [x] Video UI Overlay 叠加系统
- [x] Preview 预览服务器（HTTP + AR UI 组件）
- [x] Proactive AR Agent（Observe → Think → Act）
- [x] 三层功能性记忆系统
- [x] 三通道干预触发（前瞻/反应/信号）
- [x] 任务理解模块（TaskKnowledgeExtractor + TaskTracker）
- [x] 自动化评估框架（4 维度，Gemini LLM-as-Judge）
- [x] EGTEA 全量 GT 生成（86/86 sessions, 13,076 boundaries）
- [x] GT 数据质量核查与修复

### 当前限制

| 限制 | 原因 | 影响 |
|------|------|------|
| VLM Delta 召回率低 (11.8%) | check_every_n_frames=4, min_interval=5s | 漏掉大量干预机会 |
| 步骤检测跳跃 (1→7→8→132) | LLM 匹配视觉最相似步骤而非顺序最近的 | 步骤转换误判 |
| Helpfulness 最弱 (3.0/5) | 冗余信息、错误步骤优先级 | 内容不够有用 |
| 干预频率过高 (14/min) | 阈值不够精细 | 目标 < 5/min |
| 处理延迟 7.9× | 每次触发需 VLM 推理 ~1-2s | 不适合实时场景 |

### 进行中

- [ ] 步骤检测跳跃修复（TaskTracker 需要顺序约束）
- [ ] 干预频率优化（上下文感知内容过滤）
- [ ] 基于 GT 数据的 Agent 全量评估

### 待开展

**RQ1 — 触发机制**：单信号阈值范式存在根本局限（详见 `docs/research/streaming-video-understanding-progress.md`），探索 Information Gap Agent 架构。

**RQ2 — 记忆管理**：当前三层记忆是基于规则的，探索 Change-Driven Segment Memory（详见 `docs/memory/2026-02-26-change-driven-memory-design.md`）。

**RQ3 — GUI 一致性**：
- 干预内容 → A2UI schema → Preview 渲染（从纯文本到 GUI 组件的转换）
- A2UI 状态机（HIDDEN → ENTERING → ACTIVE → UPDATING → EXITING）
- 位置锚定策略（Screen-Fixed / Object-Anchored / Gaze-Relative）

> **核心瓶颈**：当前系统的干预输出仍是 `Intervention.content` 纯文本字符串，尚未转换为 A2UI 组件。从"文本干预 Agent"升级到"GUI 生成系统"是下一个关键里程碑。
