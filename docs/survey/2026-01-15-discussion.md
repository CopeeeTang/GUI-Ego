---
title: "Generative UI for Smart Glasses - Mentor 讨论议程 v2"
type: meeting-agenda
created: 2026-01-14
status: pending-discussion
related: brainstorm.md, mentor-discussion.md
---

# Generative UI for Smart Glasses - Mentor 讨论议程

> [!info] 文档结构
> 按照 **背景介绍 → 已有工作 → 研究问题 → 实现难点** 的逻辑组织讨论内容。

---

## 讨论概览

| 部分     | 内容                              | 预计时间   |
| ------ | ------------------------------- | ------ |
| Part 1 | 背景：设备、SDK、数据集、场景                | 10 min |
| Part 2 | 已有工作：AI Glasses + Generative UI | 15 min |
| Part 3 | 研究问题：Gap 分析与 RQ 提出              | 15 min |
| Part 4 | 实现难点与待讨论问题                      | 15 min |

---

# Part 1: 背景介绍

## 1.1 两类主要设备

### 设备形态对比

| 类型            | 代表产品                                      | 特点                 | GenUI 需求 |
| ------------- | ----------------------------------------- | ------------------ | -------- |
| **轻量级 AR 眼镜** | Ray-Ban Meta, Vuzix Z100, Snap Spectacles | 单目/双目 HUD、极简、瞥视式交互 | ★★★★★ 高  |
| **沉浸式 MR 头显** | Vision Pro, Quest 3, Magic Leap 2         | 多窗口、空间锚定、手势交互      | ★★☆☆☆ 低  |

**关键差异**：
- AR 眼镜：用户注意力主要在现实世界，UI 是辅助信息
- MR 头显：用户沉浸在虚拟环境，需要完整 UI 系统

### TODO
- [x] 确认研究聚焦 AR 眼镜还是两者都考虑？

---

## 1.2 AR 眼镜 SDK 与 UI 生成范式

### 各家 SDK 现状

| 平台                  | SDK                | UI 格式                   | 开放状态 |
| ------------------- | ------------------ | ----------------------- | ---- |
| **Snap Spectacles** | Lens Studio + SIK  | 3D (FBX) + TypeScript   | ✅ 公开 |
| **Android XR**      | Jetpack Compose XR | Spatial Panels (dp/dmm) | ✅ 公开 |
| **Meta Ray-Ban**    | 未公开 SDK            | 未知（推测内部格式）              | ❌ 封闭 |
| **Magic Leap 2**    | Unity/Unreal       | 3D Spatial              | ✅ 公开 |
|                     |                    |                         |      |

### 研究级眼动追踪设备

| 设备                              | 眼动追踪   | 场景相机            | 精度                      | SDK/API                                                         | 定位         |
| ------------------------------- | ------ | --------------- | ----------------------- | --------------------------------------------------------------- | ---------- |
| **Pupil Labs Neon**             | 200 Hz | 1600×1200 @30Hz | 1.8° (无校准) / 1.3° (校准后) | Real-time API, Plugin API, Python SDK                           | 研究级眼动设备    |
| **Goertek AI Glasses (Sparqi)** | 待确认    | 待确认             | 待确认                     | [Sparqi 平台](http://alpha.goertek.com:4433/sparqi/sparqi.html#/) | 国产 AI 眼镜方案 |

**Pupil Labs Neon 特点**：
- 模块仅重 7.3g，尺寸 35×40×10mm
- 无需校准即可使用（Calibration-free）
- 支持多种镜框适配
- 提供 Pupil Cloud 云端数据管理
- 有 XR Core Package 用于 XR 集成

**Goertek Sparqi 待调研点**：
- [x] 具体硬件规格
- [x] SDK 开放程度
- [ ] 是否有眼动追踪
- [ ] UI 开发范式

### UI 生成范式的不一致性

**问题**：各家 SDK 完全不同，没有统一的 UI 描述格式
- Snap: 3D 模型 + 脚本
- Android XR: 声明式 UI (Compose)
- Meta: 未知

### 统一的物理特性

| 特性 | Snap Spectacles | Android XR | Meta Ray-Ban | 备注 |
|------|----------------|------------|--------------|------|
| **FOV** | ~30° | 设备相关 | ~30° | 相对一致 |
| **舒适观看距离** | ~1m | 0.75-5m | 未公开 | 有共性 |
| **Billboarding** | ✅ | ✅ | ✅ | 设计共识 |
| **最小字号** | 定义 | 14dp/1mm | - | 可抽象 |
| **List 上限** | ≤7项 | - | - | 认知限制 |

### 可抽象的跨平台 UI 原语

```
Smart Glasses UI Primitives
├── Text Card（文本卡片）
├── Icon + Label（图标+标签）
├── Progress Indicator（进度指示）
├── Arrow / Direction（方向指引）
├── Notification Badge（通知徽章）
└── Confirmation Dialog（确认对话框）
```

### 待讨论 TODO
- [ ] 是否需要定义抽象 UI Schema 作为解决策略？
- [ ] 是否需要选择平台做实机验证（Snap 有 SDK）？

---

## 1.3 现有数据集

### 眼动数据集

| 数据集               | 眼动频率   | 规模        | 数据格式                                        | 特点               |
| ----------------- | ------ | --------- | ------------------------------------------- | ---------------- |
| **Project Aria**  | 60 Hz  | 100+ 小时   | MPS 格式（含 SLAM、眼动、手部追踪）                      | Meta AR 眼镜采集，高质量 |
| **Ego4D**         | 可变     | 3,670+ 小时 | 视频 + JSON 标注                                | 大规模，多场景          |
| **Pupil Labs 数据** | 200 Hz | 300+小时    | CSV (timestamp, gaze_x, gaze_y, confidence) | 高精度研究级           |

### 第一人称视频数据集

| 数据集               | 规模        | 场景   | 标注类型                                                     |
| ----------------- | --------- | ---- | -------------------------------------------------------- |
| **Ego4D**         | 3,670+ 小时 | 日常活动 | **视频**: MP4 / **标注**: JSON (3D 扫描、音频转录、Gaze 坐标、文本旁白)     |
| **Ego-Exo4D**     | 1,286 小时  | 多视角  | **视频**: 同步的第一/第三人称 (Aria + GoPro) / **标注**: 动作分割         |
| **Epic-Kitchens** | 100 小时    | 厨房   | **视频**: 1080p, 60fps / **标注**: 动作 (verb, noun)、手/物体 mask |
| **HoloAssist**    | 169 小时    | 操作指导 | **视频**: HoloLens 2 / **标注**: 动作边界、对话转录、错误检测              |

### 数据格式对比

| 数据类型 | 常见格式 | 说明 |
|---------|---------|------|
| **眼动数据** | CSV, JSON, Binary | timestamp, gaze_point (x, y), confidence, pupil_diameter |
| **场景视频** | MP4, MKV | 分辨率 720p-4K, 帧率 24-60fps |
| **3D 信息** | PLY, OBJ, GLTF | 场景重建、物体 mesh |
| **动作标注** | JSON, CSV | 动作类型、起止时间、物体 ID |
| **语音/对话** | WAV + JSON | 音频 + 转录文本 + 时间戳 |

### 关键发现
- ✅ 有场景理解、有眼动数据
- ❌ **缺失 UI Ground Truth**（所有数据集都没有"应该显示什么 UI"的标注）

### 待讨论 TODO
- [ ] 数据来源选择：现有数据集二次开发 vs 与泽宇学姐合作 vs 自采集？

---

## 1.4 应用较广泛的场景

| 场景              | Context 复杂度 | UI 复杂度 | 数据支持          | 研究价值  |
| --------------- | ----------- | ------ | ------------- | ----- |
| **生活/教学指导**     | 高（步骤+状态）    | 中      | Epic-Kitchens | ★★★★★ |
| **专业工作（医疗/工业）** | 高           | 高      | HoloAssist    | ★★★★★ |
| 导航/寻路           | 中           | 低      | 有限            | ★★★☆☆ |
| 购物/比价           | 中           | 中      | 有限            | ★★★☆☆ |
| 翻译/字幕           | 低           | 低      | 有限            | ★★☆☆☆ |

### 待讨论
- [ ] 优先选择哪个场景作为研究切入点？

---

# Part 2: 已有工作

## 2.1 AI Glasses 现有工作

### 学术工作

| 论文                     | 核心贡献                               | 与 GenUI 的关系           |
| ---------------------- | ---------------------------------- | --------------------- |
| **PILAR** (ISMAR 2025) | LLM 生成 context-aware 的个性化解释给 AR 用户 | ⚠️ 生成的是**解释文本**，不是 UI |
| **BIM+RAG AR Agent**   | 多 Agent RAG 框架做建筑导航                | ⚠️ 聚焦**导航**，不是通用 UI   |

### Gap 确认
> **没有发现** LLM 生成 Smart Glasses UI 结构/布局的工作。
> PILAR 最接近，但它生成的是自然语言解释，不是 UI 组件。

---

## 2.2 Generative UI 现有工作

### Google 论文深度解析

**论文**: "LLMs are Effective UI Generators" (Google, 2025)

#### 核心观点

> "Generative UI is a new modality where the AI model generates not only content, but **the entire user experience**. This results in custom interactive experiences, including rich formatting, images, maps, audio and even simulations and games, in response to any prompt (instead of the widely adopted 'walls-of-text')."

**关键创新**：
- 不只是生成文字回答，而是生成**完整的交互体验**
- 包括：格式、图片、地图、音频、甚至模拟和游戏

#### High-level Workflow

```
用户 Query → LLM 理解意图 → 生成 UI 代码 → 后处理修复 → 渲染展示
                ↓
         可能包含：
         - 图像生成/搜索
         - 结构化数据查询
         - 交互逻辑生成
```

#### 可能的 System Prompt 结构

```
你是一个 UI 生成助手。根据用户的请求，生成一个交互式的网页界面。

要求：
1. 使用 HTML/CSS/JavaScript
2. 界面要有良好的视觉设计
3. 信息要清晰、有层次
4. 包含必要的交互元素（按钮、表单等）

用户请求：{query}
```

#### 评估实验设计

| 维度       | 方法                            |
| -------- | ----------------------------- |
| **对比对象** | 生成 UI vs 专家制作 vs Markdown/纯文本 |
| **评估方式** | Pairwise 人类偏好（每个结果 2 位 rater） |
| **评估指标** | ELO 分数（来自国际象棋的排名系统）           |
| **关键发现** | 人类"压倒性地"偏好生成 UI；44% 情况与人类专家相当 |

#### 评估数据构建 (PAGEN Benchmark)

| 维度 | 内容 |
|------|------|
| **任务定义** | 输入用户 query → 输出交互式 UI |
| **数据规模** | 100 个人类专家制作的网站样本 |
| **数据来源** | 专家设计（非众包） |
| **覆盖场景** | 多样化的查询类型 |

#### PAGEN 具体任务类型

基于 Google 论文，PAGEN 覆盖以下典型查询场景：

| 任务类别      | 示例查询                                | 需要生成的 UI 元素    |
| --------- | ----------------------------------- | -------------- |
| **信息展示**  | "Show me the weather in Tokyo"      | 卡片、图标、数据可视化    |
| **数据比较**  | "Compare iPhone 15 and Samsung S24" | 对比表格、并列卡片      |
| **计算工具**  | "Calculate mortgage for $500k loan" | 表单输入、实时计算、结果展示 |
| **地图导航**  | "Directions from SF to LA"          | 嵌入式地图、路线列表     |
| **多媒体内容** | "Show me top 10 movies of 2024"     | 图片网格、评分、简介     |
| **交互式模拟** | "Simulate coin flip 100 times"      | 动画、统计图表、控制按钮   |

**关键特点**：
- 需要**组合多种 UI 元素**（不是单一组件）
- 包含**动态交互**（按钮、表单、实时更新）
- 需要**调用外部 API**（天气、地图、搜索）
- 重视**视觉设计**（布局、颜色、响应式）

**与 Design2Code 的区别**：
- Design2Code: 给定截图 → 还原 UI（视觉复现）
- PAGEN: 给定意图 → 生成 UI（创造性生成）

### 待讨论
- [ ] Smart Glasses 的任务类型应该如何定义？是否类似？
- [ ] ELO 评分具体如何计算？

---

## 2.3 其他 WebUI 评测 Benchmark

### Design2Code Benchmark

| 维度        | 内容                                                    |
| --------- | ----------------------------------------------------- |
| **任务**    | 输入截图 → 输出 HTML/CSS 代码                                 |
| **数据**    | 484 个真实网页截图                                           |
| **自动化指标** | CLIP 视觉相似度 + 元素匹配（block-match, text, position, color） |
| **人工评估**  | 5 位评审 majority vote                                   |

### 主流评测方法总结

| 方法 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **视觉相似度** | CLIP 等模型计算相似度 | 自动化、可扩展 | 可能与人类感知不符 |
| **元素匹配** | 对比 DOM 结构、位置、颜色 | 客观、可量化 | 只评估还原度 |
| **人类偏好** | A/B 对比 + ELO 排名 | 符合真实需求 | 成本高、不可大规模 |
| **任务完成率** | 用户能否用生成的 UI 完成任务 | 功能导向 | 需要真实用户 |

### 待讨论
- [ ] 哪些评测方法可以迁移到 Smart Glasses 场景？

---
smart glasses 有一个simulator？
simulator-to-real
world model场景下做生成的UI，避免工程场景
# Part 3: 研究问题

## 3.1 为什么在 Smart Glasses 场景做 Generative UI？

### 核心理由

1. **一次性生成、用完即弃的 UI 高度契合该场景**
   - WebUI：用户会反复访问同一页面，UI 需要持久化、可复用
   - Smart Glasses：UI 显示几秒后消失，每次都是临时生成，不需要持久化
   - 类比：WebUI 像"建筑"（长期存在），Smart Glasses UI 像"对话"（一次性消费）

2. **注意力分散场景才真正需要 Generative UI**
   > "手机/电脑的 GUI 已经很完善，人的注意力全在屏幕上；只有当注意力无法集中时（智能眼镜）才需要 Generative UI"

   - 手机/电脑：用户愿意花时间浏览、选择、点击
   - Smart Glasses：用户正在做其他事，只能瞥一眼，需要 AI 主动筛选信息


3. **Context 极度丰富，但缺乏明确意图**
   - WebUI：用户输入明确 query（"北京天气"），AI 知道该返回什么
   - Smart Glasses：可能没有明确 query，只有 context（看向窗外、准备出门）
   - 需要 AI 从多模态 context **推断**用户可能需要什么（天气？路线？日程？）


---

## 3.2 Gap 分析：WebUI → Smart Glasses 的迁移

### 三大核心 Gap

| Gap                    | WebUI          | Smart Glasses     |
| ---------------------- | -------------- | ----------------- |
| **Gap 1: 注意力分布**       | 用户注意力 100% 在屏幕 | 注意力分散在现实世界        |
| **Gap 2: Context 丰富度** | 只有用户 query     | 场景 + 注视 + 活动 + 语音 |
| **Gap 3: 真实世界交互**      | 纯虚拟交互          | 与物理环境的交互          |

### Gap 1 详解：注意力不再集中

```
WebUI:
用户 → [全神贯注] → 屏幕

Smart Glasses:
用户 → [主要注意力] → 现实世界
     → [余光/瞥视]  → HUD 显示
```

**影响**：
- UI 必须极简（Glanceable）
- 信息密度极低
- 需要在 < 2秒 内被理解

### Gap 2 详解：更丰富的 Context

| Context 类型 | WebUI 可获取 | Smart Glasses 可获取 |
|-------------|-------------|---------------------|
| 用户 Query | ✅ | ✅ |
| 视觉场景 | ❌ | ✅ 第一人称相机 |
| 注视点 | ❌ | ✅ Eye tracking |
| 用户活动 | ❌ | ✅ 活动识别 |
| 位置信息 | ⚠️ 有限 | ✅ SLAM |

### Gap 3 详解：真实世界交互

- UI 可能需要**锚定到物理物体**（如：看向咖啡机 → 显示使用说明）
- 需要考虑**物理遮挡**
- **空间布局**与真实环境相关

---

## 3.3 研究问题提出

### 大背景

> **Generative UI in Smart Glasses**：在智能眼镜场景下，如何生成合适的 UI？

### 候选研究问题

| RQ | 核心问题 | 关注点 | 技术挑战 |
|----|---------|--------|---------|
| **RQ1: Context-to-UI Mapping** | 从多模态 context 生成用户需要的 UI | **内容**（显示什么） | 多模态理解、意图推断 |
| **RQ2: Attention-Aware UI** | 根据用户注意力状态调整 UI 复杂度 | **复杂度**（如何显示） | 注意力检测、Glanceability |
| **RQ3: Real-World Adaptive UI** | UI 与真实世界环境的适配 | **位置**（在哪显示） | Long-video Understanding + Generative Agent |


问答 推荐 情绪 记录 回忆之类的
### RQ1: Context-to-UI Mapping

**形式化**：
```
f(视觉场景, 注视点, 活动识别, 用户历史, [用户 Query]) → UI 内容 + 布局
```

**与 WebUI 的区别**：
- WebUI 只需处理 Query
- Smart Glasses 需要融合多模态 Context

**示例**：
- 用户在超市看向商品 → 显示价格比较卡片
- 用户在厨房做饭 → 显示下一步操作提示

### RQ2: Attention-Aware UI

**形式化**：
```
用户专注其他事 → 极简 HUD（关键词）
用户主动看向眼镜 → 详细信息卡片
```

**核心约束**：
- **2 秒规则**：用户需要在 2 秒内理解 UI 全部信息
- 根据注意力状态动态调整信息量

**示例**：
- 用户在开会 → 只显示来电者名字
- 用户空闲 → 显示完整来电信息 + 快捷操作

### RQ3: Real-World Adaptive UI

**核心思想**：
- 结合 **Long-video Understanding** 理解用户当前活动
- 使用 **Generative Agent** 决定何时/何处显示 UI

**示例**：
- 用户组装家具（长视频理解识别当前步骤）→ 动态生成下一步指引 UI

### 待讨论
- [ ] 优先做哪个 RQ？RQ1 / RQ2 / RQ3？
- [ ] 是否组合多个 RQ？
- [ ] 是否还有其他 RQ 候选？

---

# Part 4: 实现难点

## 4.1 UI 格式不统一

**问题**：各家 AR 眼镜 SDK 不同，无法直接生成可运行的 UI 代码

**可选方案**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 聚焦单一平台 | 如 Snap Spectacles | 可实机验证 | 泛化性受限 |
| B. 定义抽象 Schema | JSON/DSL 格式 | 通用、可扩展 | 需模拟器验证 |
| C. 多平台适配 | Schema → 平台代码 | 兼顾两者 | 工程量大 |

### 待讨论
- [ ] 该生成什么样的UI
- [ ] 离线生成/在线生成？

---

## 4.2 评估方法需重构

### 现有问题

1. **大量依赖人工评估**
   - WebUI 的 PAGEN 也主要用人工评估
   - 成本高、不可大规模

2. **评估维度不适用**
   - WebUI 重视"视觉还原"
   - Smart Glasses 需要评估"Context 匹配度"、"时机判断"、"Glanceability"

### 需要新增的评估维度

| 维度 | 描述 | 可能的量化方式 |
|------|------|---------------|
| **Context-UI 匹配度** | 生成的 UI 是否与当前场景相关 | 需设计 |
| **时机判断** | 是否在正确的时刻显示/消失 | 需设计 |
| **Glanceability** | 用户能否在 2 秒内理解 | 用户研究 |
| **注意力负荷** | UI 是否干扰用户主任务 | NASA-TLX |

### 待讨论
- [ ] 是否可以设计自动化的 proxy metric？
- [ ] 人工评估的规模和形式？
- [ ] 如何进行人工评估？离线评估吗？
- [ ] 是否结合已有数据集进行人工离线评估

---

## 4.3 数据采集问题

### 三种可选路径

| 路径 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 现有 Ego 数据集二次开发 | Ego4D, Project Aria 等 | 成本低、规模大 | 缺失 UI 标注 |
| B. AI Glasses 数据集二次开发 | HoloAssist 等 | 场景接近 | 规模有限 |
| C. 合作获取 | 如泽宇学姐的眼动数据 | 定制化 | 依赖合作 |
| D. 自采集 | 设计实验采集 | 完全可控 | 成本高 |

### 二次开发所需工作

如果选择 A/B 路径：
1. 选取典型场景片段（~500 个）
2. 人工设计"理想 UI"
3. 标注 UI 出现时机
4. 标注信息优先级

---

## 4.4 Pilot Study 设计方案（基于 P1_YuePan 数据集）

### 目标

利用已有的 P1_YuePan 多模态智能眼镜数据集进行小规模验证：
1. 验证"Context → UI"生成任务在真实智能眼镜场景下的可行性
2. 利用现有推荐标注测试 UI 生成质量
3. 识别 Generative UI 在注意力分散场景下的挑战
4. 为后续大规模研究提供数据和方法依据

### 数据集优势

P1_YuePan 数据集天然适配 Generative UI 研究：

| 维度 | 内容 | 对 Pilot Study 的价值 |
|------|------|----------------------|
| **多模态数据** | 眼动 (200Hz) + 场景视频 + 语音 | 支持 Context-to-UI 的完整输入 |
| **推荐标注** | 330+ 条 AI 推荐 + 12% 用户接受率 | **自带 Ground Truth**（接受 = 好推荐，拒绝 = 坏推荐） |
| **场景多样性** | 6 个真实场景（导航、购物、学习等） | 测试 UI 适配不同场景的能力 |
| **注意力标注** | forbidden_segments（免打扰时段） | 评估注意力感知 UI 调度 |
| **物体锚点** | bbox 标注的视觉物体 | 支持 AR 物体标签生成 |

**核心洞察**：
- 用户只接受了 12% 的推荐 → **说明现有 AI（Gemini）推荐质量需要提升**
- 我们的 Pilot Study 可以验证：能否通过 Generative UI 提高接受率？

---

### Pilot Study 设计

#### 阶段 1: 数据准备与任务设计（1 周）

##### 1.1 场景选择

从 6 个场景中选取 **3 个代表性场景**：

| 场景 | 时间段数 | 推荐总数 | 接受数 | 理由 |
|------|---------|---------|--------|------|
| **场景 A: 户外导航** | 4 段 | ~60 条 | ~8 条 | 高移动性，测试动态 UI |
| **场景 C: 购物选择** | 4 段 | ~80 条 | ~12 条 | 多决策点，测试对比类 UI |
| **场景 E: 学习模式** | 3 段 | ~40 条 | ~6 条 | 高专注度，测试免打扰机制 |

**总计**：11 个时间段（每段 ~5 分钟），~180 条推荐，~26 条被接受的推荐

##### 1.2 任务定义

**主任务**：为每条**被接受的推荐**设计对应的 UI

输入（Context）：
```json
{
  "video_clip": "P1_YuePan_3307.3_3607.3.mp4",
  "gaze_data": "eyetracking_3307.3_3607.3.csv",
  "recommendation": {
    "type": "decision_support",
    "content": "这些橙子中哪一盒看起来更显新鲜？",
    "time": 3474.8
  },
  "scene_context": "超市水果区，用户正在浏览橙子",
  "object_list": [{"instance": "橙子", "bbox": [...]}]
}
```

输出（UI Schema）：
```json
{
  "ui_type": "comparison_card",
  "content": {
    "title": "新鲜度对比",
    "items": [
      {"label": "左侧橙子", "score": 4.5, "reason": "色泽鲜艳"},
      {"label": "右侧橙子", "score": 3.2, "reason": "略有压痕"}
    ]
  },
  "layout": "side_by_side",
  "anchor": "gaze_object",  // 锚定到用户注视的物体
  "display_duration": 3  // 建议显示 3 秒
}
```

##### 1.3 UI 组件库定义

基于场景分析（见 P1_YuePan_task_scenarios_analysis.md），定义初版 UI Schema：

| UI 组件类型 | 适用推荐类型 | Schema 示例 |
|-------------|-------------|-------------|
| **comparison_card** | decision_support | `{items: [{label, score}], layout: "side_by_side"}` |
| **info_card** | object_identification | `{title, content, icon, position}` |
| **step_card** | procedural_guidance | `{steps: [{index, text, status}]}` |
| **timer** | task_assistance | `{duration, type: "pomodoro"}` |
| **ar_label** | context_aware | `{text, anchor: "gaze_object", offset: [x,y]}` |
| **quick_action** | task_assistance | `{action_text, icon, callback}` |

#### 阶段 2: Baseline UI 生成（1 周）

##### 2.1 Baseline 方法

使用 **GPT-4V / Gemini Pro Vision** + **Few-shot Prompting**：

```python
# Prompt 模板
prompt = f"""
你是一个智能眼镜的 UI 设计师。用户正在{scene_context}。

当前时刻，AI 给出了这样的推荐："{recommendation_content}"

请设计一个简洁的 UI 来呈现这个推荐，要求：
1. 用户注意力有限（2-3 秒理解）
2. 不能干扰用户主任务
3. 使用合适的 UI 组件类型（从以下选择）：
   - comparison_card: 对比类信息
   - info_card: 单一信息展示
   - step_card: 步骤指引
   - timer: 计时器
   - ar_label: AR 物体标签
   - quick_action: 快捷操作按钮

输出 JSON 格式的 UI Schema（见示例）。

# Few-shot 示例
示例 1: ...
示例 2: ...

# 当前任务
场景截图: [attached image]
推荐内容: {recommendation_content}
"""
```

##### 2.2 生成规模

- **26 条被接受推荐** → 26 个 UI Schema
- **额外采样 10 条被拒绝推荐** → 作为负样本对比（共 36 个）

#### 阶段 3: 人工评估（1-2 周）

##### 3.1 评估设计

**评估者**：10 人（5 位研究者 + 5 位普通用户）

**评估任务**：
1. 观看 5 秒视频片段（推荐出现前后）
2. 查看生成的 UI（通过模拟 AR 眼镜界面展示）
3. 按 5 个维度评分

##### 3.2 评估维度

| 维度 | 评分方式 | 说明 |
|------|---------|------|
| **相关性** | 1-5 分 | UI 是否与推荐内容一致 |
| **有用性** | 1-5 分 | UI 是否能帮助用户决策 |
| **简洁性** | 1-5 分 | UI 是否足够简洁（2-3 秒理解） |
| **时机适当性** | 1-5 分 | 该时刻是否适合显示 UI（参考 forbidden_segments） |
| **组件合理性** | 1-5 分 | UI 组件类型选择是否合理 |

##### 3.3 对比评估

**被接受推荐 vs 被拒绝推荐**：
- 生成的 UI 评分差异
- 验证假设：被接受推荐应该得到更高的 UI 评分

**免打扰时段**：
- 在 forbidden_segments 时段的 UI 应该得到更低的"时机适当性"分数
- 验证注意力感知机制的必要性

##### 3.4 评估形式

- **线上问卷**（Qualtrics）+ **模拟 AR 界面**
- 每个 UI 由 3 位评估者评分
- **总评估量**：36 个 UI × 3 评估者 = 108 人次

#### 阶段 4: 结果分析（1 周）

##### 4.1 定量分析

| 分析维度 | 指标 | 目标 |
|----------|------|------|
| **UI 质量** | 各维度平均分、标准差 | 平均 > 3.5/5 |
| **推荐区分度** | 被接受 vs 被拒绝的评分差异 | t-test, p < 0.05 |
| **注意力感知** | forbidden_segments 的时机评分 | 显著低于正常时段 |
| **标注一致性** | Inter-rater Reliability (Krippendorff's α) | α > 0.7 |

##### 4.2 定性分析

收集开放式反馈：
- UI 组件类型是否覆盖全场景？
- 哪些推荐类型最难设计 UI？
- 是否需要引入新的 UI 组件？

##### 4.3 案例研究

深入分析 **3-5 个典型案例**：
- **成功案例**：评分高且与用户接受推荐一致
- **失败案例**：评分低或 UI 组件选择不当
- **边界案例**：forbidden_segments 中的推荐

**预期产出**：
- [ ] 验证 Context-to-UI 任务在智能眼镜场景的可行性
- [ ] 完善 UI Schema 定义（v2.0）
- [ ] 识别 baseline 方法的主要问题（如：是否能正确选择 UI 组件？）
- [ ] 确定需要改进的方向（更好的注意力感知？更丰富的 UI 类型？）

---

### Pilot Study 时间线

```
Week 1:    数据准备 + UI Schema 定义
Week 2:    Baseline UI 生成（36 个）
Week 3-4:  人工评估（108 人次）
Week 5:    结果分析 + 撰写报告
```

**总计**：5 周

---

### Pilot Study 成功标准

| 指标 | 阈值 | 说明 |
|------|------|------|
| **UI 质量** | 平均 > 3.5/5 | Baseline 已有一定质量 |
| **推荐区分度** | 接受 vs 拒绝差异显著 | 验证 UI 能反映推荐质量 |
| **注意力感知** | forbidden 时段评分显著降低 | 验证免打扰机制必要性 |
| **标注一致性** | Krippendorff's α > 0.7 | 评估维度可靠 |

**决策路径**：
- ✅ **达到标准** → 进入大规模数据标注 + 方法研发（RQ1: Context-to-UI Mapping）
- ⚠️ **部分达标** → 调整 UI Schema 或评估维度，迭代 Pilot v2
- ❌ **未达标** → 重新审视任务定义或更换数据集

### 待讨论
- [ ] 是否采用 P1_YuePan 作为 Pilot Study 数据集？
- [ ] UI Schema 定义是否涵盖所有场景？
- [ ] 评估维度是否需要调整？

---

# 附录：待讨论问题清单

## Part 1: 背景
- [ ] Q1: 研究聚焦 AR 眼镜还是两者都考虑？
- [ ] Q2: 是否定义抽象 UI Schema 作为贡献？
- [ ] Q3: 选择哪个平台做实机验证？
- [ ] Q4: 数据来源选择？
- [ ] Q5: 优先选择哪个场景？

## Part 2: 已有工作
- [ ] Q6: PAGEN 的具体任务类型？
- [ ] Q7: 哪些评测方法可迁移？

## Part 3: 研究问题
- [ ] Q8: 还有哪些"为什么要做"的理由？
- [ ] Q9: 优先做哪个 RQ？
- [ ] Q10: 是否组合多个 RQ？
- [ ] Q11: 还有其他 RQ 候选吗？

## Part 4: 实现难点
- [ ] Q12: UI 格式方案选择？
- [ ] Q13: 能否设计自动化评估指标？
- [ ] Q14: 人工评估的规模和形式？
- [ ] Q15: 数据策略选择？
- [ ] Q16: 标注规范如何设计？
- [ ] Q17: 是否先做 pilot study？

---

# 讨论后行动项

| 决定项 | 结论 | 负责人 | 截止日期 |
|--------|------|--------|---------|
| 设备选择 | | | |
| 场景选择 | | | |
| 核心 RQ | | | |
| UI 格式方案 | | | |
| 评估方法 | | | |
| 数据策略 | | | |
