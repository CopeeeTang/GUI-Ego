---
title: "Generative UI for Smart Glasses - Brainstorm Session"
type: research-brainstorm
created: 2026-01-13
updated: 2026-01-13
status: in-progress
tags: [generative-ui, smart-glasses, ar, research-direction]
---

# Generative UI for Smart Glasses - 研究方向探索

> [!note] 会话背景
> 基于已有的两份调研文档（Survey_Smart glasses.md 和 survey.md），探索 Smart Glasses 场景下 Generative UI 的研究方向。

---

## 1. 核心洞察

从调研中识别的交叉点：

```
Generative UI (Google论文)     ×     Smart Glasses GUI
         ↓                                  ↓
   Web页面生成                    空间交互、注意力分散
   静态2D UI                      动态3D + 物理锚定
   用户全神贯注                   用户注意力在现实世界
```

**关键洞察**（来自 CLAUDE.md）：
> "手机/电脑的 GUI 已经很完善，人的注意力全在屏幕上；只有当注意力无法集中时（智能眼镜）才需要 Generative UI"

key：从常见的WebUI场景迁移到Smart Glasses场景下，哪一个点是最核心的改变。当下Generative UI最核心的点在“Generative UI is a new modality where the AI model generates not only content, but **the entire user experience**. This results in custom interactive experiences, including rich formatting, images, maps, audio and even simulations and games, in response to any prompt (instead of the widely adopted “walls-of-text”).” Smart Glasses与WebUI最大的区别：“1.在不同场景下，注意力不再全程集中在屏幕上了。2.有了更加丰富的来自真实生活的信息，相比于context增加”
引出两个关键的研究问题：
**RQ1：Context-to-UI Mapping** 
在用户下达query的情况下，如何从多模态的context（场景，注视，活动）下，生成用户需要的UI。
**RQ2：Attention-Aware UI Complexity**
如何根据用户的注意力状态（可能根据用户的活动场景判断）动态调整生成的UI复杂度

引出几个子问题：1.常见的AI Glasses使用场景（场景分为日常生活 工作 等？任务类型分问答 推荐 情绪 记录 回忆
2

---

## 2. 第一轮提问与回答

### Q1: 目标设备形态

| 选项               | 设备代表                     | GUI特点       | 技术挑战        |
| ---------------- | ------------------------ | ----------- | ----------- |
| **A. 轻量级AR眼镜** ✅ | Ray-Ban Meta, Vuzix Z100 | 单目HUD、极简、瞥视 | 算力受限、信息密度极低 |
| B. 沉浸式MR头显       | Vision Pro, Quest 3      | 多窗口、空间锚定、手势 | 3D布局、遮挡处理   |
| C. 两者都考虑         | -                        | 自适应UI复杂度    | 需要统一抽象层     |

**用户选择**：A - 轻量级 AR 眼镜
- 理由：需要生成的 UI 更少，用户注意力需求更低
- 沉浸式 VR 头显需要完整 UI 或在已有 UI 上操作，不太需要"生成式"

Key：应用场景选择AR眼镜（Ray-Ban Meta）还是MR头显（Vision Pro），选择AR眼镜场景先做
### Q2: Generative UI 的"生成"边界

**用户选择**：A + B
- **A. 生成 UI 的代码/结构**（类似 Google 论文）
- **B. 生成 UI 的内容/布局决策**（LLM 决定显示什么、在哪里显示）

**不关注**：交互逻辑（厂商定义的问题）

**生成边界明确为**：
- What: 生成什么内容（基于 context）
- Where: 在什么位置显示（基于 gaze/attention/voice）
- How: 以什么形式输出

Key：关于具体生成UI内容的讨论：1.生成UI的代码/架构 2.生成UI的位置 。核心挑战：各家AI Glasses的UI架构没有统一；同时各家眼镜的物理特性也没有完全统一

### Q3: 与其他研究方向的协同

**用户选择**：暂不考虑与 Long Video Understanding 和 GUI Continuous Action 的联动

---

## 3. 领域调研结果

### 3.1 Benchmark 现状

| 领域 | Benchmark 现状 | 备注 |
|------|---------------|------|
| **Web UI 生成** | ✅ Design2Code, PAGEN | Google 论文使用 |
| **Mobile GUI Agent** | ✅ ScreenSpot, AndroidWorld | 主要评估操作准确率 |
| **Smart Glasses UI** | ❌ **无专门 benchmark** | 仅有零散的 user study |

**结论**：目前没有发现专门针对 Smart Glasses UI 生成或评估的 benchmark。

### 3.2 各平台 UI 开发格式

| 平台 | SDK/工具 | UI 格式 | UI 原语 |
|------|---------|---------|---------|
| **Snap Spectacles** | Lens Studio + SIK | 3D (FBX) + 脚本 | Hand Menu, Buttons, Lists (≤7项) |
| **Android XR** | Jetpack Compose XR | Spatial Panels (dp/dmm) | Panels, Orbiters, Dialogs |
| **Meta Ray-Ban Display** | **未公开 SDK** | 未知（推测是 Meta 内部格式） | Text, Multimedia, Maps |
| **Magic Leap 2** | Unity/Unreal | 3D Spatial | 各种 3D 组件 |

**关键发现**：
- Meta Ray-Ban Display 目前**没有公开的开发者 SDK**
- Snap Spectacles 使用 Lens Studio（3D 格式 + TypeScript 脚本）
- Android XR 使用标准 Android 开发范式（Jetpack Compose 扩展）

Key：各家的UI展示风格并不一致，对应7.5
### 3.3 相关学术工作

| 论文                                | 核心贡献                               | 与研究方向的关系              |
| --------------------------------- | ---------------------------------- | --------------------- |
| **PILAR** (ISMAR 2025)            | LLM 生成 context-aware 的个性化解释给 AR 用户 | ⚠️ 生成的是**解释文本**，不是 UI |
| **BIM+RAG AR Agent** (ISMAR 2025) | 多 Agent RAG 框架做建筑导航                | ⚠️ 聚焦**导航**，不是通用 UI   |


**Gap 确认**：
> **没有发现** LLM 生成 Smart Glasses UI 结构/布局的工作。PILAR 最接近，但它生成的是自然语言解释，不是 UI 组件。

Key：尚未有做该方面的工作的：PILAR 最接近，但它生成的是自然语言解释，不是 UI 组件。

### 3.4 评估指标现状

现有 AR/HUD 评估主要是 **user study** 形式：

| 指标类型 | 具体指标 | 来源 |
|---------|---------|------|
| **任务效率** | Task Completion Time | 各种 user study |
| **认知负荷** | NASA-TLX, Inattentional Blindness | HUD 安全研究 |
| **可用性** | SUS (System Usability Scale) | 标准 HCI |
| **视觉舒适度** | Visual Fatigue, VAC (辐辏调节冲突) | AR 人因工程 |

**缺失的评估维度**：
- UI 生成质量的自动化评估
- Context-UI 匹配度的量化指标
- 信息密度与注意力负荷的平衡度量

key：1.评估多为人工评估，自动化的评估方法很重要，能大大减轻人类劳动量
2.缺乏对Context-UI 匹配度的量化指标
3.缺乏对信息密度与注意力负荷的平衡度量

---

## 4. 当前研究定位

```
目标场景：轻量级 AR 眼镜（Ray-Ban Meta, Vuzix Z100 类）
生成边界：
  ├── What: 生成什么内容（基于 context）
  ├── Where: 在什么位置显示（基于 gaze/attention）
  └── How: 以什么形式输出（格式/结构）

不关注：交互逻辑（厂商定义）
```

---

## 5. 待回答的关键问题

### Q5: 研究定位选择

既然领域如此早期，倾向于哪种贡献类型？

| 选项 | 贡献类型 | 工作量 | 影响力 |
|------|---------|--------|--------|
| A. **方法创新** | 设计 LLM-based Smart Glasses UI 生成系统 | 中 | 取决于效果 |
| B. **Benchmark 建设** | 定义评估标准 + 创建数据集 | 高 | 长期引用 |
| C. **两者结合** | 方法 + 小规模验证性 benchmark | 中高 | 平衡 |
Key：不应该局限于一个研究定位，而是我们研究的方向是什么

### Q6: Context 定义

"在什么 context 下生成什么 UI"——context 包括哪些？

| Context 类型 | 示例 | 获取方式 |
|-------------|------|---------|
| **用户意图** | "我想知道这个产品的价格" | 语音/手势 |
| **注视目标** | 用户正在看一本书 | Gaze tracking |
| **场景理解** | 用户在超市/办公室/厨房 | SLAM + 场景分类 |
| **时间/活动** | 用户正在做饭/开会 | 活动识别 |
| **用户状态** | 注意力分散/专注 | 生理信号 |

Key：讨论目前的场景下/基于目前数据集开发。1.跟泽宇学姐合作，她最近在采一个眼动数据集 2.基于现有的AI Glasses数据集二次开发
### Q7: 输出格式的务实选择

既然 Meta Ray-Ban 没有公开 SDK：

- **A. 聚焦 Snap Spectacles**（有 Lens Studio，可实机验证）
- **B. 定义抽象的 UI Schema**（与具体平台无关，可模拟验证）
- **C. 使用 Android XR 作为验证平台**（标准化程度高）

Key：需要讨论没有统一的范式下该输出什么样的UI

---

## 6. 下一步行动

- [x] 回答 Q5-Q7 以明确研究定位
- [x] 深入调研 PILAR 论文的技术细节
- [ ] 调研 Snap Lens Studio 的 UI 规范
- [ ] 探索可能的 benchmark 设计
- [ ] 确定具体的技术方案

---

## 7. 第二轮提问与分析：Method vs Benchmark

### 用户核心问题

1. **先提出生成范式（模型）还是先提出 Benchmark？**
2. **没有 benchmark 来评估该场景下的生成质量，WebUI benchmark 是怎么构造的？**
3. **如何定义 Smart Glasses 场景下 UI 生成质量的好坏？**
4. **Benchmark 测的是什么能力？评测的是模型还是系统设计方法？**
5. **各家厂商有没有共性范式？是否该去评测这个共性范式？**

### 7.1 Method First vs Benchmark First

| 策略 | 适用场景 | 优点 | 风险 |
|------|---------|------|------|
| **Method First** | 问题可解但无人做过 | 快速证明可行性，定义问题边界 | 评估不充分，难以与后续工作对比 |
| **Benchmark First** | 问题定义清晰，需要标准化 | 长期引用，吸引社区 | 如果没有 baseline 方法，可能无人使用 |
| **Method + 小 Benchmark** | 新领域开拓 | 既证明可行，又提供评估基础 | 工作量较大 |

**建议**：由于 Smart Glasses GenUI 是**全新领域**（无现有方法、无现有 benchmark）：

> **Method First + 验证性 Benchmark**
>
> 即：提出一个生成范式 → 构建小规模评估集验证效果 → 开放评估集供后续研究使用

类似案例：
- **CLIP** 提出方法 + 自建评估 → 后来成为标准
- **Google Generative UI** 提出方法 + PAGEN 评估 → 一篇论文完成

Key：明确现在要做的研究工作是什么方向

### 7.2 现有 WebUI Benchmark 结构解析

#### Design2Code Benchmark

| 维度 | 内容 |
|------|------|
| **任务定义** | 输入截图 → 输出 HTML/CSS 代码 |
| **数据集** | 484 个真实网页截图 |
| **评估对象** | 模型生成的代码 vs 原始代码 |
| **自动化指标** | CLIP 高层视觉相似度 + 低层元素匹配（block-match, text, position, color） |
| **人工评估** | 5 位评审 majority vote 对比偏好 |

#### PAGEN Benchmark (Google)

| 维度 | 内容 |
|------|------|
| **任务定义** | 输入用户 query → 输出交互式 UI |
| **数据集** | 人类专家制作的 100 个网站样本 |
| **评估对象** | 生成的 UI vs 专家制作 vs baseline (Markdown/文本) |
| **指标** | ELO 分数 + Pairwise 人类偏好 |
| **人工评估** | 每个结果 2 位 rater，A/B 对比 |

Key：ELO分数是什么？
#### 关键观察

```
WebUI Benchmark 评测的是：
┌─────────────────────────────────────────────────────┐
│  1. 模型能力（能否生成符合要求的 UI）                   │
│  2. 视觉还原度（生成结果与目标的相似程度）               │
│  3. 用户偏好（人类是否喜欢这个结果）                    │
└─────────────────────────────────────────────────────┘

但 WebUI 假设的场景是：
- 用户全神贯注看屏幕
- 有明确的 query/需求
- UI 是静态的"最终产物"
```

### 7.3 Smart Glasses 场景的本质差异

> **WebUI 和 Smart Glasses 的根本区别**：
> - WebUI 是 "用户找 UI"（用户有明确需求，去网页获取信息）
> - Smart Glasses 是 "UI 找用户"（系统感知 context，主动推送信息）

| 维度 | WebUI | Smart Glasses |
|------|-------|---------------|
| **用户注意力** | 100% 在屏幕 | 分散在现实世界 |
| **触发方式** | 用户主动 query | 系统主动感知 context |
| **UI 寿命** | 持久（网页） | 一次性（Disposable） |
| **信息密度** | 高（详尽展示） | 极低（Glanceable） |
| **成功标准** | 视觉还原 + 功能完整 | 恰当时机 + 最少干扰 + 快速理解 |

Key：该评估的除了视觉还原的能力，还需要评估生成时机和生成合适度两个维度的能力，但这个应该对应的是两个研究，对应7.4
### 7.4 Smart Glasses GenUI 应该评测的能力维度

| 能力维度 | WebUI 关注 | Smart Glasses 应关注 |
|---------|-----------|---------------------|
| **视觉还原** | ★★★★★ | ★☆☆☆☆（不重要） |
| **Context 理解** | ★★☆☆☆ | ★★★★★ |
| **信息筛选** | ★☆☆☆☆ | ★★★★★（显示什么、不显示什么） |
| **时机判断** | ☆☆☆☆☆ | ★★★★★（何时出现/消失） |
| **注意力负荷** | ★★☆☆☆ | ★★★★★（不能干扰用户） |
| **Glanceability** | ☆☆☆☆☆ | ★★★★★（一眼看懂） |

#### 可能的评测任务

| 任务类型 | 输入 | 输出 | 评估标准 |
|---------|------|------|---------|
| **Context-to-UI Relevance** | (场景, 用户状态, 意图) | UI | 生成的 UI 是否回应了用户的真实需求 |
| **Information Density** | 完整信息 | 精简 UI | 保留了关键信息，去除了冗余 |
| **Timing Appropriateness** | 视频流 + 事件 | 出现时机 | 在正确的时刻出现/消失 |
| **Attention-Awareness** | 用户注意力状态 | UI 复杂度 | 根据用户忙碌程度调整 UI 量级 |
| **Glanceability** | UI 设计 | 理解时间 | 用户能在 X 秒内理解 |

### 7.5 各家厂商的"共性范式"

#### 硬件层面的共性

| 共性 | 描述 |
|------|------|
| **有限 FOV** | 都在 30-50 度之间 |
| **加法混色** | OST 设备都无法显示真黑 |
| **中心化布局** | 关键信息必须在视野中央 |
| **多模态输入** | 都支持语音 + 某种形式的手势/注视 |

#### UI 设计原则的共性

| 原则 | Snap | Android XR | Meta | Magic Leap |
|------|------|------------|------|------------|
| **Billboarding** | ✅ | ✅ | ✅ | ✅ |
| **最小字号** | 定义 | 14dp / 1mm | - | 定义 |
| **舒适距离** | ~1m | 0.75-5m | - | 0.37m+ |
| **List 上限** | ≤7项 | - | - | - |

#### UI 原语的共性（可抽象的跨平台 Schema）

```
Smart Glasses UI Primitives (抽象层)
├── Text Card（文本卡片）
├── Icon + Label（图标+标签）
├── Progress Indicator（进度指示）
├── Arrow / Direction（方向指引）
├── Notification Badge（通知徽章）
└── Confirmation Dialog（确认对话框）
```

### 7.6 研究定义建议

**推荐路径**：

```
B (生成范式) + C (评测任务) + 轻量 A (抽象 Schema)

具体来说：
1. 定义一个抽象的 UI 表示格式（不依赖具体平台）
2. 提出 Context → UI 的生成方法
3. 构建评测任务来验证方法的有效性
```

---

## 8. 第三轮待回答问题

### Q8: 评测的数据来源

| 选项 | 方法 | 可行性 | 质量 |
|------|------|--------|------|
| A. **众包标注** | 给定 context，让人设计"理想 UI" | 高成本 | 高 |
| B. **专家设计** | 雇佣 UX 设计师制作 gold standard | 中成本 | 高 |
| C. **合成数据** | LLM 生成 context + UI 对 | 低成本 | 低-中 |
| D. **从现有 App 抽取** | 分析现有 Smart Glasses App 的 UI 模式 | 中 | 中 |

### Q9: 评测形式

| 形式 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. **自动化指标** | 定义可计算的 metric | 可大规模 | 可能与人类感知不符 |
| B. **人类偏好** | A/B 对比让人选 | 符合实际 | 成本高、不可大规模 |
| C. **模拟用户研究** | 在模拟器中测试 UI 效果 | 接近真实 | 实现复杂 |

### Q10: 核心场景选择

| 场景 | Context 复杂度 | UI 复杂度 | 研究价值 |
|------|--------------|----------|---------|
| 导航/寻路 | 中（位置+目的地） | 低（箭头） | 中 |
| 购物/比价 | 中（商品+偏好） | 中（卡片） | 中 |
| 翻译/字幕 | 低（语音） | 低（文本） | 低 |
| 烹饪/操作指导 | 高（步骤+状态） | 中（指示） | 高 |
| 社交/通知 | 低（事件） | 低（badge） | 低 |
| 专业工作（医疗/工业） | 高 | 高 | 高 |

Key：评估数据制作和评估场景的选择

---

## 10. 第三轮讨论：明确核心研究问题

> [!note] 讨论背景
> 用户反馈核心研究问题还没确定，通过提问来澄清研究方向。

### 10.1 问题层次关系澄清

用户选择 **Context 理解问题** 作为核心痛点。以下是各问题之间的层次关系：

```
Context 理解 (A) ← 用户选择的核心问题
├── 「该不该显示 UI」 ← 时机判断 (C)
├── 「该显示什么」   ← 信息筛选 (B) - A 的细化/子问题
└── 「该怎么显示」   ← 格式生成 (D) - 工程问题
```

**关于 B（信息筛选）**：是 A（Context 理解）的子问题。当理解了 context 后，需要决定从所有可能的信息中筛选出最相关的。在 Smart Glasses 场景尤为重要，因为屏幕空间极其有限（FOV 30-50°）。

**关于 D（格式生成）**：用户认为是工程问题。各家 SDK 不同（Snap/Android XR/Meta），需要考虑如何统一或选择平台。

### 10.2 用户的初步贡献定位

用户表述：
> "我的贡献是提出了 Generative UI in Smart Glasses 的应用，因为这个场景非常适合 Generative UI（注意力分散时才需要生成式 UI）"

**问题**：仅"提出一个新场景"在学术上不够，因为：
1. 没有回答一个科学问题（只是说"这里适合用"，但没说"怎么做"或"为什么现有方法做不到"）
2. 没有技术创新
3. 不能被评估和复现


### 10.3 核心洞察与研究问题重构

**核心洞察**：
> Smart Glasses 场景下，用户注意力分散，UI 需要主动"找"用户

**与 WebUI 的本质区别**：

| WebUI (Google 论文) | Smart Glasses |
|---------------------|---------------|
| 用户给出明确 query | 没有明确 query，需要从 context 推断意图 |
| UI 可以很复杂（用户会仔细看） | UI 必须极简（用户只会瞥一眼） |
| 一次生成完成 | 需要根据 context 变化动态更新 |


**三个候选研究问题**：

#### RQ 候选 1: Context-to-UI Mapping
如何从多模态 context（场景、注视、活动）推断用户需要什么 UI？**

形式化：
```
f(视觉场景, 注视点, 活动识别, 用户历史) → UI 内容 + 布局
```

#### RQ 候选 2: Attention-Aware UI Complexity
> **如何根据用户的注意力状态动态调整生成 UI 的复杂度？**

适应性问题：
```
用户专注做其他事 → 极简 HUD（只显示关键词）
用户主动看向眼镜 → 详细信息卡片
```

#### RQ 候选 3: Glanceable UI Generation
> **如何生成能在 < 2 秒内被理解的极简 UI？**

信息压缩问题：
```
完整信息 → 极简表示（保留关键 + 去除冗余）
```

Key：即使有用户的query，也需要生成合适的UI

### 10.4 关于数据与评估

**用户的数据设想**：
- 可以获取 Smart Glasses 真实使用数据（视频 + 用户需求）
- 没有生成的 UI（ground truth）
- 评估方式：离线生成 UI → 人工评判

**待澄清**：
- "真实使用数据"具体是什么形式？（第一人称视频？活动标签？意图标注？）
- 如果没有 ground truth（理想 UI），生成出来的 UI 要怎么评估"好不好"？

Key：基于现有的数据（dataset/AI glasses）做一个探索
### 10.5 第三轮待回答问题

#### Q11: 倾向于哪个研究问题？

| 候选 RQ | 核心问题 | 技术挑战 |
|---------|---------|---------|
| RQ1: Context-to-UI Mapping | 从 context 推断意图 | 多模态理解、意图推断 |
| RQ2: Attention-Aware Complexity | 动态调整 UI 复杂度 | 注意力检测、复杂度度量 |
| RQ3: Glanceable Generation | 信息压缩到极简 | 信息筛选、可读性优化 |

或者用户有自己的表述？

#### Q12: "真实使用数据"具体是什么？

| 数据形式 | 是否可获取？ |
|---------|-------------|
| 第一人称视频（用户看到什么） | ❓ |
| 用户正在做什么（活动标签） | ❓ |
| 用户当时的需求/意图（ground truth） | ❓ |
| 理想的 UI 应该是什么（ground truth） | ❓ |

#### Q13: 关于 UI 范式统一

| 选项 | 描述 | 优缺点 |
|------|------|--------|
| A. 聚焦一个平台 | 如 Snap Spectacles（有 SDK 可验证） | 可实机验证，但泛化性受限 |
| B. 定义抽象 Schema | 平台无关 | 通用性强，但只能模拟器/人工评判 |
| C. 其他想法 | ❓ | - |

---

## 11. 下一步行动

- [ ] 回答 Q11-Q13 以明确核心研究问题
- [ ] 回答 Q8-Q10 以明确 benchmark 设计
- [ ] 定义抽象 UI Schema
- [ ] 设计生成范式的技术方案
- [ ] 确定数据收集策略
- [ ] 规划实验验证路径

---

## 12. Smart Glasses 数据集与 Benchmark 调研

> [!note] 调研目的
> 评估现有数据集是否可用于 Generative UI 研究的二次开发。

### 12.1 Egocentric Video 数据集

| 数据集 | 规模 | 标注类型 | 任务类型 | License | 二次开发 |
|--------|------|---------|---------|---------|---------|
| **Ego4D** | 3,670+ 小时, 923 参与者, 74 地点 | 3D 扫描、音频、**注视(Gaze)**、立体视觉、文本旁白 | Episodic Memory, Hand-Object Interaction, Social, Forecasting | 需签署协议，~48h 审批 | ✅ 研究友好 |
| **Ego-Exo4D** | 1,286 小时, 740 佩戴者, 123 地点 | 同步第一/第三人称视角 (Aria眼镜 + 4-5 GoPro) | Keystep Recognition, Proficiency Estimation, Pose | 需申请 | ✅ 多视角优势 |
| **Epic-Kitchens** | 100 小时, 45 厨房, 20M 帧, 90K 动作片段 | 97 动词类, 300 名词类, 多语言旁白, 手/物体 mask | Action Recognition/Detection/Anticipation, Retrieval | **CC BY-NC 4.0** | ✅ 明确开源 |
| **EgoSchema** | 250+ 小时, 5000+ 问答对 | 人工标注的多选题 (5选1), 3分钟视频片段 | 长视频问答 (Long-form Video QA) | 公开 | ✅ Benchmark 友好 |

### 12.2 AR/Smart Glasses 专用数据集

| 数据集 | 规模 | 场景 | 传感器/标注 | 任务 | 二次开发 |
|--------|------|------|------------|------|---------|
| **Project Aria** (Meta) | 100+ 小时真实数据 + 合成数据 | 日常活动 | **60Hz 眼动追踪**, 3D 边界框标注 | Hand-Object Interaction, Full-body Motion | ✅ 开放研究 |
| **HoloAssist** | 169 小时, 350 组指导者-执行者配对 | 物理操作任务 (组装等) | 动作标注、对话标注、**错误检测标注** | 操作指导、错误识别 | ✅ 宽松 License |
| **Aria Digital Twin** | 合成数据 | 室内场景 | 精确 3D 重建、物体标注 | 场景理解、定位 | ✅ 研究用途 |

### 12.3 关键发现：与 Generative UI 的匹配度

| 数据集 | 有场景理解 | 有用户意图 | 有注视数据 | 有 UI 标注 | GenUI 适用性 |
|--------|-----------|-----------|-----------|-----------|-------------|
| **Ego4D** | ✅ 场景分类 | ⚠️ 任务级别 | ✅ Gaze 数据 | ❌ 无 | ★★★★☆ 高 |
| **Ego-Exo4D** | ✅ | ⚠️ 任务级别 | ✅ | ❌ 无 | ★★★☆☆ 中 |
| **Epic-Kitchens** | ✅ 厨房场景 | ✅ 动作意图 | ❌ 无 | ❌ 无 | ★★★☆☆ 中 |
| **Project Aria** | ✅ | ⚠️ | ✅ **60Hz Gaze** | ❌ 无 | ★★★★★ **最高** |
| **HoloAssist** | ✅ 操作场景 | ✅ 指导意图 | ❌ | ❌ 无 | ★★★★☆ 高 |

### 12.4 二次开发可行性分析

#### ✅ 可行的方案

**方案 1: 基于 Ego4D/Project Aria 构建 Context-UI 数据集**
```
输入：第一人称视频帧 + Gaze 数据 + 场景标注
二次标注：为典型场景人工设计"理想 UI"
输出：(Context, Ideal UI) 对
```

**方案 2: 基于 HoloAssist 构建操作指导 UI 数据集**
```
输入：操作视频 + 错误检测标注 + 对话标注
二次标注：将对话/指导转化为 UI 表示
输出：(操作状态, 指导 UI) 对
```

**方案 3: 基于 Epic-Kitchens 构建烹饪场景 UI 数据集**
```
输入：厨房视频 + 动作标注
二次标注：为每个动作阶段设计辅助 UI
输出：(烹饪步骤, 辅助 UI) 对
```

#### ❌ 缺失的关键要素

| 缺失要素 | 说明 | 解决方案 |
|---------|------|---------|
| **UI Ground Truth** | 所有数据集都没有"应该显示什么 UI"的标注 | 需人工二次标注 |
| **实时意图标注** | 用户"当下想知道什么"的细粒度标注缺失 | 可从任务/对话推断 |
| **注意力状态** | "用户是否在看 UI"的标注缺失 | 可从 Gaze 数据推断 |

### 12.5 推荐的数据策略

```
推荐路径：Project Aria + HoloAssist 组合

1. Project Aria 提供：
   - 高质量 Gaze 数据 (60Hz)
   - 真实 Smart Glasses 视角
   - 3D 场景理解基础

2. HoloAssist 提供：
   - 指导场景的对话标注
   - 错误检测标注 → 可转化为"何时需要 UI 介入"
   - 明确的任务结构

3. 二次标注工作量：
   - 选取 ~500 个典型场景片段
   - 人工设计"理想 UI"（使用抽象 Schema）
   - 标注"UI 出现时机"和"信息优先级"
```

### 12.6 License 总结

| 数据集 | License 类型 | 商业使用 | 二次分发 | 研究二次开发 |
|--------|-------------|---------|---------|-------------|
| Ego4D | 定制协议 | ❌ | ❌ | ✅ |
| Epic-Kitchens | CC BY-NC 4.0 | ❌ | ✅ (署名) | ✅ |
| Project Aria | 研究协议 | ❌ | ❓ | ✅ |
| HoloAssist | 宽松协议 | ❓ | ✅ | ✅ |

**结论**：所有主要数据集都**允许研究用途的二次开发**，但需签署协议。如果构建新 Benchmark 并开源，需注意原始数据集的分发限制。

---

## 参考资料

### 数据集官网
- [Ego4D](https://ego4d-data.org/) - Meta 大规模第一人称视频数据集
- [Ego-Exo4D](https://ego-exo4d-data.org/) - 多视角第一/第三人称数据集
- [Epic-Kitchens](https://epic-kitchens.github.io/2024) - 厨房场景第一人称数据集
- [EgoSchema](https://egoschema.github.io/) - 长视频问答 Benchmark
- [Project Aria](https://www.projectaria.com/datasets/) - Meta AR 眼镜数据集
- [HoloAssist](https://holoassist.github.io/) - AR 操作指导数据集

### 调研文档
- [Survey_Smart glasses.md](2026-01-08%20Survey_Smart%20glasses.md) - 智能眼镜 GUI 调研
- [survey.md](2026-01-08%20survey.md) - Generative UI 综述

### 相关论文
- **PILAR** (arXiv:2512.17172) - LLM for context-aware AR explanations
- **BIM+RAG AR Agent** (arXiv:2508.16602) - Multi-agent RAG for AR navigation

### 开发者文档
- [Snap Spectacles Design Best Practices](https://developers.snap.com/spectacles/best-practices/design-for-spectacles/design-best-practices)
- [Android XR Spatial UI](https://developer.android.com/design/ui/xr/guides/spatial-ui)
