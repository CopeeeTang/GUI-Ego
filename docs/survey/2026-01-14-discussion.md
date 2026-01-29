---
title: "Generative UI for Smart Glasses - Mentor 讨论议程"
type: meeting-agenda
created: 2026-01-14
status: pending-discussion
related: brainstorm.md
---

# Generative UI for Smart Glasses - Mentor 讨论议程

> [!info] 文档目的
> 本文档整理了研究方向探索过程中需要与 mentor 讨论确定的核心问题，基于 brainstorm.md 中的关键批注整理而成。

---

## 讨论概览

| 议题 | 核心问题 | 优先级 | 预计讨论时间 |
|------|---------|--------|-------------|
| 1. 核心研究问题 | RQ1 vs RQ2 的选择 | P0 | 15 min |
| 2. 设备与场景 | AR 眼镜 vs MR 头显；场景选择 | P1 | 10 min |
| 3. UI 生成范式 | 抽象 Schema vs 平台特定 | P1 | 10 min |
| 4. 评估方法 | 自动化指标 vs 人工评估 | P2 | 10 min |
| 5. 数据策略 | 现有数据集 vs 自采集 | P2 | 10 min |

---

## 议题 1: 核心研究问题选择 [P0]

### 背景

**核心洞察**：Smart Glasses 与 WebUI 的本质区别
- **WebUI**: 用户找 UI（用户有明确需求，主动访问网页）
- **Smart Glasses**: UI 找用户（系统感知 context，主动推送信息）

Google 论文的 Generative UI 核心点：
> "Generative UI is a new modality where the AI model generates not only content, but **the entire user experience**."

Smart Glasses 的独特挑战：
1. 注意力不再全程集中在屏幕上
2. 有更丰富的来自真实生活的 context（场景、注视、活动）

### 两个候选研究问题

| 研究问题 | 核心挑战 | 技术方向 |
|---------|---------|---------|
| **RQ1: Context-to-UI Mapping** | 从多模态 context（场景、注视、活动）生成用户需要的 UI | 多模态理解、意图推断 |
| **RQ2: Attention-Aware UI Complexity** | 根据用户注意力状态动态调整 UI 复杂度 | 注意力检测、复杂度度量 |

**形式化表示**：
```
RQ1: f(视觉场景, 注视点, 活动识别, 用户历史) → UI 内容 + 布局
RQ2: 用户专注其他事 → 极简 HUD ｜ 用户看向眼镜 → 详细卡片
```

### 待讨论问题

**Q1**: 优先做 RQ1 还是 RQ2？还是两者结合？

**Q2**: 即使有用户的显式 query（如语音"帮我查一下这个产品"），是否仍需考虑生成"合适"的 UI？
- 如果是，那 RQ1 就不只是"无 query 时推断意图"，而是"如何生成与 context 匹配的 UI"
- 这会改变研究问题的定义

**Q3**: 研究贡献的定位？
- 仅"提出 GenUI 在 Smart Glasses 场景的应用"在学术上不够
- 需要有可验证的技术创新（新方法/新评估标准/新数据集）

### 相关材料
- brainstorm.md Section 10.3: 核心洞察与研究问题重构
- brainstorm.md Section 7.3: Smart Glasses 场景的本质差异

---

## 议题 2: 设备与场景选择 [P1]

### 背景

目前市场分为两类设备：

| 设备类型 | 代表产品 | UI 特点 | GenUI 需求 |
|---------|---------|---------|-----------|
| **轻量级 AR 眼镜** | Ray-Ban Meta, Vuzix Z100 | 单目 HUD、极简、瞥视 | ★★★★★ 高 |
| **沉浸式 MR 头显** | Vision Pro, Quest 3 | 多窗口、空间锚定、手势 | ★★☆☆☆ 低 |

**初步判断**：AR 眼镜场景更需要 Generative UI
- 沉浸式 MR 需要完整 UI 或在已有 UI 上操作，不太需要"生成式"
- AR 眼镜算力受限、信息密度极低，更需要智能生成

### 场景选择

| 场景          | Context 复杂度 | UI 复杂度 | 数据可获取性              | 研究价值 |
| ----------- | ----------- | ------ | ------------------- | ---- |
| 烹饪/操作指导     | 高（步骤+状态）    | 中      | ★★★★★ Epic-Kitchens | 高    |
| 专业工作（医疗/工业） | 高           | 高      | ★★★★☆ HoloAssist    | 高    |
| 导航/寻路       | 中           | 低      | ★★★☆☆               | 中    |
| 购物/比价       | 中           | 中      | ★★☆☆☆               | 中    |

### 待讨论问题

**Q4**: 设备选择确认——是否先聚焦轻量级 AR 眼镜？

**Q5**: 场景优先级——优先做哪个场景？
- 厨房场景有 Epic-Kitchens 数据支持
- 操作指导场景有 HoloAssist 数据支持
- 还是选择更通用的日常场景？

**Q6**: 是否限定单一场景，还是做跨场景的通用方法？

### 相关材料
- brainstorm.md Section 2 Q1: 目标设备形态
- brainstorm.md Section 12: 数据集调研

---

## 议题 3: UI 生成范式与输出格式 [P1]

### 背景：各家 SDK 不统一

| 平台 | SDK | UI 格式 | 状态 |
|------|-----|--------|------|
| Snap Spectacles | Lens Studio + SIK | 3D (FBX) + TypeScript | ✅ 公开 |
| Android XR | Jetpack Compose XR | Spatial Panels | ✅ 公开 |
| Meta Ray-Ban | **未公开 SDK** | 未知 | ❌ 封闭 |
| Magic Leap 2 | Unity/Unreal | 3D Spatial | ✅ 公开 |

**核心挑战**：
1. 各家 UI 架构没有统一
2. 各家眼镜的物理特性也不完全统一（FOV、分辨率等）
3. UI 显示风格不一致

### 三个可选方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 聚焦单一平台 | 如 Snap Spectacles | 可实机验证 | 泛化性受限 |
| B. 定义抽象 Schema | 平台无关的 UI 原语 | 通用性强 | 只能模拟器验证 |
| C. 多平台适配层 | 抽象 Schema → 平台代码 | 兼顾两者 | 工程量大 |

**可抽象的 UI 原语**（跨平台共性）：
```
├── Text Card（文本卡片）
├── Icon + Label（图标+标签）
├── Progress Indicator（进度指示）
├── Arrow / Direction（方向指引）
├── Notification Badge（通知徽章）
└── Confirmation Dialog（确认对话框）
```

### 待讨论问题

**Q7**: 在没有统一范式的情况下，应该输出什么样的 UI？
- A. 聚焦 Snap Spectacles（有 SDK 可验证）
- B. 定义抽象 UI Schema（JSON/DSL 格式）
- C. 其他方案

**Q8**: 是否需要把"UI 格式统一"作为贡献之一？

### 相关材料
- brainstorm.md Section 3.2: 各平台 UI 开发格式
- brainstorm.md Section 7.5: 各家厂商的"共性范式"

---

## 议题 4: 评估方法设计 [P2]

### 背景：现有评估的缺失

现有 AR/HUD 评估主要是 user study 形式，**缺失**的评估维度：

| 缺失维度 | 说明 |
|---------|------|
| **UI 生成质量的自动化评估** | 减少人工评估成本 |
| **Context-UI 匹配度的量化指标** | 生成的 UI 是否匹配当前场景 |
| **信息密度与注意力负荷的平衡度量** | 显示多少信息合适 |

### WebUI 评估 vs Smart Glasses 评估

| 评估维度 | WebUI 关注度 | Smart Glasses 应关注度 |
|---------|-------------|----------------------|
| 视觉还原 | ★★★★★ | ★☆☆☆☆ |
| Context 理解 | ★★☆☆☆ | ★★★★★ |
| 信息筛选 | ★☆☆☆☆ | ★★★★★ |
| **时机判断** | ☆☆☆☆☆ | ★★★★★ |
| **注意力负荷** | ★★☆☆☆ | ★★★★★ |
| **Glanceability** | ☆☆☆☆☆ | ★★★★★ |

### 三种评估形式

| 形式 | 描述 | 成本 | 可扩展性 |
|------|------|------|---------|
| A. 自动化指标 | 定义可计算的 metric | 低 | 高 |
| B. 人类偏好 | A/B 对比 + ELO 评分 | 高 | 低 |
| C. 模拟用户研究 | 在模拟器中测试 | 中 | 中 |

**补充问题**：PAGEN 论文中的 ELO 分数是什么？
- ELO 是来自国际象棋的排名系统，用于 pairwise comparison
- 多个模型/方法两两对比，根据胜负计算相对排名

### 待讨论问题

**Q9**: 评估应该测量什么能力？
- A. 视觉还原能力（生成的 UI 是否符合设计规范）
- B. Context-UI 匹配能力（生成的 UI 是否与场景相关）
- C. 时机判断能力（是否在正确的时候显示 UI）
- D. 以上全部

**Q10**: 评估形式选择？
- 考虑到没有 ground truth UI，是否只能做人工评估？
- 是否可以设计一些自动化的 proxy metric？

**Q11**: 评估数据来源？
- A. 从现有数据集选取场景 + 人工设计理想 UI
- B. 众包标注
- C. LLM 合成

### 相关材料
- brainstorm.md Section 3.4: 评估指标现状
- brainstorm.md Section 7.2: WebUI Benchmark 结构解析
- brainstorm.md Section 7.4: 应该评测的能力维度

---

## 议题 5: 数据策略 [P2]

### 背景：现有数据集调研结果

| 数据集 | 有场景理解 | 有用户意图 | 有注视数据 | GenUI 适用性 |
|--------|-----------|-----------|-----------|-------------|
| **Project Aria** | ✅ | ⚠️ | ✅ 60Hz | ★★★★★ 最高 |
| **Ego4D** | ✅ | ⚠️ | ✅ | ★★★★☆ 高 |
| **HoloAssist** | ✅ | ✅ 指导意图 | ❌ | ★★★★☆ 高 |
| **Epic-Kitchens** | ✅ 厨房 | ✅ 动作意图 | ❌ | ★★★☆☆ 中 |

**关键缺失**：所有数据集都没有 **"应该显示什么 UI"** 的标注（UI Ground Truth）

### 数据合作可能性

1. **泽宇学姐的眼动数据集**
   - 最近在采集
   - 具体内容和规模待确认

2. **现有数据集二次开发**
   - 推荐组合：Project Aria + HoloAssist
   - 需要人工二次标注"理想 UI"

### 二次开发工作量估算

```
选取 ~500 个典型场景片段
├── 人工设计"理想 UI"（使用抽象 Schema）
├── 标注"UI 出现时机"
└── 标注"信息优先级"

预计工作量：2-3 人月（取决于标注复杂度）
```

### 待讨论问题

**Q12**: 数据来源选择？
- A. 与泽宇学姐合作，使用她的眼动数据
- B. 基于 Project Aria / Ego4D 二次开发
- C. 自己采集新数据
- D. 以上组合

**Q13**: 如果二次开发，如何设计"理想 UI"的标注规范？
- 谁来标注？（研究者自己 / 众包 / 专业 UX 设计师）
- 标注格式？（自然语言描述 / 抽象 Schema / 具体 UI 代码）

**Q14**: 是否需要先做一个小规模 pilot study 验证标注流程？

### 相关材料
- brainstorm.md Section 12: Smart Glasses 数据集与 Benchmark 调研

---

## 讨论后行动项模板

| 决定项 | 结论 | 负责人 | 截止日期 |
|--------|------|--------|---------|
| 核心 RQ 选择 | | | |
| 设备/场景确定 | | | |
| UI 输出格式 | | | |
| 评估方法 | | | |
| 数据策略 | | | |

---

## 附录：关键问题速查

### 研究方向层面
- [ ] Q1: RQ1 vs RQ2 的选择
- [ ] Q2: 有 query 时是否仍需 context-aware UI
- [ ] Q3: 技术贡献的定位

### 方法层面
- [ ] Q4: 设备选择确认
- [ ] Q5: 场景优先级
- [ ] Q6: 单场景 vs 跨场景
- [ ] Q7: UI 输出格式选择
- [ ] Q8: "UI 格式统一"是否作为贡献

### 评估层面
- [ ] Q9: 评估什么能力
- [ ] Q10: 评估形式选择
- [ ] Q11: 评估数据来源

### 数据层面
- [ ] Q12: 数据来源选择
- [ ] Q13: 标注规范设计
- [ ] Q14: 是否先做 pilot study
