# Smart Glasses Generative UI 项目实现文档

## 1. 项目概述

### 1.1 核心目标

**采集 Smart Glasses 场景的 Generative UI 数据集**

不是做产品，而是为研究构建数据：
- 输入：第一人称视频 + 眼动 + 活动标签 + 用户接受的推荐
- 输出：结构化 UI JSON + Web 渲染结果

### 1.2 开发策略

```
Web 先行 → A2UI 声明式中间层 → 后续适配眼镜硬件
```

- **Phase 1**: Web 渲染验证 pipeline
- **Phase 2**: 批量生成数据集
- **Phase 3**: 迁移到 AR 眼镜 (Apple Vision Pro / AR Glasses)

---

## 2. 系统架构

### 2.1 Pipeline 总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ P1_YuePan    │    │   GPT-4o     │    │ Web Renderer │          │
│  │ Dataset      │───▶│ (组件选择)   │───▶│ (React)      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│        │                    │                   │                   │
│        ▼                    ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ • Ego Video  │    │ A2UI JSON    │    │ UI 截图/视频  │          │
│  │ • Eye Track  │    │ (结构化输出)  │    │ (数据集)      │          │
│  │ • Activity   │    │              │    │              │          │
│  │ • Accepted   │    │              │    │              │          │
│  │   Recommend  │    │              │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

| 模块               | 职责                   | 技术栈                      |
| ---------------- | -------------------- | ------------------------ |
| **Schema 定义**    | 组件 JSON Schema       | TypeScript / JSON Schema |
| **Generator**    | 调用 GPT-4o 生成 UI JSON | Python / TypeScript      |
| **Renderer**     | 将 JSON 渲染为可视化 UI     | React + shadcn/ui        |
| **Batch Runner** | 批量处理数据集              | Python Script            |

---

## 3. 技术决策与确认点

### 3.1 已确认的决策

| 决策点 | 选择 | 理由 | 状态 |
|--------|------|------|------|
| UI 生成方式 | **LLM + 组件库 (方案 C)** | 平衡灵活性与输出稳定性；LLM 从预定义组件中选择，而非自由生成 | ✅ 确认 |
| 协议格式 | **A2UI JSON** | Google 标准，便于后续迁移到眼镜 | ✅ 确认 |
| 组件粒度 | **极简 (8 个)** | 先跑通 demo，按需扩展 | ✅ 确认 |
| 渲染环境 | **Web 先行** | 快速迭代，后续适配 AR | ✅ 确认 |
| LLM 模型 | **GPT-4o** | 用户已有 API | ✅ 确认 |
| 代码仓库 | **独立于 Second Brain** | 避免上下文污染，Git 友好 | ✅ 确认 |

### 3.2 方案对比备忘

**为什么选方案 C (LLM + 组件库) 而非方案 B (LLM 自由生成)？**

```
方案 B 问题：
- LLM 可能发明 "orange_freshness_comparator" 这样的随机类型
- 每次生成的 JSON 结构可能不同
- 渲染器无法识别

方案 C 解决：
- 预定义 8 个组件类型
- LLM 只能从中选择
- 渲染器 100% 可处理
```

### 3.3 待确认/后续决策

| 待决策 | 选项 | 何时决定 |
|--------|------|----------|
| 目标硬件 | Apple Vision Pro / AR Glasses | Phase 3 开始前 |
| 人工评估方式 | 在线标注 / 离线回顾 | 数据生成后 |
| 是否需要更多组件 | 根据 demo 效果决定 | Phase 1 完成后 |

---

## 4. 组件库规范

### 4.1 组件总览 (8 个核心组件)

| 组件 ID | 覆盖的推荐类型 | 使用场景 |
|---------|---------------|----------|
| `ar.label` | object_identification, translation | 物体标注、翻译浮层 |
| `ar.infoCard` | computation, context_aware | 信息展示（营养、价格等） |
| `ar.comparisonCard` | decision_support | 对比决策（新鲜度、性价比） |
| `ar.stepCard` | procedural_guidance | 流程引导（加热步骤、维修步骤） |
| `ar.actionButton` | task_assistance | 快捷操作（一键拍照、加入清单） |
| `ar.suggestionCard` | context_aware, engaging | 情境推荐（找单车、拍照建议） |
| `ar.timer` | task_assistance | 计时器（番茄钟、微波炉倒计时） |
| `ar.memoryCard` | contextual_memory | 记忆回溯（刚才说的问题） |

### 4.2 JSON Schema 定义

#### 4.2.1 ar.label

```json
{
  "type": "ar.label",
  "properties": {
    "text": "string (required) - 标签文本",
    "anchor": {
      "type": "object | screen",
      "objectId": "string - 锚定的物体 ID (可选)",
      "position": { "x": "number", "y": "number" }
    },
    "style": {
      "variant": "default | highlight | warning",
      "size": "small | medium | large"
    }
  }
}
```

**示例**：
```json
{
  "type": "ar.label",
  "properties": {
    "text": "赣南脐橙 ¥12.9/盒",
    "anchor": { "type": "object", "objectId": "orange-box-1" },
    "style": { "variant": "default", "size": "medium" }
  }
}
```

#### 4.2.2 ar.infoCard

```json
{
  "type": "ar.infoCard",
  "properties": {
    "title": "string (required)",
    "value": "string | number (required)",
    "unit": "string (optional)",
    "icon": "string (optional) - emoji 或图标名",
    "description": "string (optional)"
  }
}
```

**示例**：
```json
{
  "type": "ar.infoCard",
  "properties": {
    "title": "热量",
    "value": 245,
    "unit": "kcal",
    "icon": "🔥",
    "description": "每 100g"
  }
}
```

#### 4.2.3 ar.comparisonCard

```json
{
  "type": "ar.comparisonCard",
  "properties": {
    "items": [
      {
        "label": "string (required)",
        "value": "number (required)",
        "highlight": "boolean (optional)"
      }
    ],
    "metric": "string (required) - 比较维度名称",
    "highlightBest": "boolean - 是否高亮最优项",
    "unit": "string (optional)"
  }
}
```

**示例**：
```json
{
  "type": "ar.comparisonCard",
  "properties": {
    "items": [
      { "label": "盒子 A", "value": 8.5 },
      { "label": "盒子 B", "value": 9.2, "highlight": true }
    ],
    "metric": "新鲜度",
    "highlightBest": true,
    "unit": "/10"
  }
}
```

#### 4.2.4 ar.stepCard

```json
{
  "type": "ar.stepCard",
  "properties": {
    "title": "string (optional)",
    "steps": [
      {
        "index": "number",
        "content": "string",
        "status": "pending | current | completed"
      }
    ],
    "currentStep": "number"
  }
}
```

**示例**：
```json
{
  "type": "ar.stepCard",
  "properties": {
    "title": "三明治加热步骤",
    "steps": [
      { "index": 1, "content": "撕开包装一角", "status": "completed" },
      { "index": 2, "content": "放入微波炉", "status": "current" },
      { "index": 3, "content": "中火加热 90 秒", "status": "pending" }
    ],
    "currentStep": 2
  }
}
```

#### 4.2.5 ar.actionButton

```json
{
  "type": "ar.actionButton",
  "properties": {
    "label": "string (required)",
    "action": "string (required) - 动作标识符",
    "icon": "string (optional)",
    "variant": "primary | secondary | ghost",
    "confirmRequired": "boolean - 是否需要二次确认"
  }
}
```

**示例**：
```json
{
  "type": "ar.actionButton",
  "properties": {
    "label": "拍照记录",
    "action": "capture_photo",
    "icon": "📷",
    "variant": "primary",
    "confirmRequired": false
  }
}
```

#### 4.2.6 ar.suggestionCard

```json
{
  "type": "ar.suggestionCard",
  "properties": {
    "title": "string (required)",
    "description": "string (required)",
    "icon": "string (optional)",
    "actions": [
      {
        "label": "string",
        "action": "string",
        "primary": "boolean"
      }
    ]
  }
}
```

**示例**：
```json
{
  "type": "ar.suggestionCard",
  "properties": {
    "title": "附近有共享单车",
    "description": "骑车可节省约 12 分钟",
    "icon": "🚲",
    "actions": [
      { "label": "查看位置", "action": "show_bike_location", "primary": true },
      { "label": "忽略", "action": "dismiss", "primary": false }
    ]
  }
}
```

#### 4.2.7 ar.timer

```json
{
  "type": "ar.timer",
  "properties": {
    "label": "string (optional)",
    "duration": "number (required) - 秒数",
    "autoStart": "boolean",
    "showControls": "boolean"
  }
}
```

**示例**：
```json
{
  "type": "ar.timer",
  "properties": {
    "label": "番茄钟",
    "duration": 1500,
    "autoStart": true,
    "showControls": true
  }
}
```

#### 4.2.8 ar.memoryCard

```json
{
  "type": "ar.memoryCard",
  "properties": {
    "content": "string (required)",
    "timestamp": "string (optional) - ISO 时间或相对时间",
    "source": "string (optional) - 来源描述",
    "jumpTo": "string (optional) - 跳转标识"
  }
}
```

**示例**：
```json
{
  "type": "ar.memoryCard",
  "properties": {
    "content": "你提到笔记本螺丝空转的问题",
    "timestamp": "10 分钟前",
    "source": "对话记录",
    "jumpTo": "conversation_12345"
  }
}
```

### 4.3 A2UI 完整消息格式

```json
{
  "type": "surfaceUpdate",
  "surfaceId": "smart-glass-overlay",
  "timestamp": "2026-01-26T10:30:00Z",
  "context": {
    "scene": "supermarket",
    "activity": "shopping",
    "gazeTarget": "orange-box-1"
  },
  "components": [
    {
      "type": "ar.comparisonCard",
      "properties": { ... }
    }
  ],
  "attention": {
    "trigger": "gaze",
    "priority": "medium",
    "autoHide": 5.0
  }
}
```

---

## 5. GPT-4o Prompt 设计

### 5.1 System Prompt

```
你是一个 Smart Glasses UI 生成器。

你的任务是：根据用户场景和推荐内容，选择合适的 UI 组件并生成 A2UI JSON。

## 可用组件

你只能从以下 8 个组件中选择：

1. ar.label - 简单标签，用于物体标注
2. ar.infoCard - 信息卡片，展示单一数据
3. ar.comparisonCard - 对比卡片，用于决策支持
4. ar.stepCard - 步骤卡片，用于流程引导
5. ar.actionButton - 操作按钮，用于快捷操作
6. ar.suggestionCard - 建议卡片，用于情境推荐
7. ar.timer - 计时器，用于倒计时任务
8. ar.memoryCard - 记忆卡片，用于回溯信息

## 输出格式

严格输出 JSON，不要包含任何解释：

{
  "type": "surfaceUpdate",
  "surfaceId": "smart-glass-overlay",
  "components": [
    {
      "type": "组件类型",
      "properties": { ... }
    }
  ]
}

## 选择原则

- 简单信息用 ar.label
- 需要比较时用 ar.comparisonCard
- 有明确步骤时用 ar.stepCard
- 需要用户操作时用 ar.actionButton 或 ar.suggestionCard
- 涉及时间时用 ar.timer
- 回忆相关用 ar.memoryCard
```

### 5.2 User Prompt 模板

```
## 场景信息
- 场景类型: {scene_type}
- 活动状态: {activity}
- 注视目标: {gaze_target}

## 用户接受的推荐
类型: {recommendation_type}
内容: {recommendation_content}

## 物体信息 (如有)
{object_list}

请生成对应的 UI JSON。
```

---

## 6. 实现计划

### Phase 1: MVP (2-3 周)

| 任务 | 输出 | 时间 |
|------|------|------|
| 1.1 Schema 定义 | `src/schema/*.json` | 2-3 天 |
| 1.2 GPT-4o 集成 | `src/generator/` | 3-5 天 |
| 1.3 Web 渲染器 | `src/renderer/` | 1 周 |
| 1.4 端到端测试 | 10 个样例验证 | 2-3 天 |

### Phase 2: 批量生成 (1-2 周)

| 任务 | 输出 | 时间 |
|------|------|------|
| 2.1 Batch Runner | `scripts/batch_generate.py` | 2-3 天 |
| 2.2 生成全量数据 | `output/*.json` + 渲染截图 | 3-5 天 |
| 2.3 质量检查 | 人工抽检 + 问题记录 | 2-3 天 |

### Phase 3: AR 适配 (待定)

- 选择目标硬件
- 开发眼镜端渲染器
- 实地验证

---

## 7. 目录结构

```
smart-glasses-genui/
├── CLAUDE.md                    # 代码开发上下文
├── README.md                    # 本文档
├── package.json                 # Node.js 依赖
├── requirements.txt             # Python 依赖 (如用 Python)
│
├── src/
│   ├── schema/                  # 组件 JSON Schema
│   │   ├── ar.label.json
│   │   ├── ar.infoCard.json
│   │   ├── ar.comparisonCard.json
│   │   └── ...
│   │
│   ├── generator/               # UI 生成逻辑
│   │   ├── prompt.ts            # System/User Prompt
│   │   ├── openai.ts            # GPT-4o API 调用
│   │   └── index.ts
│   │
│   └── renderer/                # Web 渲染器
│       ├── components/          # React 组件实现
│       │   ├── ARLabel.tsx
│       │   ├── ARInfoCard.tsx
│       │   └── ...
│       ├── A2UIRenderer.tsx     # JSON → 组件映射
│       └── App.tsx
│
├── scripts/
│   ├── batch_generate.py        # 批量生成脚本
│   └── validate_output.py       # 输出校验
│
├── data/                        # 数据目录
│   └── P1_YuePan -> (symlink)   # 链接到 Second Brain
│
├── output/                      # 生成结果
│   ├── json/                    # A2UI JSON 文件
│   └── screenshots/             # 渲染截图
│
└── tests/
    └── samples/                 # 测试样例
```

---

## 8. 数据集映射

### 8.1 P1_YuePan 数据集结构

```
数据来源: Second Brian/Research/generative-ui/dataset/
关键文件: P1_YuePan_task_scenarios_analysis.md

统计:
- 总时长: 3.17 小时
- 推荐总数: 330+ 条
- 用户接受: ~40 条 (12%)
- 场景类型: 6 类 (导航、购物、学习等)
```

### 8.2 输入输出对应

| 输入字段 | 用途 | 映射到 |
|----------|------|--------|
| `start_time`, `end_time` | 时间定位 | `context.timestamp` |
| `recommendation_type` | 组件选择依据 | 决定用哪个组件 |
| `original_content` | 推荐内容 | 组件的 text/title/content |
| `object_list` | 空间锚定 | `anchor.objectId` |
| `time_interval` | 触发时机 | `attention.trigger` |

### 8.3 推荐类型 → 组件映射

| 推荐类型 | 首选组件 | 备选组件 |
|----------|----------|----------|
| context_aware_recommendation | ar.suggestionCard | ar.actionButton |
| task_assistance | ar.actionButton | ar.stepCard |
| decision_support | ar.comparisonCard | ar.infoCard |
| object_identification | ar.label | ar.infoCard |
| procedural_guidance | ar.stepCard | - |
| computation_and_estimation | ar.infoCard | ar.comparisonCard |
| contextual_memory | ar.memoryCard | - |
| translation | ar.label | - |
| engaging_interaction | ar.suggestionCard | ar.infoCard |

---

## 9. 验收标准

### Phase 1 完成标准

- [ ] 8 个组件 Schema 定义完成
- [ ] GPT-4o 能稳定输出符合 Schema 的 JSON
- [ ] Web 渲染器能正确显示所有 8 种组件
- [ ] 10 个样例端到端跑通

### Phase 2 完成标准

- [ ] P1_YuePan 全部 40+ 条接受推荐生成对应 UI
- [ ] 输出 JSON 100% 符合 Schema
- [ ] 渲染截图保存完整
- [ ] 人工抽检通过率 > 90%

---

## 10. 参考资源

### 规范与协议

- [A2UI 规范](https://github.com/google/A2UI)
- [A2UI Specification v0.8](https://a2ui.org/specification/v0.8-a2ui/)
- [AG-UI Protocol](https://github.com/ag-ui-protocol/ag-ui)

### 相关工具

- [CopilotKit](https://docs.copilotkit.ai/) - React GenUI 框架
- [shadcn/ui](https://ui.shadcn.com/) - React 组件库
- [React XR](https://github.com/pmndrs/react-xr) - WebXR 框架

### 调研文档

- `survey/2026-01-26 Survey.md` - GenUI 框架调研
- `survey/2026-01-21 discuss.md` - 项目实施文档
- `dataset/P1_YuePan_task_scenarios_analysis.md` - 数据集分析

---

*文档版本: v1.0*
*创建日期: 2026-01-26*
*基于 Interview Mode 讨论整理*
