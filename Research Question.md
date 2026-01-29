# Smart Glasses GenUI Survey

## Pipeline 架构

```
原始数据 → VLM 理解 → GUI 生成 → Web 渲染 (→ 未来眼镜部署)
```

## 技术决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| UI 生成方式 | LLM + 组件库 (方案 C) | 平衡灵活性与输出稳定性 |
| 协议格式 | A2UI JSON | 标准化，便于后续迁移 |
| 组件粒度 | 极简 (5-8 个) | 先跑通 demo，按需扩展 |
| 渲染环境 | Web 先行 | 快速迭代，后续适配 AR |

## 组件库 Schema (8 个核心组件)

| 组件 | 用途 | 关键参数 |
|------|------|----------|
| `ar.label` | 物体标注、翻译 | text, anchor, style |
| `ar.infoCard` | 信息展示 | title, value, unit |
| `ar.comparisonCard` | 对比决策 | items[], metric, highlightBest |
| `ar.stepCard` | 流程引导 | steps[], currentStep |
| `ar.actionButton` | 快捷操作 | label, action, confirm |
| `ar.suggestionCard` | 情境推荐 | title, description, actions[] |
| `ar.timer` | 计时器 | duration, label |
| `ar.memoryCard` | 记忆回溯 | content, timestamp |

## 数据源

- **输入**: P1_YuePan 数据集 (Ego 视频 + 眼动 + 活动标签)
- **输出**: A2UI JSON + Web 渲染结果
- **参考**: `../dataset/P1_YuePan_task_scenarios_analysis.md`

## 下一步

1. 定义 8 个组件的完整 JSON Schema
2. 编写 GPT-4o system prompt (组件注册)
3. 实现 Web 渲染器 (React + shadcn/ui)
4. 批量生成 P1_YuePan 的 UI 数据

## 相关资源

- A2UI 规范: https://github.com/google/A2UI
- CopilotKit: https://docs.copilotkit.ai/
- React XR (AR): https://github.com/pmndrs/react-xr
