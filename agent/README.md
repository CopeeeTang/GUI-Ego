# Smart Glasses Generative UI Pipeline

将智能眼镜多模态数据中的 AI 推荐转换为 A2UI 格式 JSON，用于 Generative UI 研究。

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input                                    │
│  annotation_2.2_*.json  →  推荐内容 + 类型 + 时间范围             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Scene Classification                          │
│  根据时间范围判断场景: navigation (726-1533s) / shopping (3307-4703s) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Step 1: Component Selection                      │
│  LLM 从场景组件库中选择最合适的 UI 组件                            │
│  Prompt: COMPONENT_SELECTION_PROMPT                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Step 2: Props Filling                          │
│  LLM 根据组件 Schema 填充 props                                   │
│  Prompt: PROPS_FILLING_PROMPT                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Step 3: Validation                             │
│  校验生成的 JSON 符合 A2UI Schema                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Output                                   │
│  A2UI JSON  →  /home/v-tangxin/GUI/agent/output/                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Input

### 数据源

```
/home/v-tangxin/GUI/data/ego-dataset/data/P1_YuePan/annotation/annotation_2.2_P1_YuePan.json
```

### 数据结构

```json
{
  "start_time": 1024.9,
  "end_time": 1311.6,
  "recommendation_list": [
    {
      "type": "object_identification_and_recognition",
      "content": "这座体育馆是谁设计的？"
    }
  ],
  "accepted_recommendation_list": [...]
}
```

### 推荐类型 (recommendation.type)

| 类型 | 说明 | 示例 |
|------|------|------|
| `object_identification_and_recognition` | 物体识别 | "这座体育馆是谁设计的？" |
| `context_aware_recommendation` | 情境推荐 | "你已经到达档案馆了，需要调出调档流程吗？" |
| `engaging_interaction` | 互动交流 | "清华校园是在清朝前皇家园林的旧址上建立的" |
| `task_assistance` | 任务辅助 | "设置30分钟后午饭提醒" |
| `decision_support` | 决策支持 | "这些橙子中哪一盒更新鲜？" |
| `computation_and_estimation` | 计算估算 | "估计骑滑板车行驶的距离" |

---

## Output

### 文件路径

```
/home/v-tangxin/GUI/agent/output/
├── a2ui_components_YYYYMMDD_HHMMSS.json      # 汇总文件
├── a2ui_navigation_YYYYMMDD_HHMMSS.json      # 导航场景
└── a2ui_shopping_YYYYMMDD_HHMMSS.json        # 购物场景
```

### A2UI 组件结构

```json
{
  "type": "ar_label",
  "id": "ar_label_01cd8199",
  "props": {
    "text": "设计者",
    "subtext": "点击查看详情",
    "icon": "info",
    "anchor": {"type": "object", "target": "stadium"}
  },
  "metadata": {
    "source_recommendation": {
      "type": "object_identification_and_recognition",
      "content": "这座体育馆是谁设计的？",
      "time_range": [1024.9, 1311.6]
    },
    "generated": true,
    "selection": {
      "reasoning": "AR 标签适合在物理对象上悬浮显示相关信息",
      "confidence": 0.95
    }
  }
}
```

---

## Component Library

### 导航场景 (Navigation)

| 组件 | 用途 |
|------|------|
| `map_card` | 地图卡片，显示位置、导航路线、附近兴趣点 |
| `ar_label` | AR 标签，悬浮在物理对象上方显示信息 |
| `direction_arrow` | 方向指引箭头，显示前进方向和距离 |
| `comparison_card` | 对比卡片，比较步行vs骑车等选项 |

### 购物场景 (Shopping)

| 组件 | 用途 |
|------|------|
| `comparison_card` | 商品对比卡片 |
| `nutrition_card` | 营养信息卡片 |
| `price_calculator` | 价格计算器 |
| `ar_label` | 商品 AR 标签 |

---

## Prompts

### Step 1: Component Selection Prompt

**文件**: `src/component_selector.py`

**System Prompt**:
```
你是一个 UI 组件选择专家，只返回 JSON 格式的结果。
```

**User Prompt** (`COMPONENT_SELECTION_PROMPT`):
```
你是一个智能眼镜 UI 生成专家。根据用户场景和 AI 推荐内容，选择最合适的 UI 组件。

## 当前场景: {scene_name}

## 可用组件:
{component_list}

## 任务
分析以下 AI 推荐内容，选择最合适的 UI 组件类型。

## AI 推荐
- 类型: {recommendation_type}
- 内容: {recommendation_content}

## 输出要求
返回 JSON 格式:
{
    "selected_component": "组件类型名称",
    "reasoning": "选择理由（简短）",
    "confidence": 0.0-1.0 的置信度
}
```

**变量说明**:
- `{scene_name}`: 场景名称 (navigation / shopping)
- `{component_list}`: 场景可用组件列表及其描述
- `{recommendation_type}`: 推荐类型
- `{recommendation_content}`: 推荐内容

**输出示例**:
```json
{
  "selected_component": "ar_label",
  "reasoning": "AR 标签适合在物理对象上悬浮显示相关信息",
  "confidence": 0.95
}
```

---

### Step 2: Props Filling Prompt

**文件**: `src/props_filler.py`

**System Prompt**:
```
你是一个 UI 内容生成专家，只返回 JSON 格式的 props 对象。
```

**User Prompt** (`PROPS_FILLING_PROMPT`):
```
你是一个智能眼镜 UI 内容生成专家。根据 AI 推荐内容，填充 UI 组件的 props。

## 组件类型: {component_type}

## 组件 Schema:
必需字段: {required_fields}
可用属性:
{properties}

## 示例:
```json
{example}
```

## AI 推荐
- 类型: {recommendation_type}
- 内容: {recommendation_content}

## 任务
根据推荐内容，生成该组件的 props。确保：
1. 包含所有必需字段
2. 内容与推荐语义一致
3. 适合智能眼镜的简洁显示

## 输出
返回 JSON 格式的 props 对象（不需要 type 和 id，只需要 props 内容）:
```

**变量说明**:
- `{component_type}`: 组件类型 (ar_label, map_card, etc.)
- `{required_fields}`: 必需字段列表
- `{properties}`: 属性描述
- `{example}`: 示例 JSON
- `{recommendation_type}`: 推荐类型
- `{recommendation_content}`: 推荐内容

**输出示例** (ar_label):
```json
{
  "text": "设计者",
  "subtext": "点击查看详情",
  "icon": "info",
  "anchor": {"type": "object", "target": "stadium"}
}
```

---

## LLM Configuration

| 参数 | Step 1 (Selection) | Step 2 (Props) |
|------|---------------------|----------------|
| Model | GPT-4o | GPT-4o |
| Temperature | 0.3 | 0.5 |
| Response Format | JSON | JSON |
| Max Tokens | 2000 | 2000 |

---

## Usage

### 安装依赖

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate
pip install -r agent/requirements.txt
```

### 运行 Pipeline

```bash
cd agent

# MVP 测试 (10条)
python -m src.pipeline --limit 10 --scenes navigation shopping

# 完整运行 (50条)
python -m src.pipeline --limit 50 --scenes navigation shopping

# 仅导航场景
python -m src.pipeline --limit 20 --scenes navigation
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-path` | `/home/v-tangxin/GUI/data/ego-dataset` | 数据集路径 |
| `--output-path` | `/home/v-tangxin/GUI/agent/output` | 输出路径 |
| `--participant` | `P1_YuePan` | 被试 ID |
| `--scenes` | `navigation shopping` | 处理场景 |
| `--limit` | `50` | 最大处理数量 |

---

## Experiment Variations

### 可调整的 Prompt 变量

| 位置 | 变量 | 实验方向 |
|------|------|----------|
| `component_selector.py` | `COMPONENT_DEFINITIONS` | 增减组件库、修改组件描述 |
| `component_selector.py` | `COMPONENT_SELECTION_PROMPT` | 修改选择策略、增加约束 |
| `props_filler.py` | `COMPONENT_SCHEMAS` | 修改 props 结构、增加字段 |
| `props_filler.py` | `PROPS_FILLING_PROMPT` | 修改填充策略、增加上下文 |

### 可添加的 Context

1. **视觉上下文**: 添加视频帧描述 (VLM 生成)
2. **时序上下文**: 添加前后推荐的关联
3. **用户接受/拒绝标签**: 使用 `accepted_recommendation_list` 作为反馈
4. **眼动数据**: 添加 gaze 注视点信息

### Prompt 对比实验示例

```python
# Variant A: 基础 prompt (当前)
PROMPT_A = "根据推荐内容，选择最合适的 UI 组件..."

# Variant B: 增加视觉上下文
PROMPT_B = """
根据推荐内容和当前视觉场景，选择最合适的 UI 组件。

## 视觉上下文
{visual_description}

## AI 推荐
...
"""

# Variant C: 增加用户偏好
PROMPT_C = """
根据推荐内容和用户历史偏好，选择最合适的 UI 组件。

## 用户接受率
- ar_label: 80%
- map_card: 60%
...
"""
```

---

## File Structure

```
/home/v-tangxin/GUI/agent/
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # 主 Pipeline
│   ├── data_loader.py        # 数据加载 + 场景分类
│   ├── component_selector.py # Step 1: LLM 组件选择
│   ├── props_filler.py       # Step 2: LLM Props 填充
│   ├── schema_validator.py   # Step 3: Schema 校验
│   └── llm_client.py         # Azure GPT-4o 客户端
├── schemas/
│   └── a2ui_components.json  # A2UI Schema 定义
├── output/                   # 生成的 A2UI JSON
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖
└── README.md                 # 本文档
```

---

## Metrics

### 当前 MVP 结果

| 指标 | 数值 |
|------|------|
| 处理速度 | ~2.3s/条 |
| 成功率 | 100% (5/5) |
| 组件选择 confidence | 0.9-0.95 |

### 可收集的评估指标

1. **组件选择准确率**: 人工标注 ground truth 对比
2. **Props 填充质量**: 语义一致性评分
3. **用户接受率预测**: 对比 accepted_recommendation_list
4. **多样性**: 组件类型分布

---

*Created: 2026-01-28*
