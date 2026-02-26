# Generative UI Pipeline 生产流程文档

## 概述

本文档详细描述了 Smart Glasses Generative UI Pipeline 的完整生产流程，包括数据来源、场景分类、参数说明以及各版本策略（v1/v2/v3）的差异。

---

## 1. 数据来源

### 1.1 标注数据

- **路径**: `data/ego-dataset/annotations_2.2_{participant}.json`
- **格式**: JSON 格式的时序标注数据
- **内容**: 包含 AI 推荐（recommendations）及其时间戳、类型、内容等

```json
{
  "id": "rec_001",
  "type": "navigation",
  "text": "前方50米左转",
  "time_interval": {
    "start": 120.5,
    "end": 125.0
  },
  "is_accepted": true,
  "object_list": ["building", "road_sign"]
}
```

### 1.2 视频数据

- **合并视频**: `{participant}/Timeseries Data + Scene Video/{session_id}/*.mp4`
- **分段视频**: `{participant}/raw_data/{session_id}/Neon Scene Camera v1 ps*.mp4`
- **Gaze Overlay 视频**: `{participant}/raw_data/{session_id}/gaze-overlay-output-video-compressed.mp4`

### 1.3 数据加载流程

```
DataLoader → 读取 annotations_2.2_*.json
           → 按场景分类（navigation/shopping）
           → 过滤 is_accepted=true 的推荐
           → 生成 (Recommendation, SceneConfig) 对
```

---

## 2. 场景分类 (Scene Classifications)

### 2.1 Navigation（导航场景）

| 组件类型 | 描述 | 使用场景 | 锚定策略 |
|---------|------|---------|---------|
| `map_card` | 地图卡片，显示位置、路线、兴趣点 | 找共享单车、导航到目的地、显示附近设施 | 视野下方或角落 |
| `ar_label` | AR标签，悬浮在物理对象上方 | 识别建筑名称、标注地标、方向指示 | 锚定到识别出的物体上方 |
| `direction_arrow` | 方向指引箭头 | 步行导航、指示转弯方向、显示距离 | 视野中央或前进方向 |
| `comparison_card` | 对比卡片 | 比较步行vs骑车时间、比较路线选项 | 视野中央 |

### 2.2 Shopping（购物场景）

| 组件类型 | 描述 | 使用场景 | 锚定策略 |
|---------|------|---------|---------|
| `comparison_card` | 商品对比卡片 | 比较商品新鲜度、价格、营养成分 | 商品上方或视野中央 |
| `nutrition_card` | 营养信息卡片 | 显示热量、蛋白质、健康评级 | 锚定到产品包装附近 |
| `price_calculator` | 价格计算器 | 计算单价、比较性价比、估算总价 | 显示在价格标签附近 |
| `ar_label` | AR商品标签 | 显示商品名称、价格、标注推荐商品 | 锚定到商品位置 |

---

## 3. Prompt 策略版本说明

### 3.1 v1_baseline（基线策略）

**文件**: `agent/src/prompts/v1_baseline.py`

**架构**: 两步式生成
```
Step 1: ComponentSelector → 选择最合适的组件类型
Step 2: PropsFiller → 填充组件属性
```

**特点**:
- 不支持视觉上下文
- 使用独立的 LLM 调用进行组件选择和属性填充
- 兼容旧版 pipeline

**输出示例**:
```json
{
  "type": "ar_label",
  "id": "ar_label_001",
  "props": { "text": "图书馆入口", "subtext": "50米" },
  "metadata": {
    "selection": {
      "strategy": "v1_baseline",
      "selected_component": "ar_label",
      "confidence": 0.85
    }
  }
}
```

---

### 3.2 v2_google_gui（端到端策略）

**文件**: `agent/src/prompts/v2_google_gui.py`

**架构**: 单次端到端生成

**核心理念**:
- 构建交互式 UI，而非静态文本
- 最小化且清晰的设计
- 无占位符，始终包含真实内容
- AR 优化设计

**可用的原子组件**:
- `Card`: 容器组件（floating/elevated/outlined）
- `Row`/`Column`: 布局组件
- `Text`: 文本组件（h1/h2/h3/body1/body2/caption）
- `Icon`: Material 图标
- `Image`: 图片组件
- `Button`: 交互按钮
- `Divider`, `List`, `Badge`, `ProgressBar`

**可选支持视觉上下文**（取决于模板配置）

---

### 3.3 v2_smart_glasses（智能眼镜优化策略）

**文件**: `agent/src/prompts/v2_smart_glasses.py`

**架构**: 专为 HUD 优化的端到端生成

**设计约束**:
- **不遮挡视野**: 除非用户正在聚焦，否则保持中央清晰
- **高对比度**: 动态视频背景，使用深色半透明背景+亮色文字（Glassmorphism）
- **最小化文本**: 用户无法在行走时阅读长段落
- **大触控目标**: 按钮目标必须足够大

**布局策略**:
- **Peripheral（外围）**: 内容在边缘（行走时）
- **Immersive（沉浸式）**: 内容在中央（坐着/停止时）

**输出格式**: A2UI 消息流
```json
[
  { "type": "createSurface", "surfaceId": "hud" },
  { "type": "updateComponents", "surfaceId": "hud", "components": [...] }
]
```

---

### 3.4 v3_with_visual（视觉增强策略）

**文件**: `agent/src/prompts/v3_with_visual.py`

**架构**: 视觉上下文增强生成

**两种视觉模式**:

| 模式 | 描述 | 使用场景 |
|-----|------|---------|
| `DIRECT` | 直接传递 base64 图像给多模态 LLM | 需要精确视觉分析 |
| `DESCRIPTION` | 先生成文本描述，再用于生成 | 降低成本，兼容纯文本 LLM |

**视觉上下文考量因素**:
1. **物体定位**: 根据识别的物体位置确定 AR 元素锚点
2. **环境适配**: 根据光线、拥挤程度调整 UI 策略
3. **用户活动**: 根据当前活动状态选择交互方式
4. **视觉一致性**: 确保 UI 与物理场景自然融合

**输出包含 visual_anchor**:
```json
{
  "type": "nutrition_card",
  "id": "nutrition_001",
  "props": { "product_name": "有机牛奶", "calories": 120 },
  "visual_anchor": {
    "type": "object",
    "target": "牛奶包装盒",
    "position": "right",
    "reasoning": "锚定到用户正在查看的产品包装上"
  }
}
```

---

## 4. Pipeline 参数配置

### 4.1 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--data-path` | str | `/home/v-tangxin/GUI/data/ego-dataset` | 数据集路径 |
| `--output-path` | str | `/home/v-tangxin/GUI/agent/output` | 输出路径 |
| `--participant` | str | `P1_YuePan` | 被试 ID |
| `--scenes` | list | `[navigation, shopping]` | 要处理的场景 |
| `--limit` | int | `50` | 最大处理数量 |
| `--strategy` | str | `v1_baseline` | Prompt 策略 |
| `--enable-visual` | flag | `False` | 启用视频帧提取 |
| `--visual-mode` | str | `description` | 视觉上下文模式 |
| `--output-format` | str | `legacy` | 输出格式 |
| `--num-frames` | int | `3` | 每个推荐提取的帧数 |
| `--compare` | list | - | 比较多个策略 |

### 4.2 使用示例

```bash
# 基础运行（v1_baseline）
python -m agent.src.pipeline --limit 10

# 使用 v3 视觉策略
python -m agent.src.pipeline --strategy v3_with_visual --enable-visual --limit 5

# 策略比较
python -m agent.src.pipeline --compare v1_baseline v3_with_visual --limit 5

# A2UI 标准格式输出
python -m agent.src.pipeline --strategy v2_google_gui --output-format a2ui_standard
```

---

## 5. 各模式可修改部分

### 5.1 全局可修改

| 模块 | 文件 | 可修改内容 |
|-----|------|----------|
| 组件定义 | `schema.py` | `COMPONENT_REGISTRY` - 添加/修改组件类型 |
| 场景配置 | `data_loader.py` | 场景分类规则、时间戳映射 |
| 验证规则 | `schema_validator.py` | JSON Schema 验证规则 |

### 5.2 v1_baseline 可修改

| 模块 | 可修改内容 |
|-----|----------|
| `component_selector.py` | 组件选择逻辑、选择 prompt |
| `props_filler.py` | 属性填充逻辑、属性 schema |

### 5.3 v2_google_gui 可修改

| 部分 | 可修改内容 |
|-----|----------|
| `GOOGLE_A2UI_SYSTEM_PROMPT` | 系统 prompt（设计理念、组件列表、输出格式） |
| `DEFAULT_GOOGLE_GUI_TEMPLATE` | 用户 prompt 模板 |
| 外部模板文件 | 通过 `--google-gui-template` 指定自定义模板 |

### 5.4 v2_smart_glasses 可修改

| 部分 | 可修改内容 |
|-----|----------|
| `V2_SMART_GLASSES_PROMPT` | HUD 设计约束、布局策略、示例 |
| 输出解析逻辑 | `---a2ui_JSON---` 分隔符处理 |

### 5.5 v3_with_visual 可修改

| 部分 | 可修改内容 |
|-----|----------|
| `VISUAL_COMPONENT_DEFINITIONS` | 组件的视觉提示和锚定策略 |
| `COMPONENT_SCHEMAS` | 组件属性 schema |
| `VISUAL_PROMPT_TEMPLATE` | 视觉增强 prompt 模板 |
| `_infer_component_type()` | 组件类型推断逻辑 |

---

## 6. 输出文件

Pipeline 运行后生成以下文件：

```
agent/output/
├── a2ui_components_{strategy}_{timestamp}.json    # 所有组件汇总
├── a2ui_navigation_{strategy}_{timestamp}.json    # 导航场景组件
├── a2ui_shopping_{strategy}_{timestamp}.json      # 购物场景组件
├── a2ui_messages_{strategy}_{timestamp}.json      # A2UI 消息序列（可选）
└── strategy_comparison_{timestamp}.json           # 策略比较结果（可选）
```

---

## 7. 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    GenerativeUIPipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  DataLoader  │───▶│ Recommendation│───▶│  PromptStrategy  │   │
│  │              │    │ + SceneConfig │    │  (v1/v2/v3)      │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│  ┌──────────────┐    ┌──────────────┐             │             │
│  │FrameExtractor│───▶│VisualContext │─────────────┘             │
│  │  (optional)  │    │  Generator   │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                    │             │
│                      ┌──────────────┐    ┌────────▼─────────┐   │
│                      │ LLM Client   │◀───│  Generated JSON  │   │
│                      │ (Azure GPT)  │    │    Component     │   │
│                      └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                      ┌──────────────┐    ┌────────▼─────────┐   │
│                      │   Validator  │◀───│  A2UI Converter  │   │
│                      └──────────────┘    │    (optional)    │   │
│                                          └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
