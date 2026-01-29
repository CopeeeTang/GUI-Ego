# Smart Glasses Generative UI Pipeline

将智能眼镜多模态数据中的 AI 推荐转换为 A2UI 格式 JSON，用于 Generative UI 研究。

## 🚀 Quick Start

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate

# 安装依赖
pip install -r agent/requirements.txt

# 运行 Pipeline (默认 v1_baseline)
python -m agent.src.pipeline --limit 10

# 使用视觉上下文增强
python -m agent.src.pipeline --strategy v3_with_visual --enable-visual --limit 5

# 启动预览服务器
python -m agent.preview.server --port 8080
```

---

## 📊 Pipeline 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input                                    │
│  annotation_2.2_*.json + 视频文件 (可选)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  --enable-visual  │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ v1_baseline   │    │ v2_google_gui │    │ v3_with_visual│
│ 两步法        │    │ 端到端(预留)   │    │ +视觉上下文   │
│               │    │               │    │               │
│ component_    │    │ 用户提供      │    │ 视频帧提取    │
│ selector      │    │ Prompt模板    │    │ VLM描述生成   │
│ + props_filler│    │               │    │ 端到端生成    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ A2UI Converter  │ (可选: --output-format a2ui_standard)
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Output      │
                    │  JSON + Preview │
                    └─────────────────┘
```

---

## 📁 项目结构

```
/home/v-tangxin/GUI/agent/
├── config.yaml                    # 配置文件
├── requirements.txt               # 依赖列表
│
├── src/
│   ├── schema.py                  # ⭐ 统一数据模型 (单一数据源)
│   ├── data_loader.py             # 数据加载器
│   ├── llm_client.py              # LLM/VLM 客户端
│   ├── pipeline.py                # ⭐ 主入口
│   ├── schema_validator.py        # JSON 校验器
│   │
│   ├── component_selector.py      # V1: Step 1 组件选择
│   ├── props_filler.py            # V1: Step 2 Props 填充
│   │
│   ├── prompts/                   # ⭐ Prompt 策略模式
│   │   ├── base.py                # 策略基类
│   │   ├── v1_baseline.py         # 两步法封装
│   │   ├── v2_google_gui.py       # Google GUI (预留)
│   │   └── v3_with_visual.py      # 视觉增强版
│   │
│   ├── video/                     # ⭐ 视频处理
│   │   ├── extractor.py           # 帧提取器
│   │   └── visual_context.py      # VLM 上下文生成
│   │
│   └── a2ui/                      # ⭐ A2UI 格式转换
│       ├── converter.py           # 组件转换器
│       └── message_builder.py     # 消息构建器
│
├── schemas/
│   └── a2ui_components.json       # 组件 Schema
│
├── preview/
│   ├── server.py                  # HTTP 预览服务器
│   └── static/                    # (待填充) Lit 渲染器
│
└── output/                        # 生成的 JSON
```

---

## 🎯 三种 Prompt 策略

| 策略 | 说明 | 视觉支持 | 使用场景 |
|------|------|----------|----------|
| `v1_baseline` | 两步法 (选组件→填属性) | ❌ | 基准对比 |
| `v2_google_gui` | 端到端生成 | ✅ | 需用户提供 Prompt |
| `v3_with_visual` | 带视觉上下文的端到端 | ✅ | 视频+推荐联合生成 |

### 使用示例

```bash
# 策略 1: 基准两步法
python -m agent.src.pipeline --strategy v1_baseline --limit 10

# 策略 3: 视觉增强 (description 模式)
python -m agent.src.pipeline --strategy v3_with_visual --enable-visual --visual-mode description --limit 5

# 策略 3: 视觉增强 (direct 模式 - 直接传图)
python -m agent.src.pipeline --strategy v3_with_visual --enable-visual --visual-mode direct --limit 5

# 策略对比实验
python -m agent.src.pipeline --compare v1_baseline v3_with_visual --limit 5
```

---

## 🎬 视觉上下文

### 工作流程

```
视频文件 → 提取 3 帧 (等间隔) → VLM 生成描述 → 注入 Prompt
```

### 两种模式

| 模式 | 说明 | Token 消耗 |
|------|------|-----------|
| `description` | 先用 VLM 生成文本描述 | ~500 tokens |
| `direct` | 直接传图给多模态 LLM | ~2300 tokens (3帧) |

---

## 🖥️ 预览服务器

```bash
# 启动服务器
python -m agent.preview.server --port 8080

# 浏览器访问
http://localhost:8080
```

**当前状态**: 使用内联 CSS 渲染，A2UI Lit 渲染器待集成。

---

## 📊 CLI 参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strategy` | `v1_baseline` | Prompt 策略 |
| `--enable-visual` | `false` | 启用视频帧提取 |
| `--visual-mode` | `description` | 视觉模式 (direct/description) |
| `--output-format` | `legacy` | 输出格式 (legacy/a2ui_standard) |
| `--compare` | - | 对比多个策略 |
| `--limit` | `50` | 处理数量 |
| `--scenes` | `navigation shopping` | 场景 |
| `--data-path` | `/home/v-tangxin/GUI/data/ego-dataset` | 数据集路径 |
| `--output-path` | `/home/v-tangxin/GUI/agent/output` | 输出路径 |
| `--participant` | `P1_YuePan` | 被试 ID |

---

## 📥 Input

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

## 📤 Output

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

## 🧩 Component Library

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

## ⚙️ LLM Configuration

| 参数 | Step 1 (Selection) | Step 2 (Props) |
|------|---------------------|----------------|
| Model | GPT-4o | GPT-4o |
| Temperature | 0.3 | 0.5 |
| Response Format | JSON | JSON |
| Max Tokens | 2000 | 2000 |

---

## ✅ 已完成事项

- [x] 复制 A2UI Lit 渲染器到 `preview/static/`
- [x] 添加 A2UI v0.9 Schema 文件到 `schemas/`
- [x] 集成 Google GUI Prompt 模板到 `v2_google_gui.py`

---

*Updated: 2025-01-29*
