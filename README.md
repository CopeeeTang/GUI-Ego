# Smart Glasses GenUI 数据加工与生成系统

**Last Updated:** 2026-02-02

基于 Ego-centric 视频数据集，利用多模态 LLM 生成符合 A2UI 协议的智能眼镜 GUI 数据。

---

## 🚀 Quick Start

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate

# 运行 Pipeline (默认使用 Azure GPT-4o)
python3 -m agent.src.pipeline --limit 10

# 使用 Gemini
python3 -m agent.src.pipeline --model gemini:gemini-2.5-pro --limit 5

# 使用 Claude
python3 -m agent.src.pipeline --model claude:claude-sonnet-4-5 --limit 5

# 启用视觉上下文
python3 -m agent.src.pipeline --strategy v3_with_visual --enable-visual --limit 5

# 启动预览服务器
python3 -m agent.preview.server --port 8000
```

---

## 📚 研究背景

现有的 Generative UI 主要针对 Web 界面，直接应用到 Smart Glasses 场景存在显著 Gap：

- **核心 Gap**: WebUI (鼠标/触控/大屏) vs Smart Glasses (注视/语音/透视/小屏)
- **研究方向**: 探索适合智能眼镜的 "Minimal", "Anchored", "Context-aware" 的 UI 生成范式

### 关键文档
- [Gap 分析与场景定义](survey/2026-01-18%20example.md)
- [A2UI 扩展协议](A2UI-SmartGlasses-Extension.md)
- [数据集完整文档](docs/Ego-Dataset_Complete_Documentation.md)

---

## 📂 项目结构

```
/home/v-tangxin/GUI/
├── agent/                      # 核心 Pipeline
│   ├── src/
│   │   ├── pipeline.py         # 主入口
│   │   ├── example_loader.py   # 数据加载器 (100 examples)
│   │   ├── llm/                # 多 LLM 提供商支持
│   │   │   ├── azure_openai.py # Azure GPT-4o
│   │   │   ├── gemini.py       # Google Gemini
│   │   │   └── claude.py       # Anthropic Claude
│   │   ├── prompts/            # Prompt 策略
│   │   │   ├── v1_baseline.py  # 两步法基准
│   │   │   ├── v2_google_gui.py# Google GUI 端到端
│   │   │   ├── v2_smart_glasses.py # Smart Glasses 优化
│   │   │   └── v3_with_visual.py   # 视觉上下文增强
│   │   ├── video/
│   │   │   ├── extractor.py    # 视频帧提取
│   │   │   └── overlay/        # UI 叠加到视频
│   │   └── sampling/           # 数据采样工具
│   │
│   ├── preview/                # A2UI 预览渲染器
│   │   ├── server.py           # HTTP 服务器
│   │   ├── static/src/0.8/ui/  # AR Glasses UI 组件
│   │   └── web_core/           # A2UI 核心样式
│   │
│   ├── example/                # 100 个标注样本 (raw_data)
│   └── output/                 # 生成结果
│
├── data/                       # 原始数据集 (Symlink)
├── docs/                       # 技术文档
├── survey/                     # 调研笔记
└── smartfocus-glasses-ui/      # React 原型 UI
```

---

## 🎯 核心功能

### 1. 多 LLM 提供商支持

| Provider | Model Examples | 特点 |
|----------|---------------|------|
| Azure OpenAI | `azure:gpt-4o` | 企业级稳定性 |
| Google Gemini | `gemini:gemini-2.5-pro`, `gemini:gemini-2.5-flash` | 长上下文、视觉能力强 |
| Anthropic Claude | `claude:claude-sonnet-4-5`, `claude:claude-opus-4-5-thinking` | 推理能力强 |

```bash
# 切换模型
python3 -m agent.src.pipeline --model gemini:gemini-2.5-flash --limit 5
```

### 2. Prompt 策略

| 策略 | 说明 | 视觉支持 |
|------|------|----------|
| `v1_baseline` | 两步法 (选组件→填属性) | ❌ |
| `v2_google_gui` | Google GUI 端到端生成 | ✅ |
| `v2_smart_glasses` | Smart Glasses 优化版 | ✅ |
| `v3_with_visual` | 视觉上下文增强 | ✅ |

### 3. 视频 UI Overlay

将生成的 UI 叠加到原视频上，基于眼动坐标定位 UI 位置：

```bash
python3 -m agent.src.pipeline --overlay --overlay-duration 2.0 --limit 5
```

### 4. AR Glasses UI 组件

优化的组件库，专为 AR 眼镜设计：
- `Card` (glass/solid/outline/alert variants)
- `Button` (primary/secondary/ghost)
- `Badge` (info/success/warning/error)
- `ProgressBar` (default/slim)
- `Text`, `Icon`, `Row`, `Column`, `List`

---

## 📊 Pipeline 架构

```
Input (example/ + video)
         │
         ├── ExampleLoader (100 samples)
         │
         ├── Frame Extraction (可选)
         │
    ┌────┴────┐
    │ Strategy│ v1/v2/v3
    └────┬────┘
         │
    ┌────┴────┐
    │ LLM API │ Azure/Gemini/Claude
    └────┬────┘
         │
         ├── A2UI JSON Output
         │
         ├── Preview Server (port 8000)
         │
         └── Video Overlay (可选)
```

---

## 🖥️ 预览服务器

```bash
# 启动服务器 (支持自动重载)
python3 -m agent.preview.server --port 8000 --reload

# 浏览器访问
http://localhost:8000
```

支持的功能：
- 实时渲染 A2UI JSON
- AR Glasses 风格预览
- 组件属性编辑
- 截图导出

---

## 📥 数据集

### 样本结构 (example/)

```
agent/example/
├── Task2.1/
│   └── P1_YuePan/
│       └── sample_001/
│           ├── context.json      # 上下文信息
│           ├── frames/           # 提取的视频帧
│           └── metadata.json     # 元数据
```

### 数据加载

```python
from agent.src.example_loader import ExampleLoader

loader = ExampleLoader("/home/v-tangxin/GUI/agent", "P1_YuePan")
for rec, scene in loader.iter_mvp_data(limit=10):
    print(rec.content, scene.name)
```

---

## ⚙️ 环境配置

```bash
# GPU 信息
# NVIDIA A100 80GB PCIe
# CUDA 驱动版本: 13.0 (Driver 580.95.05)
# PyTorch CUDA版本: 12.8

# 环境变量 (在 .env 中配置)
AZURE_OPENAI_ENDPOINT=xxx
AZURE_OPENAI_API_KEY=xxx
GOOGLE_API_KEY=xxx
ANTHROPIC_API_KEY=xxx
```

---

## 📤 输出格式

### A2UI 组件 JSON

```json
{
  "type": "Card",
  "id": "card_01cd8199",
  "props": {
    "variant": "glass"
  },
  "children": [
    {
      "type": "Text",
      "props": {"variant": "h1", "content": "导航到图书馆"}
    },
    {
      "type": "Badge",
      "props": {"variant": "info", "text": "500m"}
    }
  ],
  "metadata": {
    "strategy": "v2_google_gui",
    "model": "gemini:gemini-2.5-pro"
  }
}
```

---

## ✅ 已完成事项

- [x] Preview 渲染端的 AR Glasses UI 优化
- [x] 初始数据集的 100 个 examples 提取
- [x] User profile 和 context 集成到生成 Prompt
- [x] Video UI Overlay 系统实现
- [x] 多 LLM 提供商支持 (Gemini, Claude, Azure)
- [x] A2UI Lit 渲染器集成
- [x] 组件 Schema 定义

---

*Updated: 2026-02-02*
