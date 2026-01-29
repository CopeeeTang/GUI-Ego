# Smart Glasses GenUI 数据加工与生成系统 (Smart Glasses GenUI Data Processing System)

**Last Updated:** 2026-01-27

本项目是一个结合了 **Research (调研验证)** 与 **Engineering (数据工程)** 的混合型项目。主要目标是基于 Ego-centric 视频数据集（P1_YuePan），利用 Agentic Workflow 和 GPT-4o 生成符合 A2UI 协议的智能眼镜 GUI 数据，以验证 Smart Glasses GenUI 的生成范式。

---

## 📚 1. 研究背景与动机 (Research Context)

我们通过调研发现，现有的 Generative UI (如 Google GenUI) 主要针对 Web 界面，直接应用到 Smart Glasses 场景存在显著 Gap。

*   **核心 Gap**: WebUI (鼠标/触控/大屏) vs Smart Glasses (注视/语音/透视/小屏)。
*   **研究方向**: 探索适合智能眼镜的 "Minimal", "Anchored", "Context-aware" 的 UI 生成范式。

### 关键调研文档
*   [**Gap 分析与场景定义**](survey/2026-01-18%20example.md): 详细分析了 P1_YuePan 数据集中的意图分布，定义了短期 vs 长期交互模式，以及 Gap 分析。
*   [**研究问题 (Research Question)**](Research%20Question.md): 明确了 Pipeline 架构、技术决策理由和组件库 Schema 设计。
*   [**A2UI 扩展协议**](A2UI-SmartGlasses-Extension.md): 在 Google A2UI 协议基础上，针对 AR 场景扩展了 `spatial` (空间锚定) 和 `attention` (注意力管理) 字段。

---

## 🗄️ 2. 数据集 (Dataset)

本项目使用 **P1_YuePan** 数据集作为原始输入。
*   **内容**: 包含 3.17 小时的 Ego-centric 视频流、眼动数据 (Gaze) 和用户活动标签。
*   **存放位置**: 数据实体位于 Second Brain，本项目通过 Symlink 引用至 `data/` 目录。
*   **样本分析**: 参见 `survey/2026-01-18 example.md` 中的 "数据集分析" 章节。

---

## 🛠️ 3. 核心方法论 (Methodology)

为了批量生产高质量的 GUI 数据，我们设计了一套 **Agentic Data Generation Pipeline**：

### 3.1 工作流 (Agent Workflow)
借鉴 Google 的设计工作流，构建一个自动化 Agent 体系：
1.  **感知 (Perception)**: 读取视频帧与眼动数据，理解当前上下文 (Context)。
2.  **规划 (Planning)**: 模仿 UX 设计师，决定"何时显示"、"显示什么"。
3.  **生成 (Generation)**: 调用 **GPT-4o API**，基于我们定义的 Prompt Template 生成 UI 描述。
4.  **结构化 (Structuring)**: 将 UI 描述转换为符合 **A2UI Protocol** 的 JSON 数据。

### 3.2 协议 (Protocol)
采用 **Standard A2UI JSON** + **Smart Glasses Extensions**。
*   **基础**: Google A2UI (声明式 UI，安全且 LLM 友好)。
*   **扩展**: 增加了 AR 锚点、注视触发、免打扰优先级等字段 (见 `A2UI-SmartGlasses-Extension.md`)。

---

## 📂 4. 工程结构 (Project Structure)

本仓库 (`/home/v-tangxin/GUI`) 包含以下核心模块：

```text
/home/v-tangxin/GUI/
├── survey/                # [Research] 调研笔记、Gap 分析、文献综述
├── data/                  # [Data] 原始数据集 (Symlinks to raw data)
├── src/                   # [Engineering] 核心代码 (规划中)
│   ├── schema/            # Zod Schemas (A2UI + Extensions)
│   ├── generator/         # GPT-4o API 调用与 Prompt 管理
│   └── renderer/          # 简单的 Web 渲染器 (用于验证生成的 JSON)
├── scripts/               # [Tools] 批量数据处理脚本
├── A2UI-SmartGlasses-Extension.md  # [Design] 协议扩展定义
├── Research Question.md   # [Design] 研究大纲
├── project-setup-guide.md # [Docs] 工程环境搭建指南
└── test1.ipynb            # [Exp] API 连通性测试与原型实验
```

---


