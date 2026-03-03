# Streaming / Online Video Understanding: Benchmark 与 Model 全景调研

> 调研日期: 2026-03-03
> 覆盖范围: 20+ Benchmark, 27+ Model (2024.06 - 2026.02)
> 数据来源: arxiv 论文、GitHub 仓库、官方 project page

---

## 表 1: Benchmark 总览

### 1.1 Proactive Benchmark（评估"何时说话"能力）

| 名称 | 年份/会议 | Proactive? | 核心能力 | 数据规模 | 视频长度 | 指标 | 代表论文 |
|------|-----------|-----------|---------|---------|---------|------|---------|
| **OVO-Bench** | CVPR 2025 | 部分 (Forward Active Responding) | 回溯+实时+前瞻, 12 任务 | 644 视频, 2814 QA | 数分钟-30min | Accuracy + 时效性奖励 | StreamBridge, LiveCC, MMDuet2 |
| **ESTP-Bench** | NeurIPS 2025 | **是** (核心能力) | Proactive Coherence, 14 任务 (3 层) | 890 视频, 2264 QA | Ego4D 验证集 | **ESTP-F1** | EyeWO, StreamBridge, PhoStream |
| **ProactiveVideoQA** | arXiv 2025.07 | **是** (核心能力) | 主动交互, 4 场景 | 1377 视频, 1427 QA | 16.6s-360s | **PAUC** (时间动态) | MMDuet2, VideoLLM-Online, MMDuet |
| **OmniMMI** | CVPR 2025 | **是** (PA+PT 子任务) | 多模态交互, 6 子任务 | 1121 视频, 2290 QA | ~324s | Cumulative Accuracy | Gemini 1.5 Pro (27.46 avg) |
| **LiveSports-3K** | CVPR 2025 | 部分 (实时解说) | 实时解说生成, 双 Track | 416 视频, 1702 事件 | 未报告 | Win Rate (LLM-Judge) | LiveCC, StreamBridge |
| **Streamo-Bench** | arXiv 2025.12 | 部分 (叙述任务) | 定位+叙述+描述+QA, 4 任务 | 300 视频, ~3000 实例 | 未报告 | mIoU / Win Rate / Acc | Streamo, Dispider |
| **StreamingBench** | NeurIPS 2024 | 部分 (1/18 Proactive Output) | 实时视觉+多源+上下文, 18 任务 | 900 视频, 4500 QA | 3s-24min | Accuracy | Gemini 1.5 Pro (67.07), 13+ 模型 |

### 1.2 Streaming QA Benchmark（实时问答，不要求主动触发）

| 名称 | 年份/会议 | Proactive? | 核心能力 | 数据规模 | 视频长度 | 指标 | 代表论文 |
|------|-----------|-----------|---------|---------|---------|------|---------|
| **SVBench** | ICLR 2025 Spotlight | No | 多轮时序对话, 9 项能力 | 1353 视频, 49979 QA | 60-240s | SA/CC/LC/TU/IC/OS | StreamingChat, GPT-4o (66.29) |
| **OVBench** | CVPR 2025 | No | 感知+记忆+推理, 16 子任务 | ~7000 QA | 未公开 | Accuracy | VideoChat-Online (54.9) |
| **COIN** | CVPR 2019 | No | 教学视频步骤分析, 180 任务 | 11827 视频, 46354 段 | avg 2.36min | mAP/FA/Accuracy | VideoLLM-Online, OVBench 上游数据 |

### 1.3 Long Video Benchmark（离线，常用于 streaming 方法评测）

| 名称 | 年份/会议 | Proactive? | 核心能力 | 数据规模 | 视频长度 | 指标 | 代表论文 |
|------|-----------|-----------|---------|---------|---------|------|---------|
| **Video-MME** | CVPR 2025 | No | 全谱多模态视频评测, 30 子领域 | 900 视频, 2700 QA | 11s-1hr | Accuracy (w/wo subs) | 57 模型评测, Gemini 1.5 Pro (81.3) |
| **MLVU** | CVPR 2025 | No | 多任务长视频理解, 9 任务 | 1730 视频, 3102 QA | 3min-2hr (avg 15min) | Accuracy / GPT-4 Score | GPT-4o (54.5), 23+ 模型 |
| **LongVideoBench** | NeurIPS 2024 | No | 交错视频语言理解, Referring Reasoning | 3763 视频, 6678 QA | avg ~473s | Accuracy | GPT-4o (66.7) |
| **LVBench** | ICCV 2025 | No | 极长视频理解, 6 维度 | 103 视频, 1549 QA | avg ~68min (>30min) | Accuracy | Gemini-2.5-Pro (67.4), Human (94.4) |

### 1.4 Temporal Grounding Benchmark（时间定位）

| 名称 | 年份/会议 | 任务类型 | 数据规模 | 视频长度 | 指标 | 与 Streaming 关系 |
|------|-----------|---------|---------|---------|------|-----------------|
| **Charades-STA** | ICCV 2017 | Moment Retrieval | 6672 视频, 16128 pairs | avg ~30s | R@1 IoU=0.5/0.7 | 低 (视频太短) |
| **ActivityNet-MR** | ICCV 2017 | MR + Dense Captioning | 20000 视频, 72000 段 | avg ~2min | R@1 IoU=0.5/0.7, mIoU | 中 (常用于 VLM 评测) |
| **QVHighlights** | NeurIPS 2021 | MR + Highlight Detection | 10148 视频, 10310 queries | ~150s | mAP, HIT@1 | 中高 (HD 对应实时评估) |
| **Ego4D NLQ** | CVPR 2022 | Episodic Memory | ~1000h, 27k queries (v2) | avg 8.7min/clip | R@1/R@5 IoU=0.3/0.5 | **最高** (第一人称+记忆) |
| **E.T. Bench** | NeurIPS 2024 | 事件级理解 (12 子任务) | 7000 视频, 7300 样本 | avg ~129s | 多指标 | 高 (EPM/VHD 对应 streaming) |
| **TACoS** | TACL 2013 | Moment Retrieval | 127 视频, 18818 pairs | avg ~4.8min | R@1 IoU=0.3/0.5/0.7 | 低 (规模太小) |

---

## 表 2: Model 总览（含 Benchmark 数据）

### 2.1 Proactive 模型

#### A. EOS/Action Token 触发

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | ESTP-Bench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|------------|-----------|------|
| **VideoLLM-Online** | CVPR 2024 | Llama-3 + SigLIP | 8B | SFT | No | — | — | 15.5 | — | COIN SOTA, 10+ FPS |
| **ProAssist** | EMNLP 2025 | LLaMA-3.1 + SigLIP + LoRA | 8B | SFT | No | — | — | — | — | WTAG F1=0.373, Ego4D F1=0.306 |
| **EyeWO** | NeurIPS 2025 | Llama-3 + SigLIP (LIVE) | 8B | Multi-stage SFT | No | — | — | **34.7** | — | +19.2% vs LIVE |
| **LiveCC** | CVPR 2025 | Qwen2-VL-7B | 7B | Pretrain+SFT | No | **59.8** | — | — | 64.1 | LiveSports-3K: 超72B模型 |
| **Streamo** | arXiv 2025.12 | Qwen2.5-VL | 3B/7B | SFT | No | — | — | — | — | +13.83% vs Dispider |
| **STREAM-VLM** | NeurIPS 2024 D&B | LLaMA-2 + 3D-CNN | 7B | 3-stage SFT | No | — | — | — | — | FitCoach: Temp F=0.59 |

#### B. 分类头/检测器触发

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | ESTP-Bench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|------------|-----------|------|
| **MMDuet** | arXiv 2024.11 | LLaVA-OV-7B | 7B | SFT | No | — | 29.44 (PO) | 17.8 | — | QVH mAP=90 |
| **Dispider** | CVPR 2025 | Qwen2-1.5B+7B | 8.5B | 2-stage SFT | No | — | 53.12 | — | 57.2 | EgoSchema 55.6 |
| **StreamBridge** | NeurIPS 2025 | LLaVA-OV-0.5B + VLM | 0.5B+7B | SFT | No | **71.30** | **77.04** | — | — | 超 GPT-4o/Gemini |
| **StreamMind** | ICCV 2025 | Mistral-7B + EPFE | 7B+ | 2-stage SFT | No | — | — | — | — | 100 fps, COIN Step=63.7 |
| **EgoSpeak** | NAACL 2025 | ResNet-50 + wav2vec2 | 34-83M | Supervised CE | No | — | — | — | — | Ego4D mAP=69.0 |
| **ViSpeak** | ICCV 2025 | VITA 1.5 (Qwen2.5) | 7B | 3-stage SFT | No | — | 62.58 | — | — | ViSpeak-Bench: 80.42% |

#### C. 视觉变化触发

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|-----------|------|
| **TimeChat-Online** | ACM MM 2025 | Qwen2.5-VL-7B | 7B | SFT | No | 47.6 | 75.28 (RVU) | — | DTD 压缩 82.8% token |
| **OpenHOUSE** | ICCV 2025 | InternVL2-40B | 40B/8B | 部分训练 | No | — | — | — | Ego4D-GS F1=43.71 |

#### D. 困惑度触发

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|-----------|------|
| **LiveStar** | NeurIPS 2025 | InternVideo2.5 | ~8B | SFT | No | — | — | — | 语义正确性+19.5%, 时序偏差-18.1% |

#### E. Agent 规划触发

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|-----------|------|
| **StreamAgent** | ICLR 2026 | Qwen2.5-VL 3B+7B | 3B+7B | Training-free | **Yes** | 49.4 | 74.28 | 62.9 | 预期式 Agent |
| **MindPower** | arXiv 2025.11 | Qwen2.5-VL-7B | 7B | SFT+RL (GRPO) | **Yes** | — | — | — | ToM 推理, 超 GPT-4o 12.77% |
| **MMDuet2** | arXiv 2025.12 | Qwen2.5-VL-3B | 3B | SFT+RL (GRPO) | No | — | 34.69 (PO) | ~67.5 | ProactiveVQA PAUC=53.3 |

### 2.2 Responsive 模型

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|-----------|------|
| **StreamChat v2** | ICLR 2025 | LongVA | ~7B | Training-free | No | — | 64.7 (online) | — | 32 FPS, 三层记忆 |
| **Flash-VStream** | ICCV 2025 | Qwen2-VL-7B | 7B | SFT | No | 33.2 | — | — | Flash Memory 双层机制 |
| **ReKV** | ICLR 2025 | LLaVA-OV | 0.5B/7B | Training-free | No | — | — | — | KV-Cache 检索, 即插即用 |
| **VideoLLaMB** | arXiv 2024.09 | Vicuna-7B | 7B | SFT | No | — | — | 41.41 | Memory Bridge, 16→320 帧 |
| **StreamingVLM** | arXiv 2025.10 | Qwen2.5-VL-7B | 7B | SFT | No | — | — | — | 8 FPS H100, 2h+ 视频 |
| **StreamForest** | NeurIPS 2025 Spotlight | InternVL2-8B | 8B | SFT | No | 55.6 | **77.3** | — | OVBench 60.5, 事件记忆森林 |

### 2.3 SFT+RL / Agent 模型

| 模型 | 年份/会议 | 基座 | 参数量 | Training | Agent? | OVO-Bench | StreamingBench | Video-MME | 其他 |
|------|-----------|------|--------|----------|--------|-----------|---------------|-----------|------|
| **VITAL** | arXiv 2025.08 | Qwen2.5-VL-7B | 7B | SFT+RL (DGRPO) | **Yes** | — | — | 64.1 | 11 benchmark, 工具增强推理 |
| **EventMemAgent** | arXiv 2026.02 | Qwen3-VL-8B | 8B | SFT+RL (GRPO) | **Yes** | **60.75** | 77.00 (RT) | — | 超 GPT-4o (59.54), 双层事件记忆 |
| **VideoARM** | arXiv 2025.12 | GPT-5/o3 (闭源) | 闭源 | Training-free | **Yes** | — | — | **82.8** | 仅 2-3% token 消耗 |

---

## 表 3: 技术路线演进脉络

```
2024.06  VideoLLM-Online (LIVE) ─── 开创 Streaming EOS 范式
         ↓  CVPR 2024 | Llama-3 + SigLIP | 首个 streaming video LLM
         ↓  关键突破: LIVE 框架 (streaming 训练目标 + 离线→在线数据转换)
         ↓
2024.07  STREAM-VLM (FitCoach) ─── 领域特定 Action Token
         ↓  NeurIPS 2024 D&B | <next>/<feedback> 双 token 机制
         ↓  解决: 健身教练等 situated interaction 场景
         ↓
2024.09  VideoLLaMB ─── Recurrent Memory Bridge
         ↓  Vicuna-7B | Memory Bridge Layers 编码 100% 视频
         ↓  解决: 长视频信息丢失 (16帧训练→320帧推理)
         ↓
2024.11  StreamingBench 发布 ─── 首个 Streaming 评测标准
         ↓  NeurIPS 2024 | 900 视频, 18 任务, 13 模型
         ↓
         MMDuet ─── 模块化双头触发
         ↓  LLaVA-OV-7B | Dual-head (Informative + Relevance)
         ↓  关键突破: Video-Text Duet 交互格式, 分类头替代 EOS
         ↓
2025.01  OVO-Bench + Dispider ─── 解耦 P/D/R 三模块
         ↓  CVPR 2025 | Qwen2-1.5B (感知) + Qwen2-7B (响应)
         ↓  关键突破: 感知/决策/反应异步处理, StreamingBench 53.12
         ↓  OVO-Bench 定义 Backward/Realtime/Forward 三场景评测
         ↓
2025.02  EgoSpeak ─── 轻量级说话时机分类器
         ↓  NAACL 2025 | 非 VLM, 34-83M 参数, 专注"何时说"
         ↓  解决: 自然对话中的说话时机预测 (vs VLM 的重量级方案)
         ↓  SVBench 发布 ─── 最大规模 Streaming 评测 (49979 QA)
         ↓
2025.03  StreamMind ─── Event-Gated 认知
         ↓  ICCV 2025, Microsoft | Cognition Gate + EPFE
         ↓  关键突破: 100 fps (A100), 仿生感知-认知交错, LLM 仅在事件时调用
         ↓
         ViSpeak ─── 视觉指令反馈
         ↓  ICCV 2025 | 从视觉模态提取指令 (手势→对话)
         ↓  解决: 突破纯文本指令的限制
         ↓
         ReKV ─── Training-free KV-Cache 检索
         ↓  ICLR 2025 | Sliding-window + KV-Cache RAM/磁盘存储
         ↓
         OmniMMI ─── 多模态主动交互评测
         ↓  CVPR 2025 | Proactive Alerting + Turn-taking
         ↓
2025.04  LiveCC ─── ASR Streaming 大规模训练
         ↓  CVPR 2025 | Qwen2-VL | Live-CC-5M 预训练
         ↓  关键突破: 利用 YouTube CC 实现 500 万级 streaming 训练
         ↓  7B 模型超越 72B 在解说质量上; LiveSports-3K benchmark
         ↓
         TimeChat-Online ─── 视觉 Token 压缩
         ↓  ACM MM 2025 | DTD 消除 82.8% 冗余 token, 1.76x 加速
         ↓
2025.05  StreamBridge ─── 离线→Streaming 通用桥接
         ↓  NeurIPS 2025, Apple | 0.5B 激活模型 + 任意离线 VLM
         ↓  关键突破: 通用框架, StreamingBench 77.04 + OVO-Bench 71.30
         ↓  超越 GPT-4o 和 Gemini 1.5 Pro
         ↓
2025.06  ProAssist ─── Ego-centric Proactive 对话
         ↓  EMNLP 2025 | w2t_prob 触发 + 大规模 egocentric 数据集
         ↓
2025.07  ProactiveVideoQA ─── PAUC 指标
         ↓  首个追踪响应质量时间动态的评测指标
         ↓  发现: 专用 proactive 模型反而不如通用大模型
         ↓
2025.08  StreamAgent ─── 预期式 Agent (Training-free)
         ↓  ICLR 2026 | Qwen2.5-VL 3B+7B | 双模型 + KV-Cache 记忆
         ↓  关键突破: 预测未来事件时空区间, Agent 规划驱动感知
         ↓
         VITAL ─── 工具增强 RL
         ↓  Qwen2.5-VL + Visual Toolbox + DGRPO
         ↓  关键突破: 端到端 Agentic 视频推理, 按需密采帧
         ↓
2025.09  StreamForest ─── 事件记忆森林
         ↓  NeurIPS 2025 Spotlight | InternVL2-8B | 树结构记忆管理
         ↓  StreamingBench 77.3, 极端压缩保持 96.8% 精度
         ↓
         OpenHOUSE ─── 层级事件理解
         ↓  ICCV 2025 | 动作边界检测 + VLM 稀疏触发
         ↓
2025.10  EyeWO ─── ESTP-Bench + 三阶段训练
         ↓  NeurIPS 2025 | 提出 ESTP-F1 指标, 34.7% (最佳)
         ↓  关键突破: 首个以 Proactive 为核心的 Ego benchmark
         ↓
         StreamingVLM ─── 无限视频流实时理解
         ↓  MIT Han Lab | Compact KV Cache + SFT 短 chunk 策略
         ↓
2025.11  LiveStar ─── 困惑度触发
         ↓  NeurIPS 2025 | SVeD 门控 (perplexity threshold)
         ↓  关键突破: 单次 forward pass 决定响应/静默
         ↓
         MindPower ─── Theory of Mind + VLM
         ↓  Qwen2.5-VL + GRPO | Robot-Centric ToM 推理
         ↓
2025.12  Streamo ─── 通用 Streaming 指令微调
         ↓  Qwen2.5-VL | Streamo-Instruct-465K
         ↓
         MMDuet2 ─── Multi-turn RL 增强主动交互
         ↓  关键突破: 首个将 GRPO RL 应用于 proactive timing
         ↓  无需手动调阈值, ProactiveVideoQA SOTA
         ↓
         VideoARM ─── Agentic 层级记忆推理
         ↓  GPT-5/o3 | O-T-A-M 循环 + 三级记忆
         ↓  Video-MME 82.8, 仅 1/34 token 消耗
         ↓
2026.02  EventMemAgent ─── 事件记忆 Agent + GRPO
         ↓  Qwen3-VL-8B | 双层事件记忆 + 自适应工具调用
         ↓  关键突破: OVO-Bench 60.75 超越 GPT-4o (59.54)
         ↓  StreamingBench Real-time 77.00 (仅 <=32 帧)
```

---

## 三维分类交叉分析

### 维度 A × 维度 B: Proactive/Responsive × Training 方式

|  | Training-free | SFT Only | SFT + RL |
|--|---------------|----------|----------|
| **Proactive - EOS Token** | — | VideoLLM-Online, ProAssist, EyeWO, LiveCC, Streamo, STREAM-VLM | — |
| **Proactive - 分类头** | — | MMDuet, Dispider, StreamBridge, StreamMind, EgoSpeak, ViSpeak | — |
| **Proactive - 视觉变化** | — | TimeChat-Online, OpenHOUSE (部分) | — |
| **Proactive - 困惑度** | — | LiveStar | — |
| **Proactive - Agent** | StreamAgent, VideoARM | — | MindPower, MMDuet2 |
| **Responsive** | StreamChat v2, ReKV | StreamChat v1, Flash-VStream, VideoLLaMB, StreamingVLM, StreamForest | VITAL, EventMemAgent |

### 维度 A × 维度 C: Proactive/Responsive × Agent

|  | Non-Agent | Agent |
|--|-----------|-------|
| **Proactive** | VideoLLM-Online, ProAssist, EyeWO, LiveCC, Streamo, STREAM-VLM, MMDuet, Dispider, StreamBridge, StreamMind, EgoSpeak, ViSpeak, TimeChat-Online, OpenHOUSE, LiveStar, MMDuet2 | StreamAgent, MindPower |
| **Responsive** | StreamChat, Flash-VStream, ReKV, VideoLLaMB, StreamingVLM, StreamForest | VITAL, EventMemAgent, VideoARM |

---

## 关键洞察

### 1. Benchmark 生态: 碎片化与标准化并存

Proactive 评测经历了快速演进：从 2024 年 StreamingBench 的 1/18 子任务（Proactive Output），到 2025 年涌现出 ESTP-Bench、ProactiveVideoQA、OmniMMI 三个以 proactive 为核心的 benchmark。然而每个模型仍倾向自建评测（LiveSports-3K, ViSpeak-Bench, Streamo-Bench, OmniStar, ODV-Bench, Inf-Streams-Eval 等），导致横向对比困难。**StreamingBench 和 OVO-Bench 是目前仅有的通用评测标准**，建议新工作至少在这两个 benchmark 上报告结果。

两个专用 Proactive 指标值得关注：
- **ESTP-F1**: F1 框架统一时间准确性和内容质量（FP=不该说时说, FN=该说时没说）
- **PAUC**: AUC 框架追踪响应质量的时间动态（时效性 vs 正确性权衡）

### 2. 触发机制: 从 EOS 到分类头到 Agent

技术路线经历了三代演进。第一代（2024H1）是 VideoLLM-Online 开创的 EOS Token 自回归范式，优势是训练推理统一，但时机决策粗糙。第二代（2024H2-2025H1）是 MMDuet/Dispider/StreamBridge 引入的分类头范式，将触发决策从生成任务中解耦，定量评测上普遍优于 EOS 方法，StreamBridge 更以 0.5B 激活模型达到 StreamingBench/OVO-Bench 双 SOTA。第三代（2025H2+）是 StreamAgent/VITAL/EventMemAgent 引入的 Agent 范式，具备规划、工具调用和多步推理能力。

值得注意的是，困惑度触发（LiveStar 的 SVeD）和视觉变化触发（TimeChat-Online 的 DTD）提供了轻量级替代方案。前者通过单次 forward pass 即可决策，后者兼顾 token 压缩（82.8%）和触发检测。

### 3. 训练范式: GRPO 成为 RL 标准

所有 SFT+RL 模型（MMDuet2, MindPower, VITAL/DGRPO, EventMemAgent）均采用 GRPO 或其变体，而非 PPO。这反映了 GRPO 在 VLM post-training 中的主导地位。Training-free 方法（StreamAgent, VideoARM, ReKV）依赖强基座模型，适合快速验证但性能上限受限。

### 4. 基座模型: Qwen 系列主导

从 Llama-2/3（2024）到 Qwen2-VL/Qwen2.5-VL/Qwen3-VL（2025-2026）的转变非常明显。27 个模型中超过 12 个使用 Qwen 系列作为基座，InternVL/InternVideo 系列是第二选择。这反映了 Qwen 视觉语言模型在视频理解领域的生态优势。

### 5. 性能天花板与人机差距

| Benchmark | 最佳模型 | 最佳分数 | 人类基线 | 差距 |
|-----------|---------|---------|---------|------|
| OVO-Bench | StreamBridge | 71.30 | 92.81 | 21.5 |
| StreamingBench | StreamForest | 77.3 | 91.66 | 14.4 |
| ESTP-Bench | EyeWO | 34.7 | — | — |
| SVBench | GPT-4o | 66.29 | — | — |
| Video-MME | VideoARM | 82.8 | — | — |

StreamingBench 的差距已缩小到 ~14%，但 ESTP-Bench 的 34.7% 表明 **proactive timing 仍是最大挑战**。

### 6. 研究空白

1. **Proactive + Agent 的交叉**：StreamAgent 是唯一同时具备 Proactive 触发和 Agent 规划的模型，但它是 training-free 的。目前没有经过 RL 训练的 Proactive Agent 模型。
2. **Streaming Temporal Grounding**：现有 temporal grounding benchmark 均为离线设置，缺少专为 streaming 设计的时间定位评测。
3. **多模态主动交互**：OmniMMI 是唯一评估音频+视频+主动推理的 benchmark，但覆盖率还很低。
4. **Proactive 训练方法的反思**：ProactiveVideoQA 发现专用 proactive 模型（VideoLLM-Online, MMDuet）反而不如通用大模型，质疑当前的 SFT 训练范式。

---

## 推荐评估组合

| 目标 | 推荐组合 |
|------|---------|
| **全面评估 streaming 能力** | StreamingBench + OVO-Bench + ESTP-Bench |
| **评估 proactive timing** | ESTP-Bench (F1) + ProactiveVideoQA (PAUC) |
| **评估 streaming + 离线泛化** | StreamingBench + Video-MME + MLVU |
| **评估 ego-centric streaming** | ESTP-Bench + Ego4D NLQ + OVBench |
| **快速基准测试** | StreamingBench (18 任务广覆盖) |

---

## 参考文献

### Benchmark
1. OVO-Bench: [arXiv:2501.05510](https://arxiv.org/abs/2501.05510) | [GitHub](https://github.com/JoeLeelyf/OVO-Bench)
2. ESTP-Bench: [arXiv:2510.14560](https://arxiv.org/abs/2510.14560)
3. StreamingBench: [arXiv:2411.03628](https://arxiv.org/abs/2411.03628) | [GitHub](https://github.com/THUNLP-MT/StreamingBench)
4. SVBench: [arXiv:2502.10810](https://arxiv.org/abs/2502.10810) | [GitHub](https://github.com/sotayang/SVBench)
5. OVBench: [arXiv:2501.00584](https://arxiv.org/abs/2501.00584)
6. ProactiveVideoQA: [arXiv:2507.09313](https://arxiv.org/abs/2507.09313)
7. OmniMMI: [arXiv:2503.22952](https://arxiv.org/abs/2503.22952)
8. LiveSports-3K: [arXiv:2504.16030](https://arxiv.org/abs/2504.16030) | [GitHub](https://github.com/showlab/livecc)
9. Streamo-Bench: [arXiv:2512.21334](https://arxiv.org/abs/2512.21334)
10. Video-MME: [arXiv:2405.21075](https://arxiv.org/abs/2405.21075) | [GitHub](https://github.com/MME-Benchmarks/Video-MME)
11. MLVU: [arXiv:2406.04264](https://arxiv.org/abs/2406.04264) | [GitHub](https://github.com/JUNJIE99/MLVU)
12. LongVideoBench: [arXiv:2407.15754](https://arxiv.org/abs/2407.15754)
13. LVBench: [arXiv:2406.08035](https://arxiv.org/abs/2406.08035)
14. E.T. Bench: [NeurIPS 2024](https://github.com/PolyU-ChenLab/ETBench)
15. COIN: [arXiv:1903.02874](https://arxiv.org/abs/1903.02874)

### Model
16. VideoLLM-Online: [arXiv:2406.11816](https://arxiv.org/abs/2406.11816) | [GitHub](https://github.com/showlab/videollm-online)
17. ProAssist: [arXiv:2506.05904](https://arxiv.org/abs/2506.05904) | [GitHub](https://github.com/pro-assist/ProAssist)
18. EyeWO: [arXiv:2510.14560](https://arxiv.org/abs/2510.14560)
19. LiveCC: [arXiv:2504.16030](https://arxiv.org/abs/2504.16030) | [GitHub](https://github.com/showlab/livecc)
20. Streamo: [arXiv:2512.21334](https://arxiv.org/abs/2512.21334)
21. STREAM-VLM: [arXiv:2407.08101](https://arxiv.org/abs/2407.08101) | [GitHub](https://github.com/Qualcomm-AI-research/FitCoach)
22. MMDuet: [arXiv:2411.17991](https://arxiv.org/abs/2411.17991) | [GitHub](https://github.com/yellow-binary-tree/MMDuet)
23. Dispider: [arXiv:2501.03218](https://arxiv.org/abs/2501.03218) | [GitHub](https://github.com/Mark12Ding/Dispider)
24. StreamBridge: [arXiv:2505.05467](https://arxiv.org/abs/2505.05467) | [GitHub](https://github.com/apple/ml-streambridge)
25. StreamMind: [arXiv:2503.06220](https://arxiv.org/abs/2503.06220) | [GitHub](https://github.com/xinding-sys/StreamMind)
26. EgoSpeak: [arXiv:2502.14892](https://arxiv.org/abs/2502.14892)
27. ViSpeak: [arXiv:2503.12769](https://arxiv.org/abs/2503.12769) | [GitHub](https://github.com/HumanMLLM/ViSpeak)
28. TimeChat-Online: [arXiv:2504.17343](https://arxiv.org/abs/2504.17343) | [GitHub](https://github.com/yaolinli/TimeChat-Online)
29. OpenHOUSE: [arXiv:2509.12145](https://arxiv.org/abs/2509.12145)
30. LiveStar: [arXiv:2511.05299](https://arxiv.org/abs/2511.05299) | [GitHub](https://github.com/sotayang/LiveStar)
31. StreamAgent: [arXiv:2508.01875](https://arxiv.org/abs/2508.01875)
32. MindPower: [arXiv:2511.23055](https://arxiv.org/abs/2511.23055)
33. MMDuet2: [arXiv:2512.06810](https://arxiv.org/abs/2512.06810) | [GitHub](https://github.com/yellow-binary-tree/mmduet2)
34. StreamChat: [arXiv:2412.08646](https://arxiv.org/abs/2412.08646) (v1) / [arXiv:2501.13468](https://arxiv.org/abs/2501.13468) (v2)
35. Flash-VStream: [arXiv:2506.23825](https://arxiv.org/abs/2506.23825) | [GitHub](https://github.com/IVGSZ/Flash-VStream)
36. ReKV: [arXiv:2503.00540](https://arxiv.org/abs/2503.00540) | [GitHub](https://github.com/Becomebright/ReKV)
37. VideoLLaMB: [arXiv:2409.01071](https://arxiv.org/abs/2409.01071) | [GitHub](https://github.com/bigai-nlco/VideoLLaMB)
38. StreamingVLM: [arXiv:2510.09608](https://arxiv.org/abs/2510.09608) | [GitHub](https://github.com/mit-han-lab/streaming-vlm)
39. StreamForest: [arXiv:2509.24871](https://arxiv.org/abs/2509.24871) | [GitHub](https://github.com/MCG-NJU/StreamForest)
40. VITAL: [arXiv:2508.04416](https://arxiv.org/abs/2508.04416)
41. EventMemAgent: [arXiv:2602.15329](https://arxiv.org/abs/2602.15329)
42. VideoARM: [arXiv:2512.12360](https://arxiv.org/abs/2512.12360)
