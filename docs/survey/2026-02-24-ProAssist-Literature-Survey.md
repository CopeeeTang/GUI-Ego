# ProAssist 实验相关文献调研

**日期**: 2026-02-24
**范围**: 2024-2026 年最新文献
**目标**: 定位 Adaptive-D + 变化调制方法在现有文献中的位置

---

## 方向一: Motion/Change-based Attention for Proactive Assistants

### ProAssist: Proactive Assistant Dialogue Generation from Streaming Egocentric Videos (2025)
- **作者/出处**: Zhang et al., EMNLP 2025 (UMich SLED Lab)
- **核心方法**: 从标注的第一人称视频合成对话数据集 (ProAssist dataset)，训练端到端模型处理流式视频输入，在关键时刻决定是否说话。通过学习决策点的 speak/silent 二分类来处理极度不平衡的数据。
- **与我们工作的关系**: 这是我们实验的核心 baseline。ProAssist 使用 VLM 端到端学习 "何时说话"，而我们的方法使用 SigLIP 变化信号 (Adaptive-D) 作为轻量级前置触发器，理论上可以与 ProAssist 的 VLM 决策互补。
- **关键发现**: ProAssist 模型支持多帧率 (I=1/5/10 tokens per frame)，L4096 context length，解决了长视频处理问题。数据不平衡是核心挑战——说话帧极其稀疏。
- **链接**: https://aclanthology.org/2025.emnlp-main.605/ | https://github.com/pro-assist/ProAssist

### Eyes Wide Open (VideoLLM-EyeWO): Ego Proactive Video-LLM for Streaming Video (2025)
- **作者/出处**: Zhang et al., arXiv 2510.14560 (Oct 2025)
- **核心方法**: 提出 ESTP-Bench 基准和 ESTP-F1 指标，以及三阶段数据引擎 (One-to-one, One-to-many, Many-to-many) + 主动动态压缩技术。定义了三个关键属性: Proactive Coherence, Just-in-Time Responsiveness, Synchronized Efficiency。
- **与我们工作的关系**: ESTP-Bench 是我们当前使用的评估基准。EyeWO 代表端到端训练方法的 SOTA，而我们的 Adaptive-D 方法是无需训练的 inference-time 方案。EyeWO 比 VideoLLM-Online 高 +19.2%，比最佳 polling baseline 高 +11.8%。
- **关键发现**: 三类主动任务 (explicit, implicit, contextual)，EyeWO 在所有类型上均显著优于基线。

---

## 方向二: Streaming VLM with Adaptive Speaking

### MMDuet: VideoLLM Knows When to Speak (2024)
- **作者/出处**: Wang, Meng et al., EMNLP 2025 Findings (Nov 2024 arXiv)
- **核心方法**: 提出 video-text duet 交互格式，视频持续播放，用户和模型都可在任意帧后插入文本消息。构建 MMDuetIT 数据集训练模型，基于 LLaVA-OneVision。提出 MAGQA (Multi-Answer Video Grounded QA) 任务。
- **与我们工作的关系**: MMDuet 定义了 "何时说话" 的 duet 范式，是流式 VLM 的重要 baseline。它的决策机制依赖端到端训练，而非显式的触发信号。
- **关键发现**: Duet 格式相比传统全视频问答在时间敏感任务上有显著提升。

### MMDuet2: Enhancing Proactive Interaction with Multi-Turn RL (2025)
- **作者/出处**: arXiv 2512.06810 (Dec 2025)
- **核心方法**: 基于 MMDuet 基础上，使用多轮强化学习 (Multi-Turn RL) 训练模型自主决定何时回复或保持沉默。关键创新: 不需要手动调整响应阈值，也不需要标注精确回复时间。
- **与我们工作的关系**: MMDuet2 通过 RL 解决了阈值调优问题，而我们的 Adaptive-D 用任务类型自适应阈值 (task-specific tau) 解决类似问题。两种方法互补: RL 是端到端方案，Adaptive-D 是可解释的模块化方案。
- **关键发现**: 在 ProactiveVideoQA benchmark 上达到 SOTA。52k 视频的 SFT+RL 训练。

### StreamBridge: Turning Offline Video-LLMs into Proactive Streaming Assistants (2025)
- **作者/出处**: Apple Research, NeurIPS 2025 (May 2025 arXiv)
- **核心方法**: 两大组件: (1) memory buffer + round-decayed compression 支持长上下文多轮交互; (2) 解耦的轻量级 activation model 决定何时主动回复。构建 Stream-IT 大规模流式视频理解数据集。
- **与我们工作的关系**: StreamBridge 的 "activation model" (决定何时激活回复) 与我们的 Adaptive-D 触发器在功能上非常相似——都是独立于主 VLM 的轻量级触发模块。关键区别是 StreamBridge 使用学习的 activation model，我们使用 SigLIP 余弦相似度 + 自适应阈值。
- **关键发现**: Qwen2-VL 经 Stream-IT 微调后在 OVO-Bench (71.30) 和 Streaming-Bench (77.04) 上超越 GPT-4o 和 Gemini 1.5 Pro。

### Dispider: Enabling Video LLMs with Active Real-Time Interaction (2025)
- **作者/出处**: Qian et al., CVPR 2025 (CUHK + Shanghai AI Lab)
- **核心方法**: 解耦感知 (Perception)、决策 (Decision)、反应 (Reaction) 三个模块并异步处理。轻量级主动流式视频处理模块跟踪视频流并识别最佳交互时机，异步交互模块提供详细回复。
- **与我们工作的关系**: Dispider 的 "Decision" 模块与我们的触发机制最接近——都试图识别 "最佳交互时机"。Dispider 使用解耦架构，我们使用 SigLIP 变化信号，但核心思想相同: 分离 "何时说" 和 "说什么"。
- **关键发现**: 在 StreamingBench 上超越 VideoLLM-Online，在 EgoSchema, VideoMME 等离线基准上也表现出色。

### VideoLLM-Online: Online Video Large Language Model for Streaming Video (2024)
- **作者/出处**: Chen et al., CVPR 2024 (ShowLab)
- **核心方法**: 提出 Learning-In-Video-Stream (LIVE) 框架，包含: (1) 连续流式输入的语言建模目标; (2) 将离线时间标注转换为流式对话格式的数据生成方案; (3) 优化的推理流水线。基于 Llama-2/Llama-3 构建。
- **与我们工作的关系**: VideoLLM-Online 是该领域的开创性工作，定义了流式视频 LLM 的基本范式。所有后续工作 (MMDuet, EyeWO, StreamBridge 等) 都以此为基础或对比对象。
- **关键发现**: 支持 5 分钟视频流对话，A100 上 10+ FPS。在识别、字幕、预测等离线基准上也是 SOTA。

### LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale (2025)
- **作者/出处**: Chen et al., CVPR 2025 (ShowLab)
- **核心方法**: 利用大规模 ASR 转录文本进行训练，将 ASR 词和视频帧按时间戳密集交错。构建 Live-CC-5M (预训练) + Live-WhisperX-526K (SFT) 数据集。
- **与我们工作的关系**: LiveCC 展示了使用语音信号辅助视频理解的可能性，虽然不直接相关，但其 "何时产生评论" 的时机决策机制是相关参考。
- **关键发现**: LiveCC-7B 在实时评论质量上超越 LLaVA-Video-72B。

### ViSpeak: Visual Instruction Feedback in Streaming Videos (2025)
- **作者/出处**: arXiv 2503.12769 (Mar 2025)
- **核心方法**: 三阶段微调，使模型能从视觉内容中提取指令并主动说话。定义了 7 个子任务: Visual Wake-Up, Anomaly Warning, Gesture Understanding, Visual Reference, Visual Interruption, Humor Reaction, Visual Termination。
- **与我们工作的关系**: ViSpeak 的 "Anomaly Warning" 子任务与我们的变化检测触发最接近——都是通过检测视觉异常/变化来触发响应。ViSpeak 使用端到端学习，我们使用 SigLIP 特征距离。
- **关键发现**: Informative head 实现 38.80 PO (Proactive Output) 分数，是少数能主动说话的模型之一。

---

## 方向三: ProAssist 的引用和后续工作

### ProAssist 论文 (EMNLP 2025 Main)
- **GitHub**: https://github.com/pro-assist/ProAssist
- **模型**: HuggingFace 594zyc/ProAssist-Model-L4096-I1 (及 I5, I10 变体)
- **数据集**: 152 GB 预处理数据，多领域合成对话
- **当前状态**: 代码和模型已开源，数据集可用。未发现直接的后续改进论文。

### ProActLLM Workshop (CIKM 2025)
- **出处**: 2025 年 11 月，首尔，与 CIKM 2025 共同举办
- **主题**: 探索从被动问答到主动信息寻求助手的转变
- **与我们工作的关系**: 这是该领域的专题研讨会，表明主动助手已成为一个独立研究方向。

### ProactiveVideoQA Benchmark (2025)
- **作者/出处**: arXiv 2507.09313 (Jul 2025)
- **核心方法**: 第一个全面评估系统主动交互能力的基准。提出 PAUC 指标，考虑模型响应的时间动态，比传统评估指标更符合人类偏好。
- **与我们工作的关系**: 这是另一个可用于评估我们方法的基准。PAUC 指标对我们的时机决策评估可能比 ESTP-F1 更合适，因为它考虑了响应的时间准确性。
- **关键发现**: 使用固定长度 chunk 的简单规则策略可将离线 VLM 适配为主动交互模型。

---

## 方向四: Uncertainty-based Triggering

### Mitigating LLM Hallucinations via Conformal Abstention (2024)
- **作者/出处**: Abbasi-Yadkori, Kuzborskij et al., NeurIPS 2024 Workshop (Google)
- **核心方法**: 使用 LLM 自评估采样响应间的相似性，结合 conformal prediction 技术开发弃权程序，对幻觉率提供严格的理论保证。
- **与我们工作的关系**: 直接相关——我们的 logprob gap 本质上是一种 uncertainty measure，而 conformal abstention 提供了更严格的理论框架来控制错误率。可以将我们的 Adaptive-D 阈值重新表述为一种 conformal abstention 策略。
- **关键发现**: 在闭卷开放域生成式问答上可靠地约束幻觉率，同时保持不过分保守的弃权率。

### Selective Generation for Controllable Language Models (2024)
- **作者/出处**: Lee et al., NeurIPS 2024 Spotlight (POSTECH)
- **核心方法**: 提出 SGen^Sup 和 SGen^Semi 两种算法，利用文本蕴含 (textual entailment) 评估生成序列正确性，使用 conformal prediction 伪标注未标注数据，控制 FDR-E (False Discovery Rate w.r.t. Entailment)。
- **与我们工作的关系**: 提供了控制生成质量的理论基础。我们的 Adaptive-D 阈值可以类比为 "selective prediction" 的特例——通过 SigLIP 变化信号决定何时 "生成" (说话)。
- **关键发现**: SGen^Semi 的 conformal pseudo-labeling 策略与我们的自适应阈值在精神上相似——都试图在不确定性高时抑制输出。

### SelectLLM: Calibrating LLMs for Selective Prediction (2025)
- **作者/出处**: OpenReview 2025
- **核心方法**: 端到端方法，将 selective prediction 集成到微调过程中，优化覆盖域上的模型性能，在预测覆盖率和效用之间取得更好的平衡。
- **与我们工作的关系**: SelectLLM 的 "何时回答/何时弃权" 与我们的 "何时说话/何时沉默" 本质上是同一问题。
- **关键发现**: 通过微调集成 selective prediction，比 post-hoc 方法更有效。

### ReCoVERR: Reducing Unnecessary Abstention in Vision-Language Reasoning (2024)
- **作者/出处**: ACL Findings 2024
- **核心方法**: 当 VLM 预测置信度低时，不直接弃权，而是通过 LLM 生成相关问题向 VLM 收集高置信度证据。如果证据足够支持预测则回答，否则弃权。
- **与我们工作的关系**: ReCoVERR 的 "收集证据再决定" 策略与我们的多信号融合思路相关——当变化信号不确定时，可以结合其他信号源做决策。
- **关键发现**: 在 VQAv2 和 A-OKVQA 上多回答 20% 的问题而不降低系统准确率。

### Estimating LLM Uncertainty with Logits (LogU) (2025)
- **作者/出处**: arXiv 2502.00290 (Feb 2025)
- **核心方法**: Logits-induced Token Uncertainty (LogU)，将 logits 视为 Dirichlet 分布的参数，区分 aleatoric uncertainty (数据固有不确定性) 和 epistemic uncertainty (知识不确定性)。实时估计，无需采样。
- **与我们工作的关系**: 非常直接相关。我们的 logprob gap 是一种简化的 uncertainty measure，而 LogU 提供了更理论化的框架。LogU 能表达 "知道但不确定" vs "完全不知道" 的区别，这与我们在不同任务类型上观察到的 gap 分布差异一致。
- **关键发现**: 相比概率方法，LogU 能区分 "确定正确", "不知道", "缺乏知识但有建议", "知道多个答案" 四种状态。

### Explicit Abstention Knobs for Predictable Reliability in Video QA (2025)
- **作者/出处**: arXiv 2601.00138 (Dec 2025)
- **核心方法**: 研究基于置信度的弃权能否在视频问答中提供可靠的错误率控制。使用 NExT-QA + Gemini 2.0 Flash，验证了 in-distribution 下置信度阈值可提供机械式控制。
- **与我们工作的关系**: 最直接相关的工作之一。该论文发现 confidence-based abstention 在分布内有效但在分布偏移下失效，这与我们在不同任务类型 (Text-Rich vs Action Recognition) 上观察到的 Adaptive-D 表现差异一致。
- **关键发现**: Conformal 方法提供的是 marginal guarantee (可交换性下)，而非 per-instance guarantee。在 evidence shift 下需要 warrant-based 表述。

### Logprobs Know Uncertainty: Fighting LLM Hallucinations (2025)
- **作者/出处**: FSE 2025 Poster
- **核心方法**: 使用 logprobs 来量化 LLM 不确定性以检测幻觉。
- **与我们工作的关系**: 直接支持我们使用 logprob gap 作为触发信号的理论合理性。
- **关键发现**: Logprobs 确实编码了有意义的不确定性信息。

### Selective Conformal Uncertainty in Large Language Models (2025)
- **作者/出处**: ACL 2025 Long Paper
- **核心方法**: 将 conformal prediction 与 selective classification 统一为两阶段风险控制框架: 第一阶段决定接受哪些样本，第二阶段为接受的样本构建 conformal prediction set。
- **与我们工作的关系**: 我们的方法可以重新表述为这个两阶段框架: 第一阶段 = Adaptive-D 决定是否触发 VLM (selection)，第二阶段 = VLM 生成回复 (prediction)。

---

## 方向五: SigLIP 变化 vs 光流 vs VLM 推理对比

### SigLIP 2: Multilingual Vision-Language Encoders (2025)
- **作者/出处**: Google, arXiv 2502.14786 (Feb 2025)
- **核心方法**: 在原始 SigLIP 基础上增加 captioning-based pretraining、self-distillation、masked prediction、online data curation，显著提升语义理解、定位和密集特征能力。
- **与我们工作的关系**: 我们使用 SigLIP 特征的余弦距离来检测帧间变化。SigLIP 2 的改进 (特别是密集特征和定位能力) 可能使我们的变化检测更精确。
- **关键发现**: 所有尺度上均超越 SigLIP v1，特别是在定位和密集预测任务上有显著提升。

### FDA-CLIP: Frame-Difference Guided Dynamic Region Perception (2025)
- **作者/出处**: arXiv 2510.21806 (Oct 2025)
- **核心方法**: 使用帧差分引导 CLIP 模型关注视频中的动态区域，避免静态背景帧和冗余帧稀释关键帧的语义信息。
- **与我们工作的关系**: 最直接相关的工作之一。FDA-CLIP 同样利用帧间差异 + CLIP 特征来识别重要变化，但用于文本-视频检索而非触发决策。验证了 "帧差分+语义特征" 组合的有效性。
- **关键发现**: 动作转换帧和物体交互帧是最有信息量的帧——这与我们的 Adaptive-D 在特定任务类型上表现好坏的观察一致。

### 时序差分网络 (TDSNet) 方法 (2024)
- **作者/出处**: Information Sciences, 2024
- **核心方法**: 避免光流和自注意力机制，通过相邻帧特征相减获取时序特征 (temporal difference)，用于视频语义分割。
- **与我们工作的关系**: 与我们的 SigLIP cosine distance 方法在精神上非常相似——都是通过特征空间中的帧间差异来捕捉变化，而非像素级光流。
- **关键发现**: 光流方法的三大问题: (1) 依赖光流预测精度导致误差累积; (2) 错误光流降低精度; (3) 额外光流网络计算量大。特征差分方法回避了这些问题。

---

## 方向六: EyeWO 和 ESTP-Bench

### ESTP-Bench 详细分析
- **定义**: Ego Streaming Proactive Benchmark
- **指标**: ESTP-F1 — 综合衡量主动响应的时机准确性和内容质量
- **三大属性**:
  1. Proactive Coherence: 处理多样问题类型，维护上下文一致性
  2. Just-in-Time Responsiveness: 基于视觉准备度确定精确回答时机
  3. Synchronized Efficiency: 确保感知和推理时间对齐

### 当前 SOTA 排名 (截至 2025.10):
| 模型 | ESTP-F1 | 备注 |
|------|---------|------|
| VideoLLM-EyeWO | 最高 | +19.2% vs VideoLLM-Online |
| 最佳 Polling Baseline | 中等 | EyeWO +11.8% 超越 |
| VideoLLM-Online | 基线 | 开创性工作 |

---

## 补充: 重要的基准和评估工具

### ProactiveVideoQA (2025)
- 第一个全面评估主动交互的 benchmark
- PAUC 指标考虑响应时间动态
- 支持将离线 VLM 通过 chunk 策略适配为主动模型

### OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video (CVPR 2025)
- 在线视频理解综合评估
- StreamBridge (Qwen2-VL) 达到 71.30 分

### Streaming-Bench
- 流式视频理解评估
- StreamBridge (Qwen2-VL) 达到 77.04 分

---

## 文献综述总结

### 1. 我们的方法在现有文献中的定位

我们的 **Adaptive-D + 变化调制** 方法位于以下交叉点:

**触发机制维度**:
- 端到端学习触发 (EyeWO, MMDuet2, ProAssist) ←→ **我们: 特征距离触发** ←→ 规则触发 (Polling)
- 学习型 activation model (StreamBridge) 与我们的 SigLIP 余弦距离触发最接近
- Dispider 的解耦 Perception-Decision-Reaction 架构与我们的模块化设计理念一致

**不确定性控制维度**:
- Conformal abstention (Abbasi-Yadkori 2024) 提供理论框架
- LogU (2025) 为 logprob-based uncertainty 提供 Dirichlet 分布解释
- Selective Generation (Lee 2024) 提供 FDR 控制的理论保证
- **我们的 logprob gap 是这些理论方法的实用近似**

**变化检测维度**:
- 像素级光流 → 特征差分 (TDSNet) → **语义级 SigLIP 距离 (我们)** → 端到端变化建模
- FDA-CLIP 验证了 "帧差分+语义特征" 组合的有效性

**独特贡献**: 我们是 (据调研) 唯一将 **视觉编码器变化信号** 与 **logprob-based 不确定性** 结合用于主动助手触发的工作。现有方法要么纯端到端 (需要大规模训练)，要么纯规则 (无自适应能力)。

### 2. 最相关的 Baseline 方法

| 优先级 | 方法 | 关系 |
|--------|------|------|
| 1 | **ProAssist** (Zhang 2025) | 核心 baseline，我们直接在其数据集/模型上改进 |
| 2 | **EyeWO** (Zhang 2025) | ESTP-Bench SOTA，端到端方案的上界 |
| 3 | **StreamBridge** (Apple 2025) | activation model 思路最接近我们的触发器 |
| 4 | **Dispider** (Qian 2025) | 解耦架构与我们的模块化设计最相似 |
| 5 | **MMDuet2** (2025) | RL-based 阈值学习 vs 我们的自适应阈值 |
| 6 | **Polling Baseline** | 最简单的定时触发策略 |

### 3. 潜在引用论文列表

**必引** (直接相关):
1. ProAssist (Zhang et al., EMNLP 2025) — 核心 baseline 和数据集
2. Eyes Wide Open / ESTP-Bench (Zhang et al., 2025) — 评估基准
3. MMDuet (Wang et al., EMNLP 2025 Findings) — 定义 duet 范式
4. MMDuet2 (2025) — RL-based 主动交互 SOTA
5. StreamBridge (Apple, NeurIPS 2025) — activation model
6. VideoLLM-Online (Chen et al., CVPR 2024) — 开创性流式 VLM
7. SigLIP / SigLIP 2 (Google, 2025) — 我们使用的视觉编码器

**强烈推荐引用** (理论支撑):
8. Mitigating LLM Hallucinations via Conformal Abstention (2024) — uncertainty 理论
9. Selective Generation (Lee et al., NeurIPS 2024) — selective prediction 理论
10. LogU: Estimating LLM Uncertainty with Logits (2025) — logprob uncertainty 理论
11. Explicit Abstention Knobs for Video QA (2025) — 视频 QA 弃权

**推荐引用** (方法对比):
12. Dispider (Qian et al., CVPR 2025) — 解耦架构参考
13. ViSpeak (2025) — 视觉指令反馈
14. LiveCC (Chen et al., CVPR 2025) — 流式评论
15. FDA-CLIP (2025) — 帧差分+CLIP 特征
16. ReCoVERR (ACL Findings 2024) — 减少过度弃权
17. ProactiveVideoQA (2025) — 额外评估基准
18. SelectLLM (2025) — 选择性预测
19. Selective Conformal Uncertainty in LLMs (ACL 2025) — 两阶段风险控制

**可选引用** (背景):
20. ProActLLM Workshop (CIKM 2025) — 领域研讨会
21. TDSNet (Info Sciences 2024) — 时序差分 vs 光流

---

## 关键洞察

1. **我们的方法填补了一个空白**: 现有流式 VLM 方法都依赖大规模端到端训练来学习 "何时说话"。我们用轻量级 SigLIP 变化信号 + 自适应阈值实现了无需额外训练的触发机制，这是一个独特的定位。

2. **StreamBridge 的 activation model 是最接近的竞品**: 但它需要训练，我们的方法是 training-free 的。

3. **Conformal prediction 可以为我们的方法提供理论包装**: 将 Adaptive-D 阈值重新表述为 conformal abstention 策略，可以获得错误率的理论保证。

4. **任务自适应阈值 (per-type tau) 与 MMDuet2 的 RL 方法互补**: MMDuet2 需要训练数据和 GPU，我们的方法只需要在验证集上做阈值搜索。

5. **FDA-CLIP 验证了我们的核心假设**: 帧间语义特征差异确实能捕捉到关键的视觉变化 (动作转换、物体交互)。
