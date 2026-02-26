# ProAssist Validation - Round 3

> 日期: 2026-02-25
> 会话时长: 约 12 轮对话（从 context overflow 续接）
> 主要方向: 探索 Agent 结果汇总 + 信号失败根因分析 + Agent 架构方向讨论

## 会话目标

1. 汇总两个并行探索 Agent（FDA-CLIP 变化检测、推理触发）的结果
2. 检查 ego4d 扫描进度，获取 τ=0.3 结果
3. 深入讨论：为什么所有信号都与 GT 反相关？应该是信号问题还是架构问题？
4. 讨论 Adaptive-D 的本质局限性
5. 将变化检测重新定位为 Memory 管理机制

## 行动路线

### 1. 两个探索 Agent 结果汇总

**Prompt**: （从上一 session 续接，两个后台 Agent 已完成）

**FDA-CLIP 变化检测结果**:
- 8 种方法全部 AUC < 0.50，所有 separation 为负
- CLS/Patch-Max/Patch-Mean/Patch-Top50/FDA(r=0.05~0.30) 无一有效
- **结论**: Talk 帧视觉变化**低于** NoTalk 帧 — 用户需要帮助时卡住了（低变化）

**推理触发结果**:
- GPT-4o on WTAG: F1=0.269, P=0.529, R=0.180
- 与 w2t_prob 相关性: r=0.031（近乎正交）
- ORACLE 组合: F1=0.561 (+7.5% over w2t_prob alone)
- w2t_prob 擅长 confirmation (66.7%), GPT-4o 擅长 correction (21.4%) 和 step transition

### 2. ego4d τ=0.3 结果 + 扫描终止

**结果**:
- ego4d τ=0.2: F1=0.2664, P=0.739, R=0.163
- **ego4d τ=0.3: F1=0.3056, P=0.222, R=0.489** (+11.1% vs 论文)
- ego4d τ=0.4: 5/65 后被用户要求停止

**跨数据集对比**: WTAG τ*=0.20, ego4d τ*=0.30 — 最优阈值因数据集而异

**阈值差异原因**: WTAG 有用户提问做强信号(w2t_prob 二值化), ego4d 纯视觉(w2t_prob 中间地带密集 [0.2,0.3])

### 3. 用户三个核心讨论点

**Prompt**: "1.关于change detection方向我认为确实不能作为proactive的研究方向，而应该作为memory保存的机制...2.你需要与我讨论一下目前推理的范式是什么样的？局限性在哪？3.我认为adaptive D实际上来说算不上一种方法..."

#### 讨论 1: Change Detection → Memory 管理机制

用户洞察: 变化检测的价值不在 proactive trigger，而在 streaming memory management。
- 低变化帧 → 跳过/稀疏采样（节省 KV-cache）
- 中变化帧 → 局部 patch token 更新（FDA-CLIP 风格）
- 高变化帧 → 全帧重编码 + 段落摘要

写入文档: `.claude/worktrees/memory/docs/survey/2026-02-25-change-detection-as-memory-mechanism.md`

#### 讨论 2: 推理范式及局限性

梳理了三种范式:
- A: 信号阈值法 (w2t_prob/logprob gap) — 快但无语义
- B: 推理触发法 (GPT-4o/Qwen3-VL) — 有语义但无时序、高成本
- C: 规则/启发式 — 方向性错误

**共同问题**: 都把 proactive timing 当 frame-level binary classification，缺少 Task Model/User Model/Dialogue Model

#### 讨论 3: Adaptive-D 的本质

用户批评: "不算真正在做 proactive research，只是取巧"
- 不提出新信号，本质是超参调优
- 不泛化到新场景（需要 GT labels 做搜索）
- "Training-free" 是误导（阈值搜索依赖 GT）
- 价值仅在于揭示"现有信号 calibration 差"这个现象

### 4. 深度分析: 为什么所有信号与 GT 反相关

**Prompt**: "我之前尝试了user state的方向在ESTP的benchmark，和change detection，以及task-aware state，但这几个信号与GT似乎都是反作用？"

**汇总 ESTP + ProAssist 全部信号实验**:

| 信号 | ESTP 相关性 | ProAssist 相关性 | 最佳 F1 |
|------|------------|-----------------|---------|
| Logprob Gap | 正 (sep=+2.608) | N/A | 0.222 |
| w2t_prob | N/A | 正 (双峰) | 0.373 |
| Goal-State | **反向** (sep=-0.636) | 未测 | 无效 |
| Change Detection | 弱正 | **反向** (AUC<0.50) | 0.121 |
| FDA-CLIP | 未测 | **反向** (8种全部) | 无效 |
| Reasoning Trigger | 仅 Text-Rich 正 | 弱正 (正交) | 0.173/0.269 |

**根因分析**:
- 外部信号检测 "用户是否遇困难" (need-based)
- GT 标注的是 "这个时刻提供信息最有效" (opportunity-based)
- 这是两个不同的问题 → 结构性 mismatch
- Goal-state on_track + GT says "speak" = 用户注意力在正确位置时正是教学机会
- 只有模型自身不确定性信号 (w2t_prob, logprob gap) 有效，因为它们编码了对话流模式

### 5. Agent 架构方向

**Prompt**: "我觉得方向B中的memory可以作为推理触发法的时序上下文，思考一个顶尖的科学家会如何规划"

提出 **Information Gap Agent** 架构:
```
Video Stream → Change-Driven Memory → Compressed Context
Current Frame → Visual Encoder → Reasoning Agent → Decision
Task Description → Task Model ──────→     ↑
Dialog History → Dialog State ──────→     ↑
```

触发条件 = f(信息缺口, 当前上下文, 任务状态, 时机合理性)

核心洞察: 从 Signal Engineering → Agent Architecture
- 不是找更好的单一信号，而是设计综合多源信息的决策 Agent
- Memory module 提供高效时序上下文（86% 帧可跳过）
- Reasoning module 提供语义理解（error detection, step transition）
- 信息缺口 = 当前相关信息 - 已传递信息

**用户最终判断**: "单纯一个信号是不太可能做到场景泛化性的" — 完全正确

## 关键决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 停止 ego4d τ=0.4 扫描 | 终止 PID 1091248 | τ=0.3 已足够验证, 用户认为不需要继续 |
| 变化检测重定位 | Proactive trigger → Memory mechanism | 实验证明对 trigger 反向，但对 memory 管理有价值 |
| Adaptive-D 定位 | 超参调优，非核心贡献 | 不提出新信号，不泛化，需要 GT |
| 研究方向 | 从信号工程 → Agent 架构 | 单信号不可能跨场景泛化 |

## 关键数据

### 最终跨数据集结果

| 数据集 | 我们最优 τ | 我们 F1 | 论文 F1 | 提升 |
|--------|-----------|---------|---------|------|
| WTAG | 0.20 | 0.3727 | 0.3340 | +11.6% |
| ego4d | 0.30 | 0.3056 | 0.2750 | +11.1% |

### 全信号失败模式总结

所有外部观察信号（goal-state, change detection, reasoning trigger）与 GT 反相关或弱相关，
只有模型自身不确定性信号（w2t_prob, logprob gap）有效。
根因: GT 标注 opportunity-based timing ≠ 信号检测 need-based distress。

## 当前状态

- [x] WTAG 完整扫描 — τ*=0.20, F1=0.3727
- [x] ego4d 部分扫描 — τ*=0.30, F1=0.3056
- [x] 变化检测 CLS — REJECTED
- [x] 变化检测 FDA-CLIP (8种) — REJECTED (全部 AUC<0.50)
- [x] 推理触发 GPT-4o — F1=0.269, 正交, ORACLE +7.5%
- [x] 跨数据集阈值差异分析 — w2t_prob 分布形态不同
- [x] 最终报告 — `proassist_experiments/results/final_report.md`
- [x] Memory 机制文档 — `.claude/worktrees/memory/docs/survey/2026-02-25-change-detection-as-memory-mechanism.md`
- [x] 全信号失败根因分析 + Agent 架构方向讨论

## 下一步（新方向）

- [ ] 在 memory worktree 中继续发展 Change-Driven Memory 机制
- [ ] 设计 Information Gap Agent 架构的具体实现方案
- [ ] 考虑: Memory module 如何与现有 Streaming VLM (ProAssist/VideoLLM-Online) 集成
- [ ] 考虑: 如何定义和量化 "information gap"
- [ ] 将 ProAssist 验证结论写入论文 Related Work

## 关键文件

| 文件 | 说明 |
|------|------|
| `proassist_experiments/results/final_report.md` | **最终报告（本 session 生成）** |
| `proassist_experiments/results/interim_findings.md` | 5 个关键发现（含 FDA-CLIP + 推理触发） |
| `proassist_experiments/results/fda_change_analysis.json` | FDA-CLIP 8 种方法数据 |
| `proassist_experiments/results/reasoning_trigger/reasoning_trigger_report.txt` | 推理触发详细报告 |
| `.claude/worktrees/memory/docs/survey/2026-02-25-change-detection-as-memory-mechanism.md` | **Memory 机制文档（本 session 生成）** |
| `docs/history/proassist-validation/round_1.md` | Session 1 历史 |
| `docs/history/proassist-validation/round_2.md` | Session 2 历史 |
