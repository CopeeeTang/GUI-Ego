# ProAssist Validation - Round 2

> 日期: 2026-02-25
> 会话时长: 约 15 轮对话
> 主要方向: 实验结果梳理 + 两个新探索方向（FDA-CLIP 变化检测、推理触发）启动

## 会话目标

1. 从 context overflow 恢复，继续未完成的阈值扫描
2. 对已有实验结果做完整梳理和解读
3. 启动两个新探索方向的并行 Agent

## 行动路线

### 1. 恢复中断任务 + ego4d 扫描

**Prompt**: "继续完成没有完成的任务同时启动中断的子agent，完成任务后进行汇总"

**执行过程**:
- 发现 WTAG 精细扫描已在上一 session 完成（τ=0.15-0.45，6个点+之前的0.3,0.5 = 8点曲线）
- τ=0.5 结果已出：F1=0.3283
- 停止了误重跑 WTAG 的 agent，直接启动 ego4d 扫描
- ego4d narration_val_L4096_I1, 65 样本, 3 阈值 [0.2, 0.3, 0.4]
- 设置 ego4d 数据 symlink, 解压 prepared.zip + generated_dialogs.zip
- 分析脚本 agent 完成: `comprehensive_analysis.py` (940行)

**结果**:
- WTAG 确认最优: **τ*=0.20, F1=0.3727** (+11.6% vs 论文)
- ego4d τ=0.2 完成: **F1=0.2664**, P=0.739, R=0.163, CIDEr=1.11
- ego4d τ=0.3 进行中 (20/65)
- 生成 3 张分析图表 + 综合报告

### 2. Key Findings 深度解读

**Prompt**: "为我解释一下key findings以及梳理一遍目前开始到现在的实验进展"

**核心梳理**:

| Finding | 结论 | 关键证据 |
|---------|------|---------|
| 变化检测 | REJECTED | SigLIP CLS separation < 0.001, r=0.017 p=0.81 |
| w2t_prob 质量 | 极好，双峰 | 86% > 0.99, τ=0.3 完美二分类 |
| Adaptive-D (WTAG) | +11.6% F1 | τ*=0.20 vs 论文 τ=0.4 |
| Adaptive-D (ego4d) | τ=0.2 F1=0.2664 | 接近论文 τ=0.3 F1=0.275 |

### 3. 信号对比讨论

**Prompt**: "使用的信号是w2T的信号？已经不是我们原来的p(yes)-p(no)的信号了？使用的模型是什么模型？w2t也是training free吗"

**澄清关键区别**:

| 维度 | ESTP-Bench (我们) | ProAssist (验证) |
|------|---|---|
| 信号 | logprob gap (VLM next-token) | w2t_prob (trained binary head) |
| 模型 | Qwen3-VL-8B (通用) | LLaMA-3.1-8B+LoRA+SigLIP (专用) |
| 训练 | 完全 training-free | head 训练过，但阈值选择 training-free |
| 阈值 | per-task-type τ | per-dataset τ |

**核心论点**: Adaptive-D 的 training-free 指的是**阈值搜索过程**，不是信号本身。不管底层信号是什么，per-dataset 阈值优化都比固定阈值好。

### 4. 两个新探索方向启动

**Prompt**: "可以启动一个agent teams对这两个方向做一个探索"

**用户指出**:
- 推理触发方法在 ProAssist 上还没验证
- 变化检测应该用 FDA-CLIP（局部注意力引导）方向而非简单全局 CLS

**并行 Agent 分发**:

| Agent | 方向 | 资源 | 方法 |
|-------|------|------|------|
| fda-clip-change-detection | 局部变化检测 | SigLIP encoder (~2GB GPU) | 3种方法对比: CLS/Patch-Max/FDA-Style |
| reasoning-trigger-proassist | 推理触发 | GPT-4o API (无GPU) | 帧级推理 → needs_help 判断 |

## 关键决策

| 决策 | 选择 | 原因 |
|------|------|------|
| ego4d 评估格式 | narration_val (非 dialog-klg-sum) | 论文 baseline 用 narration，确保可比 |
| 停止重跑 WTAG agent | 直接跳到 ego4d | WTAG 精细扫描已在上一 session 完成 |
| FDA-CLIP 3种方法对比 | CLS / Patch-Max / FDA-Style | 梯度式验证"局部>全局"假设 |
| 推理触发用 GPT-4o API | 不占 GPU | ego4d 扫描正在跑，API 无资源冲突 |

## 关键数据

### WTAG 完整 F1 曲线（8 点）
```
τ=0.15  F1=0.3520  |  τ=0.35  F1=0.3562 (次峰)
τ=0.20  F1=0.3727  |  τ=0.40  F1=0.3310 (论文默认)
τ=0.25  F1=0.3486  |  τ=0.45  F1=0.3212
τ=0.30  F1=0.3454  |  τ=0.50  F1=0.3283
```

### ego4d 已完成结果
- τ=0.2: F1=0.2664, P=0.739, R=0.163 (论文同τ: F1=0.122)
- τ=0.3: 进行中 20/65

### 图表输出
- `results/figures/f1_vs_threshold_all.png` — F1/P/R 曲线 + 论文 baseline
- `results/figures/precision_recall_curve.png` — P-R 曲线（我们在论文右上方）
- `results/figures/error_rate_decomposition.png` — Missing/Redundant 交叉点 @ τ=0.2

## 当前状态

- [x] WTAG 完整扫描 — τ*=0.20, F1=0.3727
- [x] 变化检测 REJECTED + 深度解读
- [x] 综合分析脚本 + 报告 + 图表
- [x] ego4d τ=0.2 — F1=0.2664
- [ ] ego4d τ=0.3 — 后台运行 (20/65, ~3h)
- [ ] ego4d τ=0.4 — 排队中
- [ ] **Agent: FDA-CLIP 局部变化检测** — 后台运行中
- [ ] **Agent: 推理触发 GPT-4o** — 后台运行中
- [ ] 汇总分析报告 — 等所有任务完成

## 下一步

- [ ] 等两个探索 Agent 返回结果
- [ ] 等 ego4d 扫描完成 → 运行 `comprehensive_analysis.py` 更新报告
- [ ] 如果 FDA-CLIP patch-level 有效 → 设计与 w2t_prob 的融合方案
- [ ] 如果推理触发在 WTAG 有效 → 分析与 Adaptive-D 的互补性
- [ ] 汇总所有方向的最终结论

## 关键文件

| 文件 | 说明 |
|------|------|
| `proassist_experiments/comprehensive_analysis.py` | 940行综合分析脚本 |
| `proassist_experiments/results/comprehensive_report.md` | 分析报告（含图表） |
| `proassist_experiments/results/interim_findings.md` | 详细实验发现 |
| `proassist_experiments/fda_change_detection.py` | (Agent 创建中) FDA-CLIP 探索 |
| `proassist_experiments/reasoning_trigger_proassist.py` | (Agent 创建中) 推理触发验证 |
| `docs/survey/2026-02-24-ProAssist-Literature-Survey.md` | 文献调研（280行） |
| `docs/history/proassist-validation/round_1.md` | 上一轮会话摘要 |
