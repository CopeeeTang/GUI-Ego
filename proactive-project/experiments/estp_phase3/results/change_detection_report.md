# Change Detection Trigger Pilot 分析报告

> 生成时间: 2026-02-21
> 实验脚本: `change_detection_trigger_pilot.py`
> 模型: Gemini-3-Flash-Preview
> 状态: **15/15 cases 完成**（实验已重启并完成全部 cases）

## ⚠️ 更新说明

本报告最初基于 8/15 cases 生成。实验进程因 Gemini API 调用挂起被中断，修复超时保护后已重启完成全部 15 cases。

**完整 15-case 最终结果**：
- **CD 整体 F1 = 0.121**（低于 RT 0.173 和 Adaptive-D 0.222）
- **Yes-rate = 21.0%**（大幅低于 FBF 76-83%，有效抑制 yes-bias）
- 完整结果: `results/phase3_learned/change_detection_trigger_pilot.json`
- 详细分析: `results/phase3_round2_summary.md` 第 2.5 节

以下为原始 8-case 分析（保留供参考）：

---

## 1. 实验概述

### 1.1 核心假设

传统的 Reasoning Trigger (RT) 方法在每个时间步问 VLM "现在需要帮助吗？"——这是一个**抽象的时机判断**问题，导致严重的 yes-bias。

Change Detection (CD) 方法改为问 VLM "这两帧之间有什么显著变化？"——这是一个**具体的视觉比较**问题，理论上更适合 VLM 的能力。

### 1.2 实验参数

| 参数 | 值 |
|------|-----|
| 轮询间隔 (POLL_INTERVAL) | 10s |
| 帧间隔 (DELTA_SECONDS) | 10s |
| 冷却期 (COOLDOWN) | 12s |
| 预期窗口 (ANTICIPATION) | 1.0s |
| 延迟容差 (LATENCY) | 2.0s |

### 1.3 Prompt 设计

CD 使用两帧对比 prompt，要求 VLM 识别两帧之间与用户问题相关的**具体、明确的变化**（新物体出现、文字可见、状态变化、动作完成），仅在检测到相关变化时回答 YES。

---

## 2. 逐 Case 详细结果

### 2.1 Change Detection (CD) 结果

| Case ID | 任务类型 | F1 | Precision | Recall | FP | Triggers | Yes数 | 基线F1 | Delta F1 |
|---------|----------|-----|-----------|--------|-----|----------|-------|--------|----------|
| f7328dc0ae72 | Object Recognition | 0.000 | 0.000 | 0.000 | 4 | 4 | 4 | 0.000 | +0.000 |
| 8563cf568b39 | Object State Change Recognition | 0.000 | 0.000 | 0.000 | 2 | 2 | 2 | 0.250 | -0.250 |
| 1b83f0628411 | Ego Object State Change Recognition | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | +0.000 |
| ec8c429db0d8 | Text-Rich Understanding | 0.500 | 0.500 | 0.500 | 1 | 2 | 2 | 0.000 | **+0.500** |
| 0eb97247e799 | Object Recognition | 0.200 | 0.500 | 0.125 | 1 | 2 | 2 | 0.077 | +0.123 |
| 022907170077 | Object Localization | 0.000 | 0.000 | 0.000 | 3 | 3 | 3 | 0.095 | -0.095 |
| eb1a417d26c9 | Object Recognition | 0.429 | 0.300 | 0.750 | 7 | 10 | 16 | 0.100 | **+0.329** |
| 85a99403e624 | Object State Change Recognition | 0.000 | 0.000 | 0.000 | 1 | 1 | 1 | 0.118 | -0.118 |

**平均 CD F1: 0.141** | 总 FP: 19 | 总 Triggers: 24 | 总 Yes: 30

### 2.2 与 Reasoning Trigger (RT) 逐 Case 对比

| Case ID | 任务类型 | RT F1 | CD F1 | Delta | RT FP | CD FP | RT Yes | CD Yes |
|---------|----------|-------|-------|-------|-------|-------|--------|--------|
| f7328dc0ae72 | Object Recognition | 0.167 | 0.000 | **-0.167** | 8 | 4 | 14 | 4 |
| 8563cf568b39 | Obj State Change Recog | 0.000 | 0.000 | 0.000 | 2 | 2 | 2 | 2 |
| 1b83f0628411 | Ego Obj State Change | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| ec8c429db0d8 | Text-Rich Understanding | 0.667 | 0.500 | **-0.167** | 0 | 1 | 1 | 2 |
| 0eb97247e799 | Object Recognition | 0.250 | 0.200 | -0.050 | 13 | 1 | 31 | 2 |
| 022907170077 | Object Localization | 0.000 | 0.000 | 0.000 | 1 | 3 | 1 | 3 |
| eb1a417d26c9 | Object Recognition | 0.000 | 0.429 | **+0.429** | 7 | 7 | 9 | 16 |
| 85a99403e624 | Obj State Change Recog | 0.500 | 0.000 | **-0.500** | 0 | 1 | 1 | 1 |

### 匹配 8 Cases 汇总

| 方法 | 平均 F1 | 总 FP | 总 Yes | Yes Rate |
|------|---------|-------|--------|----------|
| **Reasoning Trigger (RT)** | **0.198** | 31 | 68 | -- |
| **Change Detection (CD)** | **0.141** | 19 | 30 | -- |
| 基线 (Periodic) | 0.080 | -- | -- | -- |

**Delta (CD - RT) = -0.057**，CD 在匹配样本上劣于 RT。

---

## 3. 按任务类型分组分析

| 任务类型 | n | 基线 F1 | RT F1 | CD F1 | CD-RT | RT FP | CD FP | RT Yes | CD Yes |
|----------|---|---------|-------|-------|-------|-------|-------|--------|--------|
| Object Recognition | 3 | 0.059 | 0.139 | **0.210** | **+0.071** | 28 | 12 | 54 | 22 |
| Text-Rich Understanding | 1 | 0.000 | 0.667 | 0.500 | -0.167 | 0 | 1 | 1 | 2 |
| Object State Change Recog | 2 | 0.184 | 0.250 | 0.000 | **-0.250** | 2 | 3 | 3 | 3 |
| Ego Obj State Change Recog | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| Object Localization | 1 | 0.095 | 0.000 | 0.000 | 0.000 | 1 | 3 | 1 | 3 |

### 关键发现

**CD 优于 RT 的类型：**
- **Object Recognition (+0.071)**: CD 大幅降低 FP (28->12) 和 yes 数 (54->22)，同时维持甚至提升 F1。特别是 case eb1a417d26c9 从 F1=0.000 提升到 F1=0.429，说明变化检测能有效识别新物体出现。

**CD 劣于 RT 的类型：**
- **Object State Change Recognition (-0.250)**: CD 完全失效（F1=0.000），而 RT 在 case 85a99403e624 上表现良好（F1=0.500）。状态变化是渐进的，两帧对比难以捕捉。
- **Text-Rich Understanding (-0.167)**: RT 更精确（P=1.0, 只触发1次），CD 多了一次误触发。

---

## 4. 与 Adaptive-D 基线对比

Phase II Adaptive-D 在 53 cases 上的结果（来自 `phase2_report.txt`）：

| 方法 | 样本量 | 平均 F1 | FP 削减 | 核心机制 |
|------|--------|---------|---------|----------|
| Periodic Baseline | 53 | 0.141 | -- | 固定间隔触发 |
| Adaptive-D | 53 | 0.222 | -72% | Logprob gap 阈值 + per-type tau |
| RT (15 cases) | 15 | 0.173 | -- | VLM 推理触发 |
| **CD (8 cases)** | **8** | **0.141** | -- | VLM 两帧变化检测 |

CD 在其完成的 8 cases 上平均 F1 = 0.141，与 Periodic Baseline 持平，**远低于 Adaptive-D 的 0.222**。

CD 在匹配的 8 cases 上也低于 RT（0.141 vs 0.198），因此 CD < RT < Adaptive-D。

---

## 5. Yes-Bias 分析

| 方法 | 总 Yes | 总 Buckets (8 cases) | Yes Rate |
|------|--------|---------------------|----------|
| RT | 68 | ~254 | ~26.8% |
| CD | 30 | ~254 | ~11.8% |

CD 的 yes rate 约为 RT 的 44%，**确实大幅减少了 yes-bias**。然而，这种保守性同时也损害了 recall——在需要触发的时候也倾向于说 NO。

唯一例外是 case eb1a417d26c9（Object Recognition），CD 的 yes 数（16）反而高于 RT（9），但恰好在这个 case 上 CD 表现更好（F1=0.429 vs 0.000），说明在某些场景下 CD 的触发更准确。

---

## 6. 优劣势总结

### 优势

1. **FP 大幅降低**: 总 FP 从 31 (RT) 降至 19 (CD)，减少 39%
2. **Yes-bias 显著改善**: Yes rate 从 ~27% 降至 ~12%，说明两帧对比 prompt 确实比单帧推理更保守
3. **Object Recognition 效果好**: 在物体识别任务上 CD 优于 RT（+0.071 F1），特别是新物体出现的场景
4. **精确度更高的触发**: case 0eb97247e799 FP 从 13 降至 1，yes 从 31 降至 2

### 劣势

1. **整体 F1 低于 RT**: 匹配 8 cases 平均 CD F1=0.141 vs RT F1=0.198 (delta=-0.057)
2. **状态变化识别完全失效**: Object State Change Recognition 上 CD F1=0.000（RT=0.250），渐进变化在两帧间不明显
3. **过度保守**: 减少 yes-bias 的代价是 recall 也下降了
4. **远低于 Adaptive-D**: CD F1=0.141 vs Adaptive-D F1=0.222
5. **实验未完成**: 进程在 8/15 cases 后停止，可能是 GPU 内存或 API 限制导致

### 根因分析

CD 方法的核心问题在于**变化检测 != 事件检测**：
- 视频中许多相关事件是**渐进发生**的（如状态变化、动作进行），10 秒间隔的两帧对比可能看不出差异
- **突变事件**（如新物体出现、文字显示）适合 CD，但这只是部分任务类型
- RT 虽然有 yes-bias，但至少能在相关场景中触发；CD 太保守导致错过许多真正需要触发的时刻

---

## 7. 结论与建议

### 结论

Change Detection Trigger **未通过评估**。虽然成功减少了 yes-bias 和 FP，但整体 F1 低于 RT 和 Adaptive-D，不适合作为独立的触发方法。

### 方法排名（当前证据）

1. **Adaptive-D** (F1=0.222, 53 cases) -- 最优
2. **Reasoning Trigger** (F1=0.173, 15 cases) -- 次优
3. **Change Detection** (F1=0.141, 8 cases) -- 与基线持平
4. **Periodic Baseline** (F1=0.141, 53 cases) -- 基线

### 可能的改进方向

1. **混合策略**: 对 Object Recognition 使用 CD，对其他类型使用 RT 或 Adaptive-D
2. **调整 DELTA_SECONDS**: 当前 10s 可能太短或太长，不同任务类型可能需要不同的帧间隔
3. **多尺度变化检测**: 同时比较 5s、10s、30s 前的帧，捕捉不同速度的变化
4. **CD + Adaptive-D 联合**: 用 CD 作为 Adaptive-D 的补充信号，而非替代

---

## 附录: 原始数据

### A. CD checkpoint 完整数据

```json
[
  {"case_id": "f7328dc0ae72", "task_type": "Object Recognition", "f1": 0.000, "precision": 0.000, "recall": 0.000, "fp": 4, "tp": 0, "fn": 3, "n_triggers": 4, "n_yes": 4, "baseline_f1": 0.000},
  {"case_id": "8563cf568b39", "task_type": "Object State Change Recognition", "f1": 0.000, "precision": 0.000, "recall": 0.000, "fp": 2, "tp": 0, "fn": 3, "n_triggers": 2, "n_yes": 2, "baseline_f1": 0.250},
  {"case_id": "1b83f0628411", "task_type": "Ego Object State Change Recognition", "f1": 0.000, "precision": 0.000, "recall": 0.000, "fp": 0, "tp": 0, "fn": 1, "n_triggers": 0, "n_yes": 0, "baseline_f1": 0.000},
  {"case_id": "ec8c429db0d8", "task_type": "Text-Rich Understanding", "f1": 0.500, "precision": 0.500, "recall": 0.500, "fp": 1, "tp": 1, "fn": 1, "n_triggers": 2, "n_yes": 2, "baseline_f1": 0.000},
  {"case_id": "0eb97247e799", "task_type": "Object Recognition", "f1": 0.200, "precision": 0.500, "recall": 0.125, "fp": 1, "tp": 1, "fn": 7, "n_triggers": 2, "n_yes": 2, "baseline_f1": 0.077},
  {"case_id": "022907170077", "task_type": "Object Localization", "f1": 0.000, "precision": 0.000, "recall": 0.000, "fp": 3, "tp": 0, "fn": 3, "n_triggers": 3, "n_yes": 3, "baseline_f1": 0.095},
  {"case_id": "eb1a417d26c9", "task_type": "Object Recognition", "f1": 0.429, "precision": 0.300, "recall": 0.750, "fp": 7, "tp": 3, "fn": 1, "n_triggers": 10, "n_yes": 16, "baseline_f1": 0.100},
  {"case_id": "85a99403e624", "task_type": "Object State Change Recognition", "f1": 0.000, "precision": 0.000, "recall": 0.000, "fp": 1, "tp": 0, "fn": 3, "n_triggers": 1, "n_yes": 1, "baseline_f1": 0.118}
]
```

### B. RT (15 cases) 汇总

- 平均 F1: 0.173
- 平均基线 F1: 0.065
- 平均 Delta F1: +0.066
- 总 Yes Rate: 6.3%
