# Context-Enhanced Reasoning Trigger (CERT) - 综合对比报告

> 实验日期: 2026-02-22
> 模型: gemini-2.0-flash | 间隔: 10s | Cooldown: 12s | 5 cases

---

## 1. 总览

| 方法 | 描述 | Avg F1 | Total FP | Global Yes-Rate | 耗时(min) |
|------|------|--------|----------|-----------------|-----------|
| **Baseline** | 原始 RT (无上下文) | 0.243 | 30 | 38.2% | 3.3 |
| **Method A** | 判断历史注入 | **0.278** | 7 | 6.3% | 3.2 |
| **Method B** | 场景描述 + 判断历史 | 0.178 | 1 | 1.9% | 3.7 |
| **Method C** | 多帧对比 + 判断历史 | **0.292** | 11 | 10.1% | 3.8 |

**最佳方法: Method C (avg F1=0.292)**, 相比 Baseline 提升 +0.049 (+20.2%)。

---

## 2. 逐 Case 详细对比

### 2.1 ESTP-F1

| Case ID | Task Type | Baseline | A | B | C | 最佳 |
|---------|-----------|----------|---|---|---|------|
| 0eb97247e799 | Object Recognition | 0.250 | 0.222 | 0.222 | 0.154 | **BL** |
| ec8c429db0d8 | Text-Rich Understanding | 0.400 | **0.667** | **0.667** | 0.500 | **A/B** |
| 4a51f9ebaa22 | Object Function | 0.000 | 0.000 | 0.000 | **0.222** | **C** |
| 85a99403e624 | Object State Change | 0.400 | **0.500** | 0.000 | 0.400 | **A** |
| 2098c3e8d904 | Action Recognition | 0.167 | 0.000 | 0.000 | **0.182** | **C** |
| **平均** | | 0.243 | 0.278 | 0.178 | **0.292** | **C** |

### 2.2 Yes-Rate (越低越好, 在 F1 不降的前提下)

| Case ID | Task Type | Baseline | A | B | C |
|---------|-----------|----------|---|---|---|
| 0eb97247e799 | Object Recognition | **96.9%** | 3.1% | 3.1% | 15.6% |
| ec8c429db0d8 | Text-Rich Understanding | 9.4% | 3.1% | 3.1% | 6.2% |
| 4a51f9ebaa22 | Object Function | **59.4%** | 9.4% | 0.0% | 12.5% |
| 85a99403e624 | Object State Change | 9.4% | 3.1% | 0.0% | 6.2% |
| 2098c3e8d904 | Action Recognition | 16.1% | 12.9% | 3.2% | 9.7% |
| **全局** | | **38.2%** | **6.3%** | **1.9%** | **10.1%** |

### 2.3 FP (误触发次数)

| Case ID | Task Type | Baseline | A | B | C |
|---------|-----------|----------|---|---|---|
| 0eb97247e799 | Object Recognition | 13 | 0 | 0 | 4 |
| ec8c429db0d8 | Text-Rich Understanding | 2 | 0 | 0 | 1 |
| 4a51f9ebaa22 | Object Function | 11 | 3 | 0 | 3 |
| 85a99403e624 | Object State Change | 1 | 0 | 0 | 1 |
| 2098c3e8d904 | Action Recognition | 3 | 4 | 1 | 2 |
| **总计** | | **30** | **7** | **1** | **11** |

### 2.4 Precision / Recall 分解

| Case ID | | Baseline | | Method A | | Method B | | Method C | |
|---------|---|------|------|------|------|------|------|------|------|
| | P | R | P | R | P | R | P | R |
| 0eb97247e799 | 0.188 | 0.375 | **1.000** | 0.125 | **1.000** | 0.125 | 0.200 | 0.125 |
| ec8c429db0d8 | 0.333 | 0.500 | **1.000** | 0.500 | **1.000** | 0.500 | 0.500 | 0.500 |
| 4a51f9ebaa22 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.250** | **0.200** |
| 85a99403e624 | 0.500 | 0.333 | **1.000** | 0.333 | 0.000 | 0.000 | 0.500 | 0.333 |
| 2098c3e8d904 | 0.250 | 0.125 | 0.000 | 0.000 | 0.000 | 0.000 | **0.333** | **0.125** |

---

## 3. 关键发现

### 3.1 Yes-Bias 抑制效果

所有上下文增强方法都显著降低了 yes-rate:

| 方法 | Yes-Rate | vs Baseline 降幅 |
|------|----------|-----------------|
| Baseline | 38.2% | - |
| Method A | 6.3% | -83.5% |
| Method B | 1.9% | -95.0% |
| Method C | 10.1% | -73.6% |

- **Method B 过度抑制**: yes-rate 仅 1.9%, 导致 recall 崩溃, 85a99403e624 和 2098c3e8d904 完全没有 TP
- **Method A 精准抑制**: yes-rate 6.3%, 在 0eb97247e799 和 ec8c429db0d8 上实现了 100% precision
- **Method C 平衡抑制**: yes-rate 10.1%, 在保持触发能力的同时有效降低 FP

### 3.2 各方法优劣势

**Method A (判断历史注入):**
- 优势: FP 最低 (7), 简单有效, 0 额外图像成本
- 劣势: 对 Action Recognition 无效 (F1=0.000)
- 适合: 静态场景类任务 (Object Recognition, Text-Rich)

**Method B (场景描述 + 历史):**
- 优势: FP 极低 (1), 几乎不误触发
- 劣势: 严重过度抑制, recall 崩溃, 3/5 case F1=0
- 不适合: 作为独立方法使用 (需要调低抑制力度)

**Method C (多帧对比 + 历史):**
- 优势: **Avg F1 最高 (0.292)**, 唯一在 Object Function 和 Action Recognition 上有 TP 的增强方法
- 劣势: FP 较多 (11), 多帧增加 API 成本
- 适合: 需要时序变化感知的任务类型

### 3.3 任务类型分析

| 任务类型 | 最佳方法 | 理由 |
|----------|----------|------|
| Object Recognition | Baseline | 上下文增强反而降低了 recall (97%->3-16% yes-rate 跨度太大) |
| Text-Rich Understanding | A 或 B | 精准场景, 上下文历史帮助去重, F1=0.667 |
| Object Function | **C** | 需要看到场景变化来判断功能相关性, 仅 C 有 TP |
| Object State Change | **A** | 判断历史帮助避免重复触发, F1=0.500 |
| Action Recognition | **C** | 动作的时序特征在多帧对比中更容易被捕捉 |

---

## 4. 成功标准评估

| 标准 | 条件 | 结果 |
|------|------|------|
| **PRIMARY** | 至少一种方案 avg F1 > Baseline (0.243) | **PASS** - Method A (0.278) 和 Method C (0.292) 均超过 |
| **SECONDARY** | 高 yes-bias case 的 yes-rate 降低 >30% | **PASS** - 0eb9: 97%->3-16%, 4a51: 59%->0-13%, 2098: 16%->3-13% |
| **GUARDRAIL** | ec8c (Text-Rich) F1 不退化超过 0.1 | **PASS** - 最低 0.500 (A/B=0.667), 均 >= 0.3 |
| **BONUS** | 任一方案在所有 case 上均不差于 Baseline | **FAIL** - 所有方案在部分 case 上退化 |

---

## 5. 结论与建议

### 5.1 核心结论

1. **Method C (多帧对比 + 历史) 是综合最优方法**, avg F1=0.292, 比 Baseline 提升 20.2%
2. **Method A 是性价比最高的方法**, 无额外图像成本, avg F1=0.278, FP 仅 7 (vs Baseline 30)
3. **Method B 过度抑制**, 不建议单独使用, 但其核心思想可以被弱化后集成
4. 上下文增强方法是**任务类型相关**的, 没有单一方法在所有类型上都最优

### 5.2 后续建议

1. **自适应方法选择**: 根据任务类型自动选择 A 或 C
   - 静态语义任务 (Text-Rich, Object State Change) -> Method A
   - 时序变化任务 (Object Function, Action Recognition) -> Method C
2. **Method C + Adaptive-D 融合**: 将 Method C 与 logprob gap 过滤组合
   - Method C 从源头降低 yes-bias (38%->10%)
   - Adaptive-D 做第二层 FP 过滤
3. **扩展到 15 cases 全量评估**: PRIMARY 标准已通过, 可进入扩展实验

---

## 6. 原始数据参考

### Method C 触发详情

| Case | Trigger Times (abs) | TP | FP | FN |
|------|--------------------|----|----|----|
| 0eb97247e799 | 1644, 1724, 1774, 1834, 1904 | 1 | 4 | 7 |
| ec8c429db0d8 | 3080, 3230 | 1 | 1 | 1 |
| 4a51f9ebaa22 | 60.5, 150.5, 250.5, 310.5 | 1 | 3 | 4 |
| 85a99403e624 | 1150, 1200 | 1 | 1 | 2 |
| 2098c3e8d904 | 40.5, 160.5, 240.5 | 1 | 2 | 7 |
