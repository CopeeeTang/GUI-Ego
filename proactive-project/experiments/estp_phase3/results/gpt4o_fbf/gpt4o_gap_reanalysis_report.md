# GPT-4o FBF Pilot — Logprob Gap 重分析报告

## 1. 问题概述

原始报告中 gap 统计量全部为 NaN，原因是 `np.mean()` 遇到 NaN/inf 值后传播。
本报告过滤无效 gap 值（NaN/inf）后重新计算所有统计量。

## 2. 根因分析

| Case | Task Type | 总步数 | 有效 | NaN | Inf | 原因 |
|------|-----------|--------|------|-----|-----|------|
| 0 | Information Function | 52 | 52 | 0 | 0 | 无 |
| 1 | Text-Rich Understanding | 52 | 51 | 0 | 1 | step 18: yes 不在 top_logprobs → gap=-inf |
| 2 | Object Function | 52 | 51 | 1 | 0 | step 0: `ERROR:rate_limit` → yes/no 都为 -inf → gap=NaN |
| 3 | Action Reasoning | 12 | 12 | 0 | 0 | 无 |
| 4 | Task Understanding | 60 | 59 | 0 | 1 | step 2: no 不在 top_logprobs → gap=+inf |

**核心问题**：原始代码 `all_gaps.append(gap)` 未过滤 NaN/inf，导致 `np.mean()` 产生 NaN 传播。

**具体原因**：
1. **API rate limit 错误**（Case 2 step 0）：`top_logprobs=[]`，yes_logp 和 no_logp 都设为 `-inf`，`(-inf) - (-inf) = NaN`
2. **yes/no token 不在 top_logprobs 中**（Case 1 step 18, Case 4 step 2）：GPT-4o 有时只返回 5 个 top_logprobs，当模型极度确信某一方时，另一方的 token 可能不在 top-5 中，此时 logprob 被设为 `-inf`，gap = ±inf

## 3. 修正后的整体 Gap 统计

- 总 polling 步数: 228
- 排除无效步数: 3 (1.3%)
- 有效步数: 225 (98.7%)

### 全部有效 gap (n=225)
- mean = **5.228**
- std = 7.853
- median = 5.750
- min = -17.250, max = 19.250

### GT 窗口内 gap (n=110)
- mean = **8.705**
- std = 7.629

### 非 GT 窗口 gap (n=115)
- mean = **1.902**
- std = 6.510

### Gap 分离度
- Separation (GT - non-GT mean) = **+6.802**
- Welch t-test: t = 7.148, p = 0.0000
- **Gap 具有显著判别力** (p < 0.05)

**对比 Qwen3-VL**：
- Qwen3-VL: GT_mean=2.226, non-GT_mean=0.808, separation=+1.418
- GPT-4o (修正后): GT_mean=8.705, non-GT_mean=1.902, separation=+6.802

## 4. 修正后的逐 Case 分析

| Case | Task Type | Steps | Valid | Yes% | Gap Mean | GT Gap | Non-GT Gap | Separation |
|------|-----------|-------|-------|------|----------|--------|------------|------------|
| 0 | Information Function | 52 | 52 | 67.3% | 2.803 | 10.179 | 1.656 | 8.523 |
| 1 | Text-Rich Understanding | 52 | 51 | 67.3% | 1.623 | 4.250 | -6.058 | 10.308 |
| 2 | Object Function | 52 | 51 | 73.1% | 2.623 | 4.737 | 1.367 | 3.370 |
| 3 | Action Reasoning | 12 | 12 | 100.0% | 4.104 | N/A | 4.104 | N/A |
| 4 | Task Understanding | 60 | 59 | 95.0% | 12.962 | 13.799 | 10.000 | 3.799 |

## 5. 逐 Case 详细分析

### Case 0: Information Function
- 原始 gap_mean: `2.8028846257033355`
- 修正后 gap_mean: **2.803**
- 无无效数据点
- GT gap mean: 10.179 (n=7)
- non-GT gap mean: 1.656 (n=45)
- Separation: +8.523

### Case 1: Text-Rich Understanding
- 原始 gap_mean: `1.6225490148271542`
- 修正后 gap_mean: **1.623**
- 排除步数: 0 NaN + 1 inf
  - step 18: gap=-inf, yes_logp=-inf, no_logp=-1.9361264946837764e-07, top_tokens=['No', 'The', 'no', ' No', '**']
- GT gap mean: 4.250 (n=38)
- non-GT gap mean: -6.058 (n=13)
- Separation: +10.308

### Case 2: Object Function
- 原始 gap_mean: `nan`
- 修正后 gap_mean: **2.623**
- 排除步数: 1 NaN + 0 inf
  - step 0: response=`ERROR:rate_limit`, yes_logp=-inf, no_logp=-inf
- GT gap mean: 4.737 (n=19)
- non-GT gap mean: 1.367 (n=32)
- Separation: +3.370

### Case 3: Action Reasoning
- 原始 gap_mean: `4.104166708537377`
- 修正后 gap_mean: **4.104**
- 无无效数据点

### Case 4: Task Understanding
- 原始 gap_mean: `12.961864312245718`
- 修正后 gap_mean: **12.962**
- 排除步数: 0 NaN + 1 inf
  - step 2: gap=+inf, yes_logp=-0.000000, no_logp=-inf, top_tokens=['Yes', '**', 'The', 'YES', ' Yes']
- GT gap mean: 13.799 (n=46)
- non-GT gap mean: 10.000 (n=13)
- Separation: +3.799

## 6. 修正后的 Threshold Coverage

| Threshold | Coverage | GT Coverage | Non-GT Coverage |
|-----------|----------|-------------|-----------------|
| gap > 0.0 | 77.8% | 89.1% | 67.0% |
| gap > 0.5 | 75.6% | 86.4% | 65.2% |
| gap > 1.0 | 73.3% | 86.4% | 60.9% |
| gap > 2.0 | 69.3% | 85.5% | 53.9% |
| gap > 3.0 | 64.4% | 82.7% | 47.0% |
| gap > 5.0 | 53.3% | 77.3% | 30.4% |
| gap > 8.0 | 36.0% | 60.0% | 13.0% |
| gap > 10.0 | 27.6% | 49.1% | 7.0% |

## 7. 关键发现

### 7.1 NaN 的影响很小
- 228 步中仅 3 步异常（1.3%），不影响结论可靠性
- 异常全部由已知原因引起（rate limit 错误 / token 不在 top-5 中）

### 7.2 修正后 GPT-4o 的 gap 判别力
- GPT-4o separation = **+6.802**（正值 = GT 窗口内 gap 更大 = 判别方向正确）
- Qwen3-VL separation = +1.418
- GPT-4o / Qwen3-VL 比值: 4.80x

### 7.3 修复建议
在 `gpt4o_fbf_pilot.py` 的 `generate_report()` 中添加 NaN/inf 过滤：
```python
gap = entry['logprob_gap']
if math.isfinite(gap):  # 过滤 NaN 和 inf
    all_gaps.append(gap)
    # ... GT/non-GT 分类
```

以及在 per-case gap_mean 计算中：
```python
gaps = [e['logprob_gap'] for e in trigger_log if math.isfinite(e['logprob_gap'])]
gap_mean = np.mean(gaps) if gaps else float('nan')
```
