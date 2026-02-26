# GPT-4o Adaptive-D 离线分析报告

生成时间: 2026-02-22 04:00
数据源: GPT-4o FBF 全量 checkpoint (60 cases)
任务类型: 12 种

## 核心结论

| 指标 | GPT-4o | Qwen | 对比 |
|------|--------|------|------|
| Vanilla FBF F1 | 0.279 | 0.127 | GPT-4o 更好 |
| Adaptive-D F1 | 0.371 | 0.222 | GPT-4o 更好 |
| Delta (vs BL) | +0.082 | +0.081 | |
| 95% CI | [0.032, +0.136] | [0.046, +0.121] | |
| Gate | PASS | PASS | |
| Gap Separation | +8.527 | +1.418 | GPT-4o 更好 |
| AUC | 0.761 | 0.615 | GPT-4o 更好 |
| Yes-Rate | 53.5% | 76-83% | |

## A. Vanilla FBF F1

触发规则: response 中包含 'yes' 即触发
整体平均 F1: **0.279**
Yes-rate: 53.5%

| 任务类型 | F1 |
|----------|-----|
| Task Understanding | 0.712 |
| Action Reasoning | 0.484 |
| Object Recognition | 0.428 |
| Object State Change Recognition | 0.341 |
| Information Function | 0.253 |
| Text-Rich Understanding | 0.205 |
| Object Function | 0.194 |
| Ego Object Localization | 0.192 |
| Action Recognition | 0.175 |
| Attribute Perception | 0.164 |
| Ego Object State Change Recognition | 0.135 |
| Object Localization | 0.065 |

## B. 全局阈值扫描

全局最优: tau=3, cooldown=12s, F1=0.329

### 详细扫描结果 (cooldown=12s)

| tau | F1 | Precision | Recall | Total FP | Trig% |
|-----|-----|-----------|--------|----------|-------|
| -5 | 0.244 | 0.226 | 0.421 | 555 | 25.9% |
| -3 | 0.254 | 0.231 | 0.426 | 505 | 24.3% |
| -2 | 0.272 | 0.252 | 0.452 | 476 | 23.3% |
| -1 | 0.281 | 0.265 | 0.443 | 452 | 22.4% |
| 0 | 0.291 | 0.280 | 0.426 | 426 | 21.4% |
| 0.5 | 0.291 | 0.280 | 0.428 | 415 | 21.0% |
| 1 | 0.295 | 0.285 | 0.423 | 403 | 20.6% |
| 2 | 0.313 | 0.306 | 0.438 | 362 | 19.0% |
| 3 | 0.329 ** | 0.323 | 0.458 | 331 | 17.8% |
| 5 | 0.279 | 0.289 | 0.362 | 282 | 15.5% |
| 8 | 0.242 | 0.289 | 0.311 | 179 | 10.9% |
| 10 | 0.182 | 0.279 | 0.195 | 140 | 8.3% |
| 15 | 0.031 | 0.047 | 0.043 | 52 | 2.2% |

## C. Per-type 自适应阈值 (Adaptive-D)

Adaptive-D 平均 F1: **0.371**
Baseline 平均 F1: 0.289
Delta: +0.082, 95% CI=[0.032, +0.136]
Gate: **PASS**

| 任务类型 | n | GPT-4o opt-tau | GPT-4o F1 | BL F1 | Delta | Qwen opt-tau |
|----------|---|---------------|-----------|-------|-------|-------------|
| Task Understanding | 5 | -5 | 0.690 | 0.681 | +0.008 | 0.0 |
| Object Recognition | 5 | 2.0 | 0.563 | 0.506 | +0.057 | 2.5 |
| Object State Change Recognition | 5 | 3.5 | 0.540 | 0.369 | +0.171 | n/a |
| Action Reasoning | 5 | 3.0 | 0.508 | 0.502 | +0.006 | n/a |
| Object Function | 5 | 4.0 | 0.368 | 0.273 | +0.095 | n/a |
| Text-Rich Understanding | 5 | 0 | 0.356 | 0.356 | +0.000 | 2.0 |
| Information Function | 5 | 8.0 | 0.344 | 0.244 | +0.100 | 2.8 |
| Ego Object State Change Recognition | 5 | -5 | 0.300 | 0.033 | +0.267 | n/a |
| Ego Object Localization | 5 | 3.5 | 0.295 | 0.190 | +0.105 | n/a |
| Attribute Perception | 5 | 2.8 | 0.193 | 0.091 | +0.102 | 0.5 |
| Action Recognition | 5 | 0.5 | 0.163 | 0.161 | +0.002 | 0.5 |
| Object Localization | 5 | 8.0 | 0.128 | 0.090 | +0.038 | n/a |

### Per-type Adaptive 平均 Delta

| 任务类型 | 平均 Delta |
|----------|-----------|
| Ego Object State Change Recognition | +0.233 |
| Object State Change Recognition | +0.178 |
| Object Function | +0.161 |
| Text-Rich Understanding | +0.099 |
| Ego Object Localization | +0.094 |
| Information Function | +0.093 |
| Object Recognition | +0.067 |
| Object Localization | +0.067 |
| Attribute Perception | +0.027 |
| Action Reasoning | +0.024 |
| Task Understanding | -0.023 |
| Action Recognition | -0.041 |

Cases improved: 28/60 (47%)
Cases degraded: 18/60 (30%)
FP reduction: 1058 -> 306 (71%)

## D. Gap 判别力分析

| 指标 | GPT-4o | Qwen |
|------|--------|------|
| GT mean gap | +6.743 | +2.226 |
| Non-GT mean gap | -1.784 | +0.808 |
| Separation | +8.527 | +1.418 |
| t-stat | 26.072 | n/a |
| p-value | 0.000000 | <0.0001 |
| AUC | 0.761 | 0.615 |
| n_GT steps | 621 | n/a |
| n_non-GT steps | 2388 | n/a |

### Per-type Gap 判别力

| 任务类型 | GT_mean | nonGT_mean | Separation | AUC | 判定 |
|----------|---------|------------|------------|-----|------|
| Action Reasoning | +5.883 | +3.018 | +2.865 | 0.748 | DISCRIMINATIVE |
| Action Recognition | +2.913 | +1.450 | +1.463 | 0.573 | DISCRIMINATIVE |
| Attribute Perception | +10.683 | +4.661 | +6.021 | 0.690 | DISCRIMINATIVE |
| Ego Object Localization | -2.269 | -2.401 | +0.132 | 0.500 | weak |
| Ego Object State Change Recognition | -3.854 | -12.509 | +8.654 | 0.790 | DISCRIMINATIVE |
| Information Function | +3.016 | -3.355 | +6.370 | 0.725 | DISCRIMINATIVE |
| Object Function | +8.897 | +3.940 | +4.957 | 0.630 | DISCRIMINATIVE |
| Object Localization | +11.648 | +6.277 | +5.371 | 0.758 | DISCRIMINATIVE |
| Object Recognition | +4.194 | -2.739 | +6.933 | 0.719 | DISCRIMINATIVE |
| Object State Change Recognition | +1.917 | -4.589 | +6.506 | 0.829 | DISCRIMINATIVE |
| Task Understanding | +9.020 | +5.372 | +3.648 | 0.644 | DISCRIMINATIVE |
| Text-Rich Understanding | +6.618 | -10.640 | +17.258 | 0.881 | DISCRIMINATIVE |

## E. Bootstrap 置信区间

- Adaptive-D F1: 0.371 (95% CI: [0.291, 0.461])
- Baseline F1: 0.289
- Delta: +0.082 (95% CI: [0.032, +0.136])
- Gate: **PASS**
- Bootstrap iterations: 1000

## 最终结论

**GPT-4o Adaptive-D F1=0.371 > Qwen Adaptive-D F1=0.222**
GPT-4o + Adaptive-D 显著优于 Qwen。

Gap 判别力: GPT-4o (+8.527) > Qwen (+1.418)
GPT-4o logprob gap 更具判别力，Adaptive-D 有更大潜力。
AUC: GPT-4o (0.761) > Qwen (0.615)
