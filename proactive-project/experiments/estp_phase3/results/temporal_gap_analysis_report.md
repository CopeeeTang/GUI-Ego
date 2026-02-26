# Temporal Gap Feature Analysis Report

Date: 2026-02-20

Cases: 60 (from fullscale_d checkpoint)
Evaluation: ESTP-F1 (anticipation=1.0s, latency=2.0s)
Cooldown: 12s

## Summary

- **17 features** evaluated
- **Baseline (raw_gap)**: F1=0.252, AUC=0.615
- No feature significantly beats raw_gap

## Feature Ranking

| Rank | Feature | AUC | Best F1 | Macro F1 | Threshold | Separation |
|------|---------|-----|---------|----------|-----------|------------|
| 1 | raw_gap | 0.615 | 0.252 | 0.145 | 1.515 | 1.418 |
| 2 | gap_max_w3 | 0.599 | 0.246 | 0.188 | -0.504 | 1.240 |
| 3 | gap_max_w5 | 0.592 | 0.243 | 0.187 | -0.750 | 1.167 |
| 4 | gap_percentile | 0.525 | 0.238 | 0.196 | 0.720 | 0.021 |
| 5 | gap_mean_w5 | 0.595 | 0.233 | 0.139 | 0.672 | 1.183 |
| 6 | gap_jump | 0.529 | 0.231 | 0.191 | 0.210 | 0.198 |
| 7 | gap_zscore | 0.498 | 0.229 | 0.190 | 0.573 | 0.014 |
| 8 | gap_slope_w5 | 0.547 | 0.224 | 0.172 | -0.100 | 0.122 |
| 9 | gap_slope_w3 | 0.540 | 0.223 | 0.176 | 0.241 | 0.131 |
| 10 | gap_mean_w3 | 0.604 | 0.223 | 0.169 | -1.167 | 1.274 |
| 11 | gap_cusum | 0.489 | 0.222 | 0.183 | 0.812 | -4.647 |
| 12 | gap_x_slope | 0.523 | 0.208 | 0.152 | 0.379 | 0.389 |
| 13 | gap_jump_abs | 0.436 | 0.207 | 0.166 | 0.254 | -0.037 |
| 14 | gap_std_w5 | 0.409 | 0.203 | 0.164 | 0.000 | -0.004 |
| 15 | gap_acceleration | 0.510 | 0.200 | 0.158 | -0.041 | 0.039 |
| 16 | gap_x_jump | 0.514 | 0.200 | 0.161 | -0.014 | 0.492 |
| 17 | gap_std_w3 | 0.421 | 0.198 | 0.161 | 0.011 | -0.032 |

## Top-5 Feature Details

### #1: raw_gap
- AUC-ROC: 0.6149
- Best F1: 0.2521 (micro), 0.1452 (macro)
- Optimal threshold: 1.5151
- Separation: 1.4179
- At best threshold: P=0.187, R=0.387, TP=89, FP=387

### #2: gap_max_w3
- AUC-ROC: 0.5991
- Best F1: 0.2460 (micro), 0.1880 (macro)
- Optimal threshold: -0.5039
- Separation: 1.2402
- At best threshold: P=0.164, R=0.496, TP=114, FP=583

### #3: gap_max_w5
- AUC-ROC: 0.5922
- Best F1: 0.2426 (micro), 0.1867 (macro)
- Optimal threshold: -0.7500
- Separation: 1.1672
- At best threshold: P=0.161, R=0.496, TP=114, FP=596

### #4: gap_percentile
- AUC-ROC: 0.5254
- Best F1: 0.2378 (micro), 0.1960 (macro)
- Optimal threshold: 0.7200
- Separation: 0.0212
- At best threshold: P=0.173, R=0.383, TP=88, FP=422

### #5: gap_mean_w5
- AUC-ROC: 0.5948
- Best F1: 0.2331 (micro), 0.1387 (macro)
- Optimal threshold: 0.6720
- Separation: 1.1829
- At best threshold: P=0.168, R=0.383, TP=88, FP=437

## Per-Task-Type Analysis

Comparing best feature (gap_max_w3) vs raw_gap baseline:

| Task Type | N | raw_gap F1 | Best Feature F1 | Delta |
|-----------|---|------------|-----------------|-------|
| Action Reasoning | 5 | 0.809 | 0.816 | +0.008 |
| Action Recognition | 5 | 0.215 | 0.198 | -0.017 |
| Attribute Perception | 5 | 0.295 | 0.267 | -0.028 |
| Ego Object Localization | 5 | 0.203 | 0.182 | -0.021 |
| Ego Object State Change Recognition | 5 | 0.190 | 0.200 | +0.010 |
| Information Function | 5 | 0.400 | 0.444 | +0.044 |
| Object Function | 5 | 0.275 | 0.227 | -0.047 |
| Object Localization | 5 | 0.273 | 0.261 | -0.012 |
| Object Recognition | 5 | 0.323 | 0.317 | -0.005 |
| Object State Change Recognition | 5 | 0.174 | 0.200 | +0.026 |
| Task Understanding | 5 | 0.690 | 0.690 | -0.001 |
| Text-Rich Understanding | 5 | 0.388 | 0.388 | +0.000 |

## Interpretation

### Key Findings
- See feature ranking above for which temporal features improve over raw single-frame gap
- AUC-ROC measures how well the feature separates GT from non-GT steps (regardless of threshold)
- Best F1 measures actual triggering performance after threshold + cooldown simulation
- Separation = GT_mean - nonGT_mean: positive means feature is higher during GT windows
