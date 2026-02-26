# ProAssist Adaptive-D 综合分析报告

> 生成时间: 2026-02-25 04:17
> 模型: ProAssist-Model-L4096-I1 (LLaMA-3.1-8B + LoRA r=128 + SigLIP-SO400M)
> 假设: 通过 per-dataset 阈值优化 (training-free Adaptive-D) 提升主动对话时机判断

## 1. 数据概览

| 来源 | 文件 | 状态 | 阈值点数 |
|------|------|------|----------|
| wtag_initial | `results/threshold_sweep/wtag_dialog-klg-sum_val_L4096_I1/sweep_results.json` | OK | 6 |
| wtag_fine | `results/threshold_sweep_fine/wtag_dialog-klg-sum_val_L4096_I1/sweep_results.json` | 缺失 | — |
| ego4d | `results/threshold_sweep_ego4d/ego4d_narration_val_L4096_I1/sweep_results.json` | 缺失 | — |

已加载数据集: **wtag**

## 2. Per-Dataset 阈值优化结果

### 2.1. WTAG

**最优阈值: τ*=0.2, F1=0.3727**
- Delta F1 vs 论文: +0.0387 (+11.6%)
- 论文 baseline: τ=0.4, F1=0.3340

| τ | F1 | Precision | Recall | Missing | Redundant | CIDEr | Bleu_4 |
|---:|------:|----------:|-------:|--------:|----------:|------:|-------:|
| 0.15 | 0.3520 | 0.4910 | 0.2744 | 0.4813 | 0.0718 | 0.8653 | 0.2342 |
|  **0.2** |  **0.3727** | 0.4759 | 0.3063 | 0.4257 | 0.1076 | 0.7678 | 0.2163 |
| 0.25 | 0.3486 | 0.4198 | 0.2981 | 0.4166 | 0.1784 | 0.7782 | 0.2165 |
| 0.35 | 0.3562 | 0.3882 | 0.3291 | 0.3409 | 0.2226 | 0.7747 | 0.2035 |
| 0.4 | 0.3310 | 0.3321 | 0.3300 | 0.3500 | 0.3459 | 0.7704 | 0.2052 |
| 0.45 | 0.3212 | 0.3136 | 0.3291 | 0.3473 | 0.3779 | 0.7606 | 0.2053 |

**Precision-Recall Trade-off:**
- Precision 趋势: monotonic (范围: [0.3136, 0.4910])
- Recall 趋势: non-monotonic (范围: [0.2744, 0.3300])
- 最高 Precision 点: τ=0.15, P=0.4910, R=0.2744, F1=0.3520
- 最高 Recall 点: τ=0.4, P=0.3321, R=0.3300, F1=0.3310

**阈值敏感度 (F1 在最优点附近的稳定性):**
- 最优点梯度: -0.0343 (接近0表示稳定)
- τ*+-0.05 内 F1 最大下降: 0.0241
- τ*+-0.10 内 F1 最大下降: 0.0241


### 2.2. EGO4D

> 数据未就绪，等待 sweep 完成。


**论文 Baseline** (model: ProAssist-Model-L4096-I1 (nr0.1)):

- 最佳阈值: τ=0.3, F1=0.2750
- 数据集: narration_val_L4096_I1
## 3. 跨数据集 Adaptive-D 汇总

| 数据集 | 论文 τ | 论文 F1 | 我们 τ* | 我们 F1 | ΔF1 | 判定 |
|--------|--------|---------|---------|---------|-----|------|
| WTAG | 0.4 | 0.3340 | 0.2 | 0.3727 | +0.0387 | PASS (显著改进) |
| EGO4D | 0.3 | 0.275 | 待定 | 待定 | 待定 | 数据未就绪 |

**平均 ΔF1: +0.0387** (across 1 datasets)

## 4. 变化检测假设 (Change Detection)

**结论: REJECTED**

- SigLIP change scores 对 talk/no-talk 无区分度 (max separation=0.00082)
- w2t_prob 信号质量: 双峰分布，86% > 0.99，3.2% < 0.01，τ=0.3 处 frame-level 完美分类
- SigLIP CLS 特征的变化分数对 talk/no-talk 决策无额外贡献，模型已内化视觉变化信息。

## 5. 关键洞察

1. **WTAG**: 最优 τ*=0.2 (论文 τ=0.4)，F1 提升 +0.0387 (+11.6%)。模型应比论文默认设置更保守 (lower τ = higher confidence threshold)。
2. **WTAG P-R**: Precision 0.4759 vs Recall 0.3063，CIDEr 0.7678。低阈值牺牲 Recall 换取更高 Precision 和内容质量。
3. **Change Detection 独立信号**: 被否决。SigLIP 变化分数已被 w2t_prob 内化，无需额外 change-aware 模块。
4. **w2t_prob 标定质量**: 双峰分布，绝大多数帧的预测高度确定 (86% > 0.99, 3.2% < 0.01)，不确定区间仅 2.6%。
5. **Adaptive-D 核心价值**: Training-free，零额外推理开销，仅需少量验证样本即可找到最优 τ。从 ESTP-Bench 迁移到 ProAssist 成功。

## 6. 结论

**Adaptive-D 假设: 部分验证 (1/1 datasets PASS)**

待完成数据集: ego4d

## 7. 后续步骤

- [ ] ego4d sweep 完成后重新运行本分析脚本
- [ ] 对比 WTAG dialog_val_L0_I1 (论文原始数据格式) 的结果
- [ ] holoassist/diff 数据集测试 (如 LFS 数据可用)
- [ ] 将最优 τ 写入部署配置

## 8. 图表

图表保存在 `results/figures/` 目录:

1. `f1_vs_threshold_all.png` — F1 / Precision / Recall vs τ 曲线 + 论文 baseline
2. `precision_recall_curve.png` — P-R 曲线 (参数化 by τ)
3. `error_rate_decomposition.png` — Missing / Redundant Rate vs τ
