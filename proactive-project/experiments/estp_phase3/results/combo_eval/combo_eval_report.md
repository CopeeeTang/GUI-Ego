# GPT-4o Adaptive-D + Context-RT 组合评估报告

生成时间: 2026-02-22 20:51
数据源: GPT-4o FBF 全量 checkpoint (60 cases, 12 task types)

## 实验设计

**级联过滤架构:**
- Stage 1: GPT-4o Adaptive-D (per-type optimal tau, 离线已有数据)
- Stage 2: Gemini-2.0-Flash Context-RT (Method A: judgment history, 在线验证)
- Cooldown: 12s

**三级决策策略:**
- gap > tau + 5.0 (high margin) -> 高置信直通，不调 RT
- tau < gap <= tau + 5.0 -> 调用 Gemini Context-RT 二次验证
- gap <= tau -> 不触发

**Context-RT Prompt:** 宽松验证式（确认画面中是否可见与问题相关的内容）

## 核心结论

| 指标 | Combo (AD+RT) | Adaptive-D only | Delta |
|------|--------------|-----------------|-------|
| 平均 F1 | **0.304** | 0.323 | -0.019 |
| 总 TP | 110 | 121 | -11 |
| 总 FP | 383 | 394 | -11 |
| 总 FN | 120 | 109 | +11 |
| 95% CI | [-0.036, -0.002] | - | Gate: **FAIL** |
| RT 调用次数 | 263 | 0 | (占候选 263/1257, 20.9%) |

**结论: Combo 未优于纯 Adaptive-D，目标 F1>0.40 未达成。**

参考基准:
- GPT-4o Adaptive-D (gpt4o_adaptive_d_analysis.py): F1=0.371
- Qwen Adaptive-D: F1=0.222

注: 本脚本的 AD-only F1=0.323 与之前报告的 0.371 有差异，原因是 ESTP-F1 计算中 GT window 边界定义略有不同（本脚本使用 gs-ant ~ ge+lat，analysis 脚本使用 gs-ant ~ gs+lat）。

## Per-type 分析

| 任务类型 | n | tau | Combo F1 | AD F1 | Delta | Combo FP | AD FP |
|----------|---|-----|----------|-------|-------|----------|-------|
| Object Recognition | 5 | 2.0 | 0.546 | 0.546 | +0.000 | 28 | 28 |
| Object State Change Recognition | 5 | 3.5 | 0.540 | 0.540 | +0.000 | 5 | 5 |
| Task Understanding | 5 | -5.0 | 0.491 | 0.491 | +0.000 | 50 | 50 |
| Information Function | 5 | 8.0 | 0.300 | 0.344 | -0.044 | 13 | 13 |
| Ego Object Localization | 5 | 3.5 | 0.295 | 0.295 | +0.000 | 30 | 30 |
| Action Reasoning | 5 | 3.0 | 0.267 | 0.326 | -0.059 | 35 | 36 |
| Ego Object State Change Recog. | 5 | -5.0 | 0.267 | 0.300 | -0.033 | 17 | 20 |
| Text-Rich Understanding | 5 | 0.0 | 0.240 | 0.307 | -0.067 | 17 | 17 |
| Object Function | 5 | 4.0 | 0.214 | 0.243 | -0.029 | 41 | 40 |
| Attribute Perception | 5 | 2.8 | 0.193 | 0.193 | +0.000 | 48 | 48 |
| Action Recognition | 5 | 0.5 | 0.172 | 0.163 | **+0.008** | 48 | 55 |
| Object Localization | 5 | 8.0 | 0.125 | 0.125 | +0.000 | 51 | 52 |

**观察:**
- 82% cases: Combo = AD（完全相同，因 high-confidence bypass）
- Action Recognition: 唯一显著受益的类型（+0.008），RT 过滤了 7 个 FP
- 6 种类型退化: RT 误过滤了正确触发（尤其 Text-Rich, Action Reasoning）

## 改进/退化分析

- Cases improved: 5/60 (8%)
- Cases unchanged: 49/60 (82%)
- Cases degraded: 6/60 (10%)
- FP reduction: 394 -> 383 (仅 3%)

## 成本分析

- 总 Context-RT 调用: 263 次 Gemini 调用
- 总 Adaptive-D 候选: 1257
- RT 调用比例: 20.9%（high-confidence bypass 避免了 79% 的调用）
- 总 polling steps: 3017
- RT调用/总steps: 8.7%

## 失败原因分析

### 1. High-Confidence Bypass 覆盖率过大
- tau=-5 的 Task Understanding 和 Ego Object State Change Recognition: bypass threshold = 0.0，几乎所有候选都被 bypass
- tau=0 的 Text-Rich Understanding: bypass threshold = 5.0，大部分 gap>5 的候选被 bypass
- 结果: 82% cases 完全不受 RT 影响

### 2. RT 对中等置信区间效果差
- 当 RT 确实被调用时（263次），它的决策几乎等同于全部说 YES
- 宽松 prompt 下 RT 无法有效区分 GT 和 non-GT
- 偶尔过滤掉正确触发（TP 损失 11 个），净效果为负

### 3. 根本限制: GPT-4o logprob gap 已经很强
- GPT-4o gap separation = +8.527（远超 Qwen 的 +1.418）
- AUC = 0.761
- Adaptive-D 已经充分利用了 gap 信号，RT 无法在此基础上提供增量信息
- 离线模拟显示: 即使 RT 完美过滤 100% FP（保留所有 TP），F1 上限仅为 0.591
- 0-50% FP 过滤率对 F1 几乎无影响（因 cooldown 使 FP 分布稀疏）

### 4. Pilot 对比: 保守 RT (V1) vs 宽松 RT (V2)
- V1 (保守 RT): Recall 崩溃，FP 降 89% 但 F1=0.244
- V2 (宽松 RT + bypass): 几乎无影响，F1=0.304
- 两个极端都不 work — RT 判别力本身不足

## 理论上限分析

```
离线模拟 (Oracle RT, 60 cases):
RT filters   0% FP -> F1=0.316
RT filters  25% FP -> F1=0.316
RT filters  50% FP -> F1=0.316
RT filters  75% FP -> F1=0.591
RT filters 100% FP -> F1=0.591
```

即使 Oracle RT（完美识别 FP），也需要过滤 75%+ FP 才能显著提升。
实际 RT 的 FP 过滤率仅 ~3%，远不够。

## 结论与建议

1. **Combo 策略无效**: Context-RT 无法在 GPT-4o Adaptive-D 基础上提供有意义的增量提升
2. **GPT-4o Adaptive-D F1=0.371 已是当前最佳**: 单纯靠 logprob gap 阈值 + per-type 优化
3. **突破 F1>0.40 的可能路径**:
   - 更精细的 per-type tau 调优（交叉验证而非固定）
   - 增加 polling 频率（当前 ~5.7s/step 可能错过短 GT window）
   - 多模态特征融合（不仅靠 logprob gap，还结合视觉特征变化率）
   - 改用更强的触发判断模型（如 GPT-4o 自身做 RT 而非 Gemini）
