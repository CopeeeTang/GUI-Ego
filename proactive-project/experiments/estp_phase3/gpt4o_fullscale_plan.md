# GPT-4o + Adaptive-D 全量验证执行计划

## 1. 背景与目标

### 1.1 Pilot 结果回顾（5 cases）

| 指标 | GPT-4o | Qwen3-VL | 倍数 |
|------|--------|----------|------|
| Vanilla FBF F1 | 0.283 | 0.127 | 2.2x |
| Yes-rate | 77.6% | 76-83% | ~1x |
| Gap Separation | +6.802 | +1.418 | 4.8x |
| GT gap mean | 8.705 | 2.226 | 3.9x |
| Non-GT gap mean | 1.902 | 0.808 | 2.4x |
| 最优离线配置 | tau=0.0+cd=12 -> F1=0.357 | - | - |
| API latency/step | 3.3s (trigger) / 5.3s (answer) | ~5s/frame (GPU) | - |

### 1.2 全量验证目标

1. **验证 GPT-4o FBF 在 60 cases（5-cases dataset）上的表现**，与 Qwen3-VL fullscale_d (60 cases) 直接对比
2. **收集全量 logprob 数据**，支持离线 Adaptive-D threshold sweep
3. **对比 Qwen3-VL Adaptive-D (F1=0.222)** 的全量结果，验证 GPT-4o 是否实质性优于 Qwen

## 2. 现有代码分析

### 2.1 `gpt4o_fbf_pilot.py` 核心架构

```
数据加载 (flatten_dataset_to_cases)
  → 视频文件定位 (find_video_file)
    → FBF 循环 (run_gpt4o_fbf):
        每步:
          1. extract_single_frame(video_path, timestamp)  -- 单帧提取
          2. pil_to_base64(frame)  -- JPEG 编码
          3. gpt4o_trigger_with_logprobs(img_b64, prompt) -- API 调用，返回 yes/no logprobs
          4. 若 triggered: gpt4o_answer(img_b64, prompt)  -- 生成回答
    → checkpoint JSONL 保存
    → generate_report() 报告生成
```

### 2.2 关键差异（vs fullscale_d_runner.py）

| 特性 | fullscale_d_runner (Qwen) | gpt4o_fbf_pilot |
|------|--------------------------|-----------------|
| 推理 | 本地 GPU (Qwen3-VL) | Azure OpenAI API |
| 帧输入 | 累积帧 (_extract_frames_upto) | 单帧 (extract_single_frame) |
| logprob 提取 | forward pass logits | API top_logprobs (max 5) |
| 触发判断 | gap > tau + cooldown | 仅 vanilla (text says "yes") |
| Baseline 重建 | gap > 0 模拟 | 同上 |
| Checkpoint | JSONL atomic append | 同上 |
| 错误恢复 | resume 机制 | 同上 |

### 2.3 已发现问题

1. **NaN/inf gap 值**：228 步中 3 步异常（1.3%），由 API rate limit 和 top-5 限制导致。已在 `gpt4o_gap_reanalysis.py` 中分析，但 pilot 代码未修复。
2. **数据文件硬编码为 5-cases**：`DATA_5CASES = ESTP_DIR / "estp_bench_sq_5_cases.json"`，已覆盖 60 QA。
3. **无 Adaptive-D 离线模拟**：pilot 代码有 `simulate_threshold_offline`，但报告生成时未做 per-type 优化分析。

## 3. 资源需求评估

### 3.1 规模估算

| 参数 | 5 cases (pilot) | 60 cases (全量) | 53 cases (matched) |
|------|-----------------|-----------------|-------------------|
| QA 数 | 5 | 60 | 53 |
| 总 polling 步 | 228 | ~3,173 | ~2,669 |
| 总时长 | 28.4 min | ~6.6 h | ~5.5 h |
| Trigger API 调用 | 228 | ~3,173 | ~2,669 |
| Answer API 调用 | 177 (77.6%) | ~2,462 | ~2,071 |
| 总 API 调用 | 405 | ~5,635 | ~4,740 |

### 3.2 时间估算

基于 pilot 数据：
- **Trigger 调用**：mean=3.33s, median=2.17s, p95=8.40s
- **Answer 调用**：mean=5.31s（仅 triggered 时）
- **综合每步**：~7.5s（含 trigger + 77.6% answer 概率）

| 场景 | 60 cases | 53 cases |
|------|----------|----------|
| API 时间 | ~6.6h | ~5.5h |
| 考虑重试/限流 | ~7-8h | ~6-7h |
| 保守估计（含中断恢复） | ~8-10h | ~7-8h |

### 3.3 API 配额风险

- **Rate limit**：pilot 中仅 1 次 429 错误（228 步中），约 0.4%
- **Azure 配额**：自建 endpoint (`52.151.57.21:9999`)，需确认 TPM/RPM 限制
- **max_tokens=5 (trigger)**：极低 output token 消耗
- **top_logprobs=5**：Azure OpenAI 的上限值

### 3.4 估算费用

- 总 input tokens: ~1.6M（图片 85 token + 文本 200 token per call）
- 总 output tokens: ~0.26M
- GPT-4o 标准价格：~$6.64（自建 endpoint 可能免费）

## 4. 执行计划

### 4.1 代码修改方案

#### 修改 1: 修复 NaN/inf gap 问题

在 `gpt4o_fbf_pilot.py` 的 `generate_report()` 中添加 `math.isfinite()` 过滤。
同时在 `run_gpt4o_fbf()` 的 logprob_gap 计算中，对 inf 值做 fallback：

```python
# 当 yes 或 no 不在 top_logprobs 时，设置一个合理的 floor 值
YES_FLOOR_LOGP = -20.0  # 约 e^-20 ≈ 2e-9
NO_FLOOR_LOGP = -20.0

# 在 gpt4o_trigger_with_logprobs() 返回前:
if yes_logp == float("-inf"):
    yes_logp = YES_FLOOR_LOGP
if no_logp == float("-inf"):
    no_logp = NO_FLOOR_LOGP
```

#### 修改 2: 扩展到全量（无需改数据路径）

**现有代码已经天然支持全量**：`gpt4o_fbf_pilot.py` 使用 `estp_bench_sq_5_cases.json`（60 QA），`--limit 5` 是 pilot 限制。全量运行只需 `--limit 0` 或不传 limit。

但为了更好的对比，建议新建 `gpt4o_fullscale_runner.py`，基于 pilot 代码增强：

```python
# 关键增强点:
1. 支持 --dataset 参数切换 5-cases / 全量数据
2. 增加 Adaptive-D 离线分析（per-type 最优 tau）
3. 增加 bootstrap CI 统计
4. 增加 batch 模式和进度控制
5. 修复 NaN/inf 问题
```

#### 修改 3: Checkpoint 增强

现有 checkpoint 机制已足够：
- JSONL atomic append
- case_key 去重
- `--resume` 支持

建议增加：
- 进度统计摘要每 N 个 case 打印一次
- 可选的中间报告生成（每 10 cases）
- 异常 case 的详细日志

#### 修改 4: 分批运行策略

```bash
# 分批策略：每批 15 cases，4 批完成 60 cases
# 每批约 1.5-2h，便于监控和中断

# 批次 1（cases 0-14）
python3 .../gpt4o_fullscale_runner.py --limit 15 --verbose

# 批次 2（resume，继续到 30）
python3 .../gpt4o_fullscale_runner.py --limit 30 --resume --verbose

# 批次 3（resume，继续到 45）
python3 .../gpt4o_fullscale_runner.py --limit 45 --resume --verbose

# 批次 4（全量，resume）
python3 .../gpt4o_fullscale_runner.py --resume --verbose

# 最终报告
python3 .../gpt4o_fullscale_runner.py --report_only
```

### 4.2 新脚本设计: `gpt4o_fullscale_runner.py`

```
gpt4o_fullscale_runner.py
├── 数据加载 (复用 flatten_dataset_to_cases)
├── API 调用 (复用 gpt4o_trigger_with_logprobs, gpt4o_answer)
│   └── 增加: logprob floor 处理
├── FBF 循环 (复用 run_gpt4o_fbf)
│   └── 增加: 进度显示增强
├── Checkpoint (复用 JSONL 机制)
│   └── 增加: 中间报告
├── 离线分析 (新增)
│   ├── Adaptive-D per-type threshold sweep
│   ├── Bootstrap CI
│   └── 与 Qwen Phase II 结果对比
└── 报告生成 (增强)
    ├── 整体 vs Qwen 对比表
    ├── Per-type 分析
    ├── Gap 分离度分析
    └── Gate 判定
```

### 4.3 运行命令

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate

# 方案 A: 直接用 pilot 脚本全量运行（最简单）
python3 proactive-project/experiments/estp_phase3/gpt4o_fbf_pilot.py \
    --limit 0 --verbose 2>&1 | tee results/gpt4o_fullscale.log

# 方案 B: 新脚本（推荐）
python3 proactive-project/experiments/estp_phase3/gpt4o_fullscale_runner.py \
    --verbose 2>&1 | tee results/gpt4o_fullscale.log

# 中断后恢复
python3 .../gpt4o_fullscale_runner.py --resume --verbose

# 仅生成报告
python3 .../gpt4o_fullscale_runner.py --report_only
```

### 4.4 预期运行时间

| 阶段 | 时间 | 备注 |
|------|------|------|
| API 预热/连通性检查 | ~15s | 自动 |
| 60 cases FBF 推理 | ~6.5h | 核心 |
| Rate limit 重试开销 | ~0.5-1h | 保守估计 |
| 离线 threshold sweep | ~2min | CPU only |
| 报告生成 | ~1min | CPU only |
| **总计** | **~7-8h** | 可分批 |

## 5. 预期结果分析

### 5.1 基于 Pilot 推断全量预期

#### Vanilla FBF F1

- Pilot: F1=0.283（5 cases，每种 task type 仅 1 个）
- Qwen 全量: F1=0.127（60 cases）
- **预期 GPT-4o 全量: F1=0.20-0.30**
  - 上界受限于 yes-bias (77.6% yes-rate 仍然很高)
  - 下界因更多 hard cases（Action Recognition 在 pilot 中 F1=0.0）

#### Gap Separation

- Pilot: +6.802（极强判别力，p<0.0001）
- Qwen 全量: +1.418
- **预期 GPT-4o 全量: +4.0~+7.0**
  - 可能略降，因为 pilot 中 Action Reasoning 只有 12 步且无 GT window，分析不完整
  - Task Understanding 的 gap 异常高（12.96），全量可能回归均值

#### Adaptive-D 预期

- Qwen Adaptive-D: F1=0.222, delta=+0.081 vs baseline
- **预期 GPT-4o Adaptive-D: F1=0.30-0.40**
  - 理由：gap separation 4.8x 于 Qwen，意味着 threshold 能更精准分离 GT/non-GT
  - 关键优势：GPT-4o 在 Text-Rich (F1=0.778) 和 Task Understanding (F1=0.235) 远超 Qwen pilot

### 5.2 不同 tau 配置的预期 F1 范围

基于 pilot threshold sweep 数据外推：

| tau | cd=0 | cd=12 | 备注 |
|-----|------|-------|------|
| 0.0 | ~0.20-0.28 | ~0.28-0.36 | cd=12 是 pilot 最优 |
| 2.0 | ~0.22-0.28 | ~0.28-0.34 | |
| 5.0 | ~0.22-0.30 | ~0.25-0.32 | 高 tau 可能过滤过多 |
| Adaptive | - | ~0.30-0.40 | per-type 最优 tau |

### 5.3 与 Qwen Adaptive-D 的预期差距

| 指标 | Qwen Adaptive-D | GPT-4o Adaptive-D (预期) | 差距 |
|------|-----------------|-------------------------|------|
| Avg F1 | 0.222 | 0.30-0.40 | +35-80% |
| FP reduction | -72% | -70-85% | 类似或更好 |
| Gap separation | +1.418 | +4.0-7.0 | 3-5x |

## 6. 风险评估

### 6.1 高风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| API 长时间限流 | 中 | 运行时间 2x+ | 分批运行 + 指数退避 |
| API endpoint 不可用 | 低 | 完全阻塞 | checkpoint 机制 + 重试 |
| 网络超时 | 中 | 个别 case 失败 | max_retries=6, timeout=30s |

### 6.2 中风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| NaN/inf gap 比例增加 | 中 | 统计精度下降 | logprob floor + isfinite 过滤 |
| 某些视频文件缺失 | 低 | case 跳过 | 状态记录为 video_not_found |
| GPT-4o 全量表现不如 pilot | 中 | 结论反转 | 这是实验本身的价值 |

### 6.3 低风险

- 磁盘空间不足：checkpoint ~150KB/5cases -> ~1.8MB/60cases，可忽略
- Python 环境问题：pilot 已验证运行环境
- 费用超支：自建 endpoint，估计 <$10

## 7. 推荐执行步骤

### Step 1: 修复 pilot 脚本中的 NaN/inf 问题（5 min）
修改 `gpt4o_fbf_pilot.py` 中的 logprob floor 处理。

### Step 2: 在现有 5 cases 上验证修复（2 min）
```bash
python3 .../gpt4o_fbf_pilot.py --report_only
```

### Step 3: 全量运行（6-8 h）
```bash
# 清理旧 checkpoint（或使用新输出目录）
# 然后运行全量
python3 .../gpt4o_fbf_pilot.py --limit 0 --verbose --resume
```

### Step 4: 离线 Adaptive-D 分析（无需 API，5 min）
运行 Phase II 风格的 per-type threshold sweep。

### Step 5: 生成对比报告

## 8. 最终判断标准

### Gate 条件（统计显著性）

```
通过条件:
  1. GPT-4o Adaptive-D F1 > Qwen Adaptive-D F1 (0.222)
  2. Bootstrap 95% CI lower > 0 (paired delta)
  3. 在至少 8/12 task types 上 delta >= 0

失败条件:
  1. GPT-4o 全量 F1 < 0.20 (低于 Qwen Adaptive-D)
  2. gap separation 全量 < +2.0 (pilot 偏差太大)
  3. NaN/inf 比例 > 5%
```

---

**文档版本**: v1.0
**创建日期**: 2026-02-21
**作者**: 自动生成
**状态**: 等待审核执行
