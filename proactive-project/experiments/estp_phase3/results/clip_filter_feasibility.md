# CLIP 帧间变化作为 VLM 触发前置过滤器 — 可行性分析

## 1. 动机

当前系统在每个时间步（约 5.7s 间隔）都调用 VLM 进行 "是否需要主动提示" 的判断，这非常昂贵（A100 上约 5s/帧）。如果能用轻量级的 CLIP 视觉特征变化作为前置过滤器，只在视觉场景发生显著变化时才调用 VLM，就可以大幅降低计算成本。

**核心假设**: GT 答案窗口（需要主动提示的时刻）通常伴随着视觉场景的显著变化（新物体出现、场景切换、动作转换等），这些变化会在 CLIP 特征空间中表现为较大的帧间距离。

## 2. 方法

### 2.1 CLIP 特征提取
- 模型: CLIP ViT-B/32 (openai/clip-vit-base-patch32)
- 采样率: 2fps（与预处理视频一致）
- 指标: `clip_change = 1 - cosine_sim(frame_t, frame_{t-1})`
  - 值域 [0, 2]，实际多在 [0, 0.3] 范围
  - 0 = 完全相同场景
  - 较大值 = 场景发生显著变化

### 2.2 分析维度
1. **分布对比**: GT 窗口内 vs 非 GT 窗口的 clip_change 均值对比
2. **早期预警**: GT 窗口前 2s 内是否有 clip_change 峰值（P90+）
3. **阈值扫描**: 不同 theta 下的窗口召回率 (window recall) 和过滤率 (filter rate)
4. **任务类型细分**: 不同任务类型对 CLIP 变化的敏感度差异

### 2.3 数据集
- 60 个 case（来自 Phase I fullscale_d checkpoint）
- 12 种任务类型
- 视频: ESTP-Bench full_scale_2fps_max384

## 3. 理论分析 — 预期效果与局限

### 3.1 CLIP 应该有效的任务类型

| 任务类型 | 预期 CLIP 信号强度 | 理由 |
|----------|-------------------|------|
| Object Recognition | 强 | 新物体出现/消失 = 显著视觉变化 |
| Object Function | 中-强 | 使用物体通常涉及动作+外观变化 |
| Information Function | 中 | 信息显示场景可能有文字/界面变化 |

### 3.2 CLIP 可能不够有效的任务类型

| 任务类型 | 预期 CLIP 信号强度 | 理由 |
|----------|-------------------|------|
| Action Recognition | 弱 | 连续动作（搅拌、走路）= 持续高变化，无法区分关键时刻 |
| Attribute Perception | 弱 | 颜色/大小等微妙属性变化在 CLIP 空间中不敏感 |
| Task Understanding | 不确定 | 取决于任务步骤是否伴随场景变化 |
| Text-Rich Understanding | 中 | 文字变化可能被 CLIP 捕捉（CLIP 有 OCR 能力） |

### 3.3 关键理论限制
1. **CLIP 是语义级特征**: 检测 "什么在场景中" 而非精确的像素变化。微妙但重要的变化可能被忽略。
2. **2fps 采样 + 帧间比较**: 快速动作可能在每帧都产生大变化，导致噪声过多。
3. **单帧对比 vs 累积变化**: 渐进式场景变化（如缓慢走向新位置）单帧变化小但累积变化大。

## 4. 预期分析指标

分析脚本 `clip_filter_analysis.py` 将产出以下指标:

### 4.1 关键决策指标
- **GT/non-GT clip_change 比率**: 如果 > 1.5x，说明 CLIP 有区分能力
- **P90 峰值在 GT 前 2s 出现的比例**: 如果 > 50%，说明 CLIP 可做早期预警
- **theta=0.05 时的 window recall**: 需要 > 80% 才实用
- **theta=0.05 时的 filter rate**: 需要 > 50% 才有意义节省计算

### 4.2 理想过滤器特征
一个好的 CLIP 过滤器应该:
- window recall >= 90%（不遗漏重要触发时刻）
- filter rate >= 60%（过滤掉足够多的无用帧）
- 这意味着只需在 40% 的时间步调用 VLM，节省 60% 计算成本

## 5. 与现有方法的关系

### 5.1 与 Adaptive-D 的互补性
- **Adaptive-D** (Phase II): 基于 VLM logprob_gap 的阈值过滤，F1=0.222
  - 优势: 直接利用 VLM 判断的置信度
  - 劣势: 每步都需要 VLM 推理
- **CLIP Filter**: 视觉变化的前置过滤
  - 优势: 极低计算成本（GPU 上 ~1ms/帧 vs VLM 的 ~5s/帧）
  - 劣势: 纯视觉信号，不理解语义
- **互补组合**: CLIP filter -> VLM inference -> Adaptive-D threshold
  - 预期: 在保持 Adaptive-D F1 的同时减少 60%+ 的 VLM 调用

### 5.2 与 Reasoning Trigger 的对比
- Reasoning Trigger (Gemini-3-Flash) 在多数任务类型上表现差 (F1=0.173)
- CLIP Filter 定位不同: 不是替代 VLM 判断，而是决定 "是否值得调用 VLM"
- 即使 CLIP Filter 只是减少 FP（过滤掉明显无变化的帧），也有价值

## 6. Bug 修复记录

### 6.1 问题描述
`phase3_clip_extractor.py` 在使用 transformers backend 时报错:
```
'BaseModelOutputWithPooling' object has no attribute 'float'
```

### 6.2 根因
transformers >= 5.x 中 `CLIPModel.get_image_features()` 和 `get_text_features()` 返回 `BaseModelOutputWithPooling` 对象（包含 pooler_output、last_hidden_state 等），而非直接的 tensor。代码直接在返回值上调用 `.float()` 导致错误。

### 6.3 修复方案
在 `embed_images_batch` 和 `embed_texts_batch` 中，从返回的 `BaseModelOutputWithPooling` 对象中提取 `.pooler_output`:
```python
out = model.get_image_features(**inputs)
embs = out.pooler_output if hasattr(out, "pooler_output") else out
```
使用 `hasattr` 保持向后兼容：如果旧版本 transformers 直接返回 tensor，代码依然有效。

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `phase3_clip_extractor.py` | 原始 CLIP 提取器（已修复 bug） |
| `clip_filter_analysis.py` | 新建的 2fps CLIP 过滤器可行性分析脚本 |
| `results/clip_filter_analysis.json` | 分析脚本输出的详细结果（运行后生成） |
| `results/clip_filter_analysis_report.txt` | 分析脚本输出的文本报告（运行后生成） |

## 8. 运行说明

```bash
cd /home/v-tangxin/GUI
source ml_env/bin/activate

# GPU 运行（推荐，约 5-10 分钟）
python3 proactive-project/experiments/estp_phase3/clip_filter_analysis.py \
    --device cuda --batch-size 64 --verbose

# CPU 运行（约 30-60 分钟）
python3 proactive-project/experiments/estp_phase3/clip_filter_analysis.py \
    --device cpu --verbose

# 快速测试（3 个 case）
python3 proactive-project/experiments/estp_phase3/clip_filter_analysis.py \
    --device cuda --limit 3 --verbose
```

## 9. 下一步

1. **运行分析脚本** 获取实际数据
2. **根据结果决定**:
   - 如果 GT/non-GT 比率 > 1.5x: CLIP filter 值得集成
   - 如果 window recall > 80% at filter rate > 50%: 可作为 Adaptive-D 的前置层
   - 如果效果不显著: 考虑更复杂的视觉变化检测（如光流、多帧累积）
3. **如果可行**: 设计 CLIP + Adaptive-D 的两阶段过滤流水线
