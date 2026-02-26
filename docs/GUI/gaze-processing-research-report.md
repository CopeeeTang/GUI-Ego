# 眼动数据处理调研报告

**日期**: 2026-02-10
**范围**: Preview 系统眼动数据处理现状、StreamGaze 论文借鉴分析、Bug 清单

---

## 1. 当前系统架构

### 1.1 数据流全链路

```
原始 gaze.csv (Pupil Labs Neon, 200Hz, ns时间戳)
  │  字段: timestamp [ns], gaze x [px], gaze y [px], fixation id, worn
  │
  ▼
SignalExtractor.save_signals_to_csv()
  │  切片为 sample 级 signals/gaze.csv
  │  字段: time_s, gaze x [px], gaze y [px], worn, fixation id
  │  时间窗口: annotation时间 ±5s
  │
  ▼
server.py /api/gaze 端点
  │  返回 JSON: [{time, x, y}, ...]
  │  ⚠️ 问题: 丢弃了 fixation id 和 worn 字段
  │
  ▼
前端 video_overlay.html
  │  gazeData 数组 → getGazeAt(time) 二分查找 → 渲染
  │  ⚠️ 问题: 二分查找实现有 bug
  │
  ▼
视觉渲染
  ├── gaze-indicator (红色圆点, z-index:5)
  └── ui-overlay (UI组件, z-index:10, transform: translate(-50%,-50%))
```

### 1.2 UI 展示时机 (当前逻辑)

**代码位置**: `video_overlay.html:321-341`

```javascript
// 当前实现: 视频最后 N 秒展示 UI
const uiStartTime = duration - uiDuration;  // 硬编码为视频末尾
if (currentTime >= uiStartTime) {
    uiOverlay.classList.remove('hidden');
}
```

**问题**: UI 展示时机完全基于视频时长计算，与用户注视行为无关。用户可能在视频早期就有稳定注视，但 UI 不会显示；或者视频末尾用户在做快速扫视，UI 却强制显示。

### 1.3 UI 位置与眼动数据的关系

支持三种定位模式 (`video_overlay.html:396-406`):

| 模式 | 位置计算 | 说明 |
|------|---------|------|
| `gaze` (Follow Gaze) | `left=displayX, top=displayY` | 跟随注视点，UI 中心对准 gaze 坐标 |
| `center` | `left=50%, top=50%` | 视频中心 |
| `fixed` | `left=50%, top=70%` | 固定在视频下方 70% 位置 |

坐标转换: `displayX = gaze.x * (videoRect.width / video.videoWidth)`

### 1.4 Gaze Fallback 策略

| 策略 | 前端实现 | 后端实现 (gaze_anchor.py) |
|------|---------|--------------------------|
| `last_valid` | ✅ 向前遍历找有效点 | ✅ numpy mask 搜索 |
| `center` | ✅ 返回视频中心 | ✅ 返回 default_center |
| `interpolate` | ❌ UI 有选项但未实现 | ✅ 线性插值实现完整 |

---

## 2. StreamGaze 论文借鉴分析

**论文**: StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos
**作者**: Daeun Lee et al., UNC Chapel Hill & Adobe Research, 2025
**代码**: https://github.com/daeunni/StreamGaze

### 2.1 可借鉴的核心方法

#### (A) Fixation Extraction (注视提取)

StreamGaze 使用两阶段过滤提取稳定注视时刻:

**阶段1: 空间-时间稳定性**
```
Point-wise stability:
  max(||g_t - c_k||) ≤ θ_d    (空间散布 ≤ 阈值, 按帧宽归一化)
  t_end - t_start ≥ θ_t       (持续时间 ≥ 最小阈值)
```

**阶段2: 场景一致性**
```
Scene consistency:
  min(Pearson(H_t, H_{t+1})) ≥ θ_c   (相邻帧 HS 直方图相关性)
```

**对我们的启发**: 我们的 Pupil Labs Neon 硬件已提供 `fixation id` 字段（设备端 I-DT 算法），无需重新实现 fixation 检测。但可以借鉴 StreamGaze 的思路，在设备 fixation 基础上增加 FOV 区域约束。

#### (B) FOV/Out-of-FOV 区域划分

StreamGaze 定义注视区域 (Field of View):
```
FOV_t = {p ∈ F_t : ||p - c_k|| ≤ r_fov}
r_fov = W × tan(θ_fov) / tan(HFOV/2)
```

其中:
- `θ_fov ≈ 8°` (perifoveal 上界)
- `HFOV` = 水平视场角 (Pupil Labs Neon 约 100°)
- `W` = 帧宽 (960px)

**计算示例**:
```
r_fov = 960 × tan(8°) / tan(50°) = 960 × 0.1405 / 1.1918 ≈ 113px
```

**对我们的启发**: 使用 ~113px 的 FOV 半径来约束 UI 位置，确保 UI 不会出现在视野边角。当 gaze 坐标距离视频边缘 < r_fov 时，将 UI 位置钳制到安全区域内。

#### (C) Scanpath 构建与注意力轨迹

StreamGaze 将连续 fixation 串成 scanpath，追踪注意力如何随时间转移。这对于 UI 触发时机很有参考价值 — 不是在任意时刻触发 UI，而是在注意力稳定（fixation 开始）时触发。

#### (D) Visual Prompting 策略

StreamGaze 使用绿点标注注视中心 + 红色圆形区域标注 FOV，这是比我们当前只有红色圆点更丰富的视觉提示方案。

### 2.2 不适合直接搬用的部分

| StreamGaze 方法 | 原因 |
|----------------|------|
| I-DT fixation 检测算法 | Pupil Labs 硬件已提供 fixation id |
| 3D gaze 投影 | 我们的数据已是 2D 像素坐标 |
| HS 直方图场景一致性检测 | 我们的视频片段已经是短片段(6-10s)，场景一致 |
| MLLM object extraction | 不在 preview 系统范围内 |

---

## 3. Bug 清单

### Bug 1: 前端 `getGazeAt()` 二分查找只找 `>=` 的点 [高]

**文件**: `agent/preview/templates/video_overlay.html:347-375`

**问题**: 二分查找实现只找到第一个 `time >= target` 的点，没有与前一个点比较取最近值。当视频时间落在两个 gaze 采样点之间时，总是选择后面的点而非最近的点。

**对比**: 后端 `gaze_anchor.py:120-128` 正确实现了 closest-point 逻辑:
```python
idx = np.searchsorted(times, timestamp)
if idx > 0:
    if abs(times[idx] - timestamp) > abs(times[idx-1] - timestamp):
        idx = idx - 1
```

**修复**: 在前端 `getGazeAt()` 中添加与前一个点的距离比较。

---

### Bug 2: 前端 `interpolate` fallback 选项存在但未实现 [中]

**文件**: `agent/preview/templates/video_overlay.html:365-374`

**问题**: `<select id="gaze-fallback">` 中有 `interpolate` 选项，但 `getGazeAt()` 的 fallback 逻辑中没有 interpolate 分支，直接 fall through 到 center 返回。

**对比**: 后端 `gaze_anchor.py:156-178` 有完整的线性插值实现。

**修复**: 在前端添加 interpolate fallback 逻辑，或者移除该选项。

---

### Bug 3: UI 展示时机硬编码为"视频最后 N 秒" [高 - 设计缺陷]

**文件**: `agent/preview/templates/video_overlay.html:331-341`

**问题**: `uiStartTime = duration - uiDuration` 将 UI 展示窗口硬编码在视频末尾，与用户实际注视行为和提问时间完全无关。

**期望行为**: UI 应在用户提问/推荐结束后的第一个稳定 fixation 时刻触发显示。

**修复**: 从 rawdata.json 获取 `time_interval.end`，在 gaze 数据中找到之后的第一个 fixation，设为 UI 显示起始时间。

---

### Bug 4: `updateGazePosition` 未考虑视频 letterboxing [中]

**文件**: `agent/preview/templates/video_overlay.html:378-407`

**问题**: 使用 `video.getBoundingClientRect()` 计算缩放比时，假设视频完全填充容器。如果浏览器窗口比例与视频不一致，视频会有黑边（letterboxing），导致 gaze 位置偏移。

**修复**: 计算实际视频渲染区域（考虑 `object-fit` 属性），或将视频容器 CSS 设为精确匹配视频比例。

---

### Bug 5: Canvas 录制时 UI 位置使用 CSS 坐标而非视频坐标 [中]

**文件**: `agent/preview/templates/video_overlay.html:458-474`

**问题**: `renderCompositeFrame()` 中获取 UI overlay 位置时使用 `parseFloat(overlayStyle.left)`，这是 CSS 像素坐标，受浏览器缩放影响。导出的视频中 UI 位置与预览中不一致。

**修复**: 在录制时使用原始 gaze 坐标进行定位，而非 CSS 计算后的坐标。

---

### Bug 6: server.py 丢弃 fixation id 字段 [高 - 数据丢失]

**文件**: `agent/preview/server.py:2825-2853`

**问题**: `/api/gaze` 端点只提取 `time_s`, `gaze x [px]`, `gaze y [px]` 三个字段，丢弃了 `fixation id` 和 `worn` 字段。前端无法获知 fixation 信息，无法实现基于 fixation 的 UI 触发。

**修复**: 在 JSON 返回中添加 `fixation_id` 字段。

---

## 4. 优化方案总结

### 4.1 Fixation-Based UI 触发逻辑

```
用户提问/推荐结束时间 (annotation.time_interval.end)
  │
  ▼
在 gaze 数据中搜索 t > end_time 的第一个有效 fixation
  │  条件: fixation_id 不为空 且 连续出现 ≥ N 个采样点
  │
  ▼
计算 fixation 中心 (centroid_x, centroid_y)
  │
  ▼
FOV 安全区域约束
  │  如果距离视频边缘 < r_fov (≈113px), 钳制到安全区域
  │
  ▼
在 fixation 中心位置展示 UI
```

### 4.2 数据传递需求

需要将以下数据从后端传递到前端:
1. `fixation_id` — 在 /api/gaze 端点中添加
2. `annotation_end_time` — 从 rawdata.json 中提取 time_interval.end，通过 /api/rawdata 或模板变量传递
3. `video_resolution` — 用于 FOV 像素半径计算

### 4.3 StreamGaze 借鉴总结

| 借鉴点 | 实施方式 | 优先级 |
|--------|---------|--------|
| Fixation 作为 UI 触发信号 | 利用 Pupil Labs fixation id | 高 |
| FOV 安全区域 (~8° perifoveal) | r_fov ≈ 113px 的钳制约束 | 高 |
| 最近点查找 | 修复前端二分查找 bug | 高 |
| 插值 fallback | 从后端移植线性插值逻辑 | 中 |
| Visual Prompting 增强 | 考虑添加 FOV 圆形区域可视化 | 低 |
