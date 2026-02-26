# 视频处理与 UI Overlay 探索文档

## 概述

本文档描述了 v3_with_visual 策略相关的视频处理流程，以及将 UI 组件覆盖到视频帧的初步探索工作。

---

## 1. 视频处理模块

### 1.1 模块结构

```
agent/src/video/
├── __init__.py
├── extractor.py        # 视频帧提取
└── visual_context.py   # 视觉上下文生成
```

### 1.2 FrameExtractor（帧提取器）

**文件**: `agent/src/video/extractor.py`

**核心功能**: 从视频文件中提取指定时间戳的帧

```python
class FrameExtractor:
    def __init__(self, video_path: str | Path)
    def extract_frames(self, start_time: float, end_time: float, num_frames: int = 3) -> list[np.ndarray]
    def frames_to_base64(self, frames: list[np.ndarray], format: str = "jpeg") -> list[str]
    def frames_to_data_urls(self, frames: list[np.ndarray]) -> list[str]
    def save_frames(self, frames: list[np.ndarray], output_dir: str) -> list[Path]
```

**关键属性**:
| 属性 | 类型 | 说明 |
|-----|------|------|
| `video_path` | Path | 视频文件路径 |
| `fps` | float | 视频帧率 |
| `duration` | float | 视频总时长（秒） |
| `width` / `height` | int | 视频分辨率 |

**帧提取算法**:
```
给定 start_time, end_time, num_frames=3:
  interval = (end_time - start_time) / (num_frames + 1)
  timestamps = [start_time + interval * (i + 1) for i in range(num_frames)]

示例: start=10s, end=20s, num_frames=3
  interval = 10 / 4 = 2.5s
  timestamps = [12.5s, 15.0s, 17.5s]
```

### 1.3 SegmentedVideoExtractor（分段视频提取器）

用于处理分段录制的视频文件：

```python
class SegmentedVideoExtractor(FrameExtractor):
    def __init__(self, segment_paths: list[str | Path])
    # 自动处理跨段时间戳映射
```

**工作原理**:
- 计算每个段的累积时长
- 将全局时间戳映射到段内本地时间戳
- 从正确的段中提取帧

### 1.4 视频文件查找

```python
FrameExtractor.find_video_for_participant(
    data_root: str | Path,
    participant: str,
    session_id: Optional[str] = None
) -> Optional[Path]
```

**查找优先级**:
1. 合并视频: `{participant}/Timeseries Data + Scene Video/{session}/*.mp4`
2. 分段视频: `{participant}/raw_data/{session}/Neon Scene Camera v1 ps*.mp4`

---

## 2. 视觉上下文生成

### 2.1 VisualContextGenerator

**文件**: `agent/src/video/visual_context.py`

**两种模式**:

| 模式 | 枚举值 | 描述 | 输出 |
|-----|-------|------|------|
| DIRECT | `direct` | 直接传递图像给多模态 LLM | `frames_base64` 列表 |
| DESCRIPTION | `description` | 用 VLM 生成文本描述 | 结构化场景描述 |

### 2.2 场景描述 Prompt

```markdown
Analyze these {num_frames} first-person perspective images from smart glasses.

Describe the visual context in a structured format:

1. **Environment**: 类型、位置、光线条件
2. **Key Objects**: 主要物体及其位置、可见文字
3. **User Activity**: 用户正在做什么、移动方向
4. **Context for UI**: AR 锚点、可见性、遮挡区域
```

### 2.3 VisualContext 数据结构

```python
@dataclass
class VisualContext:
    mode: VisualContextMode           # DIRECT 或 DESCRIPTION
    frames_base64: Optional[list[str]] # DIRECT 模式下的 base64 帧
    description: Optional[str]         # DESCRIPTION 模式下的场景描述
    frame_descriptions: Optional[list[str]]  # 单帧描述（可选）
    metadata: Optional[dict[str, Any]]       # 元数据
```

### 2.4 VisualContextCache（缓存）

使用 LRU 缓存避免重复的 VLM 调用：

```python
cache_key = (video_path, start_time, end_time, num_frames, mode)
```

---

## 3. UI Overlay 探索

### 3.1 Overlay 测试输出

**位置**: `agent/output/overlay_test/`

```
overlay_test/
├── frame_original.jpg       # 原始视频帧 (881KB)
└── frame_with_ui_overlay.jpg # 叠加 UI 后的帧 (877KB)
```

### 3.2 Overlay 实现思路

将生成的 A2UI 组件渲染并覆盖到视频帧上的流程：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Video Frame │────▶│  UI Renderer │────▶│  Composite Image │
│  (原始帧)    │     │  (渲染组件)  │     │  (叠加后的帧)    │
└──────────────┘     └──────────────┘     └──────────────────┘
                            ▲
                            │
                     ┌──────┴──────┐
                     │ A2UI JSON   │
                     │ Component   │
                     └─────────────┘
```

### 3.3 技术方案

#### 方案 A: 浏览器渲染 + 截图

```python
# 1. 用 Web 组件渲染 A2UI JSON
# 2. 使用 Playwright/Selenium 截图
# 3. 用 OpenCV 合成到视频帧
```

#### 方案 B: 直接 OpenCV 绘制

```python
import cv2
import numpy as np

def overlay_ui_on_frame(frame: np.ndarray, component: dict) -> np.ndarray:
    overlay = frame.copy()

    # 创建半透明背景
    if component["type"] == "Card":
        x, y, w, h = calculate_position(component)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (30, 30, 30), -1)

        # 混合透明度
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        # 添加文字
        cv2.putText(frame, component["props"]["title"],
                    (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return frame
```

#### 方案 C: Pillow + 字体渲染

```python
from PIL import Image, ImageDraw, ImageFont

def render_glassmorphism_card(frame_pil: Image, component: dict) -> Image:
    # 创建模糊背景效果
    # 渲染圆角矩形
    # 添加文字和图标
    pass
```

### 3.4 v3 与 Overlay 的关联

v3_with_visual 策略生成的组件包含 `visual_anchor` 信息，可用于定位 Overlay：

```json
{
  "type": "ar_label",
  "props": { "text": "有机牛奶", "subtext": "¥28.5" },
  "visual_anchor": {
    "type": "object",
    "target": "牛奶包装盒",
    "position": "right",
    "reasoning": "锚定到用户视野右侧的产品上"
  }
}
```

**锚定类型映射**:

| anchor.type | 含义 | Overlay 策略 |
|-------------|------|-------------|
| `object` | 锚定到物体 | 需要物体检测定位 |
| `location` | 锚定到位置 | 固定屏幕区域 |
| `screen` | 屏幕固定 | 直接使用 position 坐标 |

**position 映射**:

| position | 屏幕位置 |
|----------|---------|
| `left` | 左侧边缘 |
| `right` | 右侧边缘 |
| `center` | 屏幕中央 |
| `top` | 顶部 |
| `bottom` | 底部 |

---

## 4. Preview Server

### 4.1 概述

**文件**: `agent/preview/server.py`

用于在浏览器中预览生成的 A2UI 组件，模拟智能眼镜 HUD 效果。

### 4.2 模拟 HUD 样式

```css
.glasses-frame {
  aspect-ratio: 16/9;
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(10px);  /* Glassmorphism 效果 */
  border-radius: 12px;
}

.ar-card {
  background: rgba(30, 30, 30, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
```

### 4.3 Web Core 组件

**位置**: `agent/preview/web_core/`

包含 TypeScript/Lit 实现的 UI 组件：

- `a2ui-video`: 视频播放组件（支持视频背景测试）
- 各类 A2UI 原子组件的 Web Component 实现

---

## 5. 数据集中的 Overlay 视频

数据集中已包含带有 Gaze Overlay 的视频：

```
data/ego-dataset/data/{participant}/raw_data/{session}/
└── gaze-overlay-output-video-compressed.mp4
```

这些视频展示了用户注视点的可视化，可作为 UI Overlay 位置参考。

---

## 6. 下一步探索方向

### 6.1 实时渲染管线

```
Video Stream → Frame Buffer → Object Detection → UI Placement → Render → Display
                                    ↓                 ↓
                              visual_anchor ← A2UI Component
```

### 6.2 待实现功能

1. **物体检测集成**: 使用 YOLO/SAM 检测锚点物体
2. **动态跟踪**: 跨帧追踪锚点位置
3. **遮挡处理**: 根据深度信息处理 UI 遮挡
4. **视频批量处理**: 批量生成带 Overlay 的视频

### 6.3 评估指标

- UI 可读性评分
- 锚点准确度（物体检测 IoU）
- 遮挡率
- 用户注意力分布（结合 Gaze 数据）

---

## 7. 相关文件索引

| 文件路径 | 说明 |
|---------|------|
| `agent/src/video/extractor.py` | 帧提取核心代码 |
| `agent/src/video/visual_context.py` | 视觉上下文生成 |
| `agent/src/prompts/v3_with_visual.py` | v3 视觉策略 |
| `agent/preview/server.py` | 预览服务器 |
| `agent/preview/web_core/` | Web 组件库 |
| `agent/output/overlay_test/` | Overlay 测试输出 |
