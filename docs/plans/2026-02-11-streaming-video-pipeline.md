# Streaming Video Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有的"标注驱动、单UI输出"pipeline 改造为"视频输入、时序UI序列输出"的模拟流式处理系统，借鉴 StreamGaze 的滑动窗口和 gaze 处理机制。

**Architecture:** 新建 `StreamingPipeline` 类，与现有 `GenerativeUIPipeline` 并行共存（不破坏现有功能）。新 pipeline 接受一段视频作为输入，沿时间轴用滑动窗口逐段处理，每个窗口提取帧+分析场景+生成UI，最终输出带时间戳的 UI 时间轴 JSON。前端在现有 `video_overlay.html` 基础上增强，支持按时间轴回放多个 UI 的切换。

**Tech Stack:** Python 3.11+, OpenCV/decord, ffmpeg, 现有 LLM 客户端 (azure/gemini/claude), 现有 A2UI 组件体系, 现有前端 HTML/JS

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────────┐
                    │              StreamingPipeline (新)                  │
                    │                                                     │
  Video File ──────►│  1. VideoSegmenter                                 │
                    │     └── 滑动窗口裁剪视频段 (借鉴 StreamGaze)         │
  Gaze CSV ───────►│  2. GazeAnalyzer                                    │
  (optional)        │     └── I-VT 注视点提取 → 触发时机检测              │
                    │  3. 现有 FrameExtractor + VisualContextGenerator    │
                    │     └── 每个窗口提取 N 帧 → VLM 场景分析            │
                    │  4. 现有 PromptStrategy.generate()                  │
                    │     └── 生成 A2UI 组件 JSON                         │
                    │  5. TimelineAssembler                               │
                    │     └── 组装时间轴 JSON                              │
                    │                                                     │
                    │  输出: ui_timeline.json                             │
                    └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────────────────┐
                    │          前端增强 (video_overlay.html)               │
                    │                                                     │
                    │  video.timeupdate → 匹配 timeline entry → 切换 UI  │
                    │  平滑过渡动画 → 看起来像实时生成                      │
                    └─────────────────────────────────────────────────────┘
```

## Data Flow

```
输入:
  video.mp4                    # 必须
  gaze.csv                     # 可选 (有则用 gaze 事件驱动, 无则用固定间隔)

处理:
  VideoSegmenter.segment()     → [(0,60), (30,90), (60,120), ...]  # 滑动窗口列表
  GazeAnalyzer.detect_events() → [FixationEvent(t=12.5), FixationEvent(t=35.2), ...]
  合并触发点                    → [12.5, 35.2, 60.0, ...]  # UI 生成时间点

  对每个触发点 t:
    window = [max(0, t-60), t]
    frames = FrameExtractor.extract_frames(window, num_frames=8)
    visual = VisualContextGenerator.generate_context(frames)
    ui     = PromptStrategy.generate(recommendation, scene, visual)

输出: ui_timeline.json
  {
    "video_path": "video.mp4",
    "duration": 180.0,
    "generation_config": {...},
    "timeline": [
      {"time": 12.5, "trigger": "fixation_change", "window": [0, 12.5], "ui": {...}},
      {"time": 35.2, "trigger": "fixation_change", "window": [0, 35.2], "ui": {...}},
      {"time": 60.0, "trigger": "interval",        "window": [0, 60.0], "ui": {...}}
    ]
  }
```

---

## Task 1: VideoSegmenter — 滑动窗口视频分段器

**Files:**
- Create: `agent/src/video/segmenter.py`
- Test: `agent/tests/test_segmenter.py`

**Why:** 当前 `FrameExtractor` 只能从固定时间段提取帧，缺少沿时间轴滑动的能力。借鉴 StreamGaze 的 `split_video()` + 60s 窗口机制。

### Step 1: Write the failing test

```python
# agent/tests/test_segmenter.py
"""Tests for VideoSegmenter."""
import pytest
from agent.src.video.segmenter import VideoSegmenter, TimeWindow


class TestTimeWindow:
    def test_time_window_creation(self):
        w = TimeWindow(start=10.0, end=70.0, trigger_time=70.0, trigger_type="interval")
        assert w.start == 10.0
        assert w.end == 70.0
        assert w.duration == 60.0

    def test_time_window_clamp(self):
        w = TimeWindow(start=-5.0, end=30.0, trigger_time=30.0, trigger_type="interval")
        assert w.start == 0.0  # clamped


class TestVideoSegmenter:
    def test_fixed_interval_windows(self):
        """60s window, 30s stride, on a 180s video."""
        seg = VideoSegmenter(window_size=60.0, stride=30.0)
        windows = seg.generate_windows(video_duration=180.0)

        # Expected trigger times: 60, 90, 120, 150, 180
        assert len(windows) >= 4
        assert windows[0].end == 60.0
        assert windows[0].start == 0.0
        assert all(w.trigger_type == "interval" for w in windows)

    def test_event_driven_windows(self):
        """Windows triggered by external events (gaze fixation changes)."""
        seg = VideoSegmenter(window_size=60.0)
        event_times = [12.5, 35.2, 80.0]
        windows = seg.generate_windows(
            video_duration=180.0,
            event_times=event_times,
        )

        assert len(windows) == 3
        assert windows[0].trigger_time == 12.5
        assert windows[0].start == 0.0   # max(0, 12.5-60)
        assert windows[0].end == 12.5
        assert windows[0].trigger_type == "event"

    def test_hybrid_windows(self):
        """Events + fill gaps with interval windows."""
        seg = VideoSegmenter(window_size=60.0, stride=30.0)
        event_times = [15.0]  # Only one event early on
        windows = seg.generate_windows(
            video_duration=120.0,
            event_times=event_times,
            fill_gaps=True,
            gap_threshold=40.0,
        )

        # Should have event at 15.0 + interval fills for the remaining 105s
        assert windows[0].trigger_time == 15.0
        assert len(windows) > 1

    def test_empty_video(self):
        seg = VideoSegmenter(window_size=60.0)
        windows = seg.generate_windows(video_duration=0.0)
        assert windows == []
```

### Step 2: Run test to verify it fails

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_segmenter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.src.video.segmenter'`

### Step 3: Write minimal implementation

```python
# agent/src/video/segmenter.py
"""Video segmentation with sliding window for streaming-style processing.

Borrows the sliding window concept from StreamGaze's 60-second window mechanism,
but generalizes it to support both fixed-interval and event-driven triggering.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeWindow:
    """A time window for video processing.

    Attributes:
        start: Window start time in seconds.
        end: Window end time in seconds (= trigger_time).
        trigger_time: The timestamp that triggered this window.
        trigger_type: "interval" or "event".
    """

    start: float
    end: float
    trigger_time: float
    trigger_type: str  # "interval" | "event"

    def __post_init__(self):
        if self.start < 0:
            self.start = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


class VideoSegmenter:
    """Generate time windows along a video timeline.

    Supports three modes:
    1. Fixed interval: windows at regular stride intervals.
    2. Event-driven: windows triggered by external events (e.g., gaze changes).
    3. Hybrid: events + interval fill for gaps.
    """

    def __init__(
        self,
        window_size: float = 60.0,
        stride: float = 30.0,
    ):
        self.window_size = window_size
        self.stride = stride

    def generate_windows(
        self,
        video_duration: float,
        event_times: Optional[list[float]] = None,
        fill_gaps: bool = False,
        gap_threshold: float = 40.0,
    ) -> list[TimeWindow]:
        """Generate time windows along the video timeline.

        Args:
            video_duration: Total video length in seconds.
            event_times: External event timestamps (e.g., fixation changes).
                         If provided, windows are centered on these events.
                         If None, uses fixed-interval mode.
            fill_gaps: If True and event_times is provided, fill large gaps
                       between events with interval windows.
            gap_threshold: Minimum gap (seconds) before inserting fill windows.

        Returns:
            Sorted list of TimeWindow objects.
        """
        if video_duration <= 0:
            return []

        if event_times is not None:
            return self._event_driven(video_duration, event_times, fill_gaps, gap_threshold)
        return self._fixed_interval(video_duration)

    def _fixed_interval(self, video_duration: float) -> list[TimeWindow]:
        windows = []
        t = self.window_size
        while t <= video_duration:
            windows.append(TimeWindow(
                start=max(0.0, t - self.window_size),
                end=t,
                trigger_time=t,
                trigger_type="interval",
            ))
            t += self.stride
        # Always include the end of the video if not already covered
        if not windows or windows[-1].end < video_duration:
            windows.append(TimeWindow(
                start=max(0.0, video_duration - self.window_size),
                end=video_duration,
                trigger_time=video_duration,
                trigger_type="interval",
            ))
        return windows

    def _event_driven(
        self,
        video_duration: float,
        event_times: list[float],
        fill_gaps: bool,
        gap_threshold: float,
    ) -> list[TimeWindow]:
        # Filter events within video bounds and sort
        events = sorted(t for t in event_times if 0 < t <= video_duration)
        windows = []
        for t in events:
            windows.append(TimeWindow(
                start=max(0.0, t - self.window_size),
                end=t,
                trigger_time=t,
                trigger_type="event",
            ))

        if fill_gaps and windows:
            windows = self._fill_gaps(windows, video_duration, gap_threshold)

        return sorted(windows, key=lambda w: w.trigger_time)

    def _fill_gaps(
        self,
        windows: list[TimeWindow],
        video_duration: float,
        gap_threshold: float,
    ) -> list[TimeWindow]:
        filled = list(windows)
        all_times = sorted(w.trigger_time for w in windows)

        # Check gap before first event
        if all_times[0] > gap_threshold:
            t = self.window_size
            while t < all_times[0] - gap_threshold / 2:
                filled.append(TimeWindow(
                    start=max(0.0, t - self.window_size),
                    end=t,
                    trigger_time=t,
                    trigger_type="interval",
                ))
                t += self.stride

        # Check gaps between events
        for i in range(len(all_times) - 1):
            gap = all_times[i + 1] - all_times[i]
            if gap > gap_threshold:
                t = all_times[i] + self.stride
                while t < all_times[i + 1] - gap_threshold / 2:
                    filled.append(TimeWindow(
                        start=max(0.0, t - self.window_size),
                        end=t,
                        trigger_time=t,
                        trigger_type="interval",
                    ))
                    t += self.stride

        # Check gap after last event
        if video_duration - all_times[-1] > gap_threshold:
            t = all_times[-1] + self.stride
            while t <= video_duration:
                filled.append(TimeWindow(
                    start=max(0.0, t - self.window_size),
                    end=t,
                    trigger_time=t,
                    trigger_type="interval",
                ))
                t += self.stride

        return filled
```

### Step 4: Run test to verify it passes

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_segmenter.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add agent/src/video/segmenter.py agent/tests/test_segmenter.py
git commit -m "feat: add VideoSegmenter with sliding window support"
```

---

## Task 2: GazeAnalyzer — 注视点提取与事件检测

**Files:**
- Create: `agent/src/video/gaze_analyzer.py`
- Test: `agent/tests/test_gaze_analyzer.py`

**Why:** 当前系统的 gaze 数据只用于 UI 定位（前端渲染时），不参与"何时生成 UI"的决策。借鉴 StreamGaze 的 `extract_fixation_segments()` I-VT 算法，提取注视点变化事件作为 UI 生成触发时机。

### Step 1: Write the failing test

```python
# agent/tests/test_gaze_analyzer.py
"""Tests for GazeAnalyzer — fixation extraction and event detection."""
import pytest
import numpy as np
import pandas as pd
from agent.src.video.gaze_analyzer import GazeAnalyzer, Fixation, GazeEvent


class TestFixationExtraction:
    def _make_gaze_df(self, points: list[tuple[float, float, float]]) -> pd.DataFrame:
        """Helper: list of (time, x, y) -> DataFrame."""
        return pd.DataFrame(points, columns=["time_seconds", "px", "py"])

    def test_stable_gaze_produces_one_fixation(self):
        """Points clustered together for 1 second → one fixation."""
        # 30 points over 1 second, all near (0.5, 0.5)
        rng = np.random.default_rng(42)
        points = [
            (i / 30.0, 0.5 + rng.normal(0, 0.005), 0.5 + rng.normal(0, 0.005))
            for i in range(30)
        ]
        analyzer = GazeAnalyzer()
        fixations = analyzer.extract_fixations(self._make_gaze_df(points))
        assert len(fixations) >= 1
        assert fixations[0].duration >= 0.5

    def test_two_distinct_fixations(self):
        """Gaze at (0.2,0.2) for 1s then jumps to (0.8,0.8) for 1s."""
        points = (
            [(i / 30.0, 0.2, 0.2) for i in range(30)]
            + [(1.0 + i / 30.0, 0.8, 0.8) for i in range(30)]
        )
        analyzer = GazeAnalyzer()
        fixations = analyzer.extract_fixations(self._make_gaze_df(points))
        assert len(fixations) == 2

    def test_empty_dataframe(self):
        analyzer = GazeAnalyzer()
        fixations = analyzer.extract_fixations(pd.DataFrame(columns=["time_seconds", "px", "py"]))
        assert fixations == []


class TestEventDetection:
    def test_fixation_changes_produce_events(self):
        """Each fixation transition should produce a GazeEvent."""
        fixations = [
            Fixation(start_time=0.0, end_time=1.0, center_x=0.2, center_y=0.2, duration=1.0),
            Fixation(start_time=1.1, end_time=2.0, center_x=0.8, center_y=0.8, duration=0.9),
            Fixation(start_time=2.2, end_time=3.5, center_x=0.5, center_y=0.5, duration=1.3),
        ]
        analyzer = GazeAnalyzer()
        events = analyzer.detect_events(fixations)

        # Events at transition points: start of fixation 2 and 3
        assert len(events) >= 2
        assert all(isinstance(e, GazeEvent) for e in events)

    def test_no_fixations_no_events(self):
        analyzer = GazeAnalyzer()
        events = analyzer.detect_events([])
        assert events == []


class TestCSVLoading:
    def test_load_from_csv(self, tmp_path):
        """Load gaze data from a CSV file matching the project's signals/gaze.csv format."""
        csv_content = "time,x,y,fixation_id\n0.0,320,240,0\n0.033,321,241,0\n0.066,322,242,0\n"
        csv_path = tmp_path / "gaze.csv"
        csv_path.write_text(csv_content)

        analyzer = GazeAnalyzer()
        df = analyzer.load_gaze_csv(csv_path)
        assert len(df) == 3
        assert "time_seconds" in df.columns
        assert "px" in df.columns
```

### Step 2: Run test to verify it fails

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_gaze_analyzer.py -v
```
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# agent/src/video/gaze_analyzer.py
"""Gaze data analysis: fixation extraction and event detection.

Borrows the I-VT (Velocity-Threshold Identification) algorithm from
StreamGaze's gaze_processing.py, adapted to our project's gaze.csv format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fixation:
    """A detected gaze fixation."""

    start_time: float
    end_time: float
    center_x: float  # normalized [0, 1]
    center_y: float  # normalized [0, 1]
    duration: float


@dataclass
class GazeEvent:
    """A gaze event that can trigger UI generation."""

    time: float       # event timestamp (seconds)
    event_type: str   # "fixation_start" | "fixation_change" | "long_fixation"
    gaze_x: float     # normalized gaze position
    gaze_y: float
    metadata: Optional[dict] = None


class GazeAnalyzer:
    """Analyze gaze data to extract fixations and detect UI-trigger events.

    Algorithm adapted from StreamGaze's extract_fixation_segments() with
    I-VT (Velocity-Threshold Identification).
    """

    def __init__(
        self,
        radius_thresh: float = 0.05,
        duration_thresh: float = 0.5,
        gap_thresh: float = 0.2,
    ):
        """
        Args:
            radius_thresh: Max distance from fixation start to still count as
                           part of the same fixation (normalized coords, 0-1).
            duration_thresh: Minimum fixation duration in seconds.
            gap_thresh: Allowable brief interruption within a fixation (seconds).
        """
        self.radius_thresh = radius_thresh
        self.duration_thresh = duration_thresh
        self.gap_thresh = gap_thresh

    def load_gaze_csv(
        self,
        csv_path: str | Path,
        video_resolution: Optional[tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Load gaze CSV and normalize to standard format.

        Handles the project's signals/gaze.csv format:
            time, x, y, fixation_id

        Returns DataFrame with columns: time_seconds, px, py
        where px/py are normalized to [0, 1].
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        # Standardize column names
        col_map = {}
        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("time", "timestamp", "time_seconds"):
                col_map[col] = "time_seconds"
            elif cl in ("x", "gaze_x", "px", "norm_pos_x"):
                col_map[col] = "px"
            elif cl in ("y", "gaze_y", "py", "norm_pos_y"):
                col_map[col] = "py"
        df = df.rename(columns=col_map)

        # Normalize pixel coordinates to [0, 1] if needed
        if video_resolution:
            w, h = video_resolution
            if df["px"].max() > 1.0:
                df["px"] = df["px"] / w
            if df["py"].max() > 1.0:
                df["py"] = df["py"] / h
        elif df["px"].max() > 1.0:
            # Auto-detect: assume max value is close to resolution
            df["px"] = df["px"] / df["px"].max()
            df["py"] = df["py"] / df["py"].max()

        return df[["time_seconds", "px", "py"]].dropna()

    def extract_fixations(self, df: pd.DataFrame) -> list[Fixation]:
        """Extract fixation segments from gaze data using I-VT algorithm.

        Adapted from StreamGaze pipeline/preprocess/gaze_processing.py
        """
        if len(df) == 0:
            return []

        timestamps = df["time_seconds"].values
        xs = df["px"].values
        ys = df["py"].values

        fixations = []
        start_idx = 0
        i = 1

        while i < len(xs):
            dist = np.sqrt(
                (xs[i] - xs[start_idx]) ** 2 + (ys[i] - ys[start_idx]) ** 2
            )

            if dist > self.radius_thresh:
                gap_duration = timestamps[i] - timestamps[i - 1]

                if gap_duration <= self.gap_thresh:
                    i += 1
                    continue

                duration = timestamps[i - 1] - timestamps[start_idx]
                if duration >= self.duration_thresh:
                    fixations.append(Fixation(
                        start_time=float(timestamps[start_idx]),
                        end_time=float(timestamps[i - 1]),
                        center_x=float(np.mean(xs[start_idx:i])),
                        center_y=float(np.mean(ys[start_idx:i])),
                        duration=float(duration),
                    ))

                start_idx = i
            i += 1

        # Last segment
        duration = timestamps[-1] - timestamps[start_idx]
        if duration >= self.duration_thresh:
            fixations.append(Fixation(
                start_time=float(timestamps[start_idx]),
                end_time=float(timestamps[-1]),
                center_x=float(np.mean(xs[start_idx:])),
                center_y=float(np.mean(ys[start_idx:])),
                duration=float(duration),
            ))

        logger.info(f"Extracted {len(fixations)} fixations from {len(df)} gaze points")
        return fixations

    def detect_events(self, fixations: list[Fixation]) -> list[GazeEvent]:
        """Detect UI-triggering events from fixation sequence.

        Events are generated at fixation transitions (when the user's gaze
        shifts to a new location), which are natural moments to update UI.
        """
        if len(fixations) < 2:
            return []

        events = []
        for i in range(1, len(fixations)):
            prev = fixations[i - 1]
            curr = fixations[i]
            shift_dist = np.sqrt(
                (curr.center_x - prev.center_x) ** 2
                + (curr.center_y - prev.center_y) ** 2
            )
            events.append(GazeEvent(
                time=curr.start_time,
                event_type="fixation_change",
                gaze_x=curr.center_x,
                gaze_y=curr.center_y,
                metadata={
                    "shift_distance": float(shift_dist),
                    "fixation_duration": curr.duration,
                    "prev_gaze_x": prev.center_x,
                    "prev_gaze_y": prev.center_y,
                },
            ))

        logger.info(f"Detected {len(events)} gaze events from {len(fixations)} fixations")
        return events
```

### Step 4: Run test to verify it passes

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_gaze_analyzer.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add agent/src/video/gaze_analyzer.py agent/tests/test_gaze_analyzer.py
git commit -m "feat: add GazeAnalyzer with I-VT fixation extraction"
```

---

## Task 3: Enhance FrameExtractor — 更密集的帧采样

**Files:**
- Modify: `agent/src/video/extractor.py`
- Test: `agent/tests/test_extractor_enhanced.py`

**Why:** 当前 `extract_frames()` 固定提取 3 帧，信息量不足。需要支持更大的 `num_frames`（如 8-16），并借鉴 StreamGaze GPT4o.py 的 `np.linspace` 均匀采样和图片尺寸限制。

### Step 1: Write the failing test

```python
# agent/tests/test_extractor_enhanced.py
"""Tests for enhanced FrameExtractor features."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from agent.src.video.extractor import FrameExtractor


class TestDenseFrameSampling:
    """Test extracting more frames (8-16) for streaming windows."""

    def test_extract_16_frames(self, tmp_path):
        """Should handle 16 frames from a time range without issue."""
        # We test the timestamp calculation logic, not actual video I/O
        extractor = FrameExtractor.__new__(FrameExtractor)
        extractor.fps = 30.0
        extractor.duration = 60.0
        extractor.total_frames = 1800
        extractor.width = 1920
        extractor.height = 1080

        # Mock the actual frame reading
        extractor.cap = MagicMock()
        extractor.cap.set = MagicMock()
        extractor.cap.read = MagicMock(return_value=(True, np.zeros((1080, 1920, 3), dtype=np.uint8)))

        frames = extractor.extract_frames(0.0, 60.0, num_frames=16)
        assert len(frames) == 16

    def test_linspace_sampling_timestamps(self):
        """Verify uniform sampling matches StreamGaze's np.linspace approach."""
        extractor = FrameExtractor.__new__(FrameExtractor)
        extractor.fps = 30.0
        extractor.duration = 60.0
        extractor.total_frames = 1800
        extractor.width = 1920
        extractor.height = 1080

        timestamps = extractor.compute_sample_timestamps(0.0, 60.0, num_frames=16)
        expected = np.linspace(0.0, 60.0, 16 + 2)[1:-1]  # exclude endpoints
        np.testing.assert_allclose(timestamps, expected, atol=0.1)


class TestImageResizing:
    def test_resize_large_frame(self):
        """Frames larger than max_size should be downscaled."""
        extractor = FrameExtractor.__new__(FrameExtractor)
        large_frame = np.zeros((4000, 6000, 3), dtype=np.uint8)
        resized = extractor.resize_frame(large_frame, max_size=2048)
        assert max(resized.shape[:2]) <= 2048

    def test_small_frame_unchanged(self):
        """Frames smaller than max_size should not be changed."""
        extractor = FrameExtractor.__new__(FrameExtractor)
        small_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        resized = extractor.resize_frame(small_frame, max_size=2048)
        assert resized.shape == (720, 1280, 3)
```

### Step 2: Run test to verify it fails

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_extractor_enhanced.py -v
```
Expected: FAIL — `AttributeError: 'FrameExtractor' object has no attribute 'compute_sample_timestamps'`

### Step 3: Add methods to existing FrameExtractor

Add to `agent/src/video/extractor.py` — insert these methods into the `FrameExtractor` class, after `frames_to_data_urls()`:

```python
    def compute_sample_timestamps(
        self,
        start_time: float,
        end_time: float,
        num_frames: int = 16,
    ) -> list[float]:
        """Compute uniformly-spaced sample timestamps.

        Uses numpy linspace (matching StreamGaze GPT4o.py pattern) to
        generate evenly distributed timestamps, excluding exact start/end.

        Args:
            start_time: Window start in seconds.
            end_time: Window end in seconds.
            num_frames: Number of timestamps to generate.

        Returns:
            List of timestamps in seconds.
        """
        import numpy as np

        if start_time < 0:
            start_time = 0.0
        if end_time > self.duration:
            end_time = self.duration

        # linspace with num_frames+2 points, exclude first and last (boundaries)
        all_points = np.linspace(start_time, end_time, num_frames + 2)
        return all_points[1:-1].tolist()

    @staticmethod
    def resize_frame(
        frame: "np.ndarray",
        max_size: int = 2048,
    ) -> "np.ndarray":
        """Resize frame if larger than max_size, preserving aspect ratio.

        Borrowed from StreamGaze GPT4o.py encode_image_base64() pattern.

        Args:
            frame: BGR numpy array.
            max_size: Maximum dimension (width or height).

        Returns:
            Resized frame (or original if already small enough).
        """
        h, w = frame.shape[:2]
        if max(h, w) <= max_size:
            return frame
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
```

### Step 4: Run test to verify it passes

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_extractor_enhanced.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add agent/src/video/extractor.py agent/tests/test_extractor_enhanced.py
git commit -m "feat: add dense frame sampling and resize to FrameExtractor"
```

---

## Task 4: StreamingPipeline — 核心流式 Pipeline 类

**Files:**
- Create: `agent/src/streaming_pipeline.py`
- Test: `agent/tests/test_streaming_pipeline.py`

**Why:** 这是整个改造的核心。新建 `StreamingPipeline` 类，编排 VideoSegmenter → GazeAnalyzer → FrameExtractor → VisualContextGenerator → PromptStrategy → 输出 timeline JSON。与现有 `GenerativeUIPipeline` 并行共存。

### Step 1: Write the failing test

```python
# agent/tests/test_streaming_pipeline.py
"""Tests for StreamingPipeline — the core streaming orchestrator."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent.src.streaming_pipeline import StreamingPipeline, UITimelineEntry, UITimeline


class TestUITimelineEntry:
    def test_entry_serialization(self):
        entry = UITimelineEntry(
            time=12.5,
            trigger_type="fixation_change",
            window_start=0.0,
            window_end=12.5,
            ui={"type": "Card", "props": {"title": "Test"}},
        )
        d = entry.to_dict()
        assert d["time"] == 12.5
        assert d["trigger"] == "fixation_change"
        assert d["ui"]["type"] == "Card"


class TestUITimeline:
    def test_timeline_serialization(self):
        timeline = UITimeline(
            video_path="test.mp4",
            duration=60.0,
            entries=[
                UITimelineEntry(10.0, "interval", 0.0, 10.0, {"type": "Card", "props": {}}),
                UITimelineEntry(30.0, "event", 0.0, 30.0, {"type": "Badge", "props": {}}),
            ],
        )
        d = timeline.to_dict()
        assert d["video_path"] == "test.mp4"
        assert len(d["timeline"]) == 2
        assert d["timeline"][0]["time"] < d["timeline"][1]["time"]

    def test_save_and_load(self, tmp_path):
        timeline = UITimeline(
            video_path="test.mp4",
            duration=60.0,
            entries=[UITimelineEntry(10.0, "interval", 0.0, 10.0, {"type": "Card", "props": {}})],
        )
        out = tmp_path / "ui_timeline.json"
        timeline.save(out)

        loaded = json.loads(out.read_text())
        assert loaded["duration"] == 60.0
        assert len(loaded["timeline"]) == 1


class TestStreamingPipelineInit:
    def test_create_with_video_only(self):
        """Minimum viable creation: just a video path."""
        with patch("agent.src.streaming_pipeline.FrameExtractor") as MockFE:
            mock_instance = MagicMock()
            mock_instance.duration = 60.0
            mock_instance.fps = 30.0
            mock_instance.width = 1920
            mock_instance.height = 1080
            MockFE.return_value = mock_instance

            pipeline = StreamingPipeline(
                video_path="/fake/video.mp4",
                model_spec="azure:gpt-4o",
                prompt_strategy="v2_google_gui",
            )
            assert pipeline.video_duration == 60.0
```

### Step 2: Run test to verify it fails

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_streaming_pipeline.py -v
```
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write implementation

```python
# agent/src/streaming_pipeline.py
"""Streaming Video Pipeline — video input → temporal UI timeline output.

Unlike GenerativeUIPipeline (annotation-driven, single UI per recommendation),
this pipeline accepts a raw video file, segments it with sliding windows,
and generates a sequence of UI components along the timeline.

Architecture:
    Video → VideoSegmenter (windows) → FrameExtractor (dense frames)
         → VisualContextGenerator (scene analysis) → PromptStrategy (UI gen)
         → UITimeline (temporal JSON output)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from .llm import create_client, LLMClientBase
from .schema import Recommendation, SceneConfig
from .video.extractor import FrameExtractor
from .video.segmenter import VideoSegmenter, TimeWindow
from .video.gaze_analyzer import GazeAnalyzer, GazeEvent
from .video.visual_context import VisualContextGenerator, VisualContextCache
from .output_validator import validate_a2ui_component, normalize_component, normalize_props, move_visual_anchor_to_metadata

# Import strategy registry from existing pipeline
from .pipeline import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UITimelineEntry:
    """A single UI entry on the timeline."""

    time: float            # trigger timestamp (seconds)
    trigger_type: str      # "interval" | "fixation_change" | ...
    window_start: float    # processing window start
    window_end: float      # processing window end
    ui: dict               # A2UI component JSON
    gaze_x: Optional[float] = None  # gaze position for UI anchoring
    gaze_y: Optional[float] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "time": self.time,
            "trigger": self.trigger_type,
            "window": [self.window_start, self.window_end],
            "ui": self.ui,
        }
        if self.gaze_x is not None:
            d["gaze"] = {"x": self.gaze_x, "y": self.gaze_y}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class UITimeline:
    """Complete UI timeline for a video."""

    video_path: str
    duration: float
    entries: list[UITimelineEntry] = field(default_factory=list)
    generation_config: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "generated_at": datetime.now().isoformat(),
            "generation_config": self.generation_config or {},
            "timeline": [e.to_dict() for e in sorted(self.entries, key=lambda e: e.time)],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved timeline ({len(self.entries)} entries) to {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """Video-in → Temporal UI Timeline-out pipeline.

    Usage:
        pipeline = StreamingPipeline(
            video_path="video.mp4",
            model_spec="azure:gpt-4o",
            prompt_strategy="v2_google_gui",
        )
        timeline = pipeline.run()
        timeline.save("output/ui_timeline.json")
    """

    def __init__(
        self,
        video_path: str,
        model_spec: str = "azure:gpt-4o",
        prompt_strategy: str = "v2_google_gui",
        # Gaze (optional)
        gaze_csv_path: Optional[str] = None,
        video_resolution: Optional[tuple[int, int]] = None,
        # Window parameters
        window_size: float = 60.0,
        stride: float = 30.0,
        # Frame extraction
        num_frames: int = 8,
        max_frame_size: int = 2048,
        # Visual context
        visual_mode: str = "direct",
        # Scene config
        scene_name: str = "general",
        # Output
        output_path: Optional[str] = None,
    ):
        """
        Args:
            video_path: Path to input video file.
            model_spec: LLM model (e.g. "azure:gpt-4o", "gemini:gemini-2.5-pro").
            prompt_strategy: Which strategy to use for UI generation.
            gaze_csv_path: Optional path to gaze CSV. If provided, uses
                           gaze-event-driven windowing; otherwise fixed interval.
            video_resolution: (width, height) for gaze coordinate normalization.
            window_size: Sliding window size in seconds (default 60).
            stride: Window stride for fixed-interval mode (default 30).
            num_frames: Frames to extract per window (default 8).
            max_frame_size: Max pixel dimension for extracted frames.
            visual_mode: "direct" (pass images to LLM) or "description" (VLM text).
            scene_name: Scene type for SceneConfig.
            output_path: Directory for output files.
        """
        self.video_path = Path(video_path)
        self.model_spec = model_spec
        self.num_frames = num_frames
        self.max_frame_size = max_frame_size
        self.scene_name = scene_name
        self.output_path = Path(output_path) if output_path else self.video_path.parent / "output"

        # Video
        self.frame_extractor = FrameExtractor(video_path)
        self.video_duration = self.frame_extractor.duration
        self.video_resolution = video_resolution or (
            self.frame_extractor.width,
            self.frame_extractor.height,
        )

        # Windowing
        self.segmenter = VideoSegmenter(window_size=window_size, stride=stride)

        # Gaze (optional)
        self.gaze_analyzer: Optional[GazeAnalyzer] = None
        self.gaze_events: list[GazeEvent] = []
        if gaze_csv_path:
            self.gaze_analyzer = GazeAnalyzer()
            df = self.gaze_analyzer.load_gaze_csv(gaze_csv_path, self.video_resolution)
            fixations = self.gaze_analyzer.extract_fixations(df)
            self.gaze_events = self.gaze_analyzer.detect_events(fixations)
            logger.info(f"Loaded {len(self.gaze_events)} gaze events from {gaze_csv_path}")

        # LLM + Strategy
        self.llm_client = create_client(model_spec)
        if prompt_strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {prompt_strategy}. Available: {list(STRATEGY_REGISTRY.keys())}")
        strategy_class = STRATEGY_REGISTRY[prompt_strategy]
        self.strategy = strategy_class(llm_client=self.llm_client)
        self.strategy_name = prompt_strategy

        # Visual context
        self.visual_generator = VisualContextGenerator(
            llm_client=self.llm_client,
            mode=visual_mode,
        )
        self.visual_cache = VisualContextCache(max_size=100)

        logger.info(
            f"StreamingPipeline initialized: video={self.video_path.name} "
            f"({self.video_duration:.1f}s), model={model_spec}, "
            f"strategy={prompt_strategy}, gaze={'yes' if gaze_csv_path else 'no'}"
        )

    def run(self) -> UITimeline:
        """Run the streaming pipeline.

        Returns:
            UITimeline with generated UI components at each trigger point.
        """
        # Step 1: Generate time windows
        event_times = [e.time for e in self.gaze_events] if self.gaze_events else None
        windows = self.segmenter.generate_windows(
            video_duration=self.video_duration,
            event_times=event_times,
            fill_gaps=True if event_times else False,
            gap_threshold=40.0,
        )

        print(f"🎬 StreamingPipeline: {len(windows)} windows for {self.video_duration:.1f}s video")
        print(f"   Model: {self.model_spec}, Strategy: {self.strategy_name}")
        print(f"   Frames per window: {self.num_frames}")
        if self.gaze_events:
            print(f"   Gaze events: {len(self.gaze_events)} fixation changes")
        print()

        # Step 2: Process each window
        timeline = UITimeline(
            video_path=str(self.video_path),
            duration=self.video_duration,
            generation_config={
                "model": self.model_spec,
                "strategy": self.strategy_name,
                "window_size": self.segmenter.window_size,
                "stride": self.segmenter.stride,
                "num_frames": self.num_frames,
                "has_gaze": bool(self.gaze_events),
            },
        )

        scene = SceneConfig(name=self.scene_name)

        for window in tqdm(windows, desc="Processing windows"):
            entry = self._process_window(window, scene)
            if entry:
                timeline.entries.append(entry)

        print(f"\n✅ Generated {len(timeline.entries)} UI entries on timeline")
        return timeline

    def _process_window(
        self,
        window: TimeWindow,
        scene: SceneConfig,
    ) -> Optional[UITimelineEntry]:
        """Process a single time window: extract frames → analyze → generate UI."""
        try:
            # Extract frames
            frames = self.frame_extractor.extract_frames(
                window.start, window.end, num_frames=self.num_frames,
            )
            if not frames:
                logger.warning(f"No frames for window [{window.start:.1f}, {window.end:.1f}]")
                return None

            # Resize frames
            frames = [FrameExtractor.resize_frame(f, self.max_frame_size) for f in frames]

            # Generate visual context
            visual_context = self._get_visual_context(frames, window)

            # Build a synthetic Recommendation for the strategy
            rec = self._build_recommendation(window)

            # Generate UI component
            component = self.strategy.generate(rec, scene, visual_context)
            if not component:
                return None

            # Validate and normalize
            is_valid, _ = validate_a2ui_component(component)
            if not is_valid:
                component = normalize_component(component)
                component = move_visual_anchor_to_metadata(component)
            component = normalize_props(component)

            # Find gaze position at trigger time
            gaze_x, gaze_y = self._get_gaze_at(window.trigger_time)

            return UITimelineEntry(
                time=window.trigger_time,
                trigger_type=window.trigger_type,
                window_start=window.start,
                window_end=window.end,
                ui=component,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                metadata={
                    "num_frames": len(frames),
                    "strategy": self.strategy_name,
                    "model": self.model_spec,
                },
            )

        except Exception as e:
            logger.error(f"Failed to process window [{window.start:.1f}, {window.end:.1f}]: {e}")
            return None

    def _get_visual_context(self, frames, window: TimeWindow) -> Optional[dict]:
        """Get visual context, using cache when possible."""
        cache_key = (str(self.video_path), window.start, window.end)
        cached = self.visual_cache.get(
            str(self.video_path), window.start, window.end,
            self.num_frames, "direct",
        )
        if cached:
            return cached.to_dict()

        context = self.visual_generator.generate_context(frames)
        self.visual_cache.set(
            str(self.video_path), window.start, window.end,
            self.num_frames, "direct", context,
        )
        return context.to_dict()

    def _build_recommendation(self, window: TimeWindow) -> Recommendation:
        """Build a synthetic Recommendation for this window.

        The existing PromptStrategy.generate() expects a Recommendation object.
        We construct one from the window context.
        """
        # Find any gaze event matching this window
        event_text = ""
        if window.trigger_type == "event":
            matching = [e for e in self.gaze_events if abs(e.time - window.trigger_time) < 1.0]
            if matching:
                event_text = f" User gaze shifted to ({matching[0].gaze_x:.2f}, {matching[0].gaze_y:.2f})."

        return Recommendation(
            id=f"stream_{window.trigger_time:.1f}",
            type="streaming_context",
            content=f"Analyze the current scene at t={window.trigger_time:.1f}s and provide contextual information.{event_text}",
            start_time=window.start,
            end_time=window.end,
            metadata={
                "trigger_type": window.trigger_type,
                "trigger_time": window.trigger_time,
                "window_duration": window.duration,
            },
        )

    def _get_gaze_at(self, time: float) -> tuple[Optional[float], Optional[float]]:
        """Get the gaze position closest to the given time."""
        if not self.gaze_events:
            return None, None
        closest = min(self.gaze_events, key=lambda e: abs(e.time - time))
        if abs(closest.time - time) < 5.0:  # within 5 seconds
            return closest.gaze_x, closest.gaze_y
        return None, None
```

### Step 4: Run test to verify it passes

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_streaming_pipeline.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add agent/src/streaming_pipeline.py agent/tests/test_streaming_pipeline.py
git commit -m "feat: add StreamingPipeline for video-to-timeline processing"
```

---

## Task 5: CLI Entry Point — 命令行接口

**Files:**
- Modify: `agent/src/pipeline.py` (add streaming subcommand to `main()`)
- Or create: `agent/src/streaming_cli.py` (standalone entry point)

**Why:** 用户需要一个简单的命令来运行新 pipeline。

### Step 1: Create standalone CLI

```python
# agent/src/streaming_cli.py
"""CLI entry point for the Streaming Video Pipeline."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Streaming Video → UI Timeline Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed-interval mode (no gaze data)
  python3 -m agent.src.streaming_cli --video video.mp4 --model azure:gpt-4o

  # Gaze-driven mode
  python3 -m agent.src.streaming_cli --video video.mp4 --gaze signals/gaze.csv

  # Custom window size and frame count
  python3 -m agent.src.streaming_cli --video video.mp4 --window-size 30 --num-frames 16
        """,
    )

    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--gaze", default=None, help="Gaze CSV file path (optional)")
    parser.add_argument("--model", default="azure:gpt-4o", help="LLM model spec")
    parser.add_argument("--strategy", default="v2_google_gui",
                        choices=["v1_baseline", "v2_google_gui", "v3_with_visual", "v2_smart_glasses"],
                        help="Prompt strategy")
    parser.add_argument("--window-size", type=float, default=60.0, help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=30.0, help="Window stride in seconds")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames per window")
    parser.add_argument("--scene", default="general", help="Scene type")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from .streaming_pipeline import StreamingPipeline

    pipeline = StreamingPipeline(
        video_path=args.video,
        model_spec=args.model,
        prompt_strategy=args.strategy,
        gaze_csv_path=args.gaze,
        window_size=args.window_size,
        stride=args.stride,
        num_frames=args.num_frames,
        scene_name=args.scene,
        output_path=args.output,
    )

    timeline = pipeline.run()

    # Save output
    from pathlib import Path
    output_dir = Path(args.output) if args.output else Path(args.video).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ui_timeline.json"
    timeline.save(output_file)
    print(f"\n📁 Output saved to: {output_file}")


if __name__ == "__main__":
    main()
```

### Step 2: Verify CLI works

```bash
cd /home/v-tangxin/GUI && python3 -m agent.src.streaming_cli --help
```
Expected: Help text printed without errors

### Step 3: Commit

```bash
git add agent/src/streaming_cli.py
git commit -m "feat: add CLI for streaming pipeline"
```

---

## Task 6: Frontend Enhancement — 时间轴回放 UI 切换

**Files:**
- Modify: `agent/preview/templates/video_overlay.html`
- Modify: `agent/preview/server.py` (add API endpoint for timeline data)

**Why:** 前端需要支持加载 `ui_timeline.json`，在视频 `timeupdate` 时按时间轴匹配和切换 UI 组件，带平滑过渡动画。

### Step 1: Add timeline API endpoint to server.py

Add a new API route that serves timeline JSON data. Find the existing API handler section (near `/api/gaze`, `/api/rawdata`, `/api/models`) and add:

```python
# In the request handler, add a new route:
# GET /api/timeline?sample=<sample_path>
# Returns the ui_timeline.json for a given sample

elif self.path.startswith("/api/timeline"):
    query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
    sample = query.get("sample", [""])[0]

    # Look for ui_timeline.json in the sample's output directory
    timeline_path = output_path / sample / "ui_timeline.json"
    if timeline_path.exists():
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        with open(timeline_path, "r") as f:
            self.wfile.write(f.read().encode())
    else:
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Timeline not found"}).encode())
```

### Step 2: Add timeline playback to video_overlay.html

Add to the JavaScript section of `video_overlay.html`, after the existing `updateUI()` function:

```javascript
// ──── Timeline Playback Mode ────
let timelineData = null;
let currentTimelineIndex = -1;

async function loadTimeline() {
    try {
        const resp = await fetch(`/api/timeline?sample=${CONFIG.sample}`);
        if (resp.ok) {
            timelineData = await resp.json();
            console.log(`📊 Loaded timeline: ${timelineData.timeline.length} entries`);
            enableTimelineMode();
        }
    } catch (e) {
        console.log('No timeline data available, using single-UI mode');
    }
}

function enableTimelineMode() {
    // Add timeline indicator bar below video
    const container = document.querySelector('.video-container');
    const bar = document.createElement('div');
    bar.id = 'timeline-bar';
    bar.style.cssText = 'width:100%;height:24px;background:#1a1a2e;position:relative;border-radius:4px;margin-top:4px;overflow:hidden;';

    // Add markers for each timeline entry
    timelineData.timeline.forEach((entry, idx) => {
        const pct = (entry.time / timelineData.duration) * 100;
        const marker = document.createElement('div');
        marker.style.cssText = `position:absolute;left:${pct}%;top:0;width:3px;height:100%;background:${entry.trigger === 'event' ? '#00ff41' : '#4a9eff'};opacity:0.8;cursor:pointer;`;
        marker.title = `${entry.trigger} @ ${entry.time.toFixed(1)}s`;
        marker.onclick = () => { video.currentTime = entry.time; };
        bar.appendChild(marker);
    });
    container.appendChild(bar);
}

function updateTimelineUI() {
    if (!timelineData) return;

    const t = video.currentTime;
    const entries = timelineData.timeline;

    // Find the active entry: latest entry whose time <= currentTime
    let newIndex = -1;
    for (let i = entries.length - 1; i >= 0; i--) {
        if (entries[i].time <= t) {
            newIndex = i;
            break;
        }
    }

    if (newIndex === currentTimelineIndex) return; // No change
    currentTimelineIndex = newIndex;

    if (newIndex < 0) {
        uiOverlay.classList.add('hidden');
        return;
    }

    const entry = entries[newIndex];

    // Check if we're within display window (show for N seconds after trigger)
    const displayDuration = parseFloat(document.getElementById('ui-duration')?.value || '5');
    if (t > entry.time + displayDuration) {
        uiOverlay.classList.add('hidden');
        return;
    }

    // Render the new UI component with transition
    renderTimelineEntry(entry);
}

function renderTimelineEntry(entry) {
    // Fade out current UI
    uiOverlay.style.transition = 'opacity 0.3s ease';
    uiOverlay.style.opacity = '0';

    setTimeout(() => {
        // Update UI content via existing /api/ui or direct render
        // The entry.ui contains the full A2UI component JSON
        renderA2UIComponent(entry.ui, uiOverlay);

        // Position based on gaze if available
        if (entry.gaze) {
            const rect = video.getBoundingClientRect();
            uiOverlay.style.left = `${entry.gaze.x * rect.width}px`;
            uiOverlay.style.top = `${entry.gaze.y * rect.height}px`;
        }

        // Fade in
        uiOverlay.classList.remove('hidden');
        uiOverlay.style.opacity = '1';
    }, 300);
}

// Hook into existing timeupdate
const originalUpdateUI = updateUI;
function updateUI() {
    if (timelineData) {
        updateTimelineUI();
        updateGazePosition(video.currentTime); // Keep gaze indicator
    } else {
        originalUpdateUI();
    }
}

// Load timeline on page load (alongside existing data loading)
loadTimeline();
```

### Step 3: Add CSS transition styles

Add to the `<style>` section:

```css
#ui-overlay {
    transition: opacity 0.3s ease, transform 0.2s ease;
}
#ui-overlay.entering {
    animation: slideIn 0.4s ease forwards;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px) scale(0.95); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
```

### Step 4: Test manually

```bash
# Start preview server
cd /home/v-tangxin/GUI && ./ml_env/bin/python3 -m agent.preview.server --port 8000
# Open browser and verify timeline bar appears for samples with ui_timeline.json
```

### Step 5: Commit

```bash
git add agent/preview/templates/video_overlay.html agent/preview/server.py
git commit -m "feat: add timeline playback mode to frontend"
```

---

## Task 7: Integration Test — 端到端验证

**Files:**
- Create: `agent/tests/test_integration_streaming.py`

**Why:** 验证完整流程：视频 → 窗口分段 → 帧提取 → UI 生成 → timeline JSON → 可被前端加载。

### Step 1: Write integration test

```python
# agent/tests/test_integration_streaming.py
"""Integration test: video → timeline JSON end-to-end."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.src.streaming_pipeline import StreamingPipeline, UITimeline
from agent.src.video.segmenter import VideoSegmenter
from agent.src.video.gaze_analyzer import GazeAnalyzer, Fixation


class TestEndToEnd:
    def test_fixed_interval_produces_valid_timeline(self, tmp_path):
        """Fixed interval mode should produce a valid timeline JSON."""
        mock_component = {
            "type": "Card",
            "id": "test-card",
            "props": {"title": "Test", "subtitle": "Generated"},
            "children": [],
        }

        with (
            patch("agent.src.streaming_pipeline.FrameExtractor") as MockFE,
            patch("agent.src.streaming_pipeline.create_client") as MockLLM,
            patch("agent.src.streaming_pipeline.VisualContextGenerator") as MockVCG,
        ):
            # Mock frame extractor
            fe = MagicMock()
            fe.duration = 120.0
            fe.fps = 30.0
            fe.width = 1920
            fe.height = 1080
            fe.extract_frames.return_value = [MagicMock()]  # fake frame
            MockFE.return_value = fe

            # Mock visual context
            vcg = MagicMock()
            vcg.generate_context.return_value = MagicMock(to_dict=lambda: {"mode": "direct", "frames_base64": ["abc"]})
            MockVCG.return_value = vcg

            # Mock strategy
            strategy = MagicMock()
            strategy.generate.return_value = mock_component

            pipeline = StreamingPipeline(
                video_path=str(tmp_path / "test.mp4"),
                model_spec="azure:gpt-4o",
                prompt_strategy="v2_google_gui",
                window_size=60.0,
                stride=30.0,
                num_frames=8,
                output_path=str(tmp_path / "output"),
            )
            pipeline.strategy = strategy

            timeline = pipeline.run()

            # Verify structure
            assert isinstance(timeline, UITimeline)
            assert len(timeline.entries) > 0
            assert timeline.duration == 120.0

            # Verify JSON serialization
            d = timeline.to_dict()
            assert "timeline" in d
            assert all("time" in e and "ui" in e for e in d["timeline"])

            # Verify it saves correctly
            output_file = tmp_path / "output" / "ui_timeline.json"
            timeline.save(output_file)
            loaded = json.loads(output_file.read_text())
            assert loaded["duration"] == 120.0


class TestSegmenterGazeIntegration:
    def test_gaze_events_drive_windowing(self):
        """Gaze fixation changes should produce event-driven windows."""
        fixations = [
            Fixation(0.0, 2.0, 0.3, 0.3, 2.0),
            Fixation(2.5, 5.0, 0.7, 0.7, 2.5),
            Fixation(6.0, 10.0, 0.5, 0.5, 4.0),
        ]
        analyzer = GazeAnalyzer()
        events = analyzer.detect_events(fixations)

        segmenter = VideoSegmenter(window_size=60.0)
        event_times = [e.time for e in events]
        windows = segmenter.generate_windows(video_duration=30.0, event_times=event_times)

        assert len(windows) == len(events)
        for w, e in zip(windows, events):
            assert w.trigger_time == e.time
            assert w.trigger_type == "event"
```

### Step 2: Run integration test

```bash
cd /home/v-tangxin/GUI && python3 -m pytest agent/tests/test_integration_streaming.py -v
```
Expected: All PASS

### Step 3: Commit

```bash
git add agent/tests/test_integration_streaming.py
git commit -m "test: add integration tests for streaming pipeline"
```

---

## Summary: File Inventory

| Action | File | Description |
|--------|------|-------------|
| **Create** | `agent/src/video/segmenter.py` | 滑动窗口视频分段器 |
| **Create** | `agent/src/video/gaze_analyzer.py` | I-VT 注视点提取 + 事件检测 |
| **Create** | `agent/src/streaming_pipeline.py` | 核心流式 Pipeline 编排器 |
| **Create** | `agent/src/streaming_cli.py` | CLI 入口 |
| **Modify** | `agent/src/video/extractor.py` | 添加 `compute_sample_timestamps()` + `resize_frame()` |
| **Modify** | `agent/preview/server.py` | 添加 `/api/timeline` endpoint |
| **Modify** | `agent/preview/templates/video_overlay.html` | 时间轴回放模式 |
| **Create** | `agent/tests/test_segmenter.py` | VideoSegmenter 测试 |
| **Create** | `agent/tests/test_gaze_analyzer.py` | GazeAnalyzer 测试 |
| **Create** | `agent/tests/test_extractor_enhanced.py` | FrameExtractor 增强测试 |
| **Create** | `agent/tests/test_streaming_pipeline.py` | StreamingPipeline 测试 |
| **Create** | `agent/tests/test_integration_streaming.py` | 端到端集成测试 |

## Execution Order

```
Task 1 (VideoSegmenter) ─┐
                          ├──► Task 4 (StreamingPipeline) ──► Task 5 (CLI) ──► Task 7 (Integration)
Task 2 (GazeAnalyzer) ───┘                                                          │
                                                                                     ▼
Task 3 (FrameExtractor) ──────────────────────────────────────────────────► Task 6 (Frontend)
```

Tasks 1, 2, 3 are independent and can be parallelized.
Task 4 depends on 1+2+3.
Task 5 depends on 4.
Task 6 is independent (frontend only).
Task 7 depends on all.
