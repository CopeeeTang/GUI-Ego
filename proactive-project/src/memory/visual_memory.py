"""
Layer 3: Visual Working Memory

Frame-level buffer with sliding window for recent frames
and compressed representation for historical frames.

Inspired by StreamBridge's round-decayed compression.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """A stored frame with metadata."""
    timestamp: float
    frame: np.ndarray           # BGR image (full resolution)
    action_label: Optional[str] = None
    is_compressed: bool = False


@dataclass
class CompressedFrameGroup:
    """A group of compressed historical frames."""
    start_time: float
    end_time: float
    num_frames: int
    representative_frame: np.ndarray    # mean or sampled frame
    action_labels: list[str]


class VisualMemory:
    """
    Layer 3 memory: visual working memory with sliding window.

    Recent frames (last N) are stored at full resolution.
    Older frames are compressed into groups via mean-pooling
    or representative sampling.
    """

    def __init__(
        self,
        recent_capacity: int = 16,
        compressed_capacity: int = 64,
        compression_method: str = "mean_pool",  # mean_pool | sample
        compression_group_size: int = 4,
    ):
        self.recent_capacity = recent_capacity
        self.compressed_capacity = compressed_capacity
        self.compression_method = compression_method
        self.compression_group_size = compression_group_size

        self.recent_frames: deque[FrameRecord] = deque(maxlen=recent_capacity)
        self.compressed_groups: list[CompressedFrameGroup] = []

    def add_frame(self, timestamp: float, frame: np.ndarray,
                  action_label: Optional[str] = None):
        """Add a new frame to visual memory."""
        record = FrameRecord(
            timestamp=timestamp,
            frame=frame.copy(),
            action_label=action_label,
        )

        # If recent buffer is full, compress the oldest frames
        if len(self.recent_frames) >= self.recent_capacity:
            self._compress_oldest()

        self.recent_frames.append(record)

    def _compress_oldest(self):
        """Compress oldest recent frames into a compressed group."""
        if len(self.recent_frames) < self.compression_group_size:
            return

        # Take oldest frames for compression
        to_compress = []
        for _ in range(self.compression_group_size):
            if self.recent_frames:
                to_compress.append(self.recent_frames.popleft())

        if not to_compress:
            return

        # Compress
        if self.compression_method == "mean_pool":
            frames_array = np.stack([f.frame for f in to_compress])
            representative = frames_array.mean(axis=0).astype(np.uint8)
        else:  # sample — take middle frame
            mid = len(to_compress) // 2
            representative = to_compress[mid].frame

        action_labels = [f.action_label for f in to_compress if f.action_label]

        group = CompressedFrameGroup(
            start_time=to_compress[0].timestamp,
            end_time=to_compress[-1].timestamp,
            num_frames=len(to_compress),
            representative_frame=representative,
            action_labels=list(set(action_labels)),
        )
        self.compressed_groups.append(group)

        # Evict oldest compressed groups if over capacity
        if len(self.compressed_groups) > self.compressed_capacity:
            self.compressed_groups = self.compressed_groups[-self.compressed_capacity:]

    def get_recent_frames(self, n: Optional[int] = None) -> list[FrameRecord]:
        """Get N most recent frames."""
        frames = list(self.recent_frames)
        if n is not None:
            frames = frames[-n:]
        return frames

    def get_recent_pil_images(self, n: Optional[int] = None) -> list[Image.Image]:
        """Get recent frames as PIL images."""
        import cv2
        frames = self.get_recent_frames(n)
        return [Image.fromarray(cv2.cvtColor(f.frame, cv2.COLOR_BGR2RGB)) for f in frames]

    def get_frame_at_time(self, target_sec: float,
                          tolerance_sec: float = 1.0) -> Optional[FrameRecord]:
        """Find the frame closest to a target timestamp."""
        best = None
        best_dist = float("inf")

        # Search recent frames
        for f in self.recent_frames:
            dist = abs(f.timestamp - target_sec)
            if dist < best_dist:
                best = f
                best_dist = dist

        # Search compressed groups
        for group in self.compressed_groups:
            mid_time = (group.start_time + group.end_time) / 2
            dist = abs(mid_time - target_sec)
            if dist < best_dist:
                best = FrameRecord(
                    timestamp=mid_time,
                    frame=group.representative_frame,
                    action_label=group.action_labels[0] if group.action_labels else None,
                    is_compressed=True,
                )
                best_dist = dist

        if best and best_dist <= tolerance_sec:
            return best
        return best  # return best even if outside tolerance

    def get_frames_in_window(self, start_sec: float, end_sec: float) -> list[FrameRecord]:
        """Get all frames (recent + compressed) within a time window."""
        result = []

        for f in self.recent_frames:
            if start_sec <= f.timestamp <= end_sec:
                result.append(f)

        for group in self.compressed_groups:
            mid = (group.start_time + group.end_time) / 2
            if start_sec <= mid <= end_sec:
                result.append(FrameRecord(
                    timestamp=mid,
                    frame=group.representative_frame,
                    action_label=group.action_labels[0] if group.action_labels else None,
                    is_compressed=True,
                ))

        result.sort(key=lambda f: f.timestamp)
        return result

    @property
    def total_frames(self) -> int:
        return len(self.recent_frames) + sum(g.num_frames for g in self.compressed_groups)

    @property
    def time_span(self) -> tuple[float, float]:
        """Get earliest and latest timestamps in memory."""
        times = [f.timestamp for f in self.recent_frames]
        for g in self.compressed_groups:
            times.extend([g.start_time, g.end_time])
        if not times:
            return (0.0, 0.0)
        return (min(times), max(times))

    def summary(self) -> str:
        start, end = self.time_span
        return (
            f"[Visual Memory] "
            f"Recent: {len(self.recent_frames)}/{self.recent_capacity} frames, "
            f"Compressed: {len(self.compressed_groups)} groups "
            f"({sum(g.num_frames for g in self.compressed_groups)} frames), "
            f"Span: {start:.1f}s - {end:.1f}s"
        )

    def reset(self):
        self.recent_frames.clear()
        self.compressed_groups.clear()
