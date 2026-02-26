"""
Stream Simulator

Simulates real-time streaming over EGTEA Gaze+ cooking sessions.
Reconstructs a session timeline from individual action clips and
yields frames in temporal order at a configurable FPS.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from data.egtea_loader import ActionClip, CookingSession, EGTEALoader
from src.streaming.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


@dataclass
class StreamFrame:
    """A single frame from the simulated stream."""
    timestamp: float            # absolute session time (seconds)
    frame: np.ndarray           # BGR image
    current_action: Optional[ActionClip] = None
    next_action: Optional[ActionClip] = None
    is_idle: bool = False       # True if between actions
    clip_path: Optional[Path] = None


class StreamSimulator:
    """
    Simulate real-time streaming for a cooking session.

    Iterates through all clips in a session in temporal order,
    extracting frames at `sample_fps` and yielding StreamFrame objects.
    """

    def __init__(
        self,
        loader: EGTEALoader,
        sample_fps: float = 2.0,
        resolution: tuple[int, int] = (640, 480),
        max_duration_sec: Optional[float] = None,
    ):
        self.loader = loader
        self.sample_fps = sample_fps
        self.processor = FrameProcessor(target_fps=sample_fps, resolution=resolution)
        self.max_duration_sec = max_duration_sec

    def stream_session(self, session_id: str) -> Iterator[StreamFrame]:
        """
        Stream frames from a cooking session in temporal order.

        Clips are played in sequence. Gaps between clips are skipped
        (no frames yielded during idle periods, but is_idle is marked
        on the first frame after a gap).
        """
        session = self.loader.get_session(session_id)
        clip_pairs = self.loader.get_session_clips_sorted(session_id)

        if not clip_pairs:
            logger.warning(f"No video clips found for session {session_id}")
            return

        prev_end_sec = 0.0
        for clip, clip_path in clip_pairs:
            if self.max_duration_sec and clip.start_sec > self.max_duration_sec:
                break

            is_after_gap = (clip.start_sec - prev_end_sec) > 2.0
            next_action = session.get_next_action(clip.end_sec)

            try:
                frames = self.processor.extract_frames(clip_path)
            except Exception as e:
                logger.warning(f"Failed to read {clip_path}: {e}")
                continue

            for i, fdata in enumerate(frames):
                # Map relative frame time to absolute session time
                relative_time = fdata["timestamp"]
                absolute_time = clip.start_sec + relative_time

                if self.max_duration_sec and absolute_time > self.max_duration_sec:
                    return

                yield StreamFrame(
                    timestamp=absolute_time,
                    frame=fdata["frame"],
                    current_action=clip,
                    next_action=next_action,
                    is_idle=(i == 0 and is_after_gap),
                    clip_path=clip_path,
                )

            prev_end_sec = clip.end_sec

    def stream_session_windowed(
        self,
        session_id: str,
        window_sec: float = 60.0,
    ) -> Iterator[tuple[StreamFrame, list[StreamFrame]]]:
        """
        Stream with a sliding window of recent frames.

        Yields (current_frame, window_frames) where window_frames
        contains the last `window_sec` seconds of frames.
        """
        window: list[StreamFrame] = []

        for frame in self.stream_session(session_id):
            # Remove frames outside the window
            cutoff = frame.timestamp - window_sec
            window = [f for f in window if f.timestamp >= cutoff]
            window.append(frame)

            yield frame, list(window)

    def get_session_info(self, session_id: str) -> dict:
        """Get metadata about a session for logging."""
        session = self.loader.get_session(session_id)
        clips = self.loader.get_session_clips_sorted(session_id)
        return {
            "session_id": session_id,
            "recipe": session.recipe,
            "participant": session.participant,
            "num_actions": session.num_actions,
            "duration_sec": session.duration_sec,
            "num_clips_with_video": len(clips),
            "gaps": session.get_gaps(),
        }
