"""Frame processor — wraps existing FrameExtractor for iterative access."""

import base64
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from agent.src.video.extractor import FrameExtractor

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Iterates over video frames at a fixed interval.

    Wraps the existing FrameExtractor to provide an iterator interface
    suitable for the Observe-Think-Act loop.
    """

    def __init__(self, video_path: str | Path, frame_interval_sec: float = 0.5, jpeg_quality: int = 80):
        self.video_path = Path(video_path)
        self.frame_interval_sec = frame_interval_sec
        self.jpeg_quality = jpeg_quality
        self._extractor = FrameExtractor(self.video_path)

    @property
    def duration(self) -> float:
        return self._extractor.duration

    @property
    def fps(self) -> float:
        return self._extractor.fps

    def iter_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (timestamp, frame_ndarray) at fixed intervals.

        Yields:
            Tuple of (timestamp_seconds, BGR numpy array).
        """
        t = 0.0
        while t < self._extractor.duration:
            frame = self._extractor._extract_frame_at_time(t)
            if frame is not None:
                yield t, frame
            else:
                logger.warning(f"Failed to extract frame at {t:.2f}s")
            t += self.frame_interval_sec

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode a single frame to base64 JPEG string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise RuntimeError("Failed to encode frame to JPEG")
        return base64.b64encode(buffer).decode("utf-8")

    def close(self):
        """Release video resources."""
        if hasattr(self._extractor, "cap") and self._extractor.cap is not None:
            self._extractor.cap.release()
