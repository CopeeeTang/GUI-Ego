"""
Frame Processor

Extracts frames from video clips at specified FPS,
encodes them for VLM input (base64 / PIL Image / tensor).
"""

import base64
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class FrameProcessor:
    """Extract and process frames from video files."""

    def __init__(self, target_fps: float = 2.0, resolution: tuple[int, int] = (640, 480)):
        self.target_fps = target_fps
        self.resolution = resolution  # (width, height)

    def extract_frames(self, video_path: str | Path,
                       start_sec: float = 0.0,
                       end_sec: Optional[float] = None) -> list[dict]:
        """
        Extract frames from video at target FPS.

        Returns list of dicts:
            {"timestamp": float, "frame": np.ndarray (BGR), "index": int}
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / native_fps if native_fps > 0 else 0

        if end_sec is None or end_sec > duration:
            end_sec = duration

        # Calculate which frames to grab
        interval = native_fps / self.target_fps
        start_frame = int(start_sec * native_fps)
        end_frame = int(end_sec * native_fps)

        frames = []
        frame_idx = start_frame
        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                break

            if (frame.shape[1], frame.shape[0]) != self.resolution:
                frame = cv2.resize(frame, self.resolution)

            timestamp = frame_idx / native_fps
            frames.append({
                "timestamp": timestamp,
                "frame": frame,
                "index": len(frames),
            })
            frame_idx += interval

        cap.release()
        return frames

    def extract_single_frame(self, video_path: str | Path,
                             time_sec: float) -> Optional[np.ndarray]:
        """Extract a single frame at given timestamp."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
        ret, frame = cap.read()
        cap.release()
        if ret and (frame.shape[1], frame.shape[0]) != self.resolution:
            frame = cv2.resize(frame, self.resolution)
        return frame if ret else None

    @staticmethod
    def frame_to_pil(frame: np.ndarray) -> Image.Image:
        """Convert BGR numpy frame to PIL Image."""
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def frame_to_base64(frame: np.ndarray, fmt: str = "jpeg") -> str:
        """Convert frame to base64 string."""
        if fmt == "jpeg":
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            _, buf = cv2.imencode(".png", frame)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
        """Convert PIL image to base64 string."""
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
