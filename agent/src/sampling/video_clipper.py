"""Video clipping module for extracting video segments and keyframes."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoClipper:
    """Extract video clips and keyframes from Ego-Dataset recordings."""

    def __init__(self, video_path: Path):
        """Initialize video clipper.

        Args:
            video_path: Path to the source video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Get video metadata
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Opened video: {self.video_path.name}, "
            f"FPS: {self.fps:.2f}, Duration: {self.duration:.2f}s, "
            f"Resolution: {self.width}x{self.height}"
        )

    def __del__(self):
        """Release video capture resources."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

    def clip_video(
        self,
        start_time: float,
        end_time: float,
        output_path: Path,
        codec: str = "libx264",
        crf: int = 23,
    ) -> Optional[Path]:
        """Extract a video clip using FFmpeg for lossless cutting.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save the output clip
            codec: Video codec (default: libx264)
            crf: Constant Rate Factor for quality (0-51, lower is better)

        Returns:
            Path to the saved clip, or None if failed
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clamp times to video duration
        start_time = max(0, start_time)
        end_time = min(end_time, self.duration)

        if start_time >= end_time:
            logger.error(f"Invalid time range: {start_time} >= {end_time}")
            return None

        duration = end_time - start_time

        # Use FFmpeg for efficient clipping
        # -ss before -i for fast seeking, -t for duration
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),
            "-i", str(self.video_path),
            "-t", str(duration),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", "fast",
            "-c:a", "aac",  # Audio codec
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None

            logger.info(f"Created video clip: {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out")
            return None
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            return None
        except Exception as e:
            logger.error(f"Error clipping video: {e}")
            return None

    def extract_keyframes(
        self,
        start_time: float,
        end_time: float,
        num_frames: int = 3,
    ) -> list[np.ndarray]:
        """Extract equidistant keyframes from a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            num_frames: Number of frames to extract (default: 3)

        Returns:
            List of frames as numpy arrays (BGR format)
        """
        # Clamp times
        start_time = max(0, start_time)
        end_time = min(end_time, self.duration)

        if start_time >= end_time:
            logger.error(f"Invalid time range: {start_time} >= {end_time}")
            return []

        # Calculate equidistant timestamps
        # For num_frames=3: at 1/4, 2/4, 3/4 of the interval
        interval = (end_time - start_time) / (num_frames + 1)
        timestamps = [start_time + interval * (i + 1) for i in range(num_frames)]

        frames = []
        for ts in timestamps:
            frame = self._extract_frame_at_time(ts)
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Failed to extract frame at timestamp {ts}")

        logger.info(f"Extracted {len(frames)}/{num_frames} keyframes")
        return frames

    def _extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract a single frame at the specified timestamp.

        Args:
            timestamp: Time in seconds from video start

        Returns:
            Frame as numpy array (BGR format), or None if failed
        """
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            logger.error(f"Failed to read frame at position {frame_number}")
            return None

        return frame

    def save_keyframes(
        self,
        start_time: float,
        end_time: float,
        output_dir: Path,
        num_frames: int = 3,
        format: str = "jpg",
        quality: int = 95,
    ) -> list[Path]:
        """Extract and save keyframes to disk.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save frames
            num_frames: Number of frames to extract
            format: Image format (jpg or png)
            quality: JPEG quality (1-100)

        Returns:
            List of paths to saved frame files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = self.extract_keyframes(start_time, end_time, num_frames)

        saved_paths = []
        for i, frame in enumerate(frames):
            filename = f"frame_{i + 1}.{format}"
            filepath = output_dir / filename

            if format.lower() == "jpg":
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                cv2.imwrite(str(filepath), frame, encode_params)
            else:
                cv2.imwrite(str(filepath), frame)

            saved_paths.append(filepath)
            logger.debug(f"Saved frame to {filepath}")

        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths

    def generate_clip_and_frames(
        self,
        start_time: float,
        end_time: float,
        output_dir: Path,
        num_frames: int = 3,
    ) -> dict:
        """Generate both video clip and keyframes.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save outputs
            num_frames: Number of keyframes to extract

        Returns:
            Dictionary with clip_path and frame_paths
        """
        output_dir = Path(output_dir)
        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Generate video clip
        clip_path = self.clip_video(
            start_time,
            end_time,
            video_dir / "clip.mp4",
        )

        # Generate keyframes
        frame_paths = self.save_keyframes(
            start_time,
            end_time,
            video_dir,
            num_frames=num_frames,
        )

        return {
            "clip_path": clip_path,
            "frame_paths": frame_paths,
        }
