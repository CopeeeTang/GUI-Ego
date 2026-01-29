"""Video frame extraction module for extracting frames at specified timestamps."""

import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files at specified timestamps.

    This class handles video frame extraction for smart glasses recordings,
    supporting both merged videos and segmented video files.

    Attributes:
        video_path: Path to the video file.
        cap: OpenCV VideoCapture object.
        fps: Frames per second of the video.
        total_frames: Total number of frames in the video.
        duration: Total duration of the video in seconds.
    """

    def __init__(self, video_path: str | Path):
        """Initialize the frame extractor with a video file.

        Args:
            video_path: Path to the video file. Supports:
                - Merged videos from "Timeseries Data + Scene Video/" directory
                - Segmented videos from "raw_data/" directory

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened.
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        # Extract video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Get frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Opened video: {self.video_path.name}, "
            f"FPS: {self.fps:.2f}, Duration: {self.duration:.2f}s, "
            f"Resolution: {self.width}x{self.height}"
        )

    def __del__(self):
        """Release video capture resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def extract_frames(
        self,
        start_time: float,
        end_time: float,
        num_frames: int = 3,
    ) -> list[np.ndarray]:
        """Extract equidistant frames from a time range.

        Extracts frames at equal intervals within the specified time range.
        For num_frames=3, extracts at: start + interval, start + 2*interval, start + 3*interval
        where interval = (end_time - start_time) / (num_frames + 1)

        Args:
            start_time: Start timestamp in seconds (from video beginning).
            end_time: End timestamp in seconds (from video beginning).
            num_frames: Number of frames to extract (default: 3).

        Returns:
            List of numpy arrays containing the extracted frames in BGR format.

        Raises:
            ValueError: If time range is invalid or outside video duration.
        """
        if start_time < 0:
            logger.warning(f"start_time {start_time} < 0, clamping to 0")
            start_time = 0

        if end_time > self.duration:
            logger.warning(
                f"end_time {end_time} > video duration {self.duration}, "
                f"clamping to {self.duration}"
            )
            end_time = self.duration

        if start_time >= end_time:
            raise ValueError(
                f"Invalid time range: start_time ({start_time}) >= end_time ({end_time})"
            )

        # Calculate equidistant timestamps
        interval = (end_time - start_time) / (num_frames + 1)
        timestamps = [start_time + interval * (i + 1) for i in range(num_frames)]

        logger.debug(f"Extracting frames at timestamps: {timestamps}")

        frames = []
        for ts in timestamps:
            frame = self._extract_frame_at_time(ts)
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Failed to extract frame at timestamp {ts}")

        logger.info(f"Extracted {len(frames)}/{num_frames} frames")
        return frames

    def _extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract a single frame at the specified timestamp.

        Args:
            timestamp: Time in seconds from video start.

        Returns:
            Frame as numpy array (BGR format), or None if extraction fails.
        """
        # Convert timestamp to frame number
        frame_number = int(timestamp * self.fps)

        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = self.cap.read()

        if not ret:
            logger.error(f"Failed to read frame at position {frame_number}")
            return None

        return frame

    def frames_to_base64(
        self,
        frames: list[np.ndarray],
        format: str = "jpeg",
        quality: int = 85,
    ) -> list[str]:
        """Convert frames to base64-encoded strings.

        Args:
            frames: List of frames as numpy arrays (BGR format).
            format: Output image format ("jpeg" or "png").
            quality: JPEG quality (1-100, only used for JPEG format).

        Returns:
            List of base64-encoded image strings.
        """
        base64_frames = []

        for i, frame in enumerate(frames):
            try:
                # Convert BGR to RGB for proper color representation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Encode frame
                if format.lower() == "jpeg":
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode(".jpg", frame_rgb, encode_params)
                else:
                    success, buffer = cv2.imencode(".png", frame_rgb)

                if success:
                    b64_string = base64.b64encode(buffer).decode("utf-8")
                    base64_frames.append(b64_string)
                else:
                    logger.error(f"Failed to encode frame {i}")

            except Exception as e:
                logger.error(f"Error encoding frame {i}: {e}")

        logger.info(f"Converted {len(base64_frames)} frames to base64")
        return base64_frames

    def frames_to_data_urls(
        self,
        frames: list[np.ndarray],
        format: str = "jpeg",
        quality: int = 85,
    ) -> list[str]:
        """Convert frames to data URLs for direct use in HTML/APIs.

        Args:
            frames: List of frames as numpy arrays (BGR format).
            format: Output image format ("jpeg" or "png").
            quality: JPEG quality (1-100, only used for JPEG format).

        Returns:
            List of data URL strings (e.g., "data:image/jpeg;base64,...").
        """
        base64_frames = self.frames_to_base64(frames, format, quality)
        mime_type = "image/jpeg" if format.lower() == "jpeg" else "image/png"
        return [f"data:{mime_type};base64,{b64}" for b64 in base64_frames]

    def save_frames(
        self,
        frames: list[np.ndarray],
        output_dir: str | Path,
        prefix: str = "frame",
        format: str = "jpg",
    ) -> list[Path]:
        """Save extracted frames to disk.

        Args:
            frames: List of frames as numpy arrays.
            output_dir: Directory to save frames.
            prefix: Filename prefix for saved frames.
            format: Output image format.

        Returns:
            List of paths to saved frame files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, frame in enumerate(frames):
            filepath = output_dir / f"{prefix}_{i:03d}.{format}"
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(filepath)
            logger.debug(f"Saved frame to {filepath}")

        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths

    @classmethod
    def find_video_for_participant(
        cls,
        data_root: str | Path,
        participant: str,
        session_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Find the appropriate video file for a participant.

        Searches for video files in the following priority:
        1. Merged video in "Timeseries Data + Scene Video/" directory
        2. Segmented videos in "raw_data/" directory

        Args:
            data_root: Root directory of the dataset.
            participant: Participant ID (e.g., "P1_YuePan").
            session_id: Optional session ID to filter by.

        Returns:
            Path to the video file, or None if not found.
        """
        data_root = Path(data_root)
        participant_dir = data_root / participant

        if not participant_dir.exists():
            logger.error(f"Participant directory not found: {participant_dir}")
            return None

        # Priority 1: Merged video in "Timeseries Data + Scene Video/"
        timeseries_dir = participant_dir / "Timeseries Data + Scene Video"
        if timeseries_dir.exists():
            # Find session directories
            session_dirs = list(timeseries_dir.glob("*"))
            if session_id:
                session_dirs = [d for d in session_dirs if session_id in d.name]

            for session_dir in session_dirs:
                if session_dir.is_dir():
                    # Find MP4 files
                    mp4_files = list(session_dir.glob("*.mp4"))
                    if mp4_files:
                        # Prefer the merged video (usually named with session ID)
                        logger.info(f"Found merged video: {mp4_files[0]}")
                        return mp4_files[0]

        # Priority 2: Segmented videos in "raw_data/"
        raw_data_dir = participant_dir / "raw_data"
        if raw_data_dir.exists():
            # Find session directories
            for session_dir in raw_data_dir.iterdir():
                if session_dir.is_dir():
                    # Find scene camera videos
                    scene_videos = list(session_dir.glob("Neon Scene Camera v1 ps*.mp4"))
                    if scene_videos:
                        # Return first segment (caller should handle concatenation)
                        logger.info(f"Found segmented video: {scene_videos[0]}")
                        return scene_videos[0]

        logger.warning(f"No video found for participant: {participant}")
        return None


class SegmentedVideoExtractor(FrameExtractor):
    """Handle segmented video files by concatenating them virtually.

    Some recordings are split into multiple segment files. This class
    handles the mapping of global timestamps to segment-local timestamps.
    """

    def __init__(self, segment_paths: list[str | Path]):
        """Initialize with multiple video segment paths.

        Args:
            segment_paths: List of paths to video segments in order.
        """
        self.segment_paths = [Path(p) for p in segment_paths]
        self.segments: list[dict] = []

        cumulative_duration = 0.0
        for path in self.segment_paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open segment: {path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            self.segments.append({
                "path": path,
                "start_time": cumulative_duration,
                "end_time": cumulative_duration + duration,
                "fps": fps,
                "total_frames": total_frames,
                "duration": duration,
            })

            cumulative_duration += duration
            cap.release()

        self.duration = cumulative_duration
        self.fps = self.segments[0]["fps"] if self.segments else 30.0

        # Use first segment for base properties
        if self.segment_paths:
            super().__init__(self.segment_paths[0])

        logger.info(
            f"Initialized segmented video with {len(self.segments)} segments, "
            f"total duration: {self.duration:.2f}s"
        )

    def _extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract frame from the appropriate segment."""
        # Find the segment containing this timestamp
        for segment in self.segments:
            if segment["start_time"] <= timestamp < segment["end_time"]:
                # Calculate local timestamp within segment
                local_timestamp = timestamp - segment["start_time"]

                # Open segment and extract frame
                cap = cv2.VideoCapture(str(segment["path"]))
                if not cap.isOpened():
                    logger.error(f"Cannot open segment: {segment['path']}")
                    return None

                frame_number = int(local_timestamp * segment["fps"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    logger.error(f"Failed to read frame from segment")
                    return None

                return frame

        logger.error(f"Timestamp {timestamp} not found in any segment")
        return None
