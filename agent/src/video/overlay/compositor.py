"""Video frame compositing with UI overlays.

Handles alpha-blending of UI images onto video frames
and encoding the result to output video files.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Composite UI overlays onto video frames.

    Uses OpenCV for frame manipulation and FFmpeg for encoding.

    Attributes:
        video_path: Path to the source video.
        output_path: Path for the output video.
        fps: Video frame rate.
        width: Video frame width.
        height: Video frame height.
        total_frames: Total number of frames.
    """

    def __init__(self, video_path: Path | str, output_path: Path | str):
        """Initialize the video compositor.

        Args:
            video_path: Path to the source video file.
            output_path: Path for the output video file.

        Raises:
            FileNotFoundError: If the source video doesn't exist.
            ValueError: If the video cannot be opened.
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open video to get properties
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        cap.release()

        logger.info(
            f"Video: {self.video_path.name}, "
            f"{self.width}x{self.height} @ {self.fps:.2f}fps, "
            f"{self.duration:.2f}s ({self.total_frames} frames)"
        )

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def time_to_frame(self, timestamp: float) -> int:
        """Convert a timestamp to frame number.

        Args:
            timestamp: Time in seconds.

        Returns:
            Frame number (0-indexed).
        """
        frame = int(timestamp * self.fps)
        return max(0, min(frame, self.total_frames - 1))

    def composite_overlay(
        self,
        ui_image: np.ndarray,
        anchor: tuple[float, float],
        start_frame: int,
        end_frame: int,
        anchor_position: str = "center",
    ) -> Path:
        """Composite a UI overlay onto the video.

        Args:
            ui_image: RGBA numpy array of the UI to overlay.
            anchor: (x, y) pixel coordinates for the anchor point.
            start_frame: First frame to show the overlay.
            end_frame: Last frame to show the overlay (inclusive).
            anchor_position: Where to position the UI relative to anchor:
                - "center": Center the UI on the anchor point
                - "top_left": Place top-left corner at anchor
                - "bottom_center": Place bottom-center at anchor

        Returns:
            Path to the output video file.
        """
        # Open source video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        # Set up video writer using a temporary file
        temp_output = self.output_path.with_suffix(".temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_output),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video: {temp_output}")

        # Pre-process UI image
        ui_height, ui_width = ui_image.shape[:2]

        # Calculate position based on anchor
        anchor_x, anchor_y = anchor
        if anchor_position == "center":
            x_offset = int(anchor_x - ui_width / 2)
            y_offset = int(anchor_y - ui_height / 2)
        elif anchor_position == "top_left":
            x_offset = int(anchor_x)
            y_offset = int(anchor_y)
        elif anchor_position == "bottom_center":
            x_offset = int(anchor_x - ui_width / 2)
            y_offset = int(anchor_y - ui_height)
        else:
            x_offset = int(anchor_x - ui_width / 2)
            y_offset = int(anchor_y - ui_height / 2)

        # Clamp to frame bounds
        x_offset = max(0, min(x_offset, self.width - ui_width))
        y_offset = max(0, min(y_offset, self.height - ui_height))

        logger.info(
            f"Compositing UI ({ui_width}x{ui_height}) at ({x_offset}, {y_offset}) "
            f"for frames {start_frame}-{end_frame}"
        )

        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply overlay for the specified frame range
            if start_frame <= frame_idx <= end_frame:
                frame = self._blend_overlay(
                    frame, ui_image, x_offset, y_offset
                )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.debug(f"Processed {frame_idx}/{self.total_frames} frames")

        cap.release()
        writer.release()

        # Re-encode with FFmpeg for better compatibility
        self._reencode_video(temp_output, self.output_path)

        # Clean up temp file
        if temp_output.exists():
            temp_output.unlink()

        logger.info(f"Output video saved: {self.output_path}")
        return self.output_path

    def _blend_overlay(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Alpha-blend an overlay onto a frame.

        Args:
            frame: BGR video frame.
            overlay: RGBA overlay image.
            x: X position for overlay placement.
            y: Y position for overlay placement.

        Returns:
            Blended frame.
        """
        h, w = overlay.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        # Calculate the visible region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

        # Calculate overlay crop region
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return frame  # No overlap

        # Extract regions
        overlay_region = overlay[oy1:oy2, ox1:ox2]
        frame_region = frame[y1:y2, x1:x2]

        # Handle RGBA overlay
        if overlay_region.shape[2] == 4:
            # Extract alpha channel and normalize to 0-1
            alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
            overlay_rgb = overlay_region[:, :, :3]

            # Convert overlay from RGBA to BGR
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

            # Alpha blending
            blended = (
                alpha * overlay_bgr.astype(np.float32) +
                (1 - alpha) * frame_region.astype(np.float32)
            ).astype(np.uint8)

            frame[y1:y2, x1:x2] = blended
        else:
            # No alpha, just copy
            overlay_bgr = cv2.cvtColor(overlay_region, cv2.COLOR_RGB2BGR)
            frame[y1:y2, x1:x2] = overlay_bgr

        return frame

    def _reencode_video(self, input_path: Path, output_path: Path) -> None:
        """Re-encode video with FFmpeg for better compatibility.

        Args:
            input_path: Path to the input video.
            output_path: Path for the output video.
        """
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"FFmpeg encoding complete: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg encoding failed: {e.stderr}")
            # Fall back to the opencv output
            if input_path.exists():
                import shutil
                shutil.move(str(input_path), str(output_path))
        except FileNotFoundError:
            logger.warning("FFmpeg not found, using OpenCV output directly")
            if input_path.exists():
                import shutil
                shutil.move(str(input_path), str(output_path))

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video.

        Args:
            frame_number: Frame number (0-indexed).

        Returns:
            BGR numpy array or None if failed.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
