"""Video overlay processor - main orchestration module.

Coordinates the full pipeline of:
1. Loading sample data and UI JSON
2. Extracting gaze anchor points
3. Rendering UI to images
4. Compositing overlays onto video
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .gaze_anchor import GazeAnchorExtractor
from .ui_renderer import UIRenderer, PLAYWRIGHT_AVAILABLE
from .compositor import VideoCompositor

logger = logging.getLogger(__name__)


class VideoOverlayProcessor:
    """Process video samples with UI overlays.

    Orchestrates the full pipeline from sample data to output video
    with UI overlays positioned at gaze anchor points.

    Attributes:
        preview_server_url: URL of the preview server for UI rendering.
        overlay_duration: Duration in seconds to show the UI overlay.
        default_viewport: Default browser viewport size.
    """

    def __init__(
        self,
        preview_server_url: str = "http://localhost:8080",
        overlay_duration: float = 2.0,
        viewport_size: tuple[int, int] = (1280, 720),
    ):
        """Initialize the overlay processor.

        Args:
            preview_server_url: URL of the A2UI preview server.
            overlay_duration: How long to display the UI overlay (seconds).
            viewport_size: Browser viewport dimensions for UI rendering.
        """
        self.preview_server_url = preview_server_url
        self.overlay_duration = overlay_duration
        self.viewport_size = viewport_size

    def process_sample(
        self,
        sample_dir: Path | str,
        ui_json_path: Path | str,
        output_dir: Path | str,
        strategy_name: Optional[str] = None,
    ) -> Optional[Path]:
        """Process a single sample and generate overlay video.

        Args:
            sample_dir: Path to the sample directory (contains rawdata.json, video/, signals/).
            ui_json_path: Path to the UI JSON file to overlay.
            output_dir: Base output directory for generated videos.
            strategy_name: Optional strategy name for output filename.

        Returns:
            Path to the generated overlay video, or None if processing failed.
        """
        sample_dir = Path(sample_dir)
        ui_json_path = Path(ui_json_path)
        output_dir = Path(output_dir)

        # Validate inputs
        if not sample_dir.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return None

        if not ui_json_path.exists():
            logger.error(f"UI JSON not found: {ui_json_path}")
            return None

        # Load rawdata.json for time interval
        rawdata_path = sample_dir / "rawdata.json"
        if not rawdata_path.exists():
            logger.error(f"rawdata.json not found: {rawdata_path}")
            return None

        try:
            with open(rawdata_path, "r", encoding="utf-8") as f:
                rawdata = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load rawdata.json: {e}")
            return None

        # Extract time interval
        time_interval = rawdata.get("time_interval", {})
        end_time = time_interval.get("end")
        if end_time is None:
            logger.error("time_interval.end not found in rawdata.json")
            return None

        # Find video file
        video_dir = sample_dir / "video"
        video_path = video_dir / "clip.mp4"
        if not video_path.exists():
            # Try to find any mp4 file
            video_files = list(video_dir.glob("*.mp4"))
            if video_files:
                video_path = video_files[0]
            else:
                logger.error(f"No video found in {video_dir}")
                return None

        # Find gaze data
        gaze_path = sample_dir / "signals" / "gaze.csv"
        if not gaze_path.exists():
            logger.warning(f"Gaze data not found: {gaze_path}")
            gaze_extractor = None
        else:
            try:
                gaze_extractor = GazeAnchorExtractor(gaze_path)
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"Failed to load gaze data: {e}")
                gaze_extractor = None

        # Determine gaze anchor at end of user's question
        if gaze_extractor:
            anchor = gaze_extractor.get_anchor_at_time(
                end_time,
                fallback="last_valid",
                default_center=(640.0, 360.0),
            )
        else:
            # Default to center of typical smart glasses resolution
            anchor = (640.0, 360.0)

        if anchor is None:
            logger.warning("No valid anchor found, using center")
            anchor = (640.0, 360.0)

        logger.info(f"Gaze anchor at t={end_time:.2f}s: ({anchor[0]:.1f}, {anchor[1]:.1f})")

        # Determine output path
        # Structure: output_dir/Task/Participant/sample_xxx/overlay_{strategy}.mp4
        parts = sample_dir.parts
        try:
            task_idx = next(i for i, p in enumerate(parts) if p.startswith("Task"))
            task_name = parts[task_idx]
            participant = parts[task_idx + 1]
            sample_name = parts[task_idx + 2]
        except (StopIteration, IndexError):
            task_name = "unknown"
            participant = "unknown"
            sample_name = sample_dir.name

        # Determine strategy name from UI JSON filename if not provided
        if strategy_name is None:
            strategy_name = ui_json_path.stem

        output_sample_dir = output_dir / task_name / participant / sample_name
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_sample_dir / f"overlay_{strategy_name}.mp4"

        # Render UI to image
        try:
            ui_image = self._render_ui(ui_json_path)
        except Exception as e:
            logger.error(f"Failed to render UI: {e}")
            return None

        if ui_image is None:
            logger.error("UI rendering returned None")
            return None

        # Create compositor and generate overlay video
        try:
            compositor = VideoCompositor(video_path, output_video_path)

            # Calculate frame range for overlay
            # Show UI for the last `overlay_duration` seconds of video
            video_duration = compositor.duration
            overlay_start_time = max(0, video_duration - self.overlay_duration)
            overlay_end_time = video_duration

            start_frame = compositor.time_to_frame(overlay_start_time)
            end_frame = compositor.time_to_frame(overlay_end_time)

            # Scale anchor to video resolution if needed
            video_width, video_height = compositor.width, compositor.height
            # Gaze coordinates are typically in the video's native resolution
            # Scale if needed based on viewport vs video dimensions
            scale_x = video_width / self.viewport_size[0]
            scale_y = video_height / self.viewport_size[1]

            # Don't scale anchor if video is same size or close to viewport
            if abs(scale_x - 1.0) > 0.1 or abs(scale_y - 1.0) > 0.1:
                scaled_anchor = (anchor[0] * scale_x, anchor[1] * scale_y)
            else:
                scaled_anchor = anchor

            # Composite the overlay
            result_path = compositor.composite_overlay(
                ui_image,
                scaled_anchor,
                start_frame,
                end_frame,
                anchor_position="center",
            )

            logger.info(f"Generated overlay video: {result_path}")
            return result_path

        except Exception as e:
            logger.error(f"Failed to composite video: {e}")
            return None

    def _render_ui(self, ui_json_path: Path) -> Optional[np.ndarray]:
        """Render UI JSON to an RGBA image.

        Args:
            ui_json_path: Path to the UI JSON file.

        Returns:
            RGBA numpy array or None if rendering failed.
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available for UI rendering")
            return self._create_fallback_ui(ui_json_path)

        async def _render():
            async with UIRenderer(
                self.preview_server_url,
                self.viewport_size,
            ) as renderer:
                return await renderer.render_to_image(ui_json_path, transparent_bg=True)

        try:
            return asyncio.run(_render())
        except Exception as e:
            logger.warning(f"Playwright rendering failed: {e}, using fallback")
            return self._create_fallback_ui(ui_json_path)

    def _create_fallback_ui(self, ui_json_path: Path) -> np.ndarray:
        """Create a simple fallback UI image when Playwright is unavailable.

        Args:
            ui_json_path: Path to the UI JSON file.

        Returns:
            RGBA numpy array with basic UI representation.
        """
        import cv2

        # Load UI JSON to extract text
        try:
            with open(ui_json_path, "r", encoding="utf-8") as f:
                ui_data = json.load(f)
        except Exception:
            ui_data = {}

        # Extract primary text content
        text = self._extract_text_from_ui(ui_data)
        if not text:
            text = "UI Response"

        # Create a simple semi-transparent card
        card_width = 400
        card_height = 150
        padding = 20

        # Create RGBA image
        img = np.zeros((card_height, card_width, 4), dtype=np.uint8)

        # Semi-transparent dark background
        img[:, :, 0] = 20   # B
        img[:, :, 1] = 20   # G
        img[:, :, 2] = 20   # R
        img[:, :, 3] = 200  # A

        # Add border
        cv2.rectangle(img, (0, 0), (card_width - 1, card_height - 1), (0, 255, 65, 255), 2)

        # Add text (OpenCV doesn't support alpha in putText, so we handle it separately)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Wrap text to fit
        words = text.split()
        lines = []
        current_line = ""
        max_width = card_width - 2 * padding

        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # Draw text lines
        y_position = padding + 25
        for line in lines[:4]:  # Limit to 4 lines
            cv2.putText(
                img,
                line,
                (padding, y_position),
                font,
                font_scale,
                (65, 255, 0, 255),  # BGRA green
                thickness,
                cv2.LINE_AA,
            )
            y_position += 28

        return img

    def _extract_text_from_ui(self, ui_data: dict, max_length: int = 100) -> str:
        """Extract primary text content from UI data.

        Args:
            ui_data: UI component dictionary.
            max_length: Maximum text length to return.

        Returns:
            Extracted text string.
        """
        texts = []

        def _extract(obj):
            if isinstance(obj, dict):
                # Check for text content
                for key in ["text", "content", "label"]:
                    if key in obj and isinstance(obj[key], str):
                        texts.append(obj[key])

                # Check props
                props = obj.get("props", {})
                if isinstance(props, dict):
                    for key in ["text", "content", "label"]:
                        if key in props and isinstance(props[key], str):
                            texts.append(props[key])

                # Recurse into children
                for key in ["children", "content"]:
                    if key in obj:
                        _extract(obj[key])

            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        _extract(ui_data)

        # Join and truncate
        result = " ".join(texts)
        if len(result) > max_length:
            result = result[:max_length - 3] + "..."
        return result

    def process_batch(
        self,
        samples: list[tuple[Path, Path]],
        output_dir: Path,
    ) -> list[Path]:
        """Process multiple samples in batch.

        Args:
            samples: List of (sample_dir, ui_json_path) tuples.
            output_dir: Base output directory.

        Returns:
            List of successfully generated output video paths.
        """
        results = []
        total = len(samples)

        for i, (sample_dir, ui_json_path) in enumerate(samples):
            logger.info(f"Processing {i + 1}/{total}: {sample_dir.name}")
            result = self.process_sample(sample_dir, ui_json_path, output_dir)
            if result:
                results.append(result)

        logger.info(f"Batch complete: {len(results)}/{total} succeeded")
        return results
