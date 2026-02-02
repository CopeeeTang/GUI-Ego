"""Video overlay module for compositing UI onto video frames.

This module provides tools for:
- Extracting gaze anchor points from eye-tracking data
- Rendering UI components to images using headless browser
- Compositing UI overlays onto video frames
- Processing complete samples with overlay generation
"""

from .gaze_anchor import GazeAnchorExtractor
from .ui_renderer import UIRenderer
from .compositor import VideoCompositor
from .processor import VideoOverlayProcessor

__all__ = [
    "GazeAnchorExtractor",
    "UIRenderer",
    "VideoCompositor",
    "VideoOverlayProcessor",
]
