"""Video processing module for frame extraction and scene analysis."""

from .frame_processor import FrameProcessor
from .change_detector import VisualChangeDetector
from .scene_analyzer import SceneAnalyzer

__all__ = ["FrameProcessor", "VisualChangeDetector", "SceneAnalyzer"]
