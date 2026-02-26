"""Visual change detector using histogram comparison (OpenCV, no LLM)."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VisualChangeDetector:
    """Compute a 0.0–1.0 change score between consecutive frames.

    Uses HSV histogram correlation — cheap and effective as a gate
    before expensive LLM vision calls.

    A score of 0.0 means identical frames; 1.0 means completely different.
    """

    def __init__(self):
        self._prev_hist: Optional[np.ndarray] = None

    def compute_change(self, frame: np.ndarray) -> float:
        """Compute visual change score vs. the previous frame.

        Args:
            frame: Current frame in BGR format.

        Returns:
            Change score in [0.0, 1.0].
            Returns 1.0 for the very first frame (maximum novelty).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute histogram on H and S channels
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        if self._prev_hist is None:
            self._prev_hist = hist
            return 1.0  # first frame is maximally novel

        # Correlation: 1.0 = identical, -1.0 = inverse
        correlation = cv2.compareHist(
            self._prev_hist, hist, cv2.HISTCMP_CORREL
        )

        self._prev_hist = hist

        # Convert correlation to change score: 1.0 correlation → 0.0 change
        change_score = max(0.0, min(1.0, 1.0 - correlation))
        return change_score

    def reset(self):
        """Reset internal state for a new video."""
        self._prev_hist = None
