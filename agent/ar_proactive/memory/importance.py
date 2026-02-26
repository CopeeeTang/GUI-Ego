"""Importance scorer — computes a 0.0–1.0 score for each frame."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords that suggest a scene/location transition
TRANSITION_KEYWORDS = {
    "entrance", "exit", "door", "doorway", "hallway", "corridor",
    "stairs", "staircase", "elevator", "escalator", "outside",
    "parking", "lobby", "crossing", "intersection", "road",
    "sidewalk", "aisle", "checkout", "counter", "register",
    "gate", "transition", "moved", "entering", "leaving",
    "arrived", "departing", "turning", "corner",
}


class ImportanceScorer:
    """Score each frame's importance using four weighted factors.

    Factors:
        1. Visual change (0.30) — histogram diff from previous frame
        2. Semantic novelty (0.30) — new objects/activities vs recent memory
        3. Signal anomaly (0.25) — gaze fixation, HR/EDA spikes
        4. Scene transition (0.15) — keywords indicating location change
    """

    def __init__(
        self,
        weight_visual_change: float = 0.30,
        weight_semantic_novelty: float = 0.30,
        weight_signal_anomaly: float = 0.25,
        weight_scene_transition: float = 0.15,
    ):
        self.w_visual = weight_visual_change
        self.w_novelty = weight_semantic_novelty
        self.w_signal = weight_signal_anomaly
        self.w_transition = weight_scene_transition

    def score(
        self,
        visual_change: float,
        detected_objects: list[str],
        detected_activities: list[str],
        recent_objects: set[str],
        recent_activities: set[str],
        signal_context: dict,
        scene_tags: list[str],
    ) -> float:
        """Compute importance score for a frame.

        Args:
            visual_change: Change score from VisualChangeDetector (0–1).
            detected_objects: Objects detected in this frame.
            detected_activities: Activities detected in this frame.
            recent_objects: Objects seen in recent short-term memory.
            recent_activities: Activities seen in recent short-term memory.
            signal_context: Dict with gaze_fixation, hr_spike, eda_spike bools.
            scene_tags: Tags from scene analysis.

        Returns:
            Importance score in [0.0, 1.0].
        """
        # Factor 1: Visual change (already 0–1)
        f_visual = min(1.0, visual_change)

        # Factor 2: Semantic novelty
        f_novelty = self._compute_novelty(
            detected_objects, detected_activities,
            recent_objects, recent_activities,
        )

        # Factor 3: Signal anomaly
        f_signal = self._compute_signal_anomaly(signal_context)

        # Factor 4: Scene transition
        f_transition = self._compute_scene_transition(scene_tags)

        score = (
            self.w_visual * f_visual
            + self.w_novelty * f_novelty
            + self.w_signal * f_signal
            + self.w_transition * f_transition
        )

        score = max(0.0, min(1.0, score))

        logger.debug(
            f"Importance: {score:.3f} "
            f"(visual={f_visual:.2f}, novelty={f_novelty:.2f}, "
            f"signal={f_signal:.2f}, transition={f_transition:.2f})"
        )

        return score

    def _compute_novelty(
        self,
        objects: list[str],
        activities: list[str],
        recent_objects: set[str],
        recent_activities: set[str],
    ) -> float:
        """Compute semantic novelty: fraction of new objects/activities."""
        current = set(o.lower() for o in objects) | set(a.lower() for a in activities)
        recent = set(o.lower() for o in recent_objects) | set(a.lower() for a in recent_activities)

        if not current:
            return 0.0

        new_items = current - recent
        return len(new_items) / len(current)

    def _compute_signal_anomaly(self, signal_context: dict) -> float:
        """Compute signal anomaly score from physiological signals."""
        if not signal_context:
            return 0.0

        anomaly_count = 0
        total_signals = 0

        for key in ("gaze_fixation", "hr_spike", "eda_spike"):
            if key in signal_context:
                total_signals += 1
                if signal_context[key]:
                    anomaly_count += 1

        if total_signals == 0:
            return 0.0

        return anomaly_count / total_signals

    def _compute_scene_transition(self, scene_tags: list[str]) -> float:
        """Detect scene transitions from keywords in tags."""
        if not scene_tags:
            return 0.0

        tag_set = set(t.lower() for t in scene_tags)
        overlap = tag_set & TRANSITION_KEYWORDS

        if overlap:
            # More transition keywords → higher score, cap at 1.0
            return min(1.0, len(overlap) * 0.5)
        return 0.0
