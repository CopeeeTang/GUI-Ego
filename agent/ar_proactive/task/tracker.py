"""Task Tracker — per-frame step detection and progress updating."""

import logging
from typing import Optional

from agent.src.llm.base import LLMClientBase
from .prompts import STEP_DETECTION_SYSTEM, build_step_detection_prompt
from ..memory.manager import MemoryManager
from ..memory.types import KeyEvent

logger = logging.getLogger(__name__)


class TaskTracker:
    """Track task progress by detecting step transitions per frame.

    Compares current scene to the task's step list.
    Only makes an LLM call when scene change is significant enough
    to potentially indicate a step transition.
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        visual_change_threshold: float = 0.15,
        max_tokens: int = 800,
    ):
        self.llm_client = llm_client
        self.visual_change_threshold = visual_change_threshold
        self.max_tokens = max_tokens
        self._frames_since_check = 0
        self._check_interval = 3  # check every N frames minimum

    def update(
        self,
        timestamp: float,
        frame_base64: str,
        scene_description: str,
        visual_change: float,
        memory: MemoryManager,
    ) -> dict:
        """Check for step transitions and update progress.

        Returns dict with: step_changed, detected_step, error_detected, etc.
        Skips LLM call if visual change is too low (no meaningful scene change).
        """
        self._frames_since_check += 1

        # Skip if no task steps are defined
        if not memory.persistent.steps:
            return {"step_changed": False, "has_task": False}

        # Gate: only check if enough visual change OR enough frames elapsed
        if (visual_change < self.visual_change_threshold
                and self._frames_since_check < self._check_interval):
            return {"step_changed": False, "skipped": True}

        self._frames_since_check = 0

        # Build step detection prompt
        step_list = [
            {
                "index": s.index,
                "description": s.description,
                "key_objects": s.key_objects,
            }
            for s in memory.persistent.steps
        ]

        user_prompt = build_step_detection_prompt(
            current_scene=scene_description,
            step_list=step_list,
            current_step=memory.progress.current_step_index,
            cumulative_state=memory.progress._cumulative_state,
        )

        try:
            result = self.llm_client.complete_json_with_images(
                user_prompt=user_prompt,
                images=[frame_base64],
                system_prompt=STEP_DETECTION_SYSTEM,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Step detection failed: {e}")
            return {"step_changed": False, "error": str(e)}

        detected_step = result.get("detected_step_index", memory.progress.current_step_index)
        step_changed = result.get("step_changed", False)
        error_detected = result.get("error_detected", False)
        cumulative_state = result.get("cumulative_state", "")

        # Update progress memory
        if cumulative_state:
            memory.progress.update_cumulative_state(cumulative_state)

        if step_changed and detected_step != memory.progress.current_step_index:
            memory.advance_step(detected_step, timestamp)

        if error_detected:
            error_desc = result.get("error_description", "Unknown error")
            memory.add_event(KeyEvent(
                timestamp=timestamp,
                description=f"Error: {error_desc}",
                event_type="error",
                related_step=detected_step,
            ))
            logger.warning(f"[{timestamp:.1f}s] Error detected: {error_desc}")

        return {
            "step_changed": step_changed,
            "detected_step": detected_step,
            "error_detected": error_detected,
            "confidence": result.get("confidence", 0.0),
        }
