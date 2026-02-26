"""Task Knowledge Extractor — identifies task and decomposes into steps."""

import logging
from typing import Optional

from agent.src.llm.base import LLMClientBase
from .prompts import TASK_IDENTIFICATION_SYSTEM, TASK_IDENTIFICATION_USER
from ..memory.persistent import PersistentMemory

logger = logging.getLogger(__name__)


class TaskKnowledgeExtractor:
    """Extract task knowledge from initial video frames.

    Called once at the start of a session to populate PersistentMemory.
    Uses multiple frames for more reliable task identification.
    """

    def __init__(self, llm_client: LLMClientBase, max_tokens: int = 1500):
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    def extract(
        self,
        frame_images: list[str],
        persistent_memory: PersistentMemory,
        scene_context: Optional[str] = None,
    ) -> bool:
        """Identify the task from initial frames and populate persistent memory.

        Args:
            frame_images: Base64-encoded initial frames (2-4 frames recommended).
            persistent_memory: PersistentMemory instance to populate.
            scene_context: Optional external context (e.g. from scene_contexts.json).

        Returns:
            True if a task was successfully identified.
        """
        user_prompt = TASK_IDENTIFICATION_USER
        if scene_context:
            user_prompt = f"Additional context: {scene_context}\n\n{user_prompt}"

        try:
            result = self.llm_client.complete_json_with_images(
                user_prompt=user_prompt,
                images=frame_images,
                system_prompt=TASK_IDENTIFICATION_SYSTEM,
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Task identification failed: {e}")
            return False

        task_goal = result.get("task_goal", "")
        task_type = result.get("task_type", "other")
        steps = result.get("steps", [])
        confidence = result.get("confidence", 0.0)

        if confidence < 0.3 or not task_goal:
            logger.warning(f"Low confidence task identification: {confidence:.2f}")
            persistent_memory.task_goal = task_goal or "Unknown task"
            persistent_memory.task_type = task_type
            persistent_memory.knowledge_source = "inferred (low confidence)"
            return False

        persistent_memory.set_task(task_goal, task_type, steps)
        persistent_memory.knowledge_source = "inferred from video"

        logger.info(
            f"Task identified: '{task_goal}' ({task_type}), "
            f"{len(steps)} steps, confidence={confidence:.2f}"
        )
        return True
