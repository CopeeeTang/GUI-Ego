"""Persistent Memory — immutable task knowledge for the entire session.

RQ2 Layer 1: task goal, recipe/procedure steps, user preferences.
Set once at session start. Never evicted.
"""

import logging
from dataclasses import dataclass, field

from .types import TaskStep, StepStatus

logger = logging.getLogger(__name__)


@dataclass
class PersistentMemory:
    """Task-level knowledge that does not change during a session.

    Populated by TaskKnowledgeExtractor at session start.
    Always included in full in LLM context.
    """

    task_goal: str = ""                           # e.g. "Cook pasta salad"
    task_type: str = ""                           # e.g. "cooking", "assembly"
    steps: list[TaskStep] = field(default_factory=list)
    user_preferences: dict = field(default_factory=dict)
    scene_constraints: dict = field(default_factory=dict)
    knowledge_source: str = ""                    # "inferred from video", "egtea_gt"

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    def set_task(self, goal: str, task_type: str, steps: list[dict]):
        """Initialize task knowledge from extracted data."""
        self.task_goal = goal
        self.task_type = task_type
        self.steps = [
            TaskStep(
                index=i,
                description=s.get("description", ""),
                key_objects=s.get("key_objects", []),
                key_actions=s.get("key_actions", []),
            )
            for i, s in enumerate(steps)
        ]
        logger.info(f"PersistentMemory: task='{goal}', {len(self.steps)} steps")

    def set_task_from_egtea(self, recipe: str, actions: list):
        """Initialize from EGTEA ground truth action sequence.

        Deduplicates consecutive identical actions to form logical steps.
        """
        self.task_goal = f"Prepare {recipe}"
        self.task_type = "cooking"
        self.knowledge_source = "egtea_gt"

        # Group consecutive actions into logical steps
        step_descriptions = []
        for action in actions:
            label = action.action_label
            # Avoid duplicate consecutive steps
            if not step_descriptions or step_descriptions[-1]["description"] != label:
                step_descriptions.append({
                    "description": label,
                    "key_objects": action.nouns[:],
                    "key_actions": [action.verb],
                })

        self.steps = [
            TaskStep(
                index=i,
                description=s["description"],
                key_objects=s["key_objects"],
                key_actions=s["key_actions"],
            )
            for i, s in enumerate(step_descriptions)
        ]
        logger.info(
            f"PersistentMemory (EGTEA): recipe='{recipe}', "
            f"{len(self.steps)} unique steps from {len(actions)} actions"
        )

    def to_context_text(self) -> str:
        """Format for LLM context. Always included in full."""
        lines = [f"Task: {self.task_goal}"]
        if self.steps:
            lines.append(f"Procedure ({len(self.steps)} steps):")
            for step in self.steps:
                status_icon = {
                    "pending": "○", "in_progress": "◉",
                    "completed": "✓", "skipped": "–", "error": "✗"
                }.get(step.status.value, "?")
                lines.append(f"  {status_icon} Step {step.index + 1}: {step.description}")
        if self.user_preferences:
            lines.append(f"User preferences: {self.user_preferences}")
        return "\n".join(lines)
