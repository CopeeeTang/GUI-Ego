"""
Layer 1: Task Memory

Always-in-context memory that tracks the current cooking task state.
Maintained as structured text, injected as system prompt context.

Contents:
  - Recipe/task name
  - Step list with completion status
  - Currently active entities (objects in use)
  - Estimated progress percentage
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from data.egtea_loader import ActionClip, CookingSession

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """A single tracked cooking step."""
    action: str
    verb: str
    nouns: list[str]
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    is_completed: bool = False


@dataclass
class TaskState:
    """Current state of the cooking task."""
    recipe: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    active_entities: set[str] = field(default_factory=set)
    completed_count: int = 0
    current_step_idx: int = -1
    last_update_time: float = 0.0

    @property
    def progress_pct(self) -> float:
        if not self.steps:
            return 0.0
        return 100.0 * self.completed_count / len(self.steps)

    def to_context_string(self) -> str:
        """Serialize to text for injection into VLM context (~200-500 tokens)."""
        lines = [f"[Task Memory] Recipe: {self.recipe}"]

        # Current progress
        lines.append(f"Progress: {self.completed_count}/{len(self.steps)} steps ({self.progress_pct:.0f}%)")

        # Recent completed steps (last 3)
        completed = [s for s in self.steps if s.is_completed]
        if completed:
            lines.append("Recently completed:")
            for s in completed[-3:]:
                lines.append(f"  ✓ {s.action}")

        # Current step
        if 0 <= self.current_step_idx < len(self.steps):
            current = self.steps[self.current_step_idx]
            lines.append(f"Current: {current.action}")

        # Upcoming steps (next 2)
        upcoming_start = self.current_step_idx + 1
        upcoming = self.steps[upcoming_start:upcoming_start + 2]
        if upcoming:
            lines.append("Next steps:")
            for s in upcoming:
                lines.append(f"  → {s.action}")

        # Active entities
        if self.active_entities:
            lines.append(f"Active objects: {', '.join(sorted(self.active_entities))}")

        return "\n".join(lines)


class TaskMemory:
    """
    Layer 1 memory: tracks cooking task progress.

    In a real streaming system, this is updated by the VLM
    when it detects step transitions. For evaluation, we can
    also update from GT annotations.
    """

    def __init__(self):
        self.state = TaskState()

    def initialize_from_session(self, session: CookingSession):
        """Initialize task memory from GT session (oracle mode)."""
        self.state = TaskState(recipe=session.recipe)

        seen_actions = set()
        for action in session.actions:
            key = action.action_label
            if key not in seen_actions:
                seen_actions.add(key)
                self.state.steps.append(StepRecord(
                    action=action.action_label,
                    verb=action.verb,
                    nouns=list(action.nouns),
                ))

    def update_from_gt(self, action: ActionClip, timestamp: float):
        """Update state using GT annotation (oracle mode for evaluation)."""
        self.state.last_update_time = timestamp

        # Track ALL entities seen throughout the session (no cap)
        for noun in action.nouns:
            self.state.active_entities.add(noun)

        # Find matching step and mark completed
        for i, step in enumerate(self.state.steps):
            if step.action == action.action_label and not step.is_completed:
                step.is_completed = True
                step.started_at = action.start_sec
                step.completed_at = action.end_sec
                self.state.completed_count += 1
                self.state.current_step_idx = i
                break

    def update_from_vlm(self, vlm_output: dict, timestamp: float):
        """
        Update state from VLM analysis output.

        Expected vlm_output format:
        {
            "detected_action": "Cut tomato",
            "is_step_complete": true,
            "entities": ["tomato", "knife", "cutting_board"],
            "estimated_progress": 0.35
        }
        """
        self.state.last_update_time = timestamp

        # Update entities
        if "entities" in vlm_output:
            for e in vlm_output["entities"]:
                self.state.active_entities.add(e)

        # Mark step complete
        if vlm_output.get("is_step_complete") and "detected_action" in vlm_output:
            for i, step in enumerate(self.state.steps):
                if step.action == vlm_output["detected_action"] and not step.is_completed:
                    step.is_completed = True
                    step.completed_at = timestamp
                    self.state.completed_count += 1
                    self.state.current_step_idx = i
                    break

    def get_context(self) -> str:
        """Get current task memory as context string for VLM prompt."""
        return self.state.to_context_string()

    def get_state(self) -> TaskState:
        return self.state

    def reset(self):
        self.state = TaskState()
