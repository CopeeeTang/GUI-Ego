"""Progress Memory — monotonically-advancing task state.

RQ2 Layer 2: current step, completed steps, key events.
Updated on step transitions and key events. Old progress replaced by new.
"""

import logging
from typing import Optional

from .types import KeyEvent, TaskStep, StepStatus, ProgressSnapshot

logger = logging.getLogger(__name__)


class ProgressMemory:
    """Tracks task progress and key events over time.

    Unlike working memory (sliding window), progress memory is cumulative:
    step completions are permanent, key events persist until compressed.
    """

    def __init__(self, max_events: int = 50):
        self.current_step_index: int = 0
        self.key_events: list[KeyEvent] = []
        self.progress_summaries: list[ProgressSnapshot] = []
        self.max_events = max_events
        self._cumulative_state: str = ""  # e.g. "eggs cracked, tomatoes diced"

    def advance_step(self, steps: list[TaskStep], new_index: int, timestamp: float):
        """Mark current step as completed and advance to new_index."""
        if new_index <= self.current_step_index and self.current_step_index > 0:
            return  # don't go backwards

        # Complete all steps up to new_index
        for i in range(self.current_step_index, min(new_index, len(steps))):
            if steps[i].status != StepStatus.COMPLETED:
                steps[i].status = StepStatus.COMPLETED
                steps[i].completed_at = timestamp

        # Start new step
        if new_index < len(steps):
            steps[new_index].status = StepStatus.IN_PROGRESS
            steps[new_index].started_at = timestamp

        old_step = self.current_step_index
        self.current_step_index = new_index

        self.add_event(KeyEvent(
            timestamp=timestamp,
            description=f"Advanced from step {old_step + 1} to step {new_index + 1}",
            event_type="step_transition",
            related_step=new_index,
        ))

        logger.info(f"Progress: step {old_step + 1} → {new_index + 1}")

    def add_event(self, event: KeyEvent):
        """Record a key event. Evicts oldest if at capacity."""
        self.key_events.append(event)
        if len(self.key_events) > self.max_events:
            self.key_events = self.key_events[-self.max_events:]

    def update_cumulative_state(self, state: str):
        """Update the cumulative state description."""
        self._cumulative_state = state

    def add_progress_snapshot(self, snapshot: ProgressSnapshot):
        """Store a compressed progress snapshot from SUMMARIZE_AND_DROP."""
        self.progress_summaries.append(snapshot)

    def to_context_text(self, steps: list[TaskStep]) -> str:
        """Format for LLM context."""
        lines = []

        # Current progress
        total = len(steps)
        completed = sum(1 for s in steps if s.status == StepStatus.COMPLETED)
        lines.append(f"Progress: {completed}/{total} steps completed")

        if 0 <= self.current_step_index < total:
            current = steps[self.current_step_index]
            lines.append(f"Current step: {current.index + 1}. {current.description}")

        if self._cumulative_state:
            lines.append(f"Cumulative state: {self._cumulative_state}")

        # Compressed summaries
        for snap in self.progress_summaries:
            lines.append(f"[Summary {snap.time_range_start:.0f}s–{snap.time_range_end:.0f}s] {snap.summary_text}")

        # Recent key events (last 5)
        recent_events = self.key_events[-5:]
        if recent_events:
            lines.append("Recent events:")
            for e in recent_events:
                lines.append(f"  [{e.timestamp:.1f}s] {e.description}")

        return "\n".join(lines)
