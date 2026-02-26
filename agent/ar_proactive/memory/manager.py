"""Memory Manager — coordinates Persistent, Progress, and Working memory."""

import logging

from .persistent import PersistentMemory
from .progress import ProgressMemory
from .working import WorkingMemory
from .types import FrameRecord, KeyEvent, StepStatus

logger = logging.getLogger(__name__)


class MemoryManager:
    """Single entry point for all memory operations.

    Replaces TieredMemoryStore + MemoryRetriever with a semantically
    structured memory system aligned with RQ2.
    """

    def __init__(self, working_capacity: int = 8, max_events: int = 50):
        self.persistent = PersistentMemory()
        self.progress = ProgressMemory(max_events=max_events)
        self.working = WorkingMemory(capacity=working_capacity)

    def add_frame(self, record: FrameRecord):
        """Add a new frame observation to working memory."""
        self.working.add(record)

    def add_event(self, event: KeyEvent):
        """Record a key event in progress memory."""
        self.progress.add_event(event)

    def advance_step(self, new_index: int, timestamp: float):
        """Advance task progress to a new step."""
        self.progress.advance_step(
            self.persistent.steps, new_index, timestamp
        )

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "task_goal": self.persistent.task_goal,
            "total_steps": self.persistent.total_steps,
            "current_step": self.progress.current_step_index,
            "completed_steps": self.persistent.completed_steps,
            "key_events": len(self.progress.key_events),
            "working_frames": len(self.working.frames),
            "working_capacity": self.working.capacity,
            "progress_snapshots": len(self.progress.progress_summaries),
        }
