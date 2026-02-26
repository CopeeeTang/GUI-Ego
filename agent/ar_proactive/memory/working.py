"""Working Memory — sliding window of recent frames.

RQ2 Layer 3: recent frames, current scene state.
Fixed capacity, oldest frames evicted automatically.
"""

import logging
from collections import deque
from typing import Optional

from .types import FrameRecord

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Sliding window of recent frame observations.

    Stores full data (including base64 images) for the last N frames.
    Provides quick access to recent objects, activities, and scene state.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.frames: deque[FrameRecord] = deque(maxlen=capacity)
        self.previous_ui: Optional[dict] = None
        self.previous_ui_timestamp: Optional[float] = None

    def add(self, record: FrameRecord):
        """Add a new frame record."""
        self.frames.append(record)

    def get_recent(self, n: int = 3) -> list[FrameRecord]:
        """Get the last N frame records."""
        return list(self.frames)[-n:]

    def get_recent_objects(self, n: int = 3) -> set[str]:
        """Objects seen in the last N frames."""
        objs: set[str] = set()
        for fr in self.get_recent(n):
            objs.update(fr.detected_objects)
        return objs

    def get_recent_activities(self, n: int = 3) -> set[str]:
        """Activities seen in the last N frames."""
        acts: set[str] = set()
        for fr in self.get_recent(n):
            acts.update(fr.detected_activities)
        return acts

    def get_current_scene(self) -> Optional[FrameRecord]:
        """Get the most recent frame record."""
        return self.frames[-1] if self.frames else None

    def set_previous_ui(self, ui_component: dict, timestamp: float):
        """Cache the last generated UI for content consistency checks (RQ3)."""
        self.previous_ui = ui_component
        self.previous_ui_timestamp = timestamp

    def to_context_text(self, n: int = 3) -> str:
        """Format recent frames for LLM context."""
        recent = self.get_recent(n)
        if not recent:
            return "(No recent observations)"
        lines = [f"Recent observations (last {len(recent)} frames):"]
        for fr in recent:
            lines.append(f"  - {fr.to_summary_line()}")
        return "\n".join(lines)
