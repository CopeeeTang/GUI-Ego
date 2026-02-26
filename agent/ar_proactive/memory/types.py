"""Data types for the three-layer memory system (RQ2).

Replaces the old MemoryEntry/LongTermSummary with function-oriented types:
  - FrameRecord:       per-frame observation (Working memory)
  - KeyEvent:          significant event (Progress memory)
  - TaskStep:          a step in a procedural task (Persistent memory)
  - ProgressSnapshot:  compressed progress summary
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class StepStatus(str, Enum):
    """Status of a task step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TaskStep:
    """A single step in a procedural task (e.g., recipe step)."""
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[float] = None   # timestamp in seconds
    completed_at: Optional[float] = None
    key_objects: list[str] = field(default_factory=list)
    key_actions: list[str] = field(default_factory=list)


@dataclass
class FrameRecord:
    """A single analyzed frame stored in working memory.

    Replaces the old MemoryEntry — focused on per-frame observation data.
    """
    timestamp: float
    frame_base64: str
    environment: str = ""
    detected_objects: list[str] = field(default_factory=list)
    detected_activities: list[str] = field(default_factory=list)
    current_action: str = ""   # what the user is actively doing right now
    text_visible: list[str] = field(default_factory=list)
    potential_hazards: list[str] = field(default_factory=list)
    people_present: bool = False
    visual_change_score: float = 0.0
    signal_context: dict = field(default_factory=dict)
    scene_tags: list[str] = field(default_factory=list)

    # Ground truth (optional, for evaluation)
    gt_action_label: str = ""
    gt_verb: str = ""
    gt_nouns: list[str] = field(default_factory=list)

    def to_summary_line(self) -> str:
        """One-line summary for context assembly."""
        parts = [f"[{self.timestamp:.1f}s] {self.environment}"]
        if self.current_action:
            parts.append(f"action: {self.current_action}")
        if self.detected_objects:
            parts.append(f"objects: {', '.join(self.detected_objects[:5])}")
        if self.detected_activities:
            parts.append(f"activities: {', '.join(self.detected_activities[:3])}")
        return " | ".join(parts)


@dataclass
class KeyEvent:
    """A significant event worth remembering in progress memory."""
    timestamp: float
    description: str
    event_type: str = "observation"  # observation, step_transition, error, hazard
    related_step: Optional[int] = None
    objects_involved: list[str] = field(default_factory=list)


@dataclass
class ProgressSnapshot:
    """Compressed progress state produced by SUMMARIZE_AND_DROP."""
    time_range_start: float
    time_range_end: float
    summary_text: str
    steps_completed: list[int] = field(default_factory=list)
    key_events_summary: list[str] = field(default_factory=list)


# ── Legacy compatibility ─────────────────────────────────────
# Keep old names as aliases so existing code doesn't break during migration

@dataclass
class MemoryEntry:
    """Legacy type — use FrameRecord instead."""
    timestamp: float
    frame_base64: str
    scene_description: str = ""
    detected_objects: list[str] = field(default_factory=list)
    detected_activities: list[str] = field(default_factory=list)
    importance_score: float = 0.0
    visual_change_score: float = 0.0
    signal_context: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_context_line(self) -> str:
        parts = [f"[{self.timestamp:.2f}s]"]
        if self.scene_description:
            desc = self.scene_description[:120]
            if len(self.scene_description) > 120:
                desc += "..."
            parts.append(desc)
        if self.detected_objects:
            parts.append(f"Objects: {', '.join(self.detected_objects[:5])}")
        if self.detected_activities:
            parts.append(f"Activities: {', '.join(self.detected_activities[:3])}")
        parts.append(f"(importance: {self.importance_score:.2f})")
        return " | ".join(parts)


@dataclass
class LongTermSummary:
    """Legacy type — use ProgressSnapshot instead."""
    time_range_start: float
    time_range_end: float
    summary_text: str
    key_events: list[str] = field(default_factory=list)
    recurring_objects: list[str] = field(default_factory=list)
    behavior_patterns: list[str] = field(default_factory=list)

    def to_context_text(self) -> str:
        lines = [
            f"Time range: {self.time_range_start:.2f}s – {self.time_range_end:.2f}s",
            f"Summary: {self.summary_text}",
        ]
        if self.key_events:
            lines.append(f"Key events: {'; '.join(self.key_events)}")
        if self.recurring_objects:
            lines.append(f"Recurring objects: {', '.join(self.recurring_objects)}")
        if self.behavior_patterns:
            lines.append(f"Behavior patterns: {'; '.join(self.behavior_patterns)}")
        return "\n".join(lines)
