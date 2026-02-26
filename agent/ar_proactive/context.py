"""StreamingContext — the bridge between video understanding and UI generation.

Assembled every frame from the three memory layers.
Passed to InterventionEngine and UIGenerator as the complete state.
"""

from dataclasses import dataclass, field
from typing import Optional

from .memory.persistent import PersistentMemory
from .memory.progress import ProgressMemory
from .memory.working import WorkingMemory
from .memory.types import FrameRecord, StepStatus


@dataclass
class StreamingContext:
    """Complete context state at a given moment in the video stream.

    This is the central data structure of the system. It contains
    everything the InterventionEngine and UIGenerator need to decide
    what to do.
    """

    # ── Current frame ────────────────────────────
    timestamp: float = 0.0
    trigger_type: str = "interval"  # interval | scene_change | fixation | step_transition
    current_frame: Optional[FrameRecord] = None

    # ── Persistent Memory (task knowledge) ───────
    task_goal: str = ""
    task_type: str = ""
    total_steps: int = 0
    step_descriptions: list[str] = field(default_factory=list)

    # ── Progress Memory (task state) ─────────────
    current_step_index: int = 0
    completed_steps: int = 0
    progress_text: str = ""
    cumulative_state: str = ""
    recent_events: list[str] = field(default_factory=list)

    # ── Working Memory (recent perception) ───────
    recent_context_text: str = ""
    recent_objects: set = field(default_factory=set)
    recent_activities: set = field(default_factory=set)

    # ── Signals ──────────────────────────────────
    signal_context: dict = field(default_factory=dict)
    visual_change_score: float = 0.0

    # ── UI State (for RQ3 consistency) ───────────
    previous_ui: Optional[dict] = None
    previous_ui_timestamp: Optional[float] = None

    def to_llm_context(self) -> str:
        """Assemble the full context text for LLM prompts."""
        sections = []

        # Persistent: task knowledge
        if self.task_goal:
            sections.append(f"## Task\nGoal: {self.task_goal}")
            if self.step_descriptions:
                steps_text = "\n".join(
                    f"  {'✓' if i < self.completed_steps else '◉' if i == self.current_step_index else '○'} "
                    f"Step {i+1}: {desc}"
                    for i, desc in enumerate(self.step_descriptions)
                )
                sections.append(f"Steps ({self.completed_steps}/{self.total_steps} done):\n{steps_text}")

        # Progress: current state
        if self.progress_text:
            sections.append(f"## Progress\n{self.progress_text}")

        if self.recent_events:
            events = "\n".join(f"  - {e}" for e in self.recent_events[-5:])
            sections.append(f"## Recent Events\n{events}")

        # Working: recent frames
        if self.recent_context_text:
            sections.append(f"## Recent Context\n{self.recent_context_text}")

        return "\n\n".join(sections) if sections else "(No context yet)"

    @classmethod
    def assemble(
        cls,
        timestamp: float,
        current_frame: FrameRecord,
        persistent: PersistentMemory,
        progress: ProgressMemory,
        working: WorkingMemory,
        signal_context: dict,
        visual_change: float,
        trigger_type: str = "interval",
    ) -> "StreamingContext":
        """Factory: assemble context from the three memory layers."""
        return cls(
            timestamp=timestamp,
            trigger_type=trigger_type,
            current_frame=current_frame,
            # Persistent
            task_goal=persistent.task_goal,
            task_type=persistent.task_type,
            total_steps=persistent.total_steps,
            step_descriptions=[s.description for s in persistent.steps],
            # Progress
            current_step_index=progress.current_step_index,
            completed_steps=sum(
                1 for s in persistent.steps
                if s.status == StepStatus.COMPLETED
            ),
            progress_text=progress.to_context_text(persistent.steps),
            cumulative_state=progress._cumulative_state,
            recent_events=[e.description for e in progress.key_events[-5:]],
            # Working
            recent_context_text=working.to_context_text(),
            recent_objects=working.get_recent_objects(),
            recent_activities=working.get_recent_activities(),
            # Signals
            signal_context=signal_context,
            visual_change_score=visual_change,
            # UI state
            previous_ui=working.previous_ui,
            previous_ui_timestamp=working.previous_ui_timestamp,
        )
