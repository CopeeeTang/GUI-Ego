# Proactive AR Agent — RQ-Driven Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the `agent/ar_proactive/` system based on three research questions (RQ1: When to Intervene, RQ2: Streaming Memory, RQ3: GUI Consistency) to transform it from a scene-description agent into a task-aware, progress-tracking, UI-generating system.

**Architecture:** Two-stage design — Stage 1 (Multimodal Brain) handles video understanding, memory management, and intervention timing; Stage 2 (Generative UI) produces A2UI component JSON with lifecycle management. The `StreamingContext` dataclass bridges the two stages. Memory is restructured from importance-based tiers to function-based layers (Persistent / Progress / Working).

**Tech Stack:** Python 3.11+, OpenCV, Anthropic Claude API (via existing LLM factory), A2UI v0.9 standard components.

**Reference:** `docs/survey/streaming-gui-research-questions.md` for full RQ details.

---

## Architecture Diagram

```
Video + Signals
    │
    ▼
┌─ OBSERVE ──────────────────────────────────┐
│  FrameProcessor     (KEEP — no changes)    │
│  VisualChangeDetector (KEEP)               │
│  SceneAnalyzer      (MODIFY — add steps)   │
│  SignalAnalyzer     (KEEP)                 │
└──────────────┬─────────────────────────────┘
               ▼
┌─ THINK ────────────────────────────────────┐
│                                            │
│  ┌─ Memory (RQ2 — 3 functional layers) ──┐ │
│  │ PersistentMemory: task, recipe, prefs │ │
│  │ ProgressMemory:   step, events, state │ │
│  │ WorkingMemory:    recent N frames     │ │
│  └───────────────────────────────────────┘ │
│                                            │
│  TaskTracker (RQ1+RQ2)                     │
│    └─ step detection, progress update      │
│                                            │
│  StreamingContext ← assemble all layers    │
│                                            │
│  TriggerDecider (RQ1)                      │
│    ├─ anticipatory: next step approaching? │
│    ├─ reactive: error detected?            │
│    └─ signal: gaze/HR/EDA anomaly?         │
└──────────────┬─────────────────────────────┘
               ▼
┌─ ACT ──────────────────────────────────────┐
│  InterventionEngine (REFACTOR)             │
│    └─ decides intervention mode + content  │
│                                            │
│  UIStateManager (NEW — RQ3)                │
│    └─ HIDDEN → ENTERING → ACTIVE → EXIT   │
│                                            │
│  A2UIGenerator (NEW — RQ3)                 │
│    └─ StreamingContext → A2UI JSON         │
└────────────────────────────────────────────┘
```

---

## New File Structure

```
agent/ar_proactive/
├── __init__.py                  (MODIFY — update exports)
├── __main__.py                  (MODIFY — new CLI flags)
├── config.py                    (MODIFY — new params)
├── agent.py                     (REWRITE — new loop)
├── context.py                   (NEW — StreamingContext bridge)
│
├── memory/
│   ├── __init__.py              (MODIFY)
│   ├── types.py                 (REWRITE — FrameRecord, KeyEvent)
│   ├── persistent.py            (NEW — PersistentMemory)
│   ├── progress.py              (NEW — ProgressMemory)
│   ├── working.py               (NEW — replace store.py)
│   ├── manager.py               (NEW — MemoryManager)
│   └── compressor.py            (NEW — SUMMARIZE_AND_DROP)
│
├── task/                        (NEW module)
│   ├── __init__.py
│   ├── tracker.py               (TaskTracker)
│   ├── knowledge.py             (TaskKnowledgeExtractor)
│   └── prompts.py               (Task understanding prompts)
│
├── video/                       (KEEP mostly)
│   ├── __init__.py
│   ├── frame_processor.py       (KEEP)
│   ├── scene_analyzer.py        (MODIFY — add step detection fields)
│   └── change_detector.py       (KEEP)
│
├── signals/                     (KEEP unchanged)
│   ├── __init__.py
│   ├── reader.py
│   └── analyzer.py
│
├── intervention/                (REFACTOR)
│   ├── __init__.py              (MODIFY)
│   ├── types.py                 (MODIFY — add InterventionMode)
│   ├── trigger.py               (NEW — TriggerDecider)
│   ├── engine.py                (REWRITE — uses StreamingContext)
│   └── prompts.py               (REWRITE — task-aware prompts)
│
└── ui/                          (NEW module — RQ3)
    ├── __init__.py
    ├── types.py                 (UIState enum, UIEvent)
    ├── state_manager.py         (UIStateManager lifecycle)
    └── generator.py             (A2UIGenerator)
```

**Files to DELETE:**
- `memory/store.py` → replaced by `persistent.py` + `progress.py` + `working.py` + `manager.py`
- `memory/importance.py` → replaced by `intervention/trigger.py`
- `memory/retriever.py` → replaced by `memory/manager.py`

---

## Phase 1: StreamingContext & Memory Types (RQ2 Foundation)

> The central refactoring: replace importance-based memory tiers with function-based memory layers, and introduce StreamingContext as the universal bridge.

### Task 1.1: Define new memory data types

**Files:**
- Rewrite: `agent/ar_proactive/memory/types.py`

**Step 1: Write the new types**

Replace `MemoryEntry` and `LongTermSummary` with function-oriented types:

```python
"""Data types for the three-layer memory system."""

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
    started_at: Optional[float] = None  # timestamp
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
    text_visible: list[str] = field(default_factory=list)
    potential_hazards: list[str] = field(default_factory=list)
    people_present: bool = False
    visual_change_score: float = 0.0
    signal_context: dict = field(default_factory=dict)
    scene_tags: list[str] = field(default_factory=list)

    def to_summary_line(self) -> str:
        """One-line summary for context assembly."""
        parts = [f"[{self.timestamp:.1f}s] {self.environment}"]
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
```

**Step 2: Verify imports work**

Run: `cd /home/v-tangxin/GUI && source ml_env/bin/activate && python3 -c "from agent.ar_proactive.memory.types import FrameRecord, KeyEvent, TaskStep, StepStatus; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add agent/ar_proactive/memory/types.py
git commit -m "refactor(memory): replace MemoryEntry with function-based types (RQ2)"
```

---

### Task 1.2: Implement PersistentMemory

**Files:**
- Create: `agent/ar_proactive/memory/persistent.py`

**What it does:** Stores immutable task knowledge that persists for the entire session — task goal, step list (recipe), user preferences, scene constraints. Corresponds to RQ2's "Layer 1: 固有记忆".

**Step 1: Write implementation**

```python
"""Persistent Memory — immutable task knowledge for the entire session.

RQ2 Layer 1: task goal, recipe/procedure steps, user preferences.
Set once at session start. Never evicted.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .types import TaskStep

logger = logging.getLogger(__name__)


@dataclass
class PersistentMemory:
    """Task-level knowledge that does not change during a session.

    Populated by TaskKnowledgeExtractor at session start.
    Always included in full in LLM context.
    """

    task_goal: str = ""                          # e.g. "Cook tomato scrambled eggs"
    task_type: str = ""                          # e.g. "cooking", "assembly", "navigation"
    steps: list[TaskStep] = field(default_factory=list)
    user_preferences: dict = field(default_factory=dict)  # e.g. {"dietary": "no spicy"}
    scene_constraints: dict = field(default_factory=dict)  # available tools/ingredients
    knowledge_source: str = ""                   # e.g. "inferred from video", "user provided"

    @property
    def total_steps(self) -> int:
        return len(self.steps)

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
```

**Step 2: Verify**

Run: `python3 -c "from agent.ar_proactive.memory.persistent import PersistentMemory; m = PersistentMemory(); m.set_task('Cook eggs', 'cooking', [{'description': 'Crack eggs'}]); print(m.to_context_text())"`

**Step 3: Commit**

---

### Task 1.3: Implement ProgressMemory

**Files:**
- Create: `agent/ar_proactive/memory/progress.py`

**What it does:** Tracks monotonically-advancing task progress — current step, completed steps, key events. Updated each time a step transition is detected. Corresponds to RQ2's "Layer 2: 进度记忆".

```python
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
        """Update the cumulative state description (e.g. 'eggs cracked, oil heating')."""
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
```

**Step 2: Verify + Commit** (same pattern)

---

### Task 1.4: Implement WorkingMemory

**Files:**
- Create: `agent/ar_proactive/memory/working.py`

**What it does:** Sliding-window buffer of recent N frames with full data (including base64). Replaces the old `store.py` short-term deque. Corresponds to RQ2's "Layer 3: 工作记忆".

```python
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
        self.previous_ui: Optional[dict] = None  # last generated UI component
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
```

---

### Task 1.5: Implement StreamingContext

**Files:**
- Create: `agent/ar_proactive/context.py`

**What it does:** The single data structure that bridges Stage 1 (understanding) and Stage 2 (UI generation). Assembled from all three memory layers every frame. This is the **core output of RQ2**.

```python
"""StreamingContext — the bridge between video understanding and UI generation.

Assembled every frame from the three memory layers.
Passed to InterventionEngine and UIGenerator as the complete state.
"""

from dataclasses import dataclass, field
from typing import Optional

from .memory.persistent import PersistentMemory
from .memory.progress import ProgressMemory
from .memory.working import WorkingMemory
from .memory.types import FrameRecord


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
                if s.status.value == "completed"
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
```

**Step 2: Verify assembly roundtrip**

Run: `python3 -c "from agent.ar_proactive.context import StreamingContext; ctx = StreamingContext(timestamp=1.0, task_goal='Cook eggs'); print(ctx.to_llm_context())"`

**Step 3: Commit**

```bash
git commit -m "feat(context): add StreamingContext bridge dataclass (RQ2)"
```

---

### Task 1.6: Implement MemoryManager

**Files:**
- Create: `agent/ar_proactive/memory/manager.py`

**What it does:** Coordinates all three memory layers. Single entry point for adding frame records, updating progress, and assembling context. Replaces the old `TieredMemoryStore` + `MemoryRetriever`.

```python
"""Memory Manager — coordinates Persistent, Progress, and Working memory."""

import logging
from typing import Optional

from .persistent import PersistentMemory
from .progress import ProgressMemory
from .working import WorkingMemory
from .types import FrameRecord, KeyEvent

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
            "completed_steps": sum(
                1 for s in self.persistent.steps
                if s.status.value == "completed"
            ),
            "key_events": len(self.progress.key_events),
            "working_frames": len(self.working.frames),
            "working_capacity": self.working.capacity,
            "progress_snapshots": len(self.progress.progress_summaries),
        }
```

**Step 2: Verify + Commit**

---

### Task 1.7: Implement ContextCompressor

**Files:**
- Create: `agent/ar_proactive/memory/compressor.py`

**What it does:** SUMMARIZE_AND_DROP strategy from ProAssist — when working memory or key events grow too large, compress older entries into a ProgressSnapshot via LLM. Preserves task progress semantics while reducing token count.

```python
"""Context Compressor — SUMMARIZE_AND_DROP strategy.

When memory grows too large, compress older entries into a text summary.
Inspired by ProAssist's progress_summary mechanism.
"""

import logging
from typing import Optional

from agent.src.llm.base import LLMClientBase
from .types import ProgressSnapshot, KeyEvent

logger = logging.getLogger(__name__)

COMPRESS_SYSTEM = """\
You are a progress summarizer for a task assistant. Compress the following \
events into a concise summary that preserves: what was done, key objects used, \
and any errors or hazards encountered. Be brief (2-3 sentences).\
"""


class ContextCompressor:
    """Compress older memory entries into text summaries."""

    def __init__(self, llm_client: LLMClientBase, max_tokens: int = 500):
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    def compress_events(self, events: list[KeyEvent]) -> Optional[ProgressSnapshot]:
        """Compress a list of key events into a ProgressSnapshot."""
        if len(events) < 3:
            return None

        events_text = "\n".join(
            f"- [{e.timestamp:.1f}s] ({e.event_type}) {e.description}"
            for e in events
        )

        user_prompt = (
            f"Compress these {len(events)} events:\n\n{events_text}\n\n"
            "Return JSON: {\"summary\": \"...\", \"key_events\": [\"...\"], "
            "\"steps_completed\": [0, 1, ...]}"
        )

        try:
            result = self.llm_client.complete_json(
                system_prompt=COMPRESS_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )

            return ProgressSnapshot(
                time_range_start=events[0].timestamp,
                time_range_end=events[-1].timestamp,
                summary_text=result.get("summary", ""),
                steps_completed=result.get("steps_completed", []),
                key_events_summary=result.get("key_events", []),
            )
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return None
```

**Step 2: Verify + Commit**

---

### Task 1.8: Update memory __init__.py

**Files:**
- Modify: `agent/ar_proactive/memory/__init__.py`

Update exports to reflect the new module structure.

```python
"""Memory module — three functional layers aligned with RQ2.

Layers:
    Persistent — task knowledge (never changes)
    Progress   — step tracking and key events (monotonically advances)
    Working    — recent frame observations (sliding window)
"""

from .types import FrameRecord, KeyEvent, TaskStep, StepStatus, ProgressSnapshot
from .persistent import PersistentMemory
from .progress import ProgressMemory
from .working import WorkingMemory
from .manager import MemoryManager
from .compressor import ContextCompressor

__all__ = [
    "FrameRecord", "KeyEvent", "TaskStep", "StepStatus", "ProgressSnapshot",
    "PersistentMemory", "ProgressMemory", "WorkingMemory",
    "MemoryManager", "ContextCompressor",
]
```

**Step 2: Verify all imports**

Run: `python3 -c "from agent.ar_proactive.memory import MemoryManager, ContextCompressor, StreamingContext; print('OK')"`

**Step 3: Commit**

```bash
git commit -m "feat(memory): complete three-layer memory system (RQ2)"
```

---

## Phase 2: Task Understanding (RQ1 + RQ2)

> Add the ability to identify what task the user is performing and track step-by-step progress. This is the prerequisite for anticipatory interventions.

### Task 2.1: Task understanding prompts

**Files:**
- Create: `agent/ar_proactive/task/__init__.py`
- Create: `agent/ar_proactive/task/prompts.py`

The prompts module contains system/user prompts for:
1. **Task identification**: Given initial frames, identify what task the user is performing
2. **Step detection**: Given current frame + task steps, detect which step the user is on

```python
# task/prompts.py
"""Prompts for task understanding — identification and step detection."""

TASK_IDENTIFICATION_SYSTEM = """\
You are a task identification module for AR smart glasses. Given the first \
few frames of an ego-centric video, identify what procedural task the user \
is performing and decompose it into sequential steps.

Focus on observable steps. Be specific about key objects and actions for each step.\
"""

TASK_IDENTIFICATION_USER = """\
Analyze these initial frames from smart glasses and identify the procedural task.

Return JSON:
{{
  "task_goal": "Brief description of the overall task",
  "task_type": "cooking|assembly|navigation|shopping|social|other",
  "confidence": 0.0-1.0,
  "steps": [
    {{
      "description": "Step description",
      "key_objects": ["object1", "object2"],
      "key_actions": ["action1", "action2"]
    }}
  ]
}}

If you cannot identify a clear procedural task, set task_type to "other" \
and provide a general description with an empty steps list.

Respond with valid JSON only.\
"""

STEP_DETECTION_SYSTEM = """\
You are a progress tracker for AR smart glasses. Given the current frame \
and the task's step list, determine which step the user is currently on.\
"""


def build_step_detection_prompt(
    current_scene: str,
    step_list: list[dict],
    current_step: int,
    cumulative_state: str,
) -> str:
    """Build the prompt for step detection."""
    steps_text = "\n".join(
        f"  Step {s['index']+1}: {s['description']} "
        f"(objects: {', '.join(s.get('key_objects', []))})"
        for s in step_list
    )

    return f"""\
Current scene: {current_scene}
Previous cumulative state: {cumulative_state}
Current step (before this frame): {current_step + 1}

Task steps:
{steps_text}

Which step is the user currently on? Has there been a step transition?

Return JSON:
{{
  "detected_step_index": 0,
  "confidence": 0.0-1.0,
  "step_changed": false,
  "cumulative_state": "Updated cumulative state description",
  "error_detected": false,
  "error_description": ""
}}

Respond with valid JSON only.\
"""
```

---

### Task 2.2: TaskKnowledgeExtractor

**Files:**
- Create: `agent/ar_proactive/task/knowledge.py`

**What it does:** Analyzes the first few frames of a video to identify the task and extract a step-by-step procedure. Populates PersistentMemory.

```python
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
```

---

### Task 2.3: TaskTracker

**Files:**
- Create: `agent/ar_proactive/task/tracker.py`

**What it does:** Per-frame step detection — compares current scene to expected steps, detects step transitions and errors. Updates ProgressMemory.

```python
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
```

**Step 2: Verify + Commit**

```bash
git commit -m "feat(task): add TaskKnowledgeExtractor + TaskTracker (RQ1+RQ2)"
```

---

## Phase 3: Intervention Refactoring (RQ1)

> Replace the rule-based importance scorer with a principled TriggerDecider that distinguishes reactive and anticipatory interventions.

### Task 3.1: Update InterventionType with modes

**Files:**
- Modify: `agent/ar_proactive/intervention/types.py`

Add `InterventionMode` enum and update `Intervention` dataclass:

```python
"""Intervention types — includes reactive/anticipatory modes (RQ1)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InterventionMode(str, Enum):
    """How the intervention was triggered (RQ1 distinction)."""
    ANTICIPATORY = "anticipatory"  # proactive: next step, intent prediction
    REACTIVE = "reactive"          # error correction, hazard response
    SIGNAL = "signal"              # triggered by physiological anomaly


class InterventionType(str, Enum):
    """What kind of intervention to show."""
    NAVIGATION_HELP = "navigation_help"
    OBJECT_INFO = "object_info"
    SAFETY_WARNING = "safety_warning"
    TASK_GUIDANCE = "task_guidance"
    SOCIAL_CUE = "social_cue"
    CONTEXTUAL_TIP = "contextual_tip"
    STEP_INSTRUCTION = "step_instruction"    # NEW: next step guidance
    ERROR_CORRECTION = "error_correction"    # NEW: mistake correction


@dataclass
class Intervention:
    """A single proactive intervention decision."""
    timestamp: float
    intervention_type: InterventionType
    intervention_mode: InterventionMode  # NEW
    confidence: float
    content: str
    reasoning: str = ""
    trigger_factors: list[str] = field(default_factory=list)
    priority: str = "medium"
    related_step: Optional[int] = None  # NEW: which task step this relates to
    ui_component: Optional[dict] = None  # NEW: A2UI JSON (RQ3)

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "intervention_type": self.intervention_type.value,
            "intervention_mode": self.intervention_mode.value,
            "confidence": round(self.confidence, 3),
            "content": self.content,
            "reasoning": self.reasoning,
            "trigger_factors": self.trigger_factors,
            "priority": self.priority,
        }
        if self.related_step is not None:
            d["related_step"] = self.related_step
        if self.ui_component is not None:
            d["ui_component"] = self.ui_component
        return d
```

---

### Task 3.2: Implement TriggerDecider

**Files:**
- Create: `agent/ar_proactive/intervention/trigger.py`

**What it does:** Replaces `ImportanceScorer` + the gate logic in `InterventionEngine`. Uses StreamingContext to make principled trigger decisions across three channels: anticipatory, reactive, and signal-based.

```python
"""Trigger Decider — principled intervention timing (RQ1).

Replaces the old ImportanceScorer + gate logic with three
independent trigger channels:
  1. Anticipatory: task progress suggests next-step guidance needed
  2. Reactive: error or hazard detected
  3. Signal: physiological anomaly (gaze fixation, HR/EDA spike)
"""

import logging
from typing import Optional
from ..context import StreamingContext

logger = logging.getLogger(__name__)


class TriggerDecision:
    """Result of the trigger evaluation."""
    def __init__(self):
        self.should_trigger = False
        self.mode: str = ""  # anticipatory, reactive, signal
        self.reasons: list[str] = []
        self.priority: str = "medium"

    def __bool__(self):
        return self.should_trigger


class TriggerDecider:
    """Evaluate whether the current frame warrants an intervention.

    Three independent channels:
      Anticipatory — task progress + context suggest proactive help
      Reactive     — error detected or hazard present
      Signal       — physiological anomaly
    """

    def __init__(
        self,
        cooldown_sec: float = 2.0,
        visual_change_threshold: float = 0.05,
    ):
        self.cooldown_sec = cooldown_sec
        self.visual_change_threshold = visual_change_threshold
        self._last_trigger_time: Optional[float] = None

    def evaluate(self, ctx: StreamingContext, task_update: dict) -> TriggerDecision:
        """Evaluate all trigger channels.

        Args:
            ctx: Current StreamingContext.
            task_update: Result from TaskTracker.update() (step_changed, error_detected, etc.)

        Returns:
            TriggerDecision with should_trigger, mode, reasons, priority.
        """
        decision = TriggerDecision()

        # Gate: cooldown
        if self._last_trigger_time is not None:
            if (ctx.timestamp - self._last_trigger_time) < self.cooldown_sec:
                return decision

        # Channel 1: Reactive — error or hazard
        if task_update.get("error_detected"):
            decision.should_trigger = True
            decision.mode = "reactive"
            decision.priority = "high"
            decision.reasons.append("Error detected in task execution")

        if ctx.current_frame and ctx.current_frame.potential_hazards:
            decision.should_trigger = True
            decision.mode = "reactive"
            decision.priority = "high"
            decision.reasons.append(
                f"Hazard: {', '.join(ctx.current_frame.potential_hazards)}"
            )

        # Channel 2: Anticipatory — step transition or approaching next step
        if task_update.get("step_changed"):
            decision.should_trigger = True
            decision.mode = "anticipatory"
            decision.reasons.append("Step transition detected — provide next step guidance")

        # Channel 3: Signal — physiological anomaly
        signals = ctx.signal_context
        if signals.get("gaze_fixation"):
            decision.should_trigger = True
            if not decision.mode:
                decision.mode = "signal"
            decision.reasons.append(
                f"Gaze fixation ({signals.get('fixation_duration_ms', '?')}ms)"
            )

        if signals.get("hr_spike"):
            decision.should_trigger = True
            if not decision.mode:
                decision.mode = "signal"
            decision.priority = "high"
            decision.reasons.append("Heart rate spike detected")

        if signals.get("eda_spike"):
            decision.should_trigger = True
            if not decision.mode:
                decision.mode = "signal"
            decision.reasons.append("EDA spike detected (stress/arousal)")

        # Fallback: significant visual change with no prior context
        if (not decision.should_trigger
                and ctx.visual_change_score > 0.5
                and ctx.timestamp > 0):
            decision.should_trigger = True
            decision.mode = "anticipatory"
            decision.reasons.append("Major scene change")

        if decision.should_trigger:
            self._last_trigger_time = ctx.timestamp
            logger.debug(
                f"[{ctx.timestamp:.1f}s] TRIGGER ({decision.mode}): "
                f"{', '.join(decision.reasons)}"
            )

        return decision
```

---

### Task 3.3: Rewrite intervention prompts

**Files:**
- Rewrite: `agent/ar_proactive/intervention/prompts.py`

Task-aware prompts that receive StreamingContext and distinguish reactive/anticipatory modes.

```python
"""Task-aware intervention prompts (RQ1)."""

INTERVENTION_SYSTEM_PROMPT = """\
You are the decision module of AR smart glasses assisting a user with a task.

You have two intervention modes:
- **Anticipatory**: Proactively guide the user to the next step, provide helpful \
information before they need it. Use when a step transition is detected or approaching.
- **Reactive**: Correct an error or warn about a hazard. Use when something \
went wrong or is dangerous. These are high priority.

Guidelines:
- Be concise — messages appear on a small AR HUD (max 2 sentences).
- Consider task progress — don't repeat guidance for completed steps.
- Safety warnings always take highest priority.
- Avoid over-intervening — only help when there's a clear reason.\
"""


def build_intervention_prompt(
    context_text: str,
    trigger_mode: str,
    trigger_reasons: list[str],
    scene_description: str,
    signal_summary: str,
) -> str:
    """Build the user prompt for intervention decision."""
    return f"""\
## Trigger
Mode: {trigger_mode}
Reasons: {'; '.join(trigger_reasons)}

## Current Scene
{scene_description}

## Physiological Signals
{signal_summary}

## Full Context
{context_text}

## Decision
Based on the trigger and context, generate an appropriate intervention.
Return JSON:
{{
  "should_intervene": true,
  "intervention_type": "step_instruction|error_correction|safety_warning|task_guidance|object_info|navigation_help|social_cue|contextual_tip",
  "intervention_mode": "{trigger_mode}",
  "confidence": 0.0-1.0,
  "content": "Short message for AR HUD (max 2 sentences)",
  "reasoning": "Why this intervention",
  "trigger_factors": ["factor1", "factor2"],
  "priority": "low|medium|high",
  "related_step": null
}}

If the trigger is a false alarm and no intervention is actually needed, \
set should_intervene to false.
Respond with valid JSON only.\
"""
```

---

### Task 3.4: Rewrite InterventionEngine

**Files:**
- Rewrite: `agent/ar_proactive/intervention/engine.py`

Simplified engine that receives StreamingContext and TriggerDecision, calls LLM, and returns Intervention. No more internal gate logic (that's in TriggerDecider now).

```python
"""Intervention Engine — generates intervention content (RQ1)."""

import logging
from typing import Optional

from agent.src.llm.base import LLMClientBase
from ..context import StreamingContext
from .trigger import TriggerDecision
from .types import Intervention, InterventionType, InterventionMode
from .prompts import INTERVENTION_SYSTEM_PROMPT, build_intervention_prompt

logger = logging.getLogger(__name__)


class InterventionEngine:
    """Generate intervention content given a trigger decision and context.

    Simplified from the old version: gate logic moved to TriggerDecider.
    This class only handles the LLM call and output parsing.
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        min_confidence: float = 0.5,
        max_tokens: int = 1500,
    ):
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.max_tokens = max_tokens

    def generate(
        self,
        ctx: StreamingContext,
        trigger: TriggerDecision,
    ) -> Optional[Intervention]:
        """Generate intervention content.

        Only called when TriggerDecider says should_trigger=True.
        """
        if not trigger:
            return None

        scene_desc = self._format_scene(ctx)
        signal_summary = self._format_signals(ctx.signal_context)

        user_prompt = build_intervention_prompt(
            context_text=ctx.to_llm_context(),
            trigger_mode=trigger.mode,
            trigger_reasons=trigger.reasons,
            scene_description=scene_desc,
            signal_summary=signal_summary,
        )

        frame_images = []
        if ctx.current_frame and ctx.current_frame.frame_base64:
            frame_images = [ctx.current_frame.frame_base64]

        try:
            result = self.llm_client.complete_json_with_images(
                user_prompt=user_prompt,
                images=frame_images,
                system_prompt=INTERVENTION_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Intervention LLM call failed: {e}")
            return None

        if not result.get("should_intervene", False):
            logger.debug(f"[{ctx.timestamp:.1f}s] LLM vetoed intervention")
            return None

        confidence = float(result.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None

        try:
            itype = InterventionType(result.get("intervention_type", "contextual_tip"))
        except ValueError:
            itype = InterventionType.CONTEXTUAL_TIP

        try:
            imode = InterventionMode(result.get("intervention_mode", trigger.mode))
        except ValueError:
            imode = InterventionMode.ANTICIPATORY

        return Intervention(
            timestamp=ctx.timestamp,
            intervention_type=itype,
            intervention_mode=imode,
            confidence=confidence,
            content=result.get("content", ""),
            reasoning=result.get("reasoning", ""),
            trigger_factors=result.get("trigger_factors", []),
            priority=result.get("priority", trigger.priority),
            related_step=result.get("related_step"),
        )

    def _format_scene(self, ctx: StreamingContext) -> str:
        if not ctx.current_frame:
            return "No scene data"
        fr = ctx.current_frame
        lines = [f"Environment: {fr.environment}"]
        if fr.detected_objects:
            lines.append(f"Objects: {', '.join(fr.detected_objects)}")
        if fr.detected_activities:
            lines.append(f"Activities: {', '.join(fr.detected_activities)}")
        if fr.text_visible:
            lines.append(f"Visible text: {', '.join(fr.text_visible)}")
        if fr.potential_hazards:
            lines.append(f"Hazards: {', '.join(fr.potential_hazards)}")
        return "\n".join(lines)

    def _format_signals(self, signals: dict) -> str:
        parts = []
        if signals.get("gaze_fixation"):
            parts.append(f"Gaze fixation ({signals.get('fixation_duration_ms', '?')}ms)")
        if signals.get("hr_spike"):
            parts.append(f"HR spike: {signals.get('hr_value', '?')} bpm")
        if signals.get("eda_spike"):
            parts.append(f"EDA spike: {signals.get('eda_value', '?')}")
        return "; ".join(parts) if parts else "No anomalies."
```

**Commit:**
```bash
git commit -m "refactor(intervention): task-aware trigger + engine (RQ1)"
```

---

## Phase 4: UI Generation (RQ3)

> Connect the intervention system to A2UI component generation with lifecycle management.

### Task 4.1: UI types and state machine

**Files:**
- Create: `agent/ar_proactive/ui/__init__.py`
- Create: `agent/ar_proactive/ui/types.py`

```python
# ui/types.py
"""UI state types for lifecycle management (RQ3)."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class UIState(str, Enum):
    """UI component lifecycle states."""
    HIDDEN = "hidden"
    ENTERING = "entering"
    ACTIVE = "active"
    UPDATING = "updating"
    ALERT = "alert"
    EXITING = "exiting"


@dataclass
class UIEvent:
    """A UI state transition event."""
    timestamp: float
    from_state: UIState
    to_state: UIState
    component: Optional[dict] = None  # A2UI JSON
    reason: str = ""
```

---

### Task 4.2: UIStateManager

**Files:**
- Create: `agent/ar_proactive/ui/state_manager.py`

```python
"""UI State Manager — lifecycle management for AR UI components (RQ3).

Manages the state machine:
  HIDDEN → ENTERING → ACTIVE → UPDATING/ALERT → EXITING → HIDDEN

Key responsibilities:
  - Prevent unnecessary regeneration (content consistency)
  - Handle alert overlays (safety warnings)
  - Track component display duration for timeout
"""

import logging
from typing import Optional

from .types import UIState, UIEvent
from ..context import StreamingContext
from ..intervention.types import Intervention

logger = logging.getLogger(__name__)


class UIStateManager:
    """Manage the lifecycle of AR UI components."""

    def __init__(self, display_timeout_sec: float = 10.0):
        self.state = UIState.HIDDEN
        self.current_component: Optional[dict] = None
        self.current_intervention: Optional[Intervention] = None
        self.display_start_time: Optional[float] = None
        self.display_timeout_sec = display_timeout_sec
        self.event_log: list[UIEvent] = []

    def process(
        self,
        timestamp: float,
        intervention: Optional[Intervention],
        ctx: StreamingContext,
    ) -> Optional[UIEvent]:
        """Process a frame and return a UI event if state changed.

        Args:
            timestamp: Current timestamp.
            intervention: New intervention (or None).
            ctx: Current StreamingContext.

        Returns:
            UIEvent if a state transition occurred, None otherwise.
        """
        event = None

        if self.state == UIState.HIDDEN:
            if intervention:
                event = self._transition(timestamp, UIState.ACTIVE,
                                        reason=f"New intervention: {intervention.intervention_type.value}")
                self.current_intervention = intervention
                self.display_start_time = timestamp

        elif self.state == UIState.ACTIVE:
            if intervention:
                if intervention.priority == "high":
                    # High priority (safety) → alert overlay
                    event = self._transition(timestamp, UIState.ALERT,
                                            reason="High priority intervention")
                    self.current_intervention = intervention
                elif self._is_content_different(intervention):
                    # Content changed → update
                    event = self._transition(timestamp, UIState.UPDATING,
                                            reason="Content update")
                    self.current_intervention = intervention
                # else: same content, stay ACTIVE (no unnecessary regeneration)

            elif self._is_timed_out(timestamp):
                event = self._transition(timestamp, UIState.EXITING,
                                        reason="Display timeout")

        elif self.state == UIState.ALERT:
            if not intervention or intervention.priority != "high":
                # Alert resolved → back to active or hidden
                if self.current_intervention:
                    event = self._transition(timestamp, UIState.ACTIVE,
                                            reason="Alert resolved")
                else:
                    event = self._transition(timestamp, UIState.EXITING,
                                            reason="Alert resolved, no base UI")

        elif self.state == UIState.UPDATING:
            event = self._transition(timestamp, UIState.ACTIVE,
                                    reason="Update complete")

        elif self.state == UIState.EXITING:
            event = self._transition(timestamp, UIState.HIDDEN,
                                    reason="Exit complete")
            self.current_intervention = None
            self.current_component = None

        return event

    def _transition(self, timestamp: float, to_state: UIState, reason: str = "") -> UIEvent:
        event = UIEvent(
            timestamp=timestamp,
            from_state=self.state,
            to_state=to_state,
            component=self.current_component,
            reason=reason,
        )
        logger.debug(f"[{timestamp:.1f}s] UI: {self.state.value} → {to_state.value} ({reason})")
        self.state = to_state
        self.event_log.append(event)
        return event

    def _is_timed_out(self, timestamp: float) -> bool:
        if self.display_start_time is None:
            return False
        return (timestamp - self.display_start_time) >= self.display_timeout_sec

    def _is_content_different(self, new_intervention: Intervention) -> bool:
        if not self.current_intervention:
            return True
        return (new_intervention.content != self.current_intervention.content
                or new_intervention.intervention_type != self.current_intervention.intervention_type)
```

---

### Task 4.3: A2UIGenerator

**Files:**
- Create: `agent/ar_proactive/ui/generator.py`

**What it does:** Converts an `Intervention` into an A2UI component JSON using the existing component catalog. Maps intervention types to A2UI component structures.

```python
"""A2UI Generator — converts Interventions to A2UI component JSON (RQ3)."""

import uuid
import logging
from typing import Optional

from ..intervention.types import Intervention, InterventionType
from ..context import StreamingContext

logger = logging.getLogger(__name__)

# Mapping: InterventionType → A2UI component structure
COMPONENT_TEMPLATES = {
    InterventionType.STEP_INSTRUCTION: {
        "component": "Card",
        "variant": "glass",
        "children": [
            {"component": "Row", "children": [
                {"component": "Icon", "name": "arrowForward"},
                {"component": "Text", "variant": "h3", "text": "{title}"},
            ]},
            {"component": "Text", "variant": "body", "text": "{content}"},
        ],
    },
    InterventionType.SAFETY_WARNING: {
        "component": "Card",
        "variant": "alert",
        "children": [
            {"component": "Row", "children": [
                {"component": "Icon", "name": "warning"},
                {"component": "Text", "variant": "h3", "text": "Warning"},
            ]},
            {"component": "Text", "variant": "body", "text": "{content}"},
        ],
    },
    InterventionType.ERROR_CORRECTION: {
        "component": "Card",
        "variant": "outline",
        "children": [
            {"component": "Row", "children": [
                {"component": "Icon", "name": "error"},
                {"component": "Text", "variant": "h3", "text": "Correction"},
            ]},
            {"component": "Text", "variant": "body", "text": "{content}"},
        ],
    },
    InterventionType.TASK_GUIDANCE: {
        "component": "Card",
        "variant": "glass",
        "children": [
            {"component": "Text", "variant": "h3", "text": "{title}"},
            {"component": "Text", "variant": "body", "text": "{content}"},
        ],
    },
}

# Fallback for types without specific templates
DEFAULT_TEMPLATE = {
    "component": "Card",
    "variant": "glass",
    "children": [
        {"component": "Badge", "text": "{type_label}", "variant": "info"},
        {"component": "Text", "variant": "body", "text": "{content}"},
    ],
}


class A2UIGenerator:
    """Generate A2UI component JSON from Intervention objects."""

    def generate(
        self,
        intervention: Intervention,
        ctx: StreamingContext,
    ) -> dict:
        """Convert an intervention to an A2UI component.

        Args:
            intervention: The intervention to render.
            ctx: Current streaming context (for progress info).

        Returns:
            A2UI-compliant component JSON dict.
        """
        template = COMPONENT_TEMPLATES.get(
            intervention.intervention_type, DEFAULT_TEMPLATE
        )

        # Build title based on context
        title = self._build_title(intervention, ctx)

        # Deep-copy and fill template
        component = self._fill_template(
            template,
            content=intervention.content,
            title=title,
            type_label=intervention.intervention_type.value.replace("_", " ").title(),
        )

        # Add metadata
        component["id"] = f"ar_{uuid.uuid4().hex[:8]}"
        component["metadata"] = {
            "timestamp": intervention.timestamp,
            "intervention_type": intervention.intervention_type.value,
            "intervention_mode": intervention.intervention_mode.value,
            "confidence": intervention.confidence,
            "priority": intervention.priority,
        }

        # Add progress bar if task has steps
        if ctx.total_steps > 0:
            component["children"].append({
                "component": "ProgressBar",
                "variant": "slim",
                "metadata": {
                    "current": ctx.completed_steps,
                    "total": ctx.total_steps,
                },
            })

        return component

    def _build_title(self, intervention: Intervention, ctx: StreamingContext) -> str:
        """Build a contextual title for the component."""
        if intervention.related_step is not None and ctx.step_descriptions:
            step_idx = intervention.related_step
            if 0 <= step_idx < len(ctx.step_descriptions):
                return f"Step {step_idx + 1}: {ctx.step_descriptions[step_idx]}"
        return intervention.intervention_type.value.replace("_", " ").title()

    def _fill_template(self, template: dict, **kwargs) -> dict:
        """Recursively fill {placeholder} values in a template."""
        if isinstance(template, str):
            for key, value in kwargs.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        elif isinstance(template, dict):
            return {k: self._fill_template(v, **kwargs) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._fill_template(item, **kwargs) for item in template]
        return template
```

**Commit:**
```bash
git commit -m "feat(ui): add UIStateManager + A2UIGenerator (RQ3)"
```

---

## Phase 5: Agent Orchestrator Rewrite + CLI

> Rewire the entire agent loop to use the new architecture.

### Task 5.1: Update config.py

**Files:**
- Modify: `agent/ar_proactive/config.py`

Add new config fields for task tracking and UI generation. Remove old importance weight fields.

Key additions:
```python
# ── Task ─────────────────────────────────────
task_identification_frames: int = 3  # frames used for initial task ID
step_detection_interval: int = 3     # check every N frames
step_visual_change_threshold: float = 0.15

# ── UI (RQ3) ─────────────────────────────────
ui_display_timeout_sec: float = 10.0

# ── Memory ───────────────────────────────────
working_memory_capacity: int = 8
max_key_events: int = 50
compress_events_threshold: int = 30
```

Remove: `weight_visual_change`, `weight_semantic_novelty`, `weight_signal_anomaly`, `weight_scene_transition`, `short_term_capacity`, `mid_term_capacity`, `long_term_summary_every` and associated `__post_init__` weight validation.

---

### Task 5.2: Enhance SceneAnalyzer output

**Files:**
- Modify: `agent/ar_proactive/video/scene_analyzer.py`

Update the scene analysis prompt to return `current_action` field (what the user is actively doing right now), useful for step detection.

Add to JSON schema:
```json
"current_action": "what the wearer is actively doing right now"
```

And update `_normalize()` to include `current_action`.

---

### Task 5.3: Rewrite agent.py

**Files:**
- Rewrite: `agent/ar_proactive/agent.py`

The new orchestrator with the complete Observe → Think → Act loop using all new modules.

Key changes:
1. **Init**: Create MemoryManager, TaskKnowledgeExtractor, TaskTracker, TriggerDecider, InterventionEngine, UIStateManager, A2UIGenerator
2. **Task identification phase**: Before the main loop, extract first N frames, call TaskKnowledgeExtractor to populate PersistentMemory
3. **Main loop**:
   - OBSERVE: same as before (frame extraction, visual change, scene analysis, signals)
   - THINK: build FrameRecord → add to WorkingMemory → TaskTracker.update() → assemble StreamingContext → TriggerDecider.evaluate()
   - ACT: if triggered → InterventionEngine.generate() → A2UIGenerator.generate() → UIStateManager.process()
4. **Output**: interventions now include `ui_component` and `intervention_mode`

---

### Task 5.4: Update __main__.py

**Files:**
- Modify: `agent/ar_proactive/__main__.py`

Add new CLI flags:
```
--no-task-tracking    Disable task identification and step tracking
--ui-timeout          UI display timeout in seconds (default: 10.0)
```

Update summary output to show task info and step progress.

---

### Task 5.5: Update __init__.py

**Files:**
- Modify: `agent/ar_proactive/__init__.py`

Update exports.

---

### Task 5.6: Delete old files

**Files:**
- Delete: `agent/ar_proactive/memory/store.py`
- Delete: `agent/ar_proactive/memory/importance.py`
- Delete: `agent/ar_proactive/memory/retriever.py`

```bash
git rm agent/ar_proactive/memory/store.py \
      agent/ar_proactive/memory/importance.py \
      agent/ar_proactive/memory/retriever.py
```

---

### Task 5.7: End-to-end test

Run the full pipeline on example data:

```bash
python3 -m agent.ar_proactive \
    --sample agent/example/Task2.1/P10_Ernesto/sample_001 \
    --frame-interval 1.0 -v
```

Verify:
1. Task identification runs on initial frames
2. Step tracking (if task identified) updates progress
3. TriggerDecider makes principled decisions (anticipatory/reactive/signal)
4. Interventions include `intervention_mode` and `ui_component`
5. UIStateManager transitions logged
6. Output JSON contains full structured data

**Final commit:**
```bash
git commit -m "feat: complete RQ-driven refactor — task-aware AR agent with UI generation"
```

---

## Summary of Changes

| Component | Before | After | RQ |
|---|---|---|---|
| Memory architecture | 3 tiers by importance | 3 layers by function | RQ2 |
| Memory types | MemoryEntry (flat) | FrameRecord + KeyEvent + TaskStep | RQ2 |
| Context bridge | None | StreamingContext | RQ2 |
| Task understanding | None | TaskKnowledgeExtractor + TaskTracker | RQ1+RQ2 |
| Intervention timing | ImportanceScorer (4 weights) | TriggerDecider (3 channels) | RQ1 |
| Intervention modes | None | Anticipatory / Reactive / Signal | RQ1 |
| UI output | Text-only Intervention | A2UI component JSON | RQ3 |
| UI lifecycle | None | UIStateManager (6-state machine) | RQ3 |
| Cost control | Importance threshold gate | TaskTracker skips + TriggerDecider gates | RQ1 |
