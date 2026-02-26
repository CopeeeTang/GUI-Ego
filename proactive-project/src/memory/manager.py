"""
Memory Manager

Orchestrates the three-layer memory system:
  Layer 1: Task Memory (always in context)
  Layer 2: Event Memory (retrieved on demand)
  Layer 3: Visual Memory (frame buffer)

Provides unified interface for the streaming pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from data.egtea_loader import ActionClip, CookingSession
from src.memory.event_memory import EventMemory
from src.memory.task_memory import TaskMemory
from src.memory.visual_memory import VisualMemory
from src.models.base import BaseVLM

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Combined context from all three memory layers."""
    task_context: str               # Layer 1: always present
    event_context: str = ""         # Layer 2: retrieved on demand
    visual_frames: list = None      # Layer 3: recent PIL images

    def __post_init__(self):
        if self.visual_frames is None:
            self.visual_frames = []

    def to_prompt(self) -> str:
        """Combine all memory layers into a single context string."""
        parts = [self.task_context]
        if self.event_context:
            parts.append(self.event_context)
        return "\n\n".join(parts)


class MemoryManager:
    """
    Unified memory manager for streaming video understanding.

    Handles:
    - Automatic frame storage (Layer 3)
    - Event recording on action changes (Layer 2)
    - Task state tracking (Layer 1)
    - Context assembly for VLM queries
    """

    def __init__(
        self,
        task_memory: Optional[TaskMemory] = None,
        event_memory: Optional[EventMemory] = None,
        visual_memory: Optional[VisualMemory] = None,
    ):
        self.task = task_memory or TaskMemory()
        self.events = event_memory or EventMemory()
        self.visual = visual_memory or VisualMemory()

        self._last_action_label: Optional[str] = None

    def initialize(self, session: CookingSession):
        """Initialize all memory layers for a new session."""
        self.reset()
        self.task.initialize_from_session(session)
        logger.info(f"Memory initialized for session: {session.session_id}")

    def process_frame(
        self,
        timestamp: float,
        frame: np.ndarray,
        current_action: Optional[ActionClip] = None,
    ):
        """
        Process a new streaming frame through all memory layers.

        Called for every frame in the stream. Handles:
        - Storing frame in visual memory (Layer 3)
        - Detecting action changes and recording events (Layer 2)
        - Updating task progress (Layer 1)
        """
        action_label = current_action.action_label if current_action else None

        # Layer 3: Always store frame
        self.visual.add_frame(timestamp, frame, action_label)

        # Layer 2 + Layer 1: Update on action change
        if current_action and action_label != self._last_action_label:
            # New action detected → record event
            self.events.add_from_action(current_action, timestamp)
            # Update task progress
            self.task.update_from_gt(current_action, timestamp)

            self._last_action_label = action_label

    def get_context(self, query: Optional[str] = None,
                    include_visual: bool = False,
                    visual_frames_n: int = 4) -> MemoryContext:
        """
        Assemble context from all memory layers.

        Args:
            query: If provided, retrieve relevant events from Layer 2
            include_visual: Whether to include recent frames from Layer 3
            visual_frames_n: Number of recent frames to include
        """
        # Layer 1: always present
        task_ctx = self.task.get_context()

        # Layer 2: retrieve if query provided
        event_ctx = ""
        if query:
            events = self.events.retrieve(query, top_k=5)
            event_ctx = self.events.to_context_string(events)

        # Layer 3: recent frames
        visual_frames = []
        if include_visual:
            visual_frames = self.visual.get_recent_pil_images(visual_frames_n)

        return MemoryContext(
            task_context=task_ctx,
            event_context=event_ctx,
            visual_frames=visual_frames,
        )

    def query_with_vlm(self, vlm: BaseVLM, question: str,
                       include_visual: bool = True) -> str:
        """
        Answer a question using all memory layers + VLM.

        This is the primary interface for memory-based QA evaluation.
        """
        ctx = self.get_context(query=question, include_visual=include_visual)

        # Build a chronological event timeline for context
        recent_events = self.events.get_recent(n=10)
        timeline = ""
        if recent_events:
            timeline_lines = ["[Chronological Event Timeline]"]
            for e in recent_events:
                action_label = e.metadata.get("action_label", e.description)
                timeline_lines.append(f"  {e.timestamp:.1f}s: {action_label}")
            timeline = "\n".join(timeline_lines)

        prompt = f"""You are a cooking assistant that has been observing a cooking session through a first-person camera.
Use the memory context below to answer the question. Be specific and concise.

{ctx.task_context}

{timeline}

{ctx.event_context}

Question: {question}

Instructions:
- Answer based ONLY on the memory context provided above
- Use specific action names and timestamps when available
- If the question asks about "what action", respond with the action name (e.g., "Cut-Tomato")
- If the question asks about "when", include the timestamp
- Keep your answer to 1-2 sentences"""

        images = ctx.visual_frames if ctx.visual_frames else None
        response = vlm.generate(prompt, images=images)
        return response.text

    def summary(self) -> str:
        """Get summary of all memory layers."""
        lines = [
            "=== Memory Manager ===",
            self.task.get_context(),
            "",
            f"Event Memory: {self.events.size} events",
            self.visual.summary(),
        ]
        return "\n".join(lines)

    def reset(self):
        """Reset all memory layers."""
        self.task.reset()
        self.events.reset()
        self.visual.reset()
        self._last_action_label = None
