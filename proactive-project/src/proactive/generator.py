"""
Proactive Content Generator

Given a trigger decision, generates the actual intervention content
(what the system should tell the user).
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from src.models.base import BaseVLM
from src.streaming.frame_processor import FrameProcessor
from src.streaming.stream_simulator import StreamFrame

logger = logging.getLogger(__name__)


@dataclass
class Intervention:
    """A generated proactive intervention."""
    timestamp: float
    content: str
    trigger_type: str
    confidence: float
    latency_ms: float = 0.0
    model: str = ""


GENERATION_PROMPT = """You are an AR cooking assistant. Based on the current video frame, generate a brief, helpful intervention for the cook.

Trigger reason: {reason}
Recent actions: {recent_actions}

Guidelines:
- Be concise (1-2 sentences max)
- Be specific about what you see and what the cook should do next
- For safety warnings, be direct and clear
- For step transitions, mention what was just completed and what's next
- For idle reminders, gently suggest the next step

Generate the intervention text only, no formatting."""


class ContentGenerator:
    """Generate intervention content using a VLM."""

    def __init__(self, vlm: BaseVLM, max_tokens: int = 200):
        self.vlm = vlm
        self.max_tokens = max_tokens

    def generate(
        self,
        frame: StreamFrame,
        trigger_reason: str,
        recent_actions: list[str],
        trigger_type: str = "unknown",
        confidence: float = 0.5,
    ) -> Intervention:
        """Generate intervention content for a triggered frame."""
        pil_img = FrameProcessor.frame_to_pil(frame.frame)

        prompt = GENERATION_PROMPT.format(
            reason=trigger_reason,
            recent_actions=", ".join(recent_actions[-3:]) or "none",
        )

        t0 = time.time()
        response = self.vlm.generate(
            prompt=prompt,
            images=[pil_img],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )
        latency = (time.time() - t0) * 1000

        return Intervention(
            timestamp=frame.timestamp,
            content=response.text,
            trigger_type=trigger_type,
            confidence=confidence,
            latency_ms=latency,
            model=response.model,
        )


class TemplateGenerator:
    """
    Baseline: generate content from templates without VLM.
    Uses GT action annotations directly.
    """

    TEMPLATES = {
        "step_transition": "You've finished {completed}. The next step is to {next_action}. Get your {nouns} ready.",
        "safety_warning": "Safety notice: You are about to {verb} the {nouns}. Please be careful and use proper technique.",
        "idle": "It looks like you've paused. The next step in the recipe is to {next_action}.",
        "progress": "Good progress! You've completed {steps} steps so far. Keep going with the recipe.",
        "default": "You are currently working on {action}. Keep up the good work.",
    }

    def generate(
        self,
        frame: StreamFrame,
        trigger_type: str = "default",
        **kwargs,
    ) -> Intervention:
        template = self.TEMPLATES.get(trigger_type, self.TEMPLATES["default"])

        # Fill from frame context
        fill = {
            "action": frame.current_action.action_label if frame.current_action else "unknown",
            "verb": frame.current_action.verb if frame.current_action else "",
            "nouns": ", ".join(frame.current_action.nouns) if frame.current_action else "",
            "next_action": frame.next_action.action_label if frame.next_action else "unknown",
            "completed": frame.current_action.action_label if frame.current_action else "",
        }
        fill.update(kwargs)

        try:
            content = template.format(**fill)
        except KeyError:
            content = f"Current: {fill['action']}"

        return Intervention(
            timestamp=frame.timestamp,
            content=content,
            trigger_type=trigger_type,
            confidence=1.0,
            model="template",
        )
