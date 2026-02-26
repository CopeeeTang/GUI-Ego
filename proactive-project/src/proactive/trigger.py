"""
Proactive Trigger Detection

Detects when the system should proactively intervene during streaming.
Multiple strategies from simple baselines to VLM-based detection.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from src.models.base import BaseVLM
from src.streaming.stream_simulator import StreamFrame

logger = logging.getLogger(__name__)


@dataclass
class TriggerDecision:
    """Result of trigger detection for a single frame."""
    should_trigger: bool
    confidence: float = 0.0         # 0-1 confidence score
    reason: str = ""                # why triggered
    latency_ms: float = 0.0        # detection time
    method: str = ""


class TriggerDetector(ABC):
    """Base class for trigger detection strategies."""

    @abstractmethod
    def detect(self, frame: StreamFrame, history: list[StreamFrame]) -> TriggerDecision:
        ...

    @abstractmethod
    def reset(self):
        """Reset state for a new session."""
        ...


class PeriodicTrigger(TriggerDetector):
    """Baseline: trigger every N seconds."""

    def __init__(self, interval_sec: float = 15.0):
        self.interval = interval_sec
        self.last_trigger_time = -float("inf")

    def detect(self, frame: StreamFrame, history: list[StreamFrame]) -> TriggerDecision:
        elapsed = frame.timestamp - self.last_trigger_time
        if elapsed >= self.interval:
            self.last_trigger_time = frame.timestamp
            return TriggerDecision(
                should_trigger=True,
                confidence=1.0,
                reason=f"Periodic trigger (every {self.interval}s)",
                method="periodic",
            )
        return TriggerDecision(should_trigger=False, method="periodic")

    def reset(self):
        self.last_trigger_time = -float("inf")


class ActionBoundaryTrigger(TriggerDetector):
    """
    Oracle baseline: trigger at action boundaries using GT annotations.
    Useful as an upper bound for trigger timing.
    """

    def __init__(self, min_interval_sec: float = 5.0):
        self.min_interval = min_interval_sec
        self.last_trigger_time = -float("inf")
        self.last_action_label: Optional[str] = None

    def detect(self, frame: StreamFrame, history: list[StreamFrame]) -> TriggerDecision:
        current_label = frame.current_action.action_label if frame.current_action else None

        # Trigger on action change
        if current_label and current_label != self.last_action_label:
            elapsed = frame.timestamp - self.last_trigger_time
            self.last_action_label = current_label

            if elapsed >= self.min_interval:
                self.last_trigger_time = frame.timestamp
                return TriggerDecision(
                    should_trigger=True,
                    confidence=1.0,
                    reason=f"Action changed to: {current_label}",
                    method="action_boundary",
                )

        if current_label:
            self.last_action_label = current_label

        return TriggerDecision(should_trigger=False, method="action_boundary")

    def reset(self):
        self.last_trigger_time = -float("inf")
        self.last_action_label = None


class VLMDeltaTrigger(TriggerDetector):
    """
    VLM-based trigger: ask the VLM to detect significant changes.
    Compares current frame context with recent history to detect
    step transitions, safety concerns, or idle states.
    """

    TRIGGER_PROMPT = """You are a cooking assistant monitoring a first-person cooking video stream.

Analyze the current frame and recent context. Determine if you should proactively intervene.

Trigger conditions (answer YES for any):
1. STEP_TRANSITION: The cook has clearly moved to a different task/action
2. SAFETY_WARNING: The cook is about to do something potentially dangerous (cutting, using hot surfaces, pouring liquids)
3. IDLE: The cook appears to be idle or confused about what to do next
4. PROGRESS: A significant milestone has been reached

Recent actions observed: {recent_actions}
Time since last intervention: {time_since_last:.0f}s

Respond in JSON:
{{"trigger": true/false, "type": "step_transition|safety_warning|idle|progress|none", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    def __init__(
        self,
        vlm: BaseVLM,
        min_interval_sec: float = 5.0,
        check_every_n_frames: int = 4,  # don't call VLM on every frame
    ):
        self.vlm = vlm
        self.min_interval = min_interval_sec
        self.check_interval = check_every_n_frames
        self.last_trigger_time = -float("inf")
        self.frame_count = 0
        self.recent_actions: list[str] = []

    def detect(self, frame: StreamFrame, history: list[StreamFrame]) -> TriggerDecision:
        self.frame_count += 1

        # Track actions
        if frame.current_action:
            label = frame.current_action.action_label
            if not self.recent_actions or self.recent_actions[-1] != label:
                self.recent_actions.append(label)
                if len(self.recent_actions) > 5:
                    self.recent_actions = self.recent_actions[-5:]

        # Only check every N frames to save compute
        if self.frame_count % self.check_interval != 0:
            return TriggerDecision(should_trigger=False, method="vlm_delta")

        # Rate limit
        time_since = frame.timestamp - self.last_trigger_time
        if time_since < self.min_interval:
            return TriggerDecision(should_trigger=False, method="vlm_delta")

        # Build prompt
        from src.streaming.frame_processor import FrameProcessor
        pil_img = FrameProcessor.frame_to_pil(frame.frame)

        prompt = self.TRIGGER_PROMPT.format(
            recent_actions=", ".join(self.recent_actions[-3:]) or "none observed",
            time_since_last=time_since,
        )

        t0 = time.time()
        try:
            result = self.vlm.generate_json(prompt, images=[pil_img])
            latency = (time.time() - t0) * 1000

            should_trigger = result.get("trigger", False)
            if should_trigger:
                self.last_trigger_time = frame.timestamp

            return TriggerDecision(
                should_trigger=should_trigger,
                confidence=float(result.get("confidence", 0.5)),
                reason=result.get("reason", ""),
                latency_ms=latency,
                method="vlm_delta",
            )
        except Exception as e:
            logger.warning(f"VLM trigger detection failed: {e}")
            return TriggerDecision(should_trigger=False, method="vlm_delta")

    def reset(self):
        self.last_trigger_time = -float("inf")
        self.frame_count = 0
        self.recent_actions = []


def create_trigger(method: str, vlm: Optional[BaseVLM] = None, **kwargs) -> TriggerDetector:
    """Factory function for trigger detectors."""
    if method == "periodic":
        return PeriodicTrigger(**kwargs)
    elif method == "action_boundary":
        return ActionBoundaryTrigger(**kwargs)
    elif method == "vlm_delta":
        if vlm is None:
            raise ValueError("VLM required for vlm_delta trigger")
        return VLMDeltaTrigger(vlm=vlm, **kwargs)
    else:
        raise ValueError(f"Unknown trigger method: {method}")
