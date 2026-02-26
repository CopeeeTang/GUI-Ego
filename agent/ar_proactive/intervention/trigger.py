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
            task_update: Result from TaskTracker.update().

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
