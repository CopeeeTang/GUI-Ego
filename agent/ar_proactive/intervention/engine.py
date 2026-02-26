"""Intervention Engine — generates intervention content (RQ1).

Simplified from the old version: gate logic moved to TriggerDecider.
This class only handles the LLM call and output parsing.
"""

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
