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
    STEP_INSTRUCTION = "step_instruction"    # next step guidance
    ERROR_CORRECTION = "error_correction"    # mistake correction


@dataclass
class Intervention:
    """A single proactive intervention decision."""
    timestamp: float
    intervention_type: InterventionType
    intervention_mode: InterventionMode
    confidence: float
    content: str
    reasoning: str = ""
    trigger_factors: list[str] = field(default_factory=list)
    priority: str = "medium"
    related_step: Optional[int] = None     # which task step this relates to
    ui_component: Optional[dict] = None    # A2UI JSON (RQ3, populated later)

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
