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
