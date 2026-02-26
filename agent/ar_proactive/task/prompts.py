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
