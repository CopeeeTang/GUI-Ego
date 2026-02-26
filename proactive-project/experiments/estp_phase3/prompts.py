"""
Task-Semantic Enriched Prompts for Hypothesis A.

Core idea: The vanilla FBF trigger prompt "Is it the right time to answer?" lacks
task-specific semantic guidance. For implicit reasoning tasks (Object Function,
Information Function, Action Reasoning, Task Understanding), the model needs
domain-specific cues about WHAT to look for, not just WHETHER to answer.
"""

# ===== Vanilla FBF prompt (baseline) =====
VANILLA_TRIGGER = (
    'Is it the right time to answer the question "{question}"? '
    'You need to answer yes or no.'
)

VANILLA_ANSWER = 'Please answer the question: "{question}"'

PROMPT_TEMPLATE = (
    "You are an advanced image question-answering AI assistant. "
    "You have been provided with image and a question related to the images. "
    "Your task is to carefully analyze the images and provide the answer to the question. "
    "You need to carefully confirm whether the images content meet the conditions of "
    "the question, and then output the correct content.\n\n"
    "Question: {query}\n\nThe answer is:\n"
)


# ===== Hypothesis A: Task-Semantic Enriched Trigger Prompts =====

SEMANTIC_TRIGGER_BY_TASK = {
    "Object Function": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether any object's FUNCTION or USAGE is currently being "
        "demonstrated, or if an object relevant to the question is currently visible "
        "and being interacted with.\n\n"
        "Is the current moment the right time to answer this question? "
        "Answer yes ONLY if you can see clear visual evidence right now. "
        "Answer no if the relevant object/function hasn't appeared yet."
    ),
    "Information Function": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether any TEXT, LABELS, SIGNS, SCREENS, or INFORMATIONAL "
        "content relevant to this question is currently visible in the frame.\n\n"
        "Is the current moment the right time to answer? "
        "Answer yes ONLY if the information needed is visually present now. "
        "Answer no if the relevant text/sign/label hasn't appeared yet."
    ),
    "Action Reasoning": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the person's current ACTIONS or their CONSEQUENCES "
        "provide enough context to reason about this question. Think about "
        "cause-and-effect: has the triggering event happened?\n\n"
        "Is the current moment the right time to answer? "
        "Answer yes ONLY if you have enough action context to reason about the answer. "
        "Answer no if the key action sequence hasn't unfolded yet."
    ),
    "Task Understanding": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether you now have enough OVERALL CONTEXT about the task "
        "being performed — the sequence of steps, the tools being used, and the "
        "goal of the activity.\n\n"
        "Is the current moment the right time to answer? "
        "Answer yes if you can provide a meaningful answer about the task. "
        "Answer no if you need to observe more steps first."
    ),
    # Explicit tasks get a lighter-touch semantic hint
    "Object Recognition": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the OBJECT being asked about is currently visible "
        "and identifiable in the frame.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Attribute Perception": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the relevant object's ATTRIBUTES (color, shape, size, "
        "material, state) are clearly observable right now.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Text-Rich Understanding": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether TEXT or WRITTEN CONTENT relevant to this question "
        "is currently readable in the frame.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Object Localization": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the object's LOCATION or SPATIAL POSITION can be "
        "clearly determined from the current frame.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Object State Change Recognition": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether a STATE CHANGE of the relevant object (moved, opened, "
        "broken, transformed, etc.) is currently happening or just happened.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Ego Object Localization": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the object's position RELATIVE TO YOU (the camera wearer) "
        "can be clearly described from the current viewpoint.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Ego Object State Change Recognition": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether an object YOU are interacting with has just undergone "
        "a STATE CHANGE (picked up, put down, opened, moved, etc.).\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
    "Action Recognition": (
        "You are watching a first-person video stream. The user wants to know: "
        '"{question}"\n\n'
        "Focus on whether the person is currently PERFORMING the action being "
        "asked about, or if it just completed.\n\n"
        "Is the current moment the right time to answer? Answer yes or no."
    ),
}


# ===== Hypothesis C: Two-Stage Confidence Calibration =====

CONFIDENCE_CHECK_PROMPT = (
    "You just indicated it's the right time to answer. Before answering, "
    "rate your CONFIDENCE that this is truly the best moment:\n\n"
    "Question: \"{question}\"\n\n"
    "On a scale of 1-5:\n"
    "1 = Very uncertain, probably too early\n"
    "2 = Somewhat uncertain\n"
    "3 = Moderately confident\n"
    "4 = Confident, good visual evidence\n"
    "5 = Very confident, clear and unambiguous\n\n"
    "Reply with just the number (1-5):"
)

CONFIDENCE_THRESHOLD = 4  # Only answer if confidence >= this


def get_trigger_prompt(question, task_type, hypothesis="vanilla"):
    """Get the trigger prompt for a given hypothesis."""
    if hypothesis == "vanilla":
        return VANILLA_TRIGGER.format(question=question)
    elif hypothesis == "semantic":
        template = SEMANTIC_TRIGGER_BY_TASK.get(task_type, VANILLA_TRIGGER)
        return template.format(question=question)
    else:
        raise ValueError(f"Unknown hypothesis: {hypothesis}")


def get_answer_prompt(question):
    """Get the answer generation prompt."""
    return VANILLA_ANSWER.format(question=question)
