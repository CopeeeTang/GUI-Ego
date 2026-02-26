"""
LLM-as-Judge for intervention content quality evaluation.

Uses Gemini as an impartial evaluator to score intervention content
across multiple quality dimensions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Scoring Dimensions ───────────────────────────────────────────────

@dataclass
class ContentScore:
    """Multi-dimensional quality score for a single intervention."""

    intervention_index: int
    timestamp: float

    # Scores (1-5 Likert scale)
    relevance: int = 0       # Is it relevant to what the user is doing?
    accuracy: int = 0        # Is the factual content correct?
    helpfulness: int = 0     # Would this actually help the user?
    timing: int = 0          # Is the timing appropriate (not too early/late)?
    conciseness: int = 0     # Is it appropriately brief for AR HUD?

    # For safety interventions only
    safety_justified: Optional[bool] = None  # Was the safety warning warranted?

    # Judge's rationale
    rationale: str = ""

    # Context used
    gt_action: str = ""
    intervention_content: str = ""
    intervention_mode: str = ""

    @property
    def avg_score(self) -> float:
        scores = [self.relevance, self.accuracy, self.helpfulness,
                  self.timing, self.conciseness]
        valid = [s for s in scores if s > 0]
        return sum(valid) / max(1, len(valid))

    def to_dict(self) -> dict:
        d = {
            "intervention_index": self.intervention_index,
            "timestamp": self.timestamp,
            "relevance": self.relevance,
            "accuracy": self.accuracy,
            "helpfulness": self.helpfulness,
            "timing": self.timing,
            "conciseness": self.conciseness,
            "avg_score": round(self.avg_score, 2),
            "rationale": self.rationale,
            "gt_action": self.gt_action,
            "intervention_content": self.intervention_content,
            "intervention_mode": self.intervention_mode,
        }
        if self.safety_justified is not None:
            d["safety_justified"] = self.safety_justified
        return d


@dataclass
class ContentJudgeResult:
    """Aggregated result from judging all interventions in a session."""

    scores: list[ContentScore] = field(default_factory=list)

    @property
    def avg_relevance(self) -> float:
        vals = [s.relevance for s in self.scores if s.relevance > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_accuracy(self) -> float:
        vals = [s.accuracy for s in self.scores if s.accuracy > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_helpfulness(self) -> float:
        vals = [s.helpfulness for s in self.scores if s.helpfulness > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_timing(self) -> float:
        vals = [s.timing for s in self.scores if s.timing > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def avg_conciseness(self) -> float:
        vals = [s.conciseness for s in self.scores if s.conciseness > 0]
        return sum(vals) / max(1, len(vals))

    @property
    def overall_avg(self) -> float:
        return sum(s.avg_score for s in self.scores) / max(1, len(self.scores))

    @property
    def safety_precision(self) -> Optional[float]:
        """Of safety warnings, what fraction were justified?"""
        safety = [s for s in self.scores if s.safety_justified is not None]
        if not safety:
            return None
        justified = sum(1 for s in safety if s.safety_justified)
        return justified / len(safety)

    def to_dict(self) -> dict:
        d = {
            "num_evaluated": len(self.scores),
            "avg_relevance": round(self.avg_relevance, 2),
            "avg_accuracy": round(self.avg_accuracy, 2),
            "avg_helpfulness": round(self.avg_helpfulness, 2),
            "avg_timing": round(self.avg_timing, 2),
            "avg_conciseness": round(self.avg_conciseness, 2),
            "overall_avg": round(self.overall_avg, 2),
            "scores": [s.to_dict() for s in self.scores],
        }
        sp = self.safety_precision
        if sp is not None:
            d["safety_precision"] = round(sp, 3)
        return d


# ── Judge Prompts ────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a Proactive AR Smart Glasses assistance system.

The system watches a user performing a cooking task through egocentric video,
and proactively generates interventions (instructions, warnings, tips) displayed
on the AR HUD at appropriate moments.

Your job is to evaluate the QUALITY of each intervention the system generated.
You will be given:
  - The ground truth action the user was performing at that moment
  - The recipe/task context
  - The intervention content and its reasoning
  - The intervention mode (anticipatory = proactive next-step, reactive = error/hazard response)

Score each dimension on a 1-5 scale:
  1 = Very Poor, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent

Dimensions:
  - relevance: Is the intervention related to the current/next task step?
  - accuracy: Is the factual content (instructions, warnings) correct?
  - helpfulness: Would this realistically help a user doing this task?
  - timing: Is the timing appropriate? (anticipatory should be slightly ahead,
            reactive should be immediate)
  - conciseness: Is it brief enough for an AR HUD? (1-2 sentences ideal)

For safety_warning type interventions, also judge:
  - safety_justified: true/false - Is the hazard real and worth warning about?

Be strict but fair. A generic "be careful" is not helpful (score 2).
A specific, actionable instruction matching the current context scores 4-5.

Respond ONLY with valid JSON."""


def _build_judge_prompt(
    intervention: dict,
    gt_action: str,
    recipe: str,
    nearby_context: list[dict],
) -> str:
    """Build the evaluation prompt for a single intervention."""
    # Build context from nearby frames
    context_lines = []
    for ctx in nearby_context:
        ts = ctx.get("timestamp", 0)
        action = ctx.get("gt_action", "unknown")
        env = ctx.get("scene_environment", "")
        context_lines.append(f"  [{ts:.1f}s] Action: {action} | Scene: {env}")

    context_text = "\n".join(context_lines) if context_lines else "  No nearby context"

    is_safety = intervention.get("intervention_type") == "safety_warning"

    prompt = f"""## Evaluation Task

**Recipe/Task**: {recipe}
**Ground Truth Action at this moment**: {gt_action or "unknown (between actions)"}

**Nearby Frame Context** (surrounding frames):
{context_text}

**Intervention to Evaluate**:
  - Timestamp: {intervention.get("timestamp", 0):.1f}s
  - Type: {intervention.get("intervention_type", "unknown")}
  - Mode: {intervention.get("intervention_mode", "unknown")}
  - Content: "{intervention.get("content", "")}"
  - Reasoning: "{intervention.get("reasoning", "")}"
  - Priority: {intervention.get("priority", "unknown")}
  - Related Step: {intervention.get("related_step", "N/A")}

**Evaluate and return JSON**:
{{
  "relevance": <1-5>,
  "accuracy": <1-5>,
  "helpfulness": <1-5>,
  "timing": <1-5>,
  "conciseness": <1-5>,
  {"\"safety_justified\": <true/false>," if is_safety else ""}
  "rationale": "<brief explanation of scores>"
}}"""

    return prompt


# ── Judge Client ─────────────────────────────────────────────────────

class ContentJudge:
    """Uses Gemini as LLM-as-Judge for content quality evaluation."""

    def __init__(
        self,
        model_spec: str = "gemini:gemini-3-flash",
        temperature: float = 0.1,
    ):
        from agent.src.llm.factory import create_client
        self.client = create_client(model_spec)
        self.temperature = temperature
        logger.info(f"ContentJudge initialized with {model_spec}")

    def evaluate_session(self, result: dict) -> ContentJudgeResult:
        """Evaluate all interventions in a session result.

        Args:
            result: Full result dict from process_egtea_session().

        Returns:
            ContentJudgeResult with per-intervention scores.
        """
        interventions = result.get("interventions", [])
        session_log = result.get("session_log", [])
        recipe = result.get("recipe", "unknown")

        judge_result = ContentJudgeResult()

        for i, intv in enumerate(interventions):
            try:
                score = self._evaluate_single(
                    intervention=intv,
                    index=i,
                    session_log=session_log,
                    recipe=recipe,
                )
                judge_result.scores.append(score)
                logger.info(
                    f"  Intervention {i+1}/{len(interventions)}: "
                    f"avg={score.avg_score:.1f} "
                    f"(rel={score.relevance} acc={score.accuracy} "
                    f"help={score.helpfulness} tim={score.timing} "
                    f"conc={score.conciseness})"
                )
            except Exception as e:
                logger.error(f"  Failed to evaluate intervention {i}: {e}")
                # Create a placeholder score
                score = ContentScore(
                    intervention_index=i,
                    timestamp=intv.get("timestamp", 0),
                    intervention_content=intv.get("content", ""),
                    intervention_mode=intv.get("intervention_mode", ""),
                    rationale=f"Evaluation failed: {e}",
                )
                judge_result.scores.append(score)

        return judge_result

    def _evaluate_single(
        self,
        intervention: dict,
        index: int,
        session_log: list[dict],
        recipe: str,
    ) -> ContentScore:
        """Evaluate a single intervention using Gemini."""
        # Find GT action and nearby context
        intv_ts = intervention.get("timestamp", 0)
        gt_action = self._find_gt_action(intv_ts, session_log)
        nearby = self._get_nearby_context(intv_ts, session_log, window_sec=5.0)

        # Build prompt
        prompt = _build_judge_prompt(intervention, gt_action, recipe, nearby)

        # Call Gemini with JSON format (higher token limit for thinking models)
        response = self.client.complete(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        # Parse response — handle potential truncation or markdown wrapping
        data = self._parse_judge_response(response)

        score = ContentScore(
            intervention_index=index,
            timestamp=intv_ts,
            relevance=int(data.get("relevance", 0)),
            accuracy=int(data.get("accuracy", 0)),
            helpfulness=int(data.get("helpfulness", 0)),
            timing=int(data.get("timing", 0)),
            conciseness=int(data.get("conciseness", 0)),
            safety_justified=data.get("safety_justified"),
            rationale=data.get("rationale", ""),
            gt_action=gt_action,
            intervention_content=intervention.get("content", ""),
            intervention_mode=intervention.get("intervention_mode", ""),
        )

        return score

    @staticmethod
    def _parse_judge_response(response: str) -> dict:
        """Robustly parse the judge's JSON response.

        Handles: clean JSON, markdown-wrapped JSON, truncated JSON,
        and thinking-model output (reasoning text mixed with JSON).
        """
        import re

        text = response.strip()

        # Strip markdown code fences if present
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the most complete JSON object using brace matching
        # This handles thinking models that output text before/after JSON
        best_json = None
        best_fields = 0

        # Find all positions of opening braces
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j+1]
                            try:
                                parsed = json.loads(candidate)
                                # Prefer the JSON with more of our expected fields
                                expected = {"relevance", "accuracy", "helpfulness",
                                            "timing", "conciseness"}
                                n_fields = len(expected & set(parsed.keys()))
                                if n_fields > best_fields:
                                    best_json = parsed
                                    best_fields = n_fields
                            except json.JSONDecodeError:
                                pass
                            break

        if best_json is not None:
            return best_json

        # Last resort: try to find and fix truncated JSON
        # Look for the start of our expected JSON
        match = re.search(r'\{\s*"relevance"', text)
        if match:
            partial = text[match.start():]
            # Try to close truncated JSON
            partial = re.sub(r',\s*"[^"]*"?\s*:?\s*[^,}]*$', '', partial)
            partial = partial.rstrip(', \n') + '}'
            try:
                return json.loads(partial)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse judge response: {text[:200]}")

    def _find_gt_action(self, timestamp: float, session_log: list[dict]) -> str:
        """Find the GT action label at or nearest to a timestamp."""
        best_entry = None
        best_dist = float("inf")
        for entry in session_log:
            dist = abs(entry["timestamp"] - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        if best_entry:
            return best_entry.get("gt_action") or "unknown (between actions)"
        return "unknown"

    def _get_nearby_context(
        self,
        timestamp: float,
        session_log: list[dict],
        window_sec: float = 5.0,
    ) -> list[dict]:
        """Get session log entries near a timestamp."""
        return [
            entry for entry in session_log
            if abs(entry["timestamp"] - timestamp) <= window_sec
        ]
