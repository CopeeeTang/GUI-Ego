"""
Evaluation runner — orchestrates all evaluation dimensions.

Can evaluate:
  1. An existing result JSON file (offline, fast)
  2. A live EGTEA session (runs agent + evaluates)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .metrics import compute_all_metrics
from .judge import ContentJudge, ContentJudgeResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Orchestrates the full evaluation pipeline.

    Usage:
        runner = EvalRunner(judge_model="gemini:gemini-2.5-flash")
        report = runner.evaluate_result_file("output/ar_proactive/result.json")
        report.print_summary()
        report.save("output/ar_proactive/eval_result.json")
    """

    def __init__(
        self,
        judge_model: str = "gemini:gemini-3-flash",
        tolerance_sec: float = 3.0,
        skip_judge: bool = False,
    ):
        """
        Args:
            judge_model: Model spec for content quality judge.
            tolerance_sec: Boundary matching tolerance in seconds.
            skip_judge: If True, skip LLM-as-judge (faster, metrics only).
        """
        self.tolerance_sec = tolerance_sec
        self.skip_judge = skip_judge
        self._judge: Optional[ContentJudge] = None
        self._judge_model = judge_model

    @property
    def judge(self) -> ContentJudge:
        """Lazy-init the judge (avoids API key check until needed)."""
        if self._judge is None:
            self._judge = ContentJudge(model_spec=self._judge_model)
        return self._judge

    def evaluate_result(self, result: dict) -> dict:
        """Run full evaluation on a result dict.

        Returns:
            Evaluation report dict with all dimensions.
        """
        logger.info(
            f"Evaluating session: {result.get('session', 'unknown')} "
            f"({result.get('intervention_count', 0)} interventions)"
        )

        start = time.time()

        # ── Dimension 1-3: Automated metrics (no LLM) ──
        auto_metrics = compute_all_metrics(result, self.tolerance_sec)

        # ── Dimension 4: Content quality (LLM judge) ──
        content_eval = None
        if not self.skip_judge and result.get("interventions"):
            logger.info("Running content quality evaluation (Gemini judge)...")
            try:
                judge_result = self.judge.evaluate_session(result)
                content_eval = judge_result.to_dict()
            except Exception as e:
                logger.error(f"Content judge failed: {e}")
                content_eval = {"error": str(e)}

        eval_time = time.time() - start

        # ── Assemble report ──
        report = {
            "session": result.get("session", "unknown"),
            "recipe": result.get("recipe", "unknown"),
            "participant": result.get("participant", "unknown"),
            "frames_processed": result.get("frames_processed", 0),
            "intervention_count": result.get("intervention_count", 0),
            "eval_time_sec": round(eval_time, 2),
            "trigger_metrics": auto_metrics["trigger"],
            "step_metrics": auto_metrics["step_detection"],
            "system_metrics": auto_metrics["system"],
        }

        if content_eval is not None:
            report["content_quality"] = content_eval

        return report

    def evaluate_result_file(self, path: str) -> dict:
        """Evaluate a saved result JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        with open(path) as f:
            result = json.load(f)

        logger.info(f"Loaded result from {path}")
        return self.evaluate_result(result)

    def evaluate_multiple(self, paths: list[str]) -> dict:
        """Evaluate multiple result files and aggregate.

        Returns:
            Dict with per-session results and aggregated summary.
        """
        sessions = []
        for path in paths:
            try:
                report = self.evaluate_result_file(path)
                sessions.append(report)
            except Exception as e:
                logger.error(f"Failed to evaluate {path}: {e}")
                sessions.append({"session": path, "error": str(e)})

        # Aggregate across sessions
        aggregate = self._aggregate(sessions)

        return {
            "num_sessions": len(sessions),
            "sessions": sessions,
            "aggregate": aggregate,
        }

    def _aggregate(self, sessions: list[dict]) -> dict:
        """Aggregate metrics across multiple sessions."""
        valid = [s for s in sessions if "error" not in s]
        if not valid:
            return {"error": "No valid sessions to aggregate"}

        # Trigger metrics
        trigger_vals = {
            "precision": [],
            "recall": [],
            "f1": [],
            "false_trigger_rate": [],
            "mean_abs_timing_error_sec": [],
        }
        for s in valid:
            tm = s.get("trigger_metrics", {})
            for k in trigger_vals:
                v = tm.get(k)
                if v is not None:
                    trigger_vals[k].append(v)

        # Step metrics
        step_vals = {
            "change_detection_recall": [],
            "change_detection_precision": [],
            "monotonicity_ratio": [],
        }
        for s in valid:
            sm = s.get("step_metrics", {})
            for k in step_vals:
                v = sm.get(k)
                if v is not None:
                    step_vals[k].append(v)

        # Content quality
        content_vals = {
            "avg_relevance": [],
            "avg_accuracy": [],
            "avg_helpfulness": [],
            "avg_timing": [],
            "avg_conciseness": [],
            "overall_avg": [],
        }
        for s in valid:
            cq = s.get("content_quality", {})
            for k in content_vals:
                v = cq.get(k)
                if v is not None:
                    content_vals[k].append(v)

        def _mean(vals):
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "num_sessions": len(valid),
            "trigger": {k: _mean(v) for k, v in trigger_vals.items()},
            "step_detection": {k: _mean(v) for k, v in step_vals.items()},
            "content_quality": {k: _mean(v) for k, v in content_vals.items()},
            "total_interventions": sum(
                s.get("intervention_count", 0) for s in valid
            ),
            "total_frames": sum(
                s.get("frames_processed", 0) for s in valid
            ),
        }
