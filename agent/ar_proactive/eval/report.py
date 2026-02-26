"""
Evaluation report formatting and output.

Supports:
  - Terminal pretty-print
  - JSON export
  - Markdown export (for documentation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EvalReport:
    """Wraps evaluation results with formatting and export methods."""

    def __init__(self, data: dict):
        self.data = data

    def print_summary(self) -> None:
        """Print a concise summary to terminal."""
        d = self.data

        print("\n" + "=" * 60)
        print(f"  EVALUATION REPORT: {d.get('session', 'unknown')}")
        print(f"  Recipe: {d.get('recipe', '?')} | "
              f"Frames: {d.get('frames_processed', 0)} | "
              f"Interventions: {d.get('intervention_count', 0)}")
        print("=" * 60)

        # ── Trigger Metrics ──
        tm = d.get("trigger_metrics", {})
        print("\n[1] TRIGGER TIMING (When to Intervene)")
        print(f"    Precision:         {tm.get('precision', 0):.1%}")
        print(f"    Recall:            {tm.get('recall', 0):.1%}")
        print(f"    F1:                {tm.get('f1', 0):.1%}")
        print(f"    False Trigger Rate:{tm.get('false_trigger_rate', 0):.1%}")
        print(f"    Avg Timing Error:  {tm.get('mean_abs_timing_error_sec', 0):.2f}s")
        print(f"    (TP={tm.get('true_positives', 0)} "
              f"FP={tm.get('false_positives', 0)} "
              f"FN={tm.get('false_negatives', 0)})")

        # ── Step Detection ──
        sm = d.get("step_metrics", {})
        print("\n[2] STEP DETECTION (Task Understanding)")
        print(f"    Change Detection Recall:    {sm.get('change_detection_recall', 0):.1%}")
        print(f"    Change Detection Precision: {sm.get('change_detection_precision', 0):.1%}")
        print(f"    Monotonicity:               {sm.get('monotonicity_ratio', 0):.1%}")
        print(f"    Backward Jumps:             {sm.get('backward_jumps', 0)}")
        jumps = sm.get("step_jumps", [])
        if jumps:
            print(f"    Step Jumps (first 5):       {jumps[:5]}")

        # ── System Metrics ──
        sys_m = d.get("system_metrics", {})
        print("\n[3] SYSTEM HEALTH")
        print(f"    Interventions/min:  {sys_m.get('interventions_per_minute', 0):.2f}")
        print(f"    Avg Confidence:     {sys_m.get('avg_confidence', 0):.2f}")
        print(f"    Processing Ratio:   {sys_m.get('processing_ratio', 0):.1f}x real-time")
        print(f"    Mode Distribution:  {sys_m.get('mode_distribution', {})}")
        print(f"    Type Distribution:  {sys_m.get('type_distribution', {})}")

        # ── Content Quality ──
        cq = d.get("content_quality")
        if cq and "error" not in cq:
            print("\n[4] CONTENT QUALITY (Gemini Judge)")
            print(f"    Relevance:    {cq.get('avg_relevance', 0):.2f}/5")
            print(f"    Accuracy:     {cq.get('avg_accuracy', 0):.2f}/5")
            print(f"    Helpfulness:  {cq.get('avg_helpfulness', 0):.2f}/5")
            print(f"    Timing:       {cq.get('avg_timing', 0):.2f}/5")
            print(f"    Conciseness:  {cq.get('avg_conciseness', 0):.2f}/5")
            print(f"    Overall:      {cq.get('overall_avg', 0):.2f}/5")
            sp = cq.get("safety_precision")
            if sp is not None:
                print(f"    Safety Precision: {sp:.1%}")

            # Per-intervention details
            scores = cq.get("scores", [])
            if scores:
                print(f"\n    Per-intervention:")
                for s in scores:
                    print(f"      [{s['timestamp']:.1f}s] {s['intervention_mode']} "
                          f"avg={s['avg_score']:.1f} | "
                          f"rel={s['relevance']} acc={s['accuracy']} "
                          f"help={s['helpfulness']} tim={s['timing']} "
                          f"conc={s['conciseness']}")
                    if s.get("rationale"):
                        print(f"        → {s['rationale'][:120]}")
        elif cq and "error" in cq:
            print(f"\n[4] CONTENT QUALITY: Error - {cq['error']}")
        else:
            print("\n[4] CONTENT QUALITY: Skipped (--skip-judge)")

        print("\n" + "=" * 60)
        print(f"  Eval time: {d.get('eval_time_sec', 0):.1f}s")
        print("=" * 60 + "\n")

    def print_aggregate(self) -> None:
        """Print aggregate summary for multi-session evaluation."""
        agg = self.data.get("aggregate", {})
        n = self.data.get("num_sessions", 0)

        print("\n" + "=" * 60)
        print(f"  AGGREGATE REPORT ({n} sessions)")
        print("=" * 60)

        t = agg.get("trigger", {})
        print(f"\n  Trigger:  P={t.get('precision', 0):.1%}  "
              f"R={t.get('recall', 0):.1%}  F1={t.get('f1', 0):.1%}")

        s = agg.get("step_detection", {})
        print(f"  Step Det: Recall={s.get('change_detection_recall', 0):.1%}  "
              f"Mono={s.get('monotonicity_ratio', 0):.1%}")

        c = agg.get("content_quality", {})
        if c.get("overall_avg"):
            print(f"  Content:  Overall={c['overall_avg']:.2f}/5  "
                  f"Rel={c.get('avg_relevance', 0):.2f}  "
                  f"Acc={c.get('avg_accuracy', 0):.2f}  "
                  f"Help={c.get('avg_helpfulness', 0):.2f}")

        print(f"\n  Total: {agg.get('total_interventions', 0)} interventions, "
              f"{agg.get('total_frames', 0)} frames")
        print("=" * 60 + "\n")

    def save(self, path: str) -> None:
        """Save full report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {path}")

    def to_markdown(self) -> str:
        """Export report as markdown string."""
        d = self.data
        lines = [
            f"# Evaluation Report: {d.get('session', 'unknown')}",
            f"",
            f"**Recipe**: {d.get('recipe', '?')} | "
            f"**Frames**: {d.get('frames_processed', 0)} | "
            f"**Interventions**: {d.get('intervention_count', 0)}",
            f"",
        ]

        # Trigger
        tm = d.get("trigger_metrics", {})
        lines.extend([
            "## 1. Trigger Timing",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Precision | {tm.get('precision', 0):.1%} |",
            f"| Recall | {tm.get('recall', 0):.1%} |",
            f"| F1 | {tm.get('f1', 0):.1%} |",
            f"| False Trigger Rate | {tm.get('false_trigger_rate', 0):.1%} |",
            f"| Avg Timing Error | {tm.get('mean_abs_timing_error_sec', 0):.2f}s |",
            "",
        ])

        # Step
        sm = d.get("step_metrics", {})
        lines.extend([
            "## 2. Step Detection",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Change Detection Recall | {sm.get('change_detection_recall', 0):.1%} |",
            f"| Change Detection Precision | {sm.get('change_detection_precision', 0):.1%} |",
            f"| Monotonicity | {sm.get('monotonicity_ratio', 0):.1%} |",
            "",
        ])

        # Content
        cq = d.get("content_quality", {})
        if cq and "error" not in cq:
            lines.extend([
                "## 3. Content Quality (Gemini Judge)",
                "",
                "| Dimension | Score |",
                "|---|---|",
                f"| Relevance | {cq.get('avg_relevance', 0):.2f}/5 |",
                f"| Accuracy | {cq.get('avg_accuracy', 0):.2f}/5 |",
                f"| Helpfulness | {cq.get('avg_helpfulness', 0):.2f}/5 |",
                f"| Timing | {cq.get('avg_timing', 0):.2f}/5 |",
                f"| Conciseness | {cq.get('avg_conciseness', 0):.2f}/5 |",
                f"| **Overall** | **{cq.get('overall_avg', 0):.2f}/5** |",
                "",
            ])

        return "\n".join(lines)
