"""
Core evaluation metrics for Proactive AR Agent.

Four dimensions:
  1. Trigger Timing   — precision, recall, F1 against GT action boundaries
  2. Step Detection    — accuracy of task tracking vs GT labels
  3. System-Level      — intervention density, mode balance, efficiency
  4. Content Quality   — delegated to judge.py (LLM-as-Judge)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Dimension 1: Trigger Timing ─────────────────────────────────────

@dataclass
class TriggerMetrics:
    """Measures when-to-intervene quality against GT action boundaries."""

    total_triggers: int = 0
    total_gt_boundaries: int = 0

    # True positives: triggers within tolerance of a GT boundary
    true_positives: int = 0
    # False positives: triggers NOT near any GT boundary
    false_positives: int = 0
    # False negatives: GT boundaries with no nearby trigger
    false_negatives: int = 0

    # Timing analysis
    timing_errors: list[float] = field(default_factory=list)  # signed, seconds

    @property
    def precision(self) -> float:
        """Of all triggers fired, what fraction were at real boundaries?"""
        if self.total_triggers == 0:
            return 0.0
        return self.true_positives / self.total_triggers

    @property
    def recall(self) -> float:
        """Of all GT boundaries, what fraction got a trigger?"""
        if self.total_gt_boundaries == 0:
            return 0.0
        return (self.total_gt_boundaries - self.false_negatives) / self.total_gt_boundaries

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def false_trigger_rate(self) -> float:
        """Fraction of triggers that are false positives."""
        if self.total_triggers == 0:
            return 0.0
        return self.false_positives / self.total_triggers

    @property
    def mean_timing_error(self) -> float:
        """Mean signed timing error (positive = trigger after boundary)."""
        if not self.timing_errors:
            return 0.0
        return sum(self.timing_errors) / len(self.timing_errors)

    @property
    def mean_abs_timing_error(self) -> float:
        """Mean absolute timing error in seconds."""
        if not self.timing_errors:
            return 0.0
        return sum(abs(e) for e in self.timing_errors) / len(self.timing_errors)

    def to_dict(self) -> dict:
        return {
            "total_triggers": self.total_triggers,
            "total_gt_boundaries": self.total_gt_boundaries,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_trigger_rate": round(self.false_trigger_rate, 4),
            "mean_timing_error_sec": round(self.mean_timing_error, 3),
            "mean_abs_timing_error_sec": round(self.mean_abs_timing_error, 3),
        }


def compute_trigger_metrics(
    session_log: list[dict],
    tolerance_sec: float = 3.0,
) -> TriggerMetrics:
    """Compute trigger precision/recall against GT action boundaries.

    Uses a bipartite matching: each GT boundary matches at most one trigger
    (the closest), and each trigger matches at most one boundary.

    Args:
        session_log: Per-frame log entries from process_egtea_session().
        tolerance_sec: Maximum seconds between trigger and boundary for a match.
    """
    # Extract trigger and boundary timestamps
    trigger_timestamps = []
    boundary_timestamps = []

    for entry in session_log:
        trigger = entry.get("trigger", {})
        if trigger.get("should_trigger"):
            trigger_timestamps.append(entry["timestamp"])
        if entry.get("gt_action_boundary"):
            boundary_timestamps.append(entry["timestamp"])

    metrics = TriggerMetrics(
        total_triggers=len(trigger_timestamps),
        total_gt_boundaries=len(boundary_timestamps),
    )

    if not trigger_timestamps or not boundary_timestamps:
        metrics.false_positives = len(trigger_timestamps)
        metrics.false_negatives = len(boundary_timestamps)
        return metrics

    # Greedy bipartite matching: for each boundary, find closest unmatched trigger
    matched_triggers = set()
    matched_boundaries = set()
    timing_errors = []

    # Build all (boundary, trigger, distance) pairs sorted by distance
    pairs = []
    for bi, bt in enumerate(boundary_timestamps):
        for ti, tt in enumerate(trigger_timestamps):
            dist = abs(tt - bt)
            if dist <= tolerance_sec:
                pairs.append((dist, bi, ti, tt - bt))  # signed error

    pairs.sort(key=lambda x: x[0])

    for dist, bi, ti, signed_error in pairs:
        if bi not in matched_boundaries and ti not in matched_triggers:
            matched_boundaries.add(bi)
            matched_triggers.add(ti)
            timing_errors.append(signed_error)

    metrics.true_positives = len(matched_triggers)
    metrics.false_positives = len(trigger_timestamps) - len(matched_triggers)
    metrics.false_negatives = len(boundary_timestamps) - len(matched_boundaries)
    metrics.timing_errors = timing_errors

    return metrics


# ── Dimension 2: Step Detection ──────────────────────────────────────

@dataclass
class StepMetrics:
    """Measures task tracking accuracy against GT action labels."""

    total_frames: int = 0
    frames_with_gt: int = 0           # frames that have a GT action
    frames_with_detection: int = 0     # frames where step was detected

    # Step change detection
    gt_step_changes: int = 0           # GT action label changed
    detected_step_changes: int = 0     # system detected step_changed
    correct_change_detections: int = 0 # detected change at actual change

    # Step-to-action alignment (how well detected step maps to GT action)
    alignment_scores: list[float] = field(default_factory=list)  # 0-1 per frame

    # Monotonicity: does step index only increase?
    step_jumps: list[tuple[int, int]] = field(default_factory=list)  # (from, to)
    backward_jumps: int = 0

    @property
    def change_detection_recall(self) -> float:
        """Of all GT step changes, how many were detected?"""
        if self.gt_step_changes == 0:
            return 0.0
        return self.correct_change_detections / self.gt_step_changes

    @property
    def change_detection_precision(self) -> float:
        """Of all detected step changes, how many were real?"""
        if self.detected_step_changes == 0:
            return 0.0
        return self.correct_change_detections / self.detected_step_changes

    @property
    def monotonicity_ratio(self) -> float:
        """Fraction of step transitions that are forward (non-decreasing)."""
        total_jumps = len(self.step_jumps)
        if total_jumps == 0:
            return 1.0
        return 1.0 - (self.backward_jumps / total_jumps)

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "frames_with_gt": self.frames_with_gt,
            "frames_with_detection": self.frames_with_detection,
            "gt_step_changes": self.gt_step_changes,
            "detected_step_changes": self.detected_step_changes,
            "correct_change_detections": self.correct_change_detections,
            "change_detection_recall": round(self.change_detection_recall, 4),
            "change_detection_precision": round(self.change_detection_precision, 4),
            "monotonicity_ratio": round(self.monotonicity_ratio, 4),
            "backward_jumps": self.backward_jumps,
            "step_jumps": self.step_jumps[:10],  # first 10 for readability
        }


def compute_step_metrics(session_log: list[dict]) -> StepMetrics:
    """Compute step detection accuracy from session log.

    Analyzes task_update entries in the session log to measure how well
    the TaskTracker follows the actual task progression.
    """
    metrics = StepMetrics()

    prev_gt_action = None
    prev_detected_step = None

    for entry in session_log:
        metrics.total_frames += 1

        gt_action = entry.get("gt_action")
        task_update = entry.get("task_update", {})
        detected_step = task_update.get("detected_step")
        step_changed = task_update.get("step_changed", False)

        # Track GT presence
        if gt_action:
            metrics.frames_with_gt += 1

        # Track detection presence
        if detected_step is not None:
            metrics.frames_with_detection += 1

        # Step change analysis
        gt_changed = (gt_action is not None and prev_gt_action is not None
                      and gt_action != prev_gt_action)

        if gt_changed:
            metrics.gt_step_changes += 1

        if step_changed:
            metrics.detected_step_changes += 1

        # Match: both GT and system agree a change happened at this frame
        if gt_changed and step_changed:
            metrics.correct_change_detections += 1

        # Step jump tracking (monotonicity)
        if detected_step is not None and prev_detected_step is not None:
            if detected_step != prev_detected_step:
                metrics.step_jumps.append((prev_detected_step, detected_step))
                if detected_step < prev_detected_step:
                    metrics.backward_jumps += 1

        prev_gt_action = gt_action
        if detected_step is not None:
            prev_detected_step = detected_step

    return metrics


# ── Dimension 3: System-Level ────────────────────────────────────────

@dataclass
class SystemMetrics:
    """System-level health indicators."""

    total_frames: int = 0
    total_interventions: int = 0
    processing_time_sec: float = 0.0
    session_duration_sec: float = 0.0

    # Mode distribution
    mode_distribution: dict[str, int] = field(default_factory=dict)
    type_distribution: dict[str, int] = field(default_factory=dict)
    priority_distribution: dict[str, int] = field(default_factory=dict)

    # Confidence stats
    confidence_values: list[float] = field(default_factory=list)

    # Memory stats
    memory_stats: dict = field(default_factory=dict)

    @property
    def interventions_per_minute(self) -> float:
        if self.session_duration_sec == 0:
            return 0.0
        return self.total_interventions / (self.session_duration_sec / 60.0)

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_values:
            return 0.0
        return sum(self.confidence_values) / len(self.confidence_values)

    @property
    def processing_ratio(self) -> float:
        """Ratio of processing time to session duration (< 1 = faster than real-time)."""
        if self.session_duration_sec == 0:
            return 0.0
        return self.processing_time_sec / self.session_duration_sec

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "total_interventions": self.total_interventions,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "session_duration_sec": round(self.session_duration_sec, 2),
            "interventions_per_minute": round(self.interventions_per_minute, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "processing_ratio": round(self.processing_ratio, 3),
            "mode_distribution": self.mode_distribution,
            "type_distribution": self.type_distribution,
            "priority_distribution": self.priority_distribution,
            "memory_stats": self.memory_stats,
        }


def compute_system_metrics(result: dict) -> SystemMetrics:
    """Compute system-level metrics from a full result dict."""
    metrics = SystemMetrics(
        total_frames=result.get("frames_processed", 0),
        total_interventions=result.get("intervention_count", 0),
        processing_time_sec=result.get("processing_time_sec", 0.0),
        memory_stats=result.get("memory_stats", {}),
    )

    # Session duration from log timestamps
    session_log = result.get("session_log", [])
    if session_log:
        timestamps = [e["timestamp"] for e in session_log]
        metrics.session_duration_sec = max(timestamps) - min(timestamps)

    # Intervention distributions
    for intv in result.get("interventions", []):
        mode = intv.get("intervention_mode", "unknown")
        itype = intv.get("intervention_type", "unknown")
        priority = intv.get("priority", "unknown")

        metrics.mode_distribution[mode] = metrics.mode_distribution.get(mode, 0) + 1
        metrics.type_distribution[itype] = metrics.type_distribution.get(itype, 0) + 1
        metrics.priority_distribution[priority] = metrics.priority_distribution.get(priority, 0) + 1
        metrics.confidence_values.append(intv.get("confidence", 0.0))

    return metrics


# ── Combined ─────────────────────────────────────────────────────────

def compute_all_metrics(
    result: dict,
    tolerance_sec: float = 3.0,
) -> dict:
    """Compute all automated metrics (no LLM calls).

    Returns a dictionary with trigger, step, and system metrics.
    """
    session_log = result.get("session_log", [])

    trigger = compute_trigger_metrics(session_log, tolerance_sec)
    step = compute_step_metrics(session_log)
    system = compute_system_metrics(result)

    return {
        "trigger": trigger.to_dict(),
        "step_detection": step.to_dict(),
        "system": system.to_dict(),
    }
