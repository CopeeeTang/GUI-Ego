"""
Ground Truth Generator for Proactive Intervention

Generates proactive intervention ground truth from EGTEA Gaze+ annotations.
Defines when a streaming system SHOULD intervene and what it should say.

Trigger Types:
  1. STEP_TRANSITION: Action boundary → notify about next step
  2. SAFETY_WARNING: Before dangerous action → warn user
  3. IDLE_REMINDER: Long gap between actions → remind next step
  4. PROGRESS_UPDATE: After completing a significant sub-task
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from data.egtea_loader import CookingSession, EGTEALoader


class TriggerType(Enum):
    STEP_TRANSITION = "step_transition"
    SAFETY_WARNING = "safety_warning"
    IDLE_REMINDER = "idle_reminder"
    PROGRESS_UPDATE = "progress_update"


@dataclass
class GroundTruthTrigger:
    """A ground truth proactive intervention point."""
    timestamp: float            # when the system should trigger (seconds)
    trigger_type: TriggerType
    expected_content: str       # what the system should say
    context: dict = field(default_factory=dict)  # extra metadata

    # For evaluation matching
    valid_window: tuple[float, float] = (0.0, 0.0)  # [start, end] for soft matching


SAFETY_VERBS = {"Cut", "Pour", "Operate", "Crack"}

# Verbs that indicate meaningful cooking progress (not just moving items)
PROGRESS_VERBS = {"Cut", "Pour", "Mix", "Spread", "Crack", "Operate", "Wash", "Compress"}


class GroundTruthGenerator:
    """Generate proactive intervention ground truth from EGTEA annotations."""

    def __init__(
        self,
        transition_gap_sec: float = 2.0,
        safety_lookahead_sec: float = 3.0,
        idle_threshold_sec: float = 8.0,
        progress_interval: int = 5,       # every N cooking actions → progress update
        soft_window_sec: float = 3.0,
    ):
        self.transition_gap_sec = transition_gap_sec
        self.safety_lookahead_sec = safety_lookahead_sec
        self.idle_threshold_sec = idle_threshold_sec
        self.progress_interval = progress_interval
        self.soft_window_sec = soft_window_sec

    def generate(self, session: CookingSession) -> list[GroundTruthTrigger]:
        """Generate all ground truth triggers for a cooking session."""
        triggers = []

        triggers.extend(self._step_transitions(session))
        triggers.extend(self._safety_warnings(session))
        triggers.extend(self._idle_reminders(session))
        triggers.extend(self._progress_updates(session))

        # Sort by timestamp and deduplicate (merge triggers within 2s)
        triggers.sort(key=lambda t: t.timestamp)
        triggers = self._deduplicate(triggers, min_gap=2.0)

        return triggers

    def _step_transitions(self, session: CookingSession) -> list[GroundTruthTrigger]:
        """Trigger at action boundaries when the next action is different."""
        triggers = []
        actions = session.actions
        for i in range(len(actions) - 1):
            curr = actions[i]
            nxt = actions[i + 1]

            # Only trigger if actions are different
            if curr.verb == nxt.verb and curr.nouns == nxt.nouns:
                continue

            # Trigger point: end of current action
            ts = curr.end_sec

            triggers.append(GroundTruthTrigger(
                timestamp=ts,
                trigger_type=TriggerType.STEP_TRANSITION,
                expected_content=(
                    f"Step completed: {curr.action_label}. "
                    f"Next step: {nxt.action_label}."
                ),
                context={
                    "completed_action": curr.action_label,
                    "next_action": nxt.action_label,
                    "gap_sec": nxt.start_sec - curr.end_sec,
                },
                valid_window=(
                    ts - self.soft_window_sec,
                    ts + self.soft_window_sec,
                ),
            ))

        return triggers

    def _safety_warnings(self, session: CookingSession) -> list[GroundTruthTrigger]:
        """Trigger before dangerous actions (Cut, Pour, Operate, Crack)."""
        triggers = []
        for action in session.actions:
            if action.verb not in SAFETY_VERBS:
                continue

            # Trigger BEFORE the action starts
            ts = max(0, action.start_sec - self.safety_lookahead_sec)

            noun_str = ", ".join(action.nouns) if action.nouns else "item"
            triggers.append(GroundTruthTrigger(
                timestamp=ts,
                trigger_type=TriggerType.SAFETY_WARNING,
                expected_content=(
                    f"Caution: About to {action.verb.lower()} {noun_str}. "
                    f"Please be careful."
                ),
                context={
                    "action": action.action_label,
                    "verb": action.verb,
                    "nouns": action.nouns,
                },
                valid_window=(
                    ts - self.soft_window_sec,
                    action.start_sec,
                ),
            ))

        return triggers

    def _idle_reminders(self, session: CookingSession) -> list[GroundTruthTrigger]:
        """Trigger during long idle gaps between actions."""
        triggers = []
        gaps = session.get_gaps(min_gap_sec=self.idle_threshold_sec)

        for gap_start, gap_end in gaps:
            # Trigger in the middle of the gap
            ts = (gap_start + gap_end) / 2

            next_action = session.get_next_action(gap_start)
            if not next_action:
                continue

            triggers.append(GroundTruthTrigger(
                timestamp=ts,
                trigger_type=TriggerType.IDLE_REMINDER,
                expected_content=(
                    f"Idle for {gap_end - gap_start:.0f}s. "
                    f"Next step: {next_action.action_label}."
                ),
                context={
                    "gap_duration": gap_end - gap_start,
                    "next_action": next_action.action_label,
                },
                valid_window=(gap_start + 1.0, gap_end - 1.0),
            ))

        return triggers

    def _progress_updates(self, session: CookingSession) -> list[GroundTruthTrigger]:
        """Trigger periodic progress updates after N cooking actions."""
        triggers = []
        cooking_count = 0

        for action in session.actions:
            if action.verb in PROGRESS_VERBS:
                cooking_count += 1

            if cooking_count > 0 and cooking_count % self.progress_interval == 0:
                ts = action.end_sec
                progress_pct = min(100, int(100 * cooking_count / max(1, len(session.actions))))

                triggers.append(GroundTruthTrigger(
                    timestamp=ts,
                    trigger_type=TriggerType.PROGRESS_UPDATE,
                    expected_content=(
                        f"Progress: ~{progress_pct}% complete. "
                        f"{cooking_count} cooking steps done."
                    ),
                    context={
                        "cooking_steps_done": cooking_count,
                        "total_actions": len(session.actions),
                        "progress_pct": progress_pct,
                    },
                    valid_window=(ts - self.soft_window_sec, ts + self.soft_window_sec),
                ))

        return triggers

    def _deduplicate(self, triggers: list[GroundTruthTrigger],
                     min_gap: float) -> list[GroundTruthTrigger]:
        """Remove triggers too close together, keeping higher-priority ones."""
        priority = {
            TriggerType.SAFETY_WARNING: 0,
            TriggerType.STEP_TRANSITION: 1,
            TriggerType.IDLE_REMINDER: 2,
            TriggerType.PROGRESS_UPDATE: 3,
        }

        if not triggers:
            return []

        result = [triggers[0]]
        for t in triggers[1:]:
            if t.timestamp - result[-1].timestamp < min_gap:
                # Keep the higher-priority one
                if priority[t.trigger_type] < priority[result[-1].trigger_type]:
                    result[-1] = t
            else:
                result.append(t)

        return result

    def generate_for_dataset(
        self,
        loader: EGTEALoader,
        recipe: Optional[str] = None,
        split: Optional[str] = None,
        max_sessions: Optional[int] = None,
    ) -> dict[str, list[GroundTruthTrigger]]:
        """Generate ground truth for all sessions in the dataset."""
        sessions = loader.iter_sessions(recipe=recipe, split=split)
        if max_sessions:
            sessions = sessions[:max_sessions]

        all_gt = {}
        for session in sessions:
            triggers = self.generate(session)
            if triggers:
                all_gt[session.session_id] = triggers

        return all_gt

    def summary(self, gt: dict[str, list[GroundTruthTrigger]]) -> str:
        """Print summary of generated ground truth."""
        from collections import Counter
        total = sum(len(v) for v in gt.values())
        type_counts = Counter()
        for triggers in gt.values():
            for t in triggers:
                type_counts[t.trigger_type.value] += 1

        lines = [
            f"Ground Truth Summary: {len(gt)} sessions, {total} triggers",
            f"  Avg triggers/session: {total / max(1, len(gt)):.1f}",
        ]
        for ttype, count in sorted(type_counts.items()):
            lines.append(f"  {ttype}: {count}")
        return "\n".join(lines)
