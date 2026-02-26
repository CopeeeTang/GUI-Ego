"""Tiered memory store: short-term → mid-term → long-term."""

import logging
from collections import deque
from typing import Optional

from .types import MemoryEntry, LongTermSummary

logger = logging.getLogger(__name__)


class TieredMemoryStore:
    """Three-tier memory with automatic promotion and eviction.

    Tiers:
        short-term — last N frames (deque), stores images, auto-evicts oldest
        mid-term   — up to M important entries, stores images, evicts lowest score
        long-term  — unlimited compressed summaries, text only, never evicted
    """

    def __init__(
        self,
        short_term_capacity: int = 6,
        mid_term_capacity: int = 20,
        importance_threshold: float = 0.4,
        long_term_summary_every: int = 10,
    ):
        self.short_term_capacity = short_term_capacity
        self.mid_term_capacity = mid_term_capacity
        self.importance_threshold = importance_threshold
        self.long_term_summary_every = long_term_summary_every

        self.short_term: deque[MemoryEntry] = deque(maxlen=short_term_capacity)
        self.mid_term: list[MemoryEntry] = []
        self.long_term: list[LongTermSummary] = []

        self._promotions_since_summary = 0

    def add(self, entry: MemoryEntry) -> bool:
        """Add a new memory entry to short-term storage.

        Automatically promotes to mid-term if importance >= threshold.

        Args:
            entry: The memory entry to store.

        Returns:
            True if the entry was promoted to mid-term.
        """
        self.short_term.append(entry)

        promoted = False
        if entry.importance_score >= self.importance_threshold:
            self._promote_to_mid_term(entry)
            promoted = True

        return promoted

    def _promote_to_mid_term(self, entry: MemoryEntry):
        """Promote an entry from short-term to mid-term."""
        if len(self.mid_term) >= self.mid_term_capacity:
            # Evict the entry with the lowest importance score
            min_idx = min(
                range(len(self.mid_term)),
                key=lambda i: self.mid_term[i].importance_score,
            )
            evicted = self.mid_term.pop(min_idx)
            logger.debug(
                f"Evicted mid-term entry at {evicted.timestamp:.2f}s "
                f"(score {evicted.importance_score:.2f})"
            )

        self.mid_term.append(entry)
        self._promotions_since_summary += 1

        logger.debug(
            f"Promoted to mid-term: {entry.timestamp:.2f}s "
            f"(score {entry.importance_score:.2f}, "
            f"mid-term size: {len(self.mid_term)})"
        )

    def needs_long_term_summary(self) -> bool:
        """Check if it's time to create a long-term summary."""
        return self._promotions_since_summary >= self.long_term_summary_every

    def add_long_term_summary(self, summary: LongTermSummary):
        """Add a compressed long-term summary and reset the promotion counter."""
        self.long_term.append(summary)
        self._promotions_since_summary = 0
        logger.info(
            f"Added long-term summary: "
            f"{summary.time_range_start:.2f}s – {summary.time_range_end:.2f}s"
        )

    def get_mid_term_descriptions(self) -> list[dict]:
        """Get all mid-term entries as dicts for LLM summarization."""
        return [
            {
                "timestamp": e.timestamp,
                "scene_description": e.scene_description,
                "detected_objects": e.detected_objects,
                "detected_activities": e.detected_activities,
                "importance_score": e.importance_score,
            }
            for e in self.mid_term
        ]

    def get_recent_objects(self, n: int = 3) -> set[str]:
        """Get the set of objects seen in the last N short-term entries."""
        objects: set[str] = set()
        recent = list(self.short_term)[-n:]
        for entry in recent:
            objects.update(entry.detected_objects)
        return objects

    def get_recent_activities(self, n: int = 3) -> set[str]:
        """Get the set of activities seen in the last N short-term entries."""
        activities: set[str] = set()
        recent = list(self.short_term)[-n:]
        for entry in recent:
            activities.update(entry.detected_activities)
        return activities

    def get_full_context_text(self, max_mid_term: int = 5, max_short_term: int = 3) -> str:
        """Assemble all memory tiers into a single text block for the LLM.

        Args:
            max_mid_term: Max mid-term entries to include (by importance).
            max_short_term: Max recent short-term entries to include.

        Returns:
            Formatted context string.
        """
        sections = []

        # Long-term summaries
        if self.long_term:
            sections.append("## Session Summary (long-term)")
            for summary in self.long_term:
                sections.append(summary.to_context_text())
            sections.append("")

        # Mid-term: top entries by importance
        if self.mid_term:
            sorted_mid = sorted(
                self.mid_term, key=lambda e: e.importance_score, reverse=True
            )
            top_mid = sorted_mid[:max_mid_term]
            sections.append(f"## Key Events (mid-term, top {len(top_mid)} by importance)")
            for entry in top_mid:
                sections.append(f"- {entry.to_context_line()}")
            sections.append("")

        # Short-term: most recent entries
        recent = list(self.short_term)[-max_short_term:]
        if recent:
            sections.append(f"## Recent Context (short-term, last {len(recent)})")
            for entry in recent:
                sections.append(f"- {entry.to_context_line()}")
            sections.append("")

        return "\n".join(sections) if sections else "(No memory context yet)"

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "short_term_count": len(self.short_term),
            "short_term_capacity": self.short_term_capacity,
            "mid_term_count": len(self.mid_term),
            "mid_term_capacity": self.mid_term_capacity,
            "long_term_count": len(self.long_term),
            "promotions_since_summary": self._promotions_since_summary,
        }
