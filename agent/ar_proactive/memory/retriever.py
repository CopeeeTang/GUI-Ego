"""Memory retriever — lightweight keyword-overlap based retrieval."""

import logging
from typing import Optional

from .types import MemoryEntry

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieve relevant memories using keyword overlap.

    No embedding model needed — uses simple set intersection
    on tags, objects, and activities for fast retrieval.
    """

    def __init__(self, store):
        """Initialize with a reference to the TieredMemoryStore.

        Args:
            store: TieredMemoryStore instance.
        """
        self.store = store

    def retrieve_relevant(
        self,
        query_description: str,
        query_objects: list[str],
        top_k: int = 3,
    ) -> list[MemoryEntry]:
        """Find mid-term entries with highest keyword overlap to the query.

        Args:
            query_description: Current frame description text.
            query_objects: Objects detected in the current frame.
            top_k: Number of entries to return.

        Returns:
            List of the most relevant MemoryEntry objects.
        """
        if not self.store.mid_term:
            return []

        # Build query keyword set from description words and objects
        query_keywords = set(w.lower() for w in query_description.split())
        query_keywords.update(o.lower() for o in query_objects)

        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self.store.mid_term:
            entry_keywords = set(t.lower() for t in entry.tags)
            entry_keywords.update(o.lower() for o in entry.detected_objects)
            entry_keywords.update(a.lower() for a in entry.detected_activities)
            # Add words from scene description
            entry_keywords.update(w.lower() for w in entry.scene_description.split())

            if not entry_keywords:
                continue

            overlap = len(query_keywords & entry_keywords)
            # Normalize by union size (Jaccard-like)
            union = len(query_keywords | entry_keywords)
            similarity = overlap / union if union > 0 else 0.0

            scored.append((similarity, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def retrieve_by_tags(self, tags: list[str]) -> list[MemoryEntry]:
        """Filter mid-term entries by tag intersection.

        Args:
            tags: Tags to search for.

        Returns:
            Entries that share at least one tag.
        """
        tag_set = set(t.lower() for t in tags)
        return [
            entry
            for entry in self.store.mid_term
            if set(t.lower() for t in entry.tags) & tag_set
        ]

    def retrieve_temporal_window(
        self,
        center_time: float,
        window_sec: float = 2.0,
    ) -> list[MemoryEntry]:
        """Get entries within a time window around center_time.

        Searches both short-term and mid-term.

        Args:
            center_time: Center timestamp in seconds.
            window_sec: Half-width of the window.

        Returns:
            Entries within [center_time - window_sec, center_time + window_sec].
        """
        t_min = center_time - window_sec
        t_max = center_time + window_sec

        results: list[MemoryEntry] = []

        for entry in self.store.short_term:
            if t_min <= entry.timestamp <= t_max:
                results.append(entry)

        for entry in self.store.mid_term:
            if t_min <= entry.timestamp <= t_max:
                # Avoid duplicates
                if entry not in results:
                    results.append(entry)

        results.sort(key=lambda e: e.timestamp)
        return results
