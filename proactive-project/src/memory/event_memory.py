"""
Layer 2: Semantic Event Memory

Structured event store with embedding-based retrieval.
Records significant events (action completions, object interactions,
anomalies) with timestamps for on-demand retrieval.

Not in VLM context by default — retrieved and injected when needed.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A recorded semantic event."""
    event_id: int
    timestamp: float
    description: str
    event_type: str             # action, object_interaction, anomaly, milestone
    entities: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # set after encoding


class EventMemory:
    """
    Layer 2 memory: structured event store with semantic retrieval.

    Uses sentence-transformer embeddings for similarity search.
    Falls back to keyword matching if embeddings unavailable.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_events: int = 200,
        similarity_threshold: float = 0.3,
    ):
        self.max_events = max_events
        self.similarity_threshold = similarity_threshold
        self.events: list[Event] = []
        self._next_id = 0

        # Lazy-load embedding model
        self._embed_model_name = embedding_model
        self._embed_model = None
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._embeddings_dirty = True

    def _get_embedder(self):
        """Lazy-load sentence-transformer model."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(self._embed_model_name)
                logger.info(f"Loaded embedding model: {self._embed_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using keyword fallback")
        return self._embed_model

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding vector."""
        model = self._get_embedder()
        if model is None:
            return None
        return model.encode(text, normalize_embeddings=True)

    def add_event(
        self,
        timestamp: float,
        description: str,
        event_type: str = "action",
        entities: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Event:
        """Record a new event."""
        event = Event(
            event_id=self._next_id,
            timestamp=timestamp,
            description=description,
            event_type=event_type,
            entities=entities or [],
            metadata=metadata or {},
            embedding=self._encode(description),
        )
        self._next_id += 1
        self.events.append(event)
        self._embeddings_dirty = True

        # Evict oldest if over capacity
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
            self._embeddings_dirty = True

        return event

    def add_from_action(self, action, timestamp: Optional[float] = None):
        """Add event from an ActionClip object."""
        ts = timestamp if timestamp is not None else action.start_sec
        # Build rich natural language description for better embedding retrieval
        nouns_str = " and ".join(action.nouns) if action.nouns else "items"
        description = (
            f"The cook performed '{action.action_label}': "
            f"{action.verb} the {nouns_str} "
            f"at {ts:.1f} seconds into the session"
        )
        self.add_event(
            timestamp=ts,
            description=description,
            event_type="action",
            entities=list(action.nouns),
            metadata={
                "verb": action.verb,
                "action_label": action.action_label,
                "nouns": list(action.nouns),
            },
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Event]:
        """Retrieve events most relevant to a query."""
        if not self.events:
            return []

        query_emb = self._encode(query)
        if query_emb is not None:
            return self._retrieve_by_embedding(query_emb, top_k)
        else:
            return self._retrieve_by_keyword(query, top_k)

    def _retrieve_by_embedding(self, query_emb: np.ndarray, top_k: int) -> list[Event]:
        """Semantic similarity retrieval — always returns top_k results."""
        # Build matrix if dirty
        if self._embeddings_dirty or self._embeddings_matrix is None:
            valid_events = [e for e in self.events if e.embedding is not None]
            if not valid_events:
                return self._retrieve_by_keyword("", top_k)
            self._embeddings_matrix = np.stack([e.embedding for e in valid_events])
            self._embeddings_dirty = False

        valid_events = [e for e in self.events if e.embedding is not None]

        # Cosine similarity (embeddings are normalized)
        scores = self._embeddings_matrix @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Always return top_k results regardless of threshold
        # (threshold filtering was too aggressive)
        return [valid_events[idx] for idx in top_indices if idx < len(valid_events)]

    def _retrieve_by_keyword(self, query: str, top_k: int) -> list[Event]:
        """Fallback: keyword-based retrieval."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for event in self.events:
            desc_words = set(event.description.lower().split())
            entity_words = set(e.lower() for e in event.entities)
            overlap = len(query_words & (desc_words | entity_words))
            if overlap > 0:
                scored.append((overlap, event))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [event for _, event in scored[:top_k]]

    def retrieve_by_time(self, start_sec: float, end_sec: float) -> list[Event]:
        """Retrieve events within a time window."""
        return [e for e in self.events
                if start_sec <= e.timestamp <= end_sec]

    def retrieve_by_entity(self, entity: str) -> list[Event]:
        """Retrieve all events involving a specific entity."""
        entity_lower = entity.lower()
        return [e for e in self.events
                if any(entity_lower in ent.lower() for ent in e.entities)]

    def get_recent(self, n: int = 5) -> list[Event]:
        """Get N most recent events."""
        return self.events[-n:]

    def to_context_string(self, events: list[Event]) -> str:
        """Format retrieved events as context string for VLM."""
        if not events:
            return "[Event Memory] No relevant events found."

        lines = [f"[Event Memory] {len(events)} relevant events:"]
        for e in events:
            lines.append(
                f"  [{e.timestamp:.1f}s] {e.description} "
                f"(entities: {', '.join(e.entities)})"
            )
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self.events)

    def reset(self):
        self.events = []
        self._next_id = 0
        self._embeddings_matrix = None
        self._embeddings_dirty = True
