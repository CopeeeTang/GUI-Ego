"""Memory module — three functional layers aligned with RQ2.

Layers:
    Persistent — task knowledge (never changes)
    Progress   — step tracking and key events (monotonically advances)
    Working    — recent frame observations (sliding window)

Legacy imports (MemoryEntry, LongTermSummary, TieredMemoryStore,
ImportanceScorer, MemoryRetriever) are preserved for backward compatibility.
"""

# New RQ2 types
from .types import FrameRecord, KeyEvent, TaskStep, StepStatus, ProgressSnapshot
from .persistent import PersistentMemory
from .progress import ProgressMemory
from .working import WorkingMemory
from .manager import MemoryManager
from .compressor import ContextCompressor

# Legacy types (still importable for old code)
from .types import MemoryEntry, LongTermSummary
from .store import TieredMemoryStore
from .importance import ImportanceScorer
from .retriever import MemoryRetriever

__all__ = [
    # New
    "FrameRecord", "KeyEvent", "TaskStep", "StepStatus", "ProgressSnapshot",
    "PersistentMemory", "ProgressMemory", "WorkingMemory",
    "MemoryManager", "ContextCompressor",
    # Legacy
    "MemoryEntry", "LongTermSummary",
    "TieredMemoryStore", "ImportanceScorer", "MemoryRetriever",
]
