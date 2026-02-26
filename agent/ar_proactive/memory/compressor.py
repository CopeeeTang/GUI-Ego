"""Context Compressor — SUMMARIZE_AND_DROP strategy.

When memory grows too large, compress older entries into a text summary.
Inspired by ProAssist's progress_summary mechanism.
"""

import logging
from typing import Optional

from agent.src.llm.base import LLMClientBase
from .types import ProgressSnapshot, KeyEvent

logger = logging.getLogger(__name__)

COMPRESS_SYSTEM = """\
You are a progress summarizer for a task assistant. Compress the following \
events into a concise summary that preserves: what was done, key objects used, \
and any errors or hazards encountered. Be brief (2-3 sentences).\
"""


class ContextCompressor:
    """Compress older memory entries into text summaries."""

    def __init__(self, llm_client: LLMClientBase, max_tokens: int = 500):
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    def compress_events(self, events: list[KeyEvent]) -> Optional[ProgressSnapshot]:
        """Compress a list of key events into a ProgressSnapshot."""
        if len(events) < 3:
            return None

        events_text = "\n".join(
            f"- [{e.timestamp:.1f}s] ({e.event_type}) {e.description}"
            for e in events
        )

        user_prompt = (
            f"Compress these {len(events)} events:\n\n{events_text}\n\n"
            'Return JSON: {"summary": "...", "key_events": ["..."], '
            '"steps_completed": [0, 1, ...]}'
        )

        try:
            result = self.llm_client.complete_json(
                system_prompt=COMPRESS_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )

            return ProgressSnapshot(
                time_range_start=events[0].timestamp,
                time_range_end=events[-1].timestamp,
                summary_text=result.get("summary", ""),
                steps_completed=result.get("steps_completed", []),
                key_events_summary=result.get("key_events", []),
            )
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return None
