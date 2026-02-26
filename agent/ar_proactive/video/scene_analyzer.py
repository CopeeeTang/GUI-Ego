"""Scene analyzer — Claude vision calls for structured frame analysis."""

import logging
from typing import Any

from agent.src.llm.base import LLMClientBase

logger = logging.getLogger(__name__)

SCENE_ANALYSIS_SYSTEM_PROMPT = """\
You are a scene analysis module for AR smart glasses. You analyze ego-centric camera frames \
and return structured JSON describing what the wearer sees.

Be concise. Focus on actionable observations: objects, activities, hazards, and readable text.\
"""

SCENE_ANALYSIS_USER_PROMPT = """\
Analyze this ego-centric camera frame from smart glasses. Return a JSON object with:

{
  "environment": "brief description of the environment/location type",
  "objects": ["list", "of", "visible", "objects"],
  "activities": ["list", "of", "observed", "activities"],
  "text_visible": ["any", "readable", "text", "or", "signs"],
  "people_present": false,
  "potential_hazards": ["any", "safety", "hazards"],
  "scene_tags": ["searchable", "keyword", "tags"]
}

Respond with valid JSON only, no markdown.\
"""


class SceneAnalyzer:
    """Analyze video frames using Claude's vision capabilities.

    Sends each frame to the LLM and returns structured scene information.
    """

    def __init__(self, llm_client: LLMClientBase, max_tokens: int = 1500):
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    def analyze(self, frame_base64: str) -> dict[str, Any]:
        """Analyze a single frame and return structured JSON.

        Args:
            frame_base64: Base64-encoded JPEG frame.

        Returns:
            Dict with keys: environment, objects, activities,
            text_visible, people_present, potential_hazards, scene_tags.
        """
        try:
            result = self.llm_client.complete_json_with_images(
                user_prompt=SCENE_ANALYSIS_USER_PROMPT,
                images=[frame_base64],
                system_prompt=SCENE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=self.max_tokens,
            )
            return self._normalize(result)
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return self._empty_result()

    def _normalize(self, raw: dict) -> dict:
        """Ensure all expected fields exist with correct types."""
        return {
            "environment": raw.get("environment", "unknown"),
            "objects": raw.get("objects", []),
            "activities": raw.get("activities", []),
            "text_visible": raw.get("text_visible", []),
            "people_present": bool(raw.get("people_present", False)),
            "potential_hazards": raw.get("potential_hazards", []),
            "scene_tags": raw.get("scene_tags", []),
        }

    def _empty_result(self) -> dict:
        """Return an empty scene analysis result."""
        return {
            "environment": "unknown",
            "objects": [],
            "activities": [],
            "text_visible": [],
            "people_present": False,
            "potential_hazards": [],
            "scene_tags": [],
        }
