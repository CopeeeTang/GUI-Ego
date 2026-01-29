"""Visual context generation using VLM for scene understanding."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


class VisualContextMode(Enum):
    """Mode for visual context generation."""

    DIRECT = "direct"  # Pass images directly to multimodal LLM
    DESCRIPTION = "description"  # Generate text description first


@dataclass
class VisualContext:
    """Container for visual context data.

    Attributes:
        mode: The mode used to generate this context.
        frames_base64: Base64-encoded frames (for DIRECT mode).
        description: Text description of the scene (for DESCRIPTION mode).
        frame_descriptions: Individual descriptions for each frame.
        metadata: Additional metadata about the visual context.
    """

    mode: VisualContextMode
    frames_base64: Optional[list[str]] = None
    description: Optional[str] = None
    frame_descriptions: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"mode": self.mode.value}

        if self.frames_base64:
            result["frames_base64"] = self.frames_base64
        if self.description:
            result["description"] = self.description
        if self.frame_descriptions:
            result["frame_descriptions"] = self.frame_descriptions
        if self.metadata:
            result["metadata"] = self.metadata

        return result


class VisualContextGenerator:
    """Generate visual context from video frames using VLM.

    This class supports two modes:
    1. DIRECT: Return base64 images for multimodal LLM consumption
    2. DESCRIPTION: Use VLM to generate text descriptions of the scene

    Attributes:
        llm_client: LLM client for VLM calls.
        mode: Visual context generation mode.
    """

    # Prompt for generating scene descriptions
    SCENE_DESCRIPTION_PROMPT = """Analyze these {num_frames} first-person perspective images from smart glasses.

Describe the visual context in a structured format:

1. **Environment**:
   - Type (indoor/outdoor)
   - Specific location (store, street, office, etc.)
   - Lighting conditions

2. **Key Objects**:
   - Main objects visible
   - Their approximate positions (left, center, right, foreground, background)
   - Any text or signage visible

3. **User Activity**:
   - What the user appears to be doing
   - Direction of movement (if any)
   - Objects they might be interacting with

4. **Context for UI**:
   - Suitable anchor points for AR labels
   - Visibility conditions
   - Potential occlusion areas

Provide a concise but comprehensive description that can inform UI generation for smart glasses."""

    SINGLE_FRAME_PROMPT = """Describe this first-person perspective image from smart glasses.

Focus on:
1. Environment type and location
2. Main objects and their positions
3. Any text or signage visible
4. What the user might be doing

Keep the description concise (2-3 sentences)."""

    def __init__(
        self,
        llm_client: "LLMClient",
        mode: str | VisualContextMode = VisualContextMode.DESCRIPTION,
    ):
        """Initialize the visual context generator.

        Args:
            llm_client: LLM client with VLM support.
            mode: Generation mode ("direct" or "description").
        """
        self.llm_client = llm_client

        if isinstance(mode, str):
            self.mode = VisualContextMode(mode.lower())
        else:
            self.mode = mode

        logger.info(f"Initialized VisualContextGenerator with mode: {self.mode.value}")

    def generate_context(
        self,
        frames: list[np.ndarray],
        recommendation: Optional[Any] = None,
        include_individual_descriptions: bool = False,
    ) -> VisualContext:
        """Generate visual context from video frames.

        Args:
            frames: List of video frames as numpy arrays (BGR format).
            recommendation: Optional recommendation object for additional context.
            include_individual_descriptions: Whether to generate per-frame descriptions.

        Returns:
            VisualContext object containing the generated context.
        """
        # Convert frames to base64
        base64_frames = self._frames_to_base64(frames)

        if self.mode == VisualContextMode.DIRECT:
            return self._generate_direct_context(base64_frames)
        else:
            return self._generate_description_context(
                frames,
                base64_frames,
                recommendation,
                include_individual_descriptions,
            )

    def _frames_to_base64(
        self,
        frames: list[np.ndarray],
        format: str = "jpeg",
        quality: int = 85,
    ) -> list[str]:
        """Convert frames to base64 strings."""
        import base64
        import cv2

        base64_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if format.lower() == "jpeg":
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, buffer = cv2.imencode(".jpg", frame_rgb, encode_params)
            else:
                success, buffer = cv2.imencode(".png", frame_rgb)

            if success:
                b64_string = base64.b64encode(buffer).decode("utf-8")
                base64_frames.append(b64_string)

        return base64_frames

    def _generate_direct_context(
        self,
        base64_frames: list[str],
    ) -> VisualContext:
        """Generate context for direct image passing mode.

        In this mode, we simply return the base64-encoded images
        for the multimodal LLM to process directly.
        """
        return VisualContext(
            mode=VisualContextMode.DIRECT,
            frames_base64=base64_frames,
            metadata={
                "num_frames": len(base64_frames),
                "encoding": "base64",
                "format": "jpeg",
            },
        )

    def _generate_description_context(
        self,
        frames: list[np.ndarray],
        base64_frames: list[str],
        recommendation: Optional[Any] = None,
        include_individual_descriptions: bool = False,
    ) -> VisualContext:
        """Generate text description context using VLM.

        This mode uses the VLM to analyze the frames and generate
        a text description that can be used with text-only LLMs.
        """
        # Generate overall scene description
        prompt = self.SCENE_DESCRIPTION_PROMPT.format(num_frames=len(frames))

        try:
            description = self.llm_client.complete_with_images(
                prompt=prompt,
                images=base64_frames,
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"Failed to generate scene description: {e}")
            description = self._generate_fallback_description(len(frames))

        # Optionally generate individual frame descriptions
        frame_descriptions = None
        if include_individual_descriptions:
            frame_descriptions = self._generate_individual_descriptions(base64_frames)

        return VisualContext(
            mode=VisualContextMode.DESCRIPTION,
            description=description,
            frame_descriptions=frame_descriptions,
            metadata={
                "num_frames": len(frames),
                "has_individual_descriptions": include_individual_descriptions,
            },
        )

    def _generate_individual_descriptions(
        self,
        base64_frames: list[str],
    ) -> list[str]:
        """Generate description for each frame individually."""
        descriptions = []

        for i, frame_b64 in enumerate(base64_frames):
            try:
                desc = self.llm_client.complete_with_images(
                    prompt=self.SINGLE_FRAME_PROMPT,
                    images=[frame_b64],
                    max_tokens=150,
                )
                descriptions.append(desc)
            except Exception as e:
                logger.error(f"Failed to describe frame {i}: {e}")
                descriptions.append(f"Frame {i + 1}: Unable to generate description")

        return descriptions

    def _generate_fallback_description(self, num_frames: int) -> str:
        """Generate a fallback description when VLM fails."""
        return (
            f"Visual context from {num_frames} first-person perspective frames. "
            "Unable to generate detailed description due to processing error. "
            "Please use default UI generation approach."
        )

    def format_for_prompt(self, context: VisualContext) -> str:
        """Format visual context for inclusion in a prompt.

        Args:
            context: The visual context to format.

        Returns:
            Formatted string suitable for prompt injection.
        """
        if context.mode == VisualContextMode.DIRECT:
            return "[Visual context: Images will be passed directly to the model]"

        parts = ["## Visual Context (from video frames)"]

        if context.description:
            parts.append(context.description)

        if context.frame_descriptions:
            parts.append("\n### Frame-by-Frame Details:")
            for i, desc in enumerate(context.frame_descriptions):
                parts.append(f"- Frame {i + 1}: {desc}")

        return "\n\n".join(parts)


class VisualContextCache:
    """Cache for visual context to avoid redundant VLM calls.

    This cache stores generated visual contexts keyed by
    (video_path, start_time, end_time, num_frames, mode) tuples.
    """

    def __init__(self, max_size: int = 100):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache.
        """
        self.max_size = max_size
        self._cache: dict[tuple, VisualContext] = {}
        self._access_order: list[tuple] = []

    def _make_key(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_frames: int,
        mode: str,
    ) -> tuple:
        """Create a cache key from parameters."""
        return (video_path, round(start_time, 2), round(end_time, 2), num_frames, mode)

    def get(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_frames: int,
        mode: str,
    ) -> Optional[VisualContext]:
        """Get cached visual context if available."""
        key = self._make_key(video_path, start_time, end_time, num_frames, mode)

        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Cache hit for visual context: {key}")
            return self._cache[key]

        return None

    def set(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        num_frames: int,
        mode: str,
        context: VisualContext,
    ) -> None:
        """Store visual context in cache."""
        key = self._make_key(video_path, start_time, end_time, num_frames, mode)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")

        self._cache[key] = context
        self._access_order.append(key)
        logger.debug(f"Cached visual context: {key}")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Visual context cache cleared")
