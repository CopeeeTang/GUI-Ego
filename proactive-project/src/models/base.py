"""
Base VLM Interface

Abstract interface for Vision-Language Models.
Supports both local models (Qwen3-VL) and API models (GPT-4o).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass
class VLMResponse:
    """Unified response from any VLM."""
    text: str
    model: str
    usage: Optional[dict] = None  # token counts if available


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> VLMResponse:
        """
        Generate text response given prompt and optional images.

        Args:
            prompt: User text prompt
            images: Optional list of PIL images
            system_prompt: Optional system instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        ...

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> dict:
        """Generate and parse JSON response."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
