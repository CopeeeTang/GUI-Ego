"""
LLM Client Base Class

Abstract base class for all LLM providers with unified interface.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    provider: str  # azure, gemini, claude
    model_name: str  # gpt-4o, gemini-2.5-pro, claude-sonnet-4-5
    is_thinking: bool = False  # Whether model supports extended thinking
    api_key: Optional[str] = None
    endpoint: Optional[str] = None  # Azure endpoint or custom base URL
    proxy: Optional[str] = None  # Proxy URL for Gemini/Claude
    max_retries: int = 3
    retry_delay: float = 1.0
    extra: dict = field(default_factory=dict)  # Provider-specific options


class LLMClientBase(ABC):
    """Abstract base class for LLM clients.

    All provider implementations must inherit from this class and implement
    the abstract methods for consistent API across providers.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def is_thinking(self) -> bool:
        return self.config.is_thinking

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {sleep_time}s..."
                    )
                    time.sleep(sleep_time)
                    continue
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} attempts: {last_error}"
        )

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """Execute a text completion.

        Args:
            system_prompt: System instruction.
            user_prompt: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional format specification (e.g., {"type": "json_object"}).

        Returns:
            Model response text.
        """
        pass

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Execute completion and parse JSON response.

        Args:
            system_prompt: System instruction.
            user_prompt: User message.
            temperature: Sampling temperature (lower for JSON).
            max_tokens: Maximum output tokens.

        Returns:
            Parsed JSON dictionary.
        """
        response = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return self._parse_json_response(response)

    @abstractmethod
    def complete_with_images(
        self,
        prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> str:
        """Execute multimodal completion with images.

        Args:
            prompt: User text prompt.
            images: List of base64-encoded images.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            image_detail: Image detail level (provider-specific).

        Returns:
            Model response text.
        """
        pass

    def complete_json_with_images(
        self,
        user_prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> dict[str, Any]:
        """Execute multimodal completion and parse JSON response.

        Args:
            user_prompt: User text prompt.
            images: List of base64-encoded images.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            image_detail: Image detail level.

        Returns:
            Parsed JSON dictionary.
        """
        # Add JSON instruction to prompt if not already present
        json_prompt = user_prompt
        if "json" not in user_prompt.lower():
            json_prompt = f"{user_prompt}\n\nRespond with valid JSON only."

        response = self.complete_with_images(
            prompt=json_prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_detail=image_detail,
        )
        return self._parse_json_response(response)

    @abstractmethod
    def _format_image_content(self, image_b64: str, detail: str = "auto") -> dict:
        """Format image for provider-specific API.

        Args:
            image_b64: Base64-encoded image string.
            detail: Detail level hint.

        Returns:
            Provider-specific image content dict.
        """
        pass

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from response, handling markdown code blocks."""
        text = response.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content (first 500 chars): {text[:500]}")
            raise ValueError(f"Invalid JSON from LLM: {str(e)}")

    def estimate_image_tokens(self, images: list[str], detail: str = "auto") -> int:
        """Estimate token usage for images.

        Based on OpenAI's documentation as baseline:
        - low detail: 85 tokens per image
        - high detail: 85 + 170 * tiles

        Args:
            images: List of base64-encoded images.
            detail: Detail level.

        Returns:
            Estimated token count.
        """
        if detail == "low":
            return len(images) * 85
        # Conservative estimate for high/auto
        return len(images) * 765
