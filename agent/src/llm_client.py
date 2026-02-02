"""
Azure OpenAI LLM Client

Backward-compatible wrapper for the new multi-provider LLM module.
This module is DEPRECATED - use `from agent.src.llm import create_client` instead.
"""

import warnings
from typing import Any, Optional

from .llm import create_client, LLMClientBase

# Re-export for backward compatibility
__all__ = ["LLMClient"]


class LLMClient:
    """Backward-compatible LLM client wrapper.

    DEPRECATED: Use `from agent.src.llm import create_client` instead.

    This class wraps the new LLM module to maintain backward compatibility
    with existing code that uses the old LLMClient interface.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment_name: str = "gpt-4o",
        api_version: str = "2024-02-15-preview",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the backward-compatible LLM client.

        Args:
            endpoint: Azure OpenAI endpoint (ignored, uses env vars).
            api_key: Azure OpenAI API key (ignored, uses env vars).
            deployment_name: Model deployment name.
            api_version: API version (passed to client).
            max_retries: Maximum retry attempts.
            retry_delay: Base delay between retries.
        """
        # Create the underlying client using the new factory
        self._client: LLMClientBase = create_client(
            f"azure:{deployment_name}",
            max_retries=max_retries,
            retry_delay=retry_delay,
            api_version=api_version,
        )

        # Store for compatibility
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        """Execute LLM call."""
        return self._client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """Execute LLM call and return JSON."""
        return self._client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_with_images(
        self,
        prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> str:
        """Execute multimodal VLM call."""
        return self._client.complete_with_images(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_detail=image_detail,
        )

    def complete_json_with_images(
        self,
        user_prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> dict[str, Any]:
        """Execute multimodal call and return JSON."""
        return self._client.complete_json_with_images(
            user_prompt=user_prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_detail=image_detail,
        )

    def estimate_image_tokens(self, images: list[str], detail: str = "auto") -> int:
        """Estimate token usage for images."""
        return self._client.estimate_image_tokens(images, detail)
