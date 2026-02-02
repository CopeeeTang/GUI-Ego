"""
Claude Client

Implementation of LLM client for Anthropic Claude API via local proxy.
"""

import logging
from typing import Optional, Any

from .base import LLMClientBase, ModelConfig

logger = logging.getLogger(__name__)


class ClaudeClient(LLMClientBase):
    """Anthropic Claude client implementation with extended thinking support."""

    # Default thinking budget for thinking models
    DEFAULT_THINKING_BUDGET = 10000

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not config.api_key:
            raise ValueError("Anthropic API key is required")

        # Import here to avoid import errors if not installed
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        # Configure client with local proxy as base_url
        # Use proxy as base_url (e.g., http://127.0.0.1:7890)
        proxy_url = config.proxy or "http://127.0.0.1:7890"

        self.client = anthropic.Anthropic(
            base_url=proxy_url,
            api_key=config.api_key,
        )

        self._thinking_budget = config.extra.get(
            "thinking_budget", self.DEFAULT_THINKING_BUDGET
        )

        logger.info(
            f"Claude client initialized: {config.model_name} "
            f"(thinking={config.is_thinking}, proxy={proxy_url})"
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """Execute a text completion."""
        messages = [{"role": "user", "content": user_prompt}]

        def _call():
            kwargs = {
                "model": self.config.model_name,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            # Handle thinking mode
            if self.config.is_thinking:
                # Thinking models require temperature=1.0
                kwargs["temperature"] = 1.0
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget,
                }
            else:
                kwargs["temperature"] = temperature

            response = self.client.messages.create(**kwargs)

            # Extract text from response (may include thinking blocks)
            return self._extract_text_response(response)

        return self._retry_with_backoff(_call)

    def complete_with_images(
        self,
        prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> str:
        """Execute multimodal completion with images."""
        # Build content with images
        content = []

        for image_b64 in images:
            content.append(self._format_image_content(image_b64, image_detail))

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        logger.info(f"Claude VLM call with {len(images)} images")

        def _call():
            kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            # Handle thinking mode
            if self.config.is_thinking:
                kwargs["temperature"] = 1.0
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget,
                }
            else:
                kwargs["temperature"] = temperature

            response = self.client.messages.create(**kwargs)
            return self._extract_text_response(response)

        return self._retry_with_backoff(_call)

    def complete_json_with_images(
        self,
        user_prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> dict[str, Any]:
        """Execute multimodal completion and parse JSON response."""
        # Add JSON instruction to prompt
        json_prompt = user_prompt
        if "json" not in user_prompt.lower():
            json_prompt = f"{user_prompt}\n\nRespond with valid JSON only, no markdown."

        # Build content with images
        content = []

        for image_b64 in images:
            content.append(self._format_image_content(image_b64, image_detail))

        content.append({"type": "text", "text": json_prompt})

        messages = [{"role": "user", "content": content}]

        logger.info(f"Claude VLM JSON call with {len(images)} images")

        def _call():
            kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            # Handle thinking mode - use lower temp but thinking requires 1.0
            if self.config.is_thinking:
                kwargs["temperature"] = 1.0
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget,
                }
            else:
                kwargs["temperature"] = temperature

            response = self.client.messages.create(**kwargs)
            return self._extract_text_response(response)

        result = self._retry_with_backoff(_call)
        return self._parse_json_response(result)

    def _format_image_content(self, image_b64: str, detail: str = "auto") -> dict:
        """Format image for Claude API."""
        # Strip data URL prefix if present
        if image_b64.startswith("data:"):
            # Extract media type and base64 data
            if "," in image_b64:
                header, data = image_b64.split(",", 1)
                # Extract media type from header like "data:image/jpeg;base64"
                media_type = header.replace("data:", "").replace(";base64", "")
            else:
                data = image_b64
                media_type = "image/jpeg"
        else:
            data = image_b64
            media_type = "image/jpeg"

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def _extract_text_response(self, response) -> str:
        """Extract text from Claude response, handling thinking blocks.

        Args:
            response: Anthropic API response object.

        Returns:
            Extracted text content.
        """
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                # Log thinking for debugging but don't include in response
                logger.debug(f"Thinking: {block.thinking[:200]}...")

        return "".join(text_parts)
