"""
Azure OpenAI Client

Implementation of LLM client for Azure OpenAI service.
"""

import logging
from typing import Optional, Any

from openai import AzureOpenAI

from .base import LLMClientBase, ModelConfig

logger = logging.getLogger(__name__)


class AzureOpenAIClient(LLMClientBase):
    """Azure OpenAI client implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not config.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        if not config.api_key:
            raise ValueError("Azure OpenAI API key is required")

        api_version = config.extra.get("api_version", "2024-02-15-preview")

        self.client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=api_version,
        )

        logger.info(f"Azure OpenAI client initialized: {config.model_name}")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """Execute a text completion."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _call():
            kwargs = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

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
        content = []

        # Add images first
        for image_b64 in images:
            content.append(self._format_image_content(image_b64, image_detail))

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.info(f"VLM call with {len(images)} images")

        def _call():
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

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
        """Execute multimodal completion and parse JSON response.

        Azure OpenAI supports response_format with vision, so we use it.
        """
        content = []

        # Add images first
        for image_b64 in images:
            content.append(self._format_image_content(image_b64, image_detail))

        # Add text prompt
        content.append({"type": "text", "text": user_prompt})

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.info(f"VLM JSON call with {len(images)} images")

        def _call():
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        result = self._retry_with_backoff(_call)
        return self._parse_json_response(result)

    def _format_image_content(self, image_b64: str, detail: str = "auto") -> dict:
        """Format image for Azure OpenAI API."""
        # Handle data URL prefix
        if image_b64.startswith("data:"):
            image_url = image_b64
        else:
            image_url = f"data:image/jpeg;base64,{image_b64}"

        return {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": detail,
            },
        }
