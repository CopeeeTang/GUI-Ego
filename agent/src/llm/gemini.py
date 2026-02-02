"""
Gemini Client

Implementation of LLM client for Google Gemini API via local proxy.
"""

import logging
from typing import Optional, Any

from .base import LLMClientBase, ModelConfig

logger = logging.getLogger(__name__)


class GeminiClient(LLMClientBase):
    """Google Gemini client implementation via local proxy."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not config.api_key:
            raise ValueError("Gemini API key is required")

        # Import here to avoid import errors if not installed
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

        # Configure with local proxy endpoint
        # Use proxy as api_endpoint (e.g., http://127.0.0.1:7890)
        proxy_endpoint = config.proxy or "http://127.0.0.1:7890"

        genai.configure(
            api_key=config.api_key,
            transport="rest",
            client_options={"api_endpoint": proxy_endpoint},
        )

        self._genai = genai
        self._model = genai.GenerativeModel(config.model_name)

        logger.info(f"Gemini client initialized: {config.model_name} (proxy: {proxy_endpoint})")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """Execute a text completion."""
        # Combine system and user prompts for Gemini
        # Gemini uses system_instruction in model config or prepends to content
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        # Handle JSON response format
        if response_format and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        def _call():
            response = self._model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            return response.text

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
        # Build content parts
        content_parts = []

        # Add images
        for image_b64 in images:
            content_parts.append(self._format_image_content(image_b64, image_detail))

        # Add text (combine system + user prompt)
        if system_prompt:
            content_parts.append(f"{system_prompt}\n\n{prompt}")
        else:
            content_parts.append(prompt)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        logger.info(f"Gemini VLM call with {len(images)} images")

        def _call():
            response = self._model.generate_content(
                content_parts,
                generation_config=generation_config,
            )
            return response.text

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
        # Build content parts
        content_parts = []

        # Add images
        for image_b64 in images:
            content_parts.append(self._format_image_content(image_b64, image_detail))

        # Add text (combine system + user prompt)
        if system_prompt:
            content_parts.append(f"{system_prompt}\n\n{user_prompt}")
        else:
            content_parts.append(user_prompt)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }

        logger.info(f"Gemini VLM JSON call with {len(images)} images")

        def _call():
            response = self._model.generate_content(
                content_parts,
                generation_config=generation_config,
            )
            return response.text

        result = self._retry_with_backoff(_call)
        return self._parse_json_response(result)

    def _format_image_content(self, image_b64: str, detail: str = "auto") -> dict:
        """Format image for Gemini API."""
        # Strip data URL prefix if present
        if image_b64.startswith("data:"):
            # Extract base64 data after the comma
            image_b64 = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64

        return {
            "mime_type": "image/jpeg",
            "data": image_b64,
        }
