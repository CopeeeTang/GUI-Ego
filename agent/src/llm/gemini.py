"""
Gemini Client

Implementation of LLM client for Google Gemini API.
Supports both direct official API access and local proxy.
"""

import logging
from typing import Optional, Any

from .base import LLMClientBase, ModelConfig

logger = logging.getLogger(__name__)

# Short name → official API model ID mapping
# Users can write either short names or full official names.
GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3-pro-image": "gemini-3-pro-image-preview",
    "gemini-3-flash-image": "gemini-3-flash-image-preview",
}


def resolve_gemini_model(model_name: str, use_proxy: bool) -> str:
    """Resolve model name, applying aliases for official API.

    When using a proxy, pass names through as-is (proxy may have its own routing).
    When direct, map short names to official `-preview` suffixed names.
    """
    if use_proxy:
        return model_name
    return GEMINI_MODEL_ALIASES.get(model_name, model_name)


class GeminiClient(LLMClientBase):
    """Google Gemini client — supports direct API and proxy modes."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not config.api_key:
            raise ValueError("Gemini API key is required")

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

        # Decide mode: proxy or direct
        use_proxy = bool(config.proxy)

        configure_kwargs = {
            "api_key": config.api_key,
            "transport": "rest",
        }

        if use_proxy:
            configure_kwargs["client_options"] = {"api_endpoint": config.proxy}
            mode_label = f"proxy: {config.proxy}"
        else:
            mode_label = "direct (official API)"

        genai.configure(**configure_kwargs)

        # Resolve model name (alias → official name when direct)
        resolved_name = resolve_gemini_model(config.model_name, use_proxy)

        self._genai = genai
        self._model = genai.GenerativeModel(resolved_name)

        logger.info(
            f"Gemini client initialized: {resolved_name} ({mode_label})"
            + (f" [alias from {config.model_name}]" if resolved_name != config.model_name else "")
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
            return self._extract_response_text(response)

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
            return self._extract_response_text(response)

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
            return self._extract_response_text(response)

        result = self._retry_with_backoff(_call)
        return self._parse_json_response(result)

    def _extract_response_text(self, response) -> str:
        """Safely extract text from Gemini response, handling filters and errors."""
        # Check if response was blocked by safety filters
        if hasattr(response, 'prompt_feedback'):
            feedback = response.prompt_feedback
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                logger.warning(f"Gemini response blocked: {feedback.block_reason}")
                raise ValueError(f"Content blocked by Gemini safety filter: {feedback.block_reason}")

        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            logger.error("Gemini response has no candidates")
            raise ValueError("Gemini returned empty response - no candidates")

        candidate = response.candidates[0]

        # Check finish reason
        if hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
            if 'SAFETY' in finish_reason:
                logger.warning(f"Response blocked by safety: {finish_reason}")
                raise ValueError(f"Content blocked by safety filter: {finish_reason}")
            elif finish_reason not in ['STOP', 'MAX_TOKENS', '1']:  # 1 is STOP enum value
                logger.warning(f"Unexpected finish reason: {finish_reason}")

        # Try to get text
        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
            parts = candidate.content.parts
            if parts:
                text = ''.join(part.text for part in parts if hasattr(part, 'text'))
                if text.strip():
                    return text

        # Fallback to response.text
        try:
            text = response.text
            if text and text.strip():
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}")

        # If we got here, response is empty
        logger.error("Gemini response is empty or has no text content")
        raise ValueError("Gemini returned empty response")

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
