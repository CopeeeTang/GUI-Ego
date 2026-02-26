"""
Vertex AI Gemini Client

Uses Google Application Default Credentials (ADC) — no API key needed.
Provides enterprise-grade quota (no 20/day free limit).

Usage:
    create_client("vertex:gemini-2.0-flash")
    create_client("vertex:gemini-2.5-flash")

ADC setup:
    gcloud auth application-default login
    # Or set GOOGLE_APPLICATION_CREDENTIALS to service account key path
"""

import base64
import logging
import os
import signal
from typing import Any, Optional

from .base import LLMClientBase, ModelConfig

logger = logging.getLogger(__name__)


# Per-request timeout (seconds) — prevents hung HTTP connections.
# Thinking models (gemini-2.5/3.x) typically finish within 60s.
REQUEST_TIMEOUT_SEC = 120


class _VertexTimeout(Exception):
    """Raised when a Vertex AI call exceeds REQUEST_TIMEOUT_SEC."""


class VertexGeminiClient(LLMClientBase):
    """Vertex AI Gemini client using Application Default Credentials."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform not installed. "
                "Run: pip install google-cloud-aiplatform"
            )

        # Resolve project and region
        project_id = config.extra.get("project_id") or os.environ.get("VERTEX_PROJECT")
        if not project_id:
            # Try to get from ADC
            try:
                import google.auth
                _, project_id = google.auth.default()
            except Exception:
                pass
        if not project_id:
            raise ValueError(
                "Vertex AI project ID not found. "
                "Set VERTEX_PROJECT env var or ensure ADC has a project."
            )

        region = config.extra.get("region") or os.environ.get("VERTEX_REGION", "global")

        vertexai.init(project=project_id, location=region)
        self._model = GenerativeModel(config.model_name)
        self._GenerativeModel = GenerativeModel
        self._model_name = config.model_name

        logger.info(f"Vertex AI client initialized: {config.model_name} ({project_id}/{region})")

    def _call_with_timeout(self, func, timeout_sec=REQUEST_TIMEOUT_SEC):
        """Run func() with a SIGALRM-based timeout (Linux only)."""
        def _handler(signum, frame):
            raise _VertexTimeout(
                f"Vertex AI call timed out after {timeout_sec}s"
            )
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout_sec)
        try:
            return func()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """Execute a text completion."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_format and response_format.get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"

        def _call():
            response = self._call_with_timeout(
                lambda: self._model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
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
        from vertexai.generative_models import Part

        content_parts = []

        for image_b64 in images:
            image_bytes = self._decode_image(image_b64)
            content_parts.append(Part.from_data(data=image_bytes, mime_type="image/jpeg"))

        if system_prompt:
            content_parts.append(f"{system_prompt}\n\n{prompt}")
        else:
            content_parts.append(prompt)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        logger.info(f"Vertex VLM call with {len(images)} images")

        def _call():
            response = self._call_with_timeout(
                lambda: self._model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                )
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
        from vertexai.generative_models import Part

        content_parts = []

        for image_b64 in images:
            image_bytes = self._decode_image(image_b64)
            content_parts.append(Part.from_data(data=image_bytes, mime_type="image/jpeg"))

        if system_prompt:
            content_parts.append(f"{system_prompt}\n\n{user_prompt}")
        else:
            content_parts.append(user_prompt)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
        }

        logger.info(f"Vertex VLM JSON call with {len(images)} images")

        def _call():
            response = self._call_with_timeout(
                lambda: self._model.generate_content(
                    content_parts,
                    generation_config=generation_config,
                )
            )
            return self._extract_response_text(response)

        result = self._retry_with_backoff(_call)
        return self._parse_json_response(result)

    def _extract_response_text(self, response) -> str:
        """Safely extract text from Vertex AI response.

        Handles thinking models (gemini-2.5/3.x) that include thought_signature
        binary fields alongside text parts.
        """
        if not hasattr(response, "candidates") or not response.candidates:
            raise ValueError("Vertex AI returned empty response - no candidates")

        candidate = response.candidates[0]

        if hasattr(candidate, "finish_reason"):
            finish_reason = str(candidate.finish_reason)
            if "SAFETY" in finish_reason:
                raise ValueError(f"Content blocked by safety filter: {finish_reason}")

        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            parts = candidate.content.parts
            if parts:
                text_parts = []
                for part in parts:
                    # Skip thought-only parts (thought_signature without text)
                    try:
                        t = part.text
                        if t:
                            text_parts.append(t)
                    except Exception:
                        pass
                text = "".join(text_parts)
                if text.strip():
                    return text

        # Fallback: use response.text (may raise on thinking models)
        try:
            text = response.text
            if text and text.strip():
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from Vertex response: {e}")

        raise ValueError("Vertex AI returned empty response")

    def _format_image_content(self, image_b64: str, detail: str = "auto") -> dict:
        """Format image for Vertex AI (returns dict for base class compatibility)."""
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        return {"mime_type": "image/jpeg", "data": image_b64}

    def _decode_image(self, image_b64: str) -> bytes:
        """Decode base64 image string to bytes."""
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        return base64.b64decode(image_b64)
