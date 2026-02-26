"""
GPT-4o API Model

Uses Azure OpenAI endpoint for GPT-4o vision capabilities.
Loads credentials from .env file.
"""

import base64
import io
import json
import logging
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image

from src.models.base import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)

# Load from project root .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


class GPT4oModel(BaseVLM):
    """GPT-4o via Azure OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_version: str = "2024-09-01-preview",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self._model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", os.getenv("azure_endpoint", ""))
        api_key = os.getenv("AZURE_OPENAI_API_KEY", os.getenv("api_key", ""))

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
            )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        logger.info(f"GPT-4o client initialized (endpoint: {endpoint[:30]}...)")

    @property
    def model_name(self) -> str:
        return f"azure:{self._model}"

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> VLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user content
        content = []
        if images:
            for img in images:
                b64 = _pil_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = resp.choices[0].message.content or ""
                usage = None
                if resp.usage:
                    usage = {
                        "input_tokens": resp.usage.prompt_tokens,
                        "output_tokens": resp.usage.completion_tokens,
                    }
                return VLMResponse(text=text.strip(), model=self.model_name, usage=usage)

            except Exception as e:
                logger.warning(f"GPT-4o attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> dict:
        if system_prompt:
            system_prompt += "\nRespond ONLY with valid JSON."
        else:
            system_prompt = "Respond ONLY with valid JSON."

        response = self.generate(prompt, images, system_prompt, max_tokens, temperature=0.1)
        return _parse_json(response.text)


def _pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 JPEG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json(text: str) -> dict:
    """Parse JSON from GPT-4o response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from: {text[:200]}")
