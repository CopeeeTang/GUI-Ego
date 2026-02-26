"""
Qwen3-VL Local Model

Runs Qwen3-VL-8B-Instruct (or other sizes) locally on A100 80GB.
Supports image and video input via the qwen_vl_utils processor.
"""

import json
import logging
import re
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.models.base import BaseVLM, VLMResponse

logger = logging.getLogger(__name__)


class QwenVLModel(BaseVLM):
    """Local Qwen3-VL model for streaming video understanding."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: Optional[str] = None,
    ):
        self._model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        logger.info(f"Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info(f"Model loaded on {device}")

    @property
    def model_name(self) -> str:
        return self._model_id

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> VLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with interleaved images
        content = []
        if images:
            for img in images:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        # Process
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if images:
            inputs = self.processor(
                text=[text_input],
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text_input],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True)

        return VLMResponse(
            text=text.strip(),
            model=self._model_id,
            usage={"input_tokens": input_len, "output_tokens": len(generated)},
        )

    def generate_json(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> dict:
        if system_prompt:
            system_prompt += "\nRespond ONLY with valid JSON."
        else:
            system_prompt = "Respond ONLY with valid JSON."

        response = self.generate(prompt, images, system_prompt, max_tokens, temperature=0.1)
        return _parse_json(response.text)


def _parse_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from: {text[:200]}")
