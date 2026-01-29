"""
Azure OpenAI LLM Client

封装 GPT-4o 调用，支持组件选择、props 填充和多模态 VLM 调用
"""

import os
import json
import time
import logging
from typing import Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Azure GPT-4o 客户端，支持文本和多模态调用"""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment_name: str = "gpt-4o",
        api_version: str = "2024-02-15-preview",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("azure_endpoint")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("api_key")
        self.deployment_name = deployment_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables."
            )

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=api_version,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: dict | None = None,
    ) -> str:
        """执行 LLM 调用"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.deployment_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {e}")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """执行 LLM 调用并返回 JSON"""
        response = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content (first 500 chars): {response[:500]}")
            raise ValueError(f"Invalid JSON from LLM: {str(e)}")

    def complete_with_images(
        self,
        prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> str:
        """执行带图像的多模态 VLM 调用

        Args:
            prompt: 用户提示文本
            images: Base64 编码的图像列表
            system_prompt: 系统提示（可选）
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            image_detail: 图像细节级别 ("low", "high", "auto")

        Returns:
            模型响应文本
        """
        # 构建多模态消息内容
        content = []

        # 添加图像
        for i, image_b64 in enumerate(images):
            # 确保图像是纯 base64，不包含 data URL 前缀
            if image_b64.startswith("data:"):
                # 已经是 data URL 格式
                image_url = image_b64
            else:
                # 添加 data URL 前缀
                image_url = f"data:image/jpeg;base64,{image_b64}"

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": image_detail,
                },
            })

        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt,
        })

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.info(f"VLM call with {len(images)} images")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"VLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"VLM call failed after {self.max_retries} attempts: {e}")

    def complete_json_with_images(
        self,
        user_prompt: str,
        images: list[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        image_detail: str = "auto",
    ) -> dict[str, Any]:
        """执行带图像的多模态调用并返回 JSON

        Args:
            user_prompt: 用户提示文本
            images: Base64 编码的图像列表
            system_prompt: 系统提示（可选）
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            image_detail: 图像细节级别

        Returns:
            解析后的 JSON 字典
        """
        # 构建多模态消息内容
        content = []

        # 添加图像
        for image_b64 in images:
            if image_b64.startswith("data:"):
                image_url = image_b64
            else:
                image_url = f"data:image/jpeg;base64,{image_b64}"

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": image_detail,
                },
            })

        # 添加文本提示
        content.append({
            "type": "text",
            "text": user_prompt,
        })

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.info(f"VLM JSON call with {len(images)} images")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                result = response.choices[0].message.content
                return json.loads(result)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to parse JSON after {self.max_retries} attempts")

            except Exception as e:
                logger.warning(f"VLM JSON call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(f"VLM call failed after {self.max_retries} attempts: {e}")

    def estimate_image_tokens(self, images: list[str], detail: str = "auto") -> int:
        """估算图像消耗的 token 数

        Based on OpenAI's documentation:
        - low detail: 85 tokens per image
        - high detail: 85 + 170 * (tiles), where tiles depend on image size

        Args:
            images: Base64 编码的图像列表
            detail: 图像细节级别

        Returns:
            估算的 token 数
        """
        if detail == "low":
            return len(images) * 85

        # For high/auto, estimate based on typical smart glasses resolution (960x720)
        # This would typically be 2 tiles (1 horizontal, 2 vertical) = 85 + 170*2 = 425
        # But we use a conservative estimate
        return len(images) * 765  # Approximate for typical resolution
