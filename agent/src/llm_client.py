"""
Azure OpenAI LLM Client

封装 GPT-4o 调用，支持组件选择和 props 填充
"""

import os
import json
import time
from typing import Any
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Azure GPT-4o 客户端"""

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
        return json.loads(response)
