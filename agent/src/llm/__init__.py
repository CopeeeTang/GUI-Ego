"""
LLM Module

Multi-provider LLM client abstraction supporting Azure OpenAI, Gemini, and Claude.

Usage:
    from agent.src.llm import create_client

    # Azure OpenAI (default)
    client = create_client("azure:gpt-4o")

    # Gemini
    client = create_client("gemini:gemini-2.5-pro")

    # Claude with extended thinking
    client = create_client("claude:claude-opus-4-5-thinking")

    # Use the client
    response = client.complete(system_prompt="...", user_prompt="...")
    json_response = client.complete_json(system_prompt="...", user_prompt="...")
    vision_response = client.complete_with_images(prompt="...", images=[...])
"""

from .base import LLMClientBase, ModelConfig
from .factory import create_client, parse_model_spec, is_thinking_model, list_available_models
from .config import LLMConfig

__all__ = [
    # Main factory function
    "create_client",
    # Base classes
    "LLMClientBase",
    "ModelConfig",
    # Utilities
    "parse_model_spec",
    "is_thinking_model",
    "list_available_models",
    "LLMConfig",
]
