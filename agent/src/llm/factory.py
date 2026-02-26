"""
LLM Client Factory

Factory pattern for creating LLM clients based on model specification.
"""

import re
import logging
from typing import TYPE_CHECKING

from .base import ModelConfig, LLMClientBase
from .config import LLMConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Thinking model patterns
THINKING_MODEL_PATTERNS = [
    r"-thinking$",  # claude-opus-4-5-thinking
    r"-thinking-",  # future models like model-thinking-v2
]


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse model specification string.

    Format: provider:model_name
    Examples:
        - "azure:gpt-4o" -> ("azure", "gpt-4o")
        - "gemini:gemini-2.5-pro" -> ("gemini", "gemini-2.5-pro")
        - "claude:claude-sonnet-4-5" -> ("claude", "claude-sonnet-4-5")
        - "gpt-4o" -> ("azure", "gpt-4o")  # Default to Azure

    Args:
        spec: Model specification string.

    Returns:
        Tuple of (provider, model_name).
    """
    if ":" in spec:
        parts = spec.split(":", 1)
        provider = parts[0].lower()
        model_name = parts[1]
    else:
        # Default to Azure for backward compatibility
        provider = "azure"
        model_name = spec

    # Validate provider
    valid_providers = ["azure", "gemini", "claude", "vertex"]
    if provider not in valid_providers:
        raise ValueError(
            f"Unknown provider: {provider}. Valid providers: {valid_providers}"
        )

    return provider, model_name


def is_thinking_model(model_name: str) -> bool:
    """Check if model supports extended thinking.

    Args:
        model_name: Model name to check.

    Returns:
        True if model is a thinking model.
    """
    for pattern in THINKING_MODEL_PATTERNS:
        if re.search(pattern, model_name, re.IGNORECASE):
            return True
    return False


def create_client(
    model_spec: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs,
) -> LLMClientBase:
    """Create an LLM client from model specification.

    Args:
        model_spec: Model specification (e.g., "azure:gpt-4o", "claude:claude-sonnet-4-5").
        max_retries: Maximum retry attempts.
        retry_delay: Base delay between retries.
        **kwargs: Additional provider-specific options.

    Returns:
        Configured LLM client instance.

    Raises:
        ValueError: If provider is unknown or configuration is invalid.
    """
    provider, model_name = parse_model_spec(model_spec)

    # Validate configuration
    is_valid, error = LLMConfig.validate_provider_config(provider)
    if not is_valid:
        raise ValueError(f"Configuration error for {provider}: {error}")

    # Get provider configuration
    provider_config = LLMConfig.get_provider_config(provider)

    # Build ModelConfig
    # For vertex, pass project_id and region via extra
    extra = dict(kwargs)
    if provider == "vertex":
        extra.setdefault("project_id", provider_config.get("project_id"))
        extra.setdefault("region", provider_config.get("region", "us-central1"))

    config = ModelConfig(
        provider=provider,
        model_name=model_name,
        is_thinking=is_thinking_model(model_name),
        api_key=provider_config.get("api_key"),
        endpoint=provider_config.get("endpoint"),
        proxy=provider_config.get("proxy"),
        max_retries=max_retries,
        retry_delay=retry_delay,
        extra=extra,
    )

    # Create provider-specific client
    if provider == "azure":
        from .azure_openai import AzureOpenAIClient

        return AzureOpenAIClient(config)
    elif provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(config)
    elif provider == "claude":
        from .claude import ClaudeClient

        return ClaudeClient(config)
    elif provider == "vertex":
        from .vertex import VertexGeminiClient

        return VertexGeminiClient(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_available_models() -> dict[str, list[str]]:
    """List commonly used models per provider.

    Returns:
        Dictionary mapping provider names to model lists.
    """
    return {
        "azure": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "gemini": [
            "gemini-3-flash",       # alias → gemini-3-flash-preview (official)
            "gemini-3-pro",         # alias → gemini-3-pro-preview (official)
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "vertex": [
            "gemini-2.0-flash",     # recommended: no thinking overhead, reliable
            "gemini-2.0-flash-001",
            "gemini-2.5-flash",     # thinking model, needs max_tokens >= 2048
            "gemini-2.5-pro",       # highest quality, thinking model
        ],
        "claude": [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-thinking",
            "claude-opus-4-5",
            "claude-opus-4-5-thinking",
        ],
    }
