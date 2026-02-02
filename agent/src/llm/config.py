"""
LLM Configuration System

Loads configuration from environment variables and optional YAML config files.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()


class LLMConfig:
    """Central configuration for LLM providers."""

    # Default proxy for providers that need it
    DEFAULT_PROXY = "http://127.0.0.1:7890"

    # Provider-specific environment variable mappings
    ENV_MAPPINGS = {
        "azure": {
            "endpoint": ["AZURE_OPENAI_ENDPOINT", "azure_endpoint"],
            "api_key": ["AZURE_OPENAI_API_KEY", "api_key"],
        },
        "gemini": {
            "api_key": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "proxy": ["GEMINI_PROXY", "HTTPS_PROXY"],
        },
        "claude": {
            "api_key": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "proxy": ["CLAUDE_PROXY", "HTTPS_PROXY"],
        },
    }

    # Default models per provider
    DEFAULT_MODELS = {
        "azure": "gpt-4o",
        "gemini": "gemini-2.5-flash",
        "claude": "claude-sonnet-4-5",
    }

    @classmethod
    def get_env(cls, keys: list[str], default: Optional[str] = None) -> Optional[str]:
        """Get first available environment variable from list of keys."""
        for key in keys:
            value = os.getenv(key)
            if value:
                return value
        return default

    @classmethod
    def get_azure_config(cls) -> dict:
        """Get Azure OpenAI configuration."""
        mapping = cls.ENV_MAPPINGS["azure"]
        return {
            "endpoint": cls.get_env(mapping["endpoint"]),
            "api_key": cls.get_env(mapping["api_key"]),
        }

    @classmethod
    def get_gemini_config(cls) -> dict:
        """Get Gemini configuration."""
        mapping = cls.ENV_MAPPINGS["gemini"]
        return {
            "api_key": cls.get_env(mapping["api_key"]),
            "proxy": cls.get_env(mapping["proxy"], cls.DEFAULT_PROXY),
        }

    @classmethod
    def get_claude_config(cls) -> dict:
        """Get Claude configuration."""
        mapping = cls.ENV_MAPPINGS["claude"]
        return {
            "api_key": cls.get_env(mapping["api_key"]),
            "proxy": cls.get_env(mapping["proxy"], cls.DEFAULT_PROXY),
        }

    @classmethod
    def get_provider_config(cls, provider: str) -> dict:
        """Get configuration for a specific provider.

        Args:
            provider: Provider name (azure, gemini, claude).

        Returns:
            Configuration dictionary for the provider.

        Raises:
            ValueError: If provider is unknown.
        """
        provider = provider.lower()
        if provider == "azure":
            return cls.get_azure_config()
        elif provider == "gemini":
            return cls.get_gemini_config()
        elif provider == "claude":
            return cls.get_claude_config()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @classmethod
    def validate_provider_config(cls, provider: str) -> tuple[bool, str]:
        """Validate that required configuration exists for a provider.

        Args:
            provider: Provider name.

        Returns:
            Tuple of (is_valid, error_message).
        """
        config = cls.get_provider_config(provider)

        if provider == "azure":
            if not config.get("endpoint"):
                return False, "AZURE_OPENAI_ENDPOINT not set"
            if not config.get("api_key"):
                return False, "AZURE_OPENAI_API_KEY not set"
        elif provider == "gemini":
            if not config.get("api_key"):
                return False, "GEMINI_API_KEY not set"
        elif provider == "claude":
            if not config.get("api_key"):
                return False, "ANTHROPIC_API_KEY not set"

        return True, ""

    @classmethod
    def get_default_model(cls, provider: str) -> str:
        """Get default model for a provider."""
        return cls.DEFAULT_MODELS.get(provider.lower(), "gpt-4o")


def load_yaml_config(config_path: Optional[str] = None) -> dict:
    """Load optional YAML configuration file.

    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml
                    in the project root.

    Returns:
        Configuration dictionary (empty if file not found).
    """
    if config_path:
        path = Path(config_path)
    else:
        # Look for config.yaml in common locations
        candidates = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]
        path = next((p for p in candidates if p.exists()), None)

    if not path or not path.exists():
        return {}

    try:
        import yaml

        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {path}")
        return config
    except ImportError:
        logger.warning("PyYAML not installed, skipping YAML config")
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {path}: {e}")
        return {}
