"""Prompt strategy module for multi-version prompt management."""

from .base import PromptStrategy
from .v1_baseline import BaselinePromptStrategy
from .v2_google_gui import GoogleGUIPromptStrategy
from .v3_with_visual import VisualPromptStrategy

__all__ = [
    "PromptStrategy",
    "BaselinePromptStrategy",
    "GoogleGUIPromptStrategy",
    "VisualPromptStrategy",
]
