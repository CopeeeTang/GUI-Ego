"""Proactive AR Smart Glasses Agent.

An autonomous agent that analyzes ego-centric video clips frame-by-frame
and proactively suggests interventions using a tiered memory system.
"""

from .agent import ProactiveARAgent
from .config import ARAgentConfig

__all__ = ["ProactiveARAgent", "ARAgentConfig"]
