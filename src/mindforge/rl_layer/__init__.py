"""Modular RL layer for MindForge agents.

Toggle with ``RLConfig(enabled=True/False)`` — zero impact when disabled.
"""

from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer

__all__ = ["RLConfig", "RLLayer"]
