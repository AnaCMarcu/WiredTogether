"""Modular RL layer for MindForge agents.

Toggle with ``RLConfig(enabled=True/False)`` — zero impact when disabled.
Toggle Hebbian social plasticity with ``HebbianConfig(enabled=True/False)``.
"""

from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer
from rl_layer.hebbian_config import HebbianConfig
from rl_layer.hebbian_graph import HebbianSocialGraph

__all__ = ["RLConfig", "RLLayer", "HebbianConfig", "HebbianSocialGraph"]
