"""RL layer for MindForge agents.

Centralised-critic MAPPO (default) and per-agent IPPO (legacy fallback). Toggle
the whole layer with ``RLConfig(enabled=True/False)``. Hebbian social plasticity
lives in the ``hebbian`` package — it is independent of this one and is
re-exported here for backward-compatible imports.
"""

from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer
from rl_layer.centralized_critic import CentralizedCritic

# Re-export Hebbian for callers that used to import everything from rl_layer.
from hebbian.config import HebbianConfig
from hebbian.graph import HebbianSocialGraph

__all__ = [
    "RLConfig", "RLLayer", "CentralizedCritic",
    "HebbianConfig", "HebbianSocialGraph",
]
