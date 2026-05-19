"""Hebbian Social Plasticity — ported from the WiredTogether parent repo.

Numpy-only implementation of a reward-modulated Hebbian update over an
N×N social graph. See HEBBIAN_MARL_PLAN.md §1 / §2 / §10 for the
mathematical content and the role this module plays in the ablation grid.
"""

from .config import HebbianConfig
from .graph import HebbianSocialGraph
from .runtime import get_graph, set_graph, clear_graph

__all__ = [
    "HebbianConfig",
    "HebbianSocialGraph",
    "get_graph",
    "set_graph",
    "clear_graph",
]
