"""Hebbian social-plasticity graph for multi-agent RL.

Maintains a learned bond matrix W between agents and uses it to:
- diffuse rewards along strong bonds,
- weight social-replay sampling during PPO updates,
- shape modulator signals for spatial co-activity and communication.

Independent of the RL/PPO machinery and the LLM agent stack — keep it that way.
"""

from hebbian.config import HebbianConfig
from hebbian.graph import HebbianSocialGraph

__all__ = ["HebbianConfig", "HebbianSocialGraph"]
