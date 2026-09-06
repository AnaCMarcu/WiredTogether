"""Environment-side accounting used by the episode loop.

Two groups. The Lua interface — ``lua_events`` (milestone / anvil / death event
drains) and ``chamber_state`` (the live puzzle-state block the prompt shows) —
is mixed into the environment adapter. The rest observes or shapes rewards from
the env contract: message validity and payment, pair-level cooperation stats,
per-episode JSONL. None of it depends on the LLM stack or the RL layer.
"""

from mindforge.env.communication_rewards import CommunicationTracker
from mindforge.env.cooperation_metric import CooperationMetric
from mindforge.env.episode_logger import EpisodeLogger

__all__ = ["CommunicationTracker", "CooperationMetric", "EpisodeLogger"]
