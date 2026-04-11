"""Fixed-size rollout buffer for MAPPO trajectories.

Stores per-step transitions for one agent.  After ``update_interval`` steps
the buffer is consumed by ``mappo.ppo_update`` and then cleared.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single environment transition."""
    prompt_text: str          # full formatted prompt (needed for recomputing logprobs)
    action_idx: int           # index into RLConfig.actions
    old_log_prob: float       # log π_old(a|s)
    old_value: float          # V_old(s)
    reward: float = 0.0
    done: bool = False
    # populated by GAE after the rollout segment
    advantage: float = 0.0
    returns: float = 0.0


class RolloutBuffer:
    """Collects transitions for one agent and computes GAE when flushed."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._buf: List[Transition] = []
        self._pending: Optional[Transition] = None  # waiting for reward

    # ── Collection API ──

    def store_action(self, prompt_text: str, action_idx: int,
                     log_prob: float, value: float) -> None:
        """Called right after action selection.  Reward comes later."""
        if self._pending is not None:
            # previous transition never got a reward – store it with 0
            self._buf.append(self._pending)
        self._pending = Transition(
            prompt_text=prompt_text,
            action_idx=action_idx,
            old_log_prob=log_prob,
            old_value=value,
        )

    def store_reward(self, reward: float, done: bool = False) -> None:
        """Called after the environment returns the reward for the last action."""
        if self._pending is None:
            logger.warning(
                "store_reward() called with no pending transition "
                "(timing bug or macro mid-execution). reward=%.4f discarded.", reward
            )
            return
        # Clamp invalid reward values to prevent NaN/inf corruption
        if not isinstance(reward, (int, float)) or reward != reward:  # NaN check
            logger.warning("Invalid reward %s, clamping to 0.0", reward)
            reward = 0.0
        reward = max(-1e6, min(1e6, reward))
        self._pending.reward = reward
        self._pending.done = done
        self._buf.append(self._pending)
        self._pending = None

    # ── Query ──

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) > 0

    # ── GAE computation ──

    def compute_gae(self, gamma: float, gae_lambda: float,
                    last_value: float = 0.0) -> None:
        """Compute Generalised Advantage Estimation in-place.

        After GAE is computed, advantages are normalized over the *full rollout*
        so that mini-batch assignment does not affect advantage scaling.  Per-
        mini-batch normalization (the previous approach) destroyed the signal
        about which parts of the rollout were actually better than others.
        """
        gae = 0.0
        for t in reversed(range(len(self._buf))):
            tr = self._buf[t]
            if t == len(self._buf) - 1:
                next_value = last_value
            else:
                next_value = self._buf[t + 1].old_value
            # Use the CURRENT transition's done flag, not the next step's.
            # tr.done=True means the episode ended after this step, so V(s_{t+1})=0.
            next_non_terminal = 1.0 - float(tr.done)
            delta = tr.reward + gamma * next_value * next_non_terminal - tr.old_value
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            tr.advantage = gae
            tr.returns = gae + tr.old_value

        # Normalize advantages over the full rollout (not per mini-batch).
        adv = torch.tensor([tr.advantage for tr in self._buf], dtype=torch.float32)
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            for i, tr in enumerate(self._buf):
                tr.advantage = adv[i].item()

    # ── Batching for training ──

    def sample_batches(self, mini_batch_size: int,
                       extra_transitions: Optional[List["Transition"]] = None):
        """Yield mini-batches of transitions (shuffled).

        Parameters
        ----------
        mini_batch_size : int
        extra_transitions : list of Transition, optional
            Social-replay transitions from neighbour buffers (Eq. 7).
            Appended to the pool before shuffling so they are distributed
            across all mini-batches proportionally.
        """
        assert mini_batch_size > 0, "mini_batch_size must be positive"
        pool = self._buf + (extra_transitions or [])
        indices = torch.randperm(len(pool)).tolist()
        for start in range(0, len(pool), mini_batch_size):
            batch_idx = indices[start:start + mini_batch_size]
            yield [pool[i] for i in batch_idx]

    def get_all(self) -> List[Transition]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()
        self._pending = None

    def filter_by_keyword(self, keyword: str) -> List[Transition]:
        """Return transitions whose prompt contains ``keyword`` (for token-opt filtering)."""
        return [t for t in self._buf if keyword.lower() in t.prompt_text.lower()]
