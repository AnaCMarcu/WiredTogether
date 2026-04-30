"""Fixed-size rollout buffer for MAPPO trajectories.

Stores per-step transitions for one agent.  After ``update_interval`` steps
the buffer is consumed by ``mappo.ppo_update`` and then cleared.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single environment transition."""
    prompt_text: str          # full formatted prompt (needed for recomputing logprobs)
    action_idx: int           # index into RLConfig.actions
    old_log_prob: float       # log π_old(a|s)
    old_value: float          # V_old(s) — per-agent value head (IPPO baseline)
    reward: float = 0.0
    done: bool = False
    # populated by GAE after the rollout segment
    advantage: float = 0.0
    returns: float = 0.0
    # reward decomposition for post-hoc analysis (training uses reward only)
    reward_task: float = 0.0
    reward_comm: float = 0.0
    # ── Centralised-critic (MAPPO) fields, populated by main loop ──
    # When set, GAE uses old_value_global instead of old_value, and
    # action_level_ppo_step skips its value-loss term (the centralised
    # critic owns its own update pipeline).
    old_value_global: Optional[float] = None
    joint_state: Optional[np.ndarray] = None


class RolloutBuffer:
    """Collects transitions for one agent and computes GAE when flushed."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self._buf: List[Transition] = []
        self._pending: Optional[Transition] = None  # waiting for reward

    # ── Collection API ──

    def store_action(self, prompt_text: str, action_idx: int,
                     log_prob: float, value: float,
                     value_global: Optional[float] = None,
                     joint_state: Optional[np.ndarray] = None) -> None:
        """Called right after action selection.  Reward comes later.

        ``value_global`` and ``joint_state`` are populated when running with a
        centralised critic; otherwise the per-agent ``value`` is used as the
        GAE baseline (IPPO behaviour).
        """
        if self._pending is not None:
            # previous transition never got a reward – store it with 0
            self._buf.append(self._pending)
        self._pending = Transition(
            prompt_text=prompt_text,
            action_idx=action_idx,
            old_log_prob=log_prob,
            old_value=value,
            old_value_global=value_global,
            joint_state=joint_state,
        )

    def set_pending_value_global(self, value_global: float,
                                 joint_state: Optional[np.ndarray] = None) -> None:
        """Attach a centralised critic value (and the joint state it was computed
        from) to the currently-pending transition.

        Called once per step from the main loop AFTER all agents' select_action
        have run, so V_global is identical across agents at the same step.
        No-op if no transition is pending (e.g. agent terminated mid-step).
        """
        if self._pending is None:
            return
        self._pending.old_value_global = float(value_global)
        if joint_state is not None:
            self._pending.joint_state = joint_state

    def store_reward(self, reward: float, done: bool = False,
                     reward_task: float = 0.0, reward_comm: float = 0.0) -> None:
        """Called after the environment returns the reward for the last action."""
        if self._pending is None:
            logger.warning(
                "store_reward() called with no pending transition "
                "(timing bug or macro mid-execution). reward=%.4f discarded.", reward
            )
            return
        # Clamp invalid reward values to prevent NaN/inf corruption
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            logger.warning("Invalid reward %s, clamping to 0.0", reward)
            reward = 0.0
        if math.isnan(reward) or math.isinf(reward):
            logger.warning("Non-finite reward %s, clamping to 0.0", reward)
            reward = 0.0
        reward = max(-1e6, min(1e6, reward))
        self._pending.reward = reward
        self._pending.done = done
        self._pending.reward_task = reward_task
        self._pending.reward_comm = reward_comm
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
                    last_value: float = 0.0,
                    use_global_value: bool = False) -> None:
        """Compute Generalised Advantage Estimation in-place.

        When ``use_global_value`` is True, GAE uses each transition's
        ``old_value_global`` (centralised critic baseline) instead of the
        per-agent ``old_value``. Falls back to ``old_value`` for any
        transition where the global value is missing.

        After GAE is computed, advantages are normalised over the *full rollout*
        so mini-batch assignment does not affect advantage scaling.
        """
        def _v(tr: Transition) -> float:
            if use_global_value and tr.old_value_global is not None:
                return tr.old_value_global
            return tr.old_value

        gae = 0.0
        for t in reversed(range(len(self._buf))):
            tr = self._buf[t]
            if t == len(self._buf) - 1:
                next_value = last_value
            else:
                next_value = _v(self._buf[t + 1])
            # tr.done=True means the episode ended after this step → V(s_{t+1})=0.
            next_non_terminal = 1.0 - float(tr.done)
            cur_value = _v(tr)
            delta = tr.reward + gamma * next_value * next_non_terminal - cur_value
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            tr.advantage = gae
            tr.returns = gae + cur_value

        # Normalise advantages over the full rollout (not per mini-batch).
        adv = torch.tensor([tr.advantage for tr in self._buf], dtype=torch.float32)
        if adv.numel() >= 1:
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
