"""Centralized critic for MAPPO.

Replaces per-agent value heads with a single shared V(joint_state). Joint state is
a two-stream vector:

  Stream A — compact features per agent:
    position (xyz), chamber one-hot, hp, inventory bag-of-words, milestone bitmap,
    raw step reward.
  Stream B — semantic features per agent:
    sentence-transformer embedding of the agent's last action + last comm.

The critic is a small MLP. It carries its own buffer of (joint_state, team_reward)
tuples and runs an MSE update once per round, independent of any agent's update.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_layer.config import RLConfig

logger = logging.getLogger(__name__)


# Five-chambers chamber → one-hot index. Matches CHAMBER_BOUNDS in
# cooperation_metric.py / communication_rewards.py.
_CHAMBERS = ("ch1", "ch2", "ch3", "ch4", "ch5")
_CHAMBER_INDEX = {c: i for i, c in enumerate(_CHAMBERS)}

# Inventory items we one-hot encode (substring match against the env's
# pickedup_object string). Captures the few items that drive milestones.
_INVENTORY_ITEMS = (
    "diamond_sword",
    "diamond_chestplate",
    "tree",
    "stone",
    "wood",
    "log",
    "cobble",
    "dirt",
)


def _per_agent_compact_dim(num_milestones: int) -> int:
    # 3 (xyz) + 5 chambers + 1 hp + 8 inv items + N milestones + 1 reward
    return 3 + len(_CHAMBERS) + 1 + len(_INVENTORY_ITEMS) + num_milestones + 1


class _CompactEncoder:
    """Build the compact (Stream A) part of the joint feature vector."""

    def __init__(self, num_agents: int, milestone_ids: Sequence[str]):
        self.num_agents = num_agents
        self.milestone_ids = list(milestone_ids)
        self._per_agent_dim = _per_agent_compact_dim(len(self.milestone_ids))
        self.dim = num_agents * self._per_agent_dim

    def encode(
        self,
        *,
        positions: Dict[int, Optional[Sequence[float]]],
        chambers: Dict[int, Optional[str]],
        hps: Dict[int, Optional[float]],
        inventories: Dict[int, Optional[str]],
        milestones_by_agent: Dict[str, set],
        raw_rewards: Dict[int, float],
    ) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.num_agents):
            base = i * self._per_agent_dim
            cursor = base

            # position (xyz) — leave as-is; the world is small (~50 blocks span).
            pos = positions.get(i)
            if pos is not None:
                for k in range(3):
                    if k < len(pos):
                        out[cursor + k] = float(pos[k])
            cursor += 3

            # chamber one-hot
            ch_idx = _CHAMBER_INDEX.get(chambers.get(i)) if chambers else None
            if ch_idx is not None:
                out[cursor + ch_idx] = 1.0
            cursor += len(_CHAMBERS)

            # hp normalised to [0, 1]
            hp = hps.get(i)
            out[cursor] = float(hp) / 20.0 if hp is not None else 1.0
            cursor += 1

            # inventory bag-of-words (substring match)
            inv_str = (inventories.get(i) or "").lower()
            for k, item in enumerate(_INVENTORY_ITEMS):
                if item in inv_str:
                    out[cursor + k] = 1.0
            cursor += len(_INVENTORY_ITEMS)

            # milestone bitmap
            mset = milestones_by_agent.get(f"agent_{i}", set()) if milestones_by_agent else set()
            for k, mid in enumerate(self.milestone_ids):
                if mid in mset:
                    out[cursor + k] = 1.0
            cursor += len(self.milestone_ids)

            # raw step reward (pre-diffusion) — clipped to a reasonable range
            r = float(raw_rewards.get(i, 0.0)) if raw_rewards else 0.0
            out[cursor] = max(-100.0, min(100.0, r))
            cursor += 1
        return out


class _CriticBuffer:
    """Per-step buffer of (joint_state, team_reward, V_t, done) for critic updates."""

    def __init__(self):
        self.joint_states: List[np.ndarray] = []
        self.team_rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def __len__(self) -> int:
        return len(self.joint_states)

    def store(self, joint_state: np.ndarray, team_reward: float,
              value: float, done: bool) -> None:
        self.joint_states.append(joint_state)
        self.team_rewards.append(float(team_reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def compute_returns(self, gamma: float, gae_lambda: float,
                        last_value: float = 0.0) -> List[float]:
        """GAE on the team-reward sequence; returns list aligned with stored steps."""
        gae = 0.0
        n = len(self.joint_states)
        returns = [0.0] * n
        for t in reversed(range(n)):
            next_value = self.values[t + 1] if t + 1 < n else last_value
            non_term = 1.0 - float(self.dones[t])
            delta = self.team_rewards[t] + gamma * next_value * non_term - self.values[t]
            gae = delta + gamma * gae_lambda * non_term * gae
            returns[t] = gae + self.values[t]
        return returns

    def clear(self) -> None:
        self.joint_states.clear()
        self.team_rewards.clear()
        self.values.clear()
        self.dones.clear()


class _CriticNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralizedCritic:
    """Single shared V(joint_state) used by all agents in MAPPO."""

    def __init__(
        self,
        *,
        num_agents: int,
        config: RLConfig,
        milestone_ids: Sequence[str],
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.num_agents = num_agents
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Sentence-transformer for last-action+comm embeddings.
        # Same ST_MODEL_NAME used by ChromaDB elsewhere — load shares cache.
        from sentence_transformers import SentenceTransformer
        from agent_modules.util import ST_MODEL_NAME
        self._sentence_model = SentenceTransformer(
            ST_MODEL_NAME, device=str(self._device)
        )
        self._semantic_dim = int(self._sentence_model.get_sentence_embedding_dimension())

        self._compact = _CompactEncoder(num_agents, milestone_ids)
        self._joint_dim = self._compact.dim + num_agents * self._semantic_dim

        self.net = _CriticNet(self._joint_dim, config.critic_hidden).to(self._device).float()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.critic_lr)
        self._buffer = _CriticBuffer()
        self._update_count = 0

        logger.info(
            "CentralizedCritic: in_dim=%d (compact=%d, semantic=%d × %d), hidden=%d, lr=%.0e",
            self._joint_dim, self._compact.dim, num_agents,
            self._semantic_dim, config.critic_hidden, config.critic_lr,
        )

    @property
    def joint_dim(self) -> int:
        return self._joint_dim

    @property
    def update_count(self) -> int:
        return self._update_count

    # ─── Encoding ─────────────────────────────────────────────────────

    def encode_joint(
        self,
        *,
        positions: Dict[int, Optional[Sequence[float]]],
        chambers: Dict[int, Optional[str]],
        hps: Dict[int, Optional[float]],
        inventories: Dict[int, Optional[str]],
        milestones_by_agent: Dict[str, set],
        raw_rewards: Dict[int, float],
        last_actions: Dict[int, str],
        last_comms: Dict[int, str],
    ) -> np.ndarray:
        """Build the full joint-state vector (compact + semantic) for one step."""
        compact = self._compact.encode(
            positions=positions,
            chambers=chambers,
            hps=hps,
            inventories=inventories,
            milestones_by_agent=milestones_by_agent,
            raw_rewards=raw_rewards,
        )
        sentences = [
            (f"{(last_actions.get(i) or 'NoOp')}. "
             f"{(last_comms.get(i) or '')}").strip()
            for i in range(self.num_agents)
        ]
        with torch.no_grad():
            sem = self._sentence_model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )  # (num_agents, semantic_dim)
        sem_flat = np.asarray(sem, dtype=np.float32).reshape(-1)
        return np.concatenate([compact, sem_flat], dtype=np.float32)

    # ─── Inline value evaluation (no grad) ────────────────────────────

    def evaluate(self, joint_state: np.ndarray) -> float:
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(joint_state).float().to(self._device).unsqueeze(0)
            return float(self.net(x).item())

    # ─── Buffer + update ──────────────────────────────────────────────

    def store_step(
        self,
        joint_state: np.ndarray,
        team_reward: float,
        value: float,
        done: bool,
    ) -> None:
        self._buffer.store(joint_state, team_reward, value, done)

    def __len__(self) -> int:
        return len(self._buffer)

    def should_update(self) -> bool:
        return len(self._buffer) >= self.config.update_interval

    def update(self, last_value: float = 0.0) -> dict:
        """Run one round of MSE updates against GAE-team-returns."""
        if len(self._buffer) == 0:
            return {}

        returns = self._buffer.compute_returns(
            self.config.gamma, self.config.gae_lambda, last_value
        )
        joint = torch.from_numpy(
            np.stack(self._buffer.joint_states)
        ).float().to(self._device)
        ret = torch.tensor(returns, dtype=torch.float32, device=self._device)
        old_v = torch.tensor(
            self._buffer.values, dtype=torch.float32, device=self._device
        )

        self.net.train()
        info = {}
        n = joint.size(0)
        for _ in range(self.config.ppo_epochs):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, self.config.mini_batch_size):
                idx = perm[start:start + self.config.mini_batch_size]
                if idx.numel() == 0:
                    continue
                v_pred = self.net(joint[idx])
                v_clipped = old_v[idx] + torch.clamp(
                    v_pred - old_v[idx],
                    -self.config.value_clip_eps,
                    self.config.value_clip_eps,
                )
                l1 = F.mse_loss(v_pred, ret[idx], reduction="none")
                l2 = F.mse_loss(v_clipped, ret[idx], reduction="none")
                loss = torch.min(l1, l2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(),
                                         self.config.max_grad_norm)
                self.optimizer.step()
                info["critic_loss"] = float(loss.item())

        self._update_count += 1
        info["critic_update_count"] = self._update_count
        info["critic_buffer_size"] = n
        info["critic_returns_mean"] = float(ret.mean().item())
        info["critic_returns_std"] = float(ret.std().item())
        self._buffer.clear()
        return info

    # ─── Persistence ──────────────────────────────────────────────────

    def save(self, path: "os.PathLike | str") -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.net.state_dict(), f"{path}/critic_net.pt")
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "update_count": self._update_count,
        }, f"{path}/critic_state.pt")

    def load(self, path: "os.PathLike | str") -> None:
        import os
        net_path = f"{path}/critic_net.pt"
        state_path = f"{path}/critic_state.pt"
        if os.path.exists(net_path):
            self.net.load_state_dict(
                torch.load(net_path, map_location=self._device, weights_only=True)
            )
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self._device, weights_only=False)
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError):
                logger.warning("CentralizedCritic: optimizer state mismatch, reinitialising.")
            self._update_count = int(state.get("update_count", 0))
