"""Hebbian Social Plasticity graph — the core thesis contribution.

Implements a reward-modulated Hebbian update rule over a social graph
W(t) ∈ [0,1]^{N×N}.  Each agent is a neuron, each weighted edge is a
synapse, and the team advantage signal acts as a dopaminergic modulator.

This module is **numpy-only** — zero torch dependency — so it works
regardless of whether the RL layer is enabled.

When ``HebbianConfig(enabled=False)``, every public method is a fast
no-op returning sensible defaults.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from hebbian.config import HebbianConfig

logger = logging.getLogger(__name__)

_EPS = 1e-8  # numerical guard for division-by-zero


def _sanitize_reward(value: float) -> float:
    """Clamp invalid reward values to prevent NaN/inf corruption."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid reward %s, clamping to 0.0", value)
        return 0.0
    if math.isnan(v):
        logger.warning("NaN reward, clamping to 0.0")
        return 0.0
    if math.isinf(v):
        logger.warning("Infinite reward %s, clamping to 0.0", value)
        return 0.0
    return max(-1e6, min(1e6, v))


class HebbianSocialGraph:
    """Adaptive social graph modelling inter-agent bonds.

    Implements the neuron-agent isomorphism described in the thesis:
    each agent is a neuron, each bond wij is a synapse, and the team
    reward signal modulates Hebbian plasticity (LTP/LTD).

    Parameters
    ----------
    config : HebbianConfig
        Full configuration (see hebbian_config.py).
    agent_roles : list[int], optional
        Role index per agent (0=gatherer, 1=hunter, 2=defender).
        Used for the modularity_proxy metric.  May be ``None``.
    """

    def __init__(self, config: HebbianConfig, agent_roles: Optional[List[int]] = None):
        self.config = config
        self._agent_roles = agent_roles
        self._step_count = 0

        # Always initialise — avoids AttributeError when disabled
        self._last_coactivity: Optional[np.ndarray] = None

        if not config.enabled:
            return

        N = config.num_agents

        # Social graph W(t) ∈ [0,1]^{N×N}, diagonal always 0
        self.W = np.full((N, N), config.init_weight, dtype=np.float32)
        np.fill_diagonal(self.W, 0.0)

        # Sustained LTD state
        self._failure_coactivation = np.zeros((N, N), dtype=np.float32)
        self._failure_window_buffer: deque = deque(
            maxlen=config.failure_memory_window,
        )

        # Running max of |ri(t)| for engagement normalisation
        self._max_reward_seen: float = 0.0

    # ──────────────────────────────────────────────
    # 2.2  Co-activity signal (refined Eq. 2)
    # ──────────────────────────────────────────────

    def _compute_coactivity(
        self,
        positions: List[Optional[Tuple[float, float, float]]],
        step_rewards: List[float],
        comm_events: Optional[List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Compute the co-activity matrix cij(t).

        Implements refined Eq. 2 from the thesis proposal:
        soft engagement score gi(t), spatial gate, and communication
        co-activity bonus for agents that communicate across distance.

        Parameters
        ----------
        positions : list of (x, y, z) or None per agent
        step_rewards : list of floats, one per agent
        comm_events : list of (sender_idx, receiver_idx) pairs, or None

        Returns
        -------
        np.ndarray shape (N, N), dtype float32, values in [0, 1]
        """
        N = self.config.num_agents
        cfg = self.config

        # ── Update running max reward ──
        for r in step_rewards:
            abs_r = abs(_sanitize_reward(r))
            if abs_r > self._max_reward_seen:
                self._max_reward_seen = abs_r

        # ── Soft engagement score gi(t) ──
        # gi(t) = clip(α · |ri(t)| / (max_reward_seen + ε) + (1-α) · comm_i(t), 0, 1)
        comm_agents = set()
        if comm_events:
            for sender, receiver in comm_events:
                comm_agents.add(sender)
                comm_agents.add(receiver)

        engagement = np.zeros(N, dtype=np.float32)
        for i in range(N):
            reward_component = cfg.engagement_reward_weight * (
                abs(_sanitize_reward(step_rewards[i]))
                / (self._max_reward_seen + _EPS)
            )
            comm_component = (1.0 - cfg.engagement_reward_weight) * (
                1.0 if i in comm_agents else 0.0
            )
            engagement[i] = np.clip(reward_component + comm_component, 0.0, 1.0)

        # ── Spatial gate: I[||pi - pj|| ≤ d] ──
        spatial_gate = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            if positions[i] is None:
                continue
            pi = np.array(positions[i], dtype=np.float32)
            for j in range(i + 1, N):
                if positions[j] is None:
                    continue
                pj = np.array(positions[j], dtype=np.float32)
                dist = np.linalg.norm(pi - pj)
                if dist <= cfg.interaction_radius:
                    spatial_gate[i, j] = 1.0
                    spatial_gate[j, i] = 1.0

        # ── Spatial co-activity ──
        # cij_spatial(t) = I[dist ≤ d] · gi(t) · gj(t)
        cij_spatial = spatial_gate * np.outer(engagement, engagement)

        # ── Communication co-activity bonus ──
        # cij_comm(t) = δ_comm · comm_pair_ij(t) · (1 - I[dist ≤ d])
        cij_comm = np.zeros((N, N), dtype=np.float32)
        if comm_events and cfg.communication_coactivity_bonus > 0.0:
            comm_pair_set = set()
            for sender, receiver in comm_events:
                comm_pair_set.add((sender, receiver))
                comm_pair_set.add((receiver, sender))  # symmetric
            for i, j in comm_pair_set:
                if i != j:
                    # Only fire when NOT already spatially co-active
                    cij_comm[i, j] = cfg.communication_coactivity_bonus * (
                        1.0 - spatial_gate[i, j]
                    )

        # ── Final co-activity ──
        cij = np.clip(cij_spatial + cij_comm, 0.0, 1.0)
        np.fill_diagonal(cij, 0.0)
        return cij

    # ──────────────────────────────────────────────
    # 2.3  Modulatory signal — asymmetric LTP/LTD (refined Eq. 4+5)
    # ──────────────────────────────────────────────

    def _compute_modulator(
        self,
        advantages: Optional[List[Optional[float]]],
        step_rewards: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compute the per-pair modulatory signal matrix m(t) ∈ R^{N×N}.

        Each entry m_ij uses only the advantage signals of agents i and j,
        not the team mean.  This gives proper credit assignment: the bond
        between i and j only strengthens when *both* involved agents had a
        positive outcome, not because a third agent happened to succeed.

        Implements the asymmetric LTP/LTD rule (Eq. 4+5) per pair:
            At_ij  = (A_i + A_j) / 2
            m_ltp  = ltp_lr  * tanh(β * max(At_ij,  0))
            m_ltd  = ltd_lr  * tanh(β * max(-At_ij, 0))
            m_ij   = m_ltp - m_ltd

        Parameters
        ----------
        advantages : list of per-agent one-step advantages (δ = r - V(s)),
            or None per agent when RL layer is disabled for that agent.
            When the whole list is None, falls back to normalised rewards.
        step_rewards : per-agent rewards — fallback for agents without RL.

        Returns
        -------
        np.ndarray shape (N, N) : per-pair modulator matrix, diagonal = 0
        """
        N = self.config.num_agents
        cfg = self.config

        # Build per-agent signal: prefer advantage, fall back to reward/max
        agent_signals = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if advantages is not None and i < len(advantages) and advantages[i] is not None:
                agent_signals[i] = _sanitize_reward(advantages[i])
            elif step_rewards is not None and i < len(step_rewards):
                agent_signals[i] = (
                    _sanitize_reward(step_rewards[i]) / (self._max_reward_seen + _EPS)
                )

        # Per-pair modulator matrix
        m = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                At_ij = (agent_signals[i] + agent_signals[j]) / 2.0
                m_ltp = cfg.ltp_lr * math.tanh(cfg.modulation_beta * max(At_ij, 0.0))
                # LTD only fires when the average advantage is clearly negative
                # (beyond ltd_threshold). This prevents zero-reward exploration
                # steps (advantage ≈ −V(s), small negative) from continuously
                # depressing bonds that were never actively failing.
                m_ltd = cfg.ltd_lr * math.tanh(
                    cfg.modulation_beta * max(-At_ij - cfg.ltd_threshold, 0.0)
                )
                m[i, j] = m_ltp - m_ltd

        return m

    # ──────────────────────────────────────────────
    # 2.4  Sustained LTD from repeated co-failure
    # ──────────────────────────────────────────────

    def _update_failure_window(
        self, cij: np.ndarray, At: float
    ) -> None:
        """Update the sliding window of failure co-activity snapshots.

        Called each step before computing sustained LTD so the history
        is current.
        """
        if At < 0.0:
            snapshot = (cij > 0.0).astype(np.float32)
            # If deque is full, subtract the oldest snapshot
            if len(self._failure_window_buffer) == self._failure_window_buffer.maxlen:
                oldest = self._failure_window_buffer[0]
                self._failure_coactivation -= oldest
            self._failure_window_buffer.append(snapshot)
            self._failure_coactivation += snapshot
        # Ensure non-negative (floating-point drift guard)
        np.clip(self._failure_coactivation, 0.0, None, out=self._failure_coactivation)

    def _compute_sustained_ltd(self) -> np.ndarray:
        """Compute sustained LTD term from repeated co-failure.

        Implements the sustained depression mechanism:
        Fij = _failure_coactivation / window_size
        Δwij_sustained = -λ_F · Fij · wij(t)

        Returns
        -------
        np.ndarray shape (N, N) : the sustained LTD delta (non-positive)
        """
        cfg = self.config
        Fij = self._failure_coactivation / float(cfg.failure_memory_window)
        return -cfg.ltd_sustained_lr * Fij * self.W

    # ──────────────────────────────────────────────
    # 2.5  Full Hebbian update (Eq. 4, refined)
    # ──────────────────────────────────────────────

    def update(
        self,
        positions: List[Optional[Tuple[float, float, float]]],
        step_rewards: List[float],
        advantages: Optional[List[float]] = None,
        comm_events: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[np.ndarray]:
        """Execute one full Hebbian plasticity step.

        Implements Eq. 4 (refined) from the thesis proposal:
        Δwij = m(t)·cij(t)·(1-wij(t)) - λ·wij(t) + Δwij_sustained
        wij(t+1) = clip(wij(t) + Δwij, 0, 1)

        Parameters
        ----------
        positions : list of (x,y,z) or None, one per agent
        step_rewards : list of floats, one per agent
        advantages : list of per-agent advantages, or None
        comm_events : list of (sender, receiver) index pairs, or None

        Returns
        -------
        np.ndarray : updated W matrix, or None if disabled
        """
        if not self.config.enabled:
            return None

        cfg = self.config

        # Co-activity
        cij = self._compute_coactivity(positions, step_rewards, comm_events)
        self._last_coactivity = cij

        # Per-pair modulator matrix (N×N) — uses per-agent advantages when available
        m = self._compute_modulator(advantages, step_rewards)

        # Team-level advantage for the failure window (scalar mean — intentional:
        # sustained LTD tracks episodes where the *whole team* repeatedly co-fails,
        # which is a team-level signal, not per-pair).
        if advantages is not None:
            valid = [a for a in advantages if a is not None]
            At_team = float(np.mean([_sanitize_reward(a) for a in valid])) if valid else 0.0
        else:
            sanitized = [_sanitize_reward(r) for r in step_rewards]
            At_team = float(np.mean(sanitized)) / (self._max_reward_seen + _EPS)

        # Update failure window BEFORE computing sustained LTD
        self._update_failure_window(cij, At_team)

        # Main Hebbian delta — m is now (N×N), element-wise with cij and W
        delta_main = (
            m * cij * (1.0 - self.W)               # advantage-gated LTP/LTD
            + cfg.base_ltp * cij * (1.0 - self.W)  # unconditional co-activity LTP
            - cfg.decay * self.W                    # passive decay
        )

        # Sustained LTD
        delta_ltd = self._compute_sustained_ltd()

        # Full update
        self.W = self.W + delta_main + delta_ltd

        # Hard constraints
        np.clip(self.W, 0.0, 1.0, out=self.W)
        np.fill_diagonal(self.W, 0.0)

        self._step_count += 1

        # Per-interval logging: show LTP vs LTD pair counts so weight trends are visible
        if self._step_count % cfg.log_graph_every == 0:
            ltp_pairs = int((delta_main > 0).sum())
            ltd_pairs = int((delta_main < 0).sum())
            mean_w = float(self.W[self.W > 0].mean()) if (self.W > 0).any() else 0.0
            max_w = float(self.W.max())
            logger.info(
                "[Hebbian step=%d] W mean=%.4f max=%.4f  LTP pairs=%d  LTD pairs=%d",
                self._step_count, mean_w, max_w, ltp_pairs, ltd_pairs,
            )

        return self.W

    # ──────────────────────────────────────────────
    # 2.6  Normalised weights (Eq. 6)
    # ──────────────────────────────────────────────

    def get_normalized_weights(self, i: int) -> np.ndarray:
        """Return normalised outgoing bond weights for agent i.

        Implements Eq. 6: w̄ij = wij / (Σ_{k≠i} wik + ε)

        Returns
        -------
        np.ndarray shape (N,) with values summing to ~1
        """
        if not self.config.enabled:
            N = self.config.num_agents
            return np.zeros(N, dtype=np.float32)
        row = self.W[i].copy()
        row[i] = 0.0
        total = row.sum() + _EPS
        return row / total

    def get_weight(self, i: int, j: int) -> float:
        """Return raw bond weight wij."""
        if not self.config.enabled:
            return 0.0
        return float(self.W[i, j])

    def get_all_weights(self) -> np.ndarray:
        """Return a copy of the full weight matrix W."""
        if not self.config.enabled:
            return np.zeros(
                (self.config.num_agents, self.config.num_agents),
                dtype=np.float32,
            )
        return self.W.copy()

    # ──────────────────────────────────────────────
    # 2.7  Social replay (Eq. 7)
    # ──────────────────────────────────────────────

    def get_social_replay_indices(
        self,
        agent_i: int,
        buffer_sizes: List[int],
        rho: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """Compute social replay sample indices for agent_i.

        Implements Eq. 7 from the thesis proposal.  Returns a list of
        (buffer_idx, agent_j) pairs to sample from neighbour buffers.
        The number of samples from agent_j is proportional to w̄ij * ρ.
        Agent_i's own buffer gets weight (1-ρ).

        Agents with wij < 0.05 are excluded to keep the graph sparse.

        Parameters
        ----------
        agent_i : int
        buffer_sizes : list of ints (transitions per agent)
        rho : float, optional (defaults to config.social_replay_rho)

        Returns
        -------
        list of (buffer_idx, agent_j) pairs
        """
        if not self.config.enabled:
            return []

        if rho is None:
            rho = self.config.social_replay_rho

        if rho <= 0.0:
            return []

        w_bar = self.get_normalized_weights(agent_i)
        N = self.config.num_agents

        # Total samples from neighbours = rho * own_buffer_size
        own_size = buffer_sizes[agent_i] if agent_i < len(buffer_sizes) else 0
        total_neighbour_samples = int(rho * own_size)
        if total_neighbour_samples <= 0:
            return []

        indices = []
        for j in range(N):
            if j == agent_i:
                continue
            if self.W[agent_i, j] < 0.05:
                continue  # sparse graph — skip weak bonds
            if j >= len(buffer_sizes) or buffer_sizes[j] <= 0:
                continue

            n_samples = max(1, int(w_bar[j] * total_neighbour_samples))
            n_samples = min(n_samples, buffer_sizes[j])

            # Sample random indices from agent_j's buffer
            sampled = np.random.randint(0, buffer_sizes[j], size=n_samples)
            for idx in sampled:
                indices.append((int(idx), j))

        return indices

    # ──────────────────────────────────────────────
    # 2.8  Reward diffusion (Eq. 8)
    # ──────────────────────────────────────────────

    def diffuse_rewards(
        self,
        raw_rewards: List[float],
        co_activity_matrix: Optional[np.ndarray] = None,
    ) -> List[float]:
        """Apply reward diffusion across the social graph.

        Implements Eq. 8: r'i(t) = (1-γ)·ri(t) + γ·Σ_{j≠i} w̄ij·cij·rj(t)

        When γ=0 or the module is disabled, returns raw_rewards unchanged.

        Parameters
        ----------
        raw_rewards : list of floats, one per agent
        co_activity_matrix : (N, N) array from current step, or None
            If None, uses self._last_coactivity.

        Returns
        -------
        list of floats : diffused rewards
        """
        if not self.config.enabled:
            return list(raw_rewards)

        gamma = self.config.reward_diffusion_gamma
        if gamma <= 0.0:
            return list(raw_rewards)

        N = self.config.num_agents
        cij = co_activity_matrix
        if cij is None:
            cij = self._last_coactivity
        if cij is None:
            return list(raw_rewards)

        rewards = [_sanitize_reward(r) for r in raw_rewards]
        diffused = []
        for i in range(N):
            w_bar = self.get_normalized_weights(i)
            social_reward = 0.0
            for j in range(N):
                if j == i:
                    continue
                social_reward += w_bar[j] * cij[i, j] * rewards[j]
            r_prime = (1.0 - gamma) * rewards[i] + gamma * social_reward
            diffused.append(r_prime)

        return diffused

    # ──────────────────────────────────────────────
    # 2.9  Graph metrics (for RQ2 analysis)
    # ──────────────────────────────────────────────

    def get_graph_metrics(self) -> Dict:
        """Compute graph-level metrics for logging and analysis.

        Returns metrics for RQ2 analysis: bond strength, sparsity,
        top pairs, per-agent centrality, modularity proxy, and the
        LTD heatmap.

        Returns
        -------
        dict with keys: mean_bond_strength, sparsity, top_3_pairs,
        per_agent_out_strength, modularity_proxy, ltd_heatmap
        """
        if not self.config.enabled:
            return {}

        N = self.config.num_agents
        W = self.W

        # Mean bond strength (excluding diagonal)
        mask = ~np.eye(N, dtype=bool)
        off_diag = W[mask]
        mean_bond = float(np.mean(off_diag)) if off_diag.size > 0 else 0.0

        # Sparsity: fraction of off-diagonal bonds < 0.1
        sparsity = float(np.mean(off_diag < 0.1)) if off_diag.size > 0 else 1.0

        # Top 3 pairs by weight
        top_3_pairs = []
        indices = np.argsort(off_diag)[::-1]
        # Convert flat off-diagonal index back to (i, j)
        row_indices, col_indices = np.where(mask)
        for k in range(min(3, len(indices))):
            flat_idx = indices[k]
            i_idx = int(row_indices[flat_idx])
            j_idx = int(col_indices[flat_idx])
            top_3_pairs.append({
                "i": i_idx,
                "j": j_idx,
                "w": float(W[i_idx, j_idx]),
            })

        # Per-agent out-strength: Σ_j wij
        per_agent_out_strength = [float(W[i].sum()) for i in range(N)]

        # Modularity proxy: mean within-role - mean cross-role
        modularity_proxy = None
        if self._agent_roles is not None and len(self._agent_roles) == N:
            within_sum = 0.0
            within_count = 0
            cross_sum = 0.0
            cross_count = 0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    if self._agent_roles[i] == self._agent_roles[j]:
                        within_sum += W[i, j]
                        within_count += 1
                    else:
                        cross_sum += W[i, j]
                        cross_count += 1
            within_mean = within_sum / max(within_count, 1)
            cross_mean = cross_sum / max(cross_count, 1)
            modularity_proxy = float(within_mean - cross_mean)

        # LTD heatmap: current Fij
        ltd_heatmap = (
            self._failure_coactivation / float(self.config.failure_memory_window)
        )

        return {
            "mean_bond_strength": mean_bond,
            "sparsity": sparsity,
            "top_3_pairs": top_3_pairs,
            "per_agent_out_strength": per_agent_out_strength,
            "modularity_proxy": modularity_proxy,
            "ltd_heatmap": ltd_heatmap.tolist(),
        }

    def get_ltd_heatmap(self) -> np.ndarray:
        """Return the failure co-occurrence matrix Fij for RQ2 plots.

        Returns
        -------
        np.ndarray shape (N, N) : Fij = _failure_coactivation / window_size
        """
        if not self.config.enabled:
            return np.zeros(
                (self.config.num_agents, self.config.num_agents),
                dtype=np.float32,
            )
        return (
            self._failure_coactivation / float(self.config.failure_memory_window)
        )

    # ──────────────────────────────────────────────
    # 2.10  Serialisation and reset
    # ──────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Serialise graph state to a JSON-compatible dict.

        Stores W, failure history, max reward, step count, and config
        fields needed for restoration.
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "W": self.W.tolist(),
            "_failure_coactivation": self._failure_coactivation.tolist(),
            "_max_reward_seen": self._max_reward_seen,
            "_step_count": self._step_count,
            "num_agents": self.config.num_agents,
            "init_weight": self.config.init_weight,
            "_last_coactivity": self._last_coactivity.tolist() if self._last_coactivity is not None else None,
        }

    def snapshot(self) -> dict:
        """Compact per-episode snapshot for hebbian_snapshots.jsonl."""
        d = self.to_dict()
        return {
            "W": d.get("W"),
            "step": d.get("_step_count", 0),
            "max_reward_seen": d.get("_max_reward_seen", 0.0),
            "num_agents": self.config.num_agents,
        }

    def from_dict(self, d: Dict) -> None:
        """Restore state from a to_dict() snapshot.

        Parameters
        ----------
        d : dict from a previous ``to_dict()`` call
        """
        if not self.config.enabled or not d.get("enabled", False):
            return

        self.W = np.array(d["W"], dtype=np.float32)
        self._failure_coactivation = np.array(
            d["_failure_coactivation"], dtype=np.float32
        )
        self._max_reward_seen = float(d.get("_max_reward_seen", 0.0))
        self._step_count = int(d.get("_step_count", 0))
        lc = d.get("_last_coactivity")
        self._last_coactivity = np.array(lc, dtype=np.float32) if lc is not None else None
        # Clear the deque — we cannot restore per-step snapshots from JSON
        self._failure_window_buffer.clear()

    def reset(self) -> None:
        """Reinitialise W to init_weight, clear all history.

        Preserves config and agent_roles.
        """
        if not self.config.enabled:
            return

        N = self.config.num_agents
        self.W = np.full((N, N), self.config.init_weight, dtype=np.float32)
        np.fill_diagonal(self.W, 0.0)
        self._failure_coactivation = np.zeros((N, N), dtype=np.float32)
        self._failure_window_buffer.clear()
        self._max_reward_seen = 0.0
        self._last_coactivity = None
        self._step_count = 0
