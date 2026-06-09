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

        if config.init_matrix is not None:
            self.W = self._load_matrix(config.init_matrix, N)
        elif config.init_preset and config.init_preset != "none":
            self.W = self.build_preset_matrix(
                config.init_preset, N,
                strong=config.preset_bond_strong,
                weak=config.preset_bond_weak,
                hub=config.preset_hub,
            )
        else:
            self.W = np.full((N, N), config.init_weight, dtype=np.float32)
        np.fill_diagonal(self.W, 0.0)

        # Sustained LTD state
        self._failure_coactivation = np.zeros((N, N), dtype=np.float32)
        self._failure_window_buffer: deque = deque(
            maxlen=config.failure_memory_window,
        )

        # Running max of |ri(t)| for engagement normalisation
        self._max_reward_seen: float = 0.0

        # ── Gated-Hebbian variant state (mode != "legacy") ──────────────
        if config.mode != "legacy" and config.init_weight <= 0.0:
            raise ValueError(
                f"mode={config.mode!r} requires init_weight (w₀) > 0; "
                f"got {config.init_weight}. W=0 is a fixed point of the gated "
                "rule, so a zero warm-start can never grow."
            )
        self._coact_window: deque = deque(maxlen=max(1, config.coop_window))
        self._reward_window: deque = deque(maxlen=max(1, config.coop_window))
        self._max_bond_reward_seen: float = 0.0
        # Realized per-branch deltas from the last gated step (analysis/tests).
        self._last_growth: Optional[np.ndarray] = None
        self._last_decay: Optional[np.ndarray] = None

        self._bond_delta_window: int = 50
        self._W_history: deque = deque(maxlen=self._bond_delta_window)
        self._W_history.append(self.W.copy())


    @staticmethod
    def _load_matrix(matrix, N: int) -> np.ndarray:
        """Validate + load an explicit N×N init matrix (diagonal forced to 0)."""
        W = np.array(matrix, dtype=np.float32)
        if W.shape != (N, N):
            raise ValueError(
                f"init_matrix shape {W.shape} != ({N}, {N}) for num_agents={N}"
            )
        np.clip(W, 0.0, 1.0, out=W)
        np.fill_diagonal(W, 0.0)
        return W

    @staticmethod
    def build_preset_matrix(
        preset: str, N: int, strong: float = 0.8, weak: float = 0.1, hub: int = 0
    ) -> np.ndarray:
        """Construct a named hardcoded social topology.

        W[i, j] is agent i's (directed) bond toward j. See HebbianConfig for the
        semantics of each preset. Diagonal is always 0.
        """
        W = np.full((N, N), weak, dtype=np.float32)
        if preset == "uniform":
            W[:] = strong
        elif preset == "star":
            if not (0 <= hub < N):
                raise ValueError(f"preset_hub {hub} out of range for N={N}")
            W[:, hub] = strong            # everyone → hub strongly
        elif preset == "ring":
            for i in range(N):
                W[i, (i + 1) % N] = strong  # i → i+1 (directed cycle)
        elif preset == "pair":
            if N >= 2:
                W[0, 1] = strong
                W[1, 0] = strong          # 0 ↔ 1 dyad; everyone else weak
        else:
            raise ValueError(f"unknown init_preset {preset!r}")
        np.fill_diagonal(W, 0.0)
        return W


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
        cij_spatial = spatial_gate * np.outer(engagement, engagement)

        # ── Communication co-activity bonus ──
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


    def _compute_modulator(
        self,
        advantages: Optional[List[Optional[float]]],
        step_rewards: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compute the per-pair modulatory signal matrix m(t) ∈ R^{N×N}.

        ASYMMETRIC (egocentric): the bond i→j is updated using agent i's
        OWN signal, not the pair mean. This produces a directed social
        graph where W[i,j] represents "agent i's attention/trust toward j",
        and W[i,j] ≠ W[j,i] in general.

        Semantics: when agent i succeeds, all of i's outgoing bonds toward
        co-active partners strengthen — i learns "the agents I was with
        when I did well are valuable". Differentiation across partners
        comes from cij (only currently-co-active pairs get the LTP).

        Update rule per direction:
            m_ij_ltp = ltp_lr  * tanh(β * max( A_i, 0))
            m_ij_ltd = ltd_lr  * tanh(β * max(-A_i - ltd_threshold, 0))
            m_ij     = m_ij_ltp - m_ij_ltd

        Parameters
        ----------
        advantages : list of per-agent one-step advantages (δ = r - V(s)),
            or None per agent when RL layer is disabled for that agent.
            When the whole list is None, falls back to normalised rewards.
        step_rewards : per-agent rewards — fallback for agents without RL.

        Returns
        -------
        np.ndarray shape (N, N) : per-pair modulator matrix, diagonal = 0.
            Asymmetric: m[i,j] depends only on agent i's signal.
        """
        N = self.config.num_agents
        cfg = self.config

        agent_signals = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if advantages is not None and i < len(advantages) and advantages[i] is not None:
                agent_signals[i] = _sanitize_reward(advantages[i])
            elif step_rewards is not None and i < len(step_rewards):
                agent_signals[i] = (
                    _sanitize_reward(step_rewards[i]) / (self._max_reward_seen + _EPS)
                )
        np.clip(agent_signals, -1.0, 1.0, out=agent_signals)

        ltp_per_agent = np.zeros(N, dtype=np.float32)
        ltd_per_agent = np.zeros(N, dtype=np.float32)
        for i in range(N):
            ltp_per_agent[i] = cfg.ltp_lr * math.tanh(
                cfg.modulation_beta * max(agent_signals[i], 0.0)
            )
            ltd_per_agent[i] = cfg.ltd_lr * math.tanh(
                cfg.modulation_beta * max(-agent_signals[i] - cfg.ltd_threshold, 0.0)
            )

        m_per_agent = ltp_per_agent - ltd_per_agent  # (N,)
        m = np.tile(m_per_agent.reshape(N, 1), (1, N))
        np.fill_diagonal(m, 0.0)
        return m


    def _update_failure_window(
        self, cij: np.ndarray, At: float
    ) -> None:
        """Update the sliding window of failure co-activity snapshots.

        Called each step before computing sustained LTD so the history
        is current.

        Treats only **clearly** negative team advantages as failures —
        gated by `ltd_threshold`, mirroring the per-step LTD rule. Without
        this gate, exploration noise (At ≈ −V(s), small negative on most
        zero-reward steps) saturates the failure window in ~50 steps and
        sustained LTD halves every co-active bond every ~350 steps,
        regardless of whether the team is genuinely failing. That's the
        bug behind the "weights climb to ~0.5, then crash to 0" pattern.
        """
        cfg = self.config
        if At < -cfg.ltd_threshold:
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


    @staticmethod
    def _chamber_gate(chambers: Optional[List[int]], N: int) -> np.ndarray:
        """1[chamber_i >= 2] as a float (N,) vector.

        Restricts every reward read to cooperative chambers (Ch2–Ch5). A
        None list means "no chamber info" → assume all cooperative (gate=1),
        which is what the unit tests and non-Craftium callers want.
        """
        if chambers is None:
            return np.ones(N, dtype=np.float32)
        gate = np.zeros(N, dtype=np.float32)
        for i in range(min(N, len(chambers))):
            ch = chambers[i]
            if ch is not None and ch >= 2:
                gate[i] = 1.0
        return gate

    def _engagement(
        self,
        bond_rewards_gated: np.ndarray,
        comm_agents: set,
    ) -> np.ndarray:
        """Engagement score g_i(t) ∈ [0,1] (shared gate, both variants).

        g_i = clip( α·|r_bond_i|/r_max·1[ch≥2] + (1−α)·1[i communicates], 0, 1)

        ``bond_rewards_gated`` is already Ch2–5-gated and DEATH-excluded, so
        engagement reflects cooperative salience only — a death spike can
        never raise g_i. ``r_max`` is a running max over the same Ch2–5
        bondable magnitudes (drift is harmless: g_i is a bounded gate).
        """
        cfg = self.config
        N = cfg.num_agents
        abs_r = np.abs(bond_rewards_gated)
        cur_max = float(abs_r.max()) if abs_r.size else 0.0
        if cur_max > self._max_bond_reward_seen:
            self._max_bond_reward_seen = cur_max
        norm = self._max_bond_reward_seen + _EPS
        reward_component = cfg.engagement_reward_weight * (abs_r / norm)
        comm_component = np.array(
            [(1.0 - cfg.engagement_reward_weight) if i in comm_agents else 0.0
             for i in range(N)],
            dtype=np.float32,
        )
        return np.clip(reward_component + comm_component, 0.0, 1.0)

    def _coactivity_gated(
        self,
        positions: List[Optional[Tuple[float, float, float]]],
        engagement: np.ndarray,
        comm_events: Optional[List[Tuple[int, int]]],
    ) -> np.ndarray:
        """Shared co-activity gate c_ij(t) ∈ [0,1].

        c_spat = 1[‖p_i−p_j‖≤d]·g_i·g_j               (close AND both engaged)
        c_comm = δ_comm·1[i↔j message]·(1−1[‖p_i−p_j‖≤d])  (talk across distance)
        c_ij   = clip(c_spat + c_comm, 0, 1)

        Vectorised over the N×N grid. Co-activity below ε is floored to 0 (ε is
        the "no co-activity" threshold), which makes the growth and decay
        branches strictly mutually exclusive.
        """
        cfg = self.config
        N = cfg.num_agents

        # Spatial gate 1[dist ≤ d], vectorised. Missing positions ⇒ no gate.
        pos = np.full((N, 3), np.nan, dtype=np.float32)
        for i in range(N):
            if positions is not None and i < len(positions) and positions[i] is not None:
                pos[i] = np.asarray(positions[i], dtype=np.float32)[:3]
        diff = pos[:, None, :] - pos[None, :, :]          # (N,N,3)
        dist = np.sqrt((diff ** 2).sum(axis=2))           # (N,N), nan if missing
        close = (dist <= cfg.interaction_radius) & np.isfinite(dist)
        spatial_gate = close.astype(np.float32)
        np.fill_diagonal(spatial_gate, 0.0)

        c_spat = spatial_gate * np.outer(engagement, engagement)

        # Communication co-activity (only when NOT spatially co-located).
        c_comm = np.zeros((N, N), dtype=np.float32)
        if comm_events and cfg.communication_coactivity_bonus > 0.0:
            for s, r in comm_events:
                if 0 <= s < N and 0 <= r < N and s != r:
                    for a, b in ((s, r), (r, s)):  # symmetric exchange
                        c_comm[a, b] = cfg.communication_coactivity_bonus * (
                            1.0 - spatial_gate[a, b]
                        )

        cij = np.clip(c_spat + c_comm, 0.0, 1.0)
        np.fill_diagonal(cij, 0.0)
        # Floor sub-ε co-activity to exactly 0 (ε = "no co-activity").
        cij[cij < cfg.coop_eps] = 0.0
        return cij

    def _growth_coeff(self, bond_rewards_gated: np.ndarray) -> np.ndarray:
        """Per-agent growth coefficient (the ONLY thing differing A vs B).

        Implemented as a shared floor + a mode-specific salience term, so the
        constant-growth ("co-presence builds bonds") logic is not duplicated:

        - Variant A ("coactivity"):       coeff_i = η₊            (flat rate)
        - Variant B ("reward_modulated"):  coeff_i = η₀ + η₊·|r_bond_i|/R

        Note A is NOT B with η₊=0: Variant A's η₊ plays the role of Variant B's
        association floor η₀ (both are the co-presence constant). The thing
        ablated between B and A is the salience term η₊·|r_bond|/R.

        ``bond_rewards_gated`` is Ch2–5-gated and DEATH-excluded, so a death
        contributes exactly 0 to the salience term in Variant B.
        """
        cfg = self.config
        N = cfg.num_agents
        if cfg.mode == "coactivity":
            return np.full(N, cfg.eta_plus, dtype=np.float32)
        if cfg.mode == "reward_modulated":
            salience = cfg.eta_plus * np.abs(bond_rewards_gated) / float(cfg.reward_norm_R)
            return (cfg.eta_0 + salience).astype(np.float32)
        raise ValueError(f"unknown gated mode {cfg.mode!r}")

    def _windowed_stats(
        self, cij: np.ndarray, total_rewards_gated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Roll the windows and return (coop_ij, neg_i).

        coop_ij(t) = max_{t-n<s≤t} c_ij(s)   — peak co-activity in window (N,N)
        neg_i(t)   = 1[Σ_{t-n<s≤t} r_i(s) < −θ] — sustained loss for i (N,)

        Both windows include the current step (appended before reduction), so
        coop_ij ≥ c_ij(t): if coop_ij < ε then c_ij(t) < ε (= 0 after flooring),
        which is why decay implies zero growth — the mutual-exclusivity claim.
        """
        cfg = self.config
        N = cfg.num_agents
        self._coact_window.append(cij.copy())
        self._reward_window.append(total_rewards_gated.copy())

        coop = np.maximum.reduce(list(self._coact_window))
        reward_sum = np.sum(self._reward_window, axis=0)
        neg = (reward_sum < -cfg.neg_theta)
        return coop, neg

    def _update_gated(
        self,
        positions: List[Optional[Tuple[float, float, float]]],
        comm_events: Optional[List[Tuple[int, int]]],
        chambers: Optional[List[int]],
        total_rewards: Optional[List[float]],
        bond_rewards: Optional[List[float]],
    ) -> np.ndarray:
        """One gated-Hebbian step (Variant A or B). Vectorised over N×N.

        ΔW_ij  = growth_ij − decay_ij
        growth_ij = coeff_i · c_ij · (1 − W_ij)
        decay_ij  = η₋ · 1[coop_ij < ε  AND  neg_i] · W_ij

        The two branches are mutually exclusive (decay needs coop<ε ⇒ c_ij=0 ⇒
        no growth), so a per-pair np.where picks exactly one — this also makes
        the exclusivity invariant exact regardless of fractional gates.

        This whole method is a pure NumPy forward rule: NO autograd, no torch.
        """
        cfg = self.config
        N = cfg.num_agents

        chamber_gate = self._chamber_gate(chambers, N)

        # Bondable reward (death-excluded), Ch2–5-gated → growth + engagement.
        if bond_rewards is not None:
            bond_g = np.array(
                [_sanitize_reward(bond_rewards[i]) if i < len(bond_rewards) else 0.0
                 for i in range(N)], dtype=np.float32,
            ) * chamber_gate
        else:
            bond_g = np.zeros(N, dtype=np.float32)

        # Total reward (death-included), Ch2–5-gated → neg_i decay gate only.
        if total_rewards is not None:
            total_g = np.array(
                [_sanitize_reward(total_rewards[i]) if i < len(total_rewards) else 0.0
                 for i in range(N)], dtype=np.float32,
            ) * chamber_gate
        else:
            total_g = np.zeros(N, dtype=np.float32)

        comm_agents = set()
        if comm_events:
            for s, r in comm_events:
                comm_agents.add(s)
                comm_agents.add(r)

        engagement = self._engagement(bond_g, comm_agents)
        cij = self._coactivity_gated(positions, engagement, comm_events)
        self._last_coactivity = cij

        coop, neg = self._windowed_stats(cij, total_g)

        # Growth (per-agent coeff broadcast over outgoing bonds j).
        coeff = self._growth_coeff(bond_g)                       # (N,)
        growth = coeff[:, None] * cij * (1.0 - self.W)           # (N,N)

        decay_mask = (coop < cfg.coop_eps) & neg[:, None]        # (N,N) bool
        decay = cfg.eta_minus * self.W                           # (N,N)

        delta = np.where(decay_mask, -decay, growth)
        np.fill_diagonal(delta, 0.0)
        self._last_growth = np.where(decay_mask, 0.0, growth)
        np.fill_diagonal(self._last_growth, 0.0)
        self._last_decay = np.where(decay_mask, decay, 0.0)
        np.fill_diagonal(self._last_decay, 0.0)

        if not self.config.freeze_weights:
            self.W = self.W + delta
            np.clip(self.W, 0.0, 1.0, out=self.W)
            np.fill_diagonal(self.W, 0.0)

        self._step_count += 1
        self._W_history.append(self.W.copy())

        if self._step_count % cfg.log_graph_every == 0:
            grew = int((delta > 0).sum())
            shrank = int((delta < 0).sum())
            mean_w = float(self.W[self.W > 0].mean()) if (self.W > 0).any() else 0.0
            logger.info(
                "[Hebbian/%s step=%d] W mean=%.4f max=%.4f  grow=%d  decay=%d",
                cfg.mode, self._step_count, mean_w, float(self.W.max()), grew, shrank,
            )

        return self.W


    def update(
        self,
        positions: List[Optional[Tuple[float, float, float]]],
        step_rewards: Optional[List[float]] = None,
        advantages: Optional[List[float]] = None,
        comm_events: Optional[List[Tuple[int, int]]] = None,
        *,
        chambers: Optional[List[int]] = None,
        bond_rewards: Optional[List[float]] = None,
        total_rewards: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """Execute one full social-graph plasticity step.

        Dispatches on ``config.mode``:

        - ``"legacy"`` — the accreted advantage-modulator + failure-window +
          grace rule (uses ``step_rewards`` / ``advantages``). Unchanged.
        - ``"coactivity"`` / ``"reward_modulated"`` — the gated-Hebbian
          variants (see :meth:`_update_gated`). These read ``chambers``,
          ``total_rewards`` and (Variant B) ``bond_rewards``; the legacy
          ``step_rewards`` / ``advantages`` args are accepted but ignored.

        Parameters
        ----------
        positions : list of (x,y,z) or None, one per agent
        step_rewards : list of floats (legacy mode), one per agent
        advantages : list of per-agent advantages (legacy mode), or None
        comm_events : list of (sender, receiver) index pairs, or None
        chambers : per-agent chamber index (1..5); 0/None ⇒ solo/unknown,
            treated as Ch1 (non-cooperative, gated OUT of every reward read).
            If the whole list is None, all agents are assumed cooperative.
        bond_rewards : per-agent BONDABLE reward this step (milestone + comm +
            futile; DEATH EXCLUDED). Drives Variant B's growth magnitude and
            the engagement normaliser. Ignored by Variant A.
        total_rewards : per-agent TOTAL reward this step (death INCLUDED). Used
            only for the neg_i decay gate. Falls back to ``step_rewards``.

        Returns
        -------
        np.ndarray : updated W matrix, or None if disabled
        """
        if not self.config.enabled:
            return None

        if self.config.mode != "legacy":
            return self._update_gated(
                positions=positions,
                comm_events=comm_events,
                chambers=chambers,
                total_rewards=(total_rewards if total_rewards is not None
                               else step_rewards),
                bond_rewards=bond_rewards,
            )

        if step_rewards is None:
            raise ValueError("legacy mode requires step_rewards")

        cfg = self.config

        # Co-activity
        cij = self._compute_coactivity(positions, step_rewards, comm_events)
        self._last_coactivity = cij

        # Per-pair modulator matrix (N×N) — uses per-agent advantages when available
        m = self._compute_modulator(advantages, step_rewards)

        if advantages is not None:
            valid = [a for a in advantages if a is not None]
            At_team = float(np.mean([_sanitize_reward(a) for a in valid])) if valid else 0.0
        else:
            sanitized = [_sanitize_reward(r) for r in step_rewards]
            At_team = float(np.mean(sanitized)) / (self._max_reward_seen + _EPS)

        # Update failure window BEFORE computing sustained LTD
        self._update_failure_window(cij, At_team)

        grace_bonus = None
        if cfg.failure_grace_enabled and At_team < -cfg.ltd_threshold:
            F_ij = self._failure_coactivation / float(cfg.failure_memory_window)
            grace_factor = np.clip(
                1.0 - F_ij / max(cfg.failure_grace_threshold, _EPS),
                0.0,
                1.0,
            )
            grace_bonus = cfg.failure_ltp_lr * grace_factor * cij * (1.0 - self.W)

        m_term        = m * cij * (1.0 - self.W)         # advantage-gated LTP/LTD
        base_ltp_term = cfg.base_ltp * cij * (1.0 - self.W)  # unconditional co-activity LTP
        decay_term    = -cfg.decay * self.W              # passive decay (≤ 0)
        delta_main = m_term + base_ltp_term + decay_term
        if grace_bonus is not None:
            delta_main = delta_main + grace_bonus

        # Sustained LTD
        delta_ltd = self._compute_sustained_ltd()

        if not cfg.freeze_weights:
            self.W = self.W + delta_main + delta_ltd
            # Hard constraints
            np.clip(self.W, 0.0, 1.0, out=self.W)
            np.fill_diagonal(self.W, 0.0)

        self._step_count += 1

        self._W_history.append(self.W.copy())

        # Per-interval logging: show LTP vs LTD pair counts so weight trends are visible
        if self._step_count % cfg.log_graph_every == 0:
            ltp_pairs = int((delta_main > 0).sum())
            ltd_pairs = int((delta_main < 0).sum())
            mean_w = float(self.W[self.W > 0].mean()) if (self.W > 0).any() else 0.0
            max_w = float(self.W.max())
            grace_str = (
                f"  grace pairs={int((grace_bonus > 0).sum())}"
                if grace_bonus is not None else ""
            )
            logger.info(
                "[Hebbian step=%d] W mean=%.4f max=%.4f  LTP pairs=%d  LTD pairs=%d%s",
                self._step_count, mean_w, max_w, ltp_pairs, ltd_pairs, grace_str,
            )
            m_pos_sum    = float(m_term[m_term > 0].sum())
            m_neg_sum    = float(m_term[m_term < 0].sum())
            base_ltp_sum = float(base_ltp_term.sum())
            decay_sum    = float(decay_term.sum())
            grace_sum    = float(grace_bonus.sum()) if grace_bonus is not None else 0.0
            ltd_sum      = float(delta_ltd.sum())
            logger.info(
                "[Hebbian step=%d] Δw sums  m+=%+.4f m-=%+.4f base_ltp=%+.4f "
                "decay=%+.4f grace=%+.4f sustained_ltd=%+.4f  net=%+.4f",
                self._step_count, m_pos_sum, m_neg_sum, base_ltp_sum,
                decay_sum, grace_sum, ltd_sum,
                m_pos_sum + m_neg_sum + base_ltp_sum + decay_sum + grace_sum + ltd_sum,
            )

        return self.W


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

    def bond_delta_row(self, agent_i: int) -> Dict[int, float]:
        """Return Δw_{i,j} over the bond-delta window for each teammate j.

        The window length is fixed at construction (default 50 steps). When
        fewer than 2 snapshots exist, returns zeros — early-episode bonds
        haven't moved meaningfully yet.

        Returns
        -------
        dict[int, float] : {teammate_idx: w_now - w_window_ago}, diagonal excluded.
        """
        if not self.config.enabled or len(self._W_history) < 2:
            N = self.config.num_agents
            return {j: 0.0 for j in range(N) if j != agent_i}
        old = self._W_history[0]
        return {
            j: float(self.W[agent_i, j] - old[agent_i, j])
            for j in range(self.config.num_agents)
            if j != agent_i
        }

    def get_all_weights(self) -> np.ndarray:
        """Return a copy of the full weight matrix W."""
        if not self.config.enabled:
            return np.zeros(
                (self.config.num_agents, self.config.num_agents),
                dtype=np.float32,
            )
        return self.W.copy()


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


    def get_graph_metrics(self) -> Dict:
        """Compute graph-level metrics for logging and analysis.

        Returns metrics for RQ2 analysis: bond strength, sparsity,
        top pairs, per-agent centrality, modularity proxy, the LTD
        heatmap, AND the full W matrix per snapshot.

        Storing the full W (~N×N floats — trivial for typical N=3..8) lets
        post-hoc plots (e.g., asymmetric W[i,j] vs W[j,i] over time) work
        without losing pairs that drop out of the top-3 mid-episode. The
        previous schema only stored top_3_pairs, which made any plot of
        a specific pair show abrupt drops to 0 whenever that pair fell
        out of top-3 — see graph_bond_evolution.png artifact.

        Returns
        -------
        dict with keys: mean_bond_strength, sparsity, top_3_pairs,
        per_agent_out_strength, modularity_proxy, ltd_heatmap, W
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
            "W": W.tolist(),
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


    def to_dict(self) -> Dict:
        """Serialise graph state to a JSON-compatible dict.

        Stores W, failure history, max reward, step count, and config
        fields needed for restoration.
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "frozen": bool(self.config.freeze_weights),
            "mode": self.config.mode,
            "W": self.W.tolist(),
            "_failure_coactivation": self._failure_coactivation.tolist(),
            "_max_reward_seen": self._max_reward_seen,
            "_max_bond_reward_seen": self._max_bond_reward_seen,
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
        self._failure_window_buffer.clear()
        self._coact_window.clear()
        self._reward_window.clear()
        self._max_bond_reward_seen = float(d.get("_max_bond_reward_seen", 0.0))

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
        # Gated-variant window state
        self._coact_window.clear()
        self._reward_window.clear()
        self._max_bond_reward_seen = 0.0
