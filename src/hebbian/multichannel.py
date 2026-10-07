"""Multi-channel three-factor Hebbian graph (AgentWorld scaling chapter).

Generalises the ``three_factor`` rule of :class:`HebbianSocialGraph` from the
fixed WIRE channels (spat / comm / obs / imit) to any set of channel-tagged
pairwise events, so AgentWorld's explicit cooperation signals can wire bonds:

    xfer    giver → receiver of a successful ``transfer_items``
    combat  i and j attacked the same target within a short window
    read    reader → author of a board post that was read (OFF by default:
            with bond-gated reading it closes a W → reads → W loop)

``graph.py`` is deliberately untouched: every existing mode stays
byte-identical, and with comm-only events and ``salience_mode="ego"`` this
class reproduces ``HebbianSocialGraph(mode="three_factor")`` exactly (pinned by
tests/test_hebbian_multichannel.py).

Two additions on top of the inherited algebra:

* **Per-channel deltas and symmetry.** c_ij = clip(c_spat + Σ_ch c_ch), where
  c_ch[i, j] = δ_ch on an (i → j, ch) event, mirrored to [j, i] when the
  channel is symmetric. A channel listed in ``channel_engages_target`` also
  counts the target as socially active for the engagement gate (messages,
  transfers and joint fights engage both parties; an observation does not).
* **Joint salience (opt-in, "v2.2").** ``salience_mode="joint"`` replaces the
  ego term η₊·|r_bond_i|/R with η₊·√(r̃_i·r̃_j)/R, where r̃ is a leaky-max
  trace r̃ ← max(ρ_r·r̃, |r_bond|). A pair is credited only when BOTH partners
  were recently rewarded, and the trace bridges the few-round lag between a
  giver's transfer and the receiver's craft.

2-D positions (AgentWorld tiles) are padded to (x, y, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from hebbian.config import HebbianConfig
from hebbian.graph import _EPS, HebbianSocialGraph, _sanitize_reward

# Channels the WIRE rule already knows; their defaults below mirror graph.py.
_WIRE_CHANNELS = ("comm", "obs", "imit")


@dataclass
class MultiChannelConfig(HebbianConfig):
    """HebbianConfig plus per-channel co-activity settings.

    ``channel_deltas`` / ``channel_symmetric`` entries override the defaults
    resolved in :meth:`resolved_deltas` / :meth:`resolved_symmetric`; channels
    not mentioned keep them (comm/obs/imit follow the inherited WIRE fields).
    """

    mode: str = "three_factor"
    social_act_channels: tuple = ("comm", "xfer", "combat")
    channel_deltas: Dict[str, float] = field(default_factory=dict)
    channel_symmetric: Dict[str, bool] = field(default_factory=dict)
    channel_engages_target: tuple = ("comm", "xfer", "combat")
    salience_mode: str = "ego"          # "ego" (Hebbian 2.0) | "joint" (v2.2)
    salience_trace_rho: float = 0.7     # ρ_r of the joint-salience trace
    delta_window: int = 50              # bond_delta_row window (updates)

    def __post_init__(self) -> None:
        if self.mode != "three_factor":
            raise ValueError(
                f"MultiChannelConfig only supports mode='three_factor', got {self.mode!r}")
        if self.salience_mode not in ("ego", "joint"):
            raise ValueError(f"unknown salience_mode {self.salience_mode!r}")

    def resolved_deltas(self) -> Dict[str, float]:
        delta_soc = (self.social_coactivity_bonus
                     if self.social_coactivity_bonus is not None
                     else self.communication_coactivity_bonus)
        out = {
            "comm": self.communication_coactivity_bonus,
            "obs": delta_soc,
            "imit": delta_soc,
            "xfer": 0.8,
            "combat": 0.5,
            "read": 0.0,
        }
        out.update(self.channel_deltas)
        return out

    def resolved_symmetric(self) -> Dict[str, bool]:
        out = {
            "comm": True,
            "obs": self.social_bidirectional,
            "imit": self.social_bidirectional,
            "xfer": True,
            "combat": True,
            "read": False,
        }
        out.update(self.channel_symmetric)
        return out


class MultiChannelHebbianGraph(HebbianSocialGraph):
    """Three-factor Hebbian graph over arbitrary channel-tagged events."""

    config: MultiChannelConfig

    def __init__(self, config: MultiChannelConfig, agent_roles: Optional[List[int]] = None):
        super().__init__(config, agent_roles)
        self._last_c: Dict[str, np.ndarray] = {}
        if not config.enabled:
            return
        N = config.num_agents
        self._deltas = config.resolved_deltas()
        self._symmetric = config.resolved_symmetric()
        channels = ["spat"] + sorted(set(self._deltas) | set(config.social_act_channels))
        self._growth_by_channel = {ch: np.zeros((N, N), dtype=np.float32) for ch in channels}
        self._salience_trace = np.zeros(N, dtype=np.float32)
        self._elig_by_channel: Dict[str, np.ndarray] = {}
        self._bond_delta_window = max(2, int(config.delta_window))
        self._W_history = type(self._W_history)(maxlen=self._bond_delta_window)
        self._W_history.append(self.W.copy())

    # ── co-activity ─────────────────────────────────────────────────────────
    @staticmethod
    def _pad_positions(positions, N: int) -> np.ndarray:
        pos = np.full((N, 3), np.nan, dtype=np.float32)
        if positions is None:
            return pos
        for i in range(min(N, len(positions))):
            p = positions[i]
            if p is None:
                continue
            arr = np.asarray(p, dtype=np.float32).ravel()[:3]
            pos[i, : arr.size] = arr
            if arr.size < 3:
                pos[i, arr.size:] = 0.0
        return pos

    def _coactivity_gated(
        self,
        positions: List[Optional[Tuple[float, ...]]],
        engagement: np.ndarray,
        credited_events: Optional[List[Tuple[int, int, str]]],
    ) -> np.ndarray:
        cfg = self.config
        N = cfg.num_agents

        pos = self._pad_positions(positions, N)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        close = (dist <= cfg.interaction_radius) & np.isfinite(dist)
        spatial_gate = close.astype(np.float32)
        np.fill_diagonal(spatial_gate, 0.0)
        if cfg.coact_floor > 0.0:
            c_spat = spatial_gate * np.maximum(np.outer(engagement, engagement), cfg.coact_floor)
        else:
            c_spat = spatial_gate * np.outer(engagement, engagement)

        terms: Dict[str, np.ndarray] = {"spat": c_spat}
        for s, r, ch in credited_events or ():
            if not (0 <= s < N and 0 <= r < N and s != r):
                continue
            delta = self._deltas.get(ch, 0.0)
            if delta <= 0.0:
                continue
            term = terms.setdefault(ch, np.zeros((N, N), dtype=np.float32))
            term[s, r] = delta
            if self._symmetric.get(ch, False):
                term[r, s] = delta

        # Fixed summation order (graph.py's spat+comm+obs+imit first) so the
        # float32 sum, and hence W, matches the parent rule bit for bit.
        order = ["spat", *_WIRE_CHANNELS, *sorted(set(terms) - {"spat", *_WIRE_CHANNELS})]
        terms = {ch: terms[ch] for ch in order if ch in terms}
        raw = terms["spat"]
        for ch, term in terms.items():
            if ch != "spat":
                raw = raw + term
        cij = np.clip(raw, 0.0, 1.0)
        np.fill_diagonal(cij, 0.0)
        cij[cij < cfg.coop_eps] = 0.0

        self._last_c = terms
        # Keep the parent's attribute names populated for existing readers.
        zeros = np.zeros((N, N), dtype=np.float32)
        self._last_c_spat = c_spat
        self._last_c_comm = terms.get("comm", zeros)
        self._last_c_obs = terms.get("obs", zeros)
        self._last_c_imit = terms.get("imit", zeros)
        return cij

    # ── update ──────────────────────────────────────────────────────────────
    def _update_gated(
        self,
        positions,
        comm_events,
        chambers,
        total_rewards,
        bond_rewards,
        social_events=None,
        death_rewards=None,
    ) -> np.ndarray:
        cfg = self.config
        N = cfg.num_agents

        chamber_gate = self._chamber_gate(chambers, N)
        bond_g = self._reward_vector(bond_rewards, N) * chamber_gate
        total_g = self._reward_vector(total_rewards, N) * chamber_gate

        events: List[Tuple[int, int, str]] = []
        if comm_events:
            events.extend((s, r, "comm") for s, r in comm_events)
        if social_events:
            events.extend(social_events)
        mask = set(cfg.social_act_channels)
        credited = [(s, r, ch) for s, r, ch in events if ch in mask]

        engages_target = set(cfg.channel_engages_target)
        social_agents = set()
        for s, r, ch in credited:
            social_agents.add(s)
            if ch in engages_target:
                social_agents.add(r)

        engagement = self._engagement(bond_g, social_agents)
        self._last_engagement = engagement
        cij = self._coactivity_gated(positions, engagement, credited)
        self._last_coactivity = cij

        coop, neg = self._windowed_stats(cij, total_g)

        coeff = self._growth_coeff(bond_g)
        growth = coeff[:, None] * cij * (1.0 - self.W)

        self._eligibility = (cfg.eligibility_rho * self._eligibility + cij).astype(np.float32)
        abs_r = np.abs(bond_g).astype(np.float32)
        if cfg.salience_mode == "joint":
            self._salience_trace = np.maximum(
                cfg.salience_trace_rho * self._salience_trace, abs_r).astype(np.float32)
            joint = np.sqrt(np.outer(self._salience_trace, self._salience_trace))
            salience = (cfg.eta_plus * joint / float(cfg.reward_norm_R)).astype(np.float32)
        else:
            salience = (cfg.eta_plus * abs_r / float(cfg.reward_norm_R)).astype(np.float32)[:, None]
        growth = growth + salience * self._eligibility * (1.0 - self.W)

        # Per-channel shares of the clipped c_ij and per-channel traces
        # (Σ_ch e_ch = e), so growth driven by PAST co-activity through the
        # trace is attributed to the channel that laid the trace down.
        raw_total = sum(self._last_c.values()) + _EPS
        shares = {ch: term / raw_total * cij for ch, term in self._last_c.items()}
        for ch in set(self._elig_by_channel) | set(shares):
            prev = self._elig_by_channel.get(ch)
            prev = cfg.eligibility_rho * prev if prev is not None else 0.0
            self._elig_by_channel[ch] = (prev + shares.get(ch, 0.0)).astype(np.float32)

        death_ltd = np.zeros((N, N), dtype=np.float32)
        if cfg.eta_minus_death > 0.0 and death_rewards is not None:
            d = np.array(
                [min(abs(_sanitize_reward(death_rewards[i])), cfg.death_cap)
                 if i < len(death_rewards) else 0.0
                 for i in range(N)], dtype=np.float32,
            )
            death_ltd = (cfg.eta_minus_death
                         * (d / float(cfg.reward_norm_R))[:, None]
                         * self._eligibility * self.W)

        decay_mask = (coop < cfg.coop_eps) & neg[:, None]
        decay = cfg.eta_minus * self.W
        homeostatic = cfg.decay * self.W
        delta = np.where(decay_mask, -decay, growth) - homeostatic - death_ltd
        np.fill_diagonal(delta, 0.0)
        self._last_growth = np.where(decay_mask, 0.0, growth)
        np.fill_diagonal(self._last_growth, 0.0)
        self._last_decay = np.where(decay_mask, decay, 0.0) + homeostatic + death_ltd
        np.fill_diagonal(self._last_decay, 0.0)

        keep = (~decay_mask).astype(np.float32)
        np.fill_diagonal(keep, 0.0)
        headroom = 1.0 - self.W
        for ch, e_ch in self._elig_by_channel.items():
            floor = coeff[:, None] * shares.get(ch, 0.0)
            acc = self._growth_by_channel.setdefault(ch, np.zeros((N, N), dtype=np.float32))
            acc += keep * (floor + salience * e_ch) * headroom

        if not cfg.freeze_weights:
            self.W = self.W + delta
            np.clip(self.W, 0.0, 1.0, out=self.W)
            np.fill_diagonal(self.W, 0.0)

        self._step_count += 1
        self._W_history.append(self.W.copy())
        return self.W

    @staticmethod
    def _reward_vector(values: Optional[List[float]], N: int) -> np.ndarray:
        if values is None:
            return np.zeros(N, dtype=np.float32)
        return np.array(
            [_sanitize_reward(values[i]) if i < len(values) else 0.0 for i in range(N)],
            dtype=np.float32,
        )

    # ── read-outs ───────────────────────────────────────────────────────────
    def channel_growth(self) -> Dict[str, np.ndarray]:
        """Cumulative realised growth split by channel (copies)."""
        return {k: v.copy() for k, v in (self._growth_by_channel or {}).items()}

    def last_channel_terms(self) -> Dict[str, np.ndarray]:
        """Pre-clip co-activity terms of the last update, keyed by channel."""
        return {k: v.copy() for k, v in self._last_c.items()}

    def top_k(self, i: int, k: int, candidates: Optional[List[int]] = None) -> List[int]:
        """Agent i's k strongest outgoing bonds (ties broken by index)."""
        if not self.config.enabled:
            return []
        row = self.W[i]
        pool = [j for j in (candidates if candidates is not None else range(len(row)))
                if j != i]
        pool.sort(key=lambda j: (-float(row[j]), j))
        return pool[:k]

    def to_dict(self) -> Dict:
        d = super().to_dict()
        if self.config.enabled:
            d["_salience_trace"] = self._salience_trace.tolist()
        return d

    def from_dict(self, d: Dict) -> None:
        super().from_dict(d)
        st = d.get("_salience_trace")
        if self.config.enabled and st is not None:
            self._salience_trace = np.array(st, dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        if self.config.enabled:
            self._salience_trace[:] = 0.0
            self._elig_by_channel = {}
            self._last_c = {}
