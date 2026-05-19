"""GRPO step-level metrics — the schema that lands in ``grpo_metrics.jsonl``.

Owns:

* ``GRPOStepMetrics`` — one record per GRPO update step
* ``_milestone_stats(batch)`` — milestone / chamber / per-agent / borrowing aggregation
* ``_hebbian_step_stats(bridge)`` — Hebbian-graph scalars from the live bridge
* ``CHAMBERS`` — canonical 6 track names matching ``MILESTONE_TRACK``

This module is import-safe in the local dev env: it does NOT import torch and
only lazily imports ``MILESTONE_TRACK`` from the legacy module (which is
import-safe — it's a pure dict).

The trainer (``src/rlvr/grpo_trainer.py``) re-exports ``GRPOStepMetrics`` and
``_milestone_stats`` for backwards-compatible imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rlvr.grpo_buffer import ScoredTrajectory

if TYPE_CHECKING:
    from rlvr.hebbian_grpo_bridge import HebbianGRPOBridge


# ──── canonical chambers ─────────────────────────────────────────────────


# Stable order matching the six tracks in ``MILESTONE_TRACK``. The thesis
# per-chamber table (T2) columns are derived from this list.
CHAMBERS: tuple[str, ...] = (
    "ch1_solo",
    "ch2_anvils",
    "ch3_switches",
    "ch4_combat",
    "ch5_boss",
    "communication",
)


def _milestone_to_chamber() -> dict[str, str]:
    """Return the milestone_id → chamber-track mapping.

    Imported lazily so the legacy ``craftium_metric`` module's heavier
    imports (matplotlib, numpy) don't run on every metrics_grpo import.
    Caches the result via a module-level singleton.
    """
    global _MID_TO_CHAMBER_CACHE
    if _MID_TO_CHAMBER_CACHE is None:
        try:
            from mindforge.agent_modules.craftium_metric import MILESTONE_TRACK
            _MID_TO_CHAMBER_CACHE = dict(MILESTONE_TRACK)
        except ImportError:
            # Local dev env or test isolation — return empty; chamber
            # breakdowns will all be zero, which is the safest fallback.
            _MID_TO_CHAMBER_CACHE = {}
    return _MID_TO_CHAMBER_CACHE


_MID_TO_CHAMBER_CACHE: dict[str, str] | None = None


# ──── the step-record dataclass ──────────────────────────────────────────


@dataclass
class GRPOStepMetrics:
    """One-step metrics record, ready to be logged or aggregated.

    Schema is the source of truth for ``grpo_metrics.jsonl`` — every field
    appears as a JSON key per line. New fields with defaults are
    backwards-compatible: older JSONL files just lack them, and the
    aggregation layer treats missing keys as zero.

    The ``milestone_*`` fields plus ``per_agent_*`` close the gaps the
    thesis tables (T1-T5) need that the original schema didn't provide.
    """

    step: int
    group_size: int
    group_mean_reward: float
    group_reward_std: float
    advantage_mean_abs: float
    surrogate_loss: float
    kl_loss: float
    total_loss: float
    fraction_clipped: float
    grad_norm: float
    milestone_fires: int = 0
    """Total milestone events across this step's group (sum of
    ``len(traj.milestone_events)`` over all scored trajectories)."""
    milestone_fire_rate: float = 0.0
    """Fraction of scored trajectories that fired ≥ 1 milestone."""
    borrowed_fraction: float = 0.0
    """Stage-4b only: fraction of the batch with ``origin_agent != None``."""
    per_agent_reward: dict = field(default_factory=dict)
    """Per-(owning_agent_id) mean reward. Keys are stringified for clean
    JSON serialisation."""
    per_agent_milestone_rate: dict = field(default_factory=dict)
    """Per-(owning_agent_id) fraction of trajectories that fired ≥ 1 milestone."""

    # ──── §A.1 extensions (thesis-side T2/T5/Hebbian-plot data) ──────────

    milestone_fires_by_chamber: dict = field(default_factory=dict)
    """``{chamber_name: count}`` across the 6 ``CHAMBERS``. Missing chambers
    map to zero. Source: each fired milestone's track in ``MILESTONE_TRACK``.
    Drives the T2 per-chamber table."""

    milestone_fires_by_id: dict = field(default_factory=dict)
    """Sparse — only milestones that fired this step appear, keyed by
    ``milestone_id``. Drives T5 time-to-first-milestone post-hoc by
    intersecting the step index with the first non-zero appearance."""

    hebbian_mean_bond: float = 0.0
    """Mean of the upper triangle of the Hebbian weight matrix W (zero
    diagonal masked). Populated from
    ``HebbianGRPOBridge.graph.get_graph_metrics()`` when active; 0 otherwise."""

    hebbian_sparsity: float = 0.0
    """Fraction of off-diagonal entries below ``HebbianConfig.sparsity_eps``
    (the "weak bond" threshold defined in the graph code). 0 when bridge
    is disabled."""

    hebbian_modularity: float = 0.0
    """Modularity proxy from ``get_graph_metrics()``. 0 when bridge is
    disabled. NB: meaningful only when the graph has formed bonds — early
    training values are close to 0 by construction."""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ──── per-batch aggregator ───────────────────────────────────────────────


def _milestone_stats(batch: list[ScoredTrajectory]) -> dict:
    """Compute milestone + borrowing + per-agent + per-chamber + per-id stats.

    Used by both single-agent and multi-agent ``_update`` paths so the
    JSONL stream has uniform schema regardless of mode. Per-agent
    breakdowns key off ``owning_agent_id`` (the borrower) so Stage-4b
    borrowed trajectories attribute to the agent training on them, not
    the agent that originally rolled them out.

    Returns a dict matching the ``GRPOStepMetrics`` field signature; the
    trainer merges this into the constructor kwargs.
    """
    if not batch:
        return {
            "milestone_fires": 0,
            "milestone_fire_rate": 0.0,
            "borrowed_fraction": 0.0,
            "per_agent_reward": {},
            "per_agent_milestone_rate": {},
            "milestone_fires_by_chamber": {ch: 0 for ch in CHAMBERS},
            "milestone_fires_by_id": {},
        }

    mid_to_chamber = _milestone_to_chamber()

    total_fires = 0
    trajectories_with_fire = 0
    borrowed = 0
    rewards_by_owner: dict[int, list[float]] = {}
    fires_by_owner: dict[int, list[int]] = {}
    fires_by_chamber: dict[str, int] = {ch: 0 for ch in CHAMBERS}
    fires_by_id: dict[str, int] = {}

    for s in batch:
        events = s.trajectory.milestone_events
        fires = len(events)
        total_fires += fires
        if fires > 0:
            trajectories_with_fire += 1
        if s.origin_agent is not None:
            borrowed += 1

        owner = s.owning_agent_id
        if owner is None:
            owner = s.trajectory.agent_id   # single-agent path fallback
        rewards_by_owner.setdefault(owner, []).append(s.reward)
        fires_by_owner.setdefault(owner, []).append(1 if fires > 0 else 0)

        # Chamber + id breakdowns — walk each event individually so we can
        # split by track / milestone. Unknown milestones (not in
        # ``MILESTONE_TRACK``) are skipped from the chamber count but still
        # counted in the per-id dict (visibility for debug).
        for ev in events:
            mid = ev.get("milestone_id")
            if not isinstance(mid, str):
                continue
            fires_by_id[mid] = fires_by_id.get(mid, 0) + 1
            chamber = mid_to_chamber.get(mid)
            if chamber and chamber in fires_by_chamber:
                fires_by_chamber[chamber] += 1

    per_agent_reward = {
        str(aid): sum(rs) / len(rs)
        for aid, rs in rewards_by_owner.items()
    }
    per_agent_milestone_rate = {
        str(aid): sum(fs) / len(fs)
        for aid, fs in fires_by_owner.items()
    }

    return {
        "milestone_fires": total_fires,
        "milestone_fire_rate": trajectories_with_fire / len(batch),
        "borrowed_fraction": borrowed / len(batch),
        "per_agent_reward": per_agent_reward,
        "per_agent_milestone_rate": per_agent_milestone_rate,
        "milestone_fires_by_chamber": fires_by_chamber,
        "milestone_fires_by_id": fires_by_id,
    }


# ──── per-step Hebbian stats ─────────────────────────────────────────────


def _hebbian_step_stats(bridge: "HebbianGRPOBridge | None") -> dict:
    """Pull mean_bond / sparsity / modularity from the live Hebbian bridge.

    Returns zeros when the bridge is None or disabled — so the JSONL
    schema is uniform across ablations (G2 has zeros; G3a/G3b/G4 have
    actual values). Failures inside ``get_graph_metrics()`` are caught
    and logged so a graph hiccup doesn't kill a training step.
    """
    zero = {
        "hebbian_mean_bond": 0.0,
        "hebbian_sparsity": 0.0,
        "hebbian_modularity": 0.0,
    }
    if bridge is None or not bridge.is_enabled():
        return zero
    try:
        metrics = bridge.graph.get_graph_metrics()
    except Exception:
        # Logged at debug level — getting here on a hot loop would flood
        # the log. The aggregation layer can detect "all zeros" runs
        # and exclude them from Hebbian plots.
        return zero
    return {
        "hebbian_mean_bond": float(metrics.get("mean_bond_strength", 0.0)),
        "hebbian_sparsity": float(metrics.get("sparsity", 0.0)),
        "hebbian_modularity": float(metrics.get("modularity_proxy", 0.0)),
    }


__all__ = [
    "CHAMBERS",
    "GRPOStepMetrics",
    "_milestone_stats",
    "_hebbian_step_stats",
]
