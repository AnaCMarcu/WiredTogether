"""HebbianGRPOBridge — keeps a ``HebbianSocialGraph`` up to date during
GRPO training, and exposes its normalised weights for Stage-4b group
composition.

Two integration points:

* **Stage 4a (reward diffusion):** the bridge calls ``graph.update()`` once
  per env step inside ``MultiAgentRolloutSampler._sample_one_joint``. The
  verifier's ``score_joint_group`` then optionally applies
  ``graph.diffuse_rewards`` per joint when
  ``VerifierConfig.hebbian_reward_diffusion`` is on. Already wired.

* **Stage 4b (group composition):** the trainer reads
  ``bridge.normalized_weights(i)`` to pick teammate buffers to borrow from
  when assembling agent i's group. See ``docs/rlvr_grpo_plan.md`` §5.4
  Stage 4b — Option 4b-i (clipped off-policy). Shared-LoRA mode makes the
  off-policy ratio trivially safe (π_i ≡ π_j), so no extra IS correction
  is needed in Stage 3+4b together.

The bridge is a thin wrapper — it doesn't own the graph, doesn't allocate
extra state, and is safe to construct even when the graph is disabled
(``observe_step`` becomes a no-op).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from hebbian.graph import HebbianSocialGraph

logger = logging.getLogger(__name__)


class HebbianGRPOBridge:
    def __init__(self, graph: "HebbianSocialGraph"):
        self.graph = graph
        self._step_count = 0
        """Count of ``observe_step`` calls — used for logging cadence and to
        decide when the graph has 'enough' history for borrowing to be
        meaningful."""

    # ──── Stage 4a: per-step graph update ────────────────────────────

    def observe_step(
        self,
        positions: list[Optional[tuple[float, float, float]]],
        step_rewards: list[float],
        comm_events: list[tuple[int, int]] | None = None,
        advantages: list[float] | None = None,
    ) -> None:
        """One env-step's worth of data. Forwards to ``graph.update``.

        Safe to call when the graph is disabled — silently skips. Failures
        are logged, not raised: a buggy Hebbian update must not crash the
        rollout.
        """
        if not self.is_enabled():
            return
        try:
            self.graph.update(
                positions=positions,
                step_rewards=step_rewards,
                advantages=advantages,
                comm_events=comm_events,
            )
        except Exception as e:
            logger.warning("HebbianGRPOBridge.observe_step failed: %s", e)
        self._step_count += 1

    # ──── Stage 4b: borrowing weights ────────────────────────────────

    def normalized_weights(self, agent_id: int) -> np.ndarray:
        """W̄[agent_id, :] — sampling probability for teammate buffers.

        ``W̄[agent_id, j]`` is the probability that a borrow slot in agent
        ``agent_id``'s group should be filled from teammate ``j``'s buffer.
        The diagonal is masked to zero (no self-borrow) by the underlying
        ``get_normalized_weights``.
        """
        if not self.is_enabled():
            n = self.graph.config.num_agents
            return np.zeros(n, dtype=np.float32)
        return self.graph.get_normalized_weights(agent_id)

    def weight_matrix(self) -> np.ndarray:
        """Full ``W`` matrix — for logging / plots only. Do not mutate."""
        return self.graph.get_all_weights()

    # ──── Stage 6 / §A.3: graph snapshots for time-series plots ──────

    def snapshot(self, step: int) -> dict:
        """Return a JSON-serializable snapshot of the graph state at ``step``.

        The trainer writes one of these per K GRPO steps to a
        ``hebbian_snapshots.jsonl`` sidecar. ``compare.py`` reads them to
        produce the ``bond_strength_evolution.png`` plot and the T3
        Hebbian-axis decomposition table.

        Schema (every value JSON-safe):
        ``{step, enabled, mean_bond_strength, sparsity, modularity_proxy,
           top_3_pairs, per_agent_out_strength, W, ltd_heatmap}``

        When the bridge is disabled, returns ``{step, enabled: False}``
        only — keeps the JSONL parseable in mixed-ablation aggregations.

        Failures inside ``get_graph_metrics`` are caught and reported as
        ``{step, enabled: True, error: <msg>}`` — never crashes training.
        """
        if not self.is_enabled():
            return {"step": int(step), "enabled": False}
        try:
            raw = self.graph.get_graph_metrics()
        except Exception as e:
            logger.warning("HebbianGRPOBridge.snapshot failed: %s", e)
            return {"step": int(step), "enabled": True, "error": str(e)}
        return {
            "step": int(step),
            "enabled": True,
            "mean_bond_strength": float(raw.get("mean_bond_strength", 0.0)),
            "sparsity": float(raw.get("sparsity", 0.0)),
            "modularity_proxy": float(raw.get("modularity_proxy", 0.0)),
            "top_3_pairs": _jsonable(raw.get("top_3_pairs", [])),
            "per_agent_out_strength": _jsonable(
                raw.get("per_agent_out_strength", [])),
            "W": _jsonable(raw.get("W")),
            "ltd_heatmap": _jsonable(raw.get("ltd_heatmap")),
        }

    # ──── status ────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return bool(getattr(self.graph, "config", None) and
                    getattr(self.graph.config, "enabled", False))

    def step_count(self) -> int:
        return self._step_count


# ──── JSON-safe coercion ──────────────────────────────────────────────


def _jsonable(value):
    """Coerce numpy arrays / numpy scalars to JSON-safe Python types.

    Used by ``snapshot()`` so the ``hebbian_snapshots.jsonl`` records are
    consumable by any JSON parser without numpy installed.
    """
    if value is None:
        return None
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover — defensive
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


# ──── extract comm events from action dicts ────────────────────────────


def comm_events_from_actions(
    actions_by_agent: dict[int, dict],
) -> list[tuple[int, int]]:
    """Build the ``(sender, receiver)`` pair list from per-agent action
    dicts. Used by the sampler to feed comm-bond updates to the Hebbian
    graph without making the graph parse JSON itself.
    """
    events: list[tuple[int, int]] = []
    for sender, action in actions_by_agent.items():
        if not isinstance(action, dict):
            continue
        target = action.get("communication_target")
        if target is None:
            continue
        if isinstance(target, bool):
            # ``True`` is technically int — guard against accidental booleans.
            continue
        if isinstance(target, int) and target != sender:
            events.append((sender, target))
    return events
