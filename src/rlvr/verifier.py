"""FiveChambersVerifier — pure function mapping trajectories to scalar rewards.

Composes verifiable signals:
    * milestone fires (from ``trajectory.milestone_events`` *and*
      ``trajectory.event_log`` — both schemas supported, dedup by
      ``(step, milestone_id)`` so a single fire counted twice doesn't
      double-credit)
    * per-step format reward (from the parsed action dicts on the
      trajectory; weighted small by config)
    * all-alive bonus (no death event for this agent in the window)
    * (Stage 4a) optional Hebbian reward diffusion in ``score_group``

The verifier is a *pure* function: scoring the same trajectory twice
yields identical results. No env calls, no LLM calls, no learned
components. This is the RLVR contract.

Reward lookup uses ``rlvr.reward_table.build_milestone_rewards()`` —
NEVER ``MILESTONE_TRACK`` directly, since that maps to track-name
strings, not floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlvr.action_parser import score_parsed_action
from rlvr.reward_table import build_milestone_rewards
from rlvr.trajectory import GRPOTrajectory

if TYPE_CHECKING:
    from hebbian.graph import HebbianSocialGraph


@dataclass
class VerifierConfig:
    """All-defaults config: scores milestones + format + alive bonus, no
    Hebbian diffusion. Override fields via YAML in Stage 2+.
    """

    use_milestone_rewards: bool = True
    use_format_reward: bool = True
    format_reward_weight: float = 0.1
    """Per-step format score (∈ {0, 0.5, 1}) is *summed* over the trajectory
    and multiplied by this weight. With weight=0.1 and a 50-step trajectory,
    max format reward is 5.0 — small relative to milestones (40–300) but
    dense, which is the bootstrapping role RLVR papers ascribe to it."""

    use_alive_bonus: bool = True
    alive_bonus_amount: float = 5.0
    """Flat bonus added if the trajectory has no death event for its agent.
    Encourages survival without dominating milestone rewards."""

    hebbian_reward_diffusion: bool = False
    """Stage 4a toggle. When True AND a ``HebbianSocialGraph`` is supplied
    AND the group has one trajectory per distinct agent_id matching
    ``hebbian.config.num_agents``, applies ``diffuse_rewards`` to spread
    reward across bonded teammates. Otherwise a no-op."""

    n_agents: int = 3
    """Used by ``score_parsed_action`` to bound ``communication_target``."""


class FiveChambersVerifier:
    def __init__(self, config: VerifierConfig,
                 hebbian: "HebbianSocialGraph | None" = None):
        self.config = config
        self.hebbian = hebbian
        self.milestone_rewards: dict[str, float] = build_milestone_rewards()

    def score(self, trajectory: GRPOTrajectory) -> float:
        """Scalar reward for one trajectory. Sum of components from ``explain``."""
        return sum(self.explain(trajectory).values())

    def explain(self, trajectory: GRPOTrajectory) -> dict[str, float]:
        """Decompose the reward by component. Idempotent.

        Returns a dict with stable keys ``{"milestone", "format", "alive"}``
        even when a component is disabled (value is then ``0.0``). This
        lets log dashboards key off a fixed schema.
        """
        parts = {"milestone": 0.0, "format": 0.0, "alive": 0.0}

        if self.config.use_milestone_rewards:
            parts["milestone"] = self._score_milestones(trajectory)

        if self.config.use_format_reward:
            total = 0.0
            for action_dict in trajectory.actions:
                total += score_parsed_action(action_dict, self.config.n_agents)
            parts["format"] = total * self.config.format_reward_weight

        if self.config.use_alive_bonus and not self._agent_died(trajectory):
            parts["alive"] = self.config.alive_bonus_amount

        return parts

    def score_group(self, trajectories: list[GRPOTrajectory]) -> list[float]:
        """Score a group of single-agent trajectories. Identical to mapping
        ``score`` over them.

        Stage-2 single-agent runs use this path. Multi-agent runs (Stage 3+)
        use ``score_joint_group`` instead — that's where Hebbian reward
        diffusion lives, because diffusion is per-joint and needs the full
        N-agent reward vector for one timestep, which only joint rollouts
        provide cleanly.
        """
        return [self.score(t) for t in trajectories]

    def score_joint_group(
        self,
        joints: list,
        team_reward: bool,
    ) -> list:
        """Stage-3 multi-agent scoring. Dispatches between 3A and 3B.

        Parameters
        ----------
        joints : list[rlvr.rollout_sampler.JointRollout]
            G joint rollouts. Each has ``per_agent: dict[int, (traj, tensors)]``.
        team_reward : bool
            * ``True`` (3A) — return ``list[float]`` of length G; each entry
              is the sum of per-agent rewards within that joint rollout.
            * ``False`` (3B) — return ``list[dict[int, float]]`` of length G;
              each entry maps trained ``agent_id → reward`` for that joint.

        Stage 4a integration: when ``self.config.hebbian_reward_diffusion``
        is on AND ``self.hebbian`` is provided AND ``team_reward`` is
        ``False``, applies ``hebbian.diffuse_rewards`` *within each joint*
        to spread reward across bonded teammates before returning.

        The import of ``JointRollout`` is local to avoid a circular import —
        ``rollout_sampler`` imports ``trajectory`` which we already use.
        """
        if team_reward:
            return [
                sum(self.score(traj) for _aid, (traj, _) in joint.per_agent.items())
                for joint in joints
            ]

        results: list[dict[int, float]] = []
        for joint in joints:
            raw = {aid: self.score(traj) for aid, (traj, _) in joint.per_agent.items()}
            if self.config.hebbian_reward_diffusion and self.hebbian is not None:
                raw = self._diffuse_within_joint(raw)
            results.append(raw)
        return results

    def _diffuse_within_joint(self, raw: dict[int, float]) -> dict[int, float]:
        """Apply ``HebbianSocialGraph.diffuse_rewards`` to the per-agent
        reward vector within one joint rollout. Idempotent on the input.
        """
        n = self.hebbian.config.num_agents
        ordered = [0.0] * n
        for aid, r in raw.items():
            if 0 <= aid < n:
                ordered[aid] = r
        diffused = self.hebbian.diffuse_rewards(ordered)
        return {aid: diffused[aid] for aid in raw if 0 <= aid < n}

    # ──── milestone crediting ────────────────────────────────────────────

    def _score_milestones(self, trajectory: GRPOTrajectory) -> float:
        """Sum milestone rewards credited to this trajectory's agent, deduped
        across both event streams.

        Two schemas supported:
          * Lua-written ``milestone_events.jsonl``: ``{step, agent_id,
            milestone_id, ...}`` — credit if ``agent_id == trajectory.agent_id``.
          * Python-written ``event_log.jsonl``: ``{step, type="milestone",
            id=<mid>, contributors=[...]}`` — credit if ``f"agent_{aid}"``
            appears in ``contributors``.

        Same (step, milestone_id) is counted once regardless of which stream
        carries it.
        """
        credited: set[tuple[int, str]] = set()
        total = 0.0
        agent_token = f"agent_{trajectory.agent_id}"

        for ev in trajectory.milestone_events:
            mid = ev.get("milestone_id")
            if mid not in self.milestone_rewards:
                continue
            if ev.get("agent_id") != trajectory.agent_id:
                continue
            key = (int(ev.get("step", -1)), mid)
            if key in credited:
                continue
            credited.add(key)
            total += self.milestone_rewards[mid]

        for ev in trajectory.event_log:
            if ev.get("type") not in ("milestone", "comm_milestone"):
                continue
            mid = ev.get("id") or ev.get("milestone_id")
            if mid not in self.milestone_rewards:
                continue
            contributors = ev.get("contributors") or []
            if agent_token not in contributors:
                continue
            key = (int(ev.get("step", -1)), mid)
            if key in credited:
                continue
            credited.add(key)
            total += self.milestone_rewards[mid]

        return total

    def _agent_died(self, trajectory: GRPOTrajectory) -> bool:
        """True if a death event in ``trajectory.event_log`` credits this agent.

        Schema flexibility: accepts ``type in {"death", "agent_died"}`` and
        agent identification via either ``agent_id`` int or ``contributors``
        list with ``f"agent_{aid}"`` string.
        """
        agent_token = f"agent_{trajectory.agent_id}"
        for ev in trajectory.event_log:
            if ev.get("type") not in ("death", "agent_died"):
                continue
            if ev.get("agent_id") == trajectory.agent_id:
                return True
            if agent_token in (ev.get("contributors") or []):
                return True
        return False


# ──── CLI entry point ──────────────────────────────────────────────────


def _main() -> int:
    """``python -m rlvr.verifier --score-file path`` — Stage-1 sanity tool.

    Loads a jsonl of trajectories (written by ``PassiveLoggerCallback``),
    scores each with default ``VerifierConfig``, prints one line per
    trajectory plus an aggregate summary.
    """
    import argparse
    from pathlib import Path

    from rlvr.passive_logger import load_trajectories

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--score-file", type=Path, required=True,
                        help="Path to grpo_trajectories.jsonl")
    parser.add_argument("--decompose", action="store_true",
                        help="Print {milestone, format, alive} breakdown")
    args = parser.parse_args()

    trajectories = load_trajectories(args.score_file)
    verifier = FiveChambersVerifier(VerifierConfig())

    total = 0.0
    by_agent: dict[int, list[float]] = {}
    for traj in trajectories:
        parts = verifier.explain(traj)
        score = sum(parts.values())
        total += score
        by_agent.setdefault(traj.agent_id, []).append(score)
        if args.decompose:
            print(f"agent={traj.agent_id} chamber={traj.chamber} "
                  f"steps={traj.n_steps()} score={score:.2f} "
                  f"milestone={parts['milestone']:.2f} "
                  f"format={parts['format']:.2f} "
                  f"alive={parts['alive']:.2f}")
        else:
            print(f"agent={traj.agent_id} chamber={traj.chamber} "
                  f"steps={traj.n_steps()} score={score:.2f}")

    print(f"--- summary: {len(trajectories)} trajectories, total={total:.2f}")
    for aid in sorted(by_agent):
        scores = by_agent[aid]
        mean = sum(scores) / len(scores)
        print(f"    agent_{aid}: n={len(scores)} mean={mean:.2f} "
              f"min={min(scores):.2f} max={max(scores):.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised by hand, not pytest
    raise SystemExit(_main())
