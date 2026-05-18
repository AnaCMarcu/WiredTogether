"""GRPOTrajectory — frozen record of one rollout segment.

A trajectory is the unit the verifier scores and the unit GRPO computes
advantages over. It is reconstructed from EpisodeLogger callbacks (Stage 1)
or sampled directly from the policy (Stage 2+).

The dataclass is ``frozen=True`` for immutability; we deliberately do NOT
override ``__hash__`` because the field types (``list[dict]``) are unhashable.
If a content-fingerprint is ever needed (e.g. for a result cache),
serialise to canonical JSON first — don't try to hash the instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Open set — accept arbitrary strings, but encourage these names for the
# common cases. Verifier / sampler code branches only on these values.
TerminationReason = Literal[
    "horizon",              # hit max trajectory length
    "milestone_fired",      # early termination on a configured milestone
    "death",                # the trajectory's agent died
    "death_all",            # the whole team wiped
    "chamber_transition",   # agent moved into a new chamber
    "episode_end",          # env episode terminated naturally
    "abandoned",            # trajectory cut short (run interrupted, etc.)
]


@dataclass(frozen=True)
class GRPOTrajectory:
    """One rollout segment under a single policy.

    Fields are the minimum the verifier needs to compute a scalar reward
    *and* the minimum the GRPO trainer needs to recompute logprobs and apply
    its loss. Anything beyond that should live in run-level metadata, not
    in every trajectory.
    """

    prompt_id: str
    """Identifier for the starting condition (chamber + position bucket etc).
    Used as the ``equivalence_class`` key by the rollout sampler when
    assembling groups of G. Two trajectories sharing a prompt_id are
    comparable for group-relative advantage."""

    agent_id: int
    """The agent whose actions populate ``actions``. For team-reward (3A)
    trajectories this is a representative; for per-agent (3B) it identifies
    the credit recipient."""

    chamber: str
    """Chamber the trajectory took place in (``ch1`` … ``ch5``). When the
    agent crosses chambers mid-trajectory, this is the *starting* chamber;
    the crossing is recorded in ``env_outputs`` and may trigger
    ``termination_reason == "chamber_transition"``."""

    start_step: int
    """Global env step at trajectory start (inclusive)."""

    end_step: int
    """Global env step at trajectory end (inclusive). ``end_step -
    start_step + 1`` is the trajectory length in env steps."""

    actions: list[dict] = field(default_factory=list)
    """Parsed LLM JSON outputs, one per env step. Each dict has at minimum
    an ``"action"`` key (str in ``_DISCRETE_ACTIONS`` ∪ {"nop"}) and may
    include ``"communication_target"``, ``"thoughts"``, etc."""

    env_outputs: list[dict] = field(default_factory=list)
    """Per-step env info aligned 1:1 with ``actions``. Includes positions,
    chambers, hp, wielded item, comm rewards. Source: step rows passed to
    ``EpisodeLogger.log_step()`` (Stage 1)."""

    milestone_events: list[dict] = field(default_factory=list)
    """Milestone fires within ``[start_step, end_step]``. Source:
    ``<world_dir>/milestone_events.jsonl``, written by the five-chambers Lua
    mod. Schema: ``{step, agent_id, milestone_id, ...}``."""

    event_log: list[dict] = field(default_factory=list)
    """Episode-logger events within ``[start_step, end_step]``. Source:
    ``runs/.../episodes/ep_*/event_log.jsonl``, written by
    ``EpisodeLogger.log_event()``. Disjoint from ``milestone_events``:
    these are Python-emitted (switches, deaths, kills, damage)."""

    termination_reason: str = "horizon"
    """One of ``TerminationReason``. Open string for forward-compat."""

    def n_steps(self) -> int:
        return self.end_step - self.start_step + 1
