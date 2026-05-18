"""RolloutSampler — produce groups of G trajectories under the current policy.

Design choices (locked in per ``docs/rlvr_grpo_plan.md`` §5.2 + the answers
to the §8 open questions):

* **Equivalence class:** same chamber + position bucket (default 2-block
  grid). Trajectories starting in the same bucket are comparable for
  group-relative advantage. Trajectories in different buckets go into
  different open groups; a group is emitted only when its bucket reaches G.
* **Horizon:** up-to-H with early termination on milestone fire, death,
  or chamber transition (in addition to the env's own ``done`` signal).
* **Streaming:** rollouts are sequential. ``sample_group()`` keeps rolling
  until *some* bucket fills to G; that bucket is popped and returned. In
  a slow env this can take many rollouts — the metric to watch is
  "rollouts per group emitted."

The env interface is deliberately minimal — see ``RolloutEnv`` below. The
real PettingZoo ParallelEnv from ``marl_craftium.openworld_multi_agents``
is wrapped to match this interface in the entry point (Stage 2.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from rlvr.trajectory import GRPOTrajectory

if TYPE_CHECKING:
    import torch


@dataclass
class RolloutTensors:
    """Torch payload aligned 1:1 with a ``GRPOTrajectory``. The trainer pairs
    these into ``ScoredTrajectory`` instances after the verifier scores.

    The fields can be ``None`` for trajectories that don't carry gradient
    info (e.g. scenery-agent rollouts in single-agent Stage 2).
    """

    prompt_text: str = ""
    response_tokens: "torch.Tensor | None" = None
    response_logprobs: "torch.Tensor | None" = None


@dataclass
class SamplerConfig:
    """Hyperparameters for the rollout sampler."""

    n_per_group: int = 4
    """Group size G — number of comparable trajectories that constitute one
    GRPO group."""

    horizon: int = 50
    """Maximum trajectory length in env steps. Early termination on
    milestone fire / death / chamber transition may cut this short."""

    position_bucket_size: float = 2.0
    """Side length of the (x, z) position grid used by the equivalence
    class. 2.0 = "within 2 blocks of a fixed reference"."""

    chamber_filter: str | None = None
    """If set, only collect trajectories starting in this chamber; reset
    the env otherwise. Stage-2 default for Ch3 training: ``"ch3"``."""

    max_resets_per_group: int = 1024
    """Safety bound: ``sample_group`` raises if it can't fill a bucket
    within this many rollouts. Prevents an infinite loop when position
    buckets are too tight for the env's spatial diversity."""


class RolloutEnv(Protocol):
    """The minimal env interface the sampler needs.

    Returns observations and infos at every step; the sampler uses ``info``
    to compute the equivalence class and detect early-termination events.
    Multi-agent envs are wrapped to expose only the trained agent's view
    (see ``multi_agent_craftium_grpo.py`` for the adapter).
    """

    def reset(self) -> tuple[object, dict]:
        """Return ``(obs, info)``. ``info`` must include ``chamber``
        (string) and ``position`` (``(x, y, z)`` tuple or array)."""
        ...

    def step(self, action: dict) -> tuple[object, float, bool, dict]:
        """Step the env. ``info`` may include ``events`` (list of dicts)
        and ``milestone_events`` (list of dicts) for the verifier's
        downstream scoring AND the sampler's early-termination logic."""
        ...


class Policy(Protocol):
    """The minimal policy interface the sampler needs.

    Implementations live in ``rlvr.reference_policy`` (Stage 2.3) and are
    backed by a PEFT-adapted LLM. For tests, a stub that emits fixed JSON
    is sufficient.
    """

    def act(self, observation, info) -> tuple[dict, RolloutTensors]:
        """Return ``(action_dict, tensors)``. The action dict is what the
        env accepts (typically ``{"action": "dig", ...}``); the tensors
        carry the LLM tokens + logprobs the trainer needs."""
        ...


class RolloutSampler:
    def __init__(
        self,
        env: RolloutEnv,
        policy: Policy,
        config: SamplerConfig,
        agent_id: int = 0,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.agent_id = agent_id
        self._buckets: dict[str, list[tuple[GRPOTrajectory, RolloutTensors]]] = {}
        self._global_step = 0

    def sample_group(self) -> list[tuple[GRPOTrajectory, RolloutTensors]]:
        """Roll trajectories until one equivalence-class bucket fills to G.
        Returns the G items from that bucket (popped); other buckets retain
        their partial contents across calls.
        """
        for _ in range(self.config.max_resets_per_group):
            traj, tensors = self._sample_one()
            bucket = self._buckets.setdefault(traj.prompt_id, [])
            bucket.append((traj, tensors))
            if len(bucket) >= self.config.n_per_group:
                full = bucket[: self.config.n_per_group]
                del bucket[: self.config.n_per_group]
                return full
        raise RuntimeError(
            f"sample_group did not fill any bucket to {self.config.n_per_group} "
            f"within {self.config.max_resets_per_group} rollouts. Buckets: "
            f"{ {pid: len(b) for pid, b in self._buckets.items()} }"
        )

    def _sample_one(self) -> tuple[GRPOTrajectory, RolloutTensors]:
        """Roll one trajectory up to ``horizon``, with early termination."""
        obs, info = self.env.reset()
        chamber = str(info.get("chamber", ""))
        position = info.get("position", (0.0, 0.0, 0.0))
        prompt_id = self._make_prompt_id(chamber, position)
        start_step = self._global_step

        actions: list[dict] = []
        env_outputs: list[dict] = []
        event_log: list[dict] = []
        milestone_events: list[dict] = []
        termination_reason = "horizon"
        accumulated_prompt = ""
        accumulated_tokens = None
        accumulated_logprobs = None

        for _ in range(self.config.horizon):
            action_dict, tensors = self.policy.act(obs, info)
            actions.append(action_dict)
            accumulated_prompt += tensors.prompt_text
            accumulated_tokens = _maybe_concat(accumulated_tokens, tensors.response_tokens)
            accumulated_logprobs = _maybe_concat(accumulated_logprobs, tensors.response_logprobs)

            obs, _reward, done, info = self.env.step(action_dict)
            env_outputs.append(_jsonable_info(info))

            new_milestones = list(info.get("milestone_events") or [])
            milestone_events.extend(new_milestones)
            new_events = list(info.get("events") or [])
            event_log.extend(new_events)

            self._global_step += 1

            term = self._classify_termination(new_milestones, new_events)
            if term is not None:
                termination_reason = term
                break
            if done:
                termination_reason = "episode_end"
                break

        end_step = self._global_step - 1
        traj = GRPOTrajectory(
            prompt_id=prompt_id,
            agent_id=self.agent_id,
            chamber=chamber,
            start_step=start_step,
            end_step=end_step,
            actions=actions,
            env_outputs=env_outputs,
            milestone_events=milestone_events,
            event_log=event_log,
            termination_reason=termination_reason,
        )
        return traj, RolloutTensors(
            prompt_text=accumulated_prompt,
            response_tokens=accumulated_tokens,
            response_logprobs=accumulated_logprobs,
        )

    def _classify_termination(
        self,
        new_milestones: list[dict],
        new_events: list[dict],
    ) -> str | None:
        """Return a termination reason if any event in this step triggers
        early termination for ``self.agent_id``, else ``None``."""
        agent_token = f"agent_{self.agent_id}"
        for me in new_milestones:
            if me.get("agent_id") == self.agent_id:
                return "milestone_fired"
        for ev in new_events:
            if ev.get("type") == "milestone":
                if agent_token in (ev.get("contributors") or []):
                    return "milestone_fired"
            if ev.get("type") in ("death", "agent_died"):
                if ev.get("agent_id") == self.agent_id:
                    return "death"
                if agent_token in (ev.get("contributors") or []):
                    return "death"
            if ev.get("type") == "chamber_transition":
                if ev.get("agent_id") == self.agent_id:
                    return "chamber_transition"
        return None

    def _make_prompt_id(self, chamber: str, position) -> str:
        try:
            x = float(position[0])
            z = float(position[2])
        except (TypeError, IndexError, ValueError):
            return f"{chamber or 'unknown'}:no_pos"
        bucket = self.config.position_bucket_size
        bx = round(x / bucket) * bucket
        bz = round(z / bucket) * bucket
        chamber_token = chamber or "unknown"
        return f"{chamber_token}:x{bx:.0f}_z{bz:.0f}"


# ──── helpers ───────────────────────────────────────────────────────────


def _maybe_concat(acc, new):
    """Concatenate two torch tensors along dim 0 if both non-None.

    Imported lazily so the module doesn't require torch for non-trainer paths.
    """
    if new is None:
        return acc
    if acc is None:
        return new
    import torch  # local import — keeps the module importable without torch

    return torch.cat([acc, new], dim=0)


def _jsonable_info(info: dict) -> dict:
    """Strip an env ``info`` dict down to the JSON-safe slice the verifier
    needs. Keeps the trajectory serializable for the passive logger / disk
    roundtrip.
    """
    if not isinstance(info, dict):
        return {}
    out = {}
    for key in ("chamber", "position", "hp", "wielded", "step", "reward",
                "task_reward", "comm_reward", "message"):
        if key in info:
            value = info[key]
            if hasattr(value, "tolist") and callable(value.tolist):
                try:
                    out[key] = value.tolist()
                    continue
                except Exception:  # pragma: no cover
                    pass
            out[key] = value
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Stage 3: multi-agent sampling
# ─────────────────────────────────────────────────────────────────────────


class MultiAgentRolloutEnv(Protocol):
    """Multi-agent env interface. Keys are int agent_ids (the entry point's
    adapter is responsible for translating between PettingZoo's str-keyed
    dicts and this int-keyed interface).
    """

    def reset(self) -> tuple[dict[int, object], dict[int, dict]]:
        """``(obs_by_agent, info_by_agent)``. ``info`` per agent must include
        ``chamber`` and ``position`` for the equivalence-class computation."""
        ...

    def step(
        self, actions: dict[int, dict]
    ) -> tuple[dict[int, object], dict[int, float], dict[int, bool], dict[int, dict]]:
        """Step ALL agents simultaneously. ``info`` per agent may include
        ``events`` and ``milestone_events`` — the sampler aggregates these
        across agents (de-duplicated by ``(step, milestone_id)``)."""
        ...


@dataclass
class MultiAgentSamplerConfig:
    """Hyperparameters for the multi-agent sampler. Mostly mirrors
    ``SamplerConfig`` plus the trained-agents list and the env's total
    agent count.
    """

    n_per_group: int = 4
    horizon: int = 50
    position_bucket_size: float = 2.0
    num_agents: int = 3
    """Total agents in the env. Must match the env's PettingZoo agent count."""

    trained_agents: tuple[int, ...] = (0, 1, 2)
    """Which agents accumulate response tensors for the GRPO update. Other
    agents act under the same policy (shared LoRA) but their outputs are
    discarded for gradient purposes — they're 'scenery' from the trainer's
    POV. Stage-2 single-agent runs use ``(0,)`` and route through the
    legacy single-agent ``RolloutSampler`` instead.
    """

    max_resets_per_group: int = 1024


@dataclass
class JointRollout:
    """One joint env rollout. Maps each trained ``agent_id`` to its
    ``(trajectory, tensors)`` pair. Scenery agents are not included.
    """

    per_agent: dict[int, tuple[GRPOTrajectory, RolloutTensors]] = field(default_factory=dict)

    @property
    def trained_agent_ids(self) -> list[int]:
        return sorted(self.per_agent.keys())

    @property
    def joint_prompt_id(self) -> str:
        """Equivalence class key for the joint rollout — pipe-joined
        per-agent prompt_ids in ascending agent_id order. Two joint rollouts
        are comparable iff every trained agent started in the same chamber
        + position bucket.
        """
        parts = [self.per_agent[aid][0].prompt_id for aid in self.trained_agent_ids]
        return "|".join(parts)


class MultiAgentRolloutSampler:
    """Stage-3 sampler. Each ``sample_joint_group()`` returns G joint
    rollouts in which every trained agent has its own trajectory + tensors.

    For Stage 3 single-policy mode (all agents share one LoRA adapter), the
    sampler holds **one** ``Policy`` and calls it for every agent. For
    per-agent LoRA (a future variant), accept a dict ``policies[agent_id]``
    — not yet wired.

    Stage 4a: pass a ``HebbianGRPOBridge`` to ``hebbian_bridge`` to feed the
    Hebbian graph per env step. The bridge is optional — when ``None`` the
    sampler is identical to the Stage-3 implementation.
    """

    def __init__(
        self,
        env: MultiAgentRolloutEnv,
        policy,            # rlvr.rollout_sampler.Policy — shared across agents
        config: MultiAgentSamplerConfig,
        hebbian_bridge=None,
    ):
        self.env = env
        self.policy = policy
        self.config = config
        self.hebbian_bridge = hebbian_bridge
        self._buckets: dict[str, list[JointRollout]] = {}
        self._global_step = 0

    def sample_joint_group(self) -> list[JointRollout]:
        for _ in range(self.config.max_resets_per_group):
            rollout = self._sample_one_joint()
            bucket = self._buckets.setdefault(rollout.joint_prompt_id, [])
            bucket.append(rollout)
            if len(bucket) >= self.config.n_per_group:
                full = bucket[: self.config.n_per_group]
                del bucket[: self.config.n_per_group]
                return full
        raise RuntimeError(
            f"sample_joint_group did not fill any bucket to "
            f"{self.config.n_per_group} within {self.config.max_resets_per_group} "
            f"rollouts. Buckets: { {pid: len(b) for pid, b in self._buckets.items()} }"
        )

    def _sample_one_joint(self) -> JointRollout:
        obs_by_agent, info_by_agent = self.env.reset()
        start_step = self._global_step
        trained = self.config.trained_agents

        # Per-trained-agent buffers.
        actions: dict[int, list[dict]] = {a: [] for a in trained}
        env_outputs: dict[int, list[dict]] = {a: [] for a in trained}
        prompt_text_acc: dict[int, str] = {a: "" for a in trained}
        tokens_acc: dict[int, object] = {a: None for a in trained}
        logprobs_acc: dict[int, object] = {a: None for a in trained}
        starting_info: dict[int, dict] = {a: info_by_agent.get(a, {}) for a in trained}

        # Shared event accumulators — every trained agent's trajectory carries
        # the same event list, then the verifier filters per agent.
        event_log: list[dict] = []
        milestone_events: list[dict] = []

        termination_reason = "horizon"

        for _t in range(self.config.horizon):
            actions_this_step: dict[int, dict] = {}
            for aid in range(self.config.num_agents):
                obs = obs_by_agent.get(aid)
                info = info_by_agent.get(aid, {})
                action_dict, tensors = self.policy.act(obs, info)
                actions_this_step[aid] = action_dict

                if aid in trained:
                    actions[aid].append(action_dict)
                    prompt_text_acc[aid] += tensors.prompt_text
                    tokens_acc[aid] = _maybe_concat(tokens_acc[aid], tensors.response_tokens)
                    logprobs_acc[aid] = _maybe_concat(logprobs_acc[aid], tensors.response_logprobs)

            obs_by_agent, rewards_by_agent, done_by_agent, info_by_agent = self.env.step(actions_this_step)

            for aid in trained:
                env_outputs[aid].append(_jsonable_info(info_by_agent.get(aid, {})))

            new_milestones, new_events = _aggregate_env_events(info_by_agent, trained)
            milestone_events.extend(new_milestones)
            event_log.extend(new_events)

            # Stage 4a: feed the Hebbian graph one step of co-activity +
            # reward + communication data (every agent, not just trained).
            if self.hebbian_bridge is not None and self.hebbian_bridge.is_enabled():
                from rlvr.hebbian_grpo_bridge import comm_events_from_actions

                positions = [
                    info_by_agent.get(aid, {}).get("position")
                    for aid in range(self.config.num_agents)
                ]
                step_rewards = [
                    float(rewards_by_agent.get(aid, 0.0))
                    for aid in range(self.config.num_agents)
                ]
                comm_evs = comm_events_from_actions(actions_this_step)
                self.hebbian_bridge.observe_step(
                    positions=positions,
                    step_rewards=step_rewards,
                    comm_events=comm_evs,
                )

            self._global_step += 1

            term = self._classify_joint_termination(new_milestones, new_events, trained)
            if term is not None:
                termination_reason = term
                break
            if any(done_by_agent.get(a, False) for a in trained):
                termination_reason = "episode_end"
                break

        end_step = self._global_step - 1
        per_agent = {}
        for aid in trained:
            info0 = starting_info[aid]
            traj = GRPOTrajectory(
                prompt_id=_position_prompt_id(
                    str(info0.get("chamber", "")),
                    info0.get("position", (0.0, 0.0, 0.0)),
                    self.config.position_bucket_size,
                ),
                agent_id=aid,
                chamber=str(info0.get("chamber", "")),
                start_step=start_step,
                end_step=end_step,
                actions=actions[aid],
                env_outputs=env_outputs[aid],
                milestone_events=list(milestone_events),
                event_log=list(event_log),
                termination_reason=termination_reason,
            )
            tensors = RolloutTensors(
                prompt_text=prompt_text_acc[aid],
                response_tokens=tokens_acc[aid],
                response_logprobs=logprobs_acc[aid],
            )
            per_agent[aid] = (traj, tensors)

        return JointRollout(per_agent=per_agent)

    def _classify_joint_termination(
        self,
        new_milestones: list[dict],
        new_events: list[dict],
        trained: tuple[int, ...],
    ) -> str | None:
        """Early-terminate the joint rollout if ANY trained agent fires a
        milestone, dies, or transitions chambers. The intuition: a joint
        rollout's "interesting moment" applies to the whole team.
        """
        trained_set = set(trained)
        trained_tokens = {f"agent_{aid}" for aid in trained}
        for me in new_milestones:
            if me.get("agent_id") in trained_set:
                return "milestone_fired"
        for ev in new_events:
            ev_type = ev.get("type")
            if ev_type == "milestone":
                if trained_tokens.intersection(ev.get("contributors") or []):
                    return "milestone_fired"
            elif ev_type in ("death", "agent_died"):
                if ev.get("agent_id") in trained_set:
                    return "death"
                if trained_tokens.intersection(ev.get("contributors") or []):
                    return "death"
            elif ev_type == "chamber_transition":
                if ev.get("agent_id") in trained_set:
                    return "chamber_transition"
        return None


# ──── helpers shared with single-agent path ────────────────────────────


def _position_prompt_id(chamber: str, position, bucket: float) -> str:
    try:
        x = float(position[0])
        z = float(position[2])
    except (TypeError, IndexError, ValueError):
        return f"{chamber or 'unknown'}:no_pos"
    bx = round(x / bucket) * bucket
    bz = round(z / bucket) * bucket
    chamber_token = chamber or "unknown"
    return f"{chamber_token}:x{bx:.0f}_z{bz:.0f}"


def _aggregate_env_events(
    info_by_agent: dict[int, dict],
    trained: tuple[int, ...],
) -> tuple[list[dict], list[dict]]:
    """Collect ``milestone_events`` + ``events`` from each agent's info,
    de-duplicated by ``(step, milestone_id)`` (milestones) and the full dict
    contents (general events).

    Why aggregate across agents: the Lua mod may write a milestone fire to
    every agent's info simultaneously; we want one record per fire, not N.
    """
    milestones_seen: set[tuple] = set()
    milestones: list[dict] = []
    events_seen: set[tuple] = set()
    events: list[dict] = []
    for aid in trained:
        info = info_by_agent.get(aid, {})
        for me in info.get("milestone_events") or []:
            key = (me.get("step"), me.get("milestone_id"), me.get("agent_id"))
            if key in milestones_seen:
                continue
            milestones_seen.add(key)
            milestones.append(me)
        for ev in info.get("events") or []:
            key = (ev.get("step"), ev.get("type"), ev.get("id") or ev.get("milestone_id"))
            if key in events_seen:
                continue
            events_seen.add(key)
            events.append(ev)
    return milestones, events
