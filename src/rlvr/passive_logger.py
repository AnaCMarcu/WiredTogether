"""Stage-1 passive observer: register as an ``EpisodeLogger`` callback,
buffer per-step data and events, then on episode finalize emit one
``GRPOTrajectory`` per agent to ``runs/<run_id>/grpo_trajectories.jsonl``.

Wiring (one line in the legacy entry point):

    from rlvr.passive_logger import attach_if_enabled
    attach_if_enabled(ep_logger, run_dir)   # respects RLVR_PASSIVE_LOG env var

The wiring is opt-in: nothing happens unless the environment variable
``RLVR_PASSIVE_LOG=1`` is set OR ``attach()`` is called explicitly.

Stage-1 limitations (documented honestly so we don't confuse ourselves later):

* The Lua-written ``milestone_events.jsonl`` is **not** consumed here. All
  milestone fires come from ``event_log.jsonl`` (Python-side, with
  ``type=="milestone"`` and ``contributors=[...]``). The verifier handles
  both schemas — Stage 1 just doesn't populate the Lua field.
* Actions on the trajectory are bare strings wrapped as ``{"action": str}``
  — we don't see the original LLM JSON output here, so per-step format
  reward will be 0.5 (action valid, optional fields absent) at best.
  Stage 2's rollout sampler will produce real ``communication_target`` /
  ``thoughts`` fields.
* Trajectories are episode-bounded (start = first step, end = last step
  for each agent). Stage 2 introduces horizon-based segmentation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from rlvr.trajectory import GRPOTrajectory

logger = logging.getLogger(__name__)


class PassiveLoggerCallback:
    """An ``EpisodeLoggerCallback``. Buffers steps and events, emits one
    trajectory per agent on ``on_finalize``.

    Safe to reuse across episodes — internal buffers are cleared after
    each finalize.
    """

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._steps: list[dict] = []
        self._events: list[dict] = []

    # ──── EpisodeLoggerCallback hooks ────────────────────────────────

    def on_step(self, step, positions, actions, messages,
                task_rewards, comm_rewards, infos):
        self._steps.append({
            "step": int(step),
            "positions": _jsonable_dict(positions),
            "actions": _jsonable_dict(actions),
            "messages": _jsonable_dict(messages),
            "task_rewards": _jsonable_dict(task_rewards),
            "comm_rewards": _jsonable_dict(comm_rewards),
            "infos": _jsonable_value(infos) if infos else {},
        })

    def on_event(self, event):
        self._events.append(_jsonable_value(event))

    def on_finalize(self, summary):
        try:
            trajectories = self._build_trajectories()
            with self.output_path.open("a", encoding="utf-8") as f:
                for traj in trajectories:
                    f.write(json.dumps(trajectory_to_jsonable(traj)) + "\n")
        finally:
            # Always reset, even on write failure — otherwise the next
            # episode would double-emit.
            self._steps.clear()
            self._events.clear()

    # ──── trajectory construction ────────────────────────────────────

    def _build_trajectories(self) -> list[GRPOTrajectory]:
        if not self._steps:
            return []

        agent_ids: set[int] = set()
        for record in self._steps:
            for aid in record["actions"]:
                # Keys are str after _jsonable_dict; convert.
                try:
                    agent_ids.add(int(aid))
                except (TypeError, ValueError):
                    continue

        start_step = self._steps[0]["step"]
        end_step = self._steps[-1]["step"]
        return [self._build_one(aid, start_step, end_step) for aid in sorted(agent_ids)]

    def _build_one(self, agent_id: int, start_step: int, end_step: int) -> GRPOTrajectory:
        key = str(agent_id)
        actions: list[dict] = []
        env_outputs: list[dict] = []

        for record in self._steps:
            raw_action = record["actions"].get(key, "")
            # Wrap the bare action string in the dict shape the verifier expects.
            # Optional fields stay absent → format reward will be 0.5 per step.
            if isinstance(raw_action, dict):
                # Forward-compatibility: if upstream ever passes a dict, use it.
                actions.append(dict(raw_action))
            else:
                actions.append({"action": str(raw_action)})

            env_outputs.append({
                "position": record["positions"].get(key),
                "chamber": (record["infos"].get("chambers") or {}).get(key),
                "hp": (record["infos"].get("hp") or {}).get(key),
                "wielded": (record["infos"].get("wielded") or {}).get(key),
                "task_reward": record["task_rewards"].get(key, 0.0),
                "comm_reward": record["comm_rewards"].get(key, 0.0),
                "message": record["messages"].get(key, ""),
            })

        # Starting chamber: first non-empty value, else empty string.
        chamber = next(
            (e["chamber"] for e in env_outputs if e["chamber"]),
            "",
        )

        return GRPOTrajectory(
            prompt_id=f"agent_{agent_id}:{chamber or 'unknown'}",
            agent_id=agent_id,
            chamber=chamber,
            start_step=start_step,
            end_step=end_step,
            actions=actions,
            env_outputs=env_outputs,
            milestone_events=[],   # Stage 1: Lua stream not consumed (see module docstring)
            event_log=list(self._events),
            termination_reason="episode_end",
        )


# ──── serialization helpers ────────────────────────────────────────────


def trajectory_to_jsonable(traj: GRPOTrajectory) -> dict:
    """Convert a ``GRPOTrajectory`` to a JSON-serializable dict.

    The returned dict round-trips through ``trajectory_from_jsonable``.
    Numeric types are preserved; list-of-dict fields are passed through
    as-is (they must already be JSON-safe — the caller is responsible).
    """
    return asdict(traj)


def trajectory_from_jsonable(data: dict) -> GRPOTrajectory:
    """Inverse of ``trajectory_to_jsonable``. Tolerant of missing optional
    fields: they fall back to dataclass defaults.
    """
    return GRPOTrajectory(
        prompt_id=data["prompt_id"],
        agent_id=int(data["agent_id"]),
        chamber=data.get("chamber", ""),
        start_step=int(data["start_step"]),
        end_step=int(data["end_step"]),
        actions=list(data.get("actions") or []),
        env_outputs=list(data.get("env_outputs") or []),
        milestone_events=list(data.get("milestone_events") or []),
        event_log=list(data.get("event_log") or []),
        termination_reason=data.get("termination_reason", "horizon"),
    )


def load_trajectories(path: Path | str) -> list[GRPOTrajectory]:
    """Read a JSONL trajectories file and return ``GRPOTrajectory`` instances."""
    out: list[GRPOTrajectory] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(trajectory_from_jsonable(json.loads(line)))
    return out


# ──── opt-in wiring helper ─────────────────────────────────────────────


def attach(episode_logger, run_dir: Path | str) -> PassiveLoggerCallback:
    """Attach a ``PassiveLoggerCallback`` to ``episode_logger`` and return it.

    The output file is ``runs/<run_id>/grpo_trajectories.jsonl`` — derived
    from ``run_dir`` (which is the run's root, with ``episodes/`` underneath).
    """
    out = Path(run_dir) / "grpo_trajectories.jsonl"
    cb = PassiveLoggerCallback(out)
    episode_logger.register_callback(cb)
    return cb


def attach_if_enabled(episode_logger, run_dir: Path | str) -> PassiveLoggerCallback | None:
    """Like ``attach`` but a no-op unless ``RLVR_PASSIVE_LOG=1`` is set.

    This is the form intended for one-line wiring in the legacy entry
    point: leaves Stage-1 inactive in production until explicitly turned
    on, so existing runs are unaffected.
    """
    if not os.environ.get("RLVR_PASSIVE_LOG"):
        return None
    return attach(episode_logger, run_dir)


# ──── internal: numpy / tuple → JSON-safe ──────────────────────────────


def _jsonable_value(value):
    """Coerce common non-JSON types to JSON-safe equivalents."""
    if value is None:
        return None
    # numpy scalars / arrays
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover — defensive
            pass
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    return str(value)


def _jsonable_dict(value):
    """``_jsonable_value`` with an extra guarantee that the result is a dict
    with str keys (raising TypeError if input isn't a mapping). Used for
    the agent-id-keyed dicts EpisodeLogger passes us.
    """
    if value is None:
        return {}
    if not hasattr(value, "items"):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return {str(k): _jsonable_value(v) for k, v in value.items()}
