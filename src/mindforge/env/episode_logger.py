"""Structured per-episode logging for the Five Chambers environment.

Writes four files per episode to runs/<run_id>/episodes/ep_{N:04d}/:
  step_log.csv       — one row per (step, agent)
  event_log.jsonl    — milestone/switch/kill events as JSON objects
  messages.jsonl     — per-message metadata (sender, receiver, text, tokens,
                       routing, was-rewarded). Replaces the run-level
                       communication_log.json for fine-grained analysis.
  episode_summary.json — end-of-episode CooperationMetric summary

Callbacks (added for the RLVR/GRPO Stage-1 passive observer — see
``docs/rlvr_grpo_plan.md`` §5.1): callers may register listeners via
``register_callback()``. Each listener may implement ``on_step``,
``on_event``, ``on_finalize`` (any subset). Callback exceptions are
swallowed with a log warning so a buggy listener never corrupts the
primary on-disk log.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EpisodeLoggerCallback(Protocol):
    """Optional hooks invoked from EpisodeLogger. Implement any subset.

    All hooks are best-effort: exceptions are caught and logged, never
    propagated. Hooks receive the same data the logger writes to disk;
    they must not mutate it.
    """

    def on_step(self, step: int, positions: dict, actions: dict,
                messages: dict, task_rewards: dict, comm_rewards: dict,
                infos: dict | None) -> None: ...

    def on_event(self, event: dict) -> None: ...

    def on_finalize(self, summary: dict) -> None: ...


class EpisodeLogger:
    def __init__(self, run_dir, episode: int):
        self.dir = Path(run_dir) / "episodes" / f"ep_{episode:04d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episode = episode

        self._step_csv = open(self.dir / "step_log.csv", "w", newline="", encoding="utf-8")
        self._step_writer = csv.DictWriter(self._step_csv, fieldnames=[
            "step", "agent_id", "chamber",
            "pos_x", "pos_y", "pos_z",
            "action", "reward_task", "reward_comm",
            "wielded_item", "hp", "message",
        ])
        self._step_writer.writeheader()

        self._event_file = open(self.dir / "event_log.jsonl", "w", encoding="utf-8")
        self._messages_file = open(self.dir / "messages.jsonl", "w", encoding="utf-8")
        self._closed = False
        self._callbacks: list = []

    def log_step(self, step, positions, actions, messages, task_rewards, comm_rewards, infos=None):
        """Write one row per agent for this step.

        All dicts are keyed by integer agent_id.
        positions: {agent_id: (x,y,z) or None}
        """
        if infos is None:
            infos = {}
        chambers = infos.get("chambers", {})
        wielded = infos.get("wielded", {})
        hp = infos.get("hp", {})

        for agent_id in sorted(positions.keys()):
            pos = positions.get(agent_id)
            # `pos` may be a numpy array of length 3 — use explicit None-check
            # because `if pos:` raises on multi-element numpy arrays.
            has_pos = pos is not None and len(pos) >= 3
            self._step_writer.writerow({
                "step": step,
                "agent_id": agent_id,
                "chamber": chambers.get(agent_id, ""),
                "pos_x": pos[0] if has_pos else "",
                "pos_y": pos[1] if has_pos else "",
                "pos_z": pos[2] if has_pos else "",
                "action": actions.get(agent_id, ""),
                "reward_task": task_rewards.get(agent_id, 0.0),
                "reward_comm": comm_rewards.get(agent_id, 0.0),
                "wielded_item": wielded.get(agent_id, ""),
                "hp": hp.get(agent_id, ""),
                "message": messages.get(agent_id, ""),
            })
        # Flush so the file on disk reflects what's been logged. SLURM
        # preemption / SIGTERM / OOM can kill the process between steps;
        # without this, the partially-buffered rows for the current step
        # are lost. The cost is one fsync per env step — negligible.
        self._step_csv.flush()
        self._fire("on_step", step, positions, actions, messages,
                   task_rewards, comm_rewards, infos)

    def log_event(self, event: dict):
        """Append one JSON event to event_log.jsonl."""
        self._event_file.write(json.dumps(event) + "\n")
        self._event_file.flush()
        self._fire("on_event", event)

    def log_message(self, msg: dict):
        """Append one JSON message record to messages.jsonl.

        Expected fields:
          t                   step counter
          sender              "agent_N" or int agent id
          receiver            "agent_N" (resolved by routing) or "all"
          text                full message text
          tokens              len(text.split()) is fine for this purpose
          valid               passed CommunicationTracker validity rules
          rewarded_base       base reward earned (0.0 if invalid or capped)
          rewarded_milestone  Tier-2 chamber milestone reward (usually 0.0)
          chamber             sender's chamber at send time
          routing             "model" / "hebbian_fallback" / "random_fallback"
        """
        self._messages_file.write(json.dumps(msg) + "\n")
        self._messages_file.flush()

    def finalize(self, summary: dict):
        """Write episode_summary.json and close file handles."""
        if self._closed:
            return
        with open(self.dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        # Also write the legacy filename for back-compat with any tooling
        # that still reads episode_summary.json.
        with open(self.dir / "episode_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        # Fire on_finalize BEFORE closing file handles so callbacks that
        # need to read alongside the logger's outputs still can.
        self._fire("on_finalize", summary)
        self._step_csv.close()
        self._event_file.close()
        self._messages_file.close()
        self._closed = True

    def register_callback(self, callback) -> None:
        """Register a passive listener. ``callback`` may implement any subset
        of ``on_step``, ``on_event``, ``on_finalize`` (see
        ``EpisodeLoggerCallback``). Registration order = invocation order.
        """
        self._callbacks.append(callback)

    def _fire(self, hook: str, *args, **kwargs) -> None:
        """Call ``hook`` on every callback that implements it.

        Failures are swallowed: a buggy listener must not corrupt the
        primary on-disk log or break the env step. We log a warning so
        the failure is visible without being fatal.
        """
        for cb in self._callbacks:
            fn = getattr(cb, hook, None)
            if fn is None:
                continue
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warning("EpisodeLogger callback %s.%s failed: %s",
                               type(cb).__name__, hook, e)

    def __del__(self):
        if not self._closed:
            try:
                self._step_csv.close()
                self._event_file.close()
                self._messages_file.close()
            except Exception:
                pass
