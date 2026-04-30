"""Structured per-episode logging for the Five Chambers environment.

Writes three files per episode to run_metrics/{run_id}/episodes/ep_{N:04d}/:
  step_log.csv       — one row per (step, agent)
  event_log.jsonl    — milestone/switch/kill events as JSON objects
  episode_summary.json — end-of-episode CooperationMetric summary
"""

import csv
import json
from pathlib import Path


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
        self._closed = False

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
            self._step_writer.writerow({
                "step": step,
                "agent_id": agent_id,
                "chamber": chambers.get(agent_id, ""),
                "pos_x": pos[0] if pos else "",
                "pos_y": pos[1] if pos else "",
                "pos_z": pos[2] if pos else "",
                "action": actions.get(agent_id, ""),
                "reward_task": task_rewards.get(agent_id, 0.0),
                "reward_comm": comm_rewards.get(agent_id, 0.0),
                "wielded_item": wielded.get(agent_id, ""),
                "hp": hp.get(agent_id, ""),
                "message": messages.get(agent_id, ""),
            })

    def log_event(self, event: dict):
        """Append one JSON event to event_log.jsonl."""
        self._event_file.write(json.dumps(event) + "\n")

    def finalize(self, summary: dict):
        """Write episode_summary.json and close file handles."""
        if self._closed:
            return
        with open(self.dir / "episode_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._step_csv.close()
        self._event_file.close()
        self._closed = True

    def __del__(self):
        if not self._closed:
            try:
                self._step_csv.close()
                self._event_file.close()
            except Exception:
                pass
