"""Single source of truth for run-output filesystem layout.

All artifacts for one training run live under `runs/<run_id>/`:

  runs/<run_id>/
  ├── config.json              CLI args + git commit + start ts
  ├── log.txt                  Python logging FileHandler
  ├── episodes/ep_NNNN/
  │   ├── step_log.jsonl       per-step per-agent record
  │   ├── event_log.jsonl      milestones, switches, doors, kills, damage
  │   ├── messages.jsonl       per-message metadata
  │   └── summary.json         end-of-episode summary
  ├── checkpoints/step_NNNNNN/ run_state.json + hebbian + curricula + rl
  ├── plots/                   PNGs rendered at run end
  ├── hebbian_snapshots.jsonl  one episode-end W matrix per line
  ├── final_metrics.json       consolidated run-level summary
  └── final_summary.txt        human-readable digest

This module's only job is to compute paths consistently. Every other file
should construct a RunPaths once at startup and consume its attributes;
nothing else should call `os.path.join` for run output paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPaths:
    root: Path           # runs/<run_id>/
    run_id: str

    # ── Top-level paths ──────────────────────────────────────────────
    @property
    def config_json(self) -> Path:        return self.root / "config.json"
    @property
    def log_txt(self) -> Path:            return self.root / "log.txt"
    @property
    def episodes_dir(self) -> Path:       return self.root / "episodes"
    @property
    def checkpoints_dir(self) -> Path:    return self.root / "checkpoints"
    @property
    def plots_dir(self) -> Path:          return self.root / "plots"
    @property
    def hebbian_snapshots(self) -> Path:  return self.root / "hebbian_snapshots.jsonl"
    @property
    def final_metrics_json(self) -> Path: return self.root / "final_metrics.json"
    @property
    def final_summary_txt(self) -> Path:  return self.root / "final_summary.txt"

    # Back-compat: many call sites still expect a flat "metrics folder"
    # equivalent to the old `./run_metrics/<run_id>/`. The new run root IS
    # that folder — alias it so legacy code keeps working.
    @property
    def metrics_dir(self) -> Path:        return self.root

    # ── Per-episode subpaths ─────────────────────────────────────────
    def episode_dir(self, episode: int) -> Path:
        return self.episodes_dir / f"ep_{episode:04d}"

    def step_log_jsonl(self, episode: int) -> Path:
        return self.episode_dir(episode) / "step_log.jsonl"

    def event_log_jsonl(self, episode: int) -> Path:
        return self.episode_dir(episode) / "event_log.jsonl"

    def messages_jsonl(self, episode: int) -> Path:
        return self.episode_dir(episode) / "messages.jsonl"

    def episode_summary(self, episode: int) -> Path:
        return self.episode_dir(episode) / "summary.json"

    # ── Per-checkpoint subpaths ──────────────────────────────────────
    def checkpoint_dir(self, step: int) -> Path:
        return self.checkpoints_dir / f"step_{step:06d}"

    def rl_dir(self, step: int) -> Path:
        return self.checkpoint_dir(step) / "rl"

    # ── Construction ─────────────────────────────────────────────────
    @classmethod
    def create(cls, run_id: str, root: str | os.PathLike = "runs") -> "RunPaths":
        """Build a RunPaths and create the directory skeleton on disk."""
        run_root = Path(root) / run_id
        rp = cls(root=run_root, run_id=run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        rp.episodes_dir.mkdir(exist_ok=True)
        rp.checkpoints_dir.mkdir(exist_ok=True)
        rp.plots_dir.mkdir(exist_ok=True)
        return rp
