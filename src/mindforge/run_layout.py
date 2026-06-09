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


def _resolve_root(root: str | os.PathLike) -> Path:
    """Resolve a possibly-relative ``root`` against ``WIREDTOGETHER_RUNS_ROOT``.

    When the caller passes a relative path like ``"runs"`` AND the env var
    is set, anchor the relative path under the env-var directory. Lets HPC
    SLURM scripts point all outputs to ``/scratch/$USER/.../runs/`` without
    every caller having to be aware of the convention. Absolute paths and
    the case of a no-env-var run are passed through unchanged.
    """
    p = Path(root)
    if p.is_absolute():
        return p
    env_root = os.environ.get("WIREDTOGETHER_RUNS_ROOT")
    if env_root:
        env = Path(env_root)
        # If caller asked for "runs" specifically, swap to the env root
        # directly (the env var IS the runs dir). Otherwise treat the env
        # root as a parent.
        if p == Path("runs"):
            return env
        return env / p
    return p


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
        run_root = _resolve_root(root) / run_id
        rp = cls(root=run_root, run_id=run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        rp.episodes_dir.mkdir(exist_ok=True)
        rp.checkpoints_dir.mkdir(exist_ok=True)
        rp.plots_dir.mkdir(exist_ok=True)
        return rp

    @classmethod
    def create_tagged(
        cls,
        *,
        tag: str,
        seed: int,
        root: str | os.PathLike | None = None,
    ) -> "RunPaths":
        """Tagged-and-seeded layout matching the GRPO pattern.

        Produces ``runs/legacy/<tag>/seed_<seed>/`` (or ``<root>/<tag>/seed_<N>/``
        when ``root`` is given). The two-level structure makes
        ``build_results.py`` + the schema bridge auto-discover legacy
        runs with the same globbing logic used for GRPO
        (``<root>/<tag>/seed_*``).

        Idempotent: calling twice with the same ``(tag, seed)`` recreates
        the same path with the directory skeleton already in place.

        ``run_id`` is set to ``"<tag>/seed_<seed>"`` so downstream code
        that logs the run id sees the canonical tag/seed identity.

        Root resolution (when ``root`` is not given):
        1. If ``WIREDTOGETHER_RUNS_ROOT`` env var is set, use ``<env>/legacy``.
        2. Otherwise fall back to relative ``runs/legacy``.
        The env-var path is what HPC SLURM scripts use to anchor outputs
        under an absolute ``/scratch/$USER/WiredTogether/runs/`` regardless
        of cwd (``_common.sh`` cds to ``src/mindforge/`` for Python's
        sake, which would otherwise scatter outputs there).
        """
        if root is not None:
            base = Path(root)
        else:
            env_root = os.environ.get("WIREDTOGETHER_RUNS_ROOT")
            base = Path(env_root) / "legacy" if env_root else Path("runs") / "legacy"
        run_root = base / tag / f"seed_{seed}"
        run_id = f"{tag}/seed_{seed}"
        rp = cls(root=run_root, run_id=run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        rp.episodes_dir.mkdir(exist_ok=True)
        rp.checkpoints_dir.mkdir(exist_ok=True)
        rp.plots_dir.mkdir(exist_ok=True)
        return rp
