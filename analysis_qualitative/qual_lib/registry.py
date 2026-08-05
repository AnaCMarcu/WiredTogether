"""Path bootstrap + run discovery for the qualitative-analysis pipeline.

Importing this module makes the repo root (for ``make_results``) and ``src/``
(for ``mindforge.*``) importable, sets the same offline guards as
``tests/conftest.py``, and re-exports the condition registry so there is a
single source of truth for condition labels and episode-slicing conventions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
for _p in (str(SRC), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Offline/env guards BEFORE any heavy import (mirrors tests/conftest.py).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import make_results  # noqa: E402  (repo root; import-safe, __main__-guarded)

CONDITIONS = make_results.CONDITIONS          # (label, dir, group, hebbian?)
MILESTONE_TRACK = make_results.MILESTONE_TRACK
PAPER_LABEL = make_results.PAPER_LABEL
COOP_TRACKS = make_results.COOP_TRACKS
ep_of_step = make_results.ep_of_step
episode_first_fire = make_results.episode_first_fire
mean_std = make_results.mean_std

DIR_TO_LABEL = {dirname: label for label, dirname, _, _ in CONDITIONS}
LABEL_TO_DIR = {label: dirname for label, dirname, _, _ in CONDITIONS}
HEBBIAN_DIRS = {dirname for _, dirname, _, heb in CONDITIONS if heb}
# Conditions whose runs carry a social_module.log (social prompt or allied
# topology controls). Detected per-run from files anyway; this is for grouping.
SOCIAL_PROMPT_DIRS = {
    "exp07_llm_2b_social_prompt",
    "exp08_llm_9b_social_prompt",
    "exp09_llm_9b_allied_all",
    "exp10_llm_9b_allied_pair",
    "exp11_llm_9b_allied_none",
    # Experiment 2 (cofire) — every arm runs --social-module prompt.
    "exp20_cofire_prc",
    "exp21_cofire_pro",
    "exp22_cofire_pri",
    "exp23_cofire_prcoi",
    "exp27_cofire_anchor",
    "exp28_cofire_null",
}

PARSER_VERSION = "2"  # bump to invalidate parse caches


@dataclass
class RunRef:
    """One (experiment, seed) run directory and what it contains."""

    exp: str                      # directory name, e.g. exp08_llm_9b_social_prompt
    seed: int
    path: Path
    label: str                    # paper label from CONDITIONS ("LLM-9B+Heb")
    degraded: bool                # missing final_metrics.json (run.log-only dir)
    episode_dirs: list = field(default_factory=list)   # sorted ep_0001.. paths
    llm_logs: dict = field(default_factory=dict)       # {module_file_stem: Path}
    is_rl: bool = False           # llm_other.log present (RL thoughts stream)
    has_social_log: bool = False

    @property
    def run_id(self) -> str:
        return f"{self.exp}/seed_{self.seed}"


def _scan_run_dir(exp: str, seed_dir: Path) -> Optional[RunRef]:
    try:
        seed = int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    ref = RunRef(
        exp=exp,
        seed=seed,
        path=seed_dir,
        label=DIR_TO_LABEL.get(exp, exp),
        degraded=not (seed_dir / "final_metrics.json").exists(),
    )
    ep_root = seed_dir / "episodes"
    if ep_root.is_dir():
        ref.episode_dirs = sorted(
            p for p in ep_root.iterdir() if p.is_dir() and p.name.startswith("ep_")
        )
    log_root = seed_dir / "llm_logs"
    if log_root.is_dir():
        ref.llm_logs = {p.stem: p for p in sorted(log_root.glob("*.log"))}
    ref.is_rl = "llm_other" in ref.llm_logs and "action_selection" not in ref.llm_logs
    ref.has_social_log = "social_module" in ref.llm_logs
    return ref


def iter_runs(runs_root: Path, only: Optional[str] = None) -> list:
    """Discover runs under runs_root, restricted to registry conditions.

    ``only`` filters to a single run, format "exp_dir/seed_N" (also accepts
    just "exp_dir" for all seeds of one condition).
    """
    runs = []
    for _, dirname, _, _ in CONDITIONS:
        exp_dir = runs_root / dirname
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            ref = _scan_run_dir(dirname, seed_dir)
            if ref is None:
                continue
            if only:
                want = only.strip("/").split("/")
                if want[0] != dirname:
                    continue
                if len(want) > 1 and want[1] != seed_dir.name:
                    continue
            runs.append(ref)
    return runs
