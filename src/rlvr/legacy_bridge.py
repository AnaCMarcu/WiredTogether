"""Translate legacy ``final_metrics.json`` runs into the Phase A sidecar
schema that ``build_results.py`` already consumes.

The legacy stack (MAPPO / IPPO / plain-LLM / LLM-with-Hebbian-prompt /
LLM-with-reward-propagation) writes one fat ``final_metrics.json`` per
run plus per-episode files under ``episodes/ep_*/``. The Phase A
pipeline expects four sidecars per seed:

    runs/<tag>/seed_<N>/
    ├── grpo_metrics.jsonl       (per-step records)
    ├── time_to_first.json       (milestone → first-fire step)
    ├── episode_summary.jsonl    (cooperation per joint rollout)
    └── hebbian_snapshots.jsonl  (graph state every K steps)

This module does the translation. It's a **pure function over the
legacy JSON** — no env, no LLM, no torch. The companion CLI lives at
``scripts/legacy_to_grpo_schema.py``.

Tag inference from CLI args follows the Phase B+ plan's truth table:

    M1 : no --rl, no --hebbian, no --reward-propagation
    L1 : --hebbian, no --rl, no --reward-propagation
    L2 : --hebbian, no --rl, --reward-propagation
    M2 : --rl --rl-critic-mode centralized, no --hebbian
    M3 : --rl --rl-critic-mode centralized, --hebbian
    M4 : --rl --rl-critic-mode independent, no --hebbian
    M5 : --rl --rl-critic-mode independent, --hebbian
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from rlvr.metrics_grpo import CHAMBERS

# Filled-in keys for translated step records. The GRPO-specific fields
# (KL loss, surrogate, fraction clipped, grad norm, etc.) are 0.0 — the
# legacy stack doesn't compute them. The aggregator + plot code already
# treats missing-or-zero as "diagnostic only", so this is honest.
_ZERO_FILL_KEYS: tuple[str, ...] = (
    "group_reward_std",
    "advantage_mean_abs",
    "surrogate_loss",
    "kl_loss",
    "total_loss",
    "fraction_clipped",
    "grad_norm",
    "borrowed_fraction",
    "hebbian_mean_bond",
    "hebbian_sparsity",
    "hebbian_modularity",
)


# ──── tag inference ─────────────────────────────────────────────────────


def _flag_in_args(cli_args, flag: str) -> bool:
    """Robust ``flag in cli_args`` check — accepts list-of-str or whole-string."""
    if cli_args is None:
        return False
    if isinstance(cli_args, str):
        return flag in cli_args.split()
    return flag in cli_args


def _arg_value(cli_args, flag: str, default: str | None = None) -> str | None:
    """Read the value following ``--flag <value>``. Returns ``default`` if
    the flag isn't present or has no value.
    """
    if cli_args is None:
        return default
    items = cli_args.split() if isinstance(cli_args, str) else list(cli_args)
    try:
        idx = items.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(items):
        return default
    return items[idx + 1]


def infer_tag(cli_args) -> str:
    """Classify a legacy run by its argv into one of M1/L1/L2/M2/M3/M4/M5.

    Raises ``ValueError`` when the combination is ambiguous (e.g. ``--rl``
    AND ``--reward-propagation`` — reward propagation is currently only
    defined as an LLM-only path).
    """
    rl = _flag_in_args(cli_args, "--rl")
    hebbian = _flag_in_args(cli_args, "--hebbian")
    reward_prop = _flag_in_args(cli_args, "--reward-propagation")
    critic_mode = _arg_value(cli_args, "--rl-critic-mode", default="centralized")

    if reward_prop and rl:
        # Phase B+ doesn't define this combination; the new --reward-propagation
        # is an LLM-only experiment. Fail loudly.
        raise ValueError(
            "Run mixes --rl and --reward-propagation — undefined in Phase B+"
        )

    if not rl:
        if reward_prop:
            return "L2"
        if hebbian:
            return "L1"
        return "M1"

    # rl is on.
    if critic_mode == "independent":
        return "M5" if hebbian else "M4"
    # centralized (default).
    return "M3" if hebbian else "M2"


def extract_seed(config_block: dict, default: int = 0) -> int:
    """Pull the seed out of ``final_metrics.json::config``.

    Falls back to the ``--seed`` flag in ``cli_args`` if the top-level
    ``seed`` field is missing/None. Returns ``default`` if neither is found.
    """
    seed = config_block.get("seed")
    if isinstance(seed, int):
        return seed
    cli_args = config_block.get("cli_args")
    raw = _arg_value(cli_args, "--seed")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return int(default)


# ──── per-section translators ──────────────────────────────────────────


def translate_time_to_first(final_metrics: dict) -> dict[str, int | None]:
    """Flatten ``steps_to_milestone[{track}][{milestone_id}] = step``
    into ``{milestone_id: step}``.

    Missing milestones map to ``None`` (Kaplan-Meier censoring marker).
    """
    out: dict[str, int | None] = {}
    s2m = final_metrics.get("steps_to_milestone") or {}
    for _track, milestones in s2m.items():
        if not isinstance(milestones, dict):
            continue
        for mid, step in milestones.items():
            if not isinstance(mid, str):
                continue
            out[mid] = int(step) if isinstance(step, int) else None
    return out


def translate_hebbian_snapshots(final_metrics: dict) -> list[dict]:
    """``final_metrics.graph_snapshots`` already matches the GRPO sidecar
    schema almost exactly. We add ``enabled: True`` for compatibility
    with the GRPO bridge's "skipped when disabled" record format.
    """
    out: list[dict] = []
    for snap in final_metrics.get("graph_snapshots") or []:
        if not isinstance(snap, dict):
            continue
        record = {"enabled": True, **snap}
        out.append(record)
    return out


def gather_episode_summaries(legacy_run_dir: Path | str) -> list[dict]:
    """Walk ``episodes/ep_*/`` and collect every ``episode_summary.json``
    or ``summary.json`` into a list of dicts in episode order.

    Both filenames are accepted — the legacy episode logger writes both
    for back-compat (see ``EpisodeLogger.finalize``).
    """
    legacy_run_dir = Path(legacy_run_dir)
    ep_root = legacy_run_dir / "episodes"
    if not ep_root.is_dir():
        return []
    out: list[dict] = []
    for ep_dir in sorted(ep_root.glob("ep_*")):
        path = ep_dir / "episode_summary.json"
        if not path.exists():
            path = ep_dir / "summary.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def translate_metrics_jsonl(final_metrics: dict) -> list[dict]:
    """Build one GRPO-style step record per legacy timestep snapshot.

    Reads ``timestep_data.timesteps`` + per-agent ``cumulative_returns``
    + ``total_milestones`` and produces records in the
    ``GRPOStepMetrics.as_dict()`` shape. GRPO-only fields (KL loss,
    fraction clipped, etc.) are zero-filled.

    For the milestone-fire rate: we use the per-snapshot DELTA in
    ``total_milestones`` divided by ``num_agents``. This gives a
    rate-per-agent-per-snapshot that's directly comparable to GRPO's
    ``milestone_fire_rate`` (fraction of trajectories firing ≥ 1
    milestone) in the limit where snapshots are dense.
    """
    ts = final_metrics.get("timestep_data") or {}
    steps = ts.get("timesteps") or []
    cum_returns = ts.get("cumulative_returns") or {}
    milestone_count = ts.get("milestone_count") or {}
    total_milestones = ts.get("total_milestones") or []
    num_agents = int(
        (final_metrics.get("config") or {}).get("num_agents") or 1
    )

    records: list[dict] = []
    prev_total = 0
    for i, step in enumerate(steps):
        # Mean cumulative return across agents at this snapshot.
        per_agent_at_step: dict[str, float] = {}
        rewards_at_step: list[float] = []
        for aid in range(num_agents):
            series = cum_returns.get(str(aid)) or cum_returns.get(aid) or []
            if i < len(series):
                val = float(series[i])
                rewards_at_step.append(val)
                per_agent_at_step[str(aid)] = val
        group_mean_reward = (
            sum(rewards_at_step) / len(rewards_at_step)
            if rewards_at_step else 0.0
        )

        # Per-agent milestone-fire rate.
        per_agent_fire_rate: dict[str, float] = {}
        for aid in range(num_agents):
            series = (milestone_count.get(str(aid))
                      or milestone_count.get(aid) or [])
            if i < len(series) and (max_steps := step) > 0:
                per_agent_fire_rate[str(aid)] = float(series[i]) / max(max_steps, 1)

        # Milestone fires delta this snapshot.
        if i < len(total_milestones):
            now_total = int(total_milestones[i])
        else:
            now_total = prev_total
        fires_this_snapshot = max(0, now_total - prev_total)
        prev_total = now_total

        record = {
            "step": int(step),
            "group_size": num_agents,
            "group_mean_reward": float(group_mean_reward),
            "milestone_fires": int(fires_this_snapshot),
            "milestone_fire_rate": (
                fires_this_snapshot / num_agents if num_agents > 0 else 0.0
            ),
            "per_agent_reward": per_agent_at_step,
            "per_agent_milestone_rate": per_agent_fire_rate,
            "milestone_fires_by_chamber": _empty_chamber_dict(),
            "milestone_fires_by_id": {},
        }
        for key in _ZERO_FILL_KEYS:
            record.setdefault(key, 0.0)
        records.append(record)
    return records


def _empty_chamber_dict() -> dict[str, int]:
    return {ch: 0 for ch in CHAMBERS}


# ──── orchestration ────────────────────────────────────────────────────


def translate_run(
    legacy_run_dir: Path | str,
    output_dir: Path | str,
    *,
    tag: str | None = None,
    seed: int | None = None,
) -> dict:
    """Translate one legacy run dir into the four GRPO sidecars.

    Reads ``<legacy_run_dir>/final_metrics.json``, classifies the run
    via ``infer_tag`` (unless ``tag`` is supplied), discovers the seed
    via ``extract_seed`` (unless ``seed`` is supplied), and writes the
    four sidecars under ``<output_dir>/<tag>/seed_<N>/``.

    Returns a small summary dict ``{tag, seed, n_steps, n_episodes,
    n_milestones_fired}`` for the CLI to log.
    """
    legacy_run_dir = Path(legacy_run_dir)
    output_dir = Path(output_dir)
    final_metrics_path = legacy_run_dir / "final_metrics.json"
    if not final_metrics_path.exists():
        raise FileNotFoundError(f"missing {final_metrics_path}")
    with final_metrics_path.open("r", encoding="utf-8") as f:
        final_metrics = json.load(f)

    config = final_metrics.get("config") or {}
    if tag is None:
        tag = infer_tag(config.get("cli_args"))
    if seed is None:
        seed = extract_seed(config)

    seed_dir = output_dir / tag / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Sidecar 1: per-step records.
    metrics_records = translate_metrics_jsonl(final_metrics)
    with (seed_dir / "grpo_metrics.jsonl").open("w", encoding="utf-8") as f:
        for rec in metrics_records:
            f.write(json.dumps(rec) + "\n")

    # Sidecar 2: time-to-first-milestone.
    ttf = translate_time_to_first(final_metrics)
    with (seed_dir / "time_to_first.json").open("w", encoding="utf-8") as f:
        json.dump(ttf, f, indent=2, sort_keys=True)

    # Sidecar 3: per-episode cooperation summaries.
    episodes = gather_episode_summaries(legacy_run_dir)
    with (seed_dir / "episode_summary.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    # Sidecar 4: Hebbian graph snapshots.
    hebbian_snaps = translate_hebbian_snapshots(final_metrics)
    with (seed_dir / "hebbian_snapshots.jsonl").open("w", encoding="utf-8") as f:
        for snap in hebbian_snaps:
            f.write(json.dumps(snap) + "\n")

    # Tag marker so downstream tools can re-discover.
    (seed_dir / "tag.txt").write_text(tag + "\n", encoding="utf-8")

    return {
        "tag": tag,
        "seed": seed,
        "n_steps": len(metrics_records),
        "n_episodes": len(episodes),
        "n_milestones_fired": sum(
            1 for v in ttf.values() if isinstance(v, int)
        ),
    }


def translate_directory(
    legacy_root: Path | str,
    output_dir: Path | str,
) -> list[dict]:
    """Translate every legacy run under ``legacy_root``.

    A "legacy run directory" is any directory containing
    ``final_metrics.json``. Recursively globs so it handles both
    layouts:

      * **Untagged (default)**: ``runs/<timestamp>_<id>/final_metrics.json``
        — one level deep below ``legacy_root``.
      * **Tagged (Phase B++)**: ``runs/legacy/<tag>/seed_<N>/final_metrics.json``
        — two levels deep below ``legacy_root``.

    Skips any path under ``output_dir`` to avoid re-translating the
    translator's own output (would cause an infinite loop when
    ``output_dir`` happens to live under ``legacy_root``).
    """
    legacy_root = Path(legacy_root).resolve()
    output_dir_resolved = Path(output_dir).resolve()
    out: list[dict] = []
    for fm in sorted(legacy_root.rglob("final_metrics.json")):
        try:
            fm_resolved = fm.resolve()
        except OSError:
            continue
        # Skip self-recursion: if the translator's output_dir lives
        # under legacy_root, its translated grpo_metrics.jsonl shouldn't
        # trigger another round-trip.
        if output_dir_resolved in fm_resolved.parents:
            continue
        out.append(translate_run(fm.parent, output_dir))
    return out


__all__ = [
    "infer_tag",
    "extract_seed",
    "translate_time_to_first",
    "translate_hebbian_snapshots",
    "gather_episode_summaries",
    "translate_metrics_jsonl",
    "translate_run",
    "translate_directory",
]
