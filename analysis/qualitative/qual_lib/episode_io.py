"""Bug-aware readers for per-episode artifacts.

IMPORTANT: all cooperation data must come through ``read_cooperation_metrics``.
The real pipeline nests CooperationMetric output under the top-level key
``"cooperation_metrics"`` in episode_summary.json (multi_agent_craftium.py
finalize call), while ``coop_eval.py`` reads those keys at top level — which is
why ``final_metrics.json["coop_metrics"]`` pair/dwell/damage values are silently
zero in every real run. Never read those from final_metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from qual_lib import registry  # noqa: F401  (path bootstrap side-effect)
from mindforge.agent_modules.comm_eval import _read_jsonl, _read_step_csv

# Some runs log a degenerate multi-hundred-KB message into a step_log.csv
# field (observed: exp06_ippo_hebbian/seed_456) — the default 128 KB csv
# field limit aborts the read. Raise it process-wide.
csv.field_size_limit(10_000_000)

# Files a stage must never read (huge / truncated / redundant).
DENYLIST = {"log.txt", "run.log", "communication_log.json"}


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def read_cooperation_metrics(ep_dir: Path) -> dict:
    """CooperationMetric summary for one episode, nesting-bug aware.

    Prefers episode_summary.json; falls back to summary.json (byte-identical
    twin). Accepts both the real nested layout and the flat layout the test
    fixtures use, so synthetic fixtures keep working.
    """
    data = _read_json(ep_dir / "episode_summary.json")
    if data is None:
        data = _read_json(ep_dir / "summary.json")
    if not isinstance(data, dict):
        return {}
    nested = data.get("cooperation_metrics")
    if isinstance(nested, dict):
        return nested
    return data  # flat layout (synthetic fixtures / future schema fix)


def read_episode_header(ep_dir: Path) -> dict:
    """Top-level episode fields: episode, final_step, total_reward_per_agent."""
    data = _read_json(ep_dir / "episode_summary.json") or _read_json(
        ep_dir / "summary.json"
    )
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k != "cooperation_metrics"}


def read_messages(ep_dir: Path) -> list:
    """messages.jsonl, deduped on (t, sender) keeping the LAST record.

    Relaunched jobs can APPEND a second attempt's records to the same file
    (observed: exp11_llm_9b_allied_none/seed_456); the last attempt is the one
    whose episode summary/final_metrics survived.
    """
    recs = _read_jsonl(ep_dir / "messages.jsonl")
    dedup = {}
    for m in recs:
        dedup[(m.get("t"), m.get("sender"))] = m
    return list(dedup.values())


def read_events(ep_dir: Path) -> list:
    """event_log.jsonl with exact-duplicate records collapsed (appended
    attempts re-log the same milestones). Order-preserving."""
    seen = set()
    out = []
    for e in _read_jsonl(ep_dir / "event_log.jsonl"):
        key = json.dumps(e, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def read_steps(ep_dir: Path) -> list:
    """step_log.csv rows with numeric fields coerced.

    Columns: step, agent_id, chamber, pos_x/y/z, action, reward_task,
    reward_comm, wielded_item, hp, message.
    """
    raw = _read_step_csv(ep_dir / "step_log.csv")
    # Dedupe (step, agent) keeping the LAST occurrence — appended attempts
    # (see read_messages) duplicate the whole step stream.
    dedup = {}
    for r in raw:
        dedup[(r.get("step"), r.get("agent_id"))] = r
    rows = list(dedup.values())
    for r in rows:
        for k in ("step", "agent_id"):
            try:
                r[k] = int(float(r.get(k, -1)))
            except (TypeError, ValueError):
                r[k] = -1
        for k in ("pos_x", "pos_y", "pos_z", "reward_task", "reward_comm"):
            try:
                r[k] = float(r.get(k)) if r.get(k) not in (None, "") else None
            except (TypeError, ValueError):
                r[k] = None
    return rows


def valid_episode_dirs(run) -> list:
    """Episode dirs trimmed to final_metrics.episode_lengths when available.

    Relaunches can leave EXTRA ep_000N dirs from an aborted longer attempt
    (observed: exp06/seed_456, exp11/seed_42+123 have 4 dirs for 3 episodes).
    """
    fm = read_final_metrics(run)
    n = len((fm or {}).get("episode_lengths", []))
    if n and len(run.episode_dirs) > n:
        return run.episode_dirs[:n]
    return run.episode_dirs


def per_agent_step_counts(run) -> list:
    """Per-episode {agent_idx: n_step_rows}, ordered by episode dir.

    step_log has exactly one row per (step, agent) while the agent is alive,
    so this is the robust per-agent step count (team episode_lengths is not,
    when an agent dies early).
    """
    out = []
    for ep_dir in valid_episode_dirs(run):
        counts: dict = {}
        for r in read_steps(ep_dir):
            a = r["agent_id"]
            counts[a] = counts.get(a, 0) + 1
        out.append(counts)
    return out


def read_hebbian_snapshots(run) -> list:
    return _read_jsonl(run.path / "hebbian_snapshots.jsonl")


def read_final_metrics(run):
    return _read_json(run.path / "final_metrics.json")


def read_config(run) -> dict:
    return _read_json(run.path / "config.json") or {}


def module_intervals(run) -> dict:
    """belief/critic/social cadences for the turn-attribution validator."""
    cfg = read_config(run)
    cli = cfg.get("cli_args") or {}
    return {
        "belief_interval": int(cli.get("belief_interval", 5) or 5),
        "critic_interval": int(cli.get("critic_interval", 20) or 20),
        "social_interval": int(cli.get("social_interval", 8) or 8),
        "num_agents": int(cli.get("num_agents", 3) or 3),
    }
