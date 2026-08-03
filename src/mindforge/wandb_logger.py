"""Thin Weights & Biases wrapper used by multi_agent_craftium.py.

Why a wrapper:
- Centralises the try/except around every wandb call so a wandb outage
  never kills a training run.
- Holds a module-level handle so the training loop doesn't have to pass
  the wandb run object through every call site.
- Honours WANDB_MODE=disabled / WANDB_API_KEY natively (wandb sees them
  before init()).

Resume across chunked SLURM jobs: wandb id is derived from the run_id
(sanitised to wandb's allowed charset) so re-launching the same
experiment+seed continues the same wandb run via resume="allow".

That resume is deliberate WITHIN a suite and wrong ACROSS suites: the
run_id is only "<tag>/seed_<N>", so a fresh suite re-running the same
exp+seed used to silently reopen the previous suite's run instead of
creating a new one (it already cost the 2000-step exp11 runs — see the
note in wandb_compute_budget.py). ``group`` namespaces the id to keep
suites apart; see :func:`_scoped_id`.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from run_layout import DEFAULT_RUN_GROUP

_log = logging.getLogger(__name__)
_WB = None  # active wandb run handle; None when disabled / init failed


def _sanitize_id(s: str) -> str:
    """wandb run ids accept [A-Za-z0-9_-], max 64 chars."""
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    return s[:64]


def _scoped_name(run_id: str, group: Optional[str]) -> str:
    """Human-facing run name, prefixed with the suite it belongs to."""
    if not group or group == DEFAULT_RUN_GROUP:
        return run_id
    return f"{group}/{run_id}"


def _scoped_id(run_id: str, group: Optional[str]) -> str:
    """wandb run id, namespaced by run group.

    ``legacy`` (and no group at all) keeps the bare ``<tag>_seed_<N>`` id so
    runs recorded before groups existed still resume into themselves. Any
    other group is prefixed, so re-running the same exp+seed under a new
    group creates a NEW wandb run rather than appending to the old suite's.
    """
    return _sanitize_id(_scoped_name(run_id, group))


def init(
    enabled: bool,
    project: str,
    entity: Optional[str],
    run_id: str,
    tags: List[str],
    config: Dict[str, Any],
    explicit_id: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """Start a wandb run. Returns True iff successfully initialised.

    No-op when ``enabled`` is False. On any failure (missing API key,
    wandb import error, network) the function logs a warning and
    returns False — the caller continues training without wandb.
    """
    global _WB
    if not enabled:
        return False
    try:
        import wandb
    except ImportError as e:
        _log.warning("[wandb] import failed: %s — disabling", e)
        return False
    try:
        wb_id = _sanitize_id(explicit_id) if explicit_id else _scoped_id(run_id, group)
        _WB = wandb.init(
            project=project,
            entity=entity,
            id=wb_id,
            resume="allow",
            name=_scoped_name(run_id, group),
            group=group or None,
            tags=tags or None,
            config=config,
        )
        url = getattr(_WB, "url", None) if _WB is not None else None
        _log.info("[wandb] init ok: project=%s group=%s id=%s url=%s",
                  project, group, wb_id, url)
        print(f"[wandb] tracking → {url} (project={project} group={group} id={wb_id})")
        return True
    except Exception as e:
        _log.warning("[wandb] init failed: %s", e)
        _WB = None
        return False


def is_active() -> bool:
    return _WB is not None


def log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a metrics dict at the given step. No-op if wandb inactive."""
    if _WB is None:
        return
    try:
        if step is not None:
            _WB.log(metrics, step=step)
        else:
            _WB.log(metrics)
    except Exception as e:
        _log.warning("[wandb] log failed: %s", e)


def upload_file(
    path: str,
    name: Optional[str] = None,
    artifact_type: str = "result",
) -> None:
    """Upload ``path`` as a wandb artifact. No-op if wandb inactive
    or the file does not exist."""
    if _WB is None:
        return
    if not os.path.exists(path):
        _log.warning("[wandb] upload_file: missing %s", path)
        return
    try:
        import wandb
        artifact_name = _sanitize_id(name or os.path.basename(path))
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_file(path)
        _WB.log_artifact(artifact)
    except Exception as e:
        _log.warning("[wandb] upload_file failed (%s): %s", path, e)


def log_rl_update(agent_id: int, info: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a single record_rl_update() event to W&B.

    Two info shapes are supported:
    - Centralized-critic update (signalled by ``critic_loss`` key, agent_id=-1):
      namespace ``rl/critic/*`` (the ``critic_`` prefix is dropped under that
      namespace to avoid ``rl/critic/critic_loss``).
    - Per-agent MAPPO action-head update (anything else): namespace
      ``rl/mappo/agent_{i}/*``.
    """
    if _WB is None or not info:
        return
    try:
        payload: Dict[str, Any] = {}
        if "critic_loss" in info:
            critic_keys = (
                "critic_loss",
                "critic_explained_variance",
                "critic_returns_mean",
                "critic_returns_std",
                "critic_returns_min",
                "critic_returns_max",
                "critic_values_mean",
                "critic_values_std",
                "critic_buffer_size",
                "critic_update_count",
            )
            for k in critic_keys:
                if k in info:
                    short = k.replace("critic_", "", 1)
                    payload[f"rl/critic/{short}"] = float(info[k])
        else:
            mappo_keys = (
                "policy_loss",
                "entropy",
                "approx_kl",
                "clip_frac",
                "ratio_max",
                "adv_mean",
                "adv_std",
                "adv_min",
                "adv_max",
                "frac_pos_advantage",
            )
            for k in mappo_keys:
                if k in info:
                    payload[f"rl/mappo/agent_{agent_id}/{k}"] = float(info[k])
        if payload:
            log(payload, step=step)
    except Exception as e:
        _log.warning("[wandb] log_rl_update failed: %s", e)


_TOKEN_OPT_COUNTS: Dict[tuple, int] = {}


def log_rl_token_opt(agent_id: int, info: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a token-optimisation decision event.

    Maintains a running counter per (agent, decision) so the W&B chart
    shows cumulative train/skip/etc. counts as monotonically-increasing
    curves. Also logs the latest reason as a string for table inspection.
    """
    if _WB is None or not info:
        return
    try:
        decision = str(info.get("decision", "unknown"))
        key = (agent_id, decision)
        _TOKEN_OPT_COUNTS[key] = _TOKEN_OPT_COUNTS.get(key, 0) + 1
        payload = {
            f"rl/token_opt/agent_{agent_id}/count_{decision}": _TOKEN_OPT_COUNTS[key],
        }
        log(payload, step=step)
    except Exception as e:
        _log.warning("[wandb] log_rl_token_opt failed: %s", e)


def finish() -> None:
    """Flush and close the wandb run. Safe to call when uninitialised."""
    global _WB
    if _WB is None:
        return
    try:
        _WB.finish()
    except Exception as e:
        _log.warning("[wandb] finish failed: %s", e)
    _WB = None
