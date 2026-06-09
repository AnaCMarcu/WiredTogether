"""Save/load helpers for ``RLLayer`` checkpoints.

Layout under ``<lora_save_dir>/<adapter_name>/``:

  adapter_config.json + adapter_model.safetensors  (LoRA weights, peft format)
  value_head.pt
  rl_state.pt    — optimizer + step counters + recent-window buffers + RMS

NOTE: ``action_head.pt`` is no longer written. The actor is the LLM itself
(constrained-generation scoring over the candidate action strings) — there
is no separate classifier head to persist. Legacy checkpoints that still
contain ``action_head.pt`` are tolerated: load logs a warning and ignores
the file rather than crashing.

The class methods on ``RLLayer`` are thin wrappers around these functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rl_layer.rl_layer import RLLayer

logger = logging.getLogger(__name__)


def save_rl_layer(rl: "RLLayer", path: Optional[str] = None) -> None:
    """Save THIS agent's LoRA adapter, value head, optimizer state, etc.

    The shared PeftModel holds adapters for every agent; we must restrict
    the save to this agent's adapter via ``selected_adapters=[name]`` so
    multi-agent checkpoints don't repeatedly write every adapter from
    every RLLayer's save() call.
    """
    if not rl.config.enabled:
        return
    save_dir = Path(path or rl.config.lora_save_dir) / rl._adapter_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Make sure THIS agent's adapter is the active one when peft serialises.
    rl.model.set_adapter(rl._adapter_name)
    try:
        rl.model.save_pretrained(
            str(save_dir),
            selected_adapters=[rl._adapter_name],
        )
    except TypeError:
        # Older PEFT versions don't support selected_adapters; fall back
        # to saving all adapters (wasteful but correct).
        rl.model.save_pretrained(str(save_dir))
    # No action_head.pt — the actor is now LLM constrained-generation.
    torch.save(rl.value_head.state_dict(), save_dir / "value_head.pt")

    rms = rl._reward_rms
    torch.save({
        "optimizer": rl.optimizer.state_dict(),
        "step_count": rl.step_count,
        "_update_count": rl._update_count,
        "_last_token_opt_step": rl._last_token_opt_step,
        "_recent_successes": list(rl._recent_successes),
        "_recent_actions": list(rl._recent_actions),
        "_recent_rewards": list(rl._recent_rewards),
        "_current_task": rl._current_task,
        "rms_mean":  rms.mean  if rms is not None else 0.0,
        "rms_var":   rms.var   if rms is not None else 1.0,
        "rms_count": rms.count if rms is not None else 1e-4,
    }, save_dir / "rl_state.pt")

    logger.info("RLLayer agent %d: saved to %s", rl.agent_id, save_dir)


def load_rl_layer(rl: "RLLayer", path: Optional[str] = None) -> None:
    """Restore value head + optimizer + RMS + recent-window state. LoRA loaded at init."""
    if not rl.config.enabled:
        return
    load_dir = Path(path or rl.config.lora_save_dir) / rl._adapter_name

    ah_path = load_dir / "action_head.pt"
    vh_path = load_dir / "value_head.pt"
    state_path = load_dir / "rl_state.pt"

    if ah_path.exists():
        # Legacy checkpoint from the pre-refactor action_head path. The
        # actor is now LLM constrained-generation — there's no classifier
        # head to restore. Log and ignore (do NOT delete the file; the
        # user may want to roll back).
        logger.warning(
            "RLLayer agent %d: legacy action_head.pt found at %s — "
            "ignoring (the actor is now LLM constrained-generation, "
            "no classifier head to restore).",
            rl.agent_id, ah_path,
        )
    if vh_path.exists():
        rl.value_head.load_state_dict(
            torch.load(vh_path, map_location=rl._device, weights_only=True)
        )
    if state_path.exists():
        _restore_rl_state(rl, state_path)


def _restore_rl_state(rl: "RLLayer", state_path: Path) -> None:
    state = torch.load(state_path, map_location=rl._device, weights_only=False)
    try:
        rl.optimizer.load_state_dict(state["optimizer"])
    except (ValueError, KeyError):
        logger.warning(
            "RLLayer agent %d: optimizer state mismatch, reinitialising.", rl.agent_id,
        )
    rl.step_count = state.get("step_count", 0)
    rl._update_count = state.get("_update_count", 0)
    rl._last_token_opt_step = state.get("_last_token_opt_step", 0)
    rl._recent_successes = list(state.get("_recent_successes", []))
    rl._recent_actions = list(state.get("_recent_actions", []))
    rl._recent_rewards = list(state.get("_recent_rewards", []))
    rl._current_task = state.get("_current_task", "Explore")
    if rl._reward_rms is not None:
        rl._reward_rms.mean  = state.get("rms_mean", 0.0)
        rl._reward_rms.var   = state.get("rms_var", 1.0)
        rl._reward_rms.count = state.get("rms_count", 1e-4)
    logger.info(
        "RLLayer agent %d: restored rl_state (step=%d, updates=%d)",
        rl.agent_id, rl.step_count, rl._update_count,
    )
