"""Save/load helpers for ``RLLayer`` checkpoints.

Layout under ``<lora_save_dir>/<adapter_name>/``:

  adapter_config.json + adapter_model.safetensors  (LoRA weights, peft format)
  action_head.pt
  value_head.pt
  rl_state.pt    — optimizer + step counters + recent-window buffers + RMS

The class methods on ``RLLayer`` are thin wrappers around these functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

from rl_layer.heads import ActionHead

if TYPE_CHECKING:
    from rl_layer.rl_layer import RLLayer

logger = logging.getLogger(__name__)


def save_rl_layer(rl: "RLLayer", path: Optional[str] = None) -> None:
    """Save LoRA adapter, heads, optimizer state, and running buffers."""
    if not rl.config.enabled:
        return
    save_dir = Path(path or rl.config.lora_save_dir) / rl._adapter_name
    save_dir.mkdir(parents=True, exist_ok=True)

    rl.model.save_pretrained(str(save_dir))
    torch.save(rl.action_head.state_dict(), save_dir / "action_head.pt")
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
    """Restore heads + optimizer + RMS + recent-window state. LoRA loaded at init."""
    if not rl.config.enabled:
        return
    load_dir = Path(path or rl.config.lora_save_dir) / rl._adapter_name

    ah_path = load_dir / "action_head.pt"
    vh_path = load_dir / "value_head.pt"
    state_path = load_dir / "rl_state.pt"

    if ah_path.exists():
        _restore_action_head(rl, ah_path)
    if vh_path.exists():
        rl.value_head.load_state_dict(
            torch.load(vh_path, map_location=rl._device, weights_only=True)
        )
    if state_path.exists():
        _restore_rl_state(rl, state_path)


def _restore_action_head(rl: "RLLayer", ah_path: Path) -> None:
    """Action head load with size-mismatch handling.

    If the checkpoint was saved with a different action count (e.g. before
    macros were added), reinitialise the head from scratch rather than crash.
    LoRA + value head are preserved.
    """
    saved_state = torch.load(ah_path, map_location=rl._device, weights_only=True)
    saved_n_actions = saved_state["net.weight"].shape[0]
    current_n_actions = len(rl.config.actions)
    if saved_n_actions != current_n_actions:
        logger.warning(
            "RLLayer agent %d: action head size mismatch — "
            "checkpoint has %d actions, config has %d. "
            "Reinitialising action head (LoRA weights preserved).",
            rl.agent_id, saved_n_actions, current_n_actions,
        )
        hidden_size = rl.model.config.hidden_size
        rl.action_head = ActionHead(hidden_size, current_n_actions).to(
            device=rl._device, dtype=torch.float32
        )
        # Rebuild optimizer with the new head's parameters.
        trainable = (
            list(filter(lambda p: p.requires_grad, rl.model.parameters()))
            + list(rl.action_head.parameters())
            + list(rl.value_head.parameters())
        )
        rl.optimizer = torch.optim.Adam(trainable, lr=rl.config.lr)
    else:
        rl.action_head.load_state_dict(saved_state)


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
