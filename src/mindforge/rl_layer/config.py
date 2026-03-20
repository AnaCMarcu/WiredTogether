"""Configuration for the modular RL layer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RLConfig:
    """All RL layer settings. When ``enabled=False`` the entire layer is a no-op."""

    # ── Master switch ──
    enabled: bool = False
    mode: str = "action"  # "action" (discrete MAPPO) or "token" (sequence PPO)

    # ── Model / LoRA ──
    model_path: Optional[str] = None  # path to base model (same one SGLang serves)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_per_role: bool = True  # separate adapter per role (gatherer/hunter/defender)
    lora_save_dir: str = "rl_checkpoints"
    dtype: str = "float16"  # "float16", "bfloat16", "float32"

    # ── Value head ──
    value_hidden: int = 256

    # ── PPO / MAPPO hyper-parameters ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 8
    lr: float = 3e-4

    # ── Rollout / update schedule ──
    buffer_size: int = 2048
    update_interval: int = 64  # steps between MAPPO updates

    # ── Token-level self-improvement ──
    auto_token_opt: bool = False  # let agent decide when to do token-level PPO
    token_opt_min_samples: int = 32
    token_opt_window: int = 10  # recent-action window for success-rate check
    token_opt_success_threshold: float = 0.3  # trigger when success < this
    token_opt_epochs: int = 2

    # ── Action space (must match ACTION_MAP in custom_environment_craftium.py) ──
    actions: tuple = (
        "NoOp", "MoveForward", "MoveBackward", "MoveLeft", "MoveRight",
        "Jump", "Sneak", "Dig", "Place",
        "Slot1", "Slot2", "Slot3", "Slot4", "Slot5",
        "TurnRight", "TurnLeft", "LookDown", "LookUp",
        "Inventory", "Drop", "Slot6", "Slot7", "Slot8",
    )
