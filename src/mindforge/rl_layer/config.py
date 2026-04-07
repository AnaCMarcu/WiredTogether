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

    # ── Memory management ──
    # RL prompts can be very long (beliefs + skills + episodes).  Truncating at
    # 512 tokens is sufficient for policy learning on discrete actions and keeps
    # PPO activation memory to a manageable size during the backward pass.
    rl_prompt_max_tokens: int = 512
    gradient_checkpointing: bool = False  # LoRA rank=8 has negligible activation memory; checkpointing adds ~35% compute overhead for no benefit

    # ── PPO / MAPPO hyper-parameters ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 1.0   # was 0.2 — after RunningMeanStd normalization returns are ~[-3,+3]; ±0.2 was too tight, value head converged in 50-100 updates instead of 5-10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2           # was 4 — with update_interval=256: 2×8 mini-batches = 16 passes (was 32, causing gradient saturation)
    mini_batch_size: int = 32     # was 8 — stable gradient estimates; 8 samples/batch gave high-variance updates
    lr: float = 1e-4              # was 3e-4 — LoRA rank=8 has ~0.1% trainable params; 3e-4 overshoots the LoRA manifold

    # ── Reward shaping ──
    normalize_rewards: bool = True   # running mean/std normalisation before buffer storage
    death_penalty: float = -50.0     # added to reward when agent terminates (done=True)

    # ── Entropy annealing ──
    # Linearly decay entropy coefficient from entropy_start to entropy_end over
    # entropy_anneal_steps PPO updates.  Set entropy_anneal_steps=0 to disable.
    entropy_start: float = 0.05
    entropy_end: float = 0.001
    entropy_anneal_steps: int = 500   # PPO update steps (not env steps)

    # ── Rollout / update schedule ──
    buffer_size: int = 2048
    update_interval: int = 256  # was 64 — more diverse experience per update; avoids 50× transition reuse

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
        "Drop", "Slot6", "Slot7", "Slot8",
    )
