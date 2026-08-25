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

    # ── Value head (per-agent IPPO baseline; reused in 'independent' critic mode) ──
    value_hidden: int = 256

    # ── Centralized critic (MAPPO) ──
    critic_mode: str = "centralized"
    critic_hidden: int = 256
    critic_lr: float = 3e-4
    critic_value_clip_eps: float = 10.0

    # ── Memory management ──
    rl_prompt_max_tokens: int = 256  # was 512 — halving sequence length reduces attention memory ~4×; 256 tokens is enough for task+status+beliefs
    gradient_checkpointing: bool = True  # re-enabled: serving model + RL model share the 80GB A100, leaving ~1.5GB free; checkpointing trades 35% compute for ~60% activation memory reduction

    # ── PPO / MAPPO hyper-parameters ──
    gamma: float = 0.995  # was 0.99 — extends effective credit horizon from ~100 to ~200 steps; needed since tasks take 50-100 steps before a completion reward fires
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 1.0   # was 0.2 — after RunningMeanStd normalization returns are ~[-3,+3]; ±0.2 was too tight, value head converged in 50-100 updates instead of 5-10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 2           # was 4 — with update_interval=256: 2×8 mini-batches = 16 passes (was 32, causing gradient saturation)
    mini_batch_size: int = 4      # was 32 — reduced to fit PPO backward pass in ~1.5GB headroom on shared A100 (serving model uses ~77GB)
    lr: float = 1e-4              # was 3e-4 — LoRA rank=8 has ~0.1% trainable params; 3e-4 overshoots the LoRA manifold

    # ── Reward shaping ──
    normalize_rewards: bool = True   # running mean/std normalisation before buffer storage
    death_penalty: float = 0.0       # added to reward when agent terminates (done=True)

    # ── Entropy annealing ──
    entropy_start: float = 0.05
    entropy_end: float = 0.001
    entropy_anneal_steps: int = 500   # PPO update steps (not env steps)

    # ── Rollout / update schedule ──
    buffer_size: int = 2048
    update_interval: int = 128
    # Stagger per-agent updates by agent_id steps (agent i updates at
    # interval+i) so the env is stepped BETWEEN updates instead of idling
    # for one long 3×update pause. Motivation: on Gemma E4B each agent's
    # PPO update takes ~10-14 min; the resulting ~40-min env idle window
    # hung the Minetest bridge within 1-2 steps afterwards in the 2026-08
    # gemma4 exp03 runs (seed_123 at step 65, seed_42 at step 129), while
    # Qwen's ~20-min windows never hung. Off by default: the completed
    # Qwen suites ran unstaggered and their gap-fill seeds must match.
    update_stagger: bool = False

    # ── Token-level self-improvement ──
    auto_token_opt: bool = False  # let agent decide when to do token-level PPO
    token_opt_min_samples: int = 32
    token_opt_window: int = 10  # recent-action window for success-rate check
    token_opt_success_threshold: float = 0.3  # trigger when success < this
    token_opt_epochs: int = 2

    # ── Action space (must match VALID_ACTIONS in custom_environment_craftium.py) ──
    actions: tuple = (
        "NoOp", "MoveForward", "MoveBackward", "MoveLeft", "MoveRight",
        "Jump", "Sneak", "Dig", "Place",
        "Slot1", "Slot2", "Slot3", "Slot4", "Slot5",
        "TurnRight", "TurnLeft", "LookDown", "LookUp",
        "Drop", "Slot6", "Slot7", "Slot8",
    )

    mask_slot_actions: bool = True
