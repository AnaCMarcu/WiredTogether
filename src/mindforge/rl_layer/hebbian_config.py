"""Configuration for the Hebbian Social Plasticity module.

All fields have defaults so that ``HebbianConfig()`` produces a disabled
no-op instance — mirroring the RLConfig pattern.
"""

from dataclasses import dataclass


@dataclass
class HebbianConfig:
    """All Hebbian social plasticity settings.

    When ``enabled=False`` the entire module is a no-op.
    """

    # ── Master switch ──
    enabled: bool = False

    # ── Population ──
    num_agents: int = 3

    # ── Co-activity spatial gate (Eq. 2) ──
    interaction_radius: float = 5.0  # d, in Minetest world units

    # ── Engagement signal blending (refined gi(t)) ──
    engagement_reward_weight: float = 0.5  # α

    # ── Communication co-activity bonus (refined cij) ──
    communication_coactivity_bonus: float = 0.5  # δ_comm

    # ── LTP — potentiation on positive advantage ──
    ltp_lr: float = 0.01  # η_+

    # ── LTD — depression on negative advantage (single-step) ──
    ltd_lr: float = 0.005  # η_-

    # ── Decay — passive bond flexibility ──
    decay: float = 0.005  # λ

    # ── Modulation sensitivity ──
    modulation_beta: float = 1.0  # β

    # ── Sustained LTD from repeated co-failure ──
    ltd_sustained_lr: float = 0.002  # λ_F
    failure_memory_window: int = 50  # rolling window size for Fij; tasks take 50-100+ steps

    # ── Social replay (Eq. 7) ──
    social_replay_rho: float = 0.3  # ρ, blend own vs neighbour buffers

    # ── Reward diffusion (Eq. 8) ──
    reward_diffusion_gamma: float = 0.2  # γ

    # ── Initialisation ──
    init_weight: float = 0.1  # initial bond strength (0.1 = warm start: reward diffusion flows from step 1)

    # ── Logging ──
    log_graph_every: int = 50  # steps between graph metric snapshots
