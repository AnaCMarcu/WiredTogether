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
    # Disabled: PPO ratio exp(log_π_i - log_π_j_old) is undefined for cross-agent transitions
    # (π_j_old is not π_i's old policy). All neighbour transitions are clipped to near-zero
    # gradient. Hebbian reward diffusion already provides the social signal via rewards.
    social_replay_rho: float = 0.0  # was 0.3 — disabled until IS correction is added

    # ── Reward diffusion (Eq. 8) ──
    reward_diffusion_gamma: float = 0.2  # γ

    # ── Initialisation ──
    init_weight: float = 0.1  # initial bond strength (0.1 = warm start: reward diffusion flows from step 1)

    # ── Logging ──
    log_graph_every: int = 50  # steps between graph metric snapshots
