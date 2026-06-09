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

    # ── Update-rule selector ───────────────────────────────────────────
    mode: str = "reward_modulated"

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

    # ── LTD threshold — LTD only fires when At_ij < -ltd_threshold ──
    ltd_threshold: float = 0.1

    # ── Base co-activity LTP — unconditional bond growth when cij > 0 ──
    base_ltp: float = 0.005

    # ── Decay — passive bond flexibility ──
    decay: float = 0.0003  # λ

    # ── Modulation sensitivity ──
    modulation_beta: float = 1.0  # β

    # ── Sustained LTD from repeated co-failure ──
    ltd_sustained_lr: float = 0.002  # λ_F
    failure_memory_window: int = 50  # rolling window size for Fij; tasks take 50-100+ steps

    # ── Failure-grace period (additive LTP bonus on co-failure) ──
    failure_grace_enabled: bool = True
    failure_grace_threshold: float = 0.3  # F_ij value where grace ends
    failure_ltp_lr: float = 0.015         # bonus LTP rate (3× ltd_lr by default)

    # ── Social replay (Eq. 7) ──
    social_replay_rho: float = 0.0  # was 0.3 — disabled until IS correction is added

    # ── Reward diffusion (Eq. 8) ── (shared by ALL modes)
    reward_diffusion_gamma: float = 0.2  # γ


    eta_plus: float = 0.05

    eta_0: float = 0.01

    eta_minus: float = 0.025

    coop_eps: float = 0.05

    coop_window: int = 50

    neg_theta: float = 5.0

    reward_norm_R: float = 300.0


    freeze_weights: bool = False

    init_preset: str = "none"
    preset_bond_strong: float = 0.8   # value for a "strong" hardcoded bond
    preset_bond_weak: float = 0.1     # value for a "weak" hardcoded bond
    preset_hub: int = 0               # hub agent index for the "star" preset

    init_matrix: list | None = None

    # ── Initialisation ──
    init_weight: float = 0.1

    # ── Logging ──
    log_graph_every: int = 50  # steps between graph metric snapshots
