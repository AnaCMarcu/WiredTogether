"""Configuration for the Hebbian Social Plasticity module.

All fields have defaults so that ``HebbianConfig()`` produces a disabled
no-op instance.

Note: this module was ported from the parent project's ``src/hebbian/``.
The mathematical content (equations, gating, asymmetric LTP/LTD) is
preserved verbatim. Docstrings that referenced parent-specific scenarios
have been generalised; per-environment tuning happens in YAML, not here.
"""

from dataclasses import dataclass


@dataclass
class HebbianConfig:
    """All Hebbian social plasticity settings.

    When ``enabled=False`` the entire module is a no-op.

    Note on units: ``interaction_radius`` is in **environment world units**.
    The parent project used a voxel world where 5.0 ≈ 5 blocks. For LBF
    (integer grid cells), set this to a small value in YAML — e.g. 2.0
    corresponds to Chebyshev distance ≤ 2.
    """

    # ── Master switch ──
    enabled: bool = False

    # ── Population ──
    num_agents: int = 3

    # ── Co-activity spatial gate (Eq. 2) ──
    interaction_radius: float = 5.0  # d, in env world units; override per env in YAML

    # ── Engagement signal blending (refined gi(t)) ──
    engagement_reward_weight: float = 0.5  # α

    # ── Communication co-activity bonus (refined cij) ──
    communication_coactivity_bonus: float = 0.5  # δ_comm

    # ── LTP — potentiation on positive advantage ──
    ltp_lr: float = 0.01  # η_+

    # ── LTD — depression on negative advantage (single-step) ──
    ltd_lr: float = 0.005  # η_-

    # ── LTD threshold — LTD only fires when advantage < -ltd_threshold ──
    # Why: prevents zero-reward steps (advantage ≈ small negative from
    # bootstrapped V(s)) from continuously depressing bonds during normal
    # exploration.
    ltd_threshold: float = 0.1

    # ── Base co-activity LTP — unconditional bond growth when cij > 0 ──
    # Why: fires regardless of advantage sign as `base_ltp * cij * (1-W)`.
    # Provides a floor against pure decay so co-presence alone builds bonds.
    # Without it, sparse-reward training runs decay W toward zero faster
    # than positive-advantage events can lift it.
    base_ltp: float = 0.005

    # ── Decay — passive bond flexibility ──
    # Half-life of an unsupported bond ≈ ln(2) / decay steps.
    # 0.0003 → ~2300 steps; 0.001 → ~700 steps; 0.005 → ~138 steps.
    # Default 0.0003 is calibrated for ~1500-step warm-up phases where
    # co-activation is sparse; tune per env if episodes are much shorter.
    decay: float = 0.0003  # λ

    # ── Modulation sensitivity ──
    modulation_beta: float = 1.0  # β

    # ── Sustained LTD from repeated co-failure ──
    ltd_sustained_lr: float = 0.002  # λ_F
    failure_memory_window: int = 50  # rolling window size for Fij

    # ── Failure-grace period (additive LTP bonus on co-failure) ──
    # Team-level failure steps add a per-pair LTP bonus that decays as
    # F_ij grows past failure_grace_threshold. Once F_ij crosses the
    # threshold the bonus is zero and only the existing sustained LTD
    # applies — the bond's per-step delta flips from positive ("we tried
    # together") to negative ("we keep failing together"). Enabled by
    # default: this is now part of the canonical Hebbian update rule.
    # Set failure_grace_enabled=False explicitly to reproduce the legacy
    # pre-grace math for ablation.
    failure_grace_enabled: bool = True
    failure_grace_threshold: float = 0.3  # F_ij value where grace ends
    failure_ltp_lr: float = 0.015         # bonus LTP rate (3× ltd_lr by default)

    # ── Social replay (Eq. 7) ──
    # Why disabled by default: PPO ratio exp(log_π_i - log_π_j_old) is
    # undefined for cross-agent transitions (π_j_old is not π_i's old
    # policy). Use the dedicated Hebbian-weighted-sharing learner
    # (`hebbian_seac_learner`) for that integration instead.
    social_replay_rho: float = 0.0

    # ── Reward diffusion (Eq. 8) ──
    reward_diffusion_gamma: float = 0.2  # γ

    # ── Initialisation ──
    init_weight: float = 0.1  # warm start so reward diffusion flows from step 1

    # ── Logging ──
    log_graph_every: int = 50  # steps between graph metric snapshots
