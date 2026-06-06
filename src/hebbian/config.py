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
    # "legacy"          → the accreted advantage-modulator + failure-window
    #                     + grace rule (everything below the "LEGACY" banner).
    #                     Kept for reproducing prior experiments.
    # "coactivity"      → Variant A: reward-free flat-rate growth
    #                     ΔW = η₊·c·(1−W) − decay_branch. "Association graph."
    # "reward_modulated"→ Variant B (DEFAULT): three-factor growth
    #                     ΔW = (η₀ + η₊·|r_bond|/R)·c·(1−W) − decay_branch.
    # The two new variants share the co-activity gate, windowed stats and the
    # failure-gated decay branch; they differ ONLY in the growth coefficient
    # (see HebbianSocialGraph._growth_coeff). There is NO passive −λW decay in
    # either new variant — the only downward pull is the failure-gated branch,
    # so warm-start init_weight (w₀) > 0 is structurally required.
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
    # Prevents zero-reward steps (advantage ≈ small negative) from
    # continuously depressing bonds during normal exploration.
    ltd_threshold: float = 0.1

    # ── Base co-activity LTP — unconditional bond growth when cij > 0 ──
    # Fires regardless of advantage sign: base_ltp * cij * (1-W).
    # Provides a floor against pure decay so co-presence alone builds bonds.
    # Raised from 0.002 → 0.005 because in Ch1 (1500 env steps of solo
    # learning where agents are rarely co-active and milestones rarely fire)
    # the previous 0.002 floor was too weak to keep init_weight=0.1 from
    # decaying to ~0.02 by Ch2 entry — observed mean weight dropped from
    # 0.093 → 0.051 over 100 env steps in the hebbian_rewards run.
    base_ltp: float = 0.005

    # ── Decay — passive bond flexibility ──
    # Half-life of an unsupported bond ≈ ln(2) / decay steps.
    # 0.0003 → ~2300 steps; 0.001 → ~700 steps; 0.005 → ~138 steps.
    # Lowered from 0.001 → 0.0003 because Ch1 alone is 1500 env steps of
    # mostly-solo activity (no spatial co-activation, sparse milestones);
    # at decay=0.001 init_weight=0.1 falls to 0.022 by Ch2 entry, leaving
    # no warm-start signal for reward diffusion when the team finally
    # regroups. 0.0003 keeps W ≈ 0.064 through Ch1.
    decay: float = 0.0003  # λ

    # ── Modulation sensitivity ──
    modulation_beta: float = 1.0  # β

    # ── Sustained LTD from repeated co-failure ──
    ltd_sustained_lr: float = 0.002  # λ_F
    failure_memory_window: int = 50  # rolling window size for Fij; tasks take 50-100+ steps

    # ── Failure-grace period (additive LTP bonus on co-failure) ──
    # Team-level failure steps add a per-pair LTP bonus that decays as
    # F_ij grows past failure_grace_threshold. Once F_ij crosses the
    # threshold the bonus is zero and only the existing sustained LTD
    # applies — the bond's per-step delta flips from positive ("we tried
    # together") to negative ("we keep failing together"). Enabled by
    # default: canonical Hebbian update rule. Set False to reproduce the
    # legacy pre-grace math.
    failure_grace_enabled: bool = True
    failure_grace_threshold: float = 0.3  # F_ij value where grace ends
    failure_ltp_lr: float = 0.015         # bonus LTP rate (3× ltd_lr by default)

    # ── Social replay (Eq. 7) ──
    # Disabled: PPO ratio exp(log_π_i - log_π_j_old) is undefined for cross-agent transitions
    # (π_j_old is not π_i's old policy). All neighbour transitions are clipped to near-zero
    # gradient. Hebbian reward diffusion already provides the social signal via rewards.
    social_replay_rho: float = 0.0  # was 0.3 — disabled until IS correction is added

    # ── Reward diffusion (Eq. 8) ── (shared by ALL modes)
    reward_diffusion_gamma: float = 0.2  # γ

    # ════════════════════════════════════════════════════════════════════
    #  Gated-Hebbian variants (mode ∈ {"coactivity", "reward_modulated"})
    #  Shared co-activity gate + failure-gated decay; growth coeff differs.
    #  Reuses init_weight (w₀), interaction_radius (d),
    #  communication_coactivity_bonus (δ_comm), engagement_reward_weight (α),
    #  reward_diffusion_gamma (γ) from above.
    # ════════════════════════════════════════════════════════════════════

    # Growth rate η₊. With full co-activity a bond reaches ~90% saturation in
    # ~ln(0.1)/ln(1−η₊) steps (≈45 at 0.05). In Variant A this IS the flat
    # growth constant; in Variant B it scales the salience term η₊·|r_bond|/R.
    # TODO(tune): set so a bond matures within one cooperative chamber — needs
    # the typical co-active duration of a real partner pair, not given.
    eta_plus: float = 0.05

    # Association floor η₀ (Variant B only): co-presence still builds bonds on
    # zero-reward steps. Conceptually plays the same role as Variant A's η₊.
    # Small: a real milestone step should outpace a silent co-active step by a
    # few×, not thousands×.
    eta_0: float = 0.01

    # Decay rate η₋ for the failure-gated branch: decay = η₋·1[no co-activity
    # in window AND sustained loss]·W. ~half of η₊ → bonds hard-won, moderately
    # hard-lost. NOTE: this is NOT a passive −λW term; it fires only on the
    # failure gate.
    eta_minus: float = 0.025

    # "No co-activity" threshold ε for coop_ij < ε in the decay gate. Also used
    # as the activity floor: a per-step co-activity below ε counts as "no
    # co-activity", which makes the growth/decay branches strictly mutually
    # exclusive.
    coop_eps: float = 0.05

    # Rolling-window length n (steps) for coop_ij (peak co-activity) and neg_i
    # (sustained-loss indicator).
    # TODO(tune): tie to a meaningful fraction of chamber length (not given);
    # 50 mirrors the legacy failure_memory_window and "tasks take 50–100+ steps".
    coop_window: int = 50

    # Negative-reward threshold θ: neg_i fires when Σ_window r_i < −θ.
    # MUST sit between |futile penalty| (1.0) and |death penalty| (10.0) so one
    # misfired camera-tilt cannot trip decay but a death (−10) does. Default 5.0.
    # TODO(tune): confirm against the live futile/death magnitudes.
    neg_theta: float = 5.0

    # Fixed reward normalizer R for Variant B's salience term (|r_bond|/R).
    # FIXED, not a running max — a drifting R would silently decay the effective
    # η₊ over training. Default = largest bondable milestone reward
    # (m27_boss_defeated = 300 in milestones.lua); death is excluded from r_bond.
    # TODO(tune): re-check if the M-schedule changes.
    reward_norm_R: float = 300.0

    # ════════════════════════════════════════════════════════════════════
    #  Hardcoded / frozen graph (LLM-only ablation; no plasticity)
    #  Use with --social-module bias to test whether an IMPOSED (not learned)
    #  social topology, routed through communication, changes coordination.
    # ════════════════════════════════════════════════════════════════════

    # When True, update() is a no-op: W is fixed for the whole run (no LTP/LTD,
    # no decay, no co-activity). bond_weights/bond_deltas the social module
    # reads are therefore constant (Δ = 0, "STABLE"). Reward diffusion is
    # inactive under freeze — pair it with --hebbian-gamma 0 for clean returns.
    freeze_weights: bool = False

    # Named starting topology for the off-diagonal of W (applied at init and,
    # under freeze, kept for the whole run). "none" ⇒ uniform init_weight fill.
    #   "uniform" — every bond = preset_bond_strong (flat: no preferred partner;
    #               the CONTROL — bias has no informative structure to exploit).
    #   "star"    — everyone bonds strongly TO preset_hub, weakly to others
    #               (centralised: all help-requests funnel to the hub).
    #   "ring"    — i bonds strongly to (i+1) mod N, weakly to the rest
    #               (a fixed directed help chain).
    #   "pair"    — agents 0 and 1 bond strongly both ways; everyone else weak
    #               (a dyad + loner; for N=3 this is the natural split).
    init_preset: str = "none"
    preset_bond_strong: float = 0.8   # value for a "strong" hardcoded bond
    preset_bond_weak: float = 0.1     # value for a "weak" hardcoded bond
    preset_hub: int = 0               # hub agent index for the "star" preset

    # Explicit N×N matrix (list of lists), highest precedence over init_preset.
    # For full manual control / loading a saved graph. Diagonal is forced to 0.
    init_matrix: list | None = None

    # ── Initialisation ──
    # Warm-start bond w₀ — MUST be > 0 (W = 0 is a fixed point of the new
    # variants: no co-activity ⇒ no growth, and the decay branch multiplies W,
    # so W=0 never moves). Legacy default kept at 0.1; the new variants
    # recommend ~0.05 (set via --hebbian-init-weight).
    init_weight: float = 0.1

    # ── Logging ──
    log_graph_every: int = 50  # steps between graph metric snapshots
