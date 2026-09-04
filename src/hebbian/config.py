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

    # ── Social-act channels credited by co-firing (Experiment 2) ──
    # Which channel-tagged social events count toward c_ij. The default
    # ("comm",) reproduces the historical rule exactly: only messages are
    # credited, obs/imit events (if any) are invisible to the wiring rule.
    social_act_channels: tuple = ("comm",)
    # δ for the directed obs/imit terms; None → use communication_coactivity_bonus
    # (one shared δ across channels, for parity between arms).
    social_coactivity_bonus: float | None = None
    # Delivery-symmetric obs/imit ("agents that co-fire wire together"): a
    # single observation/imitation event credits BOTH directions of the pair,
    # like comm already does — paired with the router notifying the target so
    # the signal actually reaches both members. False = legacy directed terms.
    social_bidirectional: bool = False
    # Drop the (1 - spatial_gate) factor from the comm term: a message is a
    # co-firing event regardless of distance, unifying comm with obs/imit
    # (c_k = δ_k·1[event]; the clip on c_ij bounds stacking with c_spat).
    # NOTE: co-located chatty pairs then stack comm on spatial credit, which
    # AMPLIFIES the proximity+chatter effect on W. False = legacy long-range-
    # only comm ("talk across distance").
    comm_distance_free: bool = False

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

    # ── Social replay (Eq. 7 weight-gated experience sharing) ──
    # 0.0 = off (the evaluated-system default; exp30/exp31 opt in at 0.3).
    # Off-policy correction is PPO's clipped ratio π_i/π_j — see
    # rl_layer.ppo_update._collect_social_replay for the full argument.
    social_replay_rho: float = 0.0

    # ── Reward diffusion (Eq. 8) ── (shared by ALL modes)
    reward_diffusion_gamma: float = 0.2  # γ


    eta_plus: float = 0.05

    eta_0: float = 0.01

    eta_minus: float = 0.025

    coop_eps: float = 0.05

    coop_window: int = 50

    neg_theta: float = 5.0

    reward_norm_R: float = 300.0

    # ── Three-factor variant (mode = "three_factor") ────────────────────
    # Eligibility-trace decay ρ_e: e_ij ← ρ_e·e_ij + c_ij, so co-activity
    # leaves a tag with a ~1/(1−ρ_e)-step memory that a later reward
    # converts into a lasting weight change (the synaptic-tagging form of
    # the three-factor rule). Growth becomes
    #     η0·c_ij·(1−W) + η+·(|r_bond_i|/R)·e_ij·(1−W),
    # replacing reward_modulated's one-step gain on the CURRENT c_ij only.
    # Because reward is sparse the trace term cannot saturate the weights,
    # which is what lets λ (decay) be set lower in this mode so that
    # reward-earned credit persists.
    eligibility_rho: float = 0.9
    # Signed death LTD (three_factor only; audit V1 "death→blame via trace"):
    #     ΔW⁻ = η₋ᵈ·(d̃_i/R)·e_ij·W_ij,   d̃_i = min(|death_i|, death_cap)
    # routed through the SAME eligibility trace as growth, so a death blames
    # the partners the dying agent was recently co-active with — the live
    # replacement for the coop<ε ∧ neg_i branch, which the closed-loop audit
    # showed never fires (comm keeps coop ≥ ε through failure episodes).
    # 0.0 = off: every existing three_factor arm is byte-identical.
    eta_minus_death: float = 0.0
    # Cap on |death signal| before /R: would-die (−10) and real death (−50)
    # carry equal blame, and one death cannot wipe a strong bond outright.
    death_cap: float = 10.0
    # Monotone co-activity for three_factor: co-location counts at least
    # this much even for a silent pair (c_spat = S·max(g_i·g_j, floor)),
    # and the comm term drops its (1−spatial) gate, so
    # near+messaging > far+messaging > near+silent > apart+silent.
    # 0.0 restores the engagement-gated spatial term of the other modes.
    coact_floor: float = 0.25

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
