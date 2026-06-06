"""Unit tests pinning the gated-Hebbian social-graph variants.

Covers the invariants in the design spec for the two toggleable variants:
``mode="coactivity"`` (Variant A, reward-free growth) and
``mode="reward_modulated"`` (Variant B, three-factor growth). Both share the
co-activity gate, the windowed statistics and the failure-gated decay branch;
they differ only in the growth coefficient.

These pin the forward rule (pure NumPy, no autograd). The legacy mode is not
exercised here — it is unchanged and predates this rule.
"""

import numpy as np
import pytest

from hebbian.config import HebbianConfig
from hebbian.graph import HebbianSocialGraph

MODES = ["coactivity", "reward_modulated"]


def make_graph(mode, n=4, **overrides):
    cfg = HebbianConfig(
        enabled=True,
        mode=mode,
        num_agents=n,
        init_weight=0.05,          # w_0 > 0 (warm start)
        interaction_radius=5.0,
        coop_eps=0.05,
        coop_window=5,             # short window so tests run a few steps
        neg_theta=5.0,             # between |futile|=1 and |death|=10
        eta_plus=0.05,
        eta_0=0.01,
        eta_minus=0.025,
        reward_norm_R=300.0,
        communication_coactivity_bonus=0.5,
        engagement_reward_weight=0.5,
        reward_diffusion_gamma=0.2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HebbianSocialGraph(cfg)


def close_positions(n):
    """All agents within interaction radius of each other."""
    return [(float(i) * 0.5, 0.0, 0.0) for i in range(n)]


def far_positions(n):
    """All agents far apart (no spatial co-activity)."""
    return [(float(i) * 1000.0, 0.0, 0.0) for i in range(n)]


# ──────────────────────────────────────────────────────────────────────
# Invariant 1: W stays in [0, 1] under random rollouts, both variants.
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", MODES)
def test_weights_stay_bounded(mode):
    rng = np.random.RandomState(0)
    n = 4
    g = make_graph(mode, n=n)
    for _ in range(300):
        positions = [
            tuple(rng.uniform(-10, 10, size=3).astype(float)) for _ in range(n)
        ]
        chambers = [int(rng.randint(1, 6)) for _ in range(n)]
        bond_rewards = list(rng.uniform(-5, 300, size=n))
        # Total includes occasional death (-10) on top of bondable reward.
        total_rewards = [
            br + (-10.0 if rng.rand() < 0.2 else 0.0) for br in bond_rewards
        ]
        comm = [(int(rng.randint(n)), int(rng.randint(n)))] if rng.rand() < 0.3 else None
        g.update(
            positions=positions,
            comm_events=comm,
            chambers=chambers,
            bond_rewards=bond_rewards,
            total_rewards=total_rewards,
        )
        assert g.W.min() >= 0.0
        assert g.W.max() <= 1.0
        assert np.allclose(np.diag(g.W), 0.0)


# ──────────────────────────────────────────────────────────────────────
# Invariant 2: asymmetry — neg_i fires but neg_j does not ⇒ W[i,j] decays
# while W[j,i] is untouched. Bonds can become one-sided.
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", MODES)
def test_decay_asymmetry(mode):
    n = 2
    g = make_graph(mode, n=n)
    # Seed both directions with a real bond so decay has something to remove.
    g.W[0, 1] = 0.5
    g.W[1, 0] = 0.5
    w_ji_before = g.W[1, 0]

    # No co-activity (far apart, no comm) so coop_ij < eps for both directions.
    # Agent 0 sustains a genuine loss every step (Ch2, total < -theta over the
    # window); agent 1 has zero reward. So neg_0 fires, neg_1 does not.
    for _ in range(g.config.coop_window + 1):
        g.update(
            positions=far_positions(n),
            comm_events=None,
            chambers=[2, 2],
            bond_rewards=[0.0, 0.0],
            total_rewards=[-10.0, 0.0],
        )

    assert g.W[0, 1] < 0.5            # i=0 → j=1 decayed (neg_0 fired)
    assert g.W[1, 0] == pytest.approx(w_ji_before)  # j=1 → i=0 untouched


# ──────────────────────────────────────────────────────────────────────
# Invariant 3: mutual exclusivity — in any step, never both growth_ij > 0
# and decay_ij > 0 for the same pair.
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", MODES)
def test_growth_decay_mutually_exclusive(mode):
    rng = np.random.RandomState(1)
    n = 4
    g = make_graph(mode, n=n)
    g.W[:] = 0.3
    np.fill_diagonal(g.W, 0.0)
    for _ in range(200):
        # Mix co-located and far agents so both branches are exercised.
        positions = [
            (0.0, 0.0, 0.0) if rng.rand() < 0.5
            else tuple(rng.uniform(50, 100, size=3).astype(float))
            for _ in range(n)
        ]
        bond = list(rng.uniform(0, 100, size=n))
        total = [b - (10.0 if rng.rand() < 0.5 else 0.0) for b in bond]
        g.update(
            positions=positions,
            comm_events=None,
            chambers=[2] * n,
            bond_rewards=bond,
            total_rewards=total,
        )
        both = (g._last_growth > 0) & (g._last_decay > 0)
        assert not both.any(), "a pair had simultaneous growth and decay"


# ──────────────────────────────────────────────────────────────────────
# Invariant 4: w_0 fixed point — a pair with zero co-activity and no neg
# keeps its bond unchanged (NO passive decay term).
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", MODES)
def test_no_passive_decay_fixed_point(mode):
    n = 2
    g = make_graph(mode, n=n)
    g.W[0, 1] = 0.42
    g.W[1, 0] = 0.42
    for _ in range(g.config.coop_window + 5):
        # Far apart (no co-activity), zero reward (no neg, no growth).
        g.update(
            positions=far_positions(n),
            comm_events=None,
            chambers=[2, 2],
            bond_rewards=[0.0, 0.0],
            total_rewards=[0.0, 0.0],
        )
    # Bond is a fixed point: unchanged to machine precision.
    assert g.W[0, 1] == pytest.approx(0.42)
    assert g.W[1, 0] == pytest.approx(0.42)


@pytest.mark.parametrize("mode", MODES)
def test_zero_bond_is_fixed_point(mode):
    """W=0 never moves (structural reason w_0 > 0 is required)."""
    n = 2
    g = make_graph(mode, n=n)
    g.W[:] = 0.0
    # Even with full co-activity, a zero bond cannot grow because the rule
    # multiplies by W on decay and... growth uses (1-W) so growth CAN raise it.
    # The point of this test: with NO co-activity and NO neg, W=0 stays 0.
    for _ in range(10):
        g.update(
            positions=far_positions(n),
            comm_events=None,
            chambers=[2, 2],
            bond_rewards=[0.0, 0.0],
            total_rewards=[0.0, 0.0],
        )
    assert np.allclose(g.W, 0.0)


# ──────────────────────────────────────────────────────────────────────
# Invariant 5 (Variant B): a co-active step with a large DEATH penalty and no
# other reward produces ZERO growth contribution from the magnitude term —
# the coefficient collapses to the eta_0 floor.
# ──────────────────────────────────────────────────────────────────────
def test_variantB_death_excluded_from_growth_magnitude():
    n = 2
    g = make_graph("reward_modulated", n=n)
    # Death lives in total_rewards, never in bond_rewards. So the bondable
    # reward seen by the growth coefficient is exactly 0.
    coeff_death = g._growth_coeff(np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(coeff_death, g.config.eta_0)   # floor only, salience = 0

    # Contrast: a real bondable milestone reward outpaces the floor.
    coeff_reward = g._growth_coeff(np.array([300.0, 0.0], dtype=np.float32))
    assert coeff_reward[0] > g.config.eta_0
    expected = g.config.eta_0 + g.config.eta_plus * 300.0 / g.config.reward_norm_R
    assert coeff_reward[0] == pytest.approx(expected)

    # End-to-end: co-active (via comm) death-only step grows by the floor only.
    g.W[0, 1] = 0.1
    before = g.W[0, 1]
    g.update(
        positions=far_positions(n),
        comm_events=[(0, 1)],                 # communicate → co-active
        chambers=[5, 5],                      # boss room (death happens here)
        bond_rewards=[0.0, 0.0],              # death excluded from bondable
        total_rewards=[-10.0, -10.0],         # the death penalty
    )
    cij = g._last_coactivity[0, 1]
    expected_growth = g.config.eta_0 * cij * (1.0 - before)
    assert g.W[0, 1] == pytest.approx(before + expected_growth)


# ──────────────────────────────────────────────────────────────────────
# Invariant 6: Ch1 gating — a Ch1 reward/penalty changes NO social signal.
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", MODES)
def test_ch1_reward_does_not_drive_social_signal(mode):
    n = 2
    g_ch1 = make_graph(mode, n=n)
    g_zero = make_graph(mode, n=n)
    for _ in range(g_ch1.config.coop_window + 2):
        # Same geometry/comm for both; one gets big Ch1 rewards, the other zero.
        g_ch1.update(
            positions=close_positions(n),
            comm_events=[(0, 1)],
            chambers=[1, 1],                  # Ch1 (solo) — must be gated out
            bond_rewards=[80.0, 80.0],
            total_rewards=[80.0, -50.0],      # incl. a big Ch1 "loss"
        )
        g_zero.update(
            positions=close_positions(n),
            comm_events=[(0, 1)],
            chambers=[1, 1],
            bond_rewards=[0.0, 0.0],
            total_rewards=[0.0, 0.0],
        )
    # Ch1 reward magnitude must not have moved the graph relative to zero reward
    # (engagement comes only from the comm term in Ch1; neg never fires).
    assert np.allclose(g_ch1.W, g_zero.W)
    assert g_ch1._max_bond_reward_seen == 0.0   # Ch1 reward never inflated r_max


# ──────────────────────────────────────────────────────────────────────
# Invariant 7: diffusion — r' == r when gamma=0; reward flows only where
# c_ij > 0.
# ──────────────────────────────────────────────────────────────────────
def test_diffusion_identity_when_gamma_zero():
    g = make_graph("reward_modulated", n=3, reward_diffusion_gamma=0.0)
    g.W[:] = 0.5
    np.fill_diagonal(g.W, 0.0)
    g._last_coactivity = np.ones((3, 3), dtype=np.float32)
    raw = [1.0, 2.0, 3.0]
    assert g.diffuse_rewards(raw) == pytest.approx(raw)


def test_diffusion_flows_only_along_active_bonds():
    g = make_graph("reward_modulated", n=3, reward_diffusion_gamma=0.5)
    # Strong, symmetric bonds everywhere.
    g.W[:] = 0.5
    np.fill_diagonal(g.W, 0.0)
    # Agent 0 is co-active ONLY with agent 1, not agent 2.
    cij = np.zeros((3, 3), dtype=np.float32)
    cij[0, 1] = cij[1, 0] = 1.0
    g._last_coactivity = cij
    raw = [0.0, 10.0, 1000.0]
    out = g.diffuse_rewards(raw)
    # Agent 0's social reward comes only from agent 1 (c_02 = 0 blocks agent 2's
    # huge reward). w_bar[0,1] = 0.5/(0.5+0.5) = 0.5.
    gamma = 0.5
    expected0 = (1 - gamma) * 0.0 + gamma * 0.5 * 1.0 * 10.0
    assert out[0] == pytest.approx(expected0)
    # None of agent 2's 1000 reward leaked to agent 0.
    assert out[0] < 5.0


# ──────────────────────────────────────────────────────────────────────
# Extra: the two variants genuinely differ (salience term), and Variant A's
# eta_plus plays the role of Variant B's eta_0 (co-presence floor).
# ──────────────────────────────────────────────────────────────────────
def test_variantA_growth_is_reward_free_flat_rate():
    g = make_graph("coactivity", n=2)
    # Growth coefficient is constant eta_plus regardless of bondable reward.
    c_lo = g._growth_coeff(np.array([0.0, 0.0], dtype=np.float32))
    c_hi = g._growth_coeff(np.array([300.0, 0.0], dtype=np.float32))
    assert np.allclose(c_lo, g.config.eta_plus)
    assert np.allclose(c_hi, g.config.eta_plus)


# ──────────────────────────────────────────────────────────────────────
# Hardcoded / frozen graph (LLM-only social-bias ablation).
# ──────────────────────────────────────────────────────────────────────
def test_preset_topologies():
    star = HebbianSocialGraph.build_preset_matrix("star", 3, strong=0.8, weak=0.1, hub=0)
    assert star[1, 0] == pytest.approx(0.8) and star[2, 0] == pytest.approx(0.8)
    assert star[0, 1] == pytest.approx(0.1)   # nobody bonds away from hub strongly
    assert np.allclose(np.diag(star), 0.0)

    uniform = HebbianSocialGraph.build_preset_matrix("uniform", 3, strong=0.5)
    off = uniform[~np.eye(3, dtype=bool)]
    assert np.allclose(off, 0.5)              # flat: no preferred partner

    ring = HebbianSocialGraph.build_preset_matrix("ring", 3, strong=0.8, weak=0.1)
    assert ring[0, 1] == pytest.approx(0.8) and ring[1, 2] == pytest.approx(0.8)
    assert ring[2, 0] == pytest.approx(0.8) and ring[0, 2] == pytest.approx(0.1)

    pair = HebbianSocialGraph.build_preset_matrix("pair", 3, strong=0.8, weak=0.1)
    assert pair[0, 1] == pytest.approx(0.8) and pair[1, 0] == pytest.approx(0.8)
    assert pair[2, 0] == pytest.approx(0.1) and pair[0, 2] == pytest.approx(0.1)


@pytest.mark.parametrize("mode", MODES + ["legacy"])
def test_freeze_is_a_noop(mode):
    """A frozen graph never changes W, in any mode, even under strong signal."""
    g = make_graph(mode, n=3, freeze_weights=True, init_preset="star",
                   preset_bond_strong=0.8, preset_bond_weak=0.1)
    W0 = g.get_all_weights().copy()
    for _ in range(20):
        g.update(
            positions=close_positions(3),
            step_rewards=[100.0, 100.0, 100.0],
            advantages=[1.0, 1.0, 1.0],
            comm_events=[(0, 1), (1, 2)],
            chambers=[2, 2, 2],
            bond_rewards=[100.0, 100.0, 100.0],
            total_rewards=[100.0, 100.0, 100.0],
        )
    assert np.allclose(g.get_all_weights(), W0)
    # Bonds read STABLE: deltas are exactly zero for the social module.
    assert all(v == 0.0 for v in g.bond_delta_row(0).values())


def test_explicit_init_matrix_loaded_and_shape_checked():
    m = [[0.0, 0.9, 0.2], [0.3, 0.0, 0.7], [0.1, 0.1, 0.0]]
    g = HebbianSocialGraph(
        HebbianConfig(enabled=True, freeze_weights=True, num_agents=3, init_matrix=m)
    )
    assert g.get_weight(0, 1) == pytest.approx(0.9)
    assert g.get_weight(1, 2) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        HebbianSocialGraph(
            HebbianConfig(enabled=True, num_agents=3, init_matrix=[[0, 1], [1, 0]])
        )


def test_zero_warm_start_rejected_for_gated_modes():
    with pytest.raises(ValueError):
        HebbianSocialGraph(
            HebbianConfig(enabled=True, mode="coactivity", num_agents=2,
                          init_weight=0.0)
        )
    # Legacy mode tolerates init_weight=0 (it has additive base_ltp/decay).
    HebbianSocialGraph(
        HebbianConfig(enabled=True, mode="legacy", num_agents=2, init_weight=0.0)
    )
