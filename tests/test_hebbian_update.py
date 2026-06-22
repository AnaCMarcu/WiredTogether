"""Unit tests for the gated Hebbian social-graph update (src/hebbian/graph.py).

Pins the actual numerical behaviour of HebbianSocialGraph in the gated modes
("coactivity" = Variant A, "reward_modulated" = Variant B): growth coefficient,
directed asymmetry, windowed failure decay, homeostatic decay, chamber gating,
death exclusion, clipping/diagonal invariants, freeze, sanitisation, and the
disabled no-op path.  All expected values are hand-computed from the source
with the config defaults (eta_0=0.01, eta_plus=0.05, eta_minus=0.025,
coop_eps=0.05, neg_theta=5.0, reward_norm_R=300.0, init_weight=0.1,
interaction_radius=5.0, engagement_reward_weight=0.5) and decay=0 unless a
test passes decay= explicitly.
"""

import numpy as np
import pytest

from hebbian import HebbianSocialGraph

# Two-agent position constants: within / beyond interaction_radius=5.0
NEAR = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
FAR = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]


# ── Growth ───────────────────────────────────────────────────────────────────

def test_growth_one_step_hand_computed(hcfg):
    """One reward_modulated step with co-located engaged agents matches W0 + coeff*c*(1-W0)."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
             chambers=[2, 2])
    # engagement g_i = 0.5*|10|/r_max = 0.5 (r_max=10) -> c = 1 * 0.5*0.5 = 0.25
    # coeff = eta_0 + eta_plus*10/300 = 0.01 + 0.0016667 = 0.0116667
    # W1 = 0.1 + 0.0116667 * 0.25 * (1 - 0.1) = 0.102625
    expected = 0.1 + (0.01 + 0.05 * 10.0 / 300.0) * 0.25 * 0.9
    assert g.W[0, 1] == pytest.approx(expected, abs=1e-4)
    assert g.W[1, 0] == pytest.approx(expected, abs=1e-4)


def test_growth_coeff_variant_b_endpoints(hcfg):
    """Variant B growth coeff maps |r_bond| of [0, 300] to [eta_0, eta_0+eta_plus] = [0.01, 0.06]."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    coeff = g._growth_coeff(np.array([0.0, 300.0], dtype=np.float32))
    assert coeff[0] == pytest.approx(0.01, rel=1e-6)
    assert coeff[1] == pytest.approx(0.06, rel=1e-6)


def test_growth_coeff_coactivity_flat(hcfg):
    """Variant A (coactivity) growth coeff is flat eta_plus regardless of bond rewards."""
    g = HebbianSocialGraph(hcfg(num_agents=2, mode="coactivity"))
    coeff = g._growth_coeff(np.array([0.0, 300.0], dtype=np.float32))
    np.testing.assert_array_equal(coeff, np.full(2, np.float32(0.05)))


def test_directed_asymmetry_reward_modulated(hcfg):
    """In reward_modulated mode, row i uses agent i's salience: W[0,1] grows more than W[1,0]."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    # comm event makes both agents engaged: g0=clip(0.5+0.5)=1.0, g1=0.05+0.5=0.55
    # c_ij = 1.0*0.55 = 0.55 (co-located, so no extra comm co-activity term)
    g.update(NEAR, comm_events=[(0, 1)], bond_rewards=[300.0, 30.0],
             total_rewards=[300.0, 30.0], chambers=[2, 2])
    # coeff0 = 0.01 + 0.05*300/300 = 0.06 ; coeff1 = 0.01 + 0.05*30/300 = 0.015
    assert g.W[0, 1] == pytest.approx(0.1 + 0.06 * 0.55 * 0.9, abs=1e-4)
    assert g.W[1, 0] == pytest.approx(0.1 + 0.015 * 0.55 * 0.9, abs=1e-4)
    assert g.W[0, 1] > g.W[1, 0]


def test_coactivity_mode_symmetric(hcfg):
    """Same asymmetric-reward inputs in coactivity mode keep W exactly symmetric."""
    g = HebbianSocialGraph(hcfg(num_agents=2, mode="coactivity"))
    g.update(NEAR, comm_events=[(0, 1)], bond_rewards=[300.0, 30.0],
             total_rewards=[300.0, 30.0], chambers=[2, 2])
    assert g.W[0, 1] == g.W[1, 0]
    assert g.W[0, 1] == pytest.approx(0.1 + 0.05 * 0.55 * 0.9, abs=1e-4)


def test_unknown_mode_raises(hcfg):
    """An unknown gated mode raises ValueError when update reaches _growth_coeff."""
    g = HebbianSocialGraph(hcfg(num_agents=2, mode="bogus"))
    with pytest.raises(ValueError, match="unknown gated mode"):
        g.update(NEAR, bond_rewards=[1.0, 1.0], total_rewards=[1.0, 1.0],
                 chambers=[2, 2])


def test_no_growth_without_coactivity(hcfg):
    """Distant agents with no comm get zero co-activity: W is unchanged despite rewards."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    w0 = g.W.copy()
    g.update(FAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
             chambers=[2, 2])
    np.testing.assert_array_equal(g.W, w0)


# ── Failure decay (windowed neg gate) ────────────────────────────────────────

def test_failure_decay_fires(hcfg):
    """Window loss sum < -neg_theta with no co-activity applies delta = -eta_minus*W exactly."""
    cfg = hcfg(num_agents=2)
    g = HebbianSocialGraph(cfg)
    g.update(FAR, total_rewards=[-3.0, -3.0], chambers=[2, 2])
    assert g.W[0, 1] == pytest.approx(0.1)  # window sum -3 > -5: no decay yet
    g.update(FAR, total_rewards=[-3.0, -3.0], chambers=[2, 2])
    # window sum -6 < -5 and coop window max 0 < eps -> W *= (1 - eta_minus)
    assert g.W[0, 1] == pytest.approx(0.1 * (1.0 - 0.025), abs=1e-6)
    assert g.W[1, 0] == pytest.approx(0.1 * (1.0 - 0.025), abs=1e-6)


def test_no_decay_when_loss_insufficient(hcfg):
    """Window loss sum of -3 (> -neg_theta=-5) never triggers the failure decay."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    for _ in range(3):
        g.update(FAR, total_rewards=[-1.0, -1.0], chambers=[2, 2])
    assert g.W[0, 1] == pytest.approx(0.1)
    assert g.W[1, 0] == pytest.approx(0.1)


def test_no_decay_while_coactive_step_in_window(hcfg):
    """A co-active step still inside the coop window blocks failure decay even when neg fires."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    # Step 1: co-located + engaged (bondable reward), zero total reward.
    g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[0.0, 0.0],
             chambers=[2, 2])
    w_after_growth = float(g.W[0, 1])
    assert w_after_growth > 0.1
    # Steps 2-3: apart, losses -> window sum 0-3-3 = -6 < -5 (neg fires), but
    # coop window max is still 0.25 >= coop_eps -> decay mask stays False.
    g.update(FAR, total_rewards=[-3.0, -3.0], chambers=[2, 2])
    g.update(FAR, total_rewards=[-3.0, -3.0], chambers=[2, 2])
    assert g.W[0, 1] == pytest.approx(w_after_growth, abs=1e-9)


# ── Homeostatic decay ────────────────────────────────────────────────────────

def test_homeostatic_decay_geometric(hcfg):
    """With decay=0.01 and no co-activity/losses, W shrinks geometrically by decay*W per step."""
    g = HebbianSocialGraph(hcfg(num_agents=2, decay=0.01))
    g.update(FAR, chambers=[2, 2])
    assert g.W[0, 1] == pytest.approx(0.1 * 0.99, rel=1e-5)
    for _ in range(4):
        g.update(FAR, chambers=[2, 2])
    assert g.W[0, 1] == pytest.approx(0.1 * 0.99 ** 5, rel=1e-5)


def test_homeostatic_fixed_point(hcfg):
    """Constant co-activity with decay>0 drives W monotonically to coeff*c/(coeff*c+decay)."""
    cfg = hcfg(num_agents=2, decay=0.01)
    g = HebbianSocialGraph(cfg)
    ws = []
    for _ in range(400):
        g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
                 chambers=[2, 2])
        ws.append(float(g.W[0, 1]))
    coeff = cfg.eta_0 + cfg.eta_plus * 10.0 / cfg.reward_norm_R  # 0.0116667
    c = 0.25  # engagement 0.5 each, co-located
    w_star = coeff * c / (coeff * c + cfg.decay)  # ~0.22581
    diffs = np.diff(ws)
    assert np.all(diffs > -1e-7)  # monotone non-decreasing trend from W0=0.1
    assert ws[-1] == pytest.approx(w_star, abs=0.01)


# ── Chamber gate ─────────────────────────────────────────────────────────────

def test_chamber_gate_blocks_growth_and_neg(hcfg):
    """Chamber 1 gates rewards to zero: no growth from big rewards, no failure decay from losses."""
    g_pos = HebbianSocialGraph(hcfg(num_agents=2))
    g_pos.update(NEAR, bond_rewards=[100.0, 100.0], total_rewards=[100.0, 100.0],
                 chambers=[1, 1])
    assert g_pos.W[0, 1] == pytest.approx(0.1)

    g_neg = HebbianSocialGraph(hcfg(num_agents=2))
    for _ in range(2):
        g_neg.update(FAR, total_rewards=[-100.0, -100.0], chambers=[1, 1])
    assert g_neg.W[0, 1] == pytest.approx(0.1)


def test_chambers_none_treated_as_cooperative(hcfg):
    """chambers=None means gate=1 for everyone: growth matches the explicit Ch2 case."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
             chambers=None)
    expected = 0.1 + (0.01 + 0.05 * 10.0 / 300.0) * 0.25 * 0.9
    assert g.W[0, 1] == pytest.approx(expected, abs=1e-4)


# ── Death exclusion ──────────────────────────────────────────────────────────

def test_death_exclusion(hcfg):
    """Death (-50 in total, 0 in bond) gives growth coeff exactly eta_0 while the loss window fills."""
    cfg = hcfg(num_agents=2)
    g = HebbianSocialGraph(cfg)
    g.update(NEAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, -50.0],
             chambers=[2, 2])
    # Salience term is 0 -> coeff is the bare association floor eta_0.
    coeff = g._growth_coeff(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(coeff, np.full(2, cfg.eta_0), rtol=1e-6)
    # The loss DID land in the reward window (death counts toward neg_i)...
    np.testing.assert_array_equal(list(g._reward_window)[0], [-50.0, -50.0])
    # ...and with zero engagement (bond=0 -> cij=0) the failure decay fired.
    assert g.W[0, 1] == pytest.approx(0.1 * (1.0 - cfg.eta_minus), abs=1e-6)


# ── Invariants / freeze / init / sanitisation / disabled ────────────────────

def test_invariants_under_max_growth(hcfg):
    """Over 320 max-growth steps W never exceeds 1.0 and the diagonal stays exactly 0."""
    g = HebbianSocialGraph(hcfg(num_agents=3))
    pos = [(0.0, 0.0, 0.0)] * 3
    comm = [(0, 1), (1, 2), (0, 2)]
    for _ in range(320):
        g.update(pos, comm_events=comm, bond_rewards=[300.0] * 3,
                 total_rewards=[300.0] * 3, chambers=[2, 2, 2])
        assert g.W.max() <= 1.0
        assert np.all(np.diag(g.W) == 0.0)
    assert g.W[0, 1] > 0.99  # asymptotes to the W=1 ceiling


def test_freeze_weights_blocks_update(hcfg):
    """freeze_weights=True: update returns W but leaves it unchanged under a growth stimulus."""
    g = HebbianSocialGraph(hcfg(num_agents=2, freeze_weights=True))
    w0 = g.W.copy()
    out = g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
                   chambers=[2, 2])
    assert out is not None
    np.testing.assert_array_equal(g.W, w0)


def test_init_weight_zero_raises_for_gated_modes(hcfg):
    """init_weight=0 raises ValueError for gated modes (W=0 is a fixed point); legacy allows it."""
    with pytest.raises(ValueError, match="init_weight"):
        HebbianSocialGraph(hcfg(num_agents=2, init_weight=0.0))
    with pytest.raises(ValueError, match="init_weight"):
        HebbianSocialGraph(hcfg(num_agents=2, mode="coactivity", init_weight=0.0))
    g = HebbianSocialGraph(hcfg(num_agents=2, mode="legacy", init_weight=0.0))
    assert g.W[0, 1] == 0.0


def test_nan_inf_rewards_sanitised(hcfg):
    """NaN/inf rewards are clamped to 0: update neither crashes nor corrupts W."""
    g = HebbianSocialGraph(hcfg(num_agents=2))
    w0 = g.W.copy()
    g.update(NEAR, bond_rewards=[float("nan"), float("inf")],
             total_rewards=[float("-inf"), float("nan")], chambers=[2, 2])
    assert np.isfinite(g.W).all()
    np.testing.assert_array_equal(g.W, w0)  # all signals clamp to 0 -> no change


def test_disabled_config_returns_none(hcfg):
    """With enabled=False, update() is a no-op returning None."""
    g = HebbianSocialGraph(hcfg(num_agents=2, enabled=False))
    assert g.update(NEAR, bond_rewards=[10.0, 10.0], total_rewards=[10.0, 10.0],
                    chambers=[2, 2]) is None
