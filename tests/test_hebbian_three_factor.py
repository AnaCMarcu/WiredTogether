"""Unit tests for the "three_factor" gated mode (eligibility trace).

The rule (src/hebbian/graph.py::_update_gated, mode="three_factor"):

    e_ij  <- rho_e * e_ij + c_ij                      (trace, current step incl.)
    dW    = eta0*c*(1-W) + eta+*(|r_bond|/R)*e*(1-W)  (growth)
            [- eta- * W  when the failure branch fires]
            - lambda * W                              (homeostatic, always)

with monotone co-activity: co-location counts at the floor even for a silent
pair, and the comm term is not gated to far-apart pairs.

Expected values are hand-computed. Unless a test says otherwise, eta_0=0 and
decay=0 isolate the trace term, R=50, eta_plus=0.05, rho=0.9, w0=0.1.
"""

import numpy as np

from hebbian import HebbianSocialGraph
from hebbian.config import HebbianConfig

NEAR = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
FAR = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
MSG = [(0, 1)]


def cfg(**kw):
    base = dict(enabled=True, mode="three_factor", num_agents=2,
                decay=0.0, eta_0=0.0, eta_plus=0.05, reward_norm_R=50.0,
                eligibility_rho=0.9, coact_floor=0.25, init_weight=0.1)
    base.update(kw)
    return HebbianConfig(**base)


def msg_steps(g, k, positions=FAR):
    for _ in range(k):
        g.update(positions, comm_events=MSG, bond_rewards=[0.0, 0.0],
                 total_rewards=[0.0, 0.0], chambers=[2, 2])


# ── The trace itself ─────────────────────────────────────────────────────────

def test_trace_accumulates_and_fades():
    g = HebbianSocialGraph(cfg())
    msg_steps(g, 10)                       # far apart, messaging: c = 0.5
    e10 = 0.5 * (1 - 0.9 ** 10) / 0.1      # geometric sum = 3.2566
    assert np.isclose(g._eligibility[0, 1], e10, atol=1e-3)
    assert np.isclose(g._eligibility[1, 0], e10, atol=1e-3)
    # with eta_0 = 0, no reward and no decay, W must not have moved
    assert np.isclose(g.W[0, 1], 0.1, atol=1e-6)
    for _ in range(5):                     # silence: trace fades by rho/step
        g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[0.0, 0.0],
                 chambers=[2, 2])
    assert np.isclose(g._eligibility[0, 1], e10 * 0.9 ** 5, atol=1e-3)


def test_reward_converts_trace_into_weight():
    g = HebbianSocialGraph(cfg())
    msg_steps(g, 10)
    e10 = 0.5 * (1 - 0.9 ** 10) / 0.1
    g.update(FAR, comm_events=MSG, bond_rewards=[50.0, 0.0],
             total_rewards=[50.0, 0.0], chambers=[2, 2])
    e_new = 0.9 * e10 + 0.5
    expected = 0.05 * (50.0 / 50.0) * e_new * (1 - 0.1)   # 0.1544
    assert np.isclose(g.W[0, 1], 0.1 + expected, atol=1e-3)
    # agent 1 earned nothing: its outgoing row gets no reward-driven growth
    assert np.isclose(g.W[1, 0], 0.1, atol=1e-6)


def test_reward_credits_past_coactivity_even_when_pair_is_apart():
    """THE distinguishing property vs reward_modulated: a reward landing on a
    step with zero current co-activity still credits the recent joint work."""
    g3 = HebbianSocialGraph(cfg())
    msg_steps(g3, 10)
    g3.update(FAR, bond_rewards=[50.0, 0.0], total_rewards=[50.0, 0.0],
              chambers=[2, 2])             # no message, far apart: c = 0
    e_new = 0.9 * (0.5 * (1 - 0.9 ** 10) / 0.1)
    assert np.isclose(g3.W[0, 1], 0.1 + 0.05 * e_new * 0.9, atol=1e-3)

    gB = HebbianSocialGraph(cfg(mode="reward_modulated", eta_0=0.005))
    msg_steps(gB, 10)
    w_before = gB.W[0, 1]
    gB.update(FAR, bond_rewards=[50.0, 0.0], total_rewards=[50.0, 0.0],
              chambers=[2, 2])
    assert np.isclose(gB.W[0, 1], w_before, atol=1e-6)   # c=0 -> no growth


# ── Monotone co-activity ─────────────────────────────────────────────────────

def test_colocated_silent_pair_has_floor_coactivity():
    g3 = HebbianSocialGraph(cfg())
    g3.update(NEAR, bond_rewards=[0.0, 0.0], total_rewards=[0.0, 0.0],
              chambers=[2, 2])
    assert np.isclose(g3._last_coactivity[0, 1], 0.25, atol=1e-6)

    gB = HebbianSocialGraph(cfg(mode="reward_modulated"))
    gB.update(NEAR, bond_rewards=[0.0, 0.0], total_rewards=[0.0, 0.0],
              chambers=[2, 2])
    assert np.isclose(gB._last_coactivity[0, 1], 0.0, atol=1e-6)


def test_near_plus_messaging_beats_far_plus_messaging():
    g3 = HebbianSocialGraph(cfg())
    g3.update(NEAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
              total_rewards=[0.0, 0.0], chambers=[2, 2])
    assert np.isclose(g3._last_coactivity[0, 1], 0.75, atol=1e-6)  # 0.25+0.5

    gB = HebbianSocialGraph(cfg(mode="reward_modulated"))
    gB.update(NEAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
              total_rewards=[0.0, 0.0], chambers=[2, 2])
    # legacy comm term is distance-gated: near+messaging = spatial 0.25 only
    assert np.isclose(gB._last_coactivity[0, 1], 0.25, atol=1e-6)


def test_coact_floor_zero_restores_gated_spatial_term():
    g = HebbianSocialGraph(cfg(coact_floor=0.0))
    g.update(NEAR, bond_rewards=[0.0, 0.0], total_rewards=[0.0, 0.0],
             chambers=[2, 2])
    assert np.isclose(g._last_coactivity[0, 1], 0.0, atol=1e-6)


# ── Isolation of the other modes ─────────────────────────────────────────────

def test_new_fields_are_invisible_to_reward_modulated():
    seq = [
        dict(positions=NEAR, comm_events=MSG, bond_rewards=[10.0, 0.0],
             total_rewards=[10.0, 0.0], chambers=[2, 2]),
        dict(positions=FAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
             total_rewards=[0.0, 0.0], chambers=[2, 2]),
        dict(positions=FAR, bond_rewards=[0.0, 40.0],
             total_rewards=[0.0, 40.0], chambers=[3, 3]),
        dict(positions=NEAR, bond_rewards=[0.0, 0.0],
             total_rewards=[-6.0, -6.0], chambers=[2, 2]),
    ] * 5
    gA = HebbianSocialGraph(cfg(mode="reward_modulated", eta_0=0.005,
                                decay=0.005))
    gB = HebbianSocialGraph(cfg(mode="reward_modulated", eta_0=0.005,
                                decay=0.005, eligibility_rho=0.42,
                                coact_floor=0.9))
    for kw in seq:
        gA.update(**kw)
        gB.update(**kw)
    assert np.array_equal(gA.W, gB.W)
    assert not gA._eligibility.any() and not gB._eligibility.any()


# ── Homeostasis and serialisation ────────────────────────────────────────────

def test_homeostatic_decay_still_applies():
    g = HebbianSocialGraph(cfg(decay=0.001))
    g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[0.0, 0.0],
             chambers=[2, 2])              # c = 0: pure decay
    assert np.isclose(g.W[0, 1], 0.1 * (1 - 0.001), atol=1e-6)


def test_eligibility_survives_serialisation_roundtrip():
    g = HebbianSocialGraph(cfg())
    msg_steps(g, 7)
    d = g.to_dict()
    g2 = HebbianSocialGraph(cfg())
    g2.from_dict(d)
    assert np.allclose(g2._eligibility, g._eligibility, atol=1e-6)
    g2.reset()
    assert not g2._eligibility.any()
