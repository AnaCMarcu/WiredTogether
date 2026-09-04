"""Unit tests for the signed death LTD term (three_factor + eta_minus_death).

The term (src/hebbian/graph.py::_update_gated, audit V1 "death->blame"):

    dW-  = eta_d * (min(|death_i|, cap) / R) * e_ij * W_ij

subtracted on the dying agent's OUTGOING row, through the same eligibility
trace the growth term uses (current step included). eta_minus_death = 0
(the default) must leave every existing three_factor arm byte-identical.

Expected values are hand-computed. eta_0=0 and decay=0 isolate the term;
R=50, rho=0.9, w0=0.5 (LTD needs weight to act on), cap=10.
"""

import numpy as np

from hebbian import HebbianSocialGraph
from hebbian.config import HebbianConfig

FAR = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
MSG = [(0, 1)]


def cfg(**kw):
    base = dict(enabled=True, mode="three_factor", num_agents=2,
                decay=0.0, eta_0=0.0, eta_plus=0.05, reward_norm_R=50.0,
                eligibility_rho=0.9, coact_floor=0.25, init_weight=0.5,
                eta_minus_death=0.05, death_cap=10.0,
                # eta_minus=0 keeps the (dead-in-practice) coop<eps ∧ neg_i
                # branch out of these tests — it would otherwise fire on a
                # fresh graph's first step, where the coop window is empty.
                eta_minus=0.0)
    base.update(kw)
    return HebbianConfig(**base)


def msg_steps(g, k):
    for _ in range(k):
        g.update(FAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
                 total_rewards=[0.0, 0.0], chambers=[2, 2])


E10 = 0.5 * (1 - 0.9 ** 10) / 0.1        # trace after 10 far-messaging steps


def test_death_converts_trace_into_weakening():
    g = HebbianSocialGraph(cfg())
    msg_steps(g, 10)
    g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, 0.0],
             chambers=[2, 2], death_rewards=[-50.0, 0.0])
    e_new = 0.9 * E10                     # no message on the death step
    expected_ltd = 0.05 * (10.0 / 50.0) * e_new * 0.5    # cap: -50 -> 10
    assert np.isclose(g.W[0, 1], 0.5 - expected_ltd, atol=1e-4)
    # egocentric: agent 1 did not die, its outgoing bond is untouched
    assert np.isclose(g.W[1, 0], 0.5, atol=1e-6)


def test_cap_makes_would_die_and_death_blame_equally():
    ga, gb = HebbianSocialGraph(cfg()), HebbianSocialGraph(cfg())
    msg_steps(ga, 10)
    msg_steps(gb, 10)
    ga.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, 0.0],
              chambers=[2, 2], death_rewards=[-50.0, 0.0])
    gb.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-10.0, 0.0],
              chambers=[2, 2], death_rewards=[-10.0, 0.0])
    assert np.array_equal(ga.W, gb.W)


def test_no_trace_no_blame():
    """A death with no recent co-activity weakens nothing (unlike old phi)."""
    g = HebbianSocialGraph(cfg())
    g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, 0.0],
             chambers=[2, 2], death_rewards=[-50.0, 0.0])
    assert np.isclose(g.W[0, 1], 0.5, atol=1e-6)


def test_default_zero_rate_is_byte_identical():
    ga = HebbianSocialGraph(cfg(eta_minus_death=0.0))
    gb = HebbianSocialGraph(cfg(eta_minus_death=0.0))
    msg_steps(ga, 10)
    msg_steps(gb, 10)
    ga.update(FAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
              total_rewards=[-50.0, 0.0], chambers=[2, 2],
              death_rewards=[-50.0, 0.0])
    gb.update(FAR, comm_events=MSG, bond_rewards=[0.0, 0.0],
              total_rewards=[-50.0, 0.0], chambers=[2, 2])
    assert np.array_equal(ga.W, gb.W)


def test_invisible_to_reward_modulated():
    ga = HebbianSocialGraph(cfg(mode="reward_modulated", eta_0=0.005))
    gb = HebbianSocialGraph(cfg(mode="reward_modulated", eta_0=0.005,
                                eta_minus_death=0.0))
    for g in (ga, gb):
        msg_steps(g, 5)
        g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, 0.0],
                 chambers=[2, 2], death_rewards=[-50.0, 0.0])
    assert np.array_equal(ga.W, gb.W)


def test_death_ltd_lands_in_last_decay_bookkeeping():
    g = HebbianSocialGraph(cfg())
    msg_steps(g, 10)
    g.update(FAR, bond_rewards=[0.0, 0.0], total_rewards=[-50.0, 0.0],
             chambers=[2, 2], death_rewards=[-50.0, 0.0])
    e_new = 0.9 * E10
    expected_ltd = 0.05 * (10.0 / 50.0) * e_new * 0.5
    assert np.isclose(g._last_decay[0, 1], expected_ltd, atol=1e-4)
    assert np.isclose(g._last_decay[1, 0], 0.0, atol=1e-8)
