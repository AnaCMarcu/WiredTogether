"""Tests for the gated co-activity math in HebbianSocialGraph (mode != "legacy").

Pins the behaviour of ``_engagement`` / ``_coactivity_gated`` / ``_chamber_gate``
as driven through ``graph.update(...)`` (gated path → ``_update_gated``), reading
back ``graph._last_coactivity``.  All expected values are hand-computed from the
defaults in ``HebbianConfig``:

    interaction_radius              d      = 5.0
    engagement_reward_weight        alpha  = 0.5
    communication_coactivity_bonus  delta  = 0.5
    coop_eps                        eps    = 0.05

Engagement (single step, so the running max == current max |bond reward|):
    g_i = clip( alpha * |r_bond_i| * 1[ch_i >= 2] / (r_max + 1e-8)
                + (1 - alpha) * 1[i in comm set], 0, 1 )
Co-activity:
    c_spat = 1[dist <= d] * g_i * g_j
    c_comm = delta * 1[i <-> j event, both indices in range] * (1 - 1[dist <= d])
    c_ij   = clip(c_spat + c_comm, 0, 1), diagonal forced 0, then floored to
             exactly 0.0 wherever c_ij < eps.
"""

import numpy as np
import pytest

from hebbian import HebbianSocialGraph


def _graph(hcfg, num_agents=2, **overrides):
    return HebbianSocialGraph(hcfg(num_agents=num_agents, **overrides))


def test_colocated_engaged_pair_spatial_coactivity(hcfg):
    """Two co-located agents with bond reward 10 in Ch2 get g=0.5 each and c01=c10~=0.25."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        chambers=[2, 2],
        bond_rewards=[10.0, 10.0],
        total_rewards=[10.0, 10.0],
    )
    c = g._last_coactivity
    # First step: running max of |bond reward| becomes 10 exactly.
    assert g._max_bond_reward_seen == 10.0
    # g_i = 0.5 * 10 / (10 + 1e-8) ~= 0.5  ->  c = g0*g1 ~= 0.25
    assert c[0, 1] == pytest.approx(0.25, rel=1e-6)
    assert c[1, 0] == pytest.approx(0.25, rel=1e-6)
    assert c[0, 1] == c[1, 0]


def test_no_coactivity_beyond_interaction_radius(hcfg):
    """Engaged agents 100 units apart with no comm produce an all-zero co-activity matrix."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        chambers=[2, 2],
        bond_rewards=[10.0, 10.0],
        total_rewards=[10.0, 10.0],
    )
    assert np.all(g._last_coactivity == 0.0)


def test_distance_exactly_radius_fires_spatial_gate(hcfg):
    """Distance exactly == interaction_radius (5.0) fires the spatial gate (source uses <=)."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)],  # ||p0-p1|| = 5.0 exactly
        chambers=[2, 2],
        bond_rewards=[10.0, 10.0],
        total_rewards=[10.0, 10.0],
    )
    assert g._last_coactivity[0, 1] == pytest.approx(0.25, rel=1e-6)


def test_comm_bonus_for_distant_pair(hcfg):
    """Distant pair with a comm event gets c = delta_comm * (1 - 0) = 0.5 in both directions."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        comm_events=[(0, 1)],
        bond_rewards=[0.0, 0.0],
        total_rewards=[0.0, 0.0],
    )
    c = g._last_coactivity
    assert c[0, 1] == pytest.approx(0.5)
    assert c[1, 0] == pytest.approx(0.5)


def test_comm_bonus_suppressed_when_colocated(hcfg):
    """Co-located comm pair gets only the spatial term (g=0.5 each -> 0.25); the 0.5 bonus is gated out."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        comm_events=[(0, 1)],
        bond_rewards=[0.0, 0.0],
        total_rewards=[0.0, 0.0],
    )
    c = g._last_coactivity
    # Engagement from comm only: g_i = (1-alpha) = 0.5 -> c_spat = 0.25.
    # Without suppression the comm bonus would push this to 0.75.
    assert c[0, 1] == pytest.approx(0.25)
    assert c[1, 0] == pytest.approx(0.25)


def test_single_directed_comm_event_is_symmetric(hcfg):
    """One directed event (0, 1) sets c[0,1] == c[1,0]; uninvolved agent 2 stays at 0."""
    g = _graph(hcfg, num_agents=3)
    g.update(
        positions=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)],
        comm_events=[(0, 1)],
        bond_rewards=[0.0, 0.0, 0.0],
        total_rewards=[0.0, 0.0, 0.0],
    )
    c = g._last_coactivity
    assert c[0, 1] == pytest.approx(0.5)
    assert c[1, 0] == c[0, 1]
    assert c[0, 2] == 0.0 and c[2, 0] == 0.0
    assert c[1, 2] == 0.0 and c[2, 1] == 0.0


def test_eps_floor_zeroes_tiny_coactivity(hcfg):
    """Co-activity below coop_eps (0.05) is floored to exactly 0.0."""
    g = _graph(hcfg)
    # r_max = 10 -> g0 ~= 0.5, g1 = 0.5*0.1/10 = 0.005 -> c = 0.0025 < 0.05 -> 0.0
    g.update(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        chambers=[2, 2],
        bond_rewards=[10.0, 0.1],
        total_rewards=[10.0, 0.1],
    )
    c = g._last_coactivity
    assert c[0, 1] == 0.0
    assert c[1, 0] == 0.0


def test_diagonal_always_zero(hcfg):
    """Diagonal of the co-activity matrix is 0 even with self-comm events and full engagement."""
    g = _graph(hcfg, num_agents=3)
    g.update(
        positions=[(0.0, 0.0, 0.0)] * 3,
        comm_events=[(0, 0), (0, 1), (1, 2)],  # includes a degenerate self-event
        chambers=[2, 2, 2],
        bond_rewards=[10.0, 10.0, 10.0],
        total_rewards=[10.0, 10.0, 10.0],
    )
    np.testing.assert_array_equal(np.diag(g._last_coactivity), np.zeros(3))


def test_none_position_disables_spatial_gate(hcfg):
    """An agent with position None gets no spatial co-activity; the located pair is unaffected."""
    g = _graph(hcfg, num_agents=3)
    g.update(
        positions=[None, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        chambers=[2, 2, 2],
        bond_rewards=[10.0, 10.0, 10.0],
        total_rewards=[10.0, 10.0, 10.0],
    )
    c = g._last_coactivity
    assert c[0, 1] == 0.0 and c[1, 0] == 0.0
    assert c[0, 2] == 0.0 and c[2, 0] == 0.0
    assert c[1, 2] == pytest.approx(0.25, rel=1e-6)


def test_chamber_gate_ch1_zeroes_reward_engagement(hcfg):
    """Chambers [1,1] gate out bond rewards entirely: g=0 -> c all zeros, r_max untouched."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        chambers=[1, 1],
        bond_rewards=[10.0, 10.0],
        total_rewards=[10.0, 10.0],
    )
    assert np.all(g._last_coactivity == 0.0)
    # The Ch1-gated rewards never reach the running-max normaliser.
    assert g._max_bond_reward_seen == 0.0


def test_comm_agent_with_zero_reward_has_half_engagement(hcfg):
    """An agent in the comm set with zero bond reward has engagement exactly (1-alpha) = 0.5."""
    g = _graph(hcfg, num_agents=3)
    eng = g._engagement(np.zeros(3, dtype=np.float32), {0})
    assert eng[0] == pytest.approx(0.5)
    assert eng[1] == 0.0
    assert eng[2] == 0.0


def test_out_of_range_comm_event_no_crash_no_bonus(hcfg):
    """A comm event with an out-of-range index (0, 99) neither crashes nor adds any bonus."""
    g = _graph(hcfg)
    g.update(
        positions=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        comm_events=[(0, 99)],
        chambers=[2, 2],
        bond_rewards=[10.0, 10.0],
        total_rewards=[10.0, 10.0],
    )
    # Distant + no valid pair event -> no spatial term, no comm bonus.
    assert np.all(g._last_coactivity == 0.0)
