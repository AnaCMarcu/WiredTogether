"""MultiChannelHebbianGraph: parity with three_factor + the AgentWorld channels."""

import numpy as np
import pytest

from hebbian import HebbianConfig, HebbianSocialGraph
from hebbian.multichannel import MultiChannelConfig, MultiChannelHebbianGraph

COMMON = dict(enabled=True, mode="three_factor", num_agents=5, eta_minus_death=0.05)


def _random_stream(rng, N, steps, channels=("comm",)):
    for _ in range(steps):
        positions = [None if rng.random() < 0.1 else tuple(rng.uniform(0, 12, 3))
                     for _ in range(N)]
        comm = [(int(a), int(b)) for a, b in rng.integers(0, N, (rng.integers(0, 4), 2))]
        social = [(int(a), int(b), str(rng.choice(channels)))
                  for a, b in rng.integers(0, N, (rng.integers(0, 3), 2))]
        bond = list(rng.choice([0.0, 0.0, 0.5, 10.0, -1.0], N))
        death = list(rng.choice([0.0] * 9 + [-10.0], N))
        total = [b + d for b, d in zip(bond, death)]
        yield positions, comm, social, bond, total, death


@pytest.mark.parametrize("channels", [("comm",), ("comm", "obs", "imit")])
def test_parity_with_three_factor(channels):
    rng = np.random.default_rng(7)
    ref = HebbianSocialGraph(HebbianConfig(**COMMON, social_act_channels=channels))
    mc = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON, social_act_channels=channels))
    for positions, comm, social, bond, total, death in _random_stream(rng, 5, 500, channels):
        kw = dict(comm_events=comm, chambers=None, bond_rewards=bond,
                  total_rewards=total, social_events=social, death_rewards=death)
        ref.update(positions, **kw)
        mc.update(positions, **kw)
        np.testing.assert_allclose(mc.W, ref.W, atol=1e-7, rtol=0)
        np.testing.assert_allclose(mc._eligibility, ref._eligibility, atol=1e-6, rtol=0)


def test_channel_attribution_sums_to_growth():
    rng = np.random.default_rng(3)
    g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON))
    total = np.zeros((5, 5), dtype=np.float64)
    for positions, comm, _, bond, tot, death in _random_stream(rng, 5, 200):
        social = [(0, 1, "xfer"), (2, 3, "combat")]
        g.update(positions, comm_events=comm, bond_rewards=bond, total_rewards=tot,
                 social_events=social, death_rewards=death)
        total += g._last_growth
    attributed = sum(v.astype(np.float64) for v in g.channel_growth().values())
    np.testing.assert_allclose(attributed, total, atol=1e-4)
    assert g.channel_growth()["xfer"][0, 1] > 0
    assert g.channel_growth()["combat"][3, 2] > 0


def test_two_d_positions_match_z_zero():
    a = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON))
    b = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON))
    pos2 = [(0, 0), (1, 1), (30, 30), None, (2, 0)]
    pos3 = [(0, 0, 0), (1, 1, 0), (30, 30, 0), None, (2, 0, 0)]
    for _ in range(20):
        a.update(pos2, bond_rewards=[1.0] * 5, total_rewards=[1.0] * 5)
        b.update(pos3, bond_rewards=[1.0] * 5, total_rewards=[1.0] * 5)
    np.testing.assert_array_equal(a.W, b.W)
    assert a.W[0, 1] > a.W[0, 2]  # near pair beats far pair


def test_xfer_is_symmetric_and_engages_both():
    g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON))
    g.update([None] * 5, social_events=[(0, 1, "xfer")])
    c = g.last_channel_terms()["xfer"]
    assert c[0, 1] == pytest.approx(0.8) and c[1, 0] == pytest.approx(0.8)
    assert g._last_engagement[1] > 0  # the receiver counts as socially active


def test_xfer_directed_option():
    cfg = MultiChannelConfig(**COMMON, channel_symmetric={"xfer": False})
    g = MultiChannelHebbianGraph(cfg)
    g.update([None] * 5, social_events=[(0, 1, "xfer")])
    c = g.last_channel_terms()["xfer"]
    assert c[0, 1] > 0 and c[1, 0] == 0


def test_read_channel_off_by_default():
    g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON))
    w0 = g.W.copy()
    for _ in range(10):
        g.update([None] * 5, social_events=[(0, 1, "read")])
    assert "read" not in g.last_channel_terms()
    assert g.W[0, 1] <= w0[0, 1]  # only homeostatic decay acts


def test_uncredited_channel_is_ignored():
    cfg = MultiChannelConfig(**COMMON, social_act_channels=("comm",))
    g = MultiChannelHebbianGraph(cfg)
    g.update([None] * 5, social_events=[(0, 1, "xfer")])
    assert "xfer" not in g.last_channel_terms()


def test_joint_salience_needs_both_partners_rewarded():
    def run(mode, rewards):
        g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON, salience_mode=mode))
        for _ in range(10):
            g.update([None] * 5, comm_events=[(0, 1)], bond_rewards=rewards,
                     total_rewards=rewards)
        return g.W[0, 1]

    only_i = [10.0, 0, 0, 0, 0]
    both = [10.0, 10.0, 0, 0, 0]
    # Ego salience credits agent 0's row from its own reward alone; joint
    # salience gives that pair nothing extra until the partner is rewarded too.
    assert run("ego", only_i) > run("joint", only_i)
    assert run("joint", both) > run("joint", only_i)


def test_joint_trace_bridges_lag():
    g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON, salience_mode="joint"))
    g.update([None] * 5, social_events=[(0, 1, "xfer")], bond_rewards=[10, 0, 0, 0, 0])
    g.update([None] * 5, bond_rewards=[0, 10, 0, 0, 0])  # receiver crafts a round later
    assert g._salience_trace[0] == pytest.approx(7.0)    # 0.7 · 10 carried over
    assert g._last_growth[0, 1] > g._last_growth[0, 2]


def test_serialisation_round_trip():
    g = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON, salience_mode="joint"))
    for _ in range(5):
        g.update([None] * 5, comm_events=[(0, 2)], bond_rewards=[3.0] * 5)
    h = MultiChannelHebbianGraph(MultiChannelConfig(**COMMON, salience_mode="joint"))
    h.from_dict(g.to_dict())
    np.testing.assert_array_equal(h.W, g.W)
    np.testing.assert_array_equal(h._salience_trace, g._salience_trace)


def test_top_k_and_large_n():
    cfg = MultiChannelConfig(**{**COMMON, "num_agents": 200})
    g = MultiChannelHebbianGraph(cfg)
    events = [(i, (i + 1) % 200, "xfer") for i in range(0, 200, 2)]
    for _ in range(5):
        g.update([(i % 20, i // 20) for i in range(200)], social_events=events,
                 bond_rewards=[5.0] * 200)
    assert g.top_k(0, 3)[0] == 1
    assert len(g.top_k(0, 6)) == 6


def test_rejects_other_modes():
    with pytest.raises(ValueError):
        MultiChannelConfig(**{**COMMON, "mode": "reward_modulated"})
