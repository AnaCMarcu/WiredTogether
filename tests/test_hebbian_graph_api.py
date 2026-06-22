"""API-level tests for ``HebbianSocialGraph`` (src/hebbian/graph.py).

Pins: preset/init-matrix construction, reward diffusion (Eq. 8), normalised
weights (Eq. 6), social replay (Eq. 7), graph metrics, to_dict/from_dict
round-trip, snapshot, reset, bond-delta window, the legacy update path, and
the disabled no-op surface.  All expected values are hand-computed from
``HebbianConfig`` defaults via the ``hcfg`` fixture (decay=0 unless noted).
"""

import math

import numpy as np
import pytest

from hebbian import HebbianConfig, HebbianSocialGraph


# ── helpers ──────────────────────────────────────────────────────────────────

_POS3 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]  # all within d=5


def _run_gated_steps(g, n):
    """Run n deterministic gated updates (all co-located, bond rewards 1/2/3)."""
    for _ in range(n):
        g.update(
            positions=_POS3,
            bond_rewards=[1.0, 2.0, 3.0],
            total_rewards=[1.0, 2.0, 3.0],
        )


# ── presets / init matrix ────────────────────────────────────────────────────

def test_preset_uniform():
    """'uniform' preset sets every off-diagonal bond to strong, diagonal 0."""
    W = HebbianSocialGraph.build_preset_matrix("uniform", 3, strong=0.8, weak=0.1)
    expected = np.full((3, 3), 0.8, dtype=np.float32)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(W, expected, rtol=1e-7)


def test_preset_star():
    """'star' preset makes everyone bond strongly toward the hub column only."""
    W = HebbianSocialGraph.build_preset_matrix("star", 4, strong=0.8, weak=0.1, hub=1)
    expected = np.full((4, 4), 0.1, dtype=np.float32)
    expected[:, 1] = 0.8
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(W, expected, rtol=1e-7)


def test_preset_ring():
    """'ring' preset is a directed cycle i -> (i+1) % N at strong, rest weak."""
    W = HebbianSocialGraph.build_preset_matrix("ring", 4, strong=0.8, weak=0.1)
    expected = np.full((4, 4), 0.1, dtype=np.float32)
    for i in range(4):
        expected[i, (i + 1) % 4] = 0.8
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(W, expected, rtol=1e-7)


def test_preset_pair():
    """'pair' preset bonds the 0<->1 dyad strongly, all other off-diag weak."""
    W = HebbianSocialGraph.build_preset_matrix("pair", 3, strong=0.8, weak=0.1)
    expected = np.full((3, 3), 0.1, dtype=np.float32)
    expected[0, 1] = expected[1, 0] = 0.8
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(W, expected, rtol=1e-7)


def test_preset_unknown_raises():
    """An unrecognised preset name raises ValueError."""
    with pytest.raises(ValueError, match="unknown init_preset"):
        HebbianSocialGraph.build_preset_matrix("mesh", 3)


def test_preset_star_hub_out_of_range_raises():
    """'star' with hub index outside [0, N) raises ValueError."""
    with pytest.raises(ValueError, match="preset_hub"):
        HebbianSocialGraph.build_preset_matrix("star", 3, hub=3)


def test_init_preset_config_wiring(hcfg):
    """Constructing with init_preset/preset_* config fields builds that topology."""
    g = HebbianSocialGraph(hcfg(
        init_preset="star", preset_hub=2,
        preset_bond_strong=0.9, preset_bond_weak=0.2,
    ))
    expected = np.full((3, 3), 0.2, dtype=np.float32)
    expected[:, 2] = 0.9
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(g.W, expected, rtol=1e-7)


def test_init_matrix_used_verbatim(hcfg):
    """An explicit in-range init_matrix with zero diagonal is loaded verbatim."""
    m = [[0.0, 0.3, 0.6], [0.2, 0.0, 0.4], [0.9, 0.7, 0.0]]
    g = HebbianSocialGraph(hcfg(init_matrix=m))
    np.testing.assert_allclose(g.W, np.array(m, dtype=np.float32), rtol=1e-7)


def test_init_matrix_wrong_shape_raises(hcfg):
    """init_matrix whose shape differs from (N, N) raises ValueError."""
    with pytest.raises(ValueError, match="init_matrix shape"):
        HebbianSocialGraph(hcfg(init_matrix=[[0.0, 0.5], [0.5, 0.0]]))


def test_init_matrix_clipped_and_diagonal_zeroed(hcfg):
    """init_matrix values are clipped to [0, 1] and the diagonal is forced to 0."""
    m = [[0.5, 1.7, -0.3], [0.2, 0.5, 1.0], [2.0, -1.0, 0.5]]
    g = HebbianSocialGraph(hcfg(init_matrix=m))
    expected = np.array(
        [[0.0, 1.0, 0.0], [0.2, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    np.testing.assert_allclose(g.W, expected, rtol=1e-7)


def test_gated_mode_requires_positive_init_weight(hcfg):
    """Gated modes reject init_weight <= 0 (W=0 is a fixed point of the rule)."""
    with pytest.raises(ValueError, match="init_weight"):
        HebbianSocialGraph(hcfg(init_weight=0.0))


# ── reward diffusion (Eq. 8) ─────────────────────────────────────────────────

def test_diffuse_rewards_hand_case(hcfg):
    """Eq. 8 hand case: W rows [.8,.2]/[.5,.5]/[0,0], c=1, gamma=.2 -> [1.24, 2.0, 2.4]."""
    g = HebbianSocialGraph(hcfg(reward_diffusion_gamma=0.2))
    g.W = np.array(
        [[0.0, 0.8, 0.2], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    cij = np.ones((3, 3), dtype=np.float32)
    out = g.diffuse_rewards([1.0, 2.0, 3.0], co_activity_matrix=cij)
    # r'0 = .8*1 + .2*(0.8*2 + 0.2*3) = 1.24 ; r'1 = .8*2 + .2*(.5*1 + .5*3) = 2.0
    # r'2 = .8*3 = 2.4 (empty row -> zero normalised weights -> no social term)
    assert out == pytest.approx([1.24, 2.0, 2.4], abs=1e-6)


def test_diffuse_rewards_uses_last_coactivity_fallback(hcfg):
    """With no matrix argument, diffuse_rewards falls back to self._last_coactivity."""
    g = HebbianSocialGraph(hcfg(reward_diffusion_gamma=0.2))
    g.W = np.array(
        [[0.0, 0.8, 0.2], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    g._last_coactivity = np.ones((3, 3), dtype=np.float32)
    assert g.diffuse_rewards([1.0, 2.0, 3.0]) == pytest.approx(
        [1.24, 2.0, 2.4], abs=1e-6
    )


def test_diffuse_rewards_gamma_zero_identity(hcfg):
    """gamma=0 returns the raw rewards unchanged (as a new list)."""
    g = HebbianSocialGraph(hcfg(reward_diffusion_gamma=0.0))
    g._last_coactivity = np.ones((3, 3), dtype=np.float32)
    raw = [1.0, -2.0, 3.5]
    out = g.diffuse_rewards(raw)
    assert out == raw and out is not raw


def test_diffuse_rewards_without_coactivity_returns_raw(hcfg):
    """No matrix argument and no stored co-activity -> raw rewards pass through."""
    g = HebbianSocialGraph(hcfg(reward_diffusion_gamma=0.2))
    assert g._last_coactivity is None
    assert g.diffuse_rewards([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


# ── weight accessors ─────────────────────────────────────────────────────────

def test_get_normalized_weights_row(hcfg):
    """get_normalized_weights(i) returns w_ij / sum_k w_ik with self entry 0."""
    g = HebbianSocialGraph(hcfg())
    g.W = np.array(
        [[0.0, 0.8, 0.2], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    w_bar = g.get_normalized_weights(0)
    assert w_bar[0] == 0.0
    assert float(w_bar.sum()) == pytest.approx(1.0, abs=1e-6)
    assert w_bar == pytest.approx([0.0, 0.8, 0.2], abs=1e-6)


def test_get_normalized_weights_zero_row(hcfg):
    """An all-zero outgoing row normalises to all zeros (epsilon guard, no NaN)."""
    g = HebbianSocialGraph(hcfg())
    g.W = np.zeros((3, 3), dtype=np.float32)
    np.testing.assert_array_equal(g.get_normalized_weights(2), np.zeros(3))


def test_get_all_weights_returns_copy(hcfg):
    """Mutating the array returned by get_all_weights does not touch graph.W."""
    g = HebbianSocialGraph(hcfg())
    W_copy = g.get_all_weights()
    W_copy[0, 1] = 0.77
    assert g.get_weight(0, 1) == pytest.approx(0.1, rel=1e-6)


# ── social replay (Eq. 7) ────────────────────────────────────────────────────

def test_social_replay_rho_zero_empty(hcfg):
    """rho=0 (explicit or the config default 0.0) yields no replay indices."""
    g = HebbianSocialGraph(hcfg())
    assert g.get_social_replay_indices(0, [10, 10, 10], rho=0.0) == []
    assert g.config.social_replay_rho == 0.0
    assert g.get_social_replay_indices(0, [10, 10, 10]) == []


def test_social_replay_excludes_weak_bonds(hcfg):
    """Bonds below the 0.05 sparsity threshold are excluded; only j=2 is sampled."""
    g = HebbianSocialGraph(hcfg())
    g.W = np.array(
        [[0.0, 0.04, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    pairs = g.get_social_replay_indices(0, [10, 10, 10], rho=0.5)
    # total neighbour samples = int(.5*10) = 5; w_bar[2] = .5/.54 -> int(4.63) = 4
    assert len(pairs) == 4
    assert all(j == 2 for _, j in pairs)
    assert all(0 <= idx < 10 for idx, _ in pairs)


# ── graph metrics ────────────────────────────────────────────────────────────

def test_graph_metrics_known_matrix(hcfg):
    """get_graph_metrics computes mean bond, sparsity(<0.1), top pairs, strengths."""
    g = HebbianSocialGraph(hcfg(), agent_roles=[0, 0, 1])
    Wm = np.array(
        [[0.0, 0.8, 0.2], [0.5, 0.0, 0.4], [0.0, 0.05, 0.0]], dtype=np.float32
    )
    g.W = Wm
    m = g.get_graph_metrics()
    assert set(m) == {
        "mean_bond_strength", "sparsity", "top_3_pairs",
        "per_agent_out_strength", "modularity_proxy", "ltd_heatmap", "W",
    }
    assert m["mean_bond_strength"] == pytest.approx(1.95 / 6, rel=1e-6)
    assert m["sparsity"] == pytest.approx(2 / 6)  # 0.0 and 0.05 are < 0.1
    assert [(p["i"], p["j"]) for p in m["top_3_pairs"]] == [(0, 1), (1, 0), (1, 2)]
    assert [p["w"] for p in m["top_3_pairs"]] == pytest.approx(
        [0.8, 0.5, 0.4], rel=1e-6
    )
    assert m["per_agent_out_strength"] == pytest.approx([1.0, 0.9, 0.05], rel=1e-6)
    # roles [0,0,1]: within = (.8+.5)/2 = .65 ; cross = (.2+0+.4+.05)/4 = .1625
    assert m["modularity_proxy"] == pytest.approx(0.4875, abs=1e-6)
    np.testing.assert_array_equal(np.array(m["ltd_heatmap"]), np.zeros((3, 3)))
    np.testing.assert_allclose(np.array(m["W"], dtype=np.float32), Wm, rtol=1e-7)


# ── serialisation ────────────────────────────────────────────────────────────

def test_to_dict_from_dict_roundtrip(hcfg):
    """from_dict restores W, counters and last co-activity from to_dict exactly."""
    g = HebbianSocialGraph(hcfg())
    _run_gated_steps(g, 5)
    assert g.get_weight(0, 1) > 0.1001  # W actually moved off the init value

    d = g.to_dict()
    assert set(d) == {
        "enabled", "frozen", "mode", "W", "_failure_coactivation",
        "_max_reward_seen", "_max_bond_reward_seen", "_step_count",
        "num_agents", "init_weight", "_last_coactivity",
    }
    assert d["enabled"] is True
    assert d["frozen"] is False
    assert d["mode"] == "reward_modulated"
    assert d["num_agents"] == 3
    assert d["_step_count"] == 5
    assert d["_max_bond_reward_seen"] == 3.0
    # Gated modes never touch the legacy running max:
    assert d["_max_reward_seen"] == 0.0
    assert d["W"] == g.W.tolist()

    g2 = HebbianSocialGraph(hcfg())
    g2.from_dict(d)
    np.testing.assert_array_equal(g2.W, g.W)
    assert g2._step_count == 5
    assert g2._max_bond_reward_seen == 3.0
    np.testing.assert_array_equal(g2._last_coactivity, g._last_coactivity)
    np.testing.assert_array_equal(g2._failure_coactivation, np.zeros((3, 3)))
    assert len(g2._coact_window) == 0 and len(g2._reward_window) == 0


def test_snapshot_compact_fields(hcfg):
    """snapshot() returns exactly {W, step, max_reward_seen, num_agents}."""
    g = HebbianSocialGraph(hcfg())
    _run_gated_steps(g, 2)
    s = g.snapshot()
    assert set(s) == {"W", "step", "max_reward_seen", "num_agents"}
    assert s["step"] == 2
    assert s["num_agents"] == 3
    assert s["max_reward_seen"] == 0.0  # gated mode leaves the legacy max at 0
    assert s["W"] == g.W.tolist()


# ── reset / bond-delta window ────────────────────────────────────────────────

def test_reset_restores_init_state(hcfg):
    """reset() rebuilds W from init_weight and zeroes counters/windows/co-activity."""
    g = HebbianSocialGraph(hcfg())
    _run_gated_steps(g, 3)
    g.reset()
    expected = np.full((3, 3), 0.1, dtype=np.float32)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(g.W, expected, rtol=1e-7)
    assert g._step_count == 0
    assert g._max_reward_seen == 0.0
    assert g._max_bond_reward_seen == 0.0
    assert g._last_coactivity is None
    np.testing.assert_array_equal(g._failure_coactivation, np.zeros((3, 3)))
    assert len(g._failure_window_buffer) == 0
    assert len(g._coact_window) == 0
    assert len(g._reward_window) == 0


def test_reset_clears_w_history(hcfg):
    """FIXED BUG: reset() clears _W_history and re-seeds it with the fresh W
    (mirroring __init__), so bond deltas right after a reset are zeros instead
    of being computed against stale pre-reset snapshots."""
    g = HebbianSocialGraph(hcfg())
    _run_gated_steps(g, 55)  # roll the maxlen-50 history past the init snapshot
    g.reset()
    assert len(g._W_history) == 1  # fresh seed snapshot only
    assert g.bond_delta_row(0) == {1: 0.0, 2: 0.0}


def test_bond_delta_row_short_history_zeros(hcfg):
    """With fewer than 2 W snapshots, bond_delta_row returns all-zero deltas."""
    g = HebbianSocialGraph(hcfg())
    assert g.bond_delta_row(0) == {1: 0.0, 2: 0.0}


def test_bond_delta_row_after_updates(hcfg):
    """After k<window steps, bond_delta_row(i) == W_now[i] - W_at_construction[i]."""
    g = HebbianSocialGraph(hcfg())
    W0 = g.get_all_weights()
    _run_gated_steps(g, 3)
    deltas = g.bond_delta_row(0)
    assert set(deltas) == {1, 2}
    for j in (1, 2):
        assert deltas[j] == float(g.W[0, j] - W0[0, j])
        assert deltas[j] > 0.0  # co-active bonds grew


# ── legacy mode ──────────────────────────────────────────────────────────────

def test_legacy_update_grows_coactive_bond(hcfg):
    """Legacy update potentiates co-located rewarded pairs and leaves far pairs flat."""
    g = HebbianSocialGraph(hcfg(mode="legacy"))
    g.update(
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        step_rewards=[1.0, 1.0, 1.0],
    )
    # signal ~= 1 -> ltp = .01*tanh(1); c01 = .5*.5; dW = (ltp+base_ltp)*c*(1-W)
    expected = 0.1 + (0.01 * math.tanh(1.0) + 0.005) * 0.25 * 0.9
    assert g.get_weight(0, 1) == pytest.approx(expected, rel=1e-5)
    assert g.get_weight(1, 0) == pytest.approx(expected, rel=1e-5)
    # Agent 2 is out of range, no comm -> c=0 and decay=0 -> bond unchanged.
    assert g.get_weight(0, 2) == pytest.approx(0.1, rel=1e-6)
    assert g._step_count == 1


def test_legacy_update_requires_step_rewards(hcfg):
    """Legacy mode raises ValueError when update is called without step_rewards."""
    g = HebbianSocialGraph(hcfg(mode="legacy"))
    with pytest.raises(ValueError, match="legacy mode requires step_rewards"):
        g.update(positions=[(0.0, 0.0, 0.0)] * 3)


# ── disabled no-op surface ───────────────────────────────────────────────────

def test_disabled_graph_noops(hcfg):
    """Disabled graph: every public method is a safe no-op with zero defaults."""
    g = HebbianSocialGraph(hcfg(enabled=False))
    assert g.update(positions=[None] * 3, step_rewards=[1.0, 1.0, 1.0]) is None
    assert g.get_weight(0, 1) == 0.0
    np.testing.assert_array_equal(g.get_all_weights(), np.zeros((3, 3)))
    np.testing.assert_array_equal(g.get_normalized_weights(0), np.zeros(3))
    assert g.get_graph_metrics() == {}
    assert g.to_dict() == {"enabled": False}
    assert g.snapshot() == {
        "W": None, "step": 0, "max_reward_seen": 0.0, "num_agents": 3,
    }
    assert g.bond_delta_row(0) == {1: 0.0, 2: 0.0}
    assert g.get_social_replay_indices(0, [10, 10, 10], rho=0.5) == []
    assert g.diffuse_rewards([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(g.get_ltd_heatmap(), np.zeros((3, 3)))
    g.reset()  # must not raise
    g.from_dict({"enabled": True, "W": [[0.0]]})  # must not raise or set state
    assert not hasattr(g, "W")
