"""Tests for the Hebbian Social Plasticity module.

All tests use numpy only — no torch, no environment dependency.
Each test docstring cites the thesis property being verified.
"""

import numpy as np
import pytest

from rl_layer.hebbian_config import HebbianConfig
from rl_layer.hebbian_graph import HebbianSocialGraph


# ── Helpers ──

def _make_graph(N=3, **overrides):
    """Create an enabled HebbianSocialGraph with sensible test defaults."""
    defaults = dict(enabled=True, num_agents=N)
    defaults.update(overrides)
    cfg = HebbianConfig(**defaults)
    return HebbianSocialGraph(cfg, agent_roles=[i % 3 for i in range(N)])


def _close_positions(N=3, spacing=1.0):
    """Return N positions all within default interaction radius."""
    return [(i * spacing, 0.0, 0.0) for i in range(N)]


def _far_positions(N=3, spacing=100.0):
    """Return N positions all far apart (>> default radius)."""
    return [(i * spacing, 0.0, 0.0) for i in range(N)]


# ══════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════


class TestWeightsBoundedUnderExtremeRewards:
    """Thesis property: W ∈ [0,1] is a hard invariant."""

    def test_weights_bounded_under_extreme_rewards(self):
        """Run 200 steps with alternating +1e6 / -1e6 rewards.
        W must stay in [0,1] throughout."""
        g = _make_graph(N=3, ltp_lr=0.1, ltd_lr=0.1)
        positions = _close_positions(3)
        for step in range(200):
            sign = 1.0 if step % 2 == 0 else -1.0
            rewards = [sign * 1e6] * 3
            g.update(positions, rewards)
            assert g.W.min() >= 0.0, f"W below 0 at step {step}"
            assert g.W.max() <= 1.0, f"W above 1 at step {step}"


class TestDiagonalAlwaysZero:
    """Thesis property: W[i,i] = 0 (no self-bonds)."""

    def test_diagonal_always_zero(self):
        """After any update, diagonal must be zero."""
        g = _make_graph(N=3)
        positions = _close_positions(3)
        for _ in range(50):
            g.update(positions, [10.0, 20.0, 30.0])
        for i in range(3):
            assert g.W[i, i] == 0.0


class TestCoactivityZeroForDistantAgents:
    """Thesis Eq. 2: spatial gate is zero when agents are far apart."""

    def test_coactivity_zero_for_distant_agents(self):
        """Two agents 100 units apart (>> radius=5). cij must be 0,
        so no bond growth regardless of reward."""
        g = _make_graph(N=2, interaction_radius=5.0)
        positions = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        initial_W = g.W.copy()
        for _ in range(50):
            g.update(positions, [100.0, 100.0])
        # Without communication, distant agents should have zero co-activity
        # and thus zero bond growth (only decay applies)
        assert g.W[0, 1] <= initial_W[0, 1] + 1e-6


class TestCoactivityCommBondAcrossDistance:
    """Thesis Eq. 2: communication bonus enables bonds across distance."""

    def test_coactivity_comm_bond_across_distance(self):
        """Two agents far apart, δ_comm > 0, comm_events=[(0,1)].
        Bond should grow after positive-advantage steps."""
        g = _make_graph(
            N=2,
            interaction_radius=5.0,
            communication_coactivity_bonus=0.5,
            ltp_lr=0.05,
        )
        positions = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        for _ in range(50):
            g.update(positions, [10.0, 10.0], comm_events=[(0, 1)])
        assert g.W[0, 1] > 0.01, "Bond should have grown via comm co-activity"


class TestCommBondDisabledForRQ4:
    """Thesis RQ4 ablation: δ_comm=0 disables communication co-activity."""

    def test_comm_bond_disabled_for_rq4(self):
        """Same setup as comm bond test but communication_coactivity_bonus=0.
        Distant agents should have no bond growth even with comm_events."""
        g = _make_graph(
            N=2,
            interaction_radius=5.0,
            communication_coactivity_bonus=0.0,
            ltp_lr=0.05,
        )
        positions = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        for _ in range(50):
            g.update(positions, [10.0, 10.0], comm_events=[(0, 1)])
        # No spatial co-activity, no comm bonus → no growth, only decay
        assert g.W[0, 1] < 0.01


class TestLtpLtdAsymmetry:
    """Thesis Eq. 4+5: η_+ > η_- biases toward accumulation."""

    def test_ltp_ltd_asymmetry(self):
        """η_+ = 0.05, η_- = 0.01. Co-active pair, alternating +1/-1.
        After 50 steps wij should be > 0.5 (net potentiation)."""
        g = _make_graph(
            N=2,
            ltp_lr=0.05,
            ltd_lr=0.01,
            init_weight=0.5,
            decay=0.0,  # disable decay to isolate LTP/LTD
        )
        positions = _close_positions(2, spacing=1.0)
        for step in range(50):
            reward = 10.0 if step % 2 == 0 else -10.0
            g.update(positions, [reward, reward])
        assert g.W[0, 1] > 0.5, (
            f"Expected net potentiation from asymmetry, got W={g.W[0,1]:.4f}"
        )


class TestSustainedLtdDissolvesRepeatedFailureBond:
    """Thesis sustained LTD: repeated co-failure dissolves bonds."""

    def test_sustained_ltd_dissolves_repeated_failure_bond(self):
        """Two co-active agents, all steps negative advantage.
        After window_size steps wij should decrease and Fij ≈ 1.0."""
        window = 10
        g = _make_graph(
            N=2,
            init_weight=0.5,
            failure_memory_window=window,
            ltd_sustained_lr=0.01,
            decay=0.0,
        )
        positions = _close_positions(2, spacing=1.0)
        for _ in range(window + 5):
            g.update(positions, [-10.0, -10.0])

        assert g.W[0, 1] < 0.5, "Bond should have decreased from sustained LTD"
        Fij = g.get_ltd_heatmap()
        # After window+5 negative steps, Fij should be close to 1
        assert Fij[0, 1] > 0.8, f"Expected Fij ≈ 1.0, got {Fij[0,1]:.2f}"


class TestRewardDiffusionIdentityAtGammaZero:
    """Thesis Eq. 8: γ=0 → diffusion is identity."""

    def test_reward_diffusion_identity_at_gamma_zero(self):
        """With reward_diffusion_gamma=0, diffuse_rewards returns raw."""
        g = _make_graph(N=3, reward_diffusion_gamma=0.0)
        raw = [1.0, 2.0, 3.0]
        diffused = g.diffuse_rewards(raw)
        for r, d in zip(raw, diffused):
            assert abs(r - d) < 1e-6


class TestRewardDiffusionSpreadsCredit:
    """Thesis Eq. 8: reward diffusion transfers credit across bonds."""

    def test_reward_diffusion_spreads_credit(self):
        """Agent 0 earns 10, agent 1 earns 0. Strong bond W[1,0]=0.8.
        With γ=0.5, agent 1 should receive diffused credit > 0."""
        g = _make_graph(N=2, reward_diffusion_gamma=0.5, init_weight=0.0)
        g.W[1, 0] = 0.8
        g.W[0, 1] = 0.0

        # Provide a co-activity matrix where both agents are co-active
        cij = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        g._last_coactivity = cij

        raw = [10.0, 0.0]
        diffused = g.diffuse_rewards(raw, cij)
        assert diffused[1] > 0.0, (
            f"Expected credit to flow to agent 1, got {diffused[1]:.4f}"
        )


class TestSerialisationRoundtrip:
    """Serialisation: to_dict → from_dict preserves state."""

    def test_serialisation_roundtrip(self):
        """Run 50 steps, serialise, restore, check W equality."""
        g1 = _make_graph(N=3)
        positions = _close_positions(3)
        for step in range(50):
            r = [float(step % 5), float(step % 3), float(step % 7)]
            g1.update(positions, r)

        snapshot = g1.to_dict()

        g2 = _make_graph(N=3)
        g2.from_dict(snapshot)

        np.testing.assert_allclose(g1.W, g2.W, atol=1e-6)
        assert g2._max_reward_seen == g1._max_reward_seen
        assert g2._step_count == g1._step_count


class TestResetReinitialisesState:
    """Reset clears all accumulated state while preserving config."""

    def test_reset_reinitialises_state(self):
        """Run 100 steps, reset, verify clean state."""
        g = _make_graph(N=3, init_weight=0.1)
        positions = _close_positions(3)
        for _ in range(100):
            g.update(positions, [10.0, -5.0, 3.0])

        # W should have changed
        assert not np.allclose(g.W, 0.1)

        g.reset()

        # Off-diagonal should be init_weight, diagonal 0
        expected = np.full((3, 3), 0.1, dtype=np.float32)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(g.W, expected)
        np.testing.assert_allclose(
            g._failure_coactivation, np.zeros((3, 3), dtype=np.float32)
        )
        assert g._max_reward_seen == 0.0


class TestNoOpWhenDisabled:
    """Guard: HebbianConfig(enabled=False) → all methods are safe no-ops."""

    def test_no_op_when_disabled(self):
        """Call every public method with enabled=False. No exceptions,
        no state mutation."""
        cfg = HebbianConfig(enabled=False, num_agents=3)
        g = HebbianSocialGraph(cfg)

        # update → None
        result = g.update([(0, 0, 0)] * 3, [1.0] * 3)
        assert result is None

        # diffuse_rewards → raw unchanged
        raw = [1.0, 2.0, 3.0]
        assert g.diffuse_rewards(raw) == raw

        # get_graph_metrics → empty dict
        assert g.get_graph_metrics() == {}

        # get_normalized_weights → zeros
        w = g.get_normalized_weights(0)
        assert np.allclose(w, 0.0)

        # get_weight → 0
        assert g.get_weight(0, 1) == 0.0

        # get_all_weights → zeros
        assert np.allclose(g.get_all_weights(), 0.0)

        # get_social_replay_indices → empty
        assert g.get_social_replay_indices(0, [10, 10, 10]) == []

        # get_ltd_heatmap → zeros
        assert np.allclose(g.get_ltd_heatmap(), 0.0)

        # to_dict → minimal
        d = g.to_dict()
        assert d == {"enabled": False}

        # from_dict → no-op
        g.from_dict(d)

        # reset → no-op
        g.reset()


class TestModularityProxyReflectsRoleStructure:
    """Thesis RQ2: modularity proxy detects role-based clustering."""

    def test_modularity_proxy_reflects_role_structure(self):
        """3 agents with roles [0,1,2]. Force W with high within-role
        and low cross-role weights. modularity_proxy should be > 0.

        Note: with 3 agents and 3 distinct roles, there are no within-role
        pairs. So we use 6 agents with roles [0,0,1,1,2,2]."""
        cfg = HebbianConfig(enabled=True, num_agents=6, init_weight=0.0)
        roles = [0, 0, 1, 1, 2, 2]
        g = HebbianSocialGraph(cfg, agent_roles=roles)

        # Set within-role pairs to 0.9, cross-role to 0.1
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                if roles[i] == roles[j]:
                    g.W[i, j] = 0.9
                else:
                    g.W[i, j] = 0.1

        metrics = g.get_graph_metrics()
        assert metrics["modularity_proxy"] is not None
        assert metrics["modularity_proxy"] > 0.0, (
            f"Expected positive modularity, got {metrics['modularity_proxy']}"
        )
