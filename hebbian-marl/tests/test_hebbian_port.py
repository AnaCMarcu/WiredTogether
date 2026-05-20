"""Verify the ported Hebbian module behaves as expected.

These tests cover the Phase 1 verification gate from HEBBIAN_MARL_PLAN.md:
  - W stays in [0, 1], diagonal is zero, mean off-diagonal grows under
    cooperative-like updates.
  - `enabled=False` returns sensible no-op values.
  - `diffuse_rewards` is identity when `reward_diffusion_gamma=0.0`
    (the actual signature takes no gamma kwarg — it reads config).
  - `to_dict` → `from_dict` roundtrip preserves W exactly.
  - Passing non-empty `comm_events` produces a different W than
    `comm_events=None` after the same number of steps with the same
    deterministic inputs (confirms the comm path is wired through).
"""

import numpy as np
import pytest

from hebbian_module import HebbianConfig, HebbianSocialGraph


def _cooperative_step(N=3, radius=5.0):
    """Three agents within interaction radius, all earning positive reward.

    Yields the kind of co-active step that should strengthen bonds.
    """
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    step_rewards = [1.0] * N
    advantages = [0.5] * N
    return positions, step_rewards, advantages


def test_disabled_is_noop():
    cfg = HebbianConfig(enabled=False, num_agents=3)
    g = HebbianSocialGraph(cfg)

    assert g.update(
        positions=[(0.0, 0.0, 0.0)] * 3,
        step_rewards=[1.0] * 3,
        advantages=[0.5] * 3,
        comm_events=[(0, 1)],
    ) is None
    assert g.get_weight(0, 1) == 0.0
    assert np.all(g.get_all_weights() == 0.0)
    assert np.all(g.get_normalized_weights(0) == 0.0)
    assert g.diffuse_rewards([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert g.get_graph_metrics() == {}
    assert g.to_dict() == {"enabled": False}


def test_w_bounded_and_diagonal_zero_after_random_updates():
    cfg = HebbianConfig(enabled=True, num_agents=3, log_graph_every=10**9)
    g = HebbianSocialGraph(cfg)
    rng = np.random.default_rng(42)

    for _ in range(100):
        positions = [
            tuple(rng.uniform(-3, 3, size=3).astype(float))
            for _ in range(3)
        ]
        step_rewards = rng.uniform(-1, 1, size=3).tolist()
        advantages = rng.uniform(-1, 1, size=3).tolist()
        # Random sender/receiver pair some of the time
        comm_events = None
        if rng.random() < 0.3:
            i = int(rng.integers(0, 3))
            j = (i + 1 + int(rng.integers(0, 2))) % 3
            comm_events = [(i, j)]
        g.update(positions, step_rewards, advantages, comm_events)

    W = g.get_all_weights()
    assert W.shape == (3, 3)
    assert np.all(W >= 0.0), f"W has negative entries: {W}"
    assert np.all(W <= 1.0), f"W has entries above 1.0: {W}"
    assert np.all(np.diag(W) == 0.0), "diagonal of W must be zero"


def test_mean_offdiag_grows_under_cooperative_updates():
    cfg = HebbianConfig(enabled=True, num_agents=3, log_graph_every=10**9)
    g = HebbianSocialGraph(cfg)
    positions, step_rewards, advantages = _cooperative_step()

    # 50 cooperative steps with all agents within radius and positive reward
    for _ in range(50):
        g.update(positions, step_rewards, advantages, comm_events=None)

    W = g.get_all_weights()
    mask = ~np.eye(3, dtype=bool)
    off_diag = W[mask]
    assert off_diag.mean() > 0.05, (
        f"expected mean off-diag W > 0.05 after 50 cooperative steps, "
        f"got {off_diag.mean():.4f}; W=\n{W}"
    )


def test_diffuse_rewards_identity_when_gamma_zero():
    """When reward_diffusion_gamma=0, diffuse_rewards must return raw_rewards.

    Note: HebbianSocialGraph.diffuse_rewards has no gamma kwarg — it reads
    config.reward_diffusion_gamma. So we set gamma=0 in the config.
    """
    cfg = HebbianConfig(enabled=True, num_agents=3, reward_diffusion_gamma=0.0,
                       log_graph_every=10**9)
    g = HebbianSocialGraph(cfg)
    # Even after some updates, gamma=0 must short-circuit to identity.
    positions, step_rewards, advantages = _cooperative_step()
    for _ in range(10):
        g.update(positions, step_rewards, advantages, comm_events=None)

    raw = [0.7, -0.3, 1.5]
    diffused = g.diffuse_rewards(raw)
    assert diffused == raw


def test_to_dict_from_dict_roundtrip_preserves_w():
    cfg = HebbianConfig(enabled=True, num_agents=3, log_graph_every=10**9)
    src = HebbianSocialGraph(cfg)
    positions, step_rewards, advantages = _cooperative_step()
    for _ in range(20):
        src.update(positions, step_rewards, advantages, comm_events=[(0, 2)])

    snap = src.to_dict()
    # Build a fresh graph and restore from snapshot
    dst = HebbianSocialGraph(cfg)
    dst.from_dict(snap)
    np.testing.assert_array_equal(src.W, dst.W)
    np.testing.assert_array_equal(src._failure_coactivation, dst._failure_coactivation)
    assert src._max_reward_seen == dst._max_reward_seen
    assert src._step_count == dst._step_count


def test_comm_events_path_changes_w():
    """Confirm the comm path is wired through.

    Two graphs run with identical deterministic inputs, except one
    receives `comm_events=[(0, 2)]` each step and the other receives
    `comm_events=None`. Agents 0 and 2 are placed OUTSIDE the interaction
    radius so the only path for them to become co-active is the comm
    bonus. The two final W matrices must differ.
    """
    cfg = HebbianConfig(
        enabled=True, num_agents=3,
        interaction_radius=1.0,         # tight: only 0-1 are co-active spatially
        communication_coactivity_bonus=0.5,
        log_graph_every=10**9,
    )
    g_no_comm = HebbianSocialGraph(cfg)
    g_with_comm = HebbianSocialGraph(cfg)

    positions = [
        (0.0, 0.0, 0.0),    # 0 and 1 within radius
        (0.5, 0.5, 0.0),
        (5.0, 5.0, 0.0),    # 2 far away from 0 and 1
    ]
    step_rewards = [1.0, 1.0, 1.0]
    advantages = [0.5, 0.5, 0.5]

    for _ in range(50):
        g_no_comm.update(positions, step_rewards, advantages, comm_events=None)
        g_with_comm.update(positions, step_rewards, advantages, comm_events=[(0, 2)])

    W_no = g_no_comm.get_all_weights()
    W_yes = g_with_comm.get_all_weights()

    # The pair (0, 2) and (2, 0) must differ between the two runs
    assert not np.isclose(W_no[0, 2], W_yes[0, 2]), (
        f"W[0,2] identical with and without comm: {W_no[0,2]} vs {W_yes[0,2]}"
    )
    assert not np.isclose(W_no[2, 0], W_yes[2, 0]), (
        f"W[2,0] identical with and without comm: {W_no[2,0]} vs {W_yes[2,0]}"
    )
    # And W_yes should have a STRONGER 0↔2 bond than W_no (comm built it up)
    assert W_yes[0, 2] > W_no[0, 2], (
        f"expected comm to raise W[0,2]: {W_yes[0,2]} <= {W_no[0,2]}"
    )


def test_normalized_weights_sum_to_one_when_bonds_exist():
    cfg = HebbianConfig(enabled=True, num_agents=3, log_graph_every=10**9)
    g = HebbianSocialGraph(cfg)
    positions, step_rewards, advantages = _cooperative_step()
    for _ in range(30):
        g.update(positions, step_rewards, advantages, comm_events=None)

    # Each agent has at least some bond strength to teammates -> w_bar sums to ~1
    for i in range(3):
        w_bar = g.get_normalized_weights(i)
        assert w_bar.shape == (3,)
        assert w_bar[i] == 0.0
        assert abs(w_bar.sum() - 1.0) < 1e-5, (
            f"agent {i}: normalized weights should sum to ~1, got {w_bar.sum()}"
        )


# ─────────────────────────────────────────────────────────────────────
# Failure-grace mechanism (HebbianConfig.failure_grace_enabled)
# ─────────────────────────────────────────────────────────────────────


def _failing_step(N=3, radius=5.0):
    """Three agents within interaction radius, positive rewards so engagement
    fires, but strongly negative advantages so At_team < -ltd_threshold and
    the failure-window / grace paths both engage."""
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    step_rewards = [1.0] * N
    advantages = [-0.5] * N
    return positions, step_rewards, advantages


def test_grace_disabled_is_identical_to_legacy():
    """Setting non-default values for failure_grace_threshold and
    failure_ltp_lr must NOT change behavior while failure_grace_enabled=False.
    Locks down the no-op-when-disabled contract — important because grace
    is now the canonical default, so the disabled path is the regression
    check for the legacy pre-grace math."""
    cfg_a = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=False,
    )
    cfg_b = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=False,
        failure_grace_threshold=0.7,
        failure_ltp_lr=0.5,
    )
    g_a = HebbianSocialGraph(cfg_a)
    g_b = HebbianSocialGraph(cfg_b)
    rng = np.random.default_rng(7)

    for _ in range(100):
        positions = [tuple(rng.uniform(-3, 3, size=3).tolist()) for _ in range(3)]
        step_rewards = rng.uniform(-1, 1, size=3).tolist()
        advantages = rng.uniform(-1, 1, size=3).tolist()
        g_a.update(positions, step_rewards, advantages, None)
        g_b.update(positions, step_rewards, advantages, None)

    np.testing.assert_array_equal(g_a.W, g_b.W)


def test_grace_bonus_lifts_w_on_first_failure():
    """One forced-failure step with grace enabled must produce strictly higher
    off-diagonal W than the same step with grace disabled. This is the
    headline behavioral test: the bonus actively grows bonds on failure."""
    cfg_legacy = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=False,
    )
    cfg_grace = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=True,
    )
    g_legacy = HebbianSocialGraph(cfg_legacy)
    g_grace = HebbianSocialGraph(cfg_grace)
    positions, step_rewards, advantages = _failing_step()

    g_legacy.update(positions, step_rewards, advantages, None)
    g_grace.update(positions, step_rewards, advantages, None)

    mask = ~np.eye(3, dtype=bool)
    assert (g_grace.W[mask] > g_legacy.W[mask]).all(), (
        f"grace failed to lift W on first failure step:\n"
        f"legacy={g_legacy.W}\ngrace={g_grace.W}"
    )


def test_grace_w_grows_above_legacy_during_grace_window():
    """Cumulative version of the first-step test. After 10 forced-failure
    steps (well inside the grace window — threshold=0.3, window=50, so F_ij
    crosses 0.3 around step 15), the grace graph's bonds must be measurably
    above the legacy graph's."""
    cfg_legacy = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=False,
    )
    cfg_grace = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=True,
    )
    g_legacy = HebbianSocialGraph(cfg_legacy)
    g_grace = HebbianSocialGraph(cfg_grace)
    positions, step_rewards, advantages = _failing_step()

    for _ in range(10):
        g_legacy.update(positions, step_rewards, advantages, None)
        g_grace.update(positions, step_rewards, advantages, None)

    mask = ~np.eye(3, dtype=bool)
    legacy_mean = g_legacy.W[mask].mean()
    grace_mean = g_grace.W[mask].mean()
    assert grace_mean > legacy_mean + 0.01, (
        f"grace failed to accumulate advantage over 10 failure steps: "
        f"legacy mean off-diag={legacy_mean:.4f}, grace mean off-diag={grace_mean:.4f}"
    )


def test_grace_threshold_zero_is_safe():
    """failure_grace_threshold=0 must not divide by zero. Construction and one
    failing step both must finish without raising; W must stay finite and
    in bounds."""
    cfg = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=True,
        failure_grace_threshold=0.0,
    )
    g = HebbianSocialGraph(cfg)
    positions, step_rewards, advantages = _failing_step()

    g.update(positions, step_rewards, advantages, None)  # must not raise

    W = g.get_all_weights()
    assert np.all(np.isfinite(W)), f"W has non-finite entries: {W}"
    assert np.all(W >= 0.0) and np.all(W <= 1.0), f"W out of bounds: {W}"


def test_to_dict_from_dict_roundtrip_with_grace():
    """to_dict/from_dict must work for grace-enabled graphs. No new state is
    serialised today, but pin the contract so a future state addition doesn't
    silently break checkpoint loading."""
    cfg = HebbianConfig(
        enabled=True, num_agents=3, log_graph_every=10**9,
        failure_grace_enabled=True,
        failure_grace_threshold=0.3,
        failure_ltp_lr=0.015,
    )
    src = HebbianSocialGraph(cfg)
    positions, step_rewards, advantages = _failing_step()
    for _ in range(20):
        src.update(positions, step_rewards, advantages, comm_events=[(0, 2)])

    snap = src.to_dict()
    dst = HebbianSocialGraph(cfg)
    dst.from_dict(snap)
    np.testing.assert_array_equal(src.W, dst.W)
    np.testing.assert_array_equal(src._failure_coactivation, dst._failure_coactivation)
    assert src._max_reward_seen == dst._max_reward_seen
    assert src._step_count == dst._step_count
