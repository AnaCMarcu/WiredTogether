"""Verify the HebbianSEACLearner sharing-weight logic.

Covers the Phase 4 verification gate from HEBBIAN_MARL_PLAN.md:

  - `_sharing_weights()` returns row-normalised Hebbian weights when
    `weighted_sharing=True` and a graph is registered.
  - Returns uniform 1/(N-1) when `uniform_sharing=True`.
  - Returns zeros when both flags are off.
  - With a strongly asymmetric W = [[0,0.9,0.1],[0.9,0,0.1],[0.1,0.1,0]],
    agent 0's sharing weights put far more weight on agent 1 than agent 2.

These tests bypass the full PPOLearner constructor (which needs a real
MAC, critic, etc.) by constructing the learner via __new__ and setting
only the attributes _sharing_weights touches.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from hebbian_module import HebbianConfig, HebbianSocialGraph, clear_graph, set_graph  # noqa: E402
from learners.hebbian_seac_learner import (  # noqa: E402
    HebbianSEACLearner,
    _flag,
    _scalar,
)


def _bare_learner(n_agents=3, uniform=False, weighted=False) -> HebbianSEACLearner:
    """Build a HebbianSEACLearner skeleton sufficient for `_sharing_weights`."""
    learner = HebbianSEACLearner.__new__(HebbianSEACLearner)
    learner.n_agents = n_agents
    learner.uniform_sharing = uniform
    learner.weighted_sharing = weighted
    return learner


def _set_explicit_w(graph: HebbianSocialGraph, W: np.ndarray) -> None:
    """Force-set the graph's W matrix (for deterministic tests)."""
    graph.W = W.astype(np.float32).copy()
    np.fill_diagonal(graph.W, 0.0)


# ── flag/scalar helpers ──

def test_flag_reads_top_level_then_hebbian_block():
    from types import SimpleNamespace
    # Top-level wins
    args = SimpleNamespace(uniform_sharing=True, hebbian=SimpleNamespace(uniform_sharing=False))
    assert _flag(args, "uniform_sharing", False) is True
    # Falls back to hebbian block
    args2 = SimpleNamespace(hebbian=SimpleNamespace(reward_diffusion=True))
    assert _flag(args2, "reward_diffusion", False) is True
    # Default
    args3 = SimpleNamespace()
    assert _flag(args3, "weighted_sharing", False) is False


def test_scalar_reads_top_level_then_hebbian_block():
    from types import SimpleNamespace
    args = SimpleNamespace(lambda_share=0.3, hebbian=SimpleNamespace(lambda_share=0.7))
    assert _scalar(args, "lambda_share", 0.5) == 0.3
    args2 = SimpleNamespace(hebbian={"reward_diffusion_gamma": 0.25})
    assert _scalar(args2, "reward_diffusion_gamma", 0.99) == 0.25


# ── _sharing_weights ──

def test_sharing_weights_zero_when_both_flags_off():
    clear_graph()
    learner = _bare_learner(n_agents=3, uniform=False, weighted=False)
    W = learner._sharing_weights()
    assert W.shape == (3, 3)
    assert np.all(W == 0.0)


def test_sharing_weights_uniform_when_uniform_flag_set():
    clear_graph()
    learner = _bare_learner(n_agents=3, uniform=True, weighted=False)
    W = learner._sharing_weights()
    expected_off = 1.0 / 2.0
    assert np.allclose(np.diag(W), 0.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.isclose(W[i, j], expected_off), (
                    f"expected W[{i},{j}] = {expected_off}, got {W[i, j]}"
                )


def test_sharing_weights_weighted_reads_hebbian_graph():
    cfg = HebbianConfig(enabled=True, num_agents=3)
    graph = HebbianSocialGraph(cfg)
    _set_explicit_w(
        graph,
        np.array(
            [[0.0, 0.9, 0.1],
             [0.9, 0.0, 0.1],
             [0.1, 0.1, 0.0]]
        ),
    )
    set_graph(graph)

    learner = _bare_learner(n_agents=3, uniform=False, weighted=True)
    W = learner._sharing_weights()

    # Row-normalised expectation
    # Row 0: [0, 0.9, 0.1] / 1.0 = [0, 0.9, 0.1]
    # Row 1: [0.9, 0, 0.1] / 1.0 = [0.9, 0, 0.1]
    # Row 2: [0.1, 0.1, 0]  / 0.2 = [0.5, 0.5, 0]
    expected = np.array(
        [[0.0, 0.9, 0.1],
         [0.9, 0.0, 0.1],
         [0.5, 0.5, 0.0]]
    )
    np.testing.assert_allclose(W, expected, atol=1e-5)

    # Agent 0's sharing puts way more weight on agent 1 than agent 2:
    assert W[0, 1] > 5 * W[0, 2], (
        f"expected agent 0 to weight agent 1 >> agent 2; got {W[0]}"
    )
    clear_graph()


def test_sharing_weights_falls_back_to_uniform_when_graph_missing():
    """If `weighted_sharing=True` but no graph is registered, fall back to
    uniform rather than crash. The learner's constructor warned about this."""
    clear_graph()
    learner = _bare_learner(n_agents=3, uniform=False, weighted=True)
    W = learner._sharing_weights()
    # Falls back to uniform 1/(N-1)
    expected_off = 1.0 / 2.0
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.isclose(W[i, j], expected_off)


def test_sharing_weights_disabled_graph_falls_back_to_uniform():
    """Even if a graph is registered but its config says enabled=False,
    we fall back to uniform — the per-row W̄ would be zero otherwise and
    the cross-agent term would silently vanish."""
    cfg = HebbianConfig(enabled=False, num_agents=3)
    graph = HebbianSocialGraph(cfg)
    set_graph(graph)
    learner = _bare_learner(n_agents=3, uniform=False, weighted=True)
    W = learner._sharing_weights()
    expected_off = 1.0 / 2.0
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.isclose(W[i, j], expected_off)
    clear_graph()
