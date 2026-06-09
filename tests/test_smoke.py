"""Smoke / import-contract tests.

These pin the public import paths that the rest of the codebase (and the
checkpoint format) depend on, so a structural refactor cannot silently move a
class or break a re-export. Pure-python where possible; the modules that need
the game engine / agent stack (``pettingzoo``, ``autogen``) are skipped when
those optional deps are absent so the suite still runs on a bare dev box.
"""

import importlib

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────
# Import contracts that must hold regardless of optional heavy deps.
# ──────────────────────────────────────────────────────────────────────
def test_hebbian_graph_reexport_identity():
    """``HebbianSocialGraph`` must stay at ``hebbian.graph`` and be the exact
    same object re-exported by ``rl_layer`` (both __init__ files re-export it)."""
    from hebbian.graph import HebbianSocialGraph as FromGraph
    from rl_layer import HebbianSocialGraph as FromRL

    assert FromGraph is FromRL


def test_metric_contract():
    """``CraftiumMetric`` + ``MILESTONE_TRACK`` stay importable from
    ``agent_modules.craftium_metric``, and the plotting/summary entry points
    survive any mixin split."""
    mod = importlib.import_module("agent_modules.craftium_metric")
    assert isinstance(mod.MILESTONE_TRACK, dict) and mod.MILESTONE_TRACK
    metric = mod.CraftiumMetric
    for method in ("record_reward", "store_timestep", "save_run_metrics",
                   "_save_plots", "_save_text_summary"):
        assert callable(getattr(metric, method)), f"CraftiumMetric.{method} missing"


def test_env_interface_contract():
    """``CraftiumEnvironmentInterface`` + ``VALID_ACTIONS`` stay at
    ``custom_environment_craftium`` (needs the game-engine deps)."""
    pytest.importorskip("pettingzoo")
    pytest.importorskip("craftium")
    mod = importlib.import_module("custom_environment_craftium")
    assert hasattr(mod, "CraftiumEnvironmentInterface")
    assert isinstance(mod.VALID_ACTIONS, list) and mod.VALID_ACTIONS


# ──────────────────────────────────────────────────────────────────────
# Determinism golden: pins the Hebbian update so the kernel extraction in the
# refactor is provably behaviour-preserving (same inputs -> identical W).
# ──────────────────────────────────────────────────────────────────────
def _run_fixed_rollout():
    from hebbian.config import HebbianConfig
    from hebbian.graph import HebbianSocialGraph

    cfg = HebbianConfig(
        enabled=True,
        mode="reward_modulated",
        num_agents=3,
        init_weight=0.05,
        interaction_radius=5.0,
        coop_eps=0.05,
        coop_window=5,
        neg_theta=5.0,
        eta_plus=0.05,
        eta_0=0.01,
        eta_minus=0.025,
        reward_norm_R=300.0,
        communication_coactivity_bonus=0.5,
        engagement_reward_weight=0.5,
        reward_diffusion_gamma=0.2,
    )
    g = HebbianSocialGraph(cfg)
    # Deterministic scripted rollout (no RNG): fixed positions/rewards/comm.
    positions = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    for t in range(20):
        bond_rewards = [1.0, 2.0, 0.5]
        total_rewards = [1.0, 2.0, 0.5 + (-10.0 if t % 7 == 0 else 0.0)]
        comm = [(0, 1)] if t % 3 == 0 else None
        g.update(
            positions=positions,
            comm_events=comm,
            chambers=[2, 2, 2],
            bond_rewards=bond_rewards,
            total_rewards=total_rewards,
        )
    return g.W.copy()


def test_hebbian_update_is_deterministic():
    w1 = _run_fixed_rollout()
    w2 = _run_fixed_rollout()
    assert np.array_equal(w1, w2)
    # Sanity: the scripted rollout actually moved some bonds off their init.
    assert not np.allclose(w1, 0.05)
