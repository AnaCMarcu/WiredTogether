"""Verify the HebbianRunner integration (a): reward diffusion + step hook.

Covers the Phase 3 verification gate from HEBBIAN_MARL_PLAN.md:

  - `_build_hebbian_config` correctly maps `args.hebbian.*` -> HebbianConfig
    (both SimpleNamespace and dict forms).
  - Replay of the runner's hook against the real comm-augmented LBF env:
    feeding `info['positions']` and `info['comm_events']` into
    `HebbianSocialGraph.update` advances W; runs with signal actions
    produce a different W than runs without.
  - `diffuse_rewards` changes rewards once W has bonds (reward-diffusion
    path produces non-identity output).
  - With `hebbian.enabled = False`, no W is built and `diffuse_rewards`
    is identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lbforaging")
pytest.importorskip("torch")  # HebbianRunner pulls in EPyMARL modules

import gymnasium as gym  # noqa: E402

from envs import lbf_comm_wrapper  # noqa: F401,E402 — register `Foraging-Comm-*` envs
from hebbian_module import HebbianConfig, HebbianSocialGraph  # noqa: E402
from runners.hebbian_runner import _build_hebbian_config, _flag  # noqa: E402


ENV_ID = "Foraging-Comm-8x8-3p-2f-v3"
N_AGENTS = 3


# ── config parsing ──

def test_build_hebbian_config_from_namespace():
    hebbian = SimpleNamespace(
        enabled=True,
        num_agents=3,
        interaction_radius=2.0,
        communication_coactivity_bonus=0.3,
        reward_diffusion_gamma=0.5,
    )
    args = SimpleNamespace(hebbian=hebbian, n_agents=3)
    cfg = _build_hebbian_config(args)
    assert cfg.enabled is True
    assert cfg.num_agents == 3
    assert cfg.interaction_radius == 2.0
    assert cfg.communication_coactivity_bonus == 0.3
    assert cfg.reward_diffusion_gamma == 0.5


def test_build_hebbian_config_from_dict():
    args = SimpleNamespace(
        hebbian={"enabled": True, "num_agents": 4, "interaction_radius": 1.5},
        n_agents=4,
    )
    cfg = _build_hebbian_config(args)
    assert cfg.enabled is True
    assert cfg.num_agents == 4
    assert cfg.interaction_radius == 1.5


def test_build_hebbian_config_disabled_when_no_block():
    args = SimpleNamespace(n_agents=3)
    cfg = _build_hebbian_config(args)
    assert cfg.enabled is False


def test_flag_helper_reads_namespace_and_dict():
    ns = SimpleNamespace(hebbian=SimpleNamespace(reward_diffusion=True))
    assert _flag(ns, "reward_diffusion", False) is True

    d = SimpleNamespace(hebbian={"reward_diffusion": True})
    assert _flag(d, "reward_diffusion", False) is True

    none = SimpleNamespace()
    assert _flag(none, "reward_diffusion", False) is False
    assert _flag(none, "reward_diffusion", True) is True


# ── runner hook replay against the real env ──

def _replay_hook_run(env, hebbian, n_steps, signal_every=None, rng_seed=0):
    """Replay HebbianRunner's update logic against `env` for n_steps.

    `signal_every`: if not None, every N-th step agent 0 picks action 6
    (signal teammate 1); other steps use random movement.
    Returns the final W matrix.
    """
    rng = np.random.default_rng(rng_seed)
    env.reset(seed=rng_seed)
    for step in range(n_steps):
        if signal_every is not None and step % signal_every == 0:
            actions = (6, 0, 0)
        else:
            actions = tuple(int(a) for a in rng.integers(0, 6, size=N_AGENTS))
        _, reward, term, trunc, info = env.step(actions)
        hebbian.update(
            positions=[tuple(p) for p in info["positions"]],
            step_rewards=[float(r) for r in reward],
            advantages=None,
            comm_events=info.get("comm_events"),
        )
        if term or trunc:
            env.reset(seed=rng_seed + step)
    return hebbian.W.copy()


def test_step_count_advances_and_w_changes_with_random_actions():
    env = gym.make(ENV_ID)
    cfg = HebbianConfig(enabled=True, num_agents=N_AGENTS, interaction_radius=2.0,
                       log_graph_every=10**9)
    hebbian = HebbianSocialGraph(cfg)
    W0 = hebbian.W.copy()
    _replay_hook_run(env, hebbian, n_steps=100)
    assert hebbian._step_count == 100
    # W must have moved off its initial values somewhere
    assert not np.allclose(hebbian.W, W0), "W did not change after 100 update steps"


def test_w_differs_between_signal_and_no_signal_runs():
    """The comm path through env -> info['comm_events'] -> hebbian.update
    must produce a different W than a run with no signal actions."""
    cfg = HebbianConfig(
        enabled=True, num_agents=N_AGENTS,
        interaction_radius=1.0,             # tight gate so comm is the differentiator
        communication_coactivity_bonus=0.5,
        log_graph_every=10**9,
    )
    # Two parallel runs with the same seed.
    env_a = gym.make(ENV_ID)
    env_b = gym.make(ENV_ID)
    hebbian_a = HebbianSocialGraph(cfg)
    hebbian_b = HebbianSocialGraph(cfg)

    W_no = _replay_hook_run(env_a, hebbian_a, n_steps=100, signal_every=None, rng_seed=7)
    W_yes = _replay_hook_run(env_b, hebbian_b, n_steps=100, signal_every=5, rng_seed=7)

    assert not np.allclose(W_no, W_yes), (
        "expected the signal-emitting run to produce a different W; "
        f"W_no=\n{W_no}\nW_yes=\n{W_yes}"
    )


def test_diffuse_rewards_changes_when_w_has_bonds():
    """When W has non-zero off-diagonals, diffuse_rewards must produce
    rewards different from the raw input (the reward-diffusion path is
    where integration (a) shows up)."""
    cfg = HebbianConfig(
        enabled=True, num_agents=N_AGENTS,
        interaction_radius=10.0,         # always co-active
        reward_diffusion_gamma=0.5,
        log_graph_every=10**9,
    )
    hebbian = HebbianSocialGraph(cfg)
    # Drive W up with cooperative-like steps.
    positions = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0)]
    for _ in range(60):
        hebbian.update(positions, [1.0, 1.0, 1.0], advantages=[0.5, 0.5, 0.5])

    raw = [1.0, 0.0, 0.0]
    diffused = hebbian.diffuse_rewards(raw)
    assert diffused != raw, (
        f"diffuse_rewards should produce different output when W has bonds; "
        f"raw={raw}, diffused={diffused}, W=\n{hebbian.W}"
    )


def test_disabled_graph_is_strict_passthrough():
    """When hebbian.enabled=False, the graph is a strict no-op."""
    cfg = HebbianConfig(enabled=False, num_agents=N_AGENTS)
    hebbian = HebbianSocialGraph(cfg)
    raw = [0.7, -0.3, 1.5]
    assert hebbian.diffuse_rewards(raw) == raw
    # update returns None and is harmless
    out = hebbian.update(
        positions=[(0.0, 0.0, 0.0)] * N_AGENTS,
        step_rewards=raw,
        advantages=None,
        comm_events=[(0, 1)],
    )
    assert out is None
