"""Tests for Stage-3 entry-point dispatch and the multi-agent YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.multi_agent_craftium_grpo import (
    MultiAgentEnvAdapter,
    PolicyConfig,
    _build_full_config,
    load_config,
)


# ──── PolicyConfig dispatch ─────────────────────────────────────────────


def test_policy_config_single_agent_fallback():
    cfg = PolicyConfig(trained_agent=2)
    assert cfg.effective_trained_agents() == [2]


def test_policy_config_explicit_list_wins():
    cfg = PolicyConfig(trained_agent=0, trained_agents=[0, 1, 2])
    assert cfg.effective_trained_agents() == [0, 1, 2]


def test_policy_config_single_element_list():
    """A 1-element ``trained_agents`` list resolves to single-agent mode."""
    cfg = PolicyConfig(trained_agents=[1])
    assert cfg.effective_trained_agents() == [1]


# ──── YAML config on disk ───────────────────────────────────────────────


def test_multi_agent_yaml_loads():
    repo_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = repo_root / "configs/rlvr/grpo_multi_agent.yaml"
    cfg = load_config(yaml_path)
    assert cfg.policy.effective_trained_agents() == [0, 1, 2]
    assert cfg.grpo.team_reward is False
    assert cfg.env.num_agents == 3


# ──── MultiAgentEnvAdapter ──────────────────────────────────────────────


class _PettingZooStub:
    """Minimal PettingZoo-style env (string-keyed dicts) for adapter tests."""

    def __init__(self):
        self.last_actions = None

    def reset(self):
        obs = {"agent_0": "obs0", "agent_1": "obs1", "agent_2": "obs2"}
        info = {f"agent_{i}": {"chamber": "ch3", "position": (float(i), 0.0, 0.0)}
                for i in range(3)}
        return obs, info

    def step(self, actions):
        self.last_actions = actions
        obs = {f"agent_{i}": f"obs{i}_after" for i in range(3)}
        reward = {f"agent_{i}": float(i) for i in range(3)}
        done = {f"agent_{i}": False for i in range(3)}
        info = {f"agent_{i}": {"chamber": "ch3"} for i in range(3)}
        return obs, reward, done, info


def test_adapter_reset_returns_int_keys():
    adapter = MultiAgentEnvAdapter(_PettingZooStub())
    obs, info = adapter.reset()
    assert set(obs.keys()) == {0, 1, 2}
    assert set(info.keys()) == {0, 1, 2}
    assert obs[0] == "obs0"


def test_adapter_step_translates_actions_to_strings():
    stub = _PettingZooStub()
    adapter = MultiAgentEnvAdapter(stub)
    adapter.reset()
    adapter.step({
        0: {"action": "dig"},
        1: {"action": "forward"},
        2: {"action": "nop"},
    })
    assert set(stub.last_actions.keys()) == {"agent_0", "agent_1", "agent_2"}
    # _DISCRETE_ACTIONS: forward=0, ..., dig=6 (0-indexed). Env action = idx + 1.
    assert stub.last_actions["agent_0"] == 7    # dig
    assert stub.last_actions["agent_1"] == 1    # forward
    assert stub.last_actions["agent_2"] == 0    # nop


def test_adapter_handles_5_tuple_step_return():
    """PettingZoo >= 1.24 returns (obs, reward, term, trunc, info)."""

    class _Modern:
        def reset(self):
            return ({"agent_0": "o", "agent_1": "o"},
                    {"agent_0": {"chamber": "ch3", "position": (0., 0., 0.)},
                     "agent_1": {"chamber": "ch3", "position": (1., 0., 0.)}})

        def step(self, actions):
            obs = {"agent_0": "o", "agent_1": "o"}
            reward = {"agent_0": 0.0, "agent_1": 0.0}
            term = {"agent_0": False, "agent_1": False}
            trunc = {"agent_0": False, "agent_1": True}    # agent 1 truncated
            info = {"agent_0": {}, "agent_1": {}}
            return obs, reward, term, trunc, info

    adapter = MultiAgentEnvAdapter(_Modern())
    adapter.reset()
    obs, reward, done, info = adapter.step({0: {"action": "nop"}, 1: {"action": "nop"}})
    # Truncated agents are flagged done.
    assert done[1] is True
    assert done[0] is False
