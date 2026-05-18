"""Tests for ``HebbianGRPOBridge`` (Stage 4a)."""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.hebbian_grpo_bridge import (
    HebbianGRPOBridge,
    comm_events_from_actions,
)


class _FakeGraphConfig:
    def __init__(self, enabled: bool = True, num_agents: int = 3):
        self.enabled = enabled
        self.num_agents = num_agents


class _FakeGraph:
    """Records every ``update`` call without doing any math."""

    def __init__(self, enabled: bool = True, num_agents: int = 3):
        self.config = _FakeGraphConfig(enabled, num_agents)
        self.update_calls: list[dict] = []
        self._weights = np.zeros((num_agents, num_agents), dtype=np.float32)

    def update(self, positions, step_rewards, advantages=None, comm_events=None):
        self.update_calls.append({
            "positions": positions,
            "step_rewards": list(step_rewards),
            "advantages": list(advantages) if advantages is not None else None,
            "comm_events": list(comm_events) if comm_events is not None else None,
        })
        return self._weights

    def get_normalized_weights(self, agent_id: int) -> np.ndarray:
        # Manually-set teammate distribution for tests.
        N = self.config.num_agents
        out = np.zeros(N, dtype=np.float32)
        if N > 1:
            for j in range(N):
                if j != agent_id:
                    out[j] = 1.0 / (N - 1)
        return out

    def get_all_weights(self) -> np.ndarray:
        return self._weights.copy()


# ──── observe_step ─────────────────────────────────────────────────────


def test_observe_step_forwards_to_graph():
    g = _FakeGraph()
    b = HebbianGRPOBridge(g)
    b.observe_step(
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        step_rewards=[0.0, 0.5, 1.0],
    )
    assert len(g.update_calls) == 1
    assert g.update_calls[0]["step_rewards"] == [0.0, 0.5, 1.0]


def test_observe_step_no_op_when_disabled():
    g = _FakeGraph(enabled=False)
    b = HebbianGRPOBridge(g)
    b.observe_step(
        positions=[(0.0, 0.0, 0.0)] * 3,
        step_rewards=[0.0, 0.0, 0.0],
    )
    assert g.update_calls == []


def test_observe_step_swallows_exceptions(caplog):
    class _BrokenGraph(_FakeGraph):
        def update(self, *a, **kw):
            raise RuntimeError("boom")

    b = HebbianGRPOBridge(_BrokenGraph())
    # Must not raise — a buggy graph should not crash the rollout.
    b.observe_step(
        positions=[(0.0, 0.0, 0.0)] * 3,
        step_rewards=[0.0, 0.0, 0.0],
    )


def test_step_count_advances():
    b = HebbianGRPOBridge(_FakeGraph())
    for _ in range(5):
        b.observe_step(positions=[(0.,)*3]*3, step_rewards=[0.]*3)
    assert b.step_count() == 5


# ──── normalized_weights ───────────────────────────────────────────────


def test_normalized_weights_pass_through():
    b = HebbianGRPOBridge(_FakeGraph(num_agents=3))
    w = b.normalized_weights(0)
    assert w.shape == (3,)
    assert w[0] == 0.0  # no self-borrow
    assert abs(w.sum() - 1.0) < 1e-6


def test_normalized_weights_zero_when_disabled():
    b = HebbianGRPOBridge(_FakeGraph(enabled=False, num_agents=3))
    w = b.normalized_weights(0)
    assert (w == 0).all()


# ──── comm_events_from_actions ────────────────────────────────────────


def test_comm_events_basic():
    actions = {
        0: {"action": "forward", "communication_target": 1},
        1: {"action": "dig", "communication_target": None},
        2: {"action": "nop", "communication_target": 0},
    }
    events = comm_events_from_actions(actions)
    assert set(events) == {(0, 1), (2, 0)}


def test_comm_events_drops_self_targets():
    actions = {
        0: {"action": "forward", "communication_target": 0},   # self → drop
        1: {"action": "forward", "communication_target": 2},
    }
    events = comm_events_from_actions(actions)
    assert events == [(1, 2)]


def test_comm_events_handles_bool_target():
    actions = {
        0: {"action": "forward", "communication_target": True},   # bool → drop
        1: {"action": "forward", "communication_target": 0},
    }
    events = comm_events_from_actions(actions)
    assert events == [(1, 0)]


def test_comm_events_handles_missing_field():
    actions = {0: {"action": "forward"}}   # no comm_target → no event
    assert comm_events_from_actions(actions) == []


# ──── sampler integration (Stage 4a wiring) ───────────────────────────


class _TinyMultiAgentEnv:
    """Minimal multi-agent env for the sampler-integration test below."""

    def __init__(self):
        self._t = 0

    def reset(self):
        self._t = 0
        obs = {0: {}, 1: {}, 2: {}}
        info = {0: {"chamber": "ch3", "position": (0.0, 0.0, 0.0)},
                1: {"chamber": "ch3", "position": (10.0, 0.0, 10.0)},
                2: {"chamber": "ch3", "position": (20.0, 0.0, 20.0)}}
        return obs, info

    def step(self, actions):
        self._t += 1
        obs = {a: {} for a in (0, 1, 2)}
        rewards = {a: 0.1 for a in (0, 1, 2)}
        done = {a: False for a in (0, 1, 2)}
        info = {0: {"chamber": "ch3", "position": (0.0, 0.0, 0.0)},
                1: {"chamber": "ch3", "position": (10.0, 0.0, 10.0)},
                2: {"chamber": "ch3", "position": (20.0, 0.0, 20.0)}}
        return obs, rewards, done, info


class _NopMultiPolicy:
    def act(self, observation, info):
        from rlvr.rollout_sampler import RolloutTensors
        return ({"action": "nop", "communication_target": None, "thoughts": ""},
                RolloutTensors(prompt_text="P"))


def test_sampler_calls_bridge_per_step():
    """End-to-end: MultiAgentRolloutSampler with a HebbianGRPOBridge fires
    one ``observe_step`` per env step."""
    from rlvr.rollout_sampler import MultiAgentRolloutSampler, MultiAgentSamplerConfig

    g = _FakeGraph(num_agents=3)
    bridge = HebbianGRPOBridge(g)

    sampler = MultiAgentRolloutSampler(
        env=_TinyMultiAgentEnv(),
        policy=_NopMultiPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=4, num_agents=3, trained_agents=(0, 1, 2),
        ),
        hebbian_bridge=bridge,
    )
    sampler.sample_joint_group()
    # One observe_step per env step. The horizon was 4 and no early
    # termination → 4 calls.
    assert len(g.update_calls) == 4
    # Step rewards vector has length N = 3.
    assert all(len(call["step_rewards"]) == 3 for call in g.update_calls)


def test_sampler_skips_bridge_when_disabled():
    """A disabled graph means observe_step is a no-op; the sampler should
    still run without crashing."""
    from rlvr.rollout_sampler import MultiAgentRolloutSampler, MultiAgentSamplerConfig

    g = _FakeGraph(enabled=False, num_agents=3)
    bridge = HebbianGRPOBridge(g)

    sampler = MultiAgentRolloutSampler(
        env=_TinyMultiAgentEnv(),
        policy=_NopMultiPolicy(),
        config=MultiAgentSamplerConfig(
            n_per_group=1, horizon=3, num_agents=3, trained_agents=(0, 1, 2),
        ),
        hebbian_bridge=bridge,
    )
    sampler.sample_joint_group()
    assert g.update_calls == []
