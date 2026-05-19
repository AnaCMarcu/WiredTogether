"""Verify the comm-augmented LBF wrapper.

Covers the Phase 2 verification gate from HEBBIAN_MARL_PLAN.md:
  - Actions 0–5 forwarded unchanged; comm_events empty.
  - Agent 0 picks action 6 (N=3): comm_events == [(0, 1)]; signal flag
    appears in agent 1's next-step observation.
  - Agent 0 picks action 7: comm_events == [(0, 2)]; flag appears in
    agent 2's next-step observation.
  - Reset clears comm flags.
  - With no signal actions, the underlying env's reward dynamics are
    identical to base LBF (regression guard).
  - info['positions'] matches env.unwrapped.players[i].position.
"""

import numpy as np
import pytest

pytest.importorskip("lbforaging")

import gymnasium as gym  # noqa: E402

# Import side-effects: this registers `Foraging-Comm-*-v3` env ids
from envs import lbf_comm_wrapper  # noqa: F401,E402


ENV_ID = "Foraging-Comm-8x8-3p-2f-v3"
BASE_ENV_ID = "Foraging-8x8-3p-2f-v3"   # plain lbforaging
N_AGENTS = 3


def _make():
    return gym.make(f"lbforaging:{BASE_ENV_ID}").unwrapped, gym.make(ENV_ID)


def test_action_space_expanded():
    env = gym.make(ENV_ID)
    assert len(env.action_space) == N_AGENTS
    for space in env.action_space:
        assert space.n == 6 + (N_AGENTS - 1)


def test_obs_space_extended_by_n_minus_one():
    env = gym.make(ENV_ID)
    base = gym.make(f"lbforaging:{BASE_ENV_ID}")
    for new_s, base_s in zip(env.observation_space, base.observation_space):
        assert new_s.shape[0] == base_s.shape[0] + (N_AGENTS - 1)


def test_movement_actions_unchanged_and_no_comm_events():
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert info["comm_events"] == []
    obs, reward, term, trunc, info = env.step((0, 0, 0))
    assert info["comm_events"] == []
    obs, reward, term, trunc, info = env.step((1, 2, 3))
    assert info["comm_events"] == []


def test_signal_action_emits_comm_event_for_first_teammate():
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    # Agent 0 picks action 6 (signal first non-self teammate → agent 1)
    obs, reward, term, trunc, info = env.step((6, 0, 0))
    assert info["comm_events"] == [(0, 1)]
    # Agent 1's next-step obs should have the signalled-by-0 flag set.
    # In agent 1's flag layout, teammates are [0, 2], so the flag slot
    # for sender=0 is index 0.
    flags_for_agent_1 = obs[1][-(N_AGENTS - 1):]
    assert flags_for_agent_1[0] == 1.0
    assert flags_for_agent_1[1] == 0.0


def test_signal_action_to_second_teammate():
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    # Agent 0 picks action 7 → signals second non-self teammate (agent 2)
    obs, reward, term, trunc, info = env.step((7, 0, 0))
    assert info["comm_events"] == [(0, 2)]
    # Agent 2's flag layout: teammates [0, 1]. sender=0 → slot 0.
    flags_for_agent_2 = obs[2][-(N_AGENTS - 1):]
    assert flags_for_agent_2[0] == 1.0
    assert flags_for_agent_2[1] == 0.0


def test_signal_flags_persist_only_one_step():
    """A signal sets the receiver's flag on the NEXT obs only."""
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    obs, *_ = env.step((6, 0, 0))            # agent 0 signals 1 → flag set
    flags = obs[1][-(N_AGENTS - 1):]
    assert flags[0] == 1.0
    obs, *_ = env.step((0, 0, 0))            # no signals; flag must clear
    flags = obs[1][-(N_AGENTS - 1):]
    assert np.all(flags == 0.0)


def test_reset_clears_signal_flags():
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    env.step((6, 0, 0))                       # cause a flag to be set next
    obs, info = env.reset(seed=0)
    for o in obs:
        flags = o[-(N_AGENTS - 1):]
        assert np.all(flags == 0.0)
    assert info["comm_events"] == []


def test_info_positions_match_player_positions():
    env = gym.make(ENV_ID)
    _, info = env.reset(seed=0)
    players = env.unwrapped.players
    assert len(info["positions"]) == len(players)
    for pos_info, player in zip(info["positions"], players):
        assert pos_info[0] == float(player.position[0])
        assert pos_info[1] == float(player.position[1])
        assert pos_info[2] == 0.0


def test_signal_action_issues_noop_for_movement():
    """A signal-acting agent does not move; underlying env saw NoOp."""
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    pre_pos = [p.position for p in env.unwrapped.players]
    env.step((6, 0, 0))                       # agent 0 signals; movement = NoOp
    post_pos = [p.position for p in env.unwrapped.players]
    assert pre_pos[0] == post_pos[0], (
        f"agent 0 picked a signal action but moved: {pre_pos[0]} -> {post_pos[0]}"
    )


def test_movement_only_run_matches_base_lbf_rewards():
    """Regression guard: with no signal actions, our wrapper preserves
    base-LBF reward dynamics step-for-step."""
    wrapped = gym.make(ENV_ID)
    base = gym.make(f"lbforaging:{BASE_ENV_ID}")

    obs_w, _ = wrapped.reset(seed=123)
    obs_b, _ = base.reset(seed=123)

    rng = np.random.default_rng(0)
    for _ in range(50):
        # Only movement actions (0-5)
        actions = tuple(int(a) for a in rng.integers(0, 6, size=N_AGENTS))
        _, r_w, t_w, tr_w, _ = wrapped.step(actions)
        _, r_b, t_b, tr_b, _ = base.step(actions)
        assert tuple(r_w) == tuple(r_b), (
            f"reward divergence: wrapper {r_w} vs base {r_b}"
        )
        if t_w or tr_w:
            break
