"""Tests for the MAPPO centralized critic (src/rl_layer/centralized_critic.py).

Every test uses the ``fake_sentence_transformers`` fixture so that
``CentralizedCritic.__init__`` (which lazily imports ``sentence_transformers``
and ``agent_modules.util``) resolves to the deterministic all-ones fake encoder
(DIM=8) instead of downloading a real model.

Joint-state layout under test (N agents, M milestones, fake semantic dim 8):

  per-agent compact block (3 + 5 + 1 + 8 + M + 1 floats):
    [0:3]   position xyz
    [3:8]   chamber one-hot over ("ch1".."ch5")
    [8]     hp / 20  (1.0 when hp is None/missing)
    [9:17]  inventory bag-of-words over 8 items (substring match)
    [17:17+M] milestone bitmap
    [17+M]  raw step reward clipped to [-100, 100]
  then N * 8 semantic floats (all 1.0 with the fake encoder).
"""

import math

import numpy as np
import pytest
import torch


def _make_critic(fake_st, num_agents=3, milestone_ids=("m1", "m2"), **cfg_overrides):
    """Build a small CPU CentralizedCritic (fixture must already be active)."""
    from rl_layer.centralized_critic import CentralizedCritic
    from rl_layer.config import RLConfig

    cfg_overrides.setdefault("critic_hidden", 32)
    cfg = RLConfig(**cfg_overrides)
    return CentralizedCritic(
        num_agents=num_agents,
        config=cfg,
        milestone_ids=list(milestone_ids),
        device=torch.device("cpu"),
    )


def _empty_encode_kwargs():
    return dict(
        positions={}, chambers={}, hps={}, inventories={},
        milestones_by_agent={}, raw_rewards={}, last_actions={}, last_comms={},
    )


def test_construction_joint_dim_n3_m2(fake_sentence_transformers):
    """joint_dim for N=3, M=2 is N*(3+5+1+8+M+1) + N*8 == 84, matching the net's input layer."""
    from rl_layer.centralized_critic import _CHAMBERS, _INVENTORY_ITEMS

    critic = _make_critic(fake_sentence_transformers)
    per_agent_compact = 3 + len(_CHAMBERS) + 1 + len(_INVENTORY_ITEMS) + 2 + 1  # = 20
    expected = 3 * per_agent_compact + 3 * 8
    assert per_agent_compact == 20
    assert expected == 84
    assert critic.joint_dim == 84
    # first Linear of the MLP consumes the full joint vector
    assert critic.net.net[0].in_features == 84


def test_encode_joint_compact_layout(fake_sentence_transformers):
    """encode_joint places pos/chamber/hp/inventory/milestones/clipped-reward at the documented offsets."""
    critic = _make_critic(fake_sentence_transformers)
    js = critic.encode_joint(
        positions={0: (1.0, 2.0, 3.0)},
        chambers={0: "ch2"},
        hps={0: 10.0, 1: None},
        inventories={0: "mcl_tools:diamond_sword"},
        milestones_by_agent={"agent_0": {"m1"}},
        raw_rewards={0: 500.0, 1: -500.0},
        last_actions={0: "Dig"},
        last_comms={},
    )
    assert js.shape == (84,)
    assert js.dtype == np.float32

    expected = np.zeros(84, dtype=np.float32)
    # agent 0 block (base 0, per-agent dim 20)
    expected[0:3] = [1.0, 2.0, 3.0]   # position
    expected[3 + 1] = 1.0             # 'ch2' -> one-hot index 1 of 5
    expected[8] = 0.5                 # hp 10 / 20
    expected[9] = 1.0                 # 'diamond_sword' is _INVENTORY_ITEMS[0]
    expected[17] = 1.0                # milestone bitmap ['m1'] -> [1, 0]
    expected[19] = 100.0              # reward 500 clipped to +100
    # agent 1 block (base 20): hp=None -> full health 1.0; reward -500 clipped
    expected[20 + 8] = 1.0
    expected[20 + 19] = -100.0
    # agent 2 block (base 40): entirely missing -> only the hp default
    expected[40 + 8] = 1.0
    # semantic stream: fake encoder returns ones
    expected[60:84] = 1.0

    np.testing.assert_allclose(js, expected, rtol=0, atol=0)


def test_encode_joint_missing_agents_defaults(fake_sentence_transformers):
    """Missing agent entries default to zeros except hp=1.0 (full health); semantic stays ones."""
    critic = _make_critic(fake_sentence_transformers)
    js = critic.encode_joint(**_empty_encode_kwargs())

    expected = np.zeros(84, dtype=np.float32)
    for i in range(3):
        expected[i * 20 + 8] = 1.0    # hp None -> 1.0
    expected[60:84] = 1.0             # fake semantic embeddings
    np.testing.assert_allclose(js, expected, rtol=0, atol=0)


def test_encode_joint_semantic_block_is_ones(fake_sentence_transformers):
    """The last N*8 entries of the joint state come from the sentence encoder (all 1.0 with the fake)."""
    critic = _make_critic(fake_sentence_transformers)
    js = critic.encode_joint(**_empty_encode_kwargs())
    sem = js[-3 * 8:]
    assert sem.shape == (24,)
    np.testing.assert_array_equal(sem, np.ones(24, dtype=np.float32))


def test_evaluate_returns_float_and_is_deterministic(fake_sentence_transformers):
    """evaluate() returns a Python float and is reproducible under a fixed torch seed."""
    torch.manual_seed(0)
    c1 = _make_critic(fake_sentence_transformers)
    state = np.linspace(-1.0, 1.0, c1.joint_dim).astype(np.float32)
    v1 = c1.evaluate(state)
    assert type(v1) is float
    # repeated call on the same net: identical
    assert c1.evaluate(state) == v1
    # fresh net built under the same seed: identical init -> identical value
    torch.manual_seed(0)
    c2 = _make_critic(fake_sentence_transformers)
    assert c2.evaluate(state) == pytest.approx(v1, abs=0.0)


def test_store_step_should_update_threshold(fake_sentence_transformers):
    """should_update() is False until the internal buffer reaches update_interval steps."""
    critic = _make_critic(fake_sentence_transformers, update_interval=8)
    state = np.zeros(critic.joint_dim, dtype=np.float32)
    for k in range(7):
        critic.store_step(state, team_reward=0.0, value=0.0, done=False)
        assert len(critic) == k + 1
        assert critic.should_update() is False
    critic.store_step(state, team_reward=0.0, value=0.0, done=False)
    assert len(critic) == 8
    assert critic.should_update() is True


def test_critic_buffer_compute_returns_hand_case(fake_sentence_transformers):
    """_CriticBuffer.compute_returns matches hand-computed GAE for gamma=lambda=0.5."""
    from rl_layer.centralized_critic import _CriticBuffer

    s = np.zeros(1, dtype=np.float32)

    # No terminal: rewards [1,1], values [1,2], last_value=3.
    #  t=1: delta = 1 + 0.5*3 - 2 = 0.5 ; gae = 0.5 ; ret = 2.5
    #  t=0: delta = 1 + 0.5*2 - 1 = 1.0 ; gae = 1.0 + 0.25*0.5 = 1.125 ; ret = 2.125
    buf = _CriticBuffer()
    buf.store(s, 1.0, 1.0, False)
    buf.store(s, 1.0, 2.0, False)
    assert buf.compute_returns(0.5, 0.5, last_value=3.0) == pytest.approx([2.125, 2.5])

    # Terminal at t=1 zeroes both bootstrap and the GAE carry:
    #  t=1: delta = 1 - 2 = -1 ; gae = -1 ; ret = 1.0
    #  t=0: delta = 1 + 0.5*2 - 1 = 1.0 ; gae = 1.0 + 0.25*(-1) = 0.75 ; ret = 1.75
    buf2 = _CriticBuffer()
    buf2.store(s, 1.0, 1.0, False)
    buf2.store(s, 1.0, 2.0, True)
    assert buf2.compute_returns(0.5, 0.5, last_value=3.0) == pytest.approx([1.75, 1.0])

    buf2.clear()
    assert len(buf2) == 0


def test_update_on_empty_buffer_is_noop(fake_sentence_transformers):
    """update() with an empty buffer returns {} and does not bump the update counter."""
    critic = _make_critic(fake_sentence_transformers)
    assert critic.update() == {}
    assert critic.update_count == 0


def test_update_info_keys_buffer_cleared_counter(fake_sentence_transformers):
    """update() on 16 stored steps returns finite diagnostics, clears the buffer, increments the counter."""
    critic = _make_critic(
        fake_sentence_transformers,
        update_interval=8, mini_batch_size=4, ppo_epochs=2,
    )
    rng = np.random.default_rng(0)
    for t in range(16):
        state = rng.normal(size=critic.joint_dim).astype(np.float32)
        critic.store_step(
            state,
            team_reward=float(rng.normal()),
            value=critic.evaluate(state),
            done=(t == 15),
        )
    info = critic.update(last_value=0.0)

    expected_keys = {
        "critic_loss", "critic_update_count", "critic_buffer_size",
        "critic_returns_mean", "critic_returns_std",
        "critic_returns_min", "critic_returns_max",
        "critic_values_mean", "critic_values_std",
        "critic_explained_variance",
    }
    assert set(info) == expected_keys
    for key in expected_keys:
        assert math.isfinite(float(info[key])), f"{key} not finite: {info[key]}"
    assert info["critic_buffer_size"] == 16
    assert info["critic_update_count"] == 1
    assert info["critic_returns_min"] <= info["critic_returns_max"]
    assert critic.update_count == 1
    assert len(critic) == 0
    assert critic.should_update() is False


@pytest.mark.slow
def test_update_learning_sanity_loss_decreases(fake_sentence_transformers):
    """5 update rounds on a constant state with constant return 1.0 reduce critic_loss."""
    critic = _make_critic(
        fake_sentence_transformers,
        critic_lr=1e-2, mini_batch_size=4, ppo_epochs=2, update_interval=8,
    )
    state = np.full(critic.joint_dim, 0.1, dtype=np.float32)
    losses = []
    for _ in range(5):
        for _ in range(8):
            # done=True at every step makes the GAE return exactly the reward,
            # so the regression target is the constant 1.0.
            critic.store_step(state, team_reward=1.0,
                              value=critic.evaluate(state), done=True)
        losses.append(critic.update(last_value=0.0)["critic_loss"])
    assert critic.update_count == 5
    assert losses[-1] < losses[0]


def test_save_load_roundtrip(fake_sentence_transformers, tmp_path):
    """save()/load() restore net weights and the update counter after a perturbation."""
    critic = _make_critic(
        fake_sentence_transformers,
        update_interval=4, mini_batch_size=4, ppo_epochs=1,
    )
    rng = np.random.default_rng(1)
    for _ in range(4):
        state = rng.normal(size=critic.joint_dim).astype(np.float32)
        critic.store_step(state, float(rng.normal()),
                          critic.evaluate(state), False)
    critic.update()
    assert critic.update_count == 1

    ckpt = tmp_path / "critic_ckpt"
    critic.save(ckpt)
    saved = {k: v.clone() for k, v in critic.net.state_dict().items()}

    # perturb everything that load() must restore
    with torch.no_grad():
        for p in critic.net.parameters():
            p.add_(1.0)
    critic._update_count = 99
    assert not all(
        torch.equal(saved[k], v) for k, v in critic.net.state_dict().items()
    )

    critic.load(ckpt)
    for k, v in critic.net.state_dict().items():
        assert torch.equal(saved[k], v), f"weight {k} not restored"
    assert critic.update_count == 1
