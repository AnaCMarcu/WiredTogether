"""Tests for rl_layer.trajectory_buffer (RolloutBuffer + GAE).

All expected values are hand-computed from the recursion in
``RolloutBuffer.compute_gae``:

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    gae_t   = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}

``tr.returns = gae_t + V(s_t)`` is set BEFORE advantage standardisation, so
``returns - baseline`` recovers the raw (un-standardised) advantage exactly.
All hand cases use dyadic rationals (gamma = lambda = 0.5) so float arithmetic
is exact unless noted.
"""

import logging
import math

import numpy as np
import pytest
import torch

from rl_layer.trajectory_buffer import RolloutBuffer, Transition


def _fill(buf, rewards, values=None, dones=None, values_global=None):
    """Store one (action, reward) pair per element of ``rewards``."""
    n = len(rewards)
    if values is None:
        values = [0.0] * n
    if dones is None:
        dones = [False] * n
    for i in range(n):
        vg = None if values_global is None else values_global[i]
        buf.store_action(f"prompt {i}", action_idx=i, log_prob=0.0,
                         value=values[i], value_global=vg)
        buf.store_reward(rewards[i], done=dones[i])
    return buf


# ── GAE hand cases ───────────────────────────────────────────────────────────

def test_gae_hand_case_zero_values():
    """gamma=lambda=0.5, V=0, r=[1,1,1], last_value=0 -> raw adv [1.3125, 1.25, 1.0]."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 1.0, 1.0])
    buf.compute_gae(gamma=0.5, gae_lambda=0.5, last_value=0.0)
    # t=2: delta=1, gae=1 | t=1: gae=1+0.25*1=1.25 | t=0: gae=1+0.25*1.25=1.3125
    raw = [tr.returns - tr.old_value for tr in buf.get_all()]
    assert raw == [1.3125, 1.25, 1.0]


def test_gae_nonzero_values_and_bootstrap():
    """V=[1,2,3], r=[1,1,1], last_value=2 bootstraps the tail; returns = raw_adv + V."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 1.0, 1.0], values=[1.0, 2.0, 3.0])
    buf.compute_gae(gamma=0.5, gae_lambda=0.5, last_value=2.0)
    # t=2: delta = 1 + 0.5*2 - 3 = -1            -> gae = -1,     ret = 2
    # t=1: delta = 1 + 0.5*3 - 2 = 0.5           -> gae = 0.5 + 0.25*(-1) = 0.25, ret = 2.25
    # t=0: delta = 1 + 0.5*2 - 1 = 1             -> gae = 1 + 0.25*0.25 = 1.0625, ret = 2.0625
    transitions = buf.get_all()
    assert [tr.returns for tr in transitions] == [2.0625, 2.25, 2.0]
    raw = [tr.returns - tr.old_value for tr in transitions]
    assert raw == [1.0625, 0.25, -1.0]


def test_gae_done_truncation_zeroes_bootstrap_and_accumulation():
    """done=True on the middle transition cuts both V(s_{t+1}) and the gae carry."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 1.0, 1.0], values=[0.0, 0.0, 5.0],
                dones=[False, True, False])
    buf.compute_gae(gamma=0.5, gae_lambda=0.5, last_value=4.0)
    # t=2: delta = 1 + 0.5*4 - 5 = -2                    -> gae = -2, ret = 3
    # t=1: done -> delta = 1 + 0 - 0 = 1, carry zeroed   -> gae = 1,  ret = 1
    #      (without done it would be 1 + 2.5 + 0.25*(-2) = 3)
    # t=0: delta = 1 + 0.5*0 - 0 = 1                     -> gae = 1 + 0.25*1 = 1.25
    assert [tr.returns for tr in buf.get_all()] == [1.25, 1.0, 3.0]


def test_gae_single_transition_unstandardised_and_finite():
    """A 1-transition buffer keeps its raw, finite advantage (numel<2 skips standardisation)."""
    buf = RolloutBuffer()
    buf.store_action("only", action_idx=0, log_prob=0.0, value=0.5)
    buf.store_reward(2.0, done=False)
    buf.compute_gae(gamma=0.9, gae_lambda=0.8, last_value=1.0)
    tr = buf.get_all()[0]
    # delta = gae = 2.0 + 0.9*1.0 - 0.5 = 2.4 ; returns = 2.4 + 0.5 = 2.9
    assert math.isfinite(tr.advantage)
    assert tr.advantage == pytest.approx(2.4)
    assert tr.returns == pytest.approx(2.9)
    assert tr.advantage == pytest.approx(tr.returns - tr.old_value)


def test_gae_standardisation_mean_zero_std_one():
    """>=2 transitions: stored advantages are standardised (mean~0, unbiased std~1)."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 1.0, 1.0])
    buf.compute_gae(gamma=0.5, gae_lambda=0.5, last_value=0.0)
    adv = torch.tensor([tr.advantage for tr in buf.get_all()], dtype=torch.float32)
    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert adv.std().item() == pytest.approx(1.0, abs=2e-3)  # eps=1e-5 in denominator


def test_gae_use_global_value_with_fallback():
    """use_global_value=True uses old_value_global as baseline, old_value when None."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 1.0, 1.0], values=[10.0, 20.0, 30.0],
                values_global=[1.0, None, 3.0])
    buf.compute_gae(gamma=0.5, gae_lambda=0.5, last_value=0.0, use_global_value=True)
    # baselines: [1 (global), 20 (fallback), 3 (global)]
    # t=2: delta = 1 + 0 - 3 = -2                        -> gae = -2,   ret = 1
    # t=1: delta = 1 + 0.5*3 - 20 = -17.5                -> gae = -18,  ret = 2
    # t=0: delta = 1 + 0.5*20 - 1 = 10                   -> gae = 10 + 0.25*(-18) = 5.5, ret = 6.5
    assert [tr.returns for tr in buf.get_all()] == [6.5, 2.0, 1.0]


# ── Pending-transition bookkeeping ───────────────────────────────────────────

def test_set_pending_value_global_attaches_to_pending():
    """set_pending_value_global writes old_value_global (+joint_state) onto the pending transition."""
    buf = RolloutBuffer()
    buf.store_action("p", action_idx=0, log_prob=0.0, value=0.0)
    js = np.arange(4, dtype=np.float32)
    buf.set_pending_value_global(7.5, joint_state=js)
    buf.store_reward(1.0)
    tr = buf.get_all()[0]
    assert tr.old_value_global == 7.5
    np.testing.assert_array_equal(tr.joint_state, js)


def test_set_pending_value_global_noop_without_pending():
    """set_pending_value_global with nothing pending is a silent no-op."""
    buf = RolloutBuffer()
    buf.set_pending_value_global(3.0, joint_state=np.zeros(2))
    assert len(buf) == 0
    assert buf._pending is None


def test_store_action_twice_flushes_pending_with_zero_reward():
    """A second store_action before store_reward flushes the first with reward 0."""
    buf = RolloutBuffer()
    buf.store_action("first", action_idx=1, log_prob=0.0, value=0.0)
    buf.store_action("second", action_idx=2, log_prob=0.0, value=0.0)
    assert len(buf) == 1
    flushed = buf.get_all()[0]
    assert flushed.action_idx == 1
    assert flushed.reward == 0.0
    assert flushed.done is False
    assert buf._pending is not None and buf._pending.action_idx == 2


def test_store_reward_without_pending_warns_and_discards(caplog):
    """store_reward with no pending transition logs a warning and leaves the buffer unchanged."""
    buf = RolloutBuffer()
    with caplog.at_level(logging.WARNING, logger="rl_layer.trajectory_buffer"):
        buf.store_reward(5.0)
    assert len(buf) == 0
    assert any("no pending transition" in r.message for r in caplog.records)


# ── Reward sanitisation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "abc", None])
def test_reward_sanitisation_to_zero(bad):
    """nan/inf/non-numeric rewards are clamped to 0.0."""
    buf = RolloutBuffer()
    buf.store_action("p", action_idx=0, log_prob=0.0, value=0.0)
    buf.store_reward(bad)
    assert buf.get_all()[0].reward == 0.0


@pytest.mark.parametrize("raw,clamped", [(1e9, 1e6), (-1e9, -1e6)])
def test_reward_clamped_to_plus_minus_1e6(raw, clamped):
    """Finite rewards are clamped into [-1e6, 1e6]."""
    buf = RolloutBuffer()
    buf.store_action("p", action_idx=0, log_prob=0.0, value=0.0)
    buf.store_reward(raw)
    assert buf.get_all()[0].reward == clamped


# ── Batching ─────────────────────────────────────────────────────────────────

def test_sample_batches_partitions_buffer():
    """10 transitions at mini_batch_size=4 yield batches of 4/4/2 covering each exactly once."""
    buf = _fill(RolloutBuffer(), rewards=[float(i) for i in range(10)])
    batches = list(buf.sample_batches(4))
    assert [len(b) for b in batches] == [4, 4, 2]
    seen = [id(tr) for batch in batches for tr in batch]
    assert sorted(seen) == sorted(id(tr) for tr in buf.get_all())


def test_sample_batches_includes_extra_transitions():
    """extra_transitions join the shuffled pool; every pool member appears exactly once."""
    buf = _fill(RolloutBuffer(), rewards=[0.0] * 10)
    extras = [Transition(prompt_text=f"extra {i}", action_idx=100 + i,
                         old_log_prob=0.0, old_value=0.0) for i in range(3)]
    batches = list(buf.sample_batches(4, extra_transitions=extras))
    assert [len(b) for b in batches] == [4, 4, 4, 1]
    seen = [id(tr) for batch in batches for tr in batch]
    expected = [id(tr) for tr in buf.get_all()] + [id(tr) for tr in extras]
    assert sorted(seen) == sorted(expected)


@pytest.mark.parametrize("bad_size", [0, -1])
def test_sample_batches_rejects_nonpositive_size(bad_size):
    """mini_batch_size <= 0 raises AssertionError (on first consumption of the generator)."""
    buf = _fill(RolloutBuffer(), rewards=[1.0])
    with pytest.raises(AssertionError):
        list(buf.sample_batches(bad_size))


# ── Misc API ─────────────────────────────────────────────────────────────────

def test_filter_by_keyword_case_insensitive():
    """filter_by_keyword lower-cases both keyword and prompt text."""
    buf = RolloutBuffer()
    for prompt in ["Go to the Anvil", "craft sword", "ANVIL nearby"]:
        buf.store_action(prompt, action_idx=0, log_prob=0.0, value=0.0)
        buf.store_reward(0.0)
    hits = buf.filter_by_keyword("aNvIl")
    assert [t.prompt_text for t in hits] == ["Go to the Anvil", "ANVIL nearby"]
    assert buf.filter_by_keyword("pickaxe") == []


def test_clear_resets_buffer_and_pending():
    """clear() empties the buffer and drops any pending transition."""
    buf = _fill(RolloutBuffer(), rewards=[1.0, 2.0])
    buf.store_action("pending", action_idx=9, log_prob=0.0, value=0.0)
    buf.clear()
    assert len(buf) == 0
    assert buf._pending is None
    assert buf.ready is False


def test_ready_property():
    """ready is False while empty (even with a pending transition), True once one is stored."""
    buf = RolloutBuffer()
    assert buf.ready is False
    buf.store_action("p", action_idx=0, log_prob=0.0, value=0.0)
    assert buf.ready is False  # pending only — nothing committed yet
    buf.store_reward(1.0)
    assert buf.ready is True
