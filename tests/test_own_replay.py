"""Matched-samples control for experience sharing (--rl-own-replay-rho).

Reviewer bQx9 asks whether experience sharing helps because of the partners'
data or only because each PPO update sees more samples. The control adds the
same number of extra transitions, m = round(n·ρ/(1−ρ)), drawn from the agent's
OWN previous update window instead of from its partners.
"""

import sys

import numpy as np
import pytest
import torch

from hebbian.config import HebbianConfig
from hebbian.graph import HebbianSocialGraph
from rl_layer import ppo_update
from rl_layer.config import RLConfig
from rl_layer.ppo_update import _collect_own_replay, _collect_social_replay
from rl_layer.trajectory_buffer import RolloutBuffer


def _filled_buffer(n, tag):
    buf = RolloutBuffer(max_size=256)
    for t in range(n):
        buf.store_action(prompt_text=f"{tag}{t}", action_idx=0,
                         log_prob=-1.0, value=0.5)
        buf.store_reward(1.0, done=False)
    return buf


class _FakeRL:
    """Just what run_ppo_update and the collectors touch; no model."""

    def __init__(self, own_rho, n=64, tag="w1-"):
        self.agent_id = 0
        self.config = RLConfig(enabled=True, ppo_epochs=1, mini_batch_size=8,
                               own_replay_rho=own_rho)
        self.buffer = _filled_buffer(n, tag)
        self._use_centralized = False
        self.centralized_critic = None
        self._update_count = 0
        self._device = torch.device("cpu")
        self._dtype = torch.bfloat16
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)


@pytest.fixture
def seen_batches(monkeypatch):
    """Record every mini-batch the PPO loop scores; skip the model."""
    seen = []

    def fake_step(rl_layer, batch, **_):
        seen.extend(batch)
        return {"n_kept": 0}

    monkeypatch.setattr(ppo_update, "action_level_ppo_step", fake_step)
    monkeypatch.setattr(ppo_update, "_bootstrap_last_value", lambda rl: 0.0)
    return seen


def test_first_update_adds_nothing_then_keeps_its_window(seen_batches):
    rl = _FakeRL(own_rho=0.3)
    info = ppo_update.run_ppo_update(rl)
    assert info["own_replay_n"] == 0
    assert len(seen_batches) == 64
    assert len(rl._own_replay_pool) == 64
    assert len(rl.buffer) == 0


def test_second_update_adds_matched_count_from_the_previous_window(seen_batches):
    rl = _FakeRL(own_rho=0.3)
    ppo_update.run_ppo_update(rl)
    seen_batches.clear()
    rl.buffer = _filled_buffer(64, "w2-")
    info = ppo_update.run_ppo_update(rl)
    m = round(64 * 0.3 / 0.7)  # 27
    assert info["own_replay_n"] == m
    assert info["social_replay_n"] == 0
    assert len(seen_batches) == 64 + m
    extra = [tr for tr in seen_batches if tr.prompt_text.startswith("w1-")]
    assert len(extra) == m
    # The previous window's transitions keep the GAE they got in their own update.
    assert any(tr.advantage != 0.0 for tr in extra)


def test_off_by_default_keeps_no_pool(seen_batches):
    rl = _FakeRL(own_rho=0.0)
    ppo_update.run_ppo_update(rl)
    rl.buffer = _filled_buffer(64, "w2-")
    info = ppo_update.run_ppo_update(rl)
    assert info["own_replay_n"] == 0
    assert getattr(rl, "_own_replay_pool", None) is None
    assert len(seen_batches) == 128


def test_count_matches_what_sharing_adds():
    """Same ρ, same own buffer: own replay adds what social replay adds from
    partners (social is a little lower after per-partner rounding)."""
    np.random.seed(0)
    rl = _FakeRL(own_rho=0.3)
    rl._own_replay_pool = _filled_buffer(64, "old-").get_all()
    own = _collect_own_replay(rl)

    cfg = HebbianConfig(enabled=True, num_agents=3, social_replay_rho=0.3)
    graph = HebbianSocialGraph(cfg)
    graph.W = np.array([[0.0, 0.38, 0.38], [0.38, 0.0, 0.38],
                        [0.38, 0.38, 0.0]], dtype=np.float32)
    partners = {1: _filled_buffer(64, "p1-"), 2: _filled_buffer(64, "p2-")}
    social = _collect_social_replay(rl, partners, graph)

    assert len(own) == 27
    assert abs(len(own) - len(social)) <= 2
    assert all(tr.prompt_text.startswith("old-") for tr in own)


def test_cli_accepts_the_flag_only_with_rl(monkeypatch):
    from mindforge import cli
    monkeypatch.setattr(sys, "argv", ["prog", "--rl", "--rl-own-replay-rho", "0.3"])
    args = cli.parse_args()
    cli.validate_args(args)
    assert args.rl_own_replay_rho == 0.3
    monkeypatch.setattr(sys, "argv", ["prog", "--rl-own-replay-rho", "0.3"])
    with pytest.raises(SystemExit):
        cli.validate_args(cli.parse_args())
    monkeypatch.setattr(sys, "argv", ["prog", "--rl", "--rl-own-replay-rho", "0.95"])
    with pytest.raises(SystemExit):
        cli.validate_args(cli.parse_args())
