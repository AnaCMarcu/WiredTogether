"""Regression-pin config defaults against thesis paper Tables 6 & 7.

These tests freeze the *source* defaults of RLConfig, HebbianConfig and the
communication-reward constants so any drift between the codebase and the
paper's hyper-parameter tables is caught at test time.

NOTE on paper consistency: paper Table 2 claims chamber comm rewards of
40/20/30/15/20 for ch1..ch5, but the source defines 10/10/20/10/10 (all with
threshold 4). The tests below pin the SOURCE values.
"""

import pytest

from hebbian.config import HebbianConfig
from mindforge.env import communication_rewards as cr
from rl_layer.config import RLConfig


def test_rlconfig_table6_defaults():
    """RLConfig() defaults match paper Table 6 field-by-field."""
    cfg = RLConfig()
    assert cfg.gamma == pytest.approx(0.995)
    assert cfg.gae_lambda == pytest.approx(0.95)
    assert cfg.clip_eps == pytest.approx(0.2)
    assert cfg.entropy_start == pytest.approx(0.05)
    assert cfg.entropy_end == pytest.approx(0.001)
    assert cfg.entropy_anneal_steps == 500
    assert cfg.value_coef == pytest.approx(0.5)
    assert cfg.value_clip_eps == pytest.approx(1.0)
    assert cfg.critic_value_clip_eps == pytest.approx(10.0)
    assert cfg.max_grad_norm == pytest.approx(0.5)
    assert cfg.ppo_epochs == 2
    assert cfg.mini_batch_size == 4
    assert cfg.update_interval == 128
    assert cfg.buffer_size == 2048
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == pytest.approx(0.05)
    assert cfg.dtype == "float16"
    assert cfg.critic_hidden == 256
    assert cfg.critic_lr == pytest.approx(3e-4)
    assert cfg.lr == pytest.approx(1e-4)  # actor (LoRA) learning rate


def test_rlconfig_actions_tuple():
    """RLConfig().actions is the exact 22-entry ordered action space."""
    actions = RLConfig().actions
    assert isinstance(actions, tuple)
    assert len(actions) == 22
    assert actions == (
        "NoOp", "MoveForward", "MoveBackward", "MoveLeft", "MoveRight",
        "Jump", "Sneak", "Dig", "Place",
        "Slot1", "Slot2", "Slot3", "Slot4", "Slot5",
        "TurnRight", "TurnLeft", "LookDown", "LookUp",
        "Drop", "Slot6", "Slot7", "Slot8",
    )


def test_hebbian_table7_defaults():
    """HebbianConfig() defaults match paper Table 7 field-by-field."""
    cfg = HebbianConfig()
    assert cfg.interaction_radius == pytest.approx(5.0)
    assert cfg.engagement_reward_weight == pytest.approx(0.5)
    assert cfg.communication_coactivity_bonus == pytest.approx(0.5)
    assert cfg.eta_0 == pytest.approx(0.01)
    assert cfg.eta_plus == pytest.approx(0.05)
    assert cfg.eta_minus == pytest.approx(0.025)
    assert cfg.reward_norm_R == pytest.approx(300.0)
    assert cfg.coop_eps == pytest.approx(0.05)
    assert cfg.coop_window == 50
    assert cfg.neg_theta == pytest.approx(5.0)
    assert cfg.reward_diffusion_gamma == pytest.approx(0.2)
    assert cfg.decay == pytest.approx(0.0003)
    assert cfg.social_replay_rho == pytest.approx(0.0)
    assert cfg.mode == "reward_modulated"
    assert cfg.init_weight == pytest.approx(0.1)
    assert cfg.init_preset == "none"


def test_configs_disabled_by_default():
    """Both HebbianConfig and RLConfig are no-op (enabled=False) by default."""
    assert HebbianConfig().enabled is False
    assert RLConfig().enabled is False


def test_comm_reward_constants():
    """Communication-reward constants and all five chamber comm thresholds match source."""
    assert cr.BASE_MSG_REWARD == pytest.approx(0.5)
    assert cr.BASE_MSG_CAP == 50
    assert cr.MIN_MSG_LEN == 5
    assert cr.RATE_LIMIT_STEPS == 2
    # Source values; paper Table 2 claims rewards 40/20/30/15/20 instead.
    assert cr.CHAMBER_COMM_THRESHOLDS == {
        "ch1": (4, 10.0, "m_comm_ch1"),
        "ch2": (4, 10.0, "m_comm_ch2"),
        "ch3": (4, 20.0, "m_comm_ch3"),
        "ch4": (4, 10.0, "m_comm_ch4"),
        "ch5": (4, 10.0, "m_comm_ch5"),
    }
