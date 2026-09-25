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


# The paper's social-plasticity hyperparameters (Table 8), by config field.
TABLE8 = {
    "mode": "three_factor",
    "interaction_radius": 5.0,           # d
    "engagement_reward_weight": 0.5,     # α
    "communication_coactivity_bonus": 0.5,  # δ_k
    "coact_floor": 0.25,                 # c_0
    "coop_eps": 0.05,                    # ε_c
    "init_weight": 0.1,                  # W_0
    "eligibility_rho": 0.9,              # ρ_e
    "eta_0": 0.001,                      # η_0
    "eta_plus": 0.05,                    # η_+
    "reward_norm_R": 50.0,               # R
    "eta_minus_death": 0.05,             # η_-^d
    "death_cap": 10.0,                   # r_cap
    "decay": 0.001,                      # λ
    "reward_diffusion_gamma": 0.2,       # γ_d
    "social_replay_rho": 0.3,            # ρ
}

# CLI flag -> config field, for the settings the paper's launchers vary.
CLI_TO_FIELD = {
    "hebbian_mode": "mode",
    "hebbian_radius": "interaction_radius",
    "hebbian_alpha": "engagement_reward_weight",
    "hebbian_coact_floor": "coact_floor",
    "hebbian_coop_eps": "coop_eps",
    "hebbian_init_weight": "init_weight",
    "hebbian_eligibility_rho": "eligibility_rho",
    "hebbian_eta_0": "eta_0",
    "hebbian_eta_plus": "eta_plus",
    "hebbian_reward_norm": "reward_norm_R",
    "hebbian_death_ltd": "eta_minus_death",
    "hebbian_death_cap": "death_cap",
    "hebbian_decay": "decay",
    "hebbian_gamma": "reward_diffusion_gamma",
    "hebbian_rho": "social_replay_rho",
}


def test_hebbian_defaults_match_paper_table8():
    """HebbianConfig() defaults are the paper's rule, field by field."""
    cfg = HebbianConfig()
    for field, value in TABLE8.items():
        got = getattr(cfg, field)
        if isinstance(value, str):
            assert got == value, field
        else:
            assert got == pytest.approx(value), field
    assert cfg.init_preset == "none"


def test_cli_defaults_match_config_defaults(monkeypatch):
    """`--hebbian` with no other flag runs the Table 8 rule: every CLI
    default equals the config default (they had drifted apart on λ)."""
    import sys
    from mindforge import cli

    monkeypatch.setattr(sys, "argv", ["x"])
    args = cli.parse_args()
    for flag, field in CLI_TO_FIELD.items():
        want = TABLE8[field]
        got = getattr(args, flag)
        if isinstance(want, str):
            assert got == want, flag
        else:
            assert got == pytest.approx(want), flag


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
