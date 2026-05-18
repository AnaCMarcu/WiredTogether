"""Tests for ``rlvr.grpo_trainer.GRPOTrainer``.

The full ``step()`` flow requires torch + a real PEFT-adapted model,
which is HPC-only. These local tests cover only the configuration /
metrics-dataclass surfaces. The plan's §5.2 task-7 weight-changes test
runs on HPC.
"""

from __future__ import annotations

from rlvr.grpo_trainer import GRPOConfig, GRPOStepMetrics, _std


def test_grpo_config_defaults_match_plan():
    cfg = GRPOConfig()
    assert cfg.clip_epsilon == 0.2
    assert cfg.kl_coefficient == 0.05
    assert cfg.learning_rate == 5e-6
    assert cfg.n_per_group == 4


def test_grpo_step_metrics_serialisable():
    m = GRPOStepMetrics(
        step=1, group_size=4, group_mean_reward=10.0,
        group_reward_std=2.0, advantage_mean_abs=0.5,
        surrogate_loss=-0.1, kl_loss=0.01, total_loss=-0.099,
        fraction_clipped=0.1, grad_norm=0.5,
    )
    d = m.as_dict()
    assert d["step"] == 1
    assert d["fraction_clipped"] == 0.1


def test_std_helper_zero_for_singleton():
    assert _std([1.0]) == 0.0


def test_std_helper_matches_population_std():
    # std of [1, 1, 0, 0] = sqrt(0.25) = 0.5
    assert abs(_std([1.0, 1.0, 0.0, 0.0]) - 0.5) < 1e-9
