"""Tests for ``mindforge.multi_agent_craftium_grpo`` config loading.

Only covers what runs without torch/gymnasium — config parsing, the
action-name → env-int mapping, and the YAML-on-disk schema. Anything that
actually instantiates the model or env is HPC-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.multi_agent_craftium_grpo import (
    EnvConfig,
    FullConfig,
    PolicyConfig,
    _action_dict_to_env,
    _apply_overrides,
    _build_full_config,
    _parse_override_value,
    load_config,
)


def test_full_config_defaults():
    cfg = _build_full_config({})
    assert isinstance(cfg, FullConfig)
    assert cfg.env.num_agents == 3
    assert cfg.policy.trained_agent == 0
    assert cfg.policy.checkpoint_path is None
    assert cfg.sampler.n_per_group == 4
    assert cfg.sampler.horizon == 50
    assert cfg.grpo.clip_epsilon == 0.2
    assert cfg.verifier.use_milestone_rewards is True
    assert cfg.hebbian == {}


def test_full_config_overrides():
    cfg = _build_full_config({
        "env": {"num_agents": 5},
        "policy": {"trained_agent": 2},
        "grpo": {"total_steps": 42},
    })
    assert cfg.env.num_agents == 5
    assert cfg.policy.trained_agent == 2
    assert cfg.grpo.total_steps == 42


def test_action_dict_to_env_nop_is_zero():
    assert _action_dict_to_env({"action": "nop"}) == 0


def test_action_dict_to_env_dig_is_seven():
    # _DISCRETE_ACTIONS: forward(0), backward(1), left(2), right(3), jump(4),
    # sneak(5), dig(6) → env action index = 6 + 1 = 7
    assert _action_dict_to_env({"action": "dig"}) == 7


def test_action_dict_to_env_invalid_falls_back_to_nop():
    assert _action_dict_to_env({"action": "fly"}) == 0
    assert _action_dict_to_env({}) == 0


def test_yaml_config_on_disk_loads(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = repo_root / "configs/rlvr/grpo_single_agent_ch3.yaml"
    cfg = load_config(yaml_path)
    assert cfg.env.num_agents == 3
    assert cfg.sampler.n_per_group == 4
    assert cfg.sampler.horizon == 50
    assert cfg.grpo.clip_epsilon == 0.2
    assert cfg.grpo.kl_coefficient == 0.05
    assert cfg.verifier.format_reward_weight == 0.1
    assert cfg.hebbian == {"enabled": False}
    assert cfg.policy.trained_agent == 0
    assert cfg.policy.checkpoint_path is None
    assert cfg.seed == 0


# ──── --set overrides ───────────────────────────────────────────────────


def test_parse_override_value_types():
    assert _parse_override_value("42") == 42
    assert _parse_override_value("3.14") == 3.14
    assert _parse_override_value("true") is True
    assert _parse_override_value("False") is False
    assert _parse_override_value("null") is None
    assert _parse_override_value("none") is None
    assert _parse_override_value("/scratch/foo") == "/scratch/foo"
    assert _parse_override_value("[1, 2, 3]") == [1, 2, 3]
    assert _parse_override_value("[]") == []


def test_apply_overrides_nested():
    data = {"env": {"num_agents": 3}}
    out = _apply_overrides(data, ["env.num_agents=5"])
    assert out["env"]["num_agents"] == 5


def test_apply_overrides_creates_missing_intermediates():
    out = _apply_overrides({}, ["llm.lora_config.r=32"])
    assert out["llm"]["lora_config"]["r"] == 32


def test_apply_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        _apply_overrides({}, ["missing_equals"])
    with pytest.raises(ValueError):
        _apply_overrides({}, ["=value_only"])


def test_load_config_applies_overrides(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = repo_root / "configs/rlvr/grpo_single_agent_ch3.yaml"
    cfg = load_config(yaml_path, overrides=[
        "seed=99",
        "llm.base_model_name=/scratch/models/Qwen3.5-2B",
        "grpo.total_steps=2000",
        "policy.checkpoint_path=null",
    ])
    assert cfg.seed == 99
    assert cfg.llm.base_model_name == "/scratch/models/Qwen3.5-2B"
    assert cfg.grpo.total_steps == 2000
    assert cfg.policy.checkpoint_path is None


def test_load_config_override_list_field(tmp_path: Path):
    """``policy.trained_agents`` is a list — override syntax handles it."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    yaml_path = repo_root / "configs/rlvr/grpo_single_agent_ch3.yaml"
    cfg = load_config(yaml_path, overrides=["policy.trained_agents=[0, 1, 2]"])
    assert cfg.policy.effective_trained_agents() == [0, 1, 2]
