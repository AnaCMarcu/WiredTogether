"""Tests for the Stage-4 Hebbian YAML configs.

Verifies the configs on disk load cleanly and route to the right
combination of 4a / 4b flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.multi_agent_craftium_grpo import load_config


def _config(name: str):
    repo_root = Path(__file__).resolve().parent.parent.parent
    return load_config(repo_root / f"configs/rlvr/{name}")


def test_full_hebbian_config():
    cfg = _config("grpo_hebbian_full.yaml")
    assert cfg.policy.effective_trained_agents() == [0, 1, 2]
    assert cfg.verifier.hebbian_reward_diffusion is True
    assert cfg.grpo.hebbian_group_composition is True
    assert cfg.grpo.hebbian_borrow_fraction == 0.25
    assert cfg.hebbian["enabled"] is True
    assert cfg.hebbian["reward_diffusion_gamma"] == 0.2


def test_diffusion_only_config():
    cfg = _config("grpo_hebbian_diffusion.yaml")
    assert cfg.verifier.hebbian_reward_diffusion is True
    assert cfg.grpo.hebbian_group_composition is False
    assert cfg.hebbian["enabled"] is True


def test_composition_only_config():
    cfg = _config("grpo_hebbian_composition.yaml")
    assert cfg.verifier.hebbian_reward_diffusion is False
    assert cfg.grpo.hebbian_group_composition is True
    assert cfg.hebbian["enabled"] is True


def test_stage2_and_stage3_configs_have_hebbian_disabled():
    """Sanity: existing non-Hebbian configs are not accidentally activating
    the bridge."""
    for name in ("grpo_single_agent_ch3.yaml", "grpo_multi_agent.yaml"):
        cfg = _config(name)
        assert cfg.verifier.hebbian_reward_diffusion is False
        assert cfg.grpo.hebbian_group_composition is False
        assert cfg.hebbian.get("enabled", False) is False, name
