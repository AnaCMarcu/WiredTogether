"""Tests for rl_layer.heads (RunningMeanStd / ActionHead / ValueHead) and
rl_layer.ppo_update._anneal_entropy.

All expected values are hand-computed from the source:
- RunningMeanStd starts with mean=0, var=1, count=eps=1e-4 (Welford with a
  tiny prior), so after n updates mean == sum(x)/(n + 1e-4).
- normalize() divides by (sqrt(var) + 1e-8) WITHOUT subtracting the mean.
- _anneal_entropy is linear from entropy_start to entropy_end over
  entropy_anneal_steps PPO updates, clamped at the end value; with
  entropy_anneal_steps <= 0 it falls back to config.entropy_coef
  (NOT entropy_start).
"""

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl_layer.config import RLConfig
from rl_layer.heads import ActionHead, RunningMeanStd, ValueHead
from rl_layer.ppo_update import _anneal_entropy


# ── RunningMeanStd ───────────────────────────────────────────────────────────

def test_running_mean_std_matches_numpy():
    """RunningMeanStd over 1000 normals matches np.mean / population np.var up to the 1e-4 prior count."""
    rms = RunningMeanStd()
    # Pin the documented initial state (prior: mean 0, var 1, count eps).
    assert rms.mean == 0.0
    assert rms.var == 1.0
    assert rms.count == pytest.approx(1e-4)

    xs = np.random.normal(loc=0.5, scale=2.0, size=1000)
    for x in xs:
        rms.update(float(x))

    assert rms.count == pytest.approx(1000 + 1e-4)
    # Welford with prior count c0 and prior mean 0 gives exactly sum(x)/(n+c0).
    assert rms.mean == pytest.approx(xs.sum() / (1000 + 1e-4), rel=1e-9)
    # The 1e-4 prior biases both stats by O(1e-7) relative — far inside rel=1e-4.
    assert rms.mean == pytest.approx(np.mean(xs), rel=1e-4)
    assert rms.var == pytest.approx(np.var(xs), rel=1e-4)  # population (ddof=0)


def test_running_mean_std_normalize_no_mean_subtraction():
    """normalize(x) == x / (sqrt(var) + 1e-8) — scales only, never subtracts the mean."""
    rms = RunningMeanStd()
    for x in (10.0, 12.0, 14.0):  # nonzero running mean
        rms.update(x)
    assert rms.mean > 0.0

    for x in (-3.0, 0.0, 7.5):
        assert rms.normalize(x) == pytest.approx(x / (rms.var ** 0.5 + 1e-8))
    # Zero maps to zero exactly; if the mean were subtracted it would not.
    assert rms.normalize(0.0) == 0.0
    # And the value differs from the mean-centred alternative.
    centred = (7.5 - rms.mean) / (rms.var ** 0.5 + 1e-8)
    assert rms.normalize(7.5) != pytest.approx(centred)


# ── ValueHead / ActionHead ───────────────────────────────────────────────────

def test_value_head_structure_default():
    """ValueHead.net is Sequential[Linear(h,256), Tanh, Linear(256,1)] with default value_hidden=256."""
    vh = ValueHead(hidden_size=16)
    layers = list(vh.net)
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear)
    assert (layers[0].in_features, layers[0].out_features) == (16, 256)
    assert isinstance(layers[1], nn.Tanh)
    assert isinstance(layers[2], nn.Linear)
    assert (layers[2].in_features, layers[2].out_features) == (256, 1)


def test_value_head_forward_shape():
    """ValueHead forward maps (B, hidden_size) -> (B, 1), honouring custom value_hidden."""
    vh = ValueHead(hidden_size=16, value_hidden=8)
    assert (vh.net[0].in_features, vh.net[0].out_features) == (16, 8)
    assert (vh.net[2].in_features, vh.net[2].out_features) == (8, 1)
    out = vh(torch.randn(5, 16))
    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


def test_action_head_structure_and_shape():
    """ActionHead is a single Linear(hidden_size, n_actions); forward maps (B,h) -> (B,n_actions)."""
    ah = ActionHead(hidden_size=16, n_actions=22)
    assert isinstance(ah.net, nn.Linear)
    assert (ah.net.in_features, ah.net.out_features) == (16, 22)
    out = ah(torch.randn(3, 16))
    assert out.shape == (3, 22)
    assert torch.isfinite(out).all()


# ── _anneal_entropy ──────────────────────────────────────────────────────────

def _rl_stub(update_count: int, **config_overrides):
    """Minimal stand-in: _anneal_entropy reads only rl.config and rl._update_count."""
    return types.SimpleNamespace(
        config=RLConfig(**config_overrides), _update_count=update_count
    )


def test_anneal_entropy_defaults_pinned():
    """RLConfig anneal defaults: start=0.05, end=0.001, steps=500, static coef=0.01."""
    cfg = RLConfig()
    assert cfg.entropy_start == 0.05
    assert cfg.entropy_end == 0.001
    assert cfg.entropy_anneal_steps == 500
    assert cfg.entropy_coef == 0.01


def test_anneal_entropy_linear_schedule():
    """_anneal_entropy decays linearly 0.05 -> 0.001 over 500 updates: k=0 gives 0.05, k=250 gives 0.0255."""
    assert _anneal_entropy(_rl_stub(0)) == pytest.approx(0.05)
    # 0.05 + 0.5 * (0.001 - 0.05) = 0.05 - 0.0245 = 0.0255
    assert _anneal_entropy(_rl_stub(250)) == pytest.approx(0.0255)
    # Quarter point: 0.05 + 0.25 * (-0.049) = 0.03775
    assert _anneal_entropy(_rl_stub(125)) == pytest.approx(0.03775)


def test_anneal_entropy_clamps_at_end():
    """Progress clamps at 1.0: k=500 and k=10000 both return entropy_end=0.001."""
    assert _anneal_entropy(_rl_stub(500)) == pytest.approx(0.001)
    assert _anneal_entropy(_rl_stub(10000)) == pytest.approx(0.001)


def test_anneal_entropy_zero_steps_falls_back_to_entropy_coef():
    """Quirk: entropy_anneal_steps <= 0 returns the STATIC config.entropy_coef (0.01), not entropy_start (0.05)."""
    assert _anneal_entropy(_rl_stub(0, entropy_anneal_steps=0)) == pytest.approx(0.01)
    assert _anneal_entropy(_rl_stub(123, entropy_anneal_steps=0)) == pytest.approx(0.01)
    assert _anneal_entropy(_rl_stub(0, entropy_anneal_steps=-5)) == pytest.approx(0.01)
    # The fallback tracks entropy_coef, not the anneal endpoints.
    assert _anneal_entropy(
        _rl_stub(0, entropy_anneal_steps=0, entropy_coef=0.42)
    ) == pytest.approx(0.42)
