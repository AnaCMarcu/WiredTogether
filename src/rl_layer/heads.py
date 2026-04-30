"""Lightweight per-agent neural components used by the RL layer.

- ``RunningMeanStd`` — online reward normaliser (Welford).
- ``ActionHead``    — Linear(hidden_size → n_actions) on the LLM's pooled hidden state.
- ``ValueHead``     — small MLP(hidden_size → 1) used as the IPPO baseline.
                      Bypassed when a ``CentralizedCritic`` is in charge.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd:
    """Online running mean/variance using Welford's algorithm.

    Used to normalise rewards before storing in the rollout buffer so the
    value head sees a roughly unit-variance signal regardless of whether
    the episode returns 0.1 (exploration) or 2048 (stage completion).
    """

    def __init__(self, eps: float = 1e-4):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = eps  # start non-zero to avoid /0 on first step

    def update(self, x: float) -> None:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (delta * (x - self.mean) - self.var) / self.count

    def normalize(self, x: float) -> float:
        return x / (self.var ** 0.5 + 1e-8)


class ActionHead(nn.Module):
    """Maps LLM pooled hidden state → logits over discrete Craftium actions."""

    def __init__(self, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Linear(hidden_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """Maps LLM pooled hidden state → scalar V-estimate (IPPO baseline)."""

    def __init__(self, hidden_size: int, value_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, value_hidden),
            nn.Tanh(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
