"""One LoRA adapter per RL agent (the method as described).

Before 2026-11-09 the adapter was named after the agent's role only. In
homogeneous teams every agent has the role "agent", so all agents trained one
shared adapter, each with its own optimiser. The default now gives every agent
its own adapter; --rl-shared-adapter restores the old naming for reproduction.
"""

import sys

import pytest

from rl_layer.config import RLConfig
from rl_layer.rl_layer import adapter_name_for


def test_homogeneous_team_gets_one_adapter_per_agent():
    cfg = RLConfig()
    names = [adapter_name_for(cfg, "agent", i) for i in range(3)]
    assert names == ["agent_0", "agent_1", "agent_2"]
    # The optimiser selects parameters by ".<adapter>." in their name:
    # no tag may be a substring match of another agent's parameters.
    for a in names:
        for b in names:
            if a != b:
                assert f".{a}." not in f"x.lora_A.{b}.weight"


def test_legacy_naming_shares_one_adapter():
    cfg = RLConfig(lora_per_agent=False)
    assert {adapter_name_for(cfg, "agent", i) for i in range(3)} == {"agent"}
    assert adapter_name_for(RLConfig(lora_per_agent=False, lora_per_role=False),
                            "hunter", 1) == "shared"


def test_heterogeneous_roles_stay_distinct_either_way():
    for per_agent in (True, False):
        cfg = RLConfig(lora_per_agent=per_agent)
        names = {adapter_name_for(cfg, r, i)
                 for i, r in enumerate(("gatherer", "hunter", "defender"))}
        assert len(names) == 3


def test_cli_flag_restores_shared_adapter(monkeypatch):
    from mindforge import cli
    monkeypatch.setattr(sys, "argv", ["prog", "--rl"])
    assert cli.parse_args().rl_shared_adapter is False
    monkeypatch.setattr(sys, "argv", ["prog", "--rl", "--rl-shared-adapter"])
    args = cli.parse_args()
    cli.validate_args(args)
    assert args.rl_shared_adapter is True
