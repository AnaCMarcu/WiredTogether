"""--disable-switch: the Chamber 3 contribution test.

Python writes the disabled switch index to ``disabled_switch.txt`` in the
world dir after each episode reset; switches.lua reads it on every press and
ignores presses of that switch. These tests cover the Python half (the flag
file and the CLI); the Lua half needs a running world.
"""

from __future__ import annotations

import sys

import pytest

from mindforge import cli
from mindforge.custom_environment_craftium import CraftiumEnvironmentInterface


def _env(world_dir) -> CraftiumEnvironmentInterface:
    env = object.__new__(CraftiumEnvironmentInterface)
    env._world_path = str(world_dir)          # what _get_world_path() caches
    return env


def test_writes_the_switch_index(tmp_path):
    assert _env(tmp_path).set_disabled_switch(2)
    assert (tmp_path / "disabled_switch.txt").read_text() == "2"
    assert not (tmp_path / "disabled_switch.txt.tmp").exists()


def test_none_removes_the_flag(tmp_path):
    env = _env(tmp_path)
    env.set_disabled_switch(0)
    assert env.set_disabled_switch(None)
    assert not (tmp_path / "disabled_switch.txt").exists()
    assert env.set_disabled_switch(None)      # removing twice is fine


def test_unresolved_world_path_is_reported(tmp_path):
    env = object.__new__(CraftiumEnvironmentInterface)   # no _world_path, no env
    assert env.set_disabled_switch(1) is False


@pytest.mark.parametrize("value", ["none", "rotate"])
def test_cli_accepts_the_flag(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["x", "--disable-switch", value])
    args = cli.parse_args()
    cli.validate_args(args)
    assert args.disable_switch == value


def test_cli_default_is_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert cli.parse_args().disable_switch == "none"
