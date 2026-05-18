"""Tests for ``rlvr.reward_table.build_milestone_rewards``.

Spot-checks one milestone per track against the values hard-coded in
``mindforge.agent_modules.craftium_metric.TRACKS``. If these tests break, it
means the reward table was edited — either update the assertions to match the
new design, or revert the change.
"""

from __future__ import annotations

from rlvr.reward_table import build_milestone_rewards


def test_known_milestone_rewards():
    r = build_milestone_rewards()
    # One assertion per track to catch any track being dropped.
    assert r["m1_move_5"] == 10.0           # ch1_solo
    assert r["m8_anvil_A1"] == 40.0         # ch2_anvils
    assert r["m17_switch_pressed"] == 40.0  # ch3_switches
    assert r["m22_all_mobs_killed"] == 150.0  # ch4_combat
    assert r["m27_boss_defeated"] == 300.0  # ch5_boss
    assert r["m_comm_ch1"] == 40.0          # communication


def test_all_values_are_float():
    r = build_milestone_rewards()
    assert r, "reward table should be non-empty"
    for mid, reward in r.items():
        assert isinstance(reward, float), f"{mid} maps to {type(reward).__name__}, not float"


def test_no_milestone_collision():
    """``TRACKS`` lists each milestone once; the flat dict's size matches the
    sum of per-track entry counts. If this fails, a milestone was duplicated
    across tracks.
    """
    from mindforge.agent_modules.craftium_metric import TRACKS

    expected_count = sum(len(entries) for entries in TRACKS.values())
    assert len(build_milestone_rewards()) == expected_count
