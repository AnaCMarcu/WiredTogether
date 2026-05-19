"""Tests for ``rlvr.reward_propagation``.

Three concerns:
    * ``per_teammate_contributions`` — math matches Eq. 8 decomposition
    * ``format_propagation_prompt`` — output string is deterministic,
      includes attribution events, collapses to ``""`` when empty
    * ``attribute_source_events`` — picks up agent_id → milestone_id mapping

All pure-python — no torch, no env, no LLM.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlvr.reward_propagation import (
    attribute_source_events,
    build_interpretability_record,
    format_propagation_prompt,
    per_teammate_contributions,
)


# ──── per_teammate_contributions ─────────────────────────────────────────


def test_contributions_sum_matches_eq8():
    """Sum of per-teammate contributions = γ · Σ_{j≠i} w̄_ij · c_ij · r_j."""
    raw = [10.0, 5.0, 2.0]
    w_bar = [0.0, 0.4, 0.6]   # agent 0's row (diagonal masked)
    coact = [1.0, 1.0, 1.0]
    gamma = 0.2
    contrib = per_teammate_contributions(
        agent_id=0, raw_rewards=raw,
        w_bar_row=w_bar, coactivity_row=coact, gamma=gamma,
    )
    # Expected: γ * (0.4 * 1.0 * 5.0 + 0.6 * 1.0 * 2.0) = 0.2 * (2.0 + 1.2) = 0.64
    total = sum(contrib.values())
    assert total == pytest.approx(0.64, abs=1e-9)


def test_contributions_skip_self():
    """The self-index never appears in the output dict."""
    raw = [1.0, 1.0, 1.0]
    contrib = per_teammate_contributions(
        agent_id=1, raw_rewards=raw,
        w_bar_row=[0.5, 0.0, 0.5],
        coactivity_row=[1.0, 1.0, 1.0],
        gamma=0.2,
    )
    assert 1 not in contrib
    assert set(contrib.keys()) == {0, 2}


def test_contributions_zero_when_coactivity_zero():
    """No co-activity → no propagation, even with strong bond."""
    contrib = per_teammate_contributions(
        agent_id=0,
        raw_rewards=[0.0, 100.0],
        w_bar_row=[0.0, 1.0],
        coactivity_row=[0.0, 0.0],
        gamma=0.5,
    )
    assert contrib[1] == 0.0


def test_contributions_zero_when_weight_zero():
    """No bond → no propagation."""
    contrib = per_teammate_contributions(
        agent_id=0,
        raw_rewards=[0.0, 100.0],
        w_bar_row=[0.0, 0.0],
        coactivity_row=[1.0, 1.0],
        gamma=0.5,
    )
    assert contrib[1] == 0.0


def test_contributions_zero_when_gamma_zero():
    """γ=0 disables the propagation channel entirely."""
    contrib = per_teammate_contributions(
        agent_id=0,
        raw_rewards=[5.0, 5.0],
        w_bar_row=[0.0, 1.0],
        coactivity_row=[1.0, 1.0],
        gamma=0.0,
    )
    assert contrib[1] == 0.0


def test_contributions_handle_numpy_arrays():
    """Both lists and numpy arrays work for w_bar / coactivity."""
    contrib = per_teammate_contributions(
        agent_id=0,
        raw_rewards=[1.0, 2.0, 3.0],
        w_bar_row=np.array([0.0, 0.5, 0.5]),
        coactivity_row=np.array([1.0, 1.0, 1.0]),
        gamma=0.2,
    )
    assert contrib[1] == pytest.approx(0.2 * 0.5 * 1.0 * 2.0, abs=1e-9)
    assert contrib[2] == pytest.approx(0.2 * 0.5 * 1.0 * 3.0, abs=1e-9)


def test_contributions_rejects_shape_mismatch():
    """If the rows are wrong-shape, fail loudly rather than silently truncate."""
    with pytest.raises(ValueError):
        per_teammate_contributions(
            agent_id=0,
            raw_rewards=[1.0, 2.0, 3.0],
            w_bar_row=[0.0, 0.5],   # too short
            coactivity_row=[1.0, 1.0, 1.0],
            gamma=0.2,
        )


def test_contributions_negative_reward_propagates_as_negative():
    """A teammate's negative reward propagates as a negative delta."""
    contrib = per_teammate_contributions(
        agent_id=0,
        raw_rewards=[0.0, -10.0],
        w_bar_row=[0.0, 1.0],
        coactivity_row=[1.0, 1.0],
        gamma=0.2,
    )
    assert contrib[1] == pytest.approx(-2.0, abs=1e-9)


# ──── format_propagation_prompt ──────────────────────────────────────────


def test_format_empty_when_no_contributions():
    """Empty dict / all-zero deltas → empty string (prompt template collapses)."""
    assert format_propagation_prompt({}) == ""
    assert format_propagation_prompt({1: 0.0, 2: 0.0}) == ""


def test_format_below_threshold_omitted():
    """Tiny deltas omitted from the line so it stays readable."""
    s = format_propagation_prompt({1: 0.0001, 2: 2.5}, threshold=0.01)
    assert "agent_1" not in s
    assert "agent_2" in s
    assert "+2.50" in s


def test_format_one_teammate_no_event():
    s = format_propagation_prompt({1: 2.5})
    assert s == "Propagated rewards this step: +2.50 from agent_1"


def test_format_one_teammate_with_event():
    s = format_propagation_prompt(
        {1: 2.5},
        source_events={1: "m17_switch_pressed"},
    )
    assert "agent_1" in s
    assert "(m17_switch_pressed)" in s


def test_format_two_teammates_ordered_by_id():
    """Stable ordering — ascending teammate id — for deterministic prompts."""
    s = format_propagation_prompt({2: 1.0, 1: 3.0})
    # agent_1 should appear before agent_2 even though 2 was inserted first.
    pos_1 = s.index("agent_1")
    pos_2 = s.index("agent_2")
    assert pos_1 < pos_2


def test_format_includes_role_when_provided():
    s = format_propagation_prompt(
        {1: 2.5},
        role_names={1: "gatherer"},
    )
    assert "(gatherer)" in s


def test_format_negative_delta_signed():
    s = format_propagation_prompt({1: -1.5})
    # Negative number includes its own sign.
    assert "-1.50 from agent_1" in s
    # Make sure we don't emit "+-1.50".
    assert "+-" not in s


def test_format_skips_reward_label_when_no_event():
    """When source_events says 'reward' (the placeholder for non-milestone),
    don't add a parenthetical — keeps the line short. (The header
    'Propagated rewards' is unrelated to the source_events sentinel.)"""
    s = format_propagation_prompt(
        {1: 2.5},
        source_events={1: "reward"},
    )
    assert "(reward)" not in s
    assert s == "Propagated rewards this step: +2.50 from agent_1"


# ──── attribute_source_events ────────────────────────────────────────────


def test_attribute_basic():
    events = [
        {"step": 5, "agent_id": 0, "milestone_id": "m17_switch_pressed"},
        {"step": 5, "agent_id": 1, "milestone_id": "m_comm_ch3"},
    ]
    out = attribute_source_events(events)
    assert out == {0: "m17_switch_pressed", 1: "m_comm_ch3"}


def test_attribute_empty():
    assert attribute_source_events([]) == {}


def test_attribute_handles_malformed_records():
    """Bad records (missing keys / wrong types) skipped silently."""
    events = [
        {"step": 5},   # no agent_id, no milestone_id
        {"agent_id": "agent_0", "milestone_id": "m1_move_5"},  # str agent_id
        {"agent_id": 0, "milestone_id": "m1_move_5"},
    ]
    out = attribute_source_events(events)
    assert out == {0: "m1_move_5"}


def test_attribute_last_wins_when_same_agent_fires_multiple():
    """Two milestones for the same agent in the same step → last wins.
    Stage-side decision; alternative would be concatenation."""
    events = [
        {"agent_id": 0, "milestone_id": "m1_move_5"},
        {"agent_id": 0, "milestone_id": "m2_dig_3_any"},
    ]
    out = attribute_source_events(events)
    assert out == {0: "m2_dig_3_any"}


# ──── legacy schema support ─────────────────────────────────────────────


def test_attribute_legacy_schema():
    """Legacy events use ``milestone`` key + ``contributors`` list."""
    events = [
        {"step": 5, "milestone": "m17_switch_pressed",
         "contributors": ["agent_0", "agent_1"]},
        {"step": 5, "milestone": "m_comm_ch3", "contributors": ["agent_2"]},
    ]
    out = attribute_source_events(events)
    # First contributor takes credit.
    assert out == {0: "m17_switch_pressed", 2: "m_comm_ch3"}


def test_attribute_legacy_no_contributors_skipped():
    events = [{"milestone": "m1_move_5", "contributors": []}]
    assert attribute_source_events(events) == {}


def test_attribute_mixed_schemas():
    """A stream with both GRPO and legacy records → both parsed correctly."""
    events = [
        {"agent_id": 0, "milestone_id": "m17_switch_pressed"},      # GRPO
        {"milestone": "m_comm_ch1", "contributors": ["agent_2"]},   # legacy
    ]
    out = attribute_source_events(events)
    assert out == {0: "m17_switch_pressed", 2: "m_comm_ch1"}


def test_attribute_legacy_malformed_contributor_skipped():
    """Contributor not starting with 'agent_' → silently skipped."""
    events = [
        {"milestone": "m1_move_5", "contributors": ["not_an_agent"]},
        {"milestone": "m2_dig_3_any", "contributors": ["agent_xyz"]},  # bad suffix
    ]
    assert attribute_source_events(events) == {}


# ──── build_interpretability_record ────────────────────────────────────


def test_record_minimal_shape():
    record = build_interpretability_record(
        step=42, agent_id=1, chamber="ch3",
        bond_row=[0.0, 0.4, 0.6],
        parsed_action={"action": "dig",
                       "communication_target": 0,
                       "thoughts": "agent_0 has the switch"},
    )
    assert record["step"] == 42
    assert record["agent_id"] == 1
    assert record["chamber"] == "ch3"
    assert record["bond_row"] == [0.0, 0.4, 0.6]
    assert record["chosen_action"] == "dig"
    assert record["communication_target"] == 0
    assert record["thoughts_excerpt"] == "agent_0 has the switch"
    assert record["propagated_delta_by_teammate"] == {}
    assert record["propagated_source_events"] == {}


def test_record_with_propagation():
    record = build_interpretability_record(
        step=10, agent_id=0, chamber="ch3",
        bond_row=[0.0, 0.5, 0.3],
        parsed_action={"action": "forward"},
        propagated_contribs={1: 2.5, 2: 0.3},
        propagated_sources={1: "m17_switch_pressed"},
    )
    # Dict keys are stringified for clean JSON.
    assert record["propagated_delta_by_teammate"] == {"1": 2.5, "2": 0.3}
    assert record["propagated_source_events"] == {"1": "m17_switch_pressed"}


def test_record_thoughts_truncated():
    """Long thoughts truncated to keep the sidecar bounded."""
    long_thoughts = "a" * 200
    record = build_interpretability_record(
        step=0, agent_id=0, chamber="ch1",
        bond_row=[0.0],
        parsed_action={"action": "nop", "thoughts": long_thoughts},
        thoughts_max_chars=20,
    )
    assert len(record["thoughts_excerpt"]) <= 20
    assert record["thoughts_excerpt"].endswith("…")


def test_record_handles_malformed_action():
    """Bad JSON / non-dict action → missing fields render as None."""
    record = build_interpretability_record(
        step=0, agent_id=0, chamber="ch1",
        bond_row=[0.0],
        parsed_action=None,
    )
    assert record["chosen_action"] is None
    assert record["communication_target"] is None
    assert record["thoughts_excerpt"] is None


def test_record_rejects_bool_comm_target():
    """``True`` would pass isinstance(int) — must be filtered."""
    record = build_interpretability_record(
        step=0, agent_id=0, chamber=None,
        bond_row=[0.0],
        parsed_action={"action": "nop", "communication_target": True},
    )
    assert record["communication_target"] is None


def test_record_coerces_numpy_bond_row():
    """numpy arrays serialize as plain lists."""
    record = build_interpretability_record(
        step=0, agent_id=0, chamber="ch1",
        bond_row=np.array([0.0, 0.1, 0.2]),
        parsed_action={"action": "nop"},
    )
    assert record["bond_row"] == [0.0, 0.1, 0.2]
    assert isinstance(record["bond_row"], list)


def test_record_json_serializable():
    """End-to-end: build → json.dumps → json.loads round-trip."""
    import json
    record = build_interpretability_record(
        step=10, agent_id=0, chamber="ch3",
        bond_row=np.array([0.0, 0.4, 0.6]),
        parsed_action={"action": "dig", "communication_target": 1,
                       "thoughts": "x"},
        propagated_contribs={1: 2.5},
        propagated_sources={1: "m17_switch_pressed"},
    )
    text = json.dumps(record)
    roundtrip = json.loads(text)
    assert roundtrip["step"] == 10
    assert roundtrip["bond_row"] == [0.0, 0.4, 0.6]


def test_record_chamber_none_serializes_as_none():
    record = build_interpretability_record(
        step=0, agent_id=0, chamber=None,
        bond_row=[0.0],
        parsed_action={"action": "nop"},
    )
    assert record["chamber"] is None
