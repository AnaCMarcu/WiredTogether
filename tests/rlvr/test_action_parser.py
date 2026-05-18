"""Tests for ``rlvr.action_parser.parse_action_json``."""

from __future__ import annotations

import json

import pytest

from rlvr.action_parser import VALID_ACTION_NAMES, parse_action_json


def test_full_valid_json_scores_one():
    text = json.dumps({
        "action": "dig",
        "communication_target": 1,
        "thoughts": "going for stone",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is not None
    assert parsed["action"] == "dig"
    assert score == 1.0


def test_missing_optional_drops_to_half():
    text = json.dumps({"action": "forward", "thoughts": "hi"})
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is not None
    assert score == 0.5


def test_missing_thoughts_drops_to_half():
    text = json.dumps({"action": "forward", "communication_target": None})
    parsed, score = parse_action_json(text, n_agents=3)
    assert score == 0.5


def test_malformed_json_returns_none_zero():
    parsed, score = parse_action_json("not json {", n_agents=3)
    assert parsed is None
    assert score == 0.0


def test_non_dict_json_returns_none_zero():
    parsed, score = parse_action_json("[1, 2, 3]", n_agents=3)
    assert parsed is None
    assert score == 0.0


def test_invalid_action_name_returns_none_zero():
    text = json.dumps({
        "action": "fly",  # not in VALID_ACTION_NAMES
        "communication_target": None,
        "thoughts": "",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is None
    assert score == 0.0


def test_missing_action_field_returns_none_zero():
    text = json.dumps({"communication_target": None, "thoughts": ""})
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is None
    assert score == 0.0


def test_nop_is_valid():
    text = json.dumps({
        "action": "nop",
        "communication_target": None,
        "thoughts": "wait",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is not None
    assert score == 1.0


def test_comm_target_out_of_range_is_partial():
    text = json.dumps({
        "action": "forward",
        "communication_target": 5,   # n_agents=3 → invalid
        "thoughts": "",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is not None     # action is valid → still a real action
    assert score == 0.5           # optional invalid → docked


def test_comm_target_bool_is_partial():
    """``True`` is technically an int in Python; we reject it explicitly."""
    text = json.dumps({
        "action": "forward",
        "communication_target": True,
        "thoughts": "",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert score == 0.5


def test_comm_target_none_is_full_score():
    text = json.dumps({
        "action": "forward",
        "communication_target": None,
        "thoughts": "",
    })
    parsed, score = parse_action_json(text, n_agents=3)
    assert score == 1.0


def test_well_formed_strictly_higher_than_partial():
    full = json.dumps({"action": "forward", "communication_target": None, "thoughts": "x"})
    partial = json.dumps({"action": "forward"})
    _, score_full = parse_action_json(full, n_agents=3)
    _, score_partial = parse_action_json(partial, n_agents=3)
    assert score_full > score_partial


@pytest.mark.parametrize("name", sorted(VALID_ACTION_NAMES))
def test_all_valid_action_names_parse(name: str):
    text = json.dumps({"action": name, "communication_target": None, "thoughts": ""})
    parsed, score = parse_action_json(text, n_agents=3)
    assert parsed is not None
    assert score == 1.0
