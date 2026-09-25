"""Tests for the orchestrator's shared plumbing: config, per-episode state,
events, the tolerant JSON parser, the text world-state renderer and the log
writer. The villager controller itself is covered in
tests/test_orchestrator_villager.py.
"""

import json

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState
from orchestrator import events as oevents
from orchestrator import core as ocore
from orchestrator import map_render as omap
from orchestrator.logging import OrchestratorLogger


def good_response(**overrides) -> dict:
    resp = {
        "ledger": {
            "task_facts": ["anvils need 2 punchers within 1s"],
            "progress": {
                "current_stage_goal": "break both anvils",
                "assignments": {},
                "issued_at_step": 12,
                "expected_signal": "milestone m8 within ~30 steps",
            },
            "stall_counter": 0,
        },
        "directives": {
            "agent_0": {"comm_target": "agent_1", "help": "punch anvil A"},
            "agent_1": {"comm_target": "agent_0", "help": "punch anvil A"},
            "agent_2": {"comm_target": "agent_0", "help": "scout door 2"},
        },
        "changed": True,
        "why": "new stage",
    }
    resp.update(overrides)
    return resp


# ── Config ────────────────────────────────────────────────────────────────

def test_config_defaults_are_disabled_noop():
    cfg = OrchestratorConfig()
    assert cfg.enabled is False
    assert cfg.variant == "villager"
    # The decomposer interval must match the social module's T_soc default
    # (--social-interval) so the two conditions are matched on call rate.
    assert cfg.decompose_min_interval == 8
    assert cfg.node_timeout_steps == 60
    assert cfg.max_open_tasks == 0
    assert cfg.model is None
    assert cfg.log_dir_name == "orchestrator"


def test_config_rejects_removed_variants():
    OrchestratorConfig().validate()
    for gone in ("task", "social", "plan"):
        with pytest.raises(ValueError):
            OrchestratorConfig(variant=gone).validate()


# ── State: per-episode memory horizon ─────────────────────────────────────

def test_state_reset_clears_everything():
    st = OrchestratorState()
    st.directives = {"agent_0": {"task_id": "t1"}}
    st.add_event(oevents.death_event(41, "agent_2"))
    st.reset()
    assert st.directives == {}
    assert st.event_buffer == []


# ── Events ────────────────────────────────────────────────────────────────

def test_message_text_truncated_to_120_chars():
    ev = oevents.message_event(1, "agent_0", "agent_1", "x" * 500)
    assert len(ev["text"]) == oevents.MESSAGE_TEXT_MAX_CHARS


def test_event_shapes():
    assert oevents.milestone_event(3, "m8", ["agent_0"]) == {
        "type": "milestone", "t": 3, "id": "m8", "contributors": ["agent_0"]}
    assert oevents.death_event(4, "agent_1")["type"] == "death"
    assert oevents.chamber_change_event(5, "ch2")["chamber"] == "ch2"


# ── Response parsing (regressions from the first smoke run) ─────────────
# The backbone closes `ledger` one brace early, emitting stall_counter as a
# sibling and continuing past the outer close. This cost 14/27 attempts in
# runs/orchestrator_smoke seed_42 before parse_orchestrator_json existed.

# Verbatim shape from that run (t=120), abbreviated but structurally exact.
PREMATURE_CLOSE = (
    '{"ledger": {"task_facts": ["Agent 1 died at t=119."], '
    '"progress": {"current_stage_goal": "clear Ch5", "issued_at_step": 114, '
    '"expected_signal": "zombies defeated within 15 steps"}}, '
    '"stall_counter": 1}, '
    '"directives": {"agent_0": {"comm_target": "agent_2", "help": "Observe."}, '
    '"agent_1": {"comm_target": "agent_0", "help": "Advance."}, '
    '"agent_2": {"comm_target": "agent_0", "help": "Engage."}}, '
    '"changed": true, "why": "Agent 1 died."}'
)

# Shape from t=106: a second document with no joining comma.
TWO_DOCUMENTS = (
    '{"ledger": {"task_facts": ["f1"], "progress": null}, "stall_counter": 2}'
    '{"directives": {"agent_0": {"comm_target": "agent_1", "help": "h0"}, '
    '"agent_1": {"comm_target": "agent_0", "help": "h1"}, '
    '"agent_2": {"comm_target": "agent_0", "help": "h2"}}, "changed": false, '
    '"why": "unchanged"}'
)


def test_parse_wellformed_response_unchanged():
    raw = json.dumps(good_response())
    assert ocore.parse_orchestrator_json(raw) == good_response()


def test_parse_recovers_premature_outer_close():
    parsed = ocore.parse_orchestrator_json(PREMATURE_CLOSE)
    assert set(parsed) == {"ledger", "stall_counter", "directives",
                           "changed", "why"}
    assert parsed["stall_counter"] == 1
    assert list(parsed["directives"]) == ["agent_0", "agent_1", "agent_2"]


def test_parse_recovers_two_concatenated_documents():
    parsed = ocore.parse_orchestrator_json(TWO_DOCUMENTS)
    assert parsed["ledger"]["task_facts"] == ["f1"]
    assert list(parsed["directives"]) == ["agent_0", "agent_1", "agent_2"]
    assert parsed["changed"] is False


def test_parse_strips_fences_and_leading_prose():
    raw = "Here is my answer:\n```json\n" + json.dumps(good_response()) + "\n```"
    assert ocore.parse_orchestrator_json(raw) == good_response()


def test_parse_garbage_returns_empty():
    assert ocore.parse_orchestrator_json("not json at all") == {}
    assert ocore.parse_orchestrator_json("") == {}


def test_default_parser_falls_back_to_load_json_first():
    # Kept byte-for-byte from the reported runs: responses without a
    # top-level "ledger" key go through load_json before the tolerant parse.
    parse = ocore._default_parse_json()
    raw = json.dumps({"tasks": [{"id": "a"}], "why": "w"})
    assert parse(raw) == {"tasks": [{"id": "a"}], "why": "w"}


# ── Text world state ──────────────────────────────────────────────────────

def _env_state():
    return {
        "step": 40,
        "agents": {
            "agent_0": {"pos": (3.0, 12.0, 4.0), "chamber": "ch1",
                        "hp": 20.0, "alive": True},
            "agent_1": {"pos": (6.0, 11.0, 19.0), "chamber": "ch2",
                        "hp": 14.0, "alive": True},
            "agent_2": {"pos": None, "chamber": None, "hp": None,
                        "alive": False},
        },
        "doors": {"door1": True, "door2": False, "door3": False,
                  "door4": False},
        "anvils": [{"kind": "sword", "hp": 12}],
        "cell_doors_open": [1],
        "recent_messages": [("agent_0", "agent_1")],
    }


def test_render_map_text_fallback():
    text = omap.render_map_text(_env_state(), num_agents=3)
    assert "agent_0" in text and "ch2" in text
    assert "DOOR1=OPEN" in text and "DOOR2=closed" in text
    assert "sword" in text and "DEAD" in text
    assert "cell 1" in text


# ── Logging ───────────────────────────────────────────────────────────────

def test_task_compliance_log_roundtrip(tmp_path):
    logger = OrchestratorLogger(str(tmp_path))
    logger.log_task_compliance({
        "episode": 1, "t": 5, "agent": "agent_0",
        "active_note": "break anvil A", "old_task": None, "new_task": "dig",
    })
    rec = json.loads(open(logger.task_compliance_path, encoding="utf-8").read())
    assert rec["active_note"] == "break anvil A"
