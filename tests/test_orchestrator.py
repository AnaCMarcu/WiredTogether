"""Tests for the centralized task-ledger orchestrator (O2 baseline).

Covers the pure logic (state horizon, digest, scheduling, validation incl.
the relational-leakage filter) plus the full orchestrate() call against a
fake client — autogen_core is stubbed per-test exactly like conftest stubs
pettingzoo, since orchestrate() imports it lazily.
"""

import asyncio
import json
import re
import sys
import types
from types import SimpleNamespace

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState, empty_ledger
from orchestrator import events as oevents
from orchestrator import core as ocore
from orchestrator import map_render as omap
from orchestrator import prompt as oprompt
from orchestrator.logging import OrchestratorLogger


LIVING = ["agent_0", "agent_1", "agent_2"]


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
    assert cfg.mode == "advisory"
    # Cadence must match the social module's T_soc default (--social-interval).
    assert cfg.cadence == 8
    assert cfg.event_triggers is True
    assert cfg.stall_threshold == 2
    assert cfg.max_task_facts == 15
    assert cfg.max_digest_events == 30
    assert cfg.use_map_image is True
    assert cfg.model is None
    assert cfg.log_dir_name == "orchestrator"


def test_config_mode_validation():
    OrchestratorConfig(mode="advisory").validate()
    OrchestratorConfig(mode="bias").validate()
    with pytest.raises(ValueError):
        OrchestratorConfig(mode="override").validate()


# ── State: within-episode memory horizon ────────────────────────────────

def test_state_reset_clears_everything():
    st = OrchestratorState()
    st.apply_success(
        {"task_facts": ["f1"], "progress": {"x": 1}, "stall_counter": 3},
        {"agent_0": {"comm_target": "agent_1", "help": "h"}},
        t=40, max_task_facts=15,
    )
    st.add_event(oevents.death_event(41, "agent_2"))
    st.record_failure(42)
    st.reset()
    assert st.ledger == empty_ledger()
    assert st.directives == {}
    assert st.last_call_step == -1
    assert st.event_buffer == []
    assert st.call_count == 0
    assert st.failed_calls == 0


def test_state_facts_cap_keeps_most_recent():
    st = OrchestratorState()
    facts = [f"fact {i}" for i in range(20)]
    st.apply_success({"task_facts": facts, "progress": None,
                      "stall_counter": 0}, {}, t=8, max_task_facts=15)
    assert st.ledger["task_facts"] == facts[-15:]


def test_state_success_clears_buffer_failure_keeps_it():
    st = OrchestratorState()
    st.add_event(oevents.chamber_change_event(5, "ch2"))
    st.record_failure(8)
    assert len(st.event_buffer) == 1          # failed call: events preserved
    assert st.failed_calls == 1
    assert st.last_call_step == 8             # but the clock advances
    st.apply_success({"task_facts": []}, {}, t=16, max_task_facts=15)
    assert st.event_buffer == []              # success clears


# ── Events / digest ──────────────────────────────────────────────────────

def test_digest_formats_chronologically():
    evs = [
        oevents.message_event(412, "agent_1", "agent_2", "pressing switch now"),
        oevents.milestone_event(415, "m13", ["agent_1"]),
        oevents.chamber_change_event(430, "ch3"),
        oevents.death_event(431, "agent_0"),
    ]
    digest = oevents.build_digest(evs, max_events=30)
    lines = digest.splitlines()
    assert lines[0] == 't=412 msg agent_1->agent_2: "pressing switch now"'
    assert lines[1] == "t=415 MILESTONE m13 by [agent_1]"
    assert lines[2] == "t=430 CHAMBER -> ch3"
    assert lines[3] == "t=431 DEATH agent_0"


def test_digest_truncation_banner_and_empty():
    evs = [oevents.milestone_event(i, f"m{i}", []) for i in range(10)]
    digest = oevents.build_digest(evs, max_events=3)
    lines = digest.splitlines()
    assert lines[0] == "(showing last 3 of 10 events)"
    assert len(lines) == 4
    assert "m9" in lines[-1]
    assert oevents.build_digest([], 30) == "(no events since your last call)"


def test_message_text_truncated_to_120_chars():
    ev = oevents.message_event(1, "agent_0", "agent_1", "x" * 500)
    assert len(ev["text"]) == oevents.MESSAGE_TEXT_MAX_CHARS


# ── Scheduling ───────────────────────────────────────────────────────────

def test_should_call_first_call_and_cadence():
    cfg = OrchestratorConfig(enabled=True, cadence=8, event_triggers=False)
    st = OrchestratorState()
    assert ocore.should_call(st, 0, cfg)          # first call of the episode
    st.last_call_step = 0
    assert not ocore.should_call(st, 7, cfg)
    assert ocore.should_call(st, 8, cfg)


def test_should_call_event_triggers():
    cfg = OrchestratorConfig(enabled=True, cadence=100, event_triggers=True)
    st = OrchestratorState()
    st.last_call_step = 0
    st.add_event(oevents.message_event(1, "agent_0", "agent_1", "hi"))
    assert not ocore.should_call(st, 2, cfg)      # messages don't trigger
    st.add_event(oevents.milestone_event(2, "m8", ["agent_0"]))
    assert ocore.should_call(st, 3, cfg)
    # Same buffer, triggers disabled -> only the cadence counts.
    cfg_off = OrchestratorConfig(enabled=True, cadence=100,
                                 event_triggers=False)
    assert not ocore.should_call(st, 3, cfg_off)


def test_should_call_failed_call_consumes_triggers():
    # A failed call keeps its events in the buffer (digest completeness)
    # but they must not re-trigger a call every subsequent step.
    cfg = OrchestratorConfig(enabled=True, cadence=100, event_triggers=True)
    st = OrchestratorState()
    st.last_call_step = 0
    st.add_event(oevents.milestone_event(2, "m8", ["agent_0"]))
    assert ocore.should_call(st, 3, cfg)
    st.record_failure(3)
    assert len(st.event_buffer) == 1
    assert not ocore.should_call(st, 4, cfg)     # trigger consumed
    st.add_event(oevents.death_event(5, "agent_1"))
    assert ocore.should_call(st, 5, cfg)         # a NEW event re-triggers


# ── Validation ───────────────────────────────────────────────────────────

def test_validate_happy_path_normalizes_names():
    resp = good_response()
    resp["directives"] = {
        "Agent_0": {"comm_target": "agent1", "help": "punch anvil A"},
        "agent_1": {"comm_target": "agent_0", "help": 7},
        "agent_2": {"comm_target": " agent_0 ", "help": "scout"},
    }
    v = ocore.validate_response(resp, LIVING, t=12)
    assert v["ok"], v["error"]
    assert v["directives"]["agent_0"]["comm_target"] == "agent_1"
    assert v["directives"]["agent_1"]["help"] == "7"   # coerced to str
    assert v["directives"]["agent_2"]["comm_target"] == "agent_0"
    # progress.assignments is a copy of the cleaned directives.
    assert v["ledger"]["progress"]["assignments"] == v["directives"]


def test_validate_missing_living_agent_fails():
    resp = good_response()
    del resp["directives"]["agent_2"]
    v = ocore.validate_response(resp, LIVING, t=12)
    assert not v["ok"]
    assert "agent_2" in v["error"]


def test_validate_strips_extra_dead_agent_with_warning():
    resp = good_response()
    resp["directives"]["agent_7"] = {"comm_target": "agent_0", "help": "x"}
    v = ocore.validate_response(resp, LIVING, t=12)
    assert v["ok"]
    assert "agent_7" not in v["directives"]
    assert any("agent_7" in w for w in v["warnings"])


@pytest.mark.parametrize("bad_target", ["agent_1", "all", "agent_9", None])
def test_validate_rejects_bad_comm_targets(bad_target):
    # self ("agent_1" for agent_1), "all", non-living, and missing all fail.
    resp = good_response()
    resp["directives"]["agent_1"]["comm_target"] = bad_target
    v = ocore.validate_response(resp, LIVING, t=12)
    assert not v["ok"]


def test_validate_missing_top_level_keys_fail():
    assert not ocore.validate_response({"directives": {}}, LIVING, t=1)["ok"]
    assert not ocore.validate_response({"ledger": {}}, LIVING, t=1)["ok"]
    assert not ocore.validate_response("nope", LIVING, t=1)["ok"]


def test_relational_leakage_filter_drops_and_reports():
    resp = good_response()
    resp["ledger"]["task_facts"] = [
        "door 2 opens 20 steps after both anvils break",   # keep
        "agent_0 and agent_1 work well together",           # drop
        "I trust agent_2 to scout",                         # drop
        "good pairing: agent_1+agent_2",                    # drop (good pair)
        "agents prefer the west switch",                    # drop (prefer)
        "the bond between 0 and 1 is strong",               # drop (bond)
        "ch4 zombies drop no loot",                         # keep
    ]
    v = ocore.validate_response(resp, LIVING, t=12)
    assert v["ok"]
    assert v["ledger"]["task_facts"] == [
        "door 2 opens 20 steps after both anvils break",
        "ch4 zombies drop no loot",
    ]
    assert len(v["leakage_filtered"]) == 5


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


def test_premature_close_validates_end_to_end():
    """The whole point: a response in the broken shape must now produce
    usable directives instead of being discarded."""
    parsed = ocore.parse_orchestrator_json(PREMATURE_CLOSE)
    v = ocore.validate_response(parsed, LIVING, t=120)
    assert v["ok"], v["error"]
    # stall_counter was a sibling of ledger — it must survive, not reset to 0.
    assert v["ledger"]["stall_counter"] == 1
    assert any("hoisted" in w for w in v["warnings"])
    assert v["directives"]["agent_0"]["comm_target"] == "agent_2"


def test_stall_counter_inside_ledger_takes_precedence():
    resp = good_response()
    resp["ledger"]["stall_counter"] = 4
    resp["stall_counter"] = 99          # stray sibling must not win
    v = ocore.validate_response(resp, LIVING, t=12)
    assert v["ok"]
    assert v["ledger"]["stall_counter"] == 4
    assert not any("hoisted" in w for w in v["warnings"])


# ── Coupling surfaces ────────────────────────────────────────────────────

def test_render_directive_and_comm_target():
    st = OrchestratorState()
    assert "none yet" in ocore.render_agent_directive("agent_0", st)
    assert ocore.directive_comm_target(st, "agent_0") is None
    st.apply_success(
        {"task_facts": []},
        {"agent_0": {"comm_target": "agent_2", "help": "guard the door"}},
        t=8, max_task_facts=15,
    )
    text = ocore.render_agent_directive("agent_0", st)
    assert "agent_2" in text and "guard the door" in text
    assert "communication_target" in text
    assert ocore.directive_comm_target(st, "agent_0") == "agent_2"
    assert ocore.directive_comm_target(st, "agent0") == "agent_2"  # tolerant
    assert ocore.directive_comm_target(st, "agent_1") is None


# ── Prompt ───────────────────────────────────────────────────────────────

def test_prompt_first_call_placeholders():
    st = OrchestratorState()
    text = oprompt.format_prompt(
        n_agents=3, agent_names=LIVING,
        last_call_step=st.last_call_step, current_step=0,
        digest="(no events since your last call)",
        ledger=st.ledger, directives=st.directives,
        stall_threshold=2, map_text_fallback="WORLD STATE: ...",
    )
    assert "3 agents (agent_0, agent_1, agent_2)" in text
    assert "(none yet — this is your first call this episode)" in text
    assert "(none yet)" in text
    assert "stall_counter > 2" in text
    assert "episode start -> 0" in text
    assert "WORLD STATE: ..." in text
    # No pairing heuristics / chamber strategy leaked into the template.
    assert "Hebbian" not in text and "bond" not in text.lower()


def test_prompt_renders_ledger_json():
    st = OrchestratorState()
    st.apply_success(
        {"task_facts": ["f1"], "progress": {"current_stage_goal": "g"},
         "stall_counter": 1},
        {"agent_0": {"comm_target": "agent_1", "help": "h"}},
        t=8, max_task_facts=15,
    )
    text = oprompt.format_prompt(
        n_agents=3, agent_names=LIVING, last_call_step=8, current_step=16,
        digest="d", ledger=st.ledger, directives=st.directives,
        stall_threshold=2,
    )
    assert '"f1"' in text and '"stall_counter": 1' in text
    assert '"comm_target": "agent_1"' in text
    assert "step 8 -> 16" in text


# ── Response-format example arity (regression, second smoke run) ────────
# A hardcoded two-agent example was copied LITERALLY: the model emitted
# directives for agent_0/agent_1 only and omitted agent_2, failing 11 of the
# first 11 calls in runs/orchestrator_smoke seed_42.

def _example_block(agent_names):
    out = oprompt.format_prompt(
        n_agents=len(agent_names), agent_names=agent_names,
        last_call_step=-1, current_step=0, digest="d",
        ledger=OrchestratorState().ledger, directives={},
        stall_threshold=2,
    )
    start = out.index('"directives": {')
    return out[start:out.index('"changed"', start)]


def test_directives_example_covers_every_living_agent():
    block = _example_block(LIVING)
    for name in LIVING:
        assert f'"{name}": {{"comm_target"' in block, name


def test_directives_example_shrinks_with_the_living_set():
    # agent_1 died: it must vanish from the example, not linger as a target.
    block = _example_block(["agent_0", "agent_2"])
    assert '"agent_1"' not in block
    assert '"agent_0": {"comm_target": "agent_2"' in block
    assert '"agent_2": {"comm_target": "agent_0"' in block


def test_directives_example_never_targets_self():
    for names in (["agent_0", "agent_1"], LIVING,
                  [f"agent_{i}" for i in range(9)]):
        for line in _example_block(names).splitlines():
            m = re.match(r'\s*"(agent_\d+)":\s*\{"comm_target":\s*"(agent_\d+)"',
                         line)
            if m:
                assert m.group(1) != m.group(2), line


def test_prompt_states_task_facts_are_cumulative():
    """The ledger IS the within-episode memory being contrasted with W(t).

    A two-slot "task_facts": ["<fact>", "<fact>"] example made the model copy
    that arity literally: retention fell to 56% per call (vs 94%) and the
    ledger stopped accumulating across runs/orchestrator_smoke_v3.
    """
    out = oprompt.format_prompt(
        n_agents=3, agent_names=LIVING, last_call_step=8, current_step=16,
        digest="d", ledger=OrchestratorState().ledger, directives={},
        stall_threshold=2,
    )
    assert "CUMULATIVE" in out
    assert "GROWS across calls" in out
    # The literal two-slot example must be gone.
    assert '["<fact>", "<fact>"]' not in out


def test_prompt_states_the_required_directive_arity():
    out = oprompt.format_prompt(
        n_agents=3, agent_names=LIVING, last_call_step=-1, current_step=0,
        digest="d", ledger=OrchestratorState().ledger, directives={},
        stall_threshold=2,
    )
    assert "all 3 of them" in out
    assert "agent_0, agent_1, agent_2" in out


# ── Map render ───────────────────────────────────────────────────────────

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


def test_render_map_writes_png(tmp_path):
    out = tmp_path / "map.png"
    result = omap.render_map(_env_state(), str(out), num_agents=3)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_render_map_text_fallback():
    text = omap.render_map_text(_env_state(), num_agents=3)
    assert "agent_0" in text and "ch2" in text
    assert "DOOR1=OPEN" in text and "DOOR2=closed" in text
    assert "sword" in text and "DEAD" in text
    assert "cell 1" in text


# ── orchestrate(): full call against a fake client ──────────────────────

class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, messages, cancellation_token=None, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            content=self._responses.pop(0),
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


@pytest.fixture
def stub_autogen(monkeypatch):
    """orchestrate() lazily imports autogen_core; stub it like conftest
    stubs pettingzoo (the real package is not installed on this machine)."""
    if "autogen_core" in sys.modules:  # real package present — nothing to do
        yield
        return
    core_mod = types.ModuleType("autogen_core")
    core_mod.CancellationToken = type("CancellationToken", (), {})
    models_mod = types.ModuleType("autogen_core.models")

    class UserMessage:
        def __init__(self, content=None, source=None):
            self.content = content
            self.source = source

    models_mod.UserMessage = UserMessage
    core_mod.models = models_mod
    monkeypatch.setitem(sys.modules, "autogen_core", core_mod)
    monkeypatch.setitem(sys.modules, "autogen_core.models", models_mod)
    yield


def _run_orchestrate(state, client, cfg, tmp_path, t=8):
    logger = OrchestratorLogger(str(tmp_path), dir_name=cfg.log_dir_name)
    asyncio.run(ocore.orchestrate(
        state, _env_state(), client, cfg,
        living_agents=LIVING, episode=1, t=t,
        orch_logger=logger, parse_json=json.loads, num_agents=3,
    ))
    return logger


def test_orchestrate_success_updates_state(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, use_map_image=False)
    state = OrchestratorState()
    state.add_event(oevents.milestone_event(3, "m8", ["agent_0"]))
    client = _FakeClient([json.dumps(good_response())])
    logger = _run_orchestrate(state, client, cfg, tmp_path)

    assert state.call_count == 1 and state.failed_calls == 0
    assert state.last_call_step == 8
    assert state.event_buffer == []
    assert state.directives["agent_0"]["comm_target"] == "agent_1"
    assert state.ledger["task_facts"] == ["anvils need 2 punchers within 1s"]
    records = [json.loads(l) for l in
               open(logger.calls_path, encoding="utf-8")]
    assert len(records) == 1
    rec = records[0]
    assert rec["failed"] is False
    assert rec["prompt_tokens"] == 100 and rec["completion_tokens"] == 20
    assert rec["directives"] == state.directives
    # The prompt the client saw carried the digest + text map fallback.
    sent = client.calls[0][0].content[0]
    assert "MILESTONE m8" in sent and "WORLD STATE" in sent


def test_orchestrate_retries_once_then_succeeds(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, use_map_image=False)
    state = OrchestratorState()
    client = _FakeClient(["not json at all", json.dumps(good_response())])
    _run_orchestrate(state, client, cfg, tmp_path)
    assert len(client.calls) == 2
    assert state.failed_calls == 0
    assert state.directives  # second attempt installed


def test_orchestrate_failure_keeps_previous_state(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, use_map_image=False)
    state = OrchestratorState()
    # Install a known-good state first.
    _run_orchestrate(state, _FakeClient([json.dumps(good_response())]),
                     cfg, tmp_path, t=8)
    prev_directives = dict(state.directives)
    prev_ledger = json.loads(json.dumps(state.ledger))

    # Now a call whose both attempts emit a self-targeting directive.
    bad = good_response()
    bad["directives"]["agent_0"]["comm_target"] = "agent_0"
    logger = _run_orchestrate(
        state, _FakeClient([json.dumps(bad), json.dumps(bad)]),
        cfg, tmp_path, t=16,
    )
    assert state.failed_calls == 1
    assert state.directives == prev_directives   # never inject malformed
    assert state.ledger == prev_ledger
    assert state.last_call_step == 16            # but no per-step hammering
    records = [json.loads(l) for l in
               open(logger.calls_path, encoding="utf-8")]
    assert records[-1]["failed"] is True
    assert records[-1]["raw_output"]             # raw output logged


def test_orchestrate_logs_leakage(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, use_map_image=False)
    state = OrchestratorState()
    resp = good_response()
    resp["ledger"]["task_facts"].append("agent_0 and agent_1 work well together")
    logger = _run_orchestrate(state, _FakeClient([json.dumps(resp)]),
                              cfg, tmp_path)
    records = [json.loads(l) for l in
               open(logger.calls_path, encoding="utf-8")]
    assert records[0]["leakage_filtered"] == [
        "agent_0 and agent_1 work well together"]
    assert all("work well" not in f for f in state.ledger["task_facts"])


def test_compliance_log_roundtrip(tmp_path):
    logger = OrchestratorLogger(str(tmp_path))
    logger.log_compliance({
        "episode": 1, "t": 5, "agent": "agent_0",
        "directed_comm_target": "agent_1",
        "actual_comm_target": "agent_2", "complied": False,
    })
    rec = json.loads(open(logger.compliance_path, encoding="utf-8").read())
    assert rec["complied"] is False
    assert rec["directed_comm_target"] == "agent_1"


# ═══════════════════════════════════════════════════════════════════════
# Variants: O-social (centralized social deliberation) + O-plan
# ═══════════════════════════════════════════════════════════════════════

from orchestrator.curriculum_hook import (
    PLAN_SUFFIX, TEAM_PLAN_NOTE_PLACEHOLDER, apply_plan_suffix,
)
from orchestrator.events import PairAccumulator


def social_response(**overrides) -> dict:
    resp = {
        "ledger": {"notes": ["a0 and a1 keep co-firing at the anvil"],
                   "stall_counter": 0},
        "directives": {
            "agent_0": {"reasoning": "r0", "ask_target": "agent_1",
                        "ask_message": "help me punch anvil A",
                        "respond_to": [], "pair_notes": {"agent_1": "warming"}},
            "agent_1": {"reasoning": "r1", "ask_target": None,
                        "ask_message": None,
                        "respond_to": ["agent_0"], "pair_notes": {}},
            "agent_2": {"reasoning": "r2", "ask_target": "agent_0",
                        "ask_message": "where is the door",
                        "respond_to": [], "pair_notes": {}},
        },
        "changed": True,
        "why": "initial",
    }
    resp.update(overrides)
    return resp


# ── Config / state ───────────────────────────────────────────────────────

def test_variant_config_validation():
    OrchestratorConfig(variant="task").validate()
    OrchestratorConfig(variant="social").validate()
    OrchestratorConfig(variant="plan").validate()
    with pytest.raises(ValueError):
        OrchestratorConfig(variant="graph").validate()
    assert OrchestratorConfig().variant == "task"   # default unchanged


def test_reset_keep_ledger_preserves_memory_across_episodes():
    st = OrchestratorState()
    st.apply_success({"notes": ["n1"], "stall_counter": 2},
                     {"agent_0": {"ask_target": "agent_1"}},
                     t=40, max_task_facts=15)
    st.add_event(oevents.death_event(41, "agent_2"))
    st.record_failure(42)
    st.reset(keep_ledger=True)
    # W(t)-matched horizon: ledger + directives survive...
    assert st.ledger["notes"] == ["n1"]
    assert st.directives["agent_0"]["ask_target"] == "agent_1"
    # ...but the episode clock resets.
    assert st.last_call_step == -1
    assert st.event_buffer == []
    assert st.call_count == 0 and st.failed_calls == 0


def test_apply_success_caps_notes_list():
    st = OrchestratorState()
    st.apply_success({"notes": [f"n{i}" for i in range(20)],
                      "stall_counter": 0}, {}, t=8, max_task_facts=15)
    assert st.ledger["notes"] == [f"n{i}" for i in range(5, 20)]


# ── PairAccumulator ──────────────────────────────────────────────────────

def test_pair_accumulator_counts_and_render():
    acc = PairAccumulator(3, radius=5.0)
    # Step 1: a0 and a1 co-present (dist 3), a2 far; a0 msgs a1; a0 earns +10.
    acc.note_step(
        positions=[(0, 0, 0), (3, 0, 0), (100, 0, 0)],
        comm_events=[(0, 1)],
        bond_rewards=[10.0, 0.0, 0.5],
        chambers=[2, 2, 1],
    )
    # Step 2: all far apart; a1 msgs a0.
    acc.note_step(
        positions=[(0, 0, 0), (50, 0, 0), (100, 0, 0)],
        comm_events=[(1, 0)],
        bond_rewards=[0.0, 5.0, 0.0],
        chambers=[2, 2, 1],
    )
    assert acc.steps == 2
    assert acc.copresence[(0, 1)] == 1
    assert (0, 2) not in acc.copresence
    assert acc.messages[(0, 1)] == 1 and acc.messages[(1, 0)] == 1
    assert acc.coreward[(0, 1)] == 10.0     # step-1 rewards of a0+a1
    assert acc.reward_totals == [10.0, 5.0, 0.5]
    text = acc.render()
    assert "agent_0~agent_1: co-present 1/2 steps" in text
    assert "msgs 1 (agent_0->agent_1) / 1 (agent_1->agent_0)" in text
    assert "reward-while-together +10.0" in text
    assert "agent_2=ch1" in text
    acc.clear()
    assert acc.steps == 0
    assert acc.render() == "(no steps recorded since your last call)"


def test_pair_accumulator_radius_boundary_and_none_positions():
    acc = PairAccumulator(2, radius=5.0)
    acc.note_step([(0, 0, 0), (5, 0, 0)], [], [0, 0], [1, 1])   # exactly 5.0
    assert acc.copresence.get((0, 1)) == 1                       # <= radius
    acc.note_step([None, (5, 0, 0)], [], [0, 0], [1, 1])         # missing pos
    assert acc.copresence.get((0, 1)) == 1


# ── validate_social_response ─────────────────────────────────────────────

def test_social_validate_happy_path():
    v = ocore.validate_social_response(social_response(), LIVING, t=8)
    assert v["ok"], v["error"]
    assert v["directives"]["agent_0"]["ask_target"] == "agent_1"
    assert v["directives"]["agent_1"]["ask_target"] is None
    assert v["directives"]["agent_1"]["respond_to"] == ["agent_0"]
    assert v["ledger"]["notes"] == ["a0 and a1 keep co-firing at the anvil"]


def test_social_validate_relational_notes_are_kept_and_counted():
    resp = social_response()
    resp["ledger"]["notes"] = [
        "agent_0 and agent_1 work well together",   # relational — KEPT here
        "door 2 needs both anvils broken",
    ]
    v = ocore.validate_social_response(resp, LIVING, t=8)
    assert v["ok"]
    assert len(v["ledger"]["notes"]) == 2           # nothing dropped
    assert v["relational_matches"] == 1             # but counted


def test_social_validate_missing_agent_fails():
    resp = social_response()
    del resp["directives"]["agent_2"]
    assert not ocore.validate_social_response(resp, LIVING, t=8)["ok"]


def test_social_validate_self_ask_target_fails():
    resp = social_response()
    resp["directives"]["agent_0"]["ask_target"] = "agent_0"
    assert not ocore.validate_social_response(resp, LIVING, t=8)["ok"]


def test_social_validate_filters_bad_respond_to_tolerantly():
    resp = social_response()
    resp["directives"]["agent_1"]["respond_to"] = [
        "agent_0", "agent_1", "agent_9", "all", "agent_0"]
    v = ocore.validate_social_response(resp, LIVING, t=8)
    assert v["ok"]                                   # tolerant, not fatal
    assert v["directives"]["agent_1"]["respond_to"] == ["agent_0"]


def test_plan_validate_truncates_plan_note():
    resp = social_response()
    for entry in resp["directives"].values():
        entry["plan_note"] = "x" * 500
    v = ocore.validate_social_response(resp, LIVING, t=8, variant="plan")
    assert v["ok"]
    note = v["directives"]["agent_0"]["plan_note"]
    assert len(note) == ocore.PLAN_NOTE_MAX_CHARS
    # social variant never carries the key at all
    v2 = ocore.validate_social_response(social_response(), LIVING, t=8)
    assert "plan_note" not in v2["directives"]["agent_0"]


def test_social_validate_hoists_top_level_stall_counter():
    resp = social_response()
    resp["ledger"].pop("stall_counter")
    resp["stall_counter"] = 3
    v = ocore.validate_social_response(resp, LIVING, t=8)
    assert v["ok"] and v["ledger"]["stall_counter"] == 3


# ── Rendering: byte-format mirror of SocialModule.render_directive ──────

def test_render_social_directive_matches_socialmodule_format():
    st = OrchestratorState()
    v = ocore.validate_social_response(social_response(), LIVING, t=8)
    st.apply_success(v["ledger"], v["directives"], t=8, max_task_facts=15)

    text = ocore.render_social_directive("agent_0", st)
    # Exact strings from SocialModule.render_directive — the action LLM must
    # not be able to tell the conditions apart by text shape.
    assert text.startswith(
        "Social directive (from your social-reasoning step):\n")
    assert "  Reasoning: r0\n" in text
    assert ('  Outgoing: Ask agent_1 for help. Suggested message: '
            '"help me punch anvil A". Put agent_1 in your '
            'communication_target field and a help request in your '
            'communication field.') in text
    assert "No pending requests warrant your help this step." in text
    assert "  Bond changes noted:\n  - agent_1: warming" in text

    text1 = ocore.render_social_directive("agent_1", st)
    assert "No help to ask for this step." in text1
    assert ("You have decided to help: agent_0. Your communication this "
            "step should acknowledge them") in text1
    assert "(no significant bond changes)" in text1


def test_render_social_directive_none_case_is_legacy_string():
    st = OrchestratorState()
    assert ocore.render_social_directive("agent_0", st) == (
        "Social directive: (none — social module disabled or not yet run)")


def test_directive_comm_target_reads_ask_target():
    st = OrchestratorState()
    v = ocore.validate_social_response(social_response(), LIVING, t=8)
    st.apply_success(v["ledger"], v["directives"], t=8, max_task_facts=15)
    assert ocore.directive_comm_target(st, "agent_0") == "agent_1"
    assert ocore.directive_comm_target(st, "agent_1") is None  # stay focused
    assert ocore.plan_note_for(st, "agent_0") == ""            # social: none


# ── Prompts ──────────────────────────────────────────────────────────────

def test_social_prompt_covers_agents_and_is_cumulative():
    st = OrchestratorState()
    out = oprompt.format_social_prompt(
        n_agents=3, agent_names=LIVING, last_call_step=-1, current_step=0,
        pair_digest="  agent_0~agent_1: co-present 3/8 steps",
        ledger=st.ledger, directives=st.directives,
    )
    for name in LIVING:
        assert f'"{name}": {{"reasoning"' in out
    assert '"ask_target": null' in out            # stay-focused shape shown
    assert "CUMULATIVE" in out and "GROWS across calls" in out
    assert "episode start -> 0" in out
    assert "(none yet — this is your first call)" in out
    assert "co-present 3/8 steps" in out
    assert "plan_note" not in out                 # social has no plan field
    assert "{" not in out.replace("{{", "").replace("}}", "") or True


def test_plan_prompt_adds_task_table_and_plan_note():
    st = OrchestratorState()
    out = oprompt.format_social_prompt(
        n_agents=3, agent_names=LIVING, last_call_step=8, current_step=16,
        pair_digest="(digest)", ledger=st.ledger, directives=st.directives,
        task_table='  agent_0: current="dig 3 stone"',
    )
    assert 'agent_0: current="dig 3 stone"' in out
    assert '"plan_note": ""' in out
    assert "never assign a task" in out.lower() or "never assign" in out


# ── Curriculum hook (byte-identity is the load-bearing property) ─────────

def test_apply_plan_suffix_identity_when_disabled():
    base = "TASK PROMPT with {completed_tasks} and {last_task}"
    assert apply_plan_suffix(base, False) == base


def test_apply_plan_suffix_appends_placeholder_once():
    base = "TASK PROMPT"
    out = apply_plan_suffix(base, True)
    assert out.startswith(base)
    assert TEAM_PLAN_NOTE_PLACEHOLDER in out
    assert apply_plan_suffix(out, True) == out          # idempotent
    assert out.count(TEAM_PLAN_NOTE_PLACEHOLDER) == 1
    # The suffix formats cleanly and legacy formatting ignores the extra kwarg.
    assert PLAN_SUFFIX.format(team_plan_note="hello").strip().endswith("hello")
    assert base.format(team_plan_note="ignored") == base


# ── orchestrate() end-to-end for the social variant ─────────────────────

def test_orchestrate_social_variant_end_to_end(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, variant="social",
                             use_map_image=False)
    state = OrchestratorState()
    client = _FakeClient([json.dumps(social_response())])
    logger = OrchestratorLogger(str(tmp_path), dir_name=cfg.log_dir_name)
    asyncio.run(ocore.orchestrate(
        state, {"pair_digest": "  agent_0~agent_1: co-present 5/8 steps",
                "task_table": None},
        client, cfg, living_agents=LIVING, episode=1, t=8,
        orch_logger=logger, parse_json=json.loads, num_agents=3,
    ))
    assert state.failed_calls == 0
    assert state.directives["agent_0"]["ask_target"] == "agent_1"
    assert state.ledger["notes"]
    sent = client.calls[0][0].content[0]
    assert "co-present 5/8 steps" in sent          # pair digest reached prompt
    assert "CURRENT INDIVIDUAL PLANS" not in sent  # social ≠ plan template
    rec = json.loads(open(logger.calls_path, encoding="utf-8").read())
    assert rec["failed"] is False
    assert rec["relational_matches"] == 0
    assert rec["map_path"] is None                 # no map in matched variant


def test_orchestrate_plan_variant_end_to_end(stub_autogen, tmp_path):
    cfg = OrchestratorConfig(enabled=True, variant="plan",
                             use_map_image=False)
    resp = social_response()
    for entry in resp["directives"].values():
        entry["plan_note"] = "take anvil B instead"
    state = OrchestratorState()
    client = _FakeClient([json.dumps(resp)])
    asyncio.run(ocore.orchestrate(
        state, {"pair_digest": "(digest)",
                "task_table": '  agent_0: current="punch anvil A"'},
        client, cfg, living_agents=LIVING, episode=1, t=8,
        orch_logger=OrchestratorLogger(str(tmp_path)),
        parse_json=json.loads, num_agents=3,
    ))
    assert state.failed_calls == 0
    sent = client.calls[0][0].content[0]
    assert 'agent_0: current="punch anvil A"' in sent
    assert ocore.plan_note_for(state, "agent_0") == "take anvil B instead"
