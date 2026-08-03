"""Unit tests for analysis_qualitative log parsing + alignment."""

import json

import pytest

from qual_fixtures import build_synth_run, LogWriter, action_prefix

from qual_lib import registry
from qual_lib import episode_io, log_parser, turns


@pytest.fixture()
def synth(tmp_path):
    run_dir, expected = build_synth_run(tmp_path)
    ref = registry._scan_run_dir("exp99_test", run_dir)
    return ref, expected


# ── log_parser ───────────────────────────────────────────────────────────


def test_action_units_count_and_retry_folding(synth):
    ref, exp = synth
    units = log_parser.iter_llm_units(ref.llm_logs["action_selection"])
    # one unit per (agent, step): retries fold into their retry-0 unit
    assert len(units) == exp["n_a0"] + exp["n_a1"]
    retried = [u for u in units if u["n_retries"] > 0]
    assert len(retried) == 1
    u = retried[0]
    assert u["agent"] == "agent_0"
    assert u["n_retries"] == 1 and u["n_errors"] == 1
    assert u["parse_ok"] is True  # last Response (pretty multi-line) won
    assert u["response"]["communication"] == "msg a0 t2"


def test_multiline_pretty_json_and_unicode(synth):
    ref, _ = synth
    units = log_parser.iter_llm_units(ref.llm_logs["critic"])
    assert units and all(u["parse_ok"] for u in units)
    action_units = log_parser.iter_llm_units(ref.llm_logs["action_selection"])
    # replacement char in thoughts survives parsing
    assert any("�" in (u["response"] or {}).get("thoughts", "")
               for u in action_units)


def test_prefix_classification():
    mod, meth, agent = log_parser._classify_prefix("Agent agent_2 on_messages: ")
    assert (mod, meth, agent) == ("action_selection", "on_messages", "agent_2")
    mod, meth, agent = log_parser._classify_prefix("Agent agent_0 rl_thoughts: ")
    assert (mod, meth, agent) == ("rl_thoughts", "rl_thoughts", "agent_0")
    mod, meth, agent = log_parser._classify_prefix("SocialModule[agent_1]: ")
    assert mod == "social_module" and agent == "agent_1"
    mod, meth, agent = log_parser._classify_prefix(
        "Belief System create_perception_beliefs: ")
    assert mod == "belief_system" and meth == "create_perception_beliefs"
    assert agent is None


def test_response_without_call_header(tmp_path):
    lw = LogWriter()
    lw.response("orphan.log", action_prefix(0), {"action": "Dig"})
    lw.dump(tmp_path)
    units = log_parser.iter_llm_units(tmp_path / "orphan.log")
    assert len(units) == 1 and units[0]["parse_ok"]


# ── turns / alignment ────────────────────────────────────────────────────


def test_ordinal_alignment_with_dying_agent(synth):
    ref, exp = synth
    units = log_parser.parse_run_llm_logs(ref)
    pa_units = turns.action_units_per_agent(units)
    pa_rows, _ = turns.build_step_index(ref)
    assert len(pa_rows[0]) == exp["n_a0"]
    assert len(pa_rows[1]) == exp["n_a1"]  # died early -> fewer rows
    align = turns.align_actions(pa_units, pa_rows)
    for a in (0, 1):
        assert align[a]["status"] == "exact"
        assert align[a]["match_rate"] == 1.0


def test_offset_repair_on_dropped_first_unit(synth):
    ref, _ = synth
    units = log_parser.parse_run_llm_logs(ref)
    pa_units = turns.action_units_per_agent(units)
    pa_rows, _ = turns.build_step_index(ref)
    pa_units[0] = pa_units[0][1:]  # simulate a lost first action unit
    align = turns.align_actions(pa_units, pa_rows)
    assert align[0]["offset"] == 1
    assert align[0]["status"] == "repaired"
    assert align[0]["match_rate"] == 1.0


def test_anonymous_attribution_and_cadence(synth):
    ref, _ = synth
    units = log_parser.parse_run_llm_logs(ref)
    pa_units = turns.action_units_per_agent(units)
    stats = turns.attribute_anonymous(units, pa_units)
    turns.social_units_turns(units, pa_units)
    assert stats["unattributed"] == 0
    cad = turns.cadence_report(units, episode_io.module_intervals(ref))
    assert cad["belief_perception"]["rate"] == 1.0
    assert cad["critic"]["rate"] == 1.0
    assert cad["social"]["rate"] == 1.0


def test_timeline_joins(synth):
    ref, exp = synth
    units = log_parser.parse_run_llm_logs(ref)
    pa_units = turns.action_units_per_agent(units)
    pa_rows, _ = turns.build_step_index(ref)
    align = turns.align_actions(pa_units, pa_rows)
    turns.attribute_anonymous(units, pa_units)
    turns.social_units_turns(units, pa_units)
    mapping = turns.map_units_to_rows(pa_units, pa_rows, align)
    assert turns.mark_stale(units, pa_units, mapping, align) == 0
    rows = turns.build_timeline(ref, units, pa_units, pa_rows, align, mapping)
    assert len(rows) == exp["n_a0"] + exp["n_a1"]
    r0 = [r for r in rows if r["agent"] == 0 and r["t"] == 0][0]
    assert r0["msg"]["text"] == "msg a0 t0"
    assert r0["thoughts"].startswith("think a0 t0")
    assert r0["task"] == "task_0_v1"          # curriculum fired at turn 1
    r_last = [r for r in rows if r["agent"] == 0][-1]
    assert r_last["task"] == "task_0_v1"      # forward-filled
    ev_rows = [r for r in rows if r["t"] == 3 and r["events"]]
    assert ev_rows and ev_rows[0]["events"][0]["id"] == "m2_dig_3_any"
    social_rows = [r for r in rows if r["social"]]
    assert social_rows, "social thoughts should attach to their turns"


# ── episode_io regression (the coop_eval nesting bug) ───────────────────


def test_read_cooperation_metrics_nested_and_flat(tmp_path):
    from qual_fixtures import write_episode
    coop = {"pair_interaction": {"messages": [[0, 1], [1, 0]]},
            "hebbian_W": [[0.0, 0.5], [0.4, 0.0]]}
    ep_n = tmp_path / "nested" / "ep_0001"
    write_episode(ep_n, [], [], [], coop, nested=True)
    got = episode_io.read_cooperation_metrics(ep_n)
    assert got.get("pair_interaction", {}).get("messages") == [[0, 1], [1, 0]]
    ep_f = tmp_path / "flat" / "ep_0001"
    write_episode(ep_f, [], [], [], coop, nested=False)
    got = episode_io.read_cooperation_metrics(ep_f)
    assert got.get("hebbian_W") == [[0.0, 0.5], [0.4, 0.0]]
