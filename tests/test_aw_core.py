"""AgentWorld harness core: tasks, actions, verifier wrapper, trajectories."""

import json
import sys
from pathlib import Path

import pytest

from agentworld.actions import TOOLS, ToolCall, parse_action, safe_text
from agentworld.tasks import (anonymise_text, compose_world, list_suite, load_task,
                              task_definition_for)
from agentworld.trajectory import RunLog, TeamTrajectory, read_jsonl, status_line
from agentworld.verify import Verifier, extract_targets, progress_from_msg

FIX = Path(__file__).parent / "fixtures" / "agentworld" / "v1.3_benchmark"
TASK = FIX / "task_02_arrow_production.yaml"


def _obs(items, hp=69, x=390, y=3):
    return {"status": "success", "location": {"x": x, "y": y},
            "playerStatus": {"level": 1, "experience": 0, "hitPoints": hp, "maxHitPoints": 69,
                             "mana": 44, "maxMana": 44},
            "inventory": {"items": [{"key": k, "count": c} for k, c in items.items()]},
            "mobs": []}


# ── tasks ────────────────────────────────────────────────────────────────────
def test_load_task_matches_runner_specs():
    t = load_task(TASK)
    assert t.task_id == "task_02" and t.n_agents == 3 and t.rounds == 55
    a1, a2, a3 = t.agents
    assert [a.key for a in t.agents] == ["agent_1", "agent_2", "agent_3"]
    assert a1.username == "t02_lumberjack_agent" and a1.location == (388, 3)
    assert a2.inventory_items == ["flask:10", "apple:15", "feather:5"]
    assert a1.equipped_items == ["axe", "leatherarmor", "leatherboots"]
    assert a3.skill_levels["fletching"] == 25
    assert t.verifier_path().name == "task_02_success_criteria.py"
    assert "10 arrows" in t.objective


def test_list_suite_orders_by_number():
    assert [p.name for p in list_suite(FIX)] == ["task_02_arrow_production.yaml"]


def test_compose_world_prefixes_replicas_and_tracks_teams():
    t = load_task(TASK)
    single = compose_world([t])
    assert single.global_names[0] == "t02_lumberjack_agent"  # benchmark names kept
    w = compose_world([t] * 4)
    assert w.n_agents == 12 and len(set(w.global_names)) == 12
    assert w.global_names[3].startswith("t01_")
    same = w.true_team_matrix()
    assert same[0, 1] == 1 and same[0, 3] == 0 and same[0, 0] == 0
    assert w.team_of(5).team == "t01"
    d = task_definition_for(w.teams[2])
    assert d["agent_1"]["username"] == "t02_t02_lumberjack_agent"
    assert t.definition["agent_1"]["username"] == "t02_lumberjack_agent"  # not mutated


def test_anonymise_text():
    text = "Consolidate feathers with t02_fletcher_agent; t02_fletcher_agent crafts."
    out = anonymise_text(text, {"agent_3": "t02_fletcher_agent"}, {"agent_3": "the fletcher"})
    assert "t02_fletcher_agent" not in out and out.count("the fletcher") == 2


# ── actions ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,name,args", [
    ('craft_item(skill="Fletching", itemKey="arrow")', "craft_item",
     {"skill": "Fletching", "itemKey": "arrow"}),
    ("move_character(x=390, y='4')", "move_character", {"x": 390, "y": 4}),
    ("transfer_items('t02_fletcher_agent', 'logs', 3)", "transfer_items",
     {"targetPlayer": "t02_fletcher_agent", "itemKey": "logs", "count": 3}),
    ("`enter_portal`", "enter_portal", {}),
    ("attack_entity(targetInstance='1-2-3')\nI attack the rat", "attack_entity",
     {"targetInstance": "1-2-3"}),
    ({"name": "sleep", "args": {"seconds": 5}}, "sleep", {"seconds": 5}),
])
def test_parse_action_ok(text, name, args):
    call = parse_action(text)
    assert call.ok, call.error
    assert call.name == name and call.args == args


@pytest.mark.parametrize("text", [
    "", "dance()", "craft_item(skill='Fletching')", "move_character(x=__import__('os'))",
    "move_character(x='left', y=2)", "harvest_resource(**{'a': 1})", "just walk north",
])
def test_parse_action_rejects(text):
    call = parse_action(text)
    assert not call.ok and call.name in TOOLS


def test_parse_drops_unknown_kwargs_and_formats_like_runner():
    call = parse_action("craft_item(skill='Fletching', itemKey='stick', why='because')")
    assert call.ok and "why" not in call.args
    assert call.traj_string() == "craft_item(skill=Fletching, itemKey=stick)"


def test_safe_text_strips_braces():
    s = safe_text('{"hp": 3} {name}', limit=12)
    assert "{" not in s and "}" not in s and len(s) == 12
    ("x" + s).format()  # must not raise


# ── verifier ─────────────────────────────────────────────────────────────────
def _traj(final_arrows):
    t = load_task(TASK)
    tr = TeamTrajectory(compose_world([t]).teams[0])
    tr.begin_round(1)
    obs = _obs({"feather": 5})
    tr.add_action("agent_2", status_line(obs), "transfer_items(targetPlayer=x, itemKey=feather, count=5)", obs)
    tr.begin_round(2)
    obs = _obs({"arrow": final_arrows})
    tr.add_dm("agent_3", "t02_hunter_agent", "thanks", status_line(obs), obs)
    tr.add_action("agent_3", status_line(obs), "craft_item(skill=Fletching, itemKey=arrow)", obs)
    return tr


def test_verifier_scores_harness_trajectory():
    v = Verifier(TASK.parent / "task_02_success_criteria.py")
    assert v(_traj(10).data) == (1, "Arrows: 10/10")
    assert v(_traj(7).data) == (0, "Arrows: 7/10")
    assert "verifier_utils" not in sys.modules or sys.modules["verifier_utils"].__name__ != "aw_verifier_utils_0"


def test_verifier_crash_is_contained():
    v = Verifier(TASK.parent / "task_02_success_criteria.py")
    ok, msg = v({"rounds": [{"actions": [{"observation": {"inventory": {"items": [None]}}}]}]})
    assert ok == 0 and msg.startswith("verifier error")


def test_progress_from_msg():
    assert progress_from_msg("Arrows: 7/10") == pytest.approx(0.7)
    assert progress_from_msg("Logs 3/3, Arrows 15/10") == pytest.approx(1.0)
    assert progress_from_msg("Ring crafted") is None


def test_extract_targets_reads_main_verifier_only():
    t = extract_targets(TASK.parent / "task_02_success_criteria.py")
    assert t.items == {"arrow": 10}  # the v1 variant's 15 is ignored


# ── trajectory + run log ─────────────────────────────────────────────────────
def test_status_line_matches_verifier_parsing():
    s = status_line(_obs({}, hp=12))
    assert s.split("❤️")[1].split("|")[0].strip() == "12/69"
    assert status_line(None).startswith("📊 Status")


def test_trajectory_schema_and_comm_records(tmp_path):
    tr = _traj(10)
    path = tr.save(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"task_id", "task_source", "task_key", "timestamp",
                         "task_definition", "rounds", "metrics"}
    acts = data["rounds"][1]["actions"]
    assert acts[0]["action"] == "chat(message=@t02_hunter_agent: thanks)"
    assert set(acts[0]) == {"agent_name", "status", "action", "observation", "thinking"}
    assert acts[-1]["action"].startswith("craft_item(")


def test_runlog_writes_jsonl(tmp_path):
    import numpy as np
    log = RunLog(tmp_path / "ep1")
    log.message({"round": 1, "sender": "a", "receiver": "b", "text": "hi"})
    log.state({"round": 1, "W": np.eye(2)})
    log.close()
    assert read_jsonl(tmp_path / "ep1" / "messages.jsonl")[0]["text"] == "hi"
    assert read_jsonl(tmp_path / "ep1" / "replay" / "state.jsonl")[0]["W"] == [[1, 0], [0, 1]]
