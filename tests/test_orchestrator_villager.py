"""Tests for the VillagerAgent-style orchestrator variant (villager).

Same conventions as tests/test_orchestrator.py: light imports only, autogen
stubbed per-test, fake clients, and every heavy runtime source (environment,
metric, craftium_metric, chamber_facts) replaced by injected overrides so the
controller's deterministic logic is exercised in isolation.
"""

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState
from orchestrator import events as oevents
from orchestrator import prompt as oprompt
from orchestrator import villager as ovillager
from orchestrator.curriculum_hook import (
    ASSIGNED_OBJECTIVE_PLACEHOLDER, PLAN_SUFFIX, VILLAGER_SUFFIX,
    apply_plan_suffix, apply_villager_suffix,
)
from orchestrator.dag import CentralTask, TaskDAG, ingest_decomposition
from orchestrator.logging import OrchestratorLogger

LIVING = ["agent_0", "agent_1", "agent_2"]
KNOWN = {"m1_move_5", "m4_dig_5_wood", "m8_anvil_A1", "m9_anvil_B1",
         "m17_switch_pressed"}
FAKE_TRACK = {m: "ch1_solo" for m in KNOWN}
FAKE_TRACKS = {"ch1_solo": [(m, 10.0) for m in sorted(KNOWN)],
               "ch2_anvils": [], "ch3_switches": [], "ch4_combat": [],
               "ch5_boss": []}


def _cfg(**overrides):
    kw = dict(enabled=True, variant="villager", node_timeout_steps=20,
              decompose_min_interval=8)
    kw.update(overrides)
    return OrchestratorConfig(**kw)


def _task(tid, milestones=("m1_move_5",), required=(), candidates=(),
          min_agents=1, status="open"):
    return CentralTask(id=tid, description=f"desc {tid}",
                       milestones=list(milestones), required=list(required),
                       candidates=list(candidates), min_agents=min_agents,
                       status=status)


def decompose_response(tasks=None):
    if tasks is None:
        tasks = [
            {"id": "explore", "description": "move around the room",
             "milestones": ["m1_move_5"], "required": [],
             "candidates": [], "min_agents": 1},
            {"id": "chop", "description": "dig 5 wood",
             "milestones": ["m4_dig_5_wood"], "required": ["explore"],
             "candidates": ["agent_0", "agent_1"], "min_agents": 1},
        ]
    return {"tasks": tasks, "why": "initial plan"}


def allocate_response(entries=None):
    if entries is None:
        entries = [{"agent": "agent_0", "task_id": "explore",
                    "role": "walk"},
                   {"agent": "agent_1", "task_id": "explore",
                    "role": "walk too"}]
    return {"assignments": entries, "why": "closest agents"}


def _env_state():
    return {"step": 0, "agents": {
        n: {"pos": (0, 0, 0), "chamber": "ch1", "hp": 20.0, "alive": True}
        for n in LIVING}}


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, messages, cancellation_token=None, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            content=self._responses.pop(0),
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
        )


@pytest.fixture
def stub_autogen(monkeypatch):
    if "autogen_core" in sys.modules:
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


def _controller(decompose_raws=(), allocate_raws=(), tmp_path=None,
                cfg=None, num_agents=3):
    logger = (OrchestratorLogger(str(tmp_path)) if tmp_path is not None
              else None)
    return ovillager.VillagerController(
        cfg or _cfg(), num_agents,
        decompose_client=_FakeClient(list(decompose_raws)),
        allocate_client=_FakeClient(list(allocate_raws)),
        orch_logger=logger,
        milestone_track=FAKE_TRACK, tracks=FAKE_TRACKS,
        chamber_describe=lambda ch, n: f"{ch}: facts",
    ), logger


def _run_tick(ctrl, state, living=LIVING, t=0, **kw):
    return asyncio.run(ctrl.tick(
        state=state, living_agents=living, episode=1, t=t,
        env_state=_env_state(), task_table="  agent_0: current=\"x\"",
        agent_milestones={}, parse_json=json.loads, **kw))


# ── Config ───────────────────────────────────────────────────────────────

def test_config_villager_defaults_and_validation():
    cfg = OrchestratorConfig()
    assert cfg.node_timeout_steps == 60
    assert cfg.max_open_tasks == 0
    assert cfg.decompose_min_interval == 8
    _cfg().validate()                                    # advisory OK
    with pytest.raises(ValueError):
        _cfg(mode="bias").validate()                     # bias invalid
    with pytest.raises(ValueError):
        _cfg(node_timeout_steps=0).validate()
    with pytest.raises(ValueError):
        _cfg(decompose_min_interval=0).validate()


# ── DAG ──────────────────────────────────────────────────────────────────

def test_dag_ready_detection():
    dag = TaskDAG()
    assert dag.add_task(_task("a"))[0]
    assert dag.add_task(_task("b", required=["a"]))[0]
    assert [t.id for t in dag.ready_tasks()] == ["a"]
    dag.assign("a", ["agent_0"], {}, t=0)
    assert dag.ready_tasks() == []                       # running blocks b
    dag.mark_success("a", t=1)
    assert [t.id for t in dag.ready_tasks()] == ["b"]
    dag.mark_failure("b", "timeout", t=2)
    assert dag.is_exhausted()


def test_dag_cycle_rejected():
    dag = TaskDAG()
    ok, reason = dag.add_task(_task("a", required=["a"]))   # self-edge
    assert not ok and reason == "cycle"
    assert dag.add_task(_task("a", required=["b"]))[0]      # b unknown: ok
    ok, reason = dag.add_task(_task("b", required=["a"]))   # closes 2-cycle
    assert not ok and reason == "cycle"
    ok, reason = dag.add_task(_task("a"))
    assert not ok and reason == "duplicate id"


def test_failure_cascades_predecessor_failed():
    dag = TaskDAG()
    dag.add_task(_task("a"))
    dag.add_task(_task("b", required=["a"]))
    dag.add_task(_task("c", required=["b"]))
    dag.add_task(_task("d"))                              # independent
    dag.assign("a", ["agent_0"], {}, t=0)
    freed = dag.mark_failure("a", "timeout", t=5)
    assert freed == ["agent_0"]
    assert dag.get("b").status == "failure"
    assert dag.get("b").failure_reason == "predecessor_failed"
    assert dag.get("c").status == "failure"               # transitive
    assert dag.get("d").status == "open"                  # untouched


# ── Ingestion ────────────────────────────────────────────────────────────

def _ingest(dag, parsed, **overrides):
    kw = dict(t=0, known_milestones=KNOWN, team_completed=set(),
              living_agents=LIVING, max_open=6)
    kw.update(overrides)
    return ingest_decomposition(dag, parsed, **kw)


def test_ingest_happy_path_and_intra_batch_required():
    dag = TaskDAG()
    r = _ingest(dag, decompose_response())
    assert r["ok"] and r["accepted"] == ["explore", "chop"]
    assert dag.get("chop").required == ["explore"]
    assert dag.get("chop").candidates == ["agent_0", "agent_1"]


def test_ingest_unknown_predecessor_edge_dropped():
    dag = TaskDAG()
    r = _ingest(dag, decompose_response([
        {"id": "t1", "description": "d", "milestones": ["m1_move_5"],
         "required": ["ghost_task"], "candidates": [], "min_agents": 1}]))
    assert r["ok"] and r["accepted"] == ["t1"]
    assert dag.get("t1").required == []
    assert any("ghost_task" in w for w in r["warnings"])


def test_ingest_requires_verifiable_milestone():
    dag = TaskDAG()
    r = _ingest(dag, decompose_response([
        {"id": "bad", "description": "d", "milestones": ["m_fake"],
         "required": [], "candidates": [], "min_agents": 1},
        {"id": "mixed", "description": "d",
         "milestones": ["m_fake", "m8_anvil_A1"],
         "required": [], "candidates": [], "min_agents": 1}]))
    assert r["ok"]
    assert ("bad", "no verifiable milestone") in r["rejected"]
    assert dag.get("mixed").milestones == ["m8_anvil_A1"]


def test_ingest_rejects_already_satisfied():
    dag = TaskDAG()
    r = _ingest(dag, decompose_response([
        {"id": "done", "description": "d", "milestones": ["m1_move_5"],
         "required": [], "candidates": [], "min_agents": 1}]),
        team_completed={"m1_move_5"})
    assert ("done", "already satisfied") in r["rejected"]


def test_ingest_empty_or_missing_tasks_fails():
    dag = TaskDAG()
    assert not _ingest(dag, {})["ok"]
    assert not _ingest(dag, {"tasks": []})["ok"]
    assert not _ingest(dag, "garbage")["ok"]
    assert len(dag.tasks) == 0                            # never mutated


def test_ingest_candidate_normalization_cap_and_clamp():
    dag = TaskDAG()
    tasks = [{"id": f"t{i}", "description": "d",
              "milestones": ["m1_move_5"], "required": [],
              "candidates": ["Agent_2", "agent9", "nonsense"],
              "min_agents": 99} for i in range(8)]
    r = _ingest(dag, decompose_response(tasks), max_open=3)
    assert r["ok"] and len(r["accepted"]) == 3            # cap enforced
    t0 = dag.get("t0")
    assert t0.candidates == ["agent_2"]                   # normalized, filtered
    assert t0.min_agents == 3                             # clamped to N
    assert any("cap" in w for w in r["warnings"])


# ── Allocation validation ────────────────────────────────────────────────

def test_allocation_drops_busy_and_double_booked():
    dag = TaskDAG()
    dag.add_task(_task("t1"))
    dag.add_task(_task("t2", candidates=["agent_1"]))
    v = ovillager.validate_allocation(
        allocate_response([
            {"agent": "agent_0", "task_id": "t1", "role": "r"},
            {"agent": "agent_0", "task_id": "t2", "role": "r"},   # dup
            {"agent": "agent_2", "task_id": "t2", "role": "r"},   # not cand
            {"agent": "agent_1", "task_id": "ghost", "role": "r"},
            {"agent": "agent_9", "task_id": "t1", "role": "r"},   # unknown
        ]),
        dag=dag, ready_ids={"t1", "t2"}, free_agents=LIVING,
        living_agents=LIVING)
    assert v["ok"]
    assert set(v["assignments"]) == {"t1"}
    assert v["assignments"]["t1"]["agents"] == ["agent_0"]
    assert len(v["warnings"]) == 4


def test_allocation_min_agents_group_rule():
    dag = TaskDAG()
    dag.add_task(_task("pair", min_agents=2))
    under = ovillager.validate_allocation(
        allocate_response([{"agent": "agent_0", "task_id": "pair",
                            "role": "left"}]),
        dag=dag, ready_ids={"pair"}, free_agents=LIVING,
        living_agents=LIVING)
    assert not under["ok"]                               # nothing survived
    full = ovillager.validate_allocation(
        allocate_response([
            {"agent": "agent_0", "task_id": "pair", "role": "left"},
            {"agent": "agent_1", "task_id": "pair", "role": "right"}]),
        dag=dag, ready_ids={"pair"}, free_agents=LIVING,
        living_agents=LIVING)
    assert full["ok"]
    assert full["assignments"]["pair"]["agents"] == ["agent_0", "agent_1"]
    assert full["assignments"]["pair"]["roles"]["agent_1"] == "right"


# ── Controller ticks ─────────────────────────────────────────────────────

def test_tick_first_call_decomposes_then_allocates(stub_autogen, tmp_path):
    ctrl, logger = _controller(
        decompose_raws=[json.dumps(decompose_response())],
        allocate_raws=[json.dumps(allocate_response(
            [{"agent": "agent_0", "task_id": "explore", "role": "walk"}]))],
        tmp_path=tmp_path)
    state = OrchestratorState()
    tick = _run_tick(ctrl, state)
    assert tick.decomposed and tick.allocated
    assert tick.reassigned == ["agent_0"]
    assert ctrl.dag.get("explore").status == "running"
    assert ctrl.dag.get("chop").status == "open"
    # Coupling surfaces:
    assert "move around the room" in ctrl.directive_text("agent_0")
    assert "MUST advance" in ctrl.directive_text("agent_0")
    assert "(none" in ctrl.assigned_objective("agent_1")
    # State mirror + logs:
    assert state.directives["agent_0"]["task_id"] == "explore"
    calls = [json.loads(l) for l in open(logger.calls_path)]
    assert [c["call_type"] for c in calls] == ["decompose", "allocate"]
    assert all(not c["failed"] for c in calls)
    assert len(open(logger.dag_path).readlines()) == 1
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert rows[0]["reason"] == "allocate" and rows[0]["agent"] == "agent_0"


def test_tick_milestone_completes_node_any_of(stub_autogen, tmp_path):
    ctrl, logger = _controller(tmp_path=tmp_path)
    ctrl.dag.add_task(_task("t1", milestones=["m1_move_5", "m4_dig_5_wood"]))
    ctrl.dag.assign("t1", ["agent_0"], {}, t=0)
    ctrl.last_decompose_step = 0                # suppress decompose this tick
    state = OrchestratorState()
    state.add_event(oevents.milestone_event(3, "m4_dig_5_wood", ["agent_0"]))
    tick = _run_tick(ctrl, state, t=4)
    assert tick.completed == ["t1"]
    assert ctrl.dag.get("t1").status == "success"
    assert state.event_buffer == []                       # drained
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert rows[0]["reason"] == "freed_success"


def test_tick_success_beats_timeout_same_tick(stub_autogen):
    ctrl, _ = _controller(cfg=_cfg(node_timeout_steps=5))
    ctrl.dag.add_task(_task("t1"))
    ctrl.dag.assign("t1", ["agent_0"], {}, t=0)
    ctrl.last_decompose_step = 0
    state = OrchestratorState()
    state.add_event(oevents.milestone_event(5, "m1_move_5", ["agent_0"]))
    _run_tick(ctrl, state, t=5)                           # timeout boundary
    assert ctrl.dag.get("t1").status == "success"


def test_tick_timeout_fails_node_and_redecomposes(stub_autogen, tmp_path):
    ctrl, logger = _controller(
        decompose_raws=[json.dumps(decompose_response())],
        allocate_raws=[json.dumps(allocate_response(
            [{"agent": "agent_0", "task_id": "explore", "role": "w"}]))],
        cfg=_cfg(node_timeout_steps=5, decompose_min_interval=2),
        tmp_path=tmp_path)
    ctrl.dag.add_task(_task("stuck", milestones=["m17_switch_pressed"]))
    ctrl.dag.assign("stuck", ["agent_1"], {}, t=0)
    ctrl.last_decompose_step = 0
    state = OrchestratorState()
    tick = _run_tick(ctrl, state, t=6)
    assert ("stuck", "timeout") in tick.failed
    assert ctrl.dag.get("stuck").status == "failure"
    assert tick.decomposed                                # replanned
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert rows[0]["reason"] == "freed_timeout"


def test_tick_death_fails_node(stub_autogen, tmp_path):
    ctrl, logger = _controller(tmp_path=tmp_path)
    ctrl.dag.add_task(_task("t1"))
    ctrl.dag.assign("t1", ["agent_2", "agent_0"], {}, t=0)
    ctrl.last_decompose_step = 0
    state = OrchestratorState()
    state.add_event(oevents.death_event(3, "agent_2"))
    tick = _run_tick(ctrl, state, living=["agent_0", "agent_1"], t=4)
    assert ("t1", "death") in tick.failed
    assert ctrl.dag.get("t1").status == "failure"
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert {r["agent"] for r in rows} == {"agent_0", "agent_2"}
    assert all(r["reason"] == "freed_death" for r in rows)


def test_tick_decompose_interval_and_failed_call_cooldown(stub_autogen,
                                                          tmp_path):
    # Both attempts of the first decompose return garbage -> DAG unchanged,
    # cooldown set, raw logged; a tick inside the interval makes NO call;
    # after the interval it retries with a good response.
    ctrl, logger = _controller(
        decompose_raws=["not json", "still not json",
                        json.dumps(decompose_response())],
        allocate_raws=[json.dumps(allocate_response(
            [{"agent": "agent_0", "task_id": "explore", "role": "w"}]))],
        cfg=_cfg(decompose_min_interval=4), tmp_path=tmp_path)
    state = OrchestratorState()
    tick0 = _run_tick(ctrl, state, t=0)
    assert not tick0.decomposed and len(ctrl.dag.tasks) == 0
    assert ctrl.failed_calls == 1
    assert len(ctrl.decompose_client.calls) == 2          # initial + 1 retry
    tick2 = _run_tick(ctrl, state, t=2)                   # inside cooldown
    assert len(ctrl.decompose_client.calls) == 2          # no new call
    assert not tick2.decomposed
    tick4 = _run_tick(ctrl, state, t=4)
    assert tick4.decomposed and len(ctrl.dag.tasks) == 2
    calls = [json.loads(l) for l in open(logger.calls_path)]
    # raw_output carries the LAST attempt's raw (the one that stood when
    # the call was abandoned).
    assert calls[0]["failed"] and calls[0]["raw_output"] == "still not json"


def test_tick_all_dead_no_llm_and_buffer_bounded(stub_autogen):
    ctrl, _ = _controller()
    ctrl.dag.add_task(_task("t1"))
    ctrl.dag.assign("t1", ["agent_0"], {}, t=0)
    state = OrchestratorState()
    for i in range(5):
        state.add_event(oevents.message_event(i, "agent_0", "agent_1", "hi"))
    tick = _run_tick(ctrl, state, living=[], t=3)
    assert ("t1", "death") in tick.failed
    assert len(ctrl.decompose_client.calls) == 0          # no LLM on wipe
    assert state.event_buffer == []                       # drained even so
    # Buffer stays bounded across repeated ticks.
    for t in range(4, 8):
        state.add_event(oevents.message_event(t, "agent_0", "agent_1", "x"))
        _run_tick(ctrl, state, living=[], t=t)
        assert state.event_buffer == []


# ── Curriculum suffix + prompt examples + directives ─────────────────────

def test_villager_suffix_identity_and_idempotence():
    base = "TASK PROMPT with {completed_tasks}"
    assert apply_villager_suffix(base, False) == base
    out = apply_villager_suffix(base, True)
    assert out.startswith(base)
    assert out.count(ASSIGNED_OBJECTIVE_PLACEHOLDER) == 1
    assert apply_villager_suffix(out, True) == out
    # Hard wording, distinct from the advisory plan suffix.
    assert "not advice" in VILLAGER_SUFFIX and "MUST" in VILLAGER_SUFFIX
    assert "ADVICE" in PLAN_SUFFIX
    assert VILLAGER_SUFFIX.format(assigned_objective="X").count("X") == 1
    # Legacy templates (no placeholder) ignore the extra kwarg.
    assert "PLAIN".format(assigned_objective="ignored") == "PLAIN"
    assert apply_plan_suffix(base, True) != out           # separate chains


def test_decompose_prompt_examples_use_real_ids():
    out = oprompt.format_decompose_prompt(
        n_agents=3, agent_names=LIVING, current_step=10,
        chamber_facts="CF", milestone_catalog="MC", env_state_text="ES",
        task_table="TT", dag_summary="DS", open_slots=4,
        example_milestones=["m8_anvil_A1", "m9_anvil_B1"])
    assert '"milestones": ["m8_anvil_A1"]' in out
    assert '"milestones": ["m9_anvil_B1"]' in out
    assert '"candidates": ["agent_0"]' in out
    assert "at most 4 new subtasks" in out
    assert "defeat the boss" in out


def test_allocate_prompt_example_covers_free_agents():
    out = oprompt.format_allocate_prompt(
        current_step=10, ready_tasks_block="RT", free_agents_block="FA",
        dag_summary="DS", free_agents=["agent_1", "agent_2"],
        ready_task_ids=["build_wall"])
    assert '"agent": "agent_1", "task_id": "build_wall"' in out
    assert '"agent": "agent_2", "task_id": "build_wall"' in out
    assert "AT MOST once" in out


def test_directive_and_objective_rendering():
    dag = TaskDAG()
    dag.add_task(_task("pair", milestones=["m8_anvil_A1"], min_agents=2))
    dag.assign("pair", ["agent_0", "agent_1"],
               {"agent_0": "punch anvil A", "agent_1": "punch with a0"}, t=0)
    text = ovillager.render_villager_directive("agent_0", dag)
    assert "Objective: desc pair" in text
    assert "Your role: punch anvil A" in text
    assert "Teammates on this task: agent_1" in text
    assert "m8_anvil_A1" in text
    note = ovillager.assigned_objective_note("agent_1", dag)
    assert note == "desc pair (your role: punch with a0)"
    assert "none right now" in ovillager.render_villager_directive(
        "agent_2", dag)
    assert "(none" in ovillager.assigned_objective_note("agent_2", dag)


def test_milestone_catalog_and_blocks():
    catalog = ovillager.build_milestone_catalog(
        {"agent_0": {"m1_move_5"}}, FAKE_TRACK, FAKE_TRACKS)
    assert "m1_move_5 (ch1_solo, +10): completed by agent_0" in catalog
    assert "m4_dig_5_wood (ch1_solo, +10): OPEN" in catalog
    assert ovillager.known_milestone_ids(FAKE_TRACK) == KNOWN

    dag = TaskDAG()
    dag.add_task(_task("t1", candidates=["agent_1"], min_agents=2))
    block = ovillager.build_ready_tasks_block(dag)
    assert "needs 2 agent(s)" in block and "candidates: agent_1" in block

    free_block = ovillager.build_free_agents_block(
        _env_state(), '  agent_0: current="dig wood" | recently completed: x',
        ["agent_0"])
    assert "agent_0: chamber=ch1, hp=20" in free_block
    assert 'current="dig wood"' in free_block


# ── Allocation fallback (regression: villager smoke seed_42) ────────────
# Two parallel min_agents=3 tasks were ready with 3 free agents; the
# allocator split the team across both on 8 consecutive calls, every group
# was under-crewed, and nobody was assigned for 60 steps.

def test_allocation_fallback_crews_first_crewable_task(stub_autogen,
                                                       tmp_path):
    split = json.dumps(allocate_response([
        {"agent": "agent_0", "task_id": "communal", "role": "go"},
        {"agent": "agent_1", "task_id": "communal", "role": "go"},
        {"agent": "agent_2", "task_id": "switches", "role": "press"},
    ]))
    ctrl, logger = _controller(allocate_raws=[split, split],
                               tmp_path=tmp_path)
    ctrl.dag.add_task(_task("communal", milestones=["m17_switch_pressed"],
                            min_agents=3))
    ctrl.dag.add_task(_task("switches", milestones=["m8_anvil_A1"],
                            min_agents=3))
    ctrl.last_decompose_step = 0
    state = OrchestratorState()
    tick = _run_tick(ctrl, state, t=8)
    assert len(ctrl.allocate_client.calls) == 2           # initial + retry
    assert tick.allocated
    assert sorted(tick.reassigned) == LIVING              # whole crew
    assert ctrl.dag.get("communal").status == "running"   # first in order
    assert ctrl.dag.get("switches").status == "open"
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert {r["reason"] for r in rows} == {"allocate_fallback"}
    calls = [json.loads(l) for l in open(logger.calls_path)]
    assert calls[-1]["failed"] and calls[-1]["fallback"]["task_id"] == "communal"


def test_allocation_fallback_respects_candidates_and_min_agents():
    dag = TaskDAG()
    dag.add_task(_task("needs_pair_of_a1", candidates=["agent_1"],
                       min_agents=2))                     # uncrewable
    dag.add_task(_task("solo", candidates=["agent_2"]))
    fb = ovillager.VillagerController._fallback_assignment(
        dag.ready_tasks(), ["agent_0", "agent_2"])
    assert fb == ("solo", ["agent_2"])
    assert ovillager.VillagerController._fallback_assignment(
        dag.ready_tasks(), ["agent_0"]) is None


def test_prompts_carry_crewing_rules():
    out = oprompt.format_allocate_prompt(
        current_step=1, ready_tasks_block="RT", free_agents_block="FA",
        dag_summary="DS", free_agents=LIVING, ready_task_ids=["t"])
    assert "pick ONE of them and give it its full crew" in out
    out2 = oprompt.format_decompose_prompt(
        n_agents=3, agent_names=LIVING, current_step=1, chamber_facts="C",
        milestone_catalog="M", env_state_text="E", task_table="T",
        dag_summary="D", open_slots=2, example_milestones=["m1_move_5"])
    assert "min_agents add up to more than the team" in out2


# ── Unreachable milestones (regression: villager smoke v2) ──────────────
# The decomposer kept proposing Ch1 dig tasks while the team was in Ch3/Ch4,
# and Ch1 tasks assigned before a teleport ran on until the timeout. The
# schedule only moves the team forward, so milestones of chambers behind the
# least-advanced living agent are unreachable: labelled in the catalog,
# rejected at ingestion, and failed immediately when already running.

CH_TRACK = {"m1_move_5": "ch1_solo", "m4_dig_5_wood": "ch1_solo",
            "m8_anvil_A1": "ch2_anvils", "m9_anvil_B1": "ch2_anvils",
            "m17_switch_pressed": "ch3_switches"}


def test_chamber_index_and_team_min():
    assert ovillager.chamber_index("ch3_communal") == 3
    assert ovillager.chamber_index("ch1") == 1
    assert ovillager.chamber_index(None) is None
    chambers = {"agent_0": "ch3_cell", "agent_1": "ch2", "agent_2": None}
    assert ovillager.team_min_chamber(chambers, LIVING) == 2   # laggard rules
    assert ovillager.team_min_chamber({}, LIVING) is None


def test_unreachable_milestones_are_behind_the_laggard():
    un = ovillager.unreachable_milestones(KNOWN, 3, CH_TRACK)
    assert un == {"m1_move_5", "m4_dig_5_wood", "m8_anvil_A1", "m9_anvil_B1"}
    assert ovillager.unreachable_milestones(KNOWN, 1, CH_TRACK) == set()
    assert ovillager.unreachable_milestones(KNOWN, None, CH_TRACK) == set()


def test_ingest_rejects_unreachable_tasks():
    dag = TaskDAG()
    r = _ingest(dag, decompose_response([
        {"id": "stale", "description": "dig in ch1",
         "milestones": ["m4_dig_5_wood"], "required": [],
         "candidates": [], "min_agents": 1},
        {"id": "mixed", "description": "d",
         "milestones": ["m1_move_5", "m17_switch_pressed"],
         "required": [], "candidates": [], "min_agents": 1}]),
        unreachable={"m1_move_5", "m4_dig_5_wood"})
    assert ("stale", "unreachable chamber") in r["rejected"]
    assert dag.get("mixed").milestones == ["m17_switch_pressed"]


def test_catalog_labels_unreachable():
    cat = ovillager.build_milestone_catalog(
        {}, FAKE_TRACK, FAKE_TRACKS, unreachable={"m1_move_5"})
    assert "m1_move_5 (ch1_solo, +10): UNREACHABLE" in cat
    assert "m4_dig_5_wood (ch1_solo, +10): OPEN" in cat
    assert "never target it" in cat


def test_tick_fails_running_and_open_tasks_left_behind(stub_autogen,
                                                       tmp_path):
    ctrl, logger = _controller(tmp_path=tmp_path)
    ctrl._milestone_track = CH_TRACK
    ctrl.dag.add_task(_task("ch1_dig", milestones=["m4_dig_5_wood"]))
    ctrl.dag.add_task(_task("ch1_open", milestones=["m1_move_5"]))
    ctrl.dag.add_task(_task("ch3_switch", milestones=["m17_switch_pressed"]))
    ctrl.dag.assign("ch1_dig", ["agent_0"], {}, t=0)
    ctrl.last_decompose_step = 0
    state = OrchestratorState()
    env = _env_state()
    for a in LIVING:                                      # team teleported
        env["agents"][a]["chamber"] = "ch3_cell"
    tick = asyncio.run(ctrl.tick(
        state=state, living_agents=LIVING, episode=1, t=3,
        env_state=env, task_table="", agent_milestones={},
        parse_json=json.loads))
    assert ("ch1_dig", "unreachable") in tick.failed      # running -> failed
    assert ("ch1_open", "unreachable") in tick.failed     # open -> failed
    rows = [json.loads(l) for l in open(logger.assignments_path)]
    assert rows[0]["reason"] == "freed_unreachable"
    # The chamber-ahead task was untouched by the sweep, became the only
    # ready task, and — with no usable allocator response (empty fake) —
    # the deterministic fallback crewed it in the same tick.
    assert ctrl.dag.get("ch3_switch").status == "running"
    assert ctrl.dag.get("ch3_switch").failure_reason == ""
    assert any(r["reason"] == "allocate_fallback"
               and r["task_id"] == "ch3_switch" for r in rows)
