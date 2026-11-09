"""Chained AgentWorld tasks (chains/): naming, renaming, state, hooks, plasticity.

Tests marked ``needs_aw`` use a real AgentWorld checkout ($AGENTWORLD_ROOT,
>= 23a07d52 for scripts/score_psr.py) and are skipped without one.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CHAINS = REPO / "chains"
sys.path.insert(0, str(CHAINS))

from chain_spec import (ChainError, ChainStep, NamingError, assign_slots,  # noqa: E402
                        make_username, parse_username, role_from_username)
from generate_tasks import rename_task_text, replacement_table, rewrite_text  # noqa: E402
from wired_agent import WiredMixin  # noqa: E402
from wired_state import (AgentSession, AgentStateStore, ChainRunError, StepContext,  # noqa: E402
                         Teammate, TeamPlasticity, addressees, interactions_for_turn,
                         parse_response)

FIX = REPO / "tests" / "fixtures" / "agentworld" / "v1.3_benchmark"
TASK02 = FIX / "task_02_arrow_production.yaml"
AW = os.environ.get("AGENTWORLD_ROOT")
needs_aw = pytest.mark.skipif(not (AW and (Path(AW) / "agents" / "run.py").is_file()),
                              reason="needs an AgentWorld checkout in $AGENTWORLD_ROOT")
BENCH = "data_v0.1_multi/v1.3_benchmark"


# ── naming ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("original,role", [
    ("t06_weapon_smith_agent", "weapon_smith"), ("t03_miner_agent", "miner"),
    ("t17_palm_logger_agent", "palm_logger"), ("t100_scout_agent", "scout"),
])
def test_role_from_task_username(original, role):
    assert role_from_username(original) == role


@pytest.mark.parametrize("bad", ["T03_Miner_agent", "t03__agent", "t03_miner-x_agent", "t03_"])
def test_bad_task_usernames_fail(bad):
    with pytest.raises(NamingError):
        role_from_username(bad)


def test_username_round_trip_and_limits():
    assert make_username("a3", "weapon_smith") == "a3_weapon_smith"
    assert parse_username("a3_weapon_smith") == ("a3", "weapon_smith")
    with pytest.raises(NamingError):
        make_username("a1", "x" * 30)                 # 33 characters
    for bad in ("A1_miner", "a1 miner", "a1_", "a1-miner", "miner", "1a_miner",
                "t06_weapon_smith_agent", "a1_miner_agent"):
        with pytest.raises(NamingError):
            parse_username(bad)
    with pytest.raises(NamingError):
        make_username("a_1", "miner")                 # id may not contain "_"


def _task02():
    return yaml.safe_load(TASK02.read_text(encoding="utf-8"))


def test_assign_slots_checks_the_map():
    data = _task02()
    out = assign_slots(data, {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"}, "t")
    assert [a.username for a in out] == ["a1_lumberjack", "a2_fletcher", "a3_bowyer"] or \
        all(a.username == f"{a.agent_id}_{a.role}" for a in out)
    with pytest.raises(ChainError):
        assign_slots(data, {"agent_1": "a1", "agent_2": "a2"}, "t")             # missing slot
    with pytest.raises(ChainError):
        assign_slots(data, {"agent_1": "a1", "agent_2": "a1", "agent_3": "a3"}, "t")
    with pytest.raises(ChainError):
        assign_slots(data, {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3",
                            "agent_4": "a4"}, "t")


# ── renaming ─────────────────────────────────────────────────────────────────
def _step(path: Path, agents) -> ChainStep:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    s = ChainStep(index=1, source=path, agents=agents)
    s.assignments = assign_slots(data, agents, "t")
    return s


def test_rewrite_is_one_pass_and_respects_word_edges():
    table = {"t06_weapon_smith_agent": "a3_weapon_smith", "weapon_smith_agent": "a3_weapon_smith",
             "smith_agent": "a9_smith"}
    text = "weapon_smith_agent and smith_agent, Smith_Agent, not blacksmith_agent2"
    assert rewrite_text(text, table) == ("a3_weapon_smith and a9_smith, a9_smith, "
                                         "not blacksmith_agent2")


def test_rename_task02_changes_only_names(tmp_path):
    step = _step(TASK02, {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"})
    raw = TASK02.read_bytes().decode("utf-8")
    new = rename_task_text(raw, step)
    orig_lines, new_lines = raw.splitlines(True), new.splitlines(True)
    assert len(orig_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(orig_lines, new_lines) if a != b]
    assert changed
    for a, b in changed:
        assert any(o in a for o in replacement_table(step)) or any(
            o in a.lower() for o in replacement_table(step))
    data = yaml.safe_load(new)
    assert [data[a.slot]["username"] for a in step.assignments] == \
        [a.username for a in step.assignments]
    # CRLF files keep their line endings
    crlf = rename_task_text(raw.replace("\r\n", "\n").replace("\n", "\r\n"), step)
    assert crlf.count("\r\n") == raw.replace("\r\n", "\n").count("\n")


def test_rename_refuses_tasks_without_new_character(tmp_path):
    src = tmp_path / "task_02_arrow_production.yaml"
    src.write_text(TASK02.read_text(encoding="utf-8").replace("new_character: true",
                                                               "new_character: false", 1),
                   encoding="utf-8")
    step = _step(src, {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"})
    with pytest.raises(ChainError, match="new_character"):
        rename_task_text(src.read_text(encoding="utf-8"), step)


# ── store, memory, prompt ────────────────────────────────────────────────────
def _session(tmp_path, arm="hebbian", step=2, state=None):
    store = AgentStateStore(tmp_path / "agent_state")
    s = AgentSession(agent_id="a1", role="miner", username="a1_miner", step=step,
                     task_key="task_06_heavy_sword", arm=arm, store=store,
                     state=state or store.load("a1"), bonds={"a2": 0.31, "a3": 0.12})
    s.start([Teammate("a2", "smelter", "a2_smelter"), Teammate("a3", "weapon_smith",
                                                                "a3_weapon_smith")])
    return s


def test_store_round_trip_and_version(tmp_path):
    store = AgentStateStore(tmp_path)
    st = store.load("a1")
    assert st["bonds"] == {} and st["saved_after_step"] is None
    st["bonds"] = {"a2": 0.5}
    store.save("a1", st)
    assert store.load("a1")["bonds"] == {"a2": 0.5}
    bad = json.loads((tmp_path / "a1.json").read_text())
    bad["version"] = 99
    (tmp_path / "a1.json").write_text(json.dumps(bad))
    with pytest.raises(ChainRunError):
        store.load("a1")


def test_prompt_block_per_arm(tmp_path):
    state = {"version": 1, "agent_id": "a1", "bonds": {"a2": 0.31}, "saved_after_step": 1,
             "history": [{"step": 1, "task": "task_03_silver_ring", "role": "miner",
                          "success": True, "message": "Silver ring: True"}],
             "memory": [{"step": 1, "task": "task_03_silver_ring", "round": 4,
                         "kind": "xfer_received", "partner": "a2", "partner_role": "smelter",
                         "own_role": "miner", "text": "2x Iron Bar", "broadcast": False},
                        {"step": 2, "task": "task_06_heavy_sword", "round": 1,
                         "kind": "msg_received", "partner": "a3", "partner_role": "weapon_smith",
                         "own_role": "miner", "text": "this step", "broadcast": True}]}
    assert _session(tmp_path, "reference", state=json.loads(json.dumps(state))).prompt_block() == ""
    mem = _session(tmp_path, "memory", state=json.loads(json.dumps(state))).prompt_block()
    heb = _session(tmp_path, "hebbian", state=json.loads(json.dumps(state))).prompt_block()
    assert "they gave you 2x Iron Bar" in mem and "now a2_smelter" in mem
    assert "this step" not in mem                       # current step stays in the chat history
    assert "step 1, task_03_silver_ring, you were miner: success" in mem
    assert "strongest bonds" not in mem
    assert heb.startswith(mem.split("=== END")[0]) and "a2_smelter (0.31), a3_weapon_smith (0.12)" in heb


def test_finish_saves_bonds_history_and_memory(tmp_path):
    s = _session(tmp_path, step=1)
    s.remember(3, {"kind": "xfer_given", "partner": "a2", "partner_role": "smelter",
                   "text": "1x Iron Ore"})
    s.bonds = {"a2": 0.4, "a3": 0.1}
    path = s.finish({"success": False, "message": "Heavy sword: False", "psr": 0.0,
                     "rounds": 55, "chats": 3})
    saved = json.loads(path.read_text())
    assert saved["saved_after_step"] == 1 and saved["bonds"] == {"a2": 0.4, "a3": 0.1}
    assert saved["memory"][0]["partner"] == "a2" and saved["memory"][0]["own_role"] == "miner"
    h = saved["history"][0]
    assert h["bonds_start"] == {"a2": 0.31, "a3": 0.12} and h["bonds_end"]["a2"] == 0.4


# ── parsing turns ────────────────────────────────────────────────────────────
TEAM = [Teammate("a1", "miner", "a1_miner"), Teammate("a2", "smelter", "a2_smelter"),
        Teammate("a3", "weapon_smith", "a3_weapon_smith")]
NAMES = {t.username: i for i, t in enumerate(TEAM)}


def test_parse_transfer_and_partner_ids():
    resp = ("thinking\n[TOOL_CALL_INFO] transfer_items(targetPlayer=a2_smelter, itemKey=ironore, "
            "count=2)\n[TOOL_RESULT] ✅ Transfer completed: 2x Iron Ore transferred from a1_miner "
            "to a2_smelter (method: api)")
    call, result = parse_response(resp)
    assert call.name == "transfer_items" and call.args["targetPlayer"] == "a2_smelter"
    turn = interactions_for_turn(0, call, result, TEAM, NAMES)
    assert turn.by_agent[0] == [{"kind": "xfer_given", "partner": "a2", "partner_role": "smelter",
                                 "text": "2x Iron Ore"}]
    assert turn.by_agent[1][0]["kind"] == "xfer_received" and turn.by_agent[1][0]["partner"] == "a1"
    assert turn.dms == []


def test_chat_addressing():
    assert addressees("a2_smelter, send bars please", 0, TEAM) == ([1], False)
    assert addressees("the weapon smith needs logs", 0, TEAM) == ([2], False)
    assert addressees("hello team", 0, TEAM) == ([1, 2], True)
    resp = "[TOOL_CALL_INFO] chat(message=a3 come here)\n[TOOL_RESULT] Global chat message sent: a3 come here"
    call, result = parse_response(resp)
    turn = interactions_for_turn(0, call, result, TEAM, NAMES)
    assert turn.dms == [(0, 2)] and turn.by_agent[2][0]["kind"] == "msg_received"


# ── plasticity and the step context ─────────────────────────────────────────
def _renamed_task02(tmp_path, agents):
    step = _step(TASK02, agents)
    d = tmp_path / "task"
    d.mkdir(parents=True, exist_ok=True)
    y = d / TASK02.name
    y.write_bytes(rename_task_text(TASK02.read_bytes().decode("utf-8"), step).encode())
    v = d / "task_02_arrow_production_success_criteria.py"
    shutil.copyfile(FIX / "task_02_success_criteria.py", v)
    return step, y, v


def _obs(x, y, items=None):
    return {"location": {"x": x, "y": y}, "playerStatus": {"hitPoints": 60},
            "inventory": {"items": [{"key": k, "count": c} for k, c in (items or {}).items()]}}


def test_team_plasticity_warm_start_by_agent_id(tmp_path):
    step, y, v = _renamed_task02(tmp_path, {"agent_1": "a2", "agent_2": "a3", "agent_3": "a1"})
    team = [Teammate(a.agent_id, a.role, a.username) for a in step.assignments]
    bonds = {"a1": {"a2": 0.7, "a3": 0.2}, "a2": {"a1": 0.6}}
    tp = TeamPlasticity(team, bonds, y, v)
    W = tp.graph.W
    i = {t.agent_id: k for k, t in enumerate(team)}
    assert W[i["a1"], i["a2"]] == pytest.approx(0.7) and W[i["a2"], i["a1"]] == pytest.approx(0.6)
    assert W[i["a3"], i["a1"]] == pytest.approx(float(tp.h["hebbian_init_weight"]))
    assert tp.bonds_of(i["a1"]) == {"a2": pytest.approx(0.7), "a3": pytest.approx(0.2)}


class _Tools:
    def __init__(self, obs):
        self.token, self._obs = "tok", obs

    def get_last_observation_data(self):
        return self._obs

    def logout_character(self):
        self.token = None


class _Base:
    def __init__(self, username):
        self.username = username
        self.conversation_history = [{"role": "system", "content": "SYS"}]

    def _update_system_prompt(self):
        self.conversation_history[0]["content"] = "SYS"


class _Agent(WiredMixin, _Base):
    def __init__(self, username, obs):
        self._wired_setup(_Base, {"username": username, "api_key": None})
        self.game_tools = _Tools(obs)


class _State:
    class _V:
        value = "waiting"
    state = _V()
    error_message = None

    def __init__(self, console):
        self.console = console


class _Console:
    def __init__(self, username, obs):
        self.username = username
        self.agent = _Agent(username, obs)


def _run_step(tmp_path, store, step_no, agents, arm="hebbian"):
    step, y, v = _renamed_task02(tmp_path / f"s{step_no}", agents)
    slots = {a.slot: {"agent_id": a.agent_id, "role": a.role, "username": a.username}
             for a in step.assignments}
    pos = [(388, 3), (390, 3), (389, 4)]
    states = {a.slot: _State(_Console(a.username, _obs(*pos[k], {"flask": 1})))
              for k, a in enumerate(step.assignments)}
    ctx = StepContext(step=step_no, task_key=step.task_stem, task_yaml=y, verifier=v, slots=slots,
                      store=store, arm=arm, verify=lambda traj: (False, "arrows: 0/10"),
                      log_path=tmp_path / f"wired_{step_no}.jsonl")
    ctx.on_agents_ready(states)
    u = {a.slot: a.username for a in step.assignments}
    traj = {"rounds": []}
    for rnd in (1, 2):
        ctx.on_action(rnd, "agent_1", f"[TOOL_CALL_INFO] chat(message=hi)\n[TOOL_RESULT] "
                      f"Global chat message sent: {u['agent_2']} I have logs", _obs(388, 3), traj)
        ctx.on_action(rnd, "agent_2", "[TOOL_CALL_INFO] transfer_items(targetPlayer="
                      f"{u['agent_3']}, itemKey=stick, count=2)\n[TOOL_RESULT] Transfer completed: "
                      f"2x Stick transferred from {u['agent_2']} to {u['agent_3']} (method: api)",
                      _obs(390, 3, {"flask": 1}), traj)
        ctx.on_action(rnd, "agent_3", "[TOOL_CALL_INFO] sleep(seconds=1)\n[TOOL_RESULT] Slept",
                      _obs(389, 4, {"flask": 1, "stick": 2 * rnd}), traj)
        traj["rounds"].append({"round": rnd})
    ctx.flush(traj)
    saved = ctx.end({"success": False, "message": "arrows: 0/10", "psr": 0.0,
                     "rounds": 2, "chats": 2})
    return ctx, step, states, saved


def test_step_context_end_to_end(tmp_path):
    store = AgentStateStore(tmp_path / "agent_state")
    ctx, step, states, saved = _run_step(tmp_path, store, 1,
                                         {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"})
    assert ctx.rounds_learned == 2 and set(saved) == {"a1", "a2", "a3"}
    assert all(not v["file_existed"] for v in ctx.loaded.values())
    a1 = store.load("a1")
    assert a1["saved_after_step"] == 1 and set(a1["bonds"]) == {"a2", "a3"}
    assert a1["bonds"]["a2"] > a1["bonds"]["a3"]          # a1 messaged a2 twice, never a3
    kinds = {(m["kind"], m["partner"]) for m in store.load("a3")["memory"]}
    assert kinds == {("xfer_received", "a2")}
    rec = [json.loads(line) for line in (tmp_path / "wired_1.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rec] == [1, 2] and ["a1", "a2"] in rec[0]["comm"]
    assert any(t == ["a2", "a3", "xfer"] for t in rec[0]["triples"])
    # the bond line reaches the system prompt (hebbian arm)
    agent = states["agent_1"].console.agent
    agent._update_system_prompt()
    assert "strongest bonds" in agent.conversation_history[0]["content"]

    # step 2 with rotated slots loads every file saved by step 1
    ctx2, *_ = _run_step(tmp_path, store, 2, {"agent_1": "a3", "agent_2": "a1", "agent_3": "a2"})
    assert all(v["file_existed"] and v["saved_after_step"] == 1 for v in ctx2.loaded.values())
    i = {s.agent_id: k for k, s in enumerate(ctx2.team)}
    W0 = ctx2.loaded["a1"]["bonds"]["a2"]
    assert W0 == pytest.approx(a1["bonds"]["a2"], abs=1e-5)
    assert [h["step"] for h in store.load("a1")["history"]] == [1, 2]
    assert store.load("a1")["history"][1]["role"] == "fletcher" or \
        store.load("a1")["history"][1]["role"] == ctx2.team[i["a1"]].role


def test_step_context_rejects_plain_agents(tmp_path):
    step, y, v = _renamed_task02(tmp_path, {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"})
    slots = {a.slot: {"agent_id": a.agent_id, "role": a.role, "username": a.username}
             for a in step.assignments}

    class Plain:
        def __init__(self, u):
            self.username, self.agent = u, type("A", (), {"game_tools": _Tools(None)})()

    states = {a.slot: _State(Plain(a.username)) for a in step.assignments}
    ctx = StepContext(step=1, task_key=step.task_stem, task_yaml=y, verifier=v, slots=slots,
                      store=AgentStateStore(tmp_path / "st"), arm="hebbian",
                      verify=lambda t: (False, ""))
    with pytest.raises(ChainRunError, match="provider: custom"):
        ctx.on_agents_ready(states)


# ── against a real AgentWorld checkout ───────────────────────────────────────
@pytest.fixture
def aw_root():
    from run_chain import bootstrap      # AgentWorld's agents/ first (src/mindforge has an agent_factory too)
    root = Path(AW).resolve()
    bootstrap(root)
    return root


def _chain_file(tmp_path, name, steps):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"chain": name, "steps": steps}))
    return p


@needs_aw
def test_generate_smithing_3(aw_root, tmp_path):
    from chain_spec import load_chain
    from generate_tasks import generate_chain
    chain = load_chain(CHAINS / "smithing_3.yaml", aw_root)
    gen = generate_chain(chain, aw_root, tmp_path / "smithing_3")
    names = [[a.username for a in g.step.assignments] for g in gen]
    assert names == [["a1_miner", "a2_smelter", "a3_smith"],
                     ["a1_miner", "a2_smelter", "a3_weapon_smith"],
                     ["a1_miner", "a2_smelter", "a3_tool_smith"]]
    for g in gen:
        assert g.verifier.read_bytes() == g.source_verifier.read_bytes()
        assert g.task_yaml.name == g.step.source.name
        assert "_agent" not in yaml.safe_load(g.task_yaml.read_text(encoding="utf-8"))["agent_3"]["username"]
    rot = load_chain(CHAINS / "smithing_3_rotate.yaml", aw_root)
    assert [a.username for a in rot.steps[2].assignments] == ["a3_miner", "a1_smelter", "a2_tool_smith"]


@needs_aw
def test_generated_verifier_resolves_in_a_fresh_process(aw_root, tmp_path):
    from chain_spec import load_chain
    from generate_tasks import generate_chain
    gen = generate_chain(load_chain(CHAINS / "smithing_3.yaml", aw_root), aw_root, tmp_path / "c")
    traj = {"task_source": str(gen[0].task_yaml), "rounds": [{"actions": [
        {"agent_name": "agent_3", "observation": {"inventory": {"items": [
            {"key": "silverring", "count": 1}]}}}]}]}
    tf = tmp_path / "traj.json"
    tf.write_text(json.dumps(traj))
    code = ("import json,sys; sys.path.insert(0, sys.argv[1]); from verification import "
            "verify_trajectory, verifier_for_task; t=json.load(open(sys.argv[2])); "
            "print(verifier_for_task(t['task_source'])); print(verify_trajectory(t))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(aw_root / "agents"), str(tf)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert Path(lines[0]).resolve() == gen[0].verifier.resolve()
    assert lines[1] == "(True, 'Silver ring: True')"


@needs_aw
@pytest.mark.parametrize("task,role_items,msg", [
    ("task_17_forest_gathering", {"palm_logger": {"palmlogs": 4}, "peach_forager": {"peach": 5}},
     "palmlogs: 4/4, peach: 5/5"),
    ("task_18_coastal_harvest", None, None),
    ("task_19_farm_defense", {"corn_harvester": {"corn": 8}, "tomato_gatherer": {"tomato": 4}},
     "corn: 8/8, tomato: 4/4"),
    ("task_20_orchard_protection", {"berry_picker": {"blueberry": 6}, "pond_fisher": {"rawtuna": 3}},
     "blueberry: 6/6, rawtuna: 3/3"),
])
def test_username_matching_verifiers_after_renaming(aw_root, tmp_path, task, role_items, msg):
    from chain_spec import load_chain
    from generate_tasks import generate_chain, load_verification
    src = yaml.safe_load((aw_root / BENCH / f"{task}.yaml").read_text(encoding="utf-8"))
    slots = sorted(k for k in src if k.startswith("agent_"))
    ids = {s: f"a{k + 1}" for k, s in enumerate(slots)}
    chain = load_chain(_chain_file(tmp_path, "c", [{"task": f"{BENCH}/{task}.yaml",
                                                     "agents": ids}]), aw_root)
    g = generate_chain(chain, aw_root, tmp_path / "out")[0]
    verification = load_verification(aw_root)
    if role_items is None:                     # task 18 sums over everyone
        role_items = {g.step.assignments[0].role: {"rawshrimp": 6, "icelogs": 3}}
        msg = "rawshrimp: 6/6, icelogs: 3/3"
    by_role = {a.role: a.slot for a in g.step.assignments}

    def traj(assign):
        acts = [{"agent_name": by_role[r], "observation": {"inventory": {"items": [
            {"key": k, "count": c} for k, c in items.items()]}}} for r, items in assign.items()]
        return {"task_source": str(g.task_yaml), "rounds": [{"actions": acts}],
                "task_definition": yaml.safe_load(g.task_yaml.read_text(encoding="utf-8"))}

    assert verification.verify_trajectory(traj(role_items)) == (True, msg)
    if len(role_items) == 2:                   # the same items on the wrong agents fail
        (r1, i1), (r2, i2) = role_items.items()
        ok, _ = verification.verify_trajectory(traj({r1: i2, r2: i1}))
        assert not ok


@needs_aw
def test_psr_matches_agentworld_checkpoint_rules(aw_root):
    from run_chain import _load_file_module, psr_for
    score = _load_file_module("aw_score_psr", aw_root / "scripts" / "score_psr.py")
    if score is None:
        pytest.skip("AgentWorld checkout older than 23a07d52 (no scripts/score_psr.py)")
    p = psr_for(score, "task_07_pickaxe_crafting", "Pickaxe count: 0", False)
    assert p["PSR"] == 0.0 and p["n_checkpoints"] == 1 and p["checkpoints"][0]["target"] == 1
    p = psr_for(score, "task_03_silver_ring", "Silver ring: True", True)
    assert p["PSR"] == 1.0 and p["n_met"] == 1
    p = psr_for(score, "task_17_forest_gathering", "palmlogs: 2/4, peach: 5/5", False)
    assert p["PSR"] == pytest.approx(0.75) and p["n_met"] == 1


@needs_aw
def test_factory_builds_the_wired_adapter(aw_root):
    import importlib
    from agent_factory import AgentFactory
    agent = AgentFactory.create_agent(
        provider="custom", api_key="x", model="m", username="a3_weapon_smith", password="p",
        base_url="http://127.0.0.1:9", llm_params={"agent_class": "wired_agent:WiredOpenAIAgent",
                                                   "base_url": "http://127.0.0.1:9/v1"})
    assert (agent.agent_id, agent.role) == ("a3", "weapon_smith")
    assert importlib.import_module("wired_agent").WiredOpenAIAgent.__mro__[2].__name__ == "OpenAIAgent"
    with pytest.raises(ValueError):
        AgentFactory.create_agent(provider="custom", api_key="x", model="m",
                                  username="t06_weapon_smith_agent", password="p",
                                  llm_params={"agent_class": "wired_agent:WiredOpenAIAgent"})


# ── check_chain_run ──────────────────────────────────────────────────────────
def _fake_run(tmp_path, start2):
    """Two-step run dir: a1 ends step 1 with a silverring; step 2 starts with ``start2``."""
    from check_chain_run import run_checks
    run = tmp_path / "run"
    task = {"agent_1": {"username": "a1_miner", "new_character": True,
                        "location": {"x": 10, "y": 10},
                        "inventory_items": [{"item": "flask", "count": 2}]}}
    steps, store = [], AgentStateStore(run / "agent_state")
    for k, (start, end) in enumerate([({"flask": 2}, {"flask": 2, "silverring": 1}),
                                      (start2, {"flask": 2})], start=1):
        d = f"step_0{k}_task_03"
        (run / "tasks" / d).mkdir(parents=True)
        (run / "tasks" / d / "task_03_silver_ring.yaml").write_text(yaml.safe_dump(task))
        (run / d / "observation_logs").mkdir(parents=True)
        (run / d / "observation_logs" / "agent_1_observations_x.jsonl").write_text(json.dumps(
            {"round_number": 0, "observation": _obs(10, 10, start)}) + "\n")
        (run / d / "task_03_trajectory.json").write_text(json.dumps(
            {"task_definition": task, "rounds": [{"actions": [
                {"agent_name": "agent_1", "observation": _obs(10, 10, end)}]}]}))
        st = store.load("a1")
        st.setdefault("history", []).append({"step": k})
        st["saved_after_step"] = k
        store.save("a1", st)
        steps.append({"step": k, "dir": d, "task_yaml": f"/elsewhere/{d}/task_03_silver_ring.yaml",
                      "slots": [{"slot": "agent_1", "agent_id": "a1", "role": "miner",
                                 "username": "a1_miner", "original_username": "t03_miner_agent"}],
                      "success": False, "verifier_message": "Silver ring: False",
                      "state": {"loaded": {"a1": {"file_existed": k > 1,
                                                  "saved_after_step": k - 1 or None}},
                                "saved": {"a1": "a1.json"}}})
    (run / "chain_summary.json").write_text(json.dumps({"steps": steps, "flags": {}}))
    return run_checks(run)


def test_check_reset_passes_a_clean_reset(tmp_path):
    r = _fake_run(tmp_path, {"flask": 2})
    assert r["ok"], r


def test_check_reset_catches_carried_items(tmp_path):
    r = _fake_run(tmp_path, {"flask": 2, "silverring": 1})
    assert not r["reset"]["ok"] and "carried over {'silverring': 1}" in r["reset"]["problems"][0]
    assert r["state"]["ok"] and r["usernames"]["ok"]


# ── skill log ────────────────────────────────────────────────────────────────
def _skill_step(tmp_path, store, step_no, inv0, arm="memory"):
    step, y, v = _renamed_task02(tmp_path / f"k{step_no}",
                                 {"agent_1": "a1", "agent_2": "a2", "agent_3": "a3"})
    slots = {a.slot: {"agent_id": a.agent_id, "role": a.role, "username": a.username}
             for a in step.assignments}
    states = {a.slot: _State(_Console(a.username, _obs(0, 0, inv0[a.slot])))
              for a in step.assignments}
    ctx = StepContext(step=step_no, task_key=step.task_stem, task_yaml=y, verifier=v,
                      slots=slots, store=store, arm=arm, verify=lambda t: (False, "arrows: 0/10"))
    ctx.on_agents_ready(states)
    return ctx, states


def test_skill_log_credits_own_work_only(tmp_path):
    store = AgentStateStore(tmp_path / "agent_state")
    inv0 = {"agent_1": {"flask": 1}, "agent_2": {"flask": 1}, "agent_3": {"flask": 1, "logs": 2}}
    ctx, _ = _skill_step(tmp_path, store, 1, inv0)
    t = {"rounds": []}
    # round 1: a1 harvests (yield not visible yet), a2's craft fails, a3 hands a1 two logs
    ctx.on_action(1, "agent_1", "[TOOL_CALL_INFO] harvest_resource(targetInstance=5)\n"
                  "[TOOL_RESULT] HARVEST IN PROGRESS", _obs(0, 0, {"flask": 1}), t)
    ctx.on_action(1, "agent_2", "[TOOL_CALL_INFO] craft_item(skill=Fletching, itemKey=stick)\n"
                  "[TOOL_RESULT] Error: you need 1x logs\nmore text", _obs(0, 0, {"flask": 1}), t)
    ctx.on_action(1, "agent_3", "[TOOL_CALL_INFO] transfer_items(targetPlayer=a1_lumberjack, "
                  "itemKey=logs, count=2)\n[TOOL_RESULT] Transfer completed: 2x Logs transferred "
                  "from a3_fletcher to a1_lumberjack (method: api)", _obs(0, 0, {"flask": 1}), t)
    t["rounds"].append({})
    # round 2: a1 now holds 3 logs (2 handed over + 1 harvested); a2's craft works
    ctx.on_action(2, "agent_1", "[TOOL_CALL_INFO] chat(message=thanks)\n[TOOL_RESULT] Global chat "
                  "message sent: thanks", _obs(0, 0, {"flask": 1, "logs": 3}), t)
    ctx.on_action(2, "agent_2", "[TOOL_CALL_INFO] craft_item(skill=Fletching, itemKey=stick)\n"
                  "[TOOL_RESULT] Successfully crafted 4x Stick!", _obs(0, 0, {"flask": 1, "stick": 4}), t)
    ctx.flush(t)
    ctx.end({"success": False, "message": "arrows: 0/10", "psr": 0.0, "rounds": 2, "chats": 1})

    a1 = store.load("a1")["skills"]
    assert [(s["key"], s["ok"], s["gained"]) for s in a1] == [("harvest_resource -> logs", True, {"logs": 1})]
    a2 = {(s["key"], s["ok"]): s for s in store.load("a2")["skills"]}
    ok = a2[("craft_item(skill=Fletching, itemKey=stick)", True)]
    bad = a2[("craft_item(skill=Fletching, itemKey=stick)", False)]
    assert ok["gained"] == {"stick": 4} and ok["by_step"] == {"1": 1}
    assert bad["error"] == "Error: you need 1x logs"
    assert store.load("a3")["skills"] == []                 # giving items away is not a skill

    # step 2: lessons from step 1 appear, in the memory and hebbian arms only
    ctx2, states2 = _skill_step(tmp_path, store, 2, inv0)
    block = ctx2.sessions["agent_2"].prompt_block()
    assert "What you did yourself in earlier tasks:" in block
    assert "craft_item(skill=Fletching, itemKey=stick): worked 1x, got 4x stick" in block
    assert 'failed 1x (as hunter in task_02_arrow_production): "Error: you need 1x logs"' in block
    ctx3, _ = _skill_step(tmp_path / "ref", AgentStateStore(tmp_path / "agent_state"), 3, inv0,
                          arm="reference")
    assert ctx3.sessions["agent_2"].prompt_block() == ""


def test_skill_lines_put_this_tasks_items_first(tmp_path):
    s = _session(tmp_path, "memory", step=3)
    s.task_text = "smelt ironbar then craft the axe"
    s.state["skills"] = [
        {"key": "harvest_resource -> peach", "ok": True, "by_step": {"2": 5}, "gained": {"peach": 5},
         "items": ["peach"], "error": "", "uses": [{"step": 2, "task": "t", "role": "forager"}]},
        {"key": "craft_item(skill=Smelting, itemKey=ironbar)", "ok": False, "by_step": {"1": 2},
         "gained": {}, "items": ["ironbar"], "error": "level too low",
         "uses": [{"step": 1, "task": "task_04_axe_crafting", "role": "miner"}]},
        {"key": "craft_item(skill=Smithing, itemKey=axe)", "ok": True, "by_step": {"3": 1},
         "gained": {"axe": 1}, "items": ["axe"], "error": "",
         "uses": [{"step": 3, "task": "now", "role": "miner"}]},   # current step: not shown
    ]
    lines = s._skill_lines()
    assert len(lines) == 2 and "Smelting, itemKey=ironbar): failed 2x" in lines[0]
    assert "peach" in lines[1]
