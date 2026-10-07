"""Tests for the HMAS-2 hard-orchestrator variant (hmas2) and its hub
plumbing.

Conventions as in tests/test_orchestrator_villager.py: light imports,
autogen stubbed per test, fake clients returning canned raw text, and every
runtime source (environment, metric, curriculum validator, tokenizer)
injected, so the protocol runs without an environment or an LLM.
"""

import asyncio
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindforge.env.comm_budget import CommBudgetLedger
from orchestrator import hmas2 as oh
from orchestrator import hub as ohub
from orchestrator import prompt as oprompt
from orchestrator.config import (
    ASSIGNED_OBJECTIVE_VARIANTS,
    CONTROLLER_VARIANTS,
    HUB_VARIANTS,
    PINNED_TASK_VARIANTS,
    OrchestratorConfig,
)
from orchestrator.logging import OrchestratorLogger
from orchestrator.state import OrchestratorState

REPO = Path(__file__).resolve().parents[1]
HPC = REPO / "hpc" / "daic" / "experiments"
LIVING = ["agent_0", "agent_1", "agent_2"]
FAKE_TRACK = {"m1_move_5": "ch1_solo", "m8_anvil_A1": "ch2_anvils"}
#: sha256 of the villager decompose template on `submission` (LF): hmas2
#: must leave villager byte-identical.
VILLAGER_DECOMPOSE_SHA = (
    "6c3d348812f023157b697cfd5d220a5f4789863f5a4b497ece7127ef68e00c5f")


# ── Fixtures and fakes ───────────────────────────────────────────────────

class _Msg:
    def __init__(self, content=None, source=None, **_kw):
        self.content = content
        self.source = source


@pytest.fixture
def stub_autogen(monkeypatch):
    """autogen_core with CancellationToken + System/User/AssistantMessage
    (the repo's other stubs lack AssistantMessage)."""
    core = sys.modules.get("autogen_core")
    models = sys.modules.get("autogen_core.models")
    if core is None or models is None:
        core = types.ModuleType("autogen_core")
        models = types.ModuleType("autogen_core.models")
        core.models = models
        monkeypatch.setitem(sys.modules, "autogen_core", core)
        monkeypatch.setitem(sys.modules, "autogen_core.models", models)
    if not hasattr(core, "CancellationToken"):
        monkeypatch.setattr(core, "CancellationToken",
                            type("CancellationToken", (), {}), raising=False)
    for name in ("SystemMessage", "UserMessage", "AssistantMessage"):
        if not hasattr(models, name):
            monkeypatch.setattr(models, name, type(name, (_Msg,), {}),
                                raising=False)
    yield


class _FakeClient:
    """Pops canned raw responses; records each call's message list."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, messages, cancellation_token=None, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(
            content=self._responses.pop(0),
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )

    def roles(self, i=0):
        return [type(m).__name__ for m in self.calls[i]]

    def text(self, i=0, j=-1):
        return self.calls[i][j].content


def _cfg(**kw):
    base = dict(enabled=True, variant="hmas2")
    base.update(kw)
    return OrchestratorConfig(**base)


def _words(text) -> int:
    return len(str(text).split())


LOCAL = {
    a: {"chamber": "ch2", "position": f"({i}, 5, {i})",
        "observe": f"SECRET_{a} anvil A1 hp 4", "holding": "nothing",
        "health": "20/20", "last_check": "not checked yet"}
    for i, a in enumerate(LIVING)
}


def _ctrl(plans=(), checks=(), tmp_path=None, cfg=None, validator=None):
    logger = OrchestratorLogger(str(tmp_path)) if tmp_path else None
    return oh.HMAS2Controller(
        cfg or _cfg(), 3,
        planner_client=_FakeClient(list(plans)),
        check_client=_FakeClient(list(checks)),
        orch_logger=logger, milestone_track=FAKE_TRACK,
        chamber_describe=lambda ch, n: f"{ch}: how it works",
        task_validator=validator or (lambda task, ch: "INVALID" not in task),
        token_counter=_words,
    )


def _tick(ctrl, *, t=1, living=LIVING, state=None, comm_budget=None):
    return asyncio.run(ctrl.tick(
        state=state or OrchestratorState(), living_agents=living,
        episode=1, t=t, local_state_fn=lambda a: dict(LOCAL[a]),
        comm_budget=comm_budget,
        env_state={"doors": {"door1": True}, "anvils": [],
                   "cell_doors_open": []},
        agent_milestones={}))


def plan(**entries):
    return json.dumps({a: {"task": t, "message": m}
                       for a, (t, m) in entries.items()})


ALL = dict(agent_0=("Dig anvil A1", ""), agent_1=("Dig anvil A1", "help agent_0"),
           agent_2=("Explore the chamber", ""))


# ── Config / variant groups / CLI ───────────────────────────────────────

def test_config_defaults_match_the_hmas2_reference_code():
    cfg = _cfg()
    cfg.validate()
    assert (cfg.hmas2_max_rounds, cfg.hmas2_syntax_retries,
            cfg.hmas2_history_tokens) == (3, 6, 3000)
    assert "hmas2" in OrchestratorConfig.VALID_VARIANTS
    assert "star" not in OrchestratorConfig.VALID_VARIANTS
    assert CONTROLLER_VARIANTS == ("villager", "hmas2")
    assert HUB_VARIANTS == PINNED_TASK_VARIANTS == ("hmas2",)
    assert ASSIGNED_OBJECTIVE_VARIANTS == ("villager",)
    with pytest.raises(ValueError):
        _cfg(mode="bias").validate()
    with pytest.raises(ValueError):
        _cfg(hmas2_max_rounds=0).validate()


def _args(monkeypatch, *extra):
    from mindforge import cli
    monkeypatch.setattr(sys, "argv", ["prog", "--orchestrator",
                                      "--orchestrator-variant", "hmas2",
                                      *extra])
    return cli, cli.parse_args()


def test_cli_hmas2_defaults_and_exclusions(monkeypatch):
    cli, args = _args(monkeypatch)
    assert (args.orchestrator_hmas2_max_rounds,
            args.orchestrator_hmas2_syntax_retries,
            args.orchestrator_hmas2_history_tokens,
            args.orchestrator_hmas2_message_words,
            args.orchestrator_hmas2_report_cap,
            args.orchestrator_hmas2_check_max_tokens) == (3, 6, 3000, 24, 2, 96)
    cli.validate_args(args)
    cli, args = _args(monkeypatch, "--comm-budget-tokens", "600")
    cli.validate_args(args)
    for extra in (["--orchestrator-mode", "bias"], ["--no-communication"],
                  ["--rl"], ["--no-simultaneous"], ["--hebbian"]):
        cli, args = _args(monkeypatch, *extra)
        with pytest.raises(SystemExit):
            cli.validate_args(args)


def test_main_loop_has_no_literal_variant_gates():
    src = (REPO / "src" / "mindforge" / "multi_agent_craftium.py").read_text(
        encoding="utf-8")
    for literal in ('variant == "villager"', 'variant != "villager"',
                    '("plan", "villager")', "VILLAGER_FAMILY", "_star_hub",
                    "star_prompts", 'variant == "hmas2"'):
        assert literal not in src, literal
    for name in ("CONTROLLER_VARIANTS", "HUB_VARIANTS",
                 "PINNED_TASK_VARIANTS", "ASSIGNED_OBJECTIVE_VARIANTS"):
        assert name in src, name


def test_villager_is_untouched():
    base = oprompt._orchestrator_decompose_prompt
    assert "{team_reports}" not in base
    assert hashlib.sha256(base.encode()).hexdigest() == VILLAGER_DECOMPOSE_SHA


# ── Paper fidelity ──────────────────────────────────────────────────────

CENTRAL_VERBATIM = [
    "You are a central planner directing agents in",
    "Actions are like:",
    "Your task is to instruct each agent to",
    "After each move, agents provide updates for the next sequence of "
    "actions. Your job is to coordinate the agents optimally.",
    "The previous state and action pairs at each step are:",
    "Please learn from previous steps. Not purely repeat the actions but "
    "learn why the state changes or remains in a dead loop. Avoid being "
    "stuck in action loops.",
    "Hence, the current state is",
    ", with the possible actions:",
    "Specify your action plan in this format:",
    "Include an agent only if it has a task next.",
    "Now, plan the next step:",
]
LOCAL_VERBATIM = [
    "in a multi-agent system",
    "A central planner coordinates all agents to achieve the goal:",
    "The current state and possible actions of yourself are: {",
    "The previous state and action pairs at each step are:",
    "Please learn from previous steps. Not purely repeat the actions but "
    "learn why the state changes or remains in a dead loop. Avoid being "
    "stuck in action loops.",
    "The central planner's current action plan",
    "If you agree with it, respond 'I Agree', without any extra words. If "
    "not, briefly explain your objections to the central planner. Your "
    "response:",
]


def _in_order(text, parts):
    pos = 0
    for part in parts:
        idx = text.find(part, pos)
        assert idx >= 0, f"missing or out of order: {part[:50]!r}"
        pos = idx + len(part)


def test_prompts_keep_the_hmas2_sentences_in_order():
    ctrl = _ctrl()
    central = ctrl.build_central_prompt(
        living=LIVING, local=LOCAL, env_state={}, completed=set(),
        budget_left=None)
    _in_order(central, CENTRAL_VERBATIM)
    local = ctrl.build_local_prompt("agent_0", LOCAL["agent_0"], "Dig A1")
    _in_order(local, LOCAL_VERBATIM)
    # The sentences the adaptation removes from the worker's prompt.
    assert "The current states and possible actions of all other agents" \
        not in local
    assert "The current state is" not in local
    # Re-prompt strings and loop constants, verbatim.
    assert oh.SYSTEM_PROMPT == "You are a helpful assistant."
    assert ("This is the feedback from local agents. If you find some errors "
            "in your previous plan, try to modify it. Otherwise, output the "
            "same plan as before. The output should have the same json "
            "format") in oh.FEEDBACK_REPROMPT
    assert ("as above. Do not explain, just directly output json directory. "
            "Your response:") in oh.FEEDBACK_REPROMPT
    assert oh.SYNTAX_BAD_ENTRY.endswith("is not in the doable action list; ")
    assert oh.SYNTAX_REPLAN == ("Please replan for all the agents again with "
                                "the same ouput format:")
    assert oh.SYNTAX_BAD_JSON.startswith(
        "Your assigned plan is not in the correct json format as before.")
    assert oh.is_agreement("I Agree") and oh.is_agreement("ok, I agree.")
    assert not oh.is_agreement("No: the anvil is behind a wall")


# ── The protocol ────────────────────────────────────────────────────────

def test_all_agree_is_one_round(stub_autogen):
    ctrl = _ctrl(plans=[plan(**ALL)], checks=["I Agree"] * 3)
    result = _tick(ctrl)
    assert ctrl.planner_client.roles() == ["SystemMessage", "UserMessage"]
    assert ctrl.planner_client.text(0, 0) == "You are a helpful assistant."
    assert len(ctrl.check_client.calls) == 3
    rec = result.hub["record"]
    assert rec["rounds"] == 1 and rec["objections"] == []
    assert sorted(result.reassigned) == LIVING
    assert ctrl.pinned_task("agent_2") == "Explore the chamber"
    assert result.hub["messages"] == {"agent_1": "help agent_0"}
    assert ctrl.assigned_objective("agent_0") == ""


def test_objection_revises_with_the_verbatim_feedback(stub_autogen):
    revised = plan(agent_0=("Move to anvil A1 then dig it", ""))
    ctrl = _ctrl(plans=[plan(**ALL), revised],
                 checks=["I cannot reach anvil A1 from here", "I Agree",
                         "I Agree", "I Agree", "I Agree", "I Agree"])
    result = _tick(ctrl)
    rec = result.hub["record"]
    assert rec["rounds"] == 2
    assert rec["objections"][0]["agent"] == "agent_0"
    assert ctrl.planner_client.roles(1) == [
        "SystemMessage", "UserMessage", "AssistantMessage", "UserMessage"]
    previous = json.loads(ctrl.planner_client.text(1, 2))
    assert previous["agent_0"]["task"] == "Dig anvil A1"
    feedback = ctrl.planner_client.text(1, 3)
    assert feedback.startswith(
        "agent_0: I cannot reach anvil A1 from here\n")
    assert oh.FEEDBACK_REPROMPT.format(
        example=oh.PLAN_FORMAT_EXAMPLE) in feedback
    # The revision only names agent_0; the others keep their entry, and
    # round 2 re-checks the whole merged plan (as HMAS-2 re-checks the
    # whole revised plan).
    assert len(ctrl.check_client.calls) == 6
    assert ctrl.pinned_task("agent_0") == "Move to anvil A1 then dig it"
    assert ctrl.pinned_task("agent_1") == "Dig anvil A1"
    assert result.hub["messages"] == {"agent_1": "help agent_0"}


def test_three_round_cap(stub_autogen):
    p = plan(agent_0=("Dig anvil A1", ""), agent_1=("Dig anvil A1", ""),
             agent_2=("Explore the chamber", ""))
    ctrl = _ctrl(plans=[p] * 4,
                 checks=["No, too far", "I Agree", "I Agree"] * 3)
    result = _tick(ctrl)
    assert result.hub["record"]["rounds"] == 3
    assert len(ctrl.planner_client.calls) == 4      # plan + 3 revisions
    assert len(ctrl.check_client.calls) == 9


def test_syntax_reprompt_uses_the_hmas2_wording(stub_autogen):
    bad = plan(agent_0=("Dig anvil A1", ""), agent_1=("INVALID hover", ""),
               agent_2=("Explore", ""))
    ctrl = _ctrl(plans=[bad, plan(**ALL)], checks=["I Agree"] * 3)
    result = _tick(ctrl)
    assert ctrl.planner_client.roles(1) == [
        "SystemMessage", "UserMessage", "AssistantMessage", "UserMessage"]
    assert ctrl.planner_client.text(1, 3) == (
        "Your assigned task for agent_1 is not in the doable action list; "
        "Please replan for all the agents again with the same ouput format:")
    assert result.hub["record"]["syntax_reprompts"] == 1
    assert result.hub["record"]["syntax_ok"] is True


def test_syntax_failure_falls_back_to_the_valid_entries(stub_autogen):
    bad = plan(agent_0=("Dig anvil A1", ""), agent_1=("INVALID hover", ""),
               agent_2=("Explore", ""))
    ctrl = _ctrl(plans=[bad] * 3, checks=["I Agree"] * 2,
                 cfg=_cfg(hmas2_syntax_retries=2))
    result = _tick(ctrl)
    assert len(ctrl.planner_client.calls) == 3
    assert result.hub["record"]["syntax_ok"] is False
    assert ctrl.stats["syntax_fallbacks"] == 1
    assert ctrl.pinned_task("agent_0") == "Dig anvil A1"
    assert ctrl.pinned_task("agent_1") is None


def test_bad_json_and_missing_agents_are_syntax_errors(stub_autogen):
    ctrl = _ctrl(plans=["I will think about it",
                        plan(agent_0=("Dig anvil A1", "")),
                        plan(**ALL)],
                 checks=["I Agree"] * 3)
    _tick(ctrl)
    assert ctrl.planner_client.text(1, 3) == (
        oh.SYNTAX_BAD_JSON + oh.SYNTAX_REPLAN)
    assert ctrl.planner_client.text(2, 5) == (
        "Your assigned task for agent_1 is missing; "
        "Your assigned task for agent_2 is missing; " + oh.SYNTAX_REPLAN)


def test_only_agents_in_the_plan_are_checked_and_reassigned(stub_autogen):
    ctrl = _ctrl(plans=[plan(**ALL),
                        json.dumps({"agent_1": {"task": "Attack the zombie",
                                                "message": ""},
                                    "agent_2": {"task": "",
                                                "message": "agent_0 is at A1"}})],
                 checks=["I Agree"] * 5)
    _tick(ctrl, t=1)
    result = _tick(ctrl, t=2)
    assert result.reassigned == ["agent_1"]
    assert result.hub["record"]["checked"] == ["agent_1", "agent_2"]
    assert ctrl.pinned_task("agent_0") == "Dig anvil A1"           # kept
    assert ctrl.pinned_task("agent_2") == "Explore the chamber"    # msg only
    assert result.hub["messages"] == {"agent_2": "agent_0 is at A1"}


def test_workers_see_only_their_own_information(stub_autogen):
    tasks = dict(agent_0=("Dig anvil A1", "partner arrives soon"),
                 agent_1=("Collect wood", "TOPSECRET relay for one"),
                 agent_2=("Explore the chamber", ""))
    ctrl = _ctrl(plans=[plan(**tasks)], checks=["I Agree"] * 3)
    _tick(ctrl)
    check_0 = ctrl.check_client.text(0, 1)
    assert "SECRET_agent_0" in check_0 and "Dig anvil A1" in check_0
    for leak in ("SECRET_agent_1", "SECRET_agent_2", "Collect wood",
                 "Explore the chamber", "TOPSECRET", "partner arrives soon",
                 "agent_1", "agent_2"):
        assert leak not in check_0, leak
    # The planner sees the whole team.
    central = ctrl.planner_client.text(0, 1)
    assert "SECRET_agent_1" in central and "SECRET_agent_2" in central
    # The directive names the agent's own task only.
    directive = ctrl.directive_text("agent_0")
    assert "Dig anvil A1" in directive and "agent_1" not in directive


def test_budget_exhausted_agents_are_not_checked_and_objections_cost(
        stub_autogen):
    ledger = CommBudgetLedger([0, 1, 2], 20, msg_cap=8, token_counter=_words,
                              topology="hub")
    ledger.state(1).spent = 20                        # agent_1 exhausted
    ctrl = _ctrl(plans=[plan(**ALL), plan(**ALL)],
                 checks=["The anvil is two rooms away", "I Agree",
                         "I Agree", "I Agree"])
    result = _tick(ctrl, comm_budget=ledger)
    assert "agent_1" not in result.hub["record"]["checked"]
    assert ledger.state(0).spent == 6                 # the objection
    central = ctrl.planner_client.text(0, 1)
    assert "agent_0=20, agent_1=0, agent_2=20" in central


def test_zero_budget_reduces_to_cmas(stub_autogen):
    ledger = CommBudgetLedger([0, 1, 2], 0, token_counter=_words,
                              topology="hub")
    ctrl = _ctrl(plans=[plan(**ALL)])
    result = _tick(ctrl, comm_budget=ledger)
    assert ctrl.check_client.calls == []
    inboxes = {0: [], 1: ["old"], 2: []}
    delivery = ohub.deliver_hub_messages(
        inboxes, result.hub["messages"], living=LIVING, step=1,
        make_message=lambda c: c, comm_budget=ledger)
    assert delivery["agent_1"]["status"] == "blocked"
    assert inboxes[1] == ["old"]
    assert sorted(result.reassigned) == LIVING     # assignments still apply


def test_history_is_state_action_pairs_within_budget(stub_autogen):
    ctrl = _ctrl(plans=[plan(**ALL), plan(**ALL), plan(**ALL)],
                 checks=["I Agree"] * 9)
    for t in (1, 2, 3):
        _tick(ctrl, t=t)
    third = ctrl.planner_client.text(2, 1)
    assert "State1: " in third and "Action2: " in third
    assert oh.format_history([("s", "a")] * 5, _words, 1) == ""
    newest = oh.format_history([("old", "x"), ("new", "y")], _words, 5)
    assert "new" in newest and "old" not in newest


def test_event_buffer_is_drained_and_team_wipe_skips(stub_autogen):
    state = OrchestratorState()
    state.add_event({"type": "milestone", "t": 1, "id": "m8_anvil_A1",
                     "contributors": ["agent_0"]})
    ctrl = _ctrl(plans=[plan(**ALL)], checks=["I Agree"] * 3)
    _tick(ctrl, state=state)
    assert state.event_buffer == []
    assert "m8_anvil_A1 by agent_0" in ctrl.planner_client.text(0, 1)
    state.add_event({"type": "death", "t": 2, "agent": "agent_2"})
    result = _tick(ctrl, state=state, living=[])
    assert result.hub is None and state.event_buffer == []
    assert len(ctrl.planner_client.calls) == 1


def test_reports_reach_the_planner_and_logs(stub_autogen, tmp_path):
    ctrl = _ctrl(plans=[plan(**ALL)], checks=["I Agree"] * 3,
                 tmp_path=tmp_path)
    ctrl.receive_report(t=0, sender="agent_2", text="door 2 is open",
                        chamber="ch2")
    result = _tick(ctrl)
    assert 't=0, agent_2 [in ch2]: "door 2 is open"' in \
        ctrl.planner_client.text(0, 1)
    assert ctrl.reports.count() == 0                   # consumed
    inboxes = {0: [], 1: [], 2: []}
    delivery = ohub.deliver_hub_messages(
        inboxes, result.hub["messages"], living=LIVING, step=1,
        make_message=lambda c: SimpleNamespace(content=c, source="orchestrator"))
    ctrl.record_delivery(result.hub, delivery)
    rows = [json.loads(line) for line in
            (tmp_path / "orchestrator" / "hmas2.jsonl").read_text()
            .splitlines()]
    for key in ("episode", "t", "rounds", "syntax_reprompts", "checked",
                "objections", "plan", "reassigned", "assignments",
                "reports_in", "tokens", "latency_s", "messages"):
        assert key in rows[0], key
    assert rows[0]["messages"]["agent_1"]["delivered"] is True
    calls = [json.loads(line)["call_type"] for line in
             (tmp_path / "orchestrator" / "calls.jsonl").read_text()
             .splitlines()]
    assert calls == ["plan", "check", "check", "check"]
    stats = ctrl.episode_stats()
    assert stats["steps"] == 1 and stats["messages_delivered"] == 1
    ctrl.reset()
    assert ctrl.assignments == {} and ctrl.history == []


# ── Hub plumbing ────────────────────────────────────────────────────────

def test_reports_cap_and_drops():
    reports = ohub.HubReports(LIVING, cap=2)
    for t in (1, 2, 3):
        reports.receive(t=t, sender="agent2", text=f"r{t}", chamber="ch2")
    assert [(r[0], r[1]) for r in reports.pending()] == [(2, "agent_2"),
                                                         (3, "agent_2")]
    assert reports.reports_dropped == 1 and reports.reports_in == 3
    reports.receive(t=4, sender="agent_1", text="   ")
    assert reports.reports_in == 3


def test_sanitize_and_deliver_replace():
    assert ohub.sanitize_hub_text("go {north} }", 40) == "go (north) )"
    assert ohub.sanitize_hub_text("a b c", 2) == "a b"
    inboxes = {0: ["x", "y"], 1: ["z"], 2: ["w"]}
    out = ohub.deliver_hub_messages(
        inboxes, {"agent_0": "go {north}", "agent_2": "dead"},
        living=["agent_0", "agent_1"], step=1, make_message=lambda c: c)
    assert inboxes[0] == ["go (north)"] and inboxes[1] == ["z"]
    assert inboxes[2] == ["w"] and out["agent_2"]["status"] == "dead"


def test_recipient_pays_truncates_then_blocks():
    ledger = CommBudgetLedger([0, 1, 2], 5, msg_cap=4, token_counter=_words,
                              topology="hub")
    inboxes = {0: [], 1: [], 2: []}
    out = ohub.deliver_hub_messages(
        inboxes, {"agent_0": "one two three four five six"}, living=LIVING,
        step=1, make_message=lambda c: c, comm_budget=ledger)
    assert out["agent_0"]["status"] == "truncated"
    assert inboxes[0] == ["one two three four"]
    st = ledger.state(0)
    assert (st.received, st.received_tokens, st.sent) == (1, 4, 0)
    ohub.deliver_hub_messages(inboxes, {"agent_0": "go now"}, living=LIVING,
                              step=2, make_message=lambda c: c,
                              comm_budget=ledger)
    out = ohub.deliver_hub_messages(inboxes, {"agent_0": "more"},
                                    living=LIVING, step=3,
                                    make_message=lambda c: c,
                                    comm_budget=ledger)
    assert out["agent_0"]["status"] == "blocked" and inboxes[0] == ["go"]


def test_ledger_topology_summary_and_action_line():
    peer = CommBudgetLedger([0], 100, token_counter=_words)
    assert "received" not in peer.summary()["per_agent"]["agent_0"]
    hub = CommBudgetLedger([0], 100, token_counter=_words, topology="hub")
    assert hub.summary()["per_agent"]["agent_0"]["received"] == 0
    assert "orchestrator" in hub.render_action_line(0, 50)
    assert "orchestrator" not in peer.render_action_line(0, 50)
    with pytest.raises(ValueError):
        CommBudgetLedger([0], 10, topology="star")


# ── Worker: pinned curriculum ───────────────────────────────────────────

def test_pinned_task_is_adopted_and_success_counted_once():
    import social_stubs  # noqa: F401
    from mindforge.custom_agent import CustomAgent

    class _Curriculum:
        def __init__(self):
            self.current_task = None
            self.completed_tasks = []
            self.saved = 0

        def save_context(self, _ctx):
            self.saved += 1

        async def adopt_task(self, task, frame, token, communications=None):
            self.current_task = task
            return task, f"context for {task}"

    agent = object.__new__(CustomAgent)
    agent.auto_curriculum = _Curriculum()
    agent.belief_system = SimpleNamespace(task_beliefs="")
    agent.metric = SimpleNamespace(log=lambda *_a, **_k: None)
    agent._pin_success_logged = False
    try:
        agent.name = "agent_0"
    except AttributeError:          # autogen's read-only property
        agent._name = "agent_0"

    def apply(pin, success):
        return asyncio.run(agent._apply_pinned_task(pin, success, None, None,
                                                    []))

    assert apply("Dig anvil A1", None) is True
    assert agent.auto_curriculum.current_task == "Dig anvil A1"
    assert agent.belief_system.task_beliefs == "context for Dig anvil A1"
    assert apply("Dig anvil A1", True) is False
    assert apply("Dig anvil A1", True) is False      # cached verdict repeats
    assert agent.auto_curriculum.completed_tasks == ["Dig anvil A1"]
    assert apply("Explore the chamber", True) is True
    assert agent.auto_curriculum.completed_tasks == ["Dig anvil A1"]
    assert apply("Explore the chamber", True) is False
    assert agent.auto_curriculum.completed_tasks == [
        "Dig anvil A1", "Explore the chamber"]


# ── Launchers ───────────────────────────────────────────────────────────

def test_launchers_support_hmas2():
    budget = (HPC / "budget_gemma.sbatch").read_text(encoding="utf-8")
    assert "base|hebbian|villager|hmas2)" in budget
    assert '--orchestrator-variant "$ARM"' in budget
    assert 'EXP_NAME="budget_gemma_${ARM}_n${NUM_AGENTS}_b${COMM_BUDGET}"' \
        in budget
    assert "star" not in budget.lower().replace("start", "")
    scale = (HPC / "scale_gemma_orch.sbatch").read_text(encoding="utf-8")
    assert "task|social|plan|villager|hmas2)" in scale
    sub = (HPC / "submit_agent_scaling_orch.sh").read_text(encoding="utf-8")
    assert "VARIANT_LIST=(${VARIANTS:-hmas2})" in sub
    assert 'if [ -z "$jobid" ]; then' in sub
    new = (HPC / "new_exp_orchestrator.sbatch").read_text(encoding="utf-8")
    assert "task|social|plan|villager|hmas2)" in new
    submit = (HPC / "submit_orchestrator.sh").read_text(encoding="utf-8")
    assert "hmas2" in submit and "96:00:00" in submit
    comm = (HPC / "submit_comm_budget.sh").read_text(encoding="utf-8")
    assert "STAR_B0" not in comm and "hmas2" in comm
