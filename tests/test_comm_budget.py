"""Communication-budget sweep (mindforge/env/comm_budget.py).

Pins four contracts:

1. LEDGER ACCOUNTING — charge = tokenized length; per-message cap cuts at a
   word boundary; a message longer than what is left is cut to the remainder
   and LOCKS the agent; every later attempt is BLOCKED; a zero budget blocks
   the first attempt; budgets are per agent; reset restores everything.
2. PROMPT NON-REGRESSION — with the switch off, every prompt file renders the
   ORIGINAL pre-placeholder bytes (the expected strings below are verbatim
   copies of the old literals, same discipline as test_team_scaling.py); with
   the switch on, the comm rule is optional + budgeted and no static
   placeholder survives. The per-step ``{comm_budget}`` line carries its own
   leading newline so "" (every legacy run) adds nothing.
3. MODULE WIRING — ActionSelection resolves the static placeholders at
   construction; SocialModule refreshes its lock even on interval-skipped
   calls and renders a "budget exhausted" directive instead of "ask X".
4. LAUNCHER SPEC — the sbatch passes the budget flags + comm reward scale 0
   on every cell; the submit script's default ladder is the tokenizer-pinned
   0/600/2600/10400 and a refused sbatch is reported as FAILED, not queued.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

from mindforge.env.comm_budget import (
    COMM_FIELD_HINT_BUDGET,
    COMM_FIELD_HINT_LEGACY,
    COMM_RULE_LEGACY,
    COMM_TARGET_RULE_BUDGET,
    COMM_TARGET_RULE_LEGACY,
    DEFAULT_MSG_CAP,
    ENV_MSG_CAP,
    ENV_SWITCH,
    STATUS_BLOCKED,
    STATUS_SENT,
    STATUS_TRUNCATED,
    CommBudgetLedger,
    apply_comm_budget_static,
    apply_comm_budget_to_prompts,
    comm_budget_enabled,
    fallback_token_count,
    make_token_counter,
    model_token_count,
    msg_cap_from_env,
    set_env_switch,
)

REPO = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO / "src" / "mindforge" / "prompts"
HPC = REPO / "hpc" / "daic" / "experiments"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ENV_SWITCH, raising=False)
    monkeypatch.delenv(ENV_MSG_CAP, raising=False)


def _words(text) -> int:
    """Stub tokenizer: one token per whitespace word."""
    return len(str(text).split())


def _ledger(budget, cap=DEFAULT_MSG_CAP, agents=(0, 1, 2)):
    return CommBudgetLedger(agents, budget, msg_cap=cap, token_counter=_words)


# ── 1. ledger accounting ─────────────────────────────────────────────────

def test_charge_basic_accounting_is_per_agent():
    L = _ledger(10)
    r = L.charge(0, "a b c", step=3)
    assert r.status == STATUS_SENT and r.text == "a b c"
    assert r.tokens_model == 3 and r.charged == 3 and r.budget_left == 7
    assert not r.exhausted_now
    assert L.left(0) == 7 and L.state(0).sent == 1 and not L.is_locked(0)
    # Teammates are untouched.
    assert L.left(1) == 10 and L.left(2) == 10


def test_per_message_cap_cuts_at_word_boundary():
    L = _ledger(100, cap=4)
    r = L.charge(0, "one two three four five six", step=1)
    assert r.status == STATUS_TRUNCATED
    assert r.text == "one two three four"
    assert r.tokens_model == 6 and r.charged == 4 and r.budget_left == 96
    assert L.state(0).truncated == 1 and L.state(0).sent == 1


def test_remainder_truncation_then_lock_then_block():
    L = _ledger(5)
    assert L.charge(0, "a b c", step=1).budget_left == 2
    r = L.charge(0, "d e f g", step=2)
    assert r.status == STATUS_TRUNCATED and r.text == "d e"
    assert r.charged == 2 and r.budget_left == 0 and r.exhausted_now
    assert L.is_locked(0) and L.state(0).exhausted_step == 2
    r3 = L.charge(0, "h", step=3)
    assert r3.status == STATUS_BLOCKED and r3.text == "" and r3.charged == 0
    assert not r3.exhausted_now                 # only the FIRST lock reports
    assert L.state(0).blocked == 1 and L.state(0).sent == 2


def test_exact_spend_reports_exhausted_now():
    L = _ledger(3)
    r = L.charge(0, "a b c", step=4)
    assert r.status == STATUS_SENT and r.exhausted_now
    assert L.state(0).exhausted_step == 4 and L.is_locked(0)


def test_zero_budget_blocks_first_attempt():
    L = _ledger(0)
    assert L.is_locked(0)
    r = L.charge(0, "hello there", step=7)
    assert r.status == STATUS_BLOCKED and r.exhausted_now
    assert L.state(0).exhausted_step == 7
    r2 = L.charge(0, "again", step=8)
    assert r2.status == STATUS_BLOCKED and not r2.exhausted_now
    assert L.state(0).blocked == 2 and L.state(0).sent == 0
    s = L.summary()
    assert s["budget_tokens"] == 0 and s["msg_cap"] == DEFAULT_MSG_CAP
    assert s["per_agent"]["agent_0"]["blocked"] == 2
    assert s["per_agent"]["agent_1"]["exhausted_step"] is None


def test_unusable_remainder_locks_instead_of_sending_nothing():
    two_per_word = lambda t: 2 * len(str(t).split())  # noqa: E731
    L = CommBudgetLedger([0], 3, msg_cap=32, token_counter=two_per_word)
    assert L.charge(0, "a", step=1).budget_left == 1
    r = L.charge(0, "b c", step=2)       # not even one word fits in 1 token
    assert r.status == STATUS_BLOCKED and r.exhausted_now and r.budget_left == 0
    assert L.is_locked(0) and L.state(0).exhausted_step == 2


def test_reset_restores_fresh_budgets():
    L = _ledger(4)
    L.charge(0, "a b c d", step=1)
    assert L.is_locked(0)
    L.reset()
    assert L.left(0) == 4 and not L.is_locked(0)
    assert L.state(0).exhausted_step is None and L.state(0).sent == 0


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        CommBudgetLedger([0], -1)
    with pytest.raises(ValueError):
        CommBudgetLedger([0], 10, msg_cap=0)


def test_fallback_token_count():
    assert fallback_token_count("") == 0
    assert fallback_token_count(None) == 0
    assert fallback_token_count("one two three") == 4    # ceil(3 x 1.3)


def test_make_token_counter_prefers_model_then_falls_back():
    assert make_token_counter(lambda t: 7)("x y z") == 7
    assert make_token_counter(lambda t: None)("x y z") == 4


def test_model_token_count_without_local_model_is_none(monkeypatch):
    name = "mindforge.agent_modules.local_model_client"
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert model_token_count("hello") is None
    fake = types.ModuleType(name)
    fake.count_text_tokens = lambda t: 11
    monkeypatch.setitem(sys.modules, name, fake)
    assert model_token_count("hello") == 11
    assert make_token_counter()("hello") == 11


# ── 2. prompt non-regression ─────────────────────────────────────────────

# Verbatim copies of the pre-placeholder literals — do NOT "fix" them here
# without also changing the *_LEGACY constants, or legacy runs would
# silently get different prompts.
ORIG_TARGET_LINE = (
    'YOU ARE: {agent_name}. Your teammates are: {teammate_names}. '
    '"communication_target" must be EXACTLY one of {teammate_names} — never '
    '"{agent_name}", never "all", never empty.'
)
ORIG_FIELD_HINT = (
    '"communication": "<short message to the teammate in communication_target '
    '— an observation, request, or commitment relevant to THEM; never empty>"'
)
ORIG_RULE_HEAD = (
    "TARGETED COMMUNICATION (REQUIRED EVERY STEP — there is NO broadcast channel):"
)
ORIG_RULE_TAIL = (
    "  same teammate are filtered as spam and never arrive.\n\n"
    "Your per-step context and the exact JSON response format"
)


def _file(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _legacy_render(name: str) -> str:
    return apply_comm_budget_static(_file(name), enabled=False).replace(
        "{comm_budget}", "")


def test_legacy_render_restores_original_instruction_prompts():
    for name in ("instruction_prompt_p2.txt", "instruction_prompt_p2_thoughts.txt"):
        txt = _legacy_render(name)
        assert txt.split("\n")[0] == ORIG_TARGET_LINE, name
        assert ORIG_FIELD_HINT in txt, name
        assert "\n{milestone_event}\n" in txt, name
        assert "{comm_" not in txt, name
    choice = _legacy_render("instruction_prompt_p2_choice.txt")
    assert "\n{milestone_event}\n" in choice and "{comm_" not in choice


def test_legacy_render_restores_original_system_rule():
    txt = _legacy_render("system_prompt.txt")
    assert COMM_RULE_LEGACY in txt
    assert ORIG_RULE_HEAD in txt and ORIG_RULE_TAIL in txt
    assert "{comm_rule}" not in txt
    assert txt.count("TARGETED COMMUNICATION") == 1


def test_legacy_render_restores_original_social_prompts():
    for name in ("social_module.txt", "social_module_choice.txt"):
        txt = _legacy_render(name)
        assert "{incoming_comms}\n" in txt and "{comm_" not in txt, name


def test_placeholders_sit_where_the_per_step_line_expects():
    for name in ("instruction_prompt_p2.txt", "instruction_prompt_p2_thoughts.txt",
                 "instruction_prompt_p2_choice.txt"):
        assert "{milestone_event}{comm_budget}\n" in _file(name), name
    for name in ("social_module.txt", "social_module_choice.txt"):
        assert "{incoming_comms}{comm_budget}\n" in _file(name), name
    assert "{comm_rule}" in _file("system_prompt.txt")
    assert "social_act" not in COMM_RULE_LEGACY     # keeps test_social_acts' guard


def test_budget_render_is_optional_and_leaves_only_per_step_placeholder():
    sysp = apply_comm_budget_static(_file("system_prompt.txt"), enabled=True,
                                    msg_cap=40)
    assert "OPTIONAL and BUDGETED" in sysp and "REQUIRED EVERY STEP" not in sysp
    assert "cut at 40" in sysp and "{cap}" not in sysp
    assert "{comm_" not in sysp
    for name in ("instruction_prompt_p2.txt", "instruction_prompt_p2_thoughts.txt"):
        txt = apply_comm_budget_static(_file(name), enabled=True)
        assert COMM_TARGET_RULE_BUDGET in txt and COMM_TARGET_RULE_LEGACY not in txt
        assert COMM_FIELD_HINT_BUDGET in txt and COMM_FIELD_HINT_LEGACY not in txt
        assert "{comm_budget}" in txt                # per-step, filled by llm_call
        assert "{comm_rule}" not in txt and "{comm_target_rule}" not in txt
        # JSON escapes untouched (str.replace, not str.format).
        assert '{{"thoughts"' in txt


def test_per_step_line_adds_nothing_when_empty():
    tpl = "{milestone_progress}\n{milestone_event}{comm_budget}\nStatus: x"
    legacy = tpl.format(milestone_progress="p", milestone_event="", comm_budget="")
    assert legacy == "p\n\nStatus: x"
    L = _ledger(100)
    line = L.render_action_line(0, steps_left=400)
    assert line.startswith("\nCommunication budget: 100 of 100 tokens left")
    rendered = tpl.format(milestone_progress="p", milestone_event="", comm_budget=line)
    assert rendered.startswith("p\n\nCommunication budget:")


def test_master_switch_env_roundtrip():
    assert not comm_budget_enabled() and msg_cap_from_env() == DEFAULT_MSG_CAP
    set_env_switch(True, 40)
    assert comm_budget_enabled() and msg_cap_from_env() == 40
    # apply_* read the switch when enabled=None.
    assert "cut at 40" in apply_comm_budget_static("{comm_rule}")
    set_env_switch(False)
    assert not comm_budget_enabled()
    assert apply_comm_budget_static("{comm_rule}") == COMM_RULE_LEGACY


def test_apply_to_prompts_dict_handles_nesting():
    prompts = {"system_template": "A {comm_rule} B", "roles": {"x": "{comm_target_rule}"},
               "n": 3}
    out = apply_comm_budget_to_prompts(prompts, enabled=False)
    assert out["system_template"] == f"A {COMM_RULE_LEGACY} B"
    assert out["roles"]["x"] == COMM_TARGET_RULE_LEGACY and out["n"] == 3


def test_action_and_social_lines_track_state():
    L = _ledger(100)
    L.charge(0, " ".join(["w"] * 4), step=1)               # 96 left
    a = L.render_action_line(0, steps_left=400)
    assert "96 of 100 tokens left" in a and "about 6 short messages" in a
    assert "about 400 steps to go" in a
    s = L.render_social_line(0)
    assert s.startswith("\nCommunication budget for agent_0: 96 of 100")
    L.charge(0, " ".join(["w"] * 200), step=2)             # cap 32 x3 → still >0
    L.charge(0, " ".join(["w"] * 200), step=3)
    L.charge(0, " ".join(["w"] * 200), step=4)             # 96-32-32-32 = 0
    assert L.is_locked(0)
    assert "EXHAUSTED (0 of 100 tokens left)" in L.render_action_line(0, 1)
    assert "EXHAUSTED" in L.render_social_line(0)
    Z = _ledger(0)
    assert "NONE this episode" in Z.render_action_line(0, 999)
    assert "NONE this episode" in Z.render_social_line(0)


# ── 3. module wiring ─────────────────────────────────────────────────────

_THOUGHT = {
    "bond_change_explanation": {}, "reasoning": "r", "referenced_bonds": {},
    "ask_target": "agent_1", "ask_message": "help me dig",
    "respond_to": ["agent_2"], "confidence": 0.5,
}


def _social_module(**kw):
    import social_stubs  # noqa: F401  (autogen/chromadb stand-ins)
    from mindforge.agent_modules.social_module import SocialModule
    return SocialModule(agent_name="agent_0", num_agents=3,
                        social_model_client=object(), **kw)


def test_social_directive_locked_branch():
    m = _social_module()
    open_ = m.render_directive(_THOUGHT)
    assert "Ask agent_1 for help" in open_ and "acknowledge them" in open_
    m._comm_locked = True
    locked = m.render_directive(_THOUGHT)
    assert "Communication budget exhausted" in locked
    assert "Suggested message" not in locked and "communication_target" not in locked
    assert "cannot message them" in locked
    # The parts that do not depend on messaging are unchanged.
    assert "Reasoning: r" in locked and "Bond changes noted" in locked


def test_deliberate_refreshes_lock_on_interval_skipped_call():
    m = _social_module(social_interval=8)
    m.last_thought = dict(_THOUGHT)
    m._call_count = 0                       # next call: 1 % 8 != 0 → cached path
    out = asyncio.run(m.deliberate(
        bond_weights={"agent_1": 0.5}, bond_deltas={}, incoming=[],
        teammate_names="agent_1, agent_2", last_action="NoOp", last_reward="0",
        picked_object="", position_text="", cancellation_token=None,
        comm_budget_text="\nCommunication budget for agent_0: EXHAUSTED",
        comm_budget_locked=True,
    ))
    assert out is m.last_thought and m._comm_locked
    assert "budget exhausted" in m.render_directive()


def test_action_selection_resolves_static_placeholders_at_construction():
    import social_stubs  # noqa: F401
    from mindforge.agent_modules import action_selection as asel
    legacy = asel.ActionSelection(action_model_client=object())
    assert "{comm_rule}" not in legacy.system_prompt
    assert COMM_RULE_LEGACY in legacy.system_prompt
    for tpl in (legacy.user_prompt_template, legacy.thoughts_prompt_template):
        assert COMM_TARGET_RULE_LEGACY in tpl and COMM_FIELD_HINT_LEGACY in tpl
        assert "{comm_target_rule}" not in tpl and "{comm_field_hint}" not in tpl
        assert "{comm_budget}" in tpl              # per-step, stays for llm_call
    set_env_switch(True, 24)
    try:
        budget = asel.ActionSelection(action_model_client=object())
        assert "OPTIONAL and BUDGETED" in budget.system_prompt
        assert "cut at 24" in budget.system_prompt
        for tpl in (budget.user_prompt_template, budget.thoughts_prompt_template):
            assert COMM_FIELD_HINT_BUDGET in tpl and COMM_TARGET_RULE_BUDGET in tpl
        # A caller-supplied template (choice mode) is passed through the same
        # substitution; without static placeholders it is unchanged.
        choice = asel.ActionSelection(action_model_client=object(),
                                      user_prompt_template="X {comm_budget} Y")
        assert choice.user_prompt_template == "X {comm_budget} Y"
    finally:
        set_env_switch(False)


def test_cli_flags_default_to_legacy_and_reject_no_communication(monkeypatch):
    from mindforge import cli
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = cli.parse_args()
    assert args.comm_budget_tokens is None and args.comm_budget_msg_cap == 32
    monkeypatch.setattr(sys, "argv", ["prog", "--comm-budget-tokens", "0"])
    assert cli.parse_args().comm_budget_tokens == 0
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--comm-budget-tokens", "800", "--no-communication"])
    with pytest.raises(SystemExit):
        cli.validate_args(cli.parse_args())


# ── 4. launcher spec ─────────────────────────────────────────────────────

def test_budget_sbatch_passes_the_sweep_flags():
    txt = (HPC / "budget_gemma.sbatch").read_text(encoding="utf-8")
    for flag in ('--comm-budget-tokens "$COMM_BUDGET"',
                 '--comm-budget-msg-cap "$COMM_MSG_CAP"',
                 "--comm-reward-scale 0", "--team-scaling", "--simultaneous",
                 '--ch4-mob-count "$CH4_MOBS"'):
        assert flag in txt, flag
    # Hebbian arm = Hebbian 2.0 (three_factor + signed death LTD), as in
    # scale_gemma_3f.sbatch; base arm passes no Hebbian flag at all.
    for flag in ("--hebbian-mode three_factor", "--hebbian-death-ltd 0.05",
                 "--hebbian-eta-0 0.001", "--hebbian-decay 0.001"):
        assert flag in txt, flag
    assert 'EXP_NAME="budget_gemma_hebbian_n${NUM_AGENTS}_b${COMM_BUDGET}"' in txt
    assert 'EXP_NAME="budget_gemma_base_n${NUM_AGENTS}_b${COMM_BUDGET}"' in txt
    assert ': "${RUN_GROUP:=comm_budget}"' in txt
    assert ': "${MAX_STEPS:=1000}"' in txt and ': "${EPISODES:=3}"' in txt


def test_submit_script_default_ladder_and_pilot_seed():
    txt = (HPC / "submit_comm_budget.sh").read_text(encoding="utf-8")
    assert "NS_LIST=(${NS:-3 5 7})" in txt
    assert "BUDGET_LIST=(${BUDGETS:-0 600 2600 10400})" in txt
    assert 'if [ -z "$jobid" ]; then' in txt   # a refused sbatch is FAILED, not queued
    assert "ARM_LIST=(${ARMS:-base hebbian})" in txt
    assert "SEEDS_LIST=(${SEEDS:-42})" in txt
    assert "budget_gemma.sbatch" in txt
