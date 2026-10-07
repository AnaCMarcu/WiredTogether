"""Hub-topology prompt rendering: rewrites, comm placeholders, Ch3 facts.

Pins that (1) every hub rewrite still matches its prompt file exactly once,
(2) nothing changes when the hub topology is off, and (3) the full prompt
pipeline (hub rewrite -> team scaling -> comm placeholders) leaves no peer
wording and no unresolved placeholder in hub mode, with and without a
budget. Prompt files are read directly (no agent_factory import: its chain
needs autogen/chromadb, which the dev machine does not have).
"""

import string
from pathlib import Path

import pytest

from mindforge.agent_modules import chamber_facts
from mindforge.agent_modules.hub_prompts import (
    HUB_COMM_HEADER,
    HUB_PROMPT_REWRITES,
    PROMPT_KEY_FILES,
    apply_hub_rewrites,
    apply_hub_rewrites_to_prompts,
)
from mindforge.agent_modules.team_scaling import apply_team_scaling_to_prompts
from mindforge.env.comm_budget import (
    COMM_FIELD_HINT_HUB_LEGACY,
    COMM_FIELD_HINT_LEGACY,
    COMM_RULE_LEGACY,
    COMM_TARGET_RULE_HUB_BUDGET,
    COMM_TARGET_RULE_HUB_LEGACY,
    COMM_TARGET_RULE_LEGACY,
    ENV_MSG_CAP,
    ENV_SWITCH,
    ENV_TOPOLOGY,
    apply_comm_budget_static,
    apply_comm_budget_to_prompts,
    comm_topology,
    set_comm_topology,
    set_env_switch,
    static_placeholders,
)

REPO = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO / "src" / "mindforge" / "prompts"
BELIEF_FILES = {"perception_beliefs.txt", "partner_beliefs.txt",
                "interaction_belief.txt"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (ENV_SWITCH, ENV_MSG_CAP, ENV_TOPOLOGY):
        monkeypatch.delenv(var, raising=False)


def _path(file_name: str) -> Path:
    sub = "belief_system" if file_name in BELIEF_FILES else ""
    return PROMPT_DIR / sub / file_name


def _read(file_name: str) -> str:
    # Universal-newline text read, exactly like load_prompts().
    with open(_path(file_name), "r", encoding="utf-8") as f:
        return f.read()


def _prompts() -> dict:
    """The load_prompts() entries the hub rewrites touch, plus an untouched
    one to prove identity elsewhere."""
    out = {key: _read(name) for key, name in PROMPT_KEY_FILES.items()}
    out["environment"] = _read("environment_prompt.txt")
    return out


def _fields(template: str) -> set:
    return {f for _, f, _, _ in string.Formatter().parse(template)
            if f is not None}


# ── 1. drift guard ───────────────────────────────────────────────────────

def test_every_rewrite_source_occurs_exactly_once():
    for file_name, rewrites in HUB_PROMPT_REWRITES.items():
        text = _read(file_name)
        for source, _ in rewrites:
            assert text.count(source) == 1, (file_name, source[:50])


def test_apply_hub_rewrites_raises_on_drift():
    with pytest.raises(RuntimeError):
        apply_hub_rewrites("no such text here", "critic_prompt.txt")
    assert apply_hub_rewrites("untouched", "unknown_file.txt") == "untouched"


# ── 2. identity when off ────────────────────────────────────────────────

def test_rewrites_are_identity_when_disabled():
    prompts = _prompts()
    assert apply_hub_rewrites_to_prompts(prompts, enabled=False) is prompts
    hub = apply_hub_rewrites_to_prompts(prompts, enabled=True)
    assert hub["environment"] == prompts["environment"]
    for key in PROMPT_KEY_FILES:
        assert hub[key] != prompts[key], key


# ── 3. full pipeline in hub mode ────────────────────────────────────────

@pytest.mark.parametrize("team_scaling", [False, True])
@pytest.mark.parametrize("n", [3, 7])
@pytest.mark.parametrize("budget", [False, True])
def test_hub_pipeline_leaves_no_peer_wording(team_scaling, n, budget):
    p = apply_hub_rewrites_to_prompts(_prompts(), enabled=True)
    p = apply_team_scaling_to_prompts(p, n, enabled=team_scaling)
    p = apply_comm_budget_to_prompts(p, enabled=budget, msg_cap=32,
                                     topology="hub")
    system = p["system_template"]
    assert "{comm_" not in system and "HUB COMMUNICATION" in system
    assert "EXACTLY ONE teammate's name" not in system
    assert "from the orchestrator's messages" in system
    assert "report to the orchestrator" in p["critic"]
    assert "message a teammate" not in p["critic"]
    assert "Message from the orchestrator" in p["perception"]
    assert "exchanged messages" not in p["partner"]
    assert "exchanged messages" not in p["interaction"]

    legacy_instr = apply_comm_budget_static(
        _read("instruction_prompt_p2.txt"), enabled=False, topology="peer")
    instr = apply_comm_budget_static(
        apply_hub_rewrites(_read("instruction_prompt_p2.txt"),
                           "instruction_prompt_p2.txt"),
        enabled=budget, msg_cap=32, topology="hub")
    assert "{comm_target_rule}" not in instr
    assert "{comm_field_hint}" not in instr
    assert '"communication_target": "orchestrator"' in instr
    assert "teammate in communication_target" not in instr
    assert "never \"{agent_name}\"" not in instr
    assert (COMM_TARGET_RULE_HUB_BUDGET if budget
            else COMM_TARGET_RULE_HUB_LEGACY) in instr
    assert '{{"thoughts"' in instr and "{comm_budget}" in instr
    # Only per-step fields the legacy template already uses remain.
    assert _fields(instr) <= _fields(legacy_instr)

    cur = apply_hub_rewrites(_read("curriculum_info.txt"),
                             "curriculum_info.txt")
    assert "Last message from the orchestrator" in cur
    assert _fields(cur) == _fields(_read("curriculum_info.txt"))


def test_header_constant():
    assert HUB_COMM_HEADER == "Message from the orchestrator"


# ── 4. Ch3 chamber facts ────────────────────────────────────────────────

def test_ch3_facts_switch_wording_only_in_hub(monkeypatch):
    peer = chamber_facts.describe_chamber("ch3", 3)
    assert "Targeted communication is the only channel here." in peer
    monkeypatch.setenv(ENV_TOPOLOGY, "hub")
    hub = chamber_facts.describe_chamber("ch3", 3)
    assert "Targeted communication" not in hub
    assert "Messages to the orchestrator, which relays them" in hub
    assert hub.replace(chamber_facts._CH3_HUB_CHANNEL,
                       chamber_facts._CH3_PEER_CHANNEL) == peer


# ── 5. comm-budget topology switch ──────────────────────────────────────

def test_peer_renderings_are_unchanged_and_hub_differs():
    for enabled in (False, True):
        assert static_placeholders(enabled, 32) == static_placeholders(
            enabled, 32, "peer")
    legacy = static_placeholders(False, 32)
    assert legacy == {"comm_rule": COMM_RULE_LEGACY,
                      "comm_target_rule": COMM_TARGET_RULE_LEGACY,
                      "comm_field_hint": COMM_FIELD_HINT_LEGACY}
    hub = static_placeholders(False, 32, "hub")
    assert hub["comm_field_hint"] == COMM_FIELD_HINT_HUB_LEGACY
    assert "REQUIRED EVERY STEP" in hub["comm_rule"]
    hub_b = static_placeholders(True, 24, "hub")
    assert "cut at 24" in hub_b["comm_rule"]
    assert "neither send nor receive" in hub_b["comm_rule"]
    assert "objections you raise" in hub_b["comm_rule"]


def test_topology_env_roundtrip():
    assert comm_topology() == "peer"
    set_comm_topology("hub")
    try:
        assert comm_topology() == "hub"
        assert "HUB COMMUNICATION" in apply_comm_budget_static("{comm_rule}")
    finally:
        set_comm_topology("peer")
    assert comm_topology() == "peer"
    assert apply_comm_budget_static("{comm_rule}") == COMM_RULE_LEGACY
    with pytest.raises(ValueError):
        set_comm_topology("star")


def test_action_selection_renders_hub_wording_from_env():
    import social_stubs  # noqa: F401
    from mindforge.agent_modules import action_selection as asel
    set_comm_topology("hub")
    try:
        sel = asel.ActionSelection(
            action_model_client=object(),
            user_prompt_template=apply_hub_rewrites(
                asel.instruction_prompt_p2, "instruction_prompt_p2.txt"),
        )
        assert "HUB COMMUNICATION" in sel.system_prompt
        assert COMM_TARGET_RULE_HUB_LEGACY in sel.user_prompt_template
        assert '"communication_target": "orchestrator"' in \
            sel.user_prompt_template
        set_env_switch(True, 32)
        budget = asel.ActionSelection(action_model_client=object())
        assert "OPTIONAL and BUDGETED" in budget.system_prompt
        assert "orchestrator" in budget.system_prompt
    finally:
        set_comm_topology("peer")
        set_env_switch(False)
