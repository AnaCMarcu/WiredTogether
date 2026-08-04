"""Choice-mode social acts (Experiment 2) — helpers, schemas, templates.

Pins the ``agent_modules.social_acts`` helper surface (CSV parsing, act
normalization against the MENU, horizon clamp, the imitation gate, menu/schema
rendering, template pre-rendering) and the two schema subclasses.

The load-bearing non-regression assertions live here too:

- ``AgentResponse``'s pydantic schema gains NO new fields — the local model
  client injects the schema into the system prompt, so a field on the base
  class would change every legacy run's prompt (that is why choice mode uses
  the ``SocialAgentResponse`` SUBCLASS).
- The legacy prompt template files contain no choice-mode placeholders —
  choice mode loads PARALLEL ``*_choice.txt`` files instead of editing them.
"""

import social_stubs  # noqa: F401  (installs autogen/chromadb stand-ins)

import pytest

from agent_modules import social_acts as sa
from agent_modules.util import (
    AgentResponse,
    SocialAgentResponse,
    SocialThought,
    SocialThoughtChoice,
)

from social_stubs import REPO

PROMPTS = REPO / "src" / "mindforge" / "prompts"


# ── parse_channels_csv ──────────────────────────────────────────────────────

def test_parse_channels_full_menu():
    assert sa.parse_channels_csv("comm,obs,imit") == ("comm", "obs", "imit")


def test_parse_channels_accepts_verbs_and_normalises_order():
    assert sa.parse_channels_csv("imitate,communicate") == ("comm", "imit")


def test_parse_channels_none_and_empty():
    assert sa.parse_channels_csv("none") == ()
    assert sa.parse_channels_csv("") == ()
    assert sa.parse_channels_csv(None) == ()


def test_parse_channels_unknown_raises():
    with pytest.raises(ValueError):
        sa.parse_channels_csv("comm,telepathy")


# ── normalize_social_act ────────────────────────────────────────────────────

def test_normalize_act_case_and_decoration():
    menu = ("comm", "obs", "imit")
    assert sa.normalize_social_act("Observe agent_1", menu) == "observe"
    assert sa.normalize_social_act("IMITATE", menu) == "imitate"
    assert sa.normalize_social_act("communicate:", menu) == "communicate"


def test_normalize_act_disabled_channel_degrades_to_none():
    """The affordance ablation is enforced in code, not trusted to the prompt."""
    assert sa.normalize_social_act("observe", ("comm",)) == "none"
    assert sa.normalize_social_act("communicate", ()) == "none"


def test_normalize_act_junk_is_none():
    menu = ("comm", "obs", "imit")
    assert sa.normalize_social_act("dance", menu) == "none"
    assert sa.normalize_social_act(None, menu) == "none"
    assert sa.normalize_social_act(17, menu) == "none"


# ── clamp_horizon ───────────────────────────────────────────────────────────

def test_clamp_horizon_bounds():
    assert sa.clamp_horizon(0) == 1
    assert sa.clamp_horizon(3) == 3
    assert sa.clamp_horizon(20) == sa.IMITATE_MAX_HORIZON == 5
    assert sa.clamp_horizon("nope") == 1
    assert sa.clamp_horizon(None) == 1


# ── imitation_gate ──────────────────────────────────────────────────────────

def test_gate_near_same_chamber_passes():
    assert sa.imitation_gate((0, 0, 0), (3, 0, 0), "ch2", "ch2", radius=5.0)


def test_gate_radius_boundary_inclusive():
    assert sa.imitation_gate((0, 0, 0), (5, 0, 0), "ch2", "ch2", radius=5.0)
    assert not sa.imitation_gate((0, 0, 0), (5.01, 0, 0), "ch2", "ch2",
                                 radius=5.0)


def test_gate_different_chamber_fails():
    assert not sa.imitation_gate((0, 0, 0), (1, 0, 0), "ch2", "ch3",
                                 radius=5.0)


def test_gate_missing_state_fails():
    assert not sa.imitation_gate(None, (0, 0, 0), "ch2", "ch2", radius=5.0)
    assert not sa.imitation_gate((0, 0, 0), (1, 0, 0), None, "ch2", radius=5.0)
    assert not sa.imitation_gate((0, 0, 0), (1, 0, 0), "ch2", None, radius=5.0)


# ── menu / schema rendering ─────────────────────────────────────────────────

def test_menu_lists_enabled_acts_only():
    menu = sa.render_social_act_menu(("obs",))
    assert '"observe"' in menu
    assert '"communicate"' not in menu
    assert '"imitate"' not in menu
    assert '"none"' in menu  # none is always a choice


def test_menu_has_no_stray_braces():
    """The menu is spliced into a template later passed through safe_format —
    a literal brace would be parsed as a placeholder."""
    for channels in ((), ("comm",), ("comm", "obs", "imit")):
        block = sa.render_social_act_menu(channels)
        assert "{" not in block and "}" not in block


def test_schema_line_acts_and_doubled_braces():
    line = sa.render_social_act_schema(("comm", "imit"))
    assert '"communicate" | "imitate" | "none"' in line
    assert '"observe"' not in line
    # Doubled braces so the per-step safe_format emits a literal JSON object.
    assert line.count("{{") == 1 and line.count("}}") == 1


def test_schema_line_empty_menu():
    line = sa.render_social_act_schema(())
    assert '"social_act": "none"' in line


# ── template pre-rendering ──────────────────────────────────────────────────

def test_load_choice_templates_prerenders_static_parts():
    system_txt, instruction_txt = sa.load_choice_templates(("comm", "obs"))
    assert "{social_act_menu}" not in system_txt
    assert "{environment_prompt}" in system_txt        # filled at construction
    assert "{social_act_schema}" not in instruction_txt
    assert "{social_returns}" in instruction_txt       # filled per step
    assert '"observe"' in instruction_txt
    assert '"imitate"' not in instruction_txt


def test_social_module_choice_prompt_renders_menu():
    txt = sa.load_social_module_choice_prompt(("imit",))
    assert "{enabled_acts_menu}" not in txt
    assert '"imitate"' in txt
    assert '"observe"' not in txt
    assert "{bond_table}" in txt   # per-call placeholders survive


# ── non-regression: legacy templates and schemas are untouched ──────────────

def test_legacy_templates_have_no_choice_placeholders():
    for fname in ("system_prompt.txt", "instruction_prompt_p2.txt",
                  "social_module.txt"):
        txt = (PROMPTS / fname).read_text(encoding="utf-8")
        assert "social_act" not in txt, f"{fname} leaked a choice placeholder"
        assert "{social_returns}" not in txt


def test_agent_response_schema_gains_no_fields():
    """_inject_json_instruction derives the prompt's schema block from the
    pydantic model — a new field on AgentResponse would change every legacy
    prompt. Choice mode must use the subclass."""
    base = set(AgentResponse.model_json_schema()["properties"])
    assert base == {"thoughts", "action", "communication",
                    "communication_target"}
    social = set(SocialAgentResponse.model_json_schema()["properties"])
    assert social == base | {"social_act", "social_target", "imitate_horizon"}


def test_social_thought_schema_gains_no_fields():
    base = set(SocialThought.model_json_schema()["properties"])
    assert "suggest_act" not in base and "suggest_target" not in base
    choice = set(SocialThoughtChoice.model_json_schema()["properties"])
    assert choice == base | {"suggest_act", "suggest_target"}


def test_social_response_defaults_validate_legacy_payload():
    """A legacy-shaped response (no social fields) must validate unchanged."""
    r = SocialAgentResponse(thoughts="t", action="NoOp",
                            communication="", communication_target="")
    assert r.social_act == "none"
    assert r.social_target == ""
    assert r.imitate_horizon == 0
    t = SocialThoughtChoice(bond_change_explanation={}, reasoning="r",
                            referenced_bonds={})
    assert t.suggest_act is None and t.suggest_target is None
