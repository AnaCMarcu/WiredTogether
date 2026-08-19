"""team_scaling: the master switch must separate the two rendering modes.

* switch OFF (legacy, the default): every substituted prompt file renders
  the ORIGINAL pre-placeholder wording BYTE-EXACTLY — including the original
  line wrapping and the hardcoded-3 text — for EVERY num_agents value
  (the N=2 pair and N=6 transplant suites ran with that text historically).
* switch ON (agent-count scaling suite): text is truthful for the actual
  team size, with no leftover placeholders.

The expected strings below are verbatim copies of the pre-placeholder prompt
literals — do not "fix" them here without also changing LEGACY_PLACEHOLDERS,
or legacy runs would silently get different prompts.
"""

from pathlib import Path

import pytest

from mindforge.agent_modules.chamber_facts import ch4_zombie_count
from mindforge.agent_modules.team_scaling import (
    LEGACY_PLACEHOLDERS,
    apply_team_scaling,
    apply_team_scaling_to_prompts,
    cell_letters_or,
    cell_letters_slash,
    num_word,
    regroup_teammates,
    scaling_placeholders,
    switch_rotation,
    team_scaling_enabled,
)

PROMPT_DIR = Path(__file__).resolve().parents[1] / "src" / "mindforge" / "prompts"
SCALING_KEYS = scaling_placeholders(3).keys()

# The original role-file Ch3 fragment, with its original line wrap.
ORIG_ROTATION_WRAPPED = "A's switch opens B's\n  door, B→C, C→A"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FC_CH4_MOB_COUNT", raising=False)
    monkeypatch.delenv("WT_TEAM_SCALING", raising=False)


def _render(filename: str, n: int, enabled: bool) -> str:
    text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    return apply_team_scaling(text, n, enabled=enabled)


def _assert_no_scaling_placeholders(text: str):
    for key in SCALING_KEYS:
        assert "{" + key + "}" not in text


# ── helpers ──────────────────────────────────────────────────────────────

def test_num_word():
    assert num_word(2) == "two"
    assert num_word(3) == "three"
    assert num_word(9) == "nine"
    assert num_word(42) == "42"


def test_cell_letter_lists():
    assert cell_letters_slash(3) == "A/B/C"
    assert cell_letters_slash(9) == "A/B/C/D/E/F/G/H/I"
    assert cell_letters_or(3) == "A, B, or C"
    assert cell_letters_or(2) == "A or B"
    assert cell_letters_or(9) == "A, B, C, D, E, F, G, H, or I"


def test_switch_rotation():
    assert switch_rotation(3) == "A's switch opens B's door, B→C, C→A"
    assert switch_rotation(2) == "A's switch opens B's door, B→A"
    assert switch_rotation(9) == ("A's switch opens B's door, B→C, C→D, D→E, "
                                  "E→F, F→G, G→H, H→I, I→A")


def test_regroup_teammates():
    assert regroup_teammates(3) == "both teammates"   # original scouter text
    assert regroup_teammates(2) == "your teammate"
    assert regroup_teammates(9) == "all 8 teammates"


def test_legacy_map_covers_every_placeholder():
    assert set(LEGACY_PLACEHOLDERS) == set(SCALING_KEYS)
    # Legacy values match scaling values at N=3 except the wrapped rotation.
    n3 = scaling_placeholders(3)
    for key, legacy in LEGACY_PLACEHOLDERS.items():
        if key == "switch_rotation":
            assert legacy == ORIG_ROTATION_WRAPPED
            assert legacy.replace("\n  ", " ") == n3[key]
        else:
            assert legacy == n3[key], key


# ── the master switch ────────────────────────────────────────────────────

def test_switch_defaults_off_and_reads_env(monkeypatch):
    assert not team_scaling_enabled()
    text = "one of {num_agents_word} agents"
    # enabled=None → env → off → legacy wording regardless of N.
    assert apply_team_scaling(text, 9) == "one of three agents"
    monkeypatch.setenv("WT_TEAM_SCALING", "1")
    assert team_scaling_enabled()
    assert apply_team_scaling(text, 9) == "one of nine agents"


# ── FC_CH4_MOB_COUNT pin ─────────────────────────────────────────────────

def test_zombie_count_default_mirrors_lua():
    assert ch4_zombie_count(2) == 2
    assert ch4_zombie_count(3) == 3
    assert ch4_zombie_count(6) == 6
    assert ch4_zombie_count(9) == 6  # 6 spawn positions in mobs.lua


def test_zombie_count_pinned(monkeypatch):
    monkeypatch.setenv("FC_CH4_MOB_COUNT", "3")
    for n in (2, 3, 4, 5, 6, 9):
        assert ch4_zombie_count(n) == 3


def test_zombie_count_invalid_pin_falls_back(monkeypatch):
    monkeypatch.setenv("FC_CH4_MOB_COUNT", "banana")
    assert ch4_zombie_count(4) == 4
    monkeypatch.setenv("FC_CH4_MOB_COUNT", "0")
    assert ch4_zombie_count(4) == 4


# ── switch OFF: byte-exact legacy wording at EVERY team size ─────────────

@pytest.mark.parametrize("n", [2, 3, 6, 9])
@pytest.mark.parametrize("role", ["agent", "hunter", "harvester", "scouter"])
def test_legacy_role_prompts_byte_exact(role, n):
    text = _render(f"role_{role}.txt", n, enabled=False).format(num_agents=n)
    assert "cells A/B/C. Rotational — " + ORIG_ROTATION_WRAPPED + ";" in text
    assert "3 zombies" in text
    assert "when all 3 are dead" in text
    assert "all 3 agents are alive" in text
    _assert_no_scaling_placeholders(text)


@pytest.mark.parametrize("n", [2, 9])
def test_legacy_other_prompts_byte_exact(n):
    env = _render("environment_prompt.txt", n, enabled=False)
    assert "You are one of three agents in a five-chamber cooperative dungeon." in env

    q = _render("curriculum_questions.txt", n, enabled=False)
    assert "I am one of three agents progressing cooperatively" in q
    assert "kill 3 zombies together" in q
    assert "(Cell A, B, or C)" in q

    partner = _render("belief_system/partner_beliefs.txt", n, enabled=False)
    assert "which cell (A, B, or C) they are in" in partner

    interaction = _render("belief_system/interaction_belief.txt", n, enabled=False)
    assert '"agent_2 is in Cell C"' in interaction
    for text in (env, q, partner, interaction):
        _assert_no_scaling_placeholders(text)


def test_legacy_role_agent_and_scouter_specifics():
    agent = _render("role_agent.txt", 2, enabled=False).format(num_agents=2)
    assert "(all three needed to unlock Ch4)" in agent
    scouter = _render("role_scouter.txt", 2, enabled=False).format(num_agents=2)
    assert "regroup in the communal room with both teammates" in scouter


# ── switch ON: original wording at N=3 ───────────────────────────────────

def test_environment_prompt_n3_original():
    text = _render("environment_prompt.txt", 3, enabled=True)
    assert "You are one of three agents in a five-chamber cooperative dungeon." in text
    _assert_no_scaling_placeholders(text)


def test_curriculum_questions_n3_original():
    text = _render("curriculum_questions.txt", 3, enabled=True)
    assert "I am one of three agents progressing cooperatively" in text
    assert "kill 3 zombies together" in text
    assert "(Cell A, B, or C)" in text
    _assert_no_scaling_placeholders(text)


def test_belief_prompts_n3_original():
    partner = _render("belief_system/partner_beliefs.txt", 3, enabled=True)
    assert "You are one of three agents in the Five Chambers" in partner
    assert "which cell (A, B, or C) they are in" in partner
    _assert_no_scaling_placeholders(partner)

    interaction = _render("belief_system/interaction_belief.txt", 3, enabled=True)
    assert "You are one of three agents in the Five Chambers" in interaction
    assert '"agent_2 is in Cell C"' in interaction
    assert '"agent_2 contributed to pair-dig' in interaction
    _assert_no_scaling_placeholders(interaction)


@pytest.mark.parametrize("role", ["agent", "hunter", "harvester", "scouter"])
def test_role_prompts_n3_original(role):
    # Mimic build_role_configs: team-scaling substitution first (done on the
    # load_prompts() dict), then .format(num_agents=N).
    text = _render(f"role_{role}.txt", 3, enabled=True).format(num_agents=3)
    assert "cells A/B/C" in text
    assert "A's switch opens B's door, B→C, C→A" in text
    assert "3 zombies" in text
    assert "when all 3 are dead" in text
    assert "all 3 agents are alive" in text
    _assert_no_scaling_placeholders(text)


# ── switch ON: truthful text at other team sizes ────────────────────────

@pytest.mark.parametrize("n", [2, 4, 5, 6, 9])
def test_role_prompt_scales(n, monkeypatch):
    monkeypatch.setenv("FC_CH4_MOB_COUNT", "3")  # scaling-suite pin
    text = _render("role_agent.txt", n, enabled=True).format(num_agents=n)
    assert f"cells {cell_letters_slash(n)}" in text
    assert switch_rotation(n) in text
    assert f"(all {num_word(n)} needed to unlock Ch4)" in text
    # Ch4 stays the pinned 3-agent design for every team size.
    assert "3 zombies" in text
    assert "when all 3 are dead" in text
    assert f"all {n} agents are alive" in text
    _assert_no_scaling_placeholders(text)


def test_environment_prompt_n9():
    text = _render("environment_prompt.txt", 9, enabled=True)
    assert "You are one of nine agents" in text


def test_interaction_belief_n2_names_existing_agent():
    text = _render("belief_system/interaction_belief.txt", 2, enabled=True)
    assert '"agent_1 is in Cell B"' in text
    assert "agent_2 is in Cell C" not in text


def test_apply_to_prompts_dict_handles_nesting():
    prompts = {
        "environment": "one of {num_agents_word} agents",
        "roles": {"agent": "cells {cell_letters}"},
        "other": 42,
    }
    out = apply_team_scaling_to_prompts(prompts, 2, enabled=True)
    assert out["environment"] == "one of two agents"
    assert out["roles"]["agent"] == "cells A/B"
    assert out["other"] == 42
    legacy = apply_team_scaling_to_prompts(prompts, 2, enabled=False)
    assert legacy["environment"] == "one of three agents"
    assert legacy["roles"]["agent"] == "cells A/B/C"
