"""chamber_facts: N=3 output must be byte-identical to the pre-extraction
hardcoded prompt text; N=6 must extend the cell ring / zombie counts.

The expected strings below are verbatim copies of the original
custom_agent.py literals — do not "fix" them here without also changing the
generator, or 3-agent runs would silently get different prompts.
"""

from mindforge.agent_modules.chamber_facts import (
    _ch3_facts,
    _ch4_facts,
    _ch4_whitelist,
    _ch5_facts,
    describe_chamber,
)

ORIG_CH3_FACTS = (
    "Communication puzzle. You are teleported into a SEALED CELL by id (agent_0=Cell A, "
    "agent_1=B, agent_2=C) and cannot see teammates. Each cell has ONE blue switch cube "
    "on the south wall (press by facing it and using Dig, bare hands work). Switches are "
    "wired rotationally: A opens B's door, B opens C's, C opens A's — you CANNOT open "
    "your own door, only a teammate can free you. Targeted communication is the only "
    "channel here. chamber_state shows your cell door LOCKED/OPEN (the only proof a "
    "teammate's press worked; a \"[SYSTEM] Switch X was pressed.\" broadcast also "
    "fires); walk north when it reads OPEN."
)

ORIG_CH4_FACTS = (
    "Combat — 3 zombies. Attack with your wielded diamond sword (Slot1 if needed); the "
    "chestplate reduces incoming damage. The door to Ch5 opens when all 3 zombies are "
    "dead, and a team bonus is awarded if all 3 agents are still alive at the clear."
)

ORIG_CH5_FACTS = (
    "Boss fight — one strong zombie, 60 HP, 3 damage per hit; it takes damage from every "
    "agent attacking it. The episode ends when the boss is defeated, with a large bonus "
    "if all 3 agents are alive at defeat."
)

ORIG_CH4_WHITELIST = (
    "3 zombies, the locked red Door 4 to Ch5, bedrock walls. There are "
    "NO trees, NO anvils, NO switches, NO boss in Ch4"
)


def test_n3_ch3_byte_identical():
    assert _ch3_facts(3) == ORIG_CH3_FACTS


def test_n3_ch4_byte_identical():
    assert _ch4_facts(3) == ORIG_CH4_FACTS
    assert _ch4_whitelist(3) == ORIG_CH4_WHITELIST


def test_n3_ch5_byte_identical():
    assert _ch5_facts(3) == ORIG_CH5_FACTS


def test_n3_describe_chamber_composes_head_and_facts():
    block = describe_chamber("ch4", 3)
    assert block == (
        f"ch4 — VISIBLE HERE: {ORIG_CH4_WHITELIST}."
        f"\nROOM FACTS: {ORIG_CH4_FACTS}"
    )


def test_describe_chamber_default_is_three_agents():
    assert describe_chamber("ch3") == describe_chamber("ch3", 3)


def test_n6_ch3_extends_cells_and_ring():
    facts = _ch3_facts(6)
    assert "agent_0=Cell A, agent_1=B, agent_2=C, agent_3=D, agent_4=E, " \
           "agent_5=F" in facts
    assert ("A opens B's door, B opens C's, C opens D's, D opens E's, "
            "E opens F's, F opens A's") in facts


def test_n6_ch4_and_ch5_counts():
    facts4 = _ch4_facts(6)
    assert "Combat — 6 zombies." in facts4
    assert "all 6 zombies are dead" in facts4
    assert "all 6 agents are still alive" in facts4
    assert "6 zombies, the locked red Door 4" in _ch4_whitelist(6)
    assert "all 6 agents are alive at defeat" in _ch5_facts(6)


def test_n2_pair_run_counts():
    # Phase A pair runs (N=2): ring collapses to a two-cycle.
    facts = _ch3_facts(2)
    assert "(agent_0=Cell A, agent_1=B)" in facts
    assert "A opens B's door, B opens A's" in facts
    assert "Combat — 2 zombies." in _ch4_facts(2)


def test_unknown_chamber_fallback_unchanged():
    assert describe_chamber(None).startswith("Unknown (position not yet locked")
