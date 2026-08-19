"""Chamber prompt text: visible-object whitelists + room facts.

Extracted from custom_agent.py so the N-dependent text (Ch3 cell mapping and
switch ring, Ch4 zombie count, all-N bonus phrasing) can be generated from
``num_agents`` and unit-tested without importing the autogen-based agent.
At num_agents=3 every generated string is byte-identical to the original
hardcoded text (pinned by tests/test_chamber_facts.py).

Stdlib-only (os, for the FC_CH4_MOB_COUNT pin) — safe to import anywhere.
"""

import os

_CHAMBER_OBJECT_WHITELIST = {
    "ch1": (
        "trees (brown trunks), stone blocks (grey textured), chickens, sheep, "
        "the locked red Door 1 in the north wall, bedrock walls/floor/ceiling. "
        "There are NO purple anvils, NO blue switches, NO zombies, NO boss, "
        "NO diamond sword or chestplate, NO red conveyor belts here. If you "
        "'see' any of those, you are hallucinating from the HUD overlay or "
        "from a stale belief — ignore it"
    ),
    "ch2": (
        "exactly 2 purple anvils (Row A front + Row B back, both centred "
        "along x), bedrock walls/floor/ceiling. There are NO trees, NO stone "
        "blocks, NO animals, NO switches, NO zombies, NO boss in Ch2"
    ),
    "ch3": (
        "your isolated cell with ONE blue switch cube on the south wall, "
        "the locked red cell door, bedrock walls. There are NO anvils, NO "
        "animals, NO zombies, NO boss in your cell. You CANNOT see your "
        "teammates from inside the cell"
    ),
    "ch3_communal": (
        "the communal regroup room (no puzzle objects here), the locked "
        "red Door 3 to Ch4, bedrock walls. There are NO anvils, NO switches, "
        "NO zombies in the communal room"
    ),
    "ch5": (
        "1 boss (large zombie variant), bedrock walls. There are NO trees, "
        "NO anvils, NO switches, NO other zombies in Ch5"
    ),
}


_CHAMBER_FACTS = {
    "ch1": (
        "Solo learning — each reward is individual (not shared). Dig trees/stone, "
        "collect drops, kill chickens/sheep. The RED locked Door 1 in the north wall "
        "unlocks the moment ANY agent fires a Ch1 milestone (dig 3 blocks, pick up 3 "
        "items, dig 5 wood, kill an animal, or dig 3 stone); the first to unlock it earns "
        "a bonus, and once open every agent can walk north through it. Safety net: if the "
        "whole team is still in Ch1 at the episode halfway point, everyone is teleported to "
        "Ch2 but unearned Ch1 rewards are forfeit."
    ),
    "ch2": (
        "Cooperative gear production. Two purple anvils at the centre: Row A (front, Z~19) "
        "drops a DIAMOND SWORD on break, Row B (back, Z~22) drops a DIAMOND CHESTPLATE. An "
        "anvil only breaks when TWO agents Dig the SAME anvil at the same time, within 3 "
        "blocks of each other (three is faster); a single digger makes zero progress. Gear "
        "AUTO-EQUIPS across the whole team on break (no pickup) and both pieces are needed "
        "to survive Ch4/Ch5. chamber_state reports each anvil's HP, its change over the "
        "last 3 steps (Δhp_last3), and a punchers list (who else is digging it). Door 2 "
        "opens once both anvils are broken (chamber_state shows Door 2 OPEN/LOCKED; walk "
        "north when OPEN)."
    ),
    "ch3_communal": (
        "The communal regroup room (no puzzle objects). Door 3 to Ch4 opens once all agents "
        "are in the communal room together; chamber_state shows Door 3 LOCKED/OPEN — walk "
        "north when OPEN."
    ),
}


def _cell_letter(i: int) -> str:
    return chr(ord("A") + i)


def _zombie_count(num_agents: int) -> int:
    # Mirrors Lua's spawn_ch4_mobs: min(CH4_MOB_COUNT or NUM_AGENTS,
    # #CH4_SPAWN_POSITIONS) with 6 spawn positions defined in mobs.lua.
    # FC_CH4_MOB_COUNT is the agent-count-scaling pin (set by
    # --ch4-mob-count via multi_agent_craftium.py, read here at call time
    # so prompt text always matches what the Lua server actually spawns).
    # Unset/invalid -> legacy one-zombie-per-agent behavior.
    try:
        pinned = int(os.environ.get("FC_CH4_MOB_COUNT", ""))
    except ValueError:
        pinned = 0
    if pinned > 0:
        return min(pinned, 6)
    return min(num_agents, 6)


def ch4_zombie_count(num_agents: int) -> int:
    """Public alias: how many zombies Ch4 spawns for this run."""
    return _zombie_count(num_agents)


def _ch4_whitelist(num_agents: int) -> str:
    return (
        f"{_zombie_count(num_agents)} zombies, the locked red Door 4 to Ch5, "
        "bedrock walls. There are "
        "NO trees, NO anvils, NO switches, NO boss in Ch4"
    )


def _ch3_facts(num_agents: int) -> str:
    cells = ", ".join(
        f"agent_{i}=" + (f"Cell {_cell_letter(i)}" if i == 0 else _cell_letter(i))
        for i in range(num_agents)
    )
    ring = ", ".join(
        f"{_cell_letter(i)} opens {_cell_letter((i + 1) % num_agents)}'s"
        + (" door" if i == 0 else "")
        for i in range(num_agents)
    )
    return (
        f"Communication puzzle. You are teleported into a SEALED CELL by id ({cells}"
        ") and cannot see teammates. Each cell has ONE blue switch cube "
        "on the south wall (press by facing it and using Dig, bare hands work). Switches are "
        f"wired rotationally: {ring} — you CANNOT open "
        "your own door, only a teammate can free you. Targeted communication is the only "
        "channel here. chamber_state shows your cell door LOCKED/OPEN (the only proof a "
        "teammate's press worked; a \"[SYSTEM] Switch X was pressed.\" broadcast also "
        "fires); walk north when it reads OPEN."
    )


def _ch4_facts(num_agents: int) -> str:
    zombies = _zombie_count(num_agents)
    return (
        f"Combat — {zombies} zombies. Attack with your wielded diamond sword (Slot1 if needed); the "
        f"chestplate reduces incoming damage. The door to Ch5 opens when all {zombies} zombies are "
        f"dead, and a team bonus is awarded if all {num_agents} agents are still alive at the clear."
    )


def _ch5_facts(num_agents: int) -> str:
    return (
        "Boss fight — one strong zombie, 60 HP, 3 damage per hit; it takes damage from every "
        "agent attacking it. The episode ends when the boss is defeated, with a large bonus "
        f"if all {num_agents} agents are alive at defeat."
    )


def describe_chamber(chamber, num_agents=3):
    """Return a self-contained `Chamber:` block: name + visible-objects whitelist
    + this room's objective/mechanics (ROOM FACTS).

    Substituted into the ``{current_chamber}`` placeholder in
    instruction_prompt_p2.txt. Multi-line: the whitelist (negative grounding to
    fight hallucination) followed by the current room's facts, so the agent
    carries ONLY the chamber it is standing in rather than all five.
    """
    if not chamber:
        return ("Unknown (position not yet locked — assume Ch1; only "
                "trees / stone / animals / locked red Door 1 are valid targets)")
    if chamber == "ch4":
        whitelist = _ch4_whitelist(num_agents)
    else:
        whitelist = _CHAMBER_OBJECT_WHITELIST.get(chamber)
    if chamber == "ch3":
        facts = _ch3_facts(num_agents)
    elif chamber == "ch4":
        facts = _ch4_facts(num_agents)
    elif chamber == "ch5":
        facts = _ch5_facts(num_agents)
    else:
        facts = _CHAMBER_FACTS.get(chamber)
    head = f"{chamber} — VISIBLE HERE: {whitelist}." if whitelist else chamber
    if facts:
        return f"{head}\nROOM FACTS: {facts}"
    return head
