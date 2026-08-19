"""Team-size-dependent prompt text for the agent-count scaling suite.

The prompt corpus was written for the original 3-agent team ("one of three
agents", "cells A/B/C", "A's switch opens B's door, B→C, C→A"). Running the
same environment with N ∈ {2,...,9} agents must not feed the model stale
3-agent facts, so the N-specific fragments in the prompt files were replaced
by ``{placeholder}`` tokens that :func:`apply_team_scaling` substitutes
literally (plain ``str.replace`` — no ``str.format`` parsing, so the JSON
``{{...}}`` escapes and the runtime placeholders like ``{convo}`` that
``llm_call`` fills later are left untouched).

GATED by the master switch ``--team-scaling`` / ``WT_TEAM_SCALING=1`` (set
only by the agent-count scaling launcher, scale_gemma.sbatch):

* switch OFF (every legacy suite, the default): placeholders substitute the
  frozen :data:`LEGACY_PLACEHOLDERS` — the exact pre-placeholder bytes,
  including the original intra-sentence line wrapping and the historical
  hardcoded-3 wording that the N=2 pair / N=6 transplant arms also ran
  with. Rendered prompts are byte-identical to the pre-refactor files for
  EVERY num_agents value.
* switch ON: placeholders substitute :func:`scaling_placeholders` — text
  truthful for the actual team size (``{ch4_zombies}`` additionally honors
  the ``FC_CH4_MOB_COUNT`` pin).

Both modes are pinned by tests/test_team_scaling.py.

Imports chamber_facts (stdlib-only) + os — safe to import anywhere.
"""

import os

try:
    # Runtime path: multi_agent_craftium.py runs as a script from
    # src/mindforge, so agent_modules is a top-level package.
    from agent_modules.chamber_facts import ch4_zombie_count
except ImportError:  # pragma: no cover — test-suite path (PYTHONPATH=src)
    from mindforge.agent_modules.chamber_facts import ch4_zombie_count

# Frozen pre-placeholder text (the original 3-agent wording, byte-exact —
# note switch_rotation carries the original line wrap of the role files).
# Do NOT derive these from scaling_placeholders(3): they must stay literal
# so legacy rendering can never drift.
LEGACY_PLACEHOLDERS = {
    "num_agents_word": "three",
    "cell_letters": "A/B/C",
    "cell_letters_or": "A, B, or C",
    "switch_rotation": "A's switch opens B's\n  door, B→C, C→A",
    "ch4_zombies": "3",
    "regroup_teammates": "both teammates",
    "example_last_agent": "agent_2",
    "example_last_cell": "C",
    "team_size": "3",
}

_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def num_word(n: int) -> str:
    """English word for small team sizes ("three"); digits beyond twelve."""
    return _NUM_WORDS.get(n, str(n))


def cell_letter(i: int) -> str:
    return chr(ord("A") + i)


def cell_letters_slash(n: int) -> str:
    """"A/B/C" — the compact cell list used in the role prompts."""
    return "/".join(cell_letter(i) for i in range(n))


def cell_letters_or(n: int) -> str:
    """"A, B, or C" — the spoken-style list used in questions/beliefs.

    n=2 collapses to "A or B" (no comma), matching how English reads.
    """
    letters = [cell_letter(i) for i in range(n)]
    if n <= 1:
        return letters[0] if letters else ""
    if n == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + ", or " + letters[-1]


def switch_rotation(n: int) -> str:
    """"A's switch opens B's door, B→C, C→A" for any ring size."""
    parts = [f"{cell_letter(0)}'s switch opens {cell_letter(1)}'s door"]
    for i in range(1, n):
        parts.append(f"{cell_letter(i)}→{cell_letter((i + 1) % n)}")
    return ", ".join(parts)


def regroup_teammates(n: int) -> str:
    """Phrase for "regroup ... with <the other agents>" in role_scouter.

    N=3 keeps the original "both teammates"; other sizes spell the count.
    """
    others = n - 1
    if others == 2:
        return "both teammates"
    if others == 1:
        return "your teammate"
    return f"all {others} teammates"


def scaling_placeholders(num_agents: int) -> dict:
    """Placeholder → replacement map for :func:`apply_team_scaling`.

    ``ch4_zombies`` honors the ``FC_CH4_MOB_COUNT`` pin (set by
    ``--ch4-mob-count``); with the pin unset it mirrors the Lua default
    ``min(NUM_AGENTS, 6)``.
    """
    n = num_agents
    zombies = ch4_zombie_count(n)
    return {
        "num_agents_word": num_word(n),
        "cell_letters": cell_letters_slash(n),
        "cell_letters_or": cell_letters_or(n),
        "switch_rotation": switch_rotation(n),
        "ch4_zombies": str(zombies),
        "regroup_teammates": regroup_teammates(n),
        # Example-text placeholders (belief prompts illustrate with the
        # highest-index agent so the example never names a nonexistent agent).
        "example_last_agent": f"agent_{n - 1}",
        "example_last_cell": cell_letter(n - 1),
        # Digit form for "all {team_size} agents are alive" (the legacy
        # files hardcoded "3" there; {num_agents} could not be reused
        # because build_role_configs always fills it with the true N).
        "team_size": str(n),
    }


def team_scaling_enabled() -> bool:
    """The master switch, as seen via the environment (set by
    --team-scaling in multi_agent_craftium.py; also readable by Lua)."""
    return os.environ.get("WT_TEAM_SCALING") == "1"


def apply_team_scaling(text: str, num_agents: int, enabled: bool | None = None) -> str:
    """Literal-substitute every scaling placeholder in ``text``.

    ``enabled=None`` (default) reads the WT_TEAM_SCALING master switch;
    False renders the frozen legacy 3-agent wording byte-exactly, True the
    truthful text for ``num_agents``. Non-str values (nested dicts in the
    prompts mapping) should be handled by the caller; this operates on one
    template string.
    """
    if enabled is None:
        enabled = team_scaling_enabled()
    mapping = scaling_placeholders(num_agents) if enabled else LEGACY_PLACEHOLDERS
    for key, value in mapping.items():
        text = text.replace("{" + key + "}", value)
    return text


def apply_team_scaling_to_prompts(prompts: dict, num_agents: int,
                                  enabled: bool | None = None) -> dict:
    """Substitute scaling placeholders across the load_prompts() dict
    (one level of nesting for the "roles" sub-dict)."""
    out = {}
    for key, value in prompts.items():
        if isinstance(value, str):
            out[key] = apply_team_scaling(value, num_agents, enabled)
        elif isinstance(value, dict):
            out[key] = {
                k: (apply_team_scaling(v, num_agents, enabled)
                    if isinstance(v, str) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out
