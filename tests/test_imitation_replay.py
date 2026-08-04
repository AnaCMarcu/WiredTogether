"""Imitation replay bookkeeping on CustomAgent (Experiment 2).

Exercises the replay API surface (queue fill/drain, abort semantics, the
return-and-clear abort reason used by the router's social_acts.jsonl rows)
plus the HP parser feeding the damage-abort check.

The agent is built via ``__new__`` with only the replay attributes set —
``CustomAgent.__init__`` would construct the whole cognitive stack (LLM
clients, ChromaDB stores), none of which the replay machinery touches. The
router-side behavior (gate, per-replayed-step credit) is pinned in
test_social_acts.py / test_hebbian_channels.py.
"""

import social_stubs  # noqa: F401  (installs autogen/chromadb stand-ins)

from collections import deque

from custom_agent import CustomAgent, _parse_hp


def _bare_agent():
    a = CustomAgent.__new__(CustomAgent)
    a._replay_queue = deque()
    a._replay_target = None
    a._replay_total = 0
    a._replay_abort_reason = None
    a._last_hp = None
    return a


# ── begin_imitation ─────────────────────────────────────────────────────────

def test_begin_imitation_fills_queue_in_order():
    a = _bare_agent()
    a.begin_imitation(["Dig", "MoveForward", "TurnLeft"], "agent_1")
    assert a.replay_active
    assert list(a._replay_queue) == ["Dig", "MoveForward", "TurnLeft"]
    assert a._replay_target == "agent_1"
    assert a._replay_total == 3


def test_begin_imitation_filters_empty_actions():
    a = _bare_agent()
    a.begin_imitation(["Dig", None, "", "Jump"], "agent_2")
    assert list(a._replay_queue) == ["Dig", "Jump"]
    assert a._replay_total == 2


def test_begin_imitation_replaces_previous_replay():
    """A newly chosen social act aborts the replay already in progress."""
    a = _bare_agent()
    a.begin_imitation(["Dig", "Dig", "Dig"], "agent_1")
    a.begin_imitation(["Jump"], "agent_2")
    assert list(a._replay_queue) == ["Jump"]
    assert a._replay_target == "agent_2"
    assert a._replay_total == 1


# ── abort semantics ─────────────────────────────────────────────────────────

def test_abort_clears_queue_and_records_reason():
    a = _bare_agent()
    a.begin_imitation(["Dig", "Jump"], "agent_1")
    a._abort_replay("hp_drop")
    assert not a.replay_active
    assert a._replay_target is None
    assert a._replay_total == 0
    assert a.pop_replay_abort() == "hp_drop"


def test_pop_replay_abort_returns_and_clears():
    a = _bare_agent()
    a._abort_replay("chamber_change")
    assert a.pop_replay_abort() == "chamber_change"
    assert a.pop_replay_abort() is None  # cleared on read


def test_begin_imitation_clears_stale_abort_reason():
    a = _bare_agent()
    a._abort_replay("chamber_change")
    a.begin_imitation(["Dig"], "agent_1")
    assert a.pop_replay_abort() is None


# ── HP parsing (damage-abort input) ─────────────────────────────────────────

def test_parse_hp_typical_status():
    assert _parse_hp("Health: 12/20 | Time: day") == 12
    assert _parse_hp("Health: 20/20") == 20


def test_parse_hp_missing_or_junk():
    assert _parse_hp(None) is None
    assert _parse_hp("") is None
    assert _parse_hp("Health: ?/20 | Time: Unknown") is None
    assert _parse_hp(42) is None
