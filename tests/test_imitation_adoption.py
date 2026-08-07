"""Guided imitation (Experiment 2, 2026-08-07 redesign) — adoption tracking.

imitate(j, h) no longer force-replays: the agent receives j's last h actions
plus j's state and instructions to judge applicability, and stays in control
of every action. The imit co-firing channel is credited on ADOPTION — each
step the imitator's own chosen action matches the next unconsumed element of
the delivered sequence. ``PendingImitation`` is the matcher the router uses;
``render_imitation_payload`` is the delivered text.
"""

import social_stubs  # noqa: F401  (installs autogen/chromadb stand-ins)

from agent_modules.social_acts import (
    ADOPTION_WINDOW_FACTOR,
    PendingImitation,
    render_imitation_payload,
)


# ── in-order matching ───────────────────────────────────────────────────────

def test_adoption_matches_in_order():
    p = PendingImitation(1, ["Dig", "MoveForward", "Dig"], start_step=10)
    assert p.note_action("Dig") is True          # 1/3
    assert p.note_action("MoveForward") is True  # 2/3
    assert p.note_action("Dig") is True          # 3/3
    assert p.done()


def test_adoption_tolerates_interleaved_actions():
    """Non-matching steps (aiming/turning between copied actions) neither
    advance nor reset the pointer."""
    p = PendingImitation(1, ["Dig", "Place"], start_step=0)
    assert p.note_action("TurnLeft") is False
    assert p.note_action("Dig") is True
    assert p.note_action("MoveForward") is False
    assert p.note_action("Place") is True
    assert p.done()


def test_adoption_does_not_skip_ahead():
    """An out-of-order match must not consume a later element."""
    p = PendingImitation(1, ["Dig", "Place"], start_step=0)
    assert p.note_action("Place") is False   # Place is 2nd, not next
    assert p.ptr == 0
    assert p.note_action("Dig") is True


def test_adoption_stops_when_done():
    p = PendingImitation(1, ["Jump"], start_step=0)
    assert p.note_action("Jump") is True
    assert p.done()
    assert p.note_action("Jump") is False    # no double credit


# ── window expiry ───────────────────────────────────────────────────────────

def test_window_scales_with_sequence_length():
    p = PendingImitation(1, ["Dig", "Dig", "Dig"], start_step=100)
    limit = 100 + ADOPTION_WINDOW_FACTOR * 3
    assert not p.expired(limit)
    assert p.expired(limit + 1)


def test_window_minimum_one_element():
    p = PendingImitation(1, [], start_step=50)
    assert not p.expired(50 + ADOPTION_WINDOW_FACTOR)
    assert p.expired(50 + ADOPTION_WINDOW_FACTOR + 1)


# ── construction hygiene ────────────────────────────────────────────────────

def test_empty_and_none_actions_filtered():
    p = PendingImitation(2, ["Dig", None, "", "Jump"], start_step=0)
    assert p.sequence == ["Dig", "Jump"]


def test_none_action_never_matches():
    p = PendingImitation(1, ["Dig"], start_step=0)
    assert p.note_action(None) is False
    assert p.note_action("") is False
    assert p.ptr == 0


# ── payload rendering ───────────────────────────────────────────────────────

def test_payload_carries_sequence_state_and_instructions():
    txt = render_imitation_payload(
        "agent_1", ["Dig", "MoveForward"], "break the anvil",
        "x=3 z=19 facing N", "ch2",
    )
    assert "agent_1" in txt
    assert "Dig -> MoveForward" in txt
    assert "break the anvil" in txt
    assert "ch2" in txt
    # The judge-then-re-enact instructions ARE the mechanism — the agent
    # must be told to check state match, and on mismatch to RETAIN the
    # recipe for a similar future situation (not discard it).
    assert "HOW TO USE THIS" in txt
    assert "RE-ENACT" in txt
    assert "similar situation" in txt


def test_payload_handles_missing_state_and_empty_sequence():
    txt = render_imitation_payload("agent_2", [], None, None, None)
    assert "(no recorded actions yet)" in txt
    assert "(unknown)" in txt
    assert "{" not in txt and "}" not in txt   # safe for safe_format prompts


def test_payload_has_no_braces():
    txt = render_imitation_payload(
        "agent_1", ["Dig"], "task", "pos", "ch3",
    )
    assert "{" not in txt and "}" not in txt
