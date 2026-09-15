"""analyze_wiring: message matrices, seatmate preference, co-milestones."""

import json

import pytest

from mindforge.tools.analyze_wiring import (
    agent_index,
    interior_seats,
    load_co_milestone_matrix,
    load_message_matrix,
    mean_preference,
    ring_conditioned_preference,
    seat_pair_table,
    seatmate_preference,
)


def test_agent_index_spellings():
    # Lua spells contributors 'agentN', Python 'agent_N' — both must resolve.
    assert agent_index("agent_3") == 3
    assert agent_index("agent3") == 3
    assert agent_index("Agent 3") == 3
    assert agent_index("singleplayer") is None
    assert agent_index("") is None


def _write_run(tmp_path, messages_by_ep, milestone_events):
    for ep, msgs in messages_by_ep.items():
        d = tmp_path / "episodes" / ep
        d.mkdir(parents=True)
        with open(d / "messages.jsonl", "w") as f:
            for t, s, r in msgs:
                f.write(json.dumps(
                    {"t": t, "sender": s, "receiver": r, "text": "x"}) + "\n")
    with open(tmp_path / "final_metrics.json", "w") as f:
        json.dump({"milestone_events": milestone_events}, f)
    return tmp_path


def test_message_matrix_and_preference(tmp_path):
    run = _write_run(tmp_path, {
        "ep_0001": [(0, "agent_0", "agent_1"), (1, "agent_0", "agent_1"),
                    (2, "agent_0", "agent_2"), (0, "agent_1", "agent_0"),
                    (5, "agent_3", "agent_0"),
                    # skipped: self-send, non-agent, out of range
                    (6, "agent_0", "agent_0"), (7, "server", "agent_1"),
                    (8, "agent_0", "agent_9")],
        "ep_0002": [(0, "agent_0", "agent_1")],
    }, [])
    total, eps = load_message_matrix(run, 4)
    assert total[0][1] == 3          # 2 in ep1 + 1 in ep2
    assert total[0][2] == 1
    assert total[1][0] == 1
    assert total[3][0] == 1
    assert sum(sum(r) for r in total) == 6
    assert eps["ep_0001"][0][1] == 2 and eps["ep_0002"][0][1] == 1

    prefs = seatmate_preference(total)
    assert prefs[0][0] == pytest.approx(3 / 4)   # 3 of 4 to seatmate 1
    # agent_3's seatmate is agent_2; its one message went to agent_0.
    assert prefs[3][0] == pytest.approx(0.0)
    assert prefs[2] == (None, 0)                 # sent nothing


def test_co_milestone_matrix_selective(tmp_path):
    run = _write_run(tmp_path, {}, [
        # pair co-earn — Lua spelling, same (id, step)
        {"step": 10, "milestone_id": "m9_anvil_B1", "contributor": "agent0"},
        {"step": 10, "milestone_id": "m9_anvil_B1", "contributor": "agent1"},
        # solo milestone — no pair signal
        {"step": 20, "milestone_id": "m1_move_5", "contributor": "agent_2"},
        # all-hands milestone — dropped by selective, kept otherwise
        {"step": 30, "milestone_id": "m_comm_ch1", "contributor": "agent_0"},
        {"step": 30, "milestone_id": "m_comm_ch1", "contributor": "agent_1"},
        {"step": 30, "milestone_id": "m_comm_ch1", "contributor": "agent_2"},
        {"step": 30, "milestone_id": "m_comm_ch1", "contributor": "agent_3"},
    ])
    sel = load_co_milestone_matrix(run, 4, selective=True)
    assert sel[0][1] == 1 and sel[1][0] == 1
    assert sum(sum(r) for r in sel) == 2         # only the anvil pair
    full = load_co_milestone_matrix(run, 4, selective=False)
    assert full[0][1] == 2                       # anvil + all-hands
    assert full[2][3] == 1                       # all-hands only


def test_seat_pair_table_labels():
    msg = [[0, 5, 1, 0], [4, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0]]
    co = [[0, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    seat_pairs = [
        {"seats": [0, 1], "same_source_run": True, "cofired": True},
        {"seats": [2, 3], "same_source_run": True, "cofired": False},
    ]
    rows = seat_pair_table(msg, co, seat_pairs)
    assert rows[0]["label"] == "GENUINE"
    assert rows[0]["messages_within"] == 9
    assert rows[0]["co_milestones"] == 3
    assert rows[1]["label"] == "CONTROL"
    # Shuffled manifests mark seat pairs as cross-run -> strangers.
    rows_s = seat_pair_table(msg, co, [
        {"seats": [0, 1], "same_source_run": False, "cofired": None}])
    assert rows_s[0]["label"] == "strangers"


# ── ring-conditioned preference (geometry-free partner choice) ─────────────

def _ring_only_team(n=6, per_neighbour=10):
    """A team with NO relationships that talks only to its ring neighbours.

    WIRE's Ch3 switch ring and its index-ordered Ch4/Ch5 spawn rows make the
    two index-adjacent agents an agent's task partners. This is what a purely
    task-driven, relationship-free team looks like.
    """
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in ((i + 1) % n, (i - 1) % n):
            m[i][j] = per_neighbour
    return m


def test_ring_conditioning_strips_the_puzzle_geometry():
    m = _ring_only_team()
    # Raw seatmate preference reads as a strong effect: 0.50 against a
    # nominal chance of 1/(N-1) = 0.20, purely from the environment's wiring.
    raw = seatmate_preference(m)
    assert mean_preference(raw) == pytest.approx(0.5)
    assert mean_preference(raw) > 2 * (1.0 / 5)
    # Conditioning on the target already being a ring neighbour puts the same
    # team exactly at its own chance level, which is 0.50 by construction.
    ring = ring_conditioned_preference(m)
    assert mean_preference(ring) == pytest.approx(0.5)
    for i in range(6):
        assert ring[i][0] == pytest.approx(0.5)
        assert ring[i][1] == 20


def test_ring_conditioning_ignores_non_ring_traffic():
    n = 6
    m = [[0] * n for _ in range(n)]
    # agent_0: 3 to seatmate 1 (ring), 1 to ring neighbour 5, 96 to agent 3
    # (not a ring neighbour). The 96 must not enter the denominator.
    m[0][1], m[0][5], m[0][3] = 3, 1, 96
    assert seatmate_preference(m)[0][0] == pytest.approx(3 / 100)
    pref, sent = ring_conditioned_preference(m)[0]
    assert sent == 4
    assert pref == pytest.approx(0.75)


def test_ring_conditioning_undefined_when_the_ring_collapses():
    # n=2: (i+1)%2 == (i-1)%2, so there is no two-way choice to measure.
    assert ring_conditioned_preference([[0, 3], [2, 0]]) == {0: (None, 0),
                                                            1: (None, 0)}
    # Silent agents stay None rather than counting as 0 preference.
    assert ring_conditioned_preference([[0] * 6 for _ in range(6)])[2] == (None, 0)


def test_end_seats_are_excluded_from_the_interior_mean():
    # The Ch4/Ch5 spawn row is linear in seat index, so seats 0 and n-1 have
    # one ring neighbour beside them and one at the far end of the row.
    assert interior_seats(6) == [1, 2, 3, 4]
    assert interior_seats(4) == [1, 2]
    m = _ring_only_team()
    m[0][1], m[0][5] = 20, 0        # end seat, maximally seatmate-biased
    ring = ring_conditioned_preference(m)
    assert mean_preference(ring) > mean_preference(
        ring, agents=set(interior_seats(6)))
    assert mean_preference(ring, agents=set(interior_seats(6))) == pytest.approx(0.5)


def test_mean_preference_skips_missing_values():
    assert mean_preference({0: (0.4, 5), 1: (None, 0), 2: (0.6, 5)}) == pytest.approx(0.5)
    assert mean_preference({0: (None, 0)}) is None
    assert mean_preference({0: (0.4, 5), 1: (0.8, 5)}, agents={1}) == pytest.approx(0.8)
