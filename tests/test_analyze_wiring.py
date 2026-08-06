"""analyze_wiring: message matrices, seatmate preference, co-milestones."""

import json

import pytest

from mindforge.tools.analyze_wiring import (
    agent_index,
    load_co_milestone_matrix,
    load_message_matrix,
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
