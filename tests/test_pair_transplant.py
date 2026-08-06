"""mindforge.tools.pair_transplant: W merge, name remap, shuffle, ranking."""

import json

import numpy as np
import pytest

from mindforge.tools.pair_transplant import (
    build_merged_manifest,
    build_slot_assignment,
    merge_hebbian_W,
    rank_pair_runs,
    remap_agent_names,
)


# ── merge_hebbian_W ──────────────────────────────────────────────────────────

W_A = [[0.0, 0.9], [0.7, 0.0]]
W_B = [[0.0, 0.5], [0.3, 0.0]]
W_C = [[0.0, 0.4], [0.4, 0.0]]


def test_merge_shape_blocks_and_cross():
    W, prov = merge_hebbian_W([W_A, W_B, W_C], normalize="none")
    assert W.shape == (6, 6)
    # Blocks at (0,1), (2,3), (4,5) — faithful magnitudes with normalize=none.
    assert W[0, 1] == pytest.approx(0.9)
    assert W[1, 0] == pytest.approx(0.7)
    assert W[2, 3] == pytest.approx(0.5)
    assert W[3, 2] == pytest.approx(0.3)
    assert W[4, 5] == pytest.approx(0.4)
    assert W[5, 4] == pytest.approx(0.4)
    # Diagonal zero; every other entry is the cross weight.
    assert np.all(np.diag(W) == 0.0)
    block_positions = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)}
    for i in range(6):
        for j in range(6):
            if i != j and (i, j) not in block_positions:
                assert W[i, j] == pytest.approx(0.1), (i, j)
    assert prov["rescale_factors"] == [1.0, 1.0, 1.0]


def test_merge_block_mean_normalization():
    W, prov = merge_hebbian_W([W_A, W_B, W_C], normalize="block_mean")
    # Block means: 0.8, 0.4, 0.4 → target = mean = 0.5333…
    target = (0.8 + 0.4 + 0.4) / 3.0
    for k in range(3):
        base = 2 * k
        block_mean = (W[base, base + 1] + W[base + 1, base]) / 2.0
        assert block_mean == pytest.approx(target)
    # Within-block asymmetry preserved: W01/W10 ratio unchanged.
    assert W[0, 1] / W[1, 0] == pytest.approx(0.9 / 0.7)
    assert W[2, 3] / W[3, 2] == pytest.approx(0.5 / 0.3)
    # Provenance records the factors.
    assert prov["block_means"] == pytest.approx([0.8, 0.4, 0.4])
    assert prov["rescale_factors"] == pytest.approx(
        [target / 0.8, target / 0.4, target / 0.4]
    )


def test_merge_entries_stay_in_unit_interval():
    hot = [[0.0, 1.0], [1.0, 0.0]]
    cold = [[0.0, 0.05], [0.05, 0.0]]
    W, _ = merge_hebbian_W([hot, cold, cold], normalize="block_mean")
    assert np.all(W >= 0.0) and np.all(W <= 1.0)


def test_merge_rejects_zero_cross_weight():
    # W=0 is a fixed point of the gated rule — zero cross-pair bonds could
    # never grow, which would rig the persistence comparison.
    with pytest.raises(ValueError, match="cross_weight"):
        merge_hebbian_W([W_A, W_B, W_C], cross_weight=0.0)


def test_merge_rejects_bad_shapes_and_modes():
    with pytest.raises(ValueError, match="2x2"):
        merge_hebbian_W([[[0.0] * 3] * 3])
    with pytest.raises(ValueError, match="normalize"):
        merge_hebbian_W([W_A], normalize="softmax")


# ── remap_agent_names ────────────────────────────────────────────────────────

def test_remap_spelling_variants():
    mapping = {0: 2, 1: 3}
    assert remap_agent_names("agent_0 met agent_1", mapping) == \
        "agent_2 met agent_3"
    assert remap_agent_names("Agent_0 and AGENT_1", mapping) == \
        "agent_2 and agent_3"
    assert remap_agent_names("agent 0 waved at agent1", mapping) == \
        "agent_2 waved at agent_3"


def test_remap_simultaneous_swap_does_not_chain():
    # 0→1 and 1→0 applied in one pass: "agent_0 agent_1" must become
    # "agent_1 agent_0", never "agent_0 agent_0".
    assert remap_agent_names("agent_0 agent_1", {0: 1, 1: 0}) == \
        "agent_1 agent_0"


def test_remap_unknown_indices_untouched():
    assert remap_agent_names("agent_7 stays", {0: 2}) == "agent_7 stays"


def test_remap_word_boundaries():
    mapping = {0: 2}
    # Embedded in identifiers on the left: no match.
    assert remap_agent_names("management_0", mapping) == "management_0"
    # "agent_01" is agent 1 by int value — not in mapping here, untouched.
    assert remap_agent_names("agent_01", {0: 2}) == "agent_01"
    assert remap_agent_names("agent_01", {1: 3}) == "agent_3"


def test_remap_empty_and_none_safe():
    assert remap_agent_names("", {0: 1}) == ""
    assert remap_agent_names(None, {0: 1}) is None


# ── build_slot_assignment ────────────────────────────────────────────────────

def test_identity_assignment():
    assert build_slot_assignment(3, shuffled=False) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]


@pytest.mark.parametrize("seed", range(10))
def test_shuffled_assignment_invariants(seed):
    assignment = build_slot_assignment(3, shuffled=True, seed=seed)
    # Every source agent seated exactly once.
    assert sorted(assignment) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]
    # No seat-pair holds two agents from the same source run.
    for k in range(3):
        a, b = assignment[2 * k], assignment[2 * k + 1]
        assert a[0] != b[0], f"seat-pair {k} holds two agents of run {a[0]}"


def test_shuffled_assignment_deterministic_per_seed():
    a1 = build_slot_assignment(3, shuffled=True, seed=7)
    a2 = build_slot_assignment(3, shuffled=True, seed=7)
    assert a1 == a2


# ── build_merged_manifest ────────────────────────────────────────────────────

def _pair_state(run_idx, agent_idx):
    """Minimal exported agent state whose free text names self and partner."""
    me, partner = agent_idx, 1 - agent_idx
    return {
        "agent_name": f"agent_{agent_idx}",
        "skills": {
            "dig": {
                "code": "Dig",
                "description": f"Action: Dig. Context: agent_{me} dug next "
                               f"to agent_{partner}.",
            },
        },
        "episodes": [
            {"id": "episode_0", "text": f"Task: help Agent_{partner} break "
                                        f"the anvil.", "episode": 0,
             "success": 1},
        ],
        "curriculum": {
            "current_context": f"run {run_idx} context",
            "completed_tasks": [f"align with Agent_{partner}"],
            "failed_tasks": [f"agent {partner} did not respond"],
        },
    }


def _pair_states():
    return [
        {0: _pair_state(r, 0), 1: _pair_state(r, 1)} for r in range(3)
    ]


def test_manifest_transplant_remaps_partner_to_seatmate():
    assignment = build_slot_assignment(3, shuffled=False)
    manifest = build_merged_manifest(_pair_states(), assignment)
    assert manifest["num_agents"] == 6
    assert set(manifest["agents"]) == {str(i) for i in range(6)}
    # Seat 2 = run 1's agent_0; its old partner (agent_1) is now seat 3.
    seat2 = manifest["agents"]["2"]
    assert seat2["source"] == "pair_run_1/agent_0"
    assert "agent_3" in seat2["skills"]["dig"]["description"]
    assert "agent_2" in seat2["skills"]["dig"]["description"]
    assert seat2["episodes"][0]["text"] == \
        "Task: help agent_3 break the anvil."
    assert seat2["curriculum"]["completed_tasks"] == ["align with agent_3"]
    assert seat2["curriculum"]["failed_tasks"] == ["agent_3 did not respond"]


def test_manifest_shuffled_partner_points_at_stranger_seatmate():
    assignment = build_slot_assignment(3, shuffled=True, seed=0)
    manifest = build_merged_manifest(
        _pair_states(), assignment, condition="shuffled"
    )
    assert manifest["condition"] == "shuffled"
    # In every seat, the transplanted memories must reference the SEATMATE's
    # seat index (a stranger), keeping memory<->bond consistency per seat.
    for seat in range(6):
        seatmate = seat + 1 if seat % 2 == 0 else seat - 1
        entry = manifest["agents"][str(seat)]
        assert f"agent_{seatmate}" in entry["episodes"][0]["text"], (
            seat, entry["episodes"][0]["text"]
        )


# ── rank_pair_runs ───────────────────────────────────────────────────────────

def _write_run(tmp_path, name, w01, w10, episodes=None):
    """episodes: list of (joint_dig, co_action, proximity) per episode."""
    d = tmp_path / name
    d.mkdir()
    with open(d / "hebbian_graph_final.json", "w") as f:
        json.dump({"num_agents": 2,
                   "W": [[0.0, w01], [w10, 0.0]]}, f)
    for k, (jd, ca, px) in enumerate(episodes or [], start=1):
        ep = d / "episodes" / f"ep_{k:04d}"
        ep.mkdir(parents=True)
        with open(ep / "summary.json", "w") as f:
            json.dump({"episode": k, "cooperation_metrics": {
                "joint_dig_events": jd,
                "co_action_events": ca,
                "proximity_events": px,
            }}, f)
    return d


def test_rank_orders_by_mean_bond(tmp_path):
    weak = _write_run(tmp_path, "weak", 0.2, 0.2, episodes=[(1, 10, 20)])
    strong = _write_run(tmp_path, "strong", 0.9, 0.8, episodes=[(5, 30, 40)])
    mid = _write_run(tmp_path, "mid", 0.5, 0.5)
    broken = tmp_path / "broken"
    broken.mkdir()

    rows = rank_pair_runs([weak, strong, mid, broken])
    assert [r["run_dir"] for r in rows[:3]] == [
        str(strong), str(mid), str(weak)
    ]
    assert rows[0]["bond"] == pytest.approx(0.85)
    assert rows[0]["joint_dig"] == 5
    # No episodes/ at all -> None, not a misleading 0.
    assert rows[1]["joint_dig"] is None
    # Unreadable runs sort last with an error note, not dropped.
    assert rows[3]["run_dir"] == str(broken)
    assert rows[3]["error"] is not None and rows[3]["bond"] is None


def test_rank_sums_behavioural_counters_across_episodes(tmp_path):
    run = _write_run(tmp_path, "r", 0.3, 0.3, episodes=[(1, 48, 58), (3, 74, 59)])
    row = rank_pair_runs([run])[0]
    assert row["joint_dig"] == 4
    assert row["co_action"] == 122
    assert row["proximity"] == 117
    assert row["episodes"] == 2


def test_cofired_flag_from_anvil_milestones(tmp_path):
    from mindforge.tools.pair_transplant import (
        pair_cofired, read_cofiring_milestones,
    )
    genuine = _write_run(tmp_path, "genuine", 0.25, 0.25, episodes=[(1, 5, 5)])
    with open(genuine / "final_metrics.json", "w") as f:
        json.dump({"milestones_per_agent": {
            "agent_0": ["m1_move_5", "m9_anvil_B1"],
            "agent_1": ["m9_anvil_B1", "m15_chestplate_equipped"],
        }}, f)
    control = _write_run(tmp_path, "control", 0.29, 0.29, episodes=[(6, 9, 9)])
    with open(control / "final_metrics.json", "w") as f:
        json.dump({"milestones_per_agent": {
            "agent_0": ["m1_move_5", "m2_dig_3_any"], "agent_1": ["m1_move_5"],
        }}, f)

    assert pair_cofired(genuine) is True
    assert read_cofiring_milestones(genuine) == [
        "m15_chestplate_equipped", "m9_anvil_B1"
    ]
    # Solo-earnable milestones and a high joint_dig do NOT make it co-firing.
    assert pair_cofired(control) is False
    assert read_cofiring_milestones(control) == []
    # Missing final_metrics.json must not raise.
    assert pair_cofired(tmp_path / "nonexistent") is False


def test_manifest_records_seat_pair_provenance(tmp_path):
    states = _pair_states()
    meta = {
        0: {"run_dir": "/runs/seed_123", "cofired": True,
            "cofiring_milestones": ["m9_anvil_B1"]},
        1: {"run_dir": "/runs/seed_1415", "cofired": True,
            "cofiring_milestones": ["m9_anvil_B1"]},
        2: {"run_dir": "/runs/seed_42", "cofired": False,
            "cofiring_milestones": []},
    }
    m = build_merged_manifest(states, build_slot_assignment(3), pair_meta=meta)
    sp = m["seat_pairs"]
    assert [p["cofired"] for p in sp] == [True, True, False]
    assert all(p["same_source_run"] for p in sp)
    # Seat 4/5 are the built-in control dyad.
    assert m["agents"]["4"]["source_cofired"] is False
    assert m["agents"]["0"]["source_cofired"] is True

    # Shuffled: no seat pair shares a source run, so "genuine" is undefined.
    ms = build_merged_manifest(
        states, build_slot_assignment(3, shuffled=True, seed=0),
        condition="shuffled", pair_meta=meta)
    assert all(p["cofired"] is None for p in ms["seat_pairs"])
    assert not any(p["same_source_run"] for p in ms["seat_pairs"])


def test_rank_surfaces_bond_vs_behaviour_mismatch(tmp_path):
    # The real 2026-08-05 smoke shape: higher bond, zero joint digs.
    hi_bond = _write_run(tmp_path, "seed_42", 0.2647, 0.2645,
                         episodes=[(0, 46, 39), (0, 57, 43)])
    lo_bond = _write_run(tmp_path, "seed_123", 0.2483, 0.2484,
                         episodes=[(1, 48, 58), (3, 74, 59)])
    rows = rank_pair_runs([lo_bond, hi_bond])
    assert rows[0]["run_dir"] == str(hi_bond)   # bond ranks it first
    assert rows[0]["joint_dig"] == 0            # despite never co-digging
    assert rows[1]["joint_dig"] == 4
