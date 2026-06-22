"""Post-hoc cooperation/communication evaluation over synthetic run dirs.

Pins `mindforge.agent_modules.coop_eval.compute_coop_metrics` and
`mindforge.agent_modules.comm_eval.compute_comm_metrics` (plus their pure
helpers) against hand-built episode directories that follow the exact
on-disk layout written by `mindforge.env.episode_logger.EpisodeLogger`:

    runs/<run_id>/episodes/ep_{N:04d}/
        step_log.csv          (columns: step, agent_id, chamber, pos_x..z,
                               action, reward_task, reward_comm,
                               wielded_item, hp, message)
        event_log.jsonl       ({"step", "type": "milestone"|"comm_milestone",
                               "id"|"milestone", "contributors"|"agent",
                               "reward"})
        messages.jsonl        ({"t", "sender", "receiver", "text", "tokens",
                               "valid", "chamber", "routing", ...})
        summary.json + episode_summary.json   (CooperationMetric summary)

All expected values are hand-computed. Message clustering is forced onto
its hash-bin fallback (no sentence-transformers / sklearn) so everything
stays offline and deterministic.
"""

import csv
import json
import math
import sys
from collections import Counter

import pytest

from mindforge.agent_modules.comm_eval import (
    _agent_idx,
    _cluster_messages,
    _entropy,
    _mutual_information,
    compute_comm_metrics,
)
from mindforge.agent_modules.coop_eval import compute_coop_metrics

# Exact column order written by EpisodeLogger.__init__ (step_log.csv).
STEP_FIELDS = [
    "step", "agent_id", "chamber",
    "pos_x", "pos_y", "pos_z",
    "action", "reward_task", "reward_comm",
    "wielded_item", "hp", "message",
]


# ── Synthetic run-dir builder ────────────────────────────────────────────────
def _ep_dir(run_root, idx):
    d = run_root / "episodes" / f"ep_{idx:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_episode(run_root, idx, summary=None, events=None, messages=None,
                   steps=None, summary_name="summary.json",
                   raw_event_lines=()):
    """Create episodes/ep_{idx:04d}/ with the EpisodeLogger file layout."""
    d = _ep_dir(run_root, idx)
    if summary is not None:
        (d / summary_name).write_text(json.dumps(summary), encoding="utf-8")
    if events is not None or raw_event_lines:
        lines = [json.dumps(e) for e in (events or [])]
        lines.extend(raw_event_lines)
        (d / "event_log.jsonl").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    if messages is not None:
        (d / "messages.jsonl").write_text(
            "".join(json.dumps(m) + "\n" for m in messages), encoding="utf-8")
    if steps is not None:
        with open(d / "step_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STEP_FIELDS)
            w.writeheader()
            for row in steps:
                w.writerow({k: row.get(k, "") for k in STEP_FIELDS})
    return d


@pytest.fixture
def offline_clustering(monkeypatch):
    """Force _cluster_messages onto its hash-bin fallback by making
    `import sentence_transformers` raise (None in sys.modules => ImportError)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


# ── coop_eval: milestone credit ──────────────────────────────────────────────
def test_equal_credit_split_team_milestone(tmp_path):
    """m19_all_in_communal (equal rule, reward 100, two contributors) is split 50/50; malformed JSONL lines are skipped."""
    _write_episode(
        tmp_path, 0, summary={},
        events=[{"step": 5, "type": "milestone", "id": "m19_all_in_communal",
                 "contributors": ["agent_0", "agent_1"], "reward": 100}],
        raw_event_lines=["{this is not json", ""],
    )
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["milestone_credit"]["agent_0"]["m19_all_in_communal"] == pytest.approx(50.0)
    assert res["milestone_credit"]["agent_1"]["m19_all_in_communal"] == pytest.approx(50.0)
    assert res["milestone_credit_total"] == pytest.approx(
        {"agent_0": 50.0, "agent_1": 50.0})


def test_damage_share_split_boss_defeated(tmp_path):
    """m27_boss_defeated reward 300 with ch5 damage 40/20 splits 200/100 when contributor names match the summary's damage keys."""
    _write_episode(
        tmp_path, 0,
        summary={"ch5_damage_per_agent": {"0": 40.0, "1": 20.0}},
        events=[{"step": 9, "type": "milestone", "id": "m27_boss_defeated",
                 "contributors": ["0", "1"], "reward": 300}],
    )
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["milestone_credit"]["0"]["m27_boss_defeated"] == pytest.approx(200.0)
    assert res["milestone_credit"]["1"]["m27_boss_defeated"] == pytest.approx(100.0)
    # Damage carried through to the run-level report unchanged.
    assert res["ch5_damage_per_agent"] == pytest.approx({"0": 40.0, "1": 20.0})


def test_damage_share_falls_back_to_equal_without_damage_info(tmp_path):
    """m27_boss_defeated with no ch5_damage_per_agent in any summary falls back to an equal 150/150 split."""
    _write_episode(
        tmp_path, 0, summary={},
        events=[{"step": 9, "type": "milestone", "id": "m27_boss_defeated",
                 "contributors": ["agent_0", "agent_1"], "reward": 300}],
    )
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["milestone_credit"]["agent_0"]["m27_boss_defeated"] == pytest.approx(150.0)
    assert res["milestone_credit"]["agent_1"]["m27_boss_defeated"] == pytest.approx(150.0)


def test_damage_share_real_pipeline_keys_match(tmp_path):
    """FIXED BUG: real-pipeline contributor names ('agent0' from Lua, 'agent_0'
    from Python) are normalised against the summary damage keys ('0'/'1' —
    JSON-stringified int ids) via coop_eval._agent_key, so the boss-damage rule
    splits 300 @ {40, 20} into 200/100 instead of degrading to equal."""
    _write_episode(
        tmp_path, 0,
        summary={"ch5_damage_per_agent": {"0": 40.0, "1": 20.0}},
        events=[{"step": 9, "type": "milestone", "id": "m27_boss_defeated",
                 "contributors": ["agent0", "agent1"], "reward": 300}],
    )
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["milestone_credit"]["agent0"]["m27_boss_defeated"] == pytest.approx(200.0)
    assert res["milestone_credit"]["agent1"]["m27_boss_defeated"] == pytest.approx(100.0)


def test_single_agent_event_credited_fully(tmp_path):
    """A comm_milestone event with only an 'agent' field (real schema) credits that agent with the full reward."""
    _write_episode(
        tmp_path, 0, summary={},
        events=[{"step": 4, "type": "comm_milestone",
                 "milestone": "m_comm_ch2", "agent": "agent_1", "reward": 7.5}],
    )
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["milestone_credit"] == {"agent_1": {"m_comm_ch2": pytest.approx(7.5)}}
    assert res["milestone_credit_total"] == {"agent_1": pytest.approx(7.5)}


def test_credit_gini_single_earner(tmp_path):
    """With 3 agents in the credit table and one earner, credit Gini == (N-1)/N == 2/3."""
    _write_episode(
        tmp_path, 0, summary={},
        events=[
            {"step": 5, "type": "milestone", "id": "m19_all_in_communal",
             "contributors": ["agent_0"], "reward": 90},
            # Zero-reward event drags agents 1 and 2 into the table at 0.0.
            {"step": 6, "type": "milestone", "id": "m23_all_alive_ch4",
             "contributors": ["agent_1", "agent_2"], "reward": 0},
        ],
    )
    res = compute_coop_metrics(tmp_path, num_agents=3)
    assert res["milestone_credit_total"] == pytest.approx(
        {"agent_0": 90.0, "agent_1": 0.0, "agent_2": 0.0})
    assert res["credit_gini"] == pytest.approx(2.0 / 3.0)


# ── coop_eval: cross-episode aggregation ─────────────────────────────────────
def test_pair_tensor_summed_and_missing_summary_skipped(tmp_path):
    """Pair-interaction planes are summed elementwise across episodes (episode_summary.json fallback honoured); an episode dir without any summary file is skipped entirely, including its events."""
    _write_episode(tmp_path, 0,
                   summary={"pair_interaction": {"messages": [[0, 2], [1, 0]]}})
    _write_episode(tmp_path, 1,
                   summary={"pair_interaction": {"messages": [[0, 3], [4, 0]],
                                                 "proximity": [[0, 1], [1, 0]]}},
                   summary_name="episode_summary.json")
    # No summary at all -> whole episode skipped (milestone NOT credited).
    _write_episode(tmp_path, 2,
                   events=[{"step": 1, "type": "milestone",
                            "id": "m19_all_in_communal",
                            "contributors": ["agent_0"], "reward": 100}])
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["pair_interaction_total"]["messages"] == [[0, 5], [5, 0]]
    assert res["pair_interaction_total"]["proximity"] == [[0, 1], [1, 0]]
    assert res["pair_interaction_total"]["joint_dig"] == [[0, 0], [0, 0]]
    assert res["milestone_credit"] == {}
    assert res["credit_gini"] == 0.0


def test_dwell_and_action_hist_aggregated_across_episodes(tmp_path):
    """dwell_steps_total and action_hist_total sum per-agent per-chamber counts over all episodes."""
    _write_episode(tmp_path, 0, summary={
        "dwell_steps": {"0": {"ch1": 5}},
        "action_hist": {"0": {"ch1": {"Dig": 2}}},
    })
    _write_episode(tmp_path, 1, summary={
        "dwell_steps": {"0": {"ch1": 3, "ch2": 2}, "1": {"ch1": 1}},
        "action_hist": {"0": {"ch1": {"Dig": 1, "Forward": 4}}},
    })
    res = compute_coop_metrics(tmp_path, num_agents=2)
    assert res["dwell_steps_total"] == {"0": {"ch1": 8, "ch2": 2},
                                        "1": {"ch1": 1}}
    assert res["action_hist_total"] == {"0": {"ch1": {"Dig": 3, "Forward": 4}}}


# ── comm_eval: pure helpers ──────────────────────────────────────────────────
def test_entropy_uniform_over_k_is_ln_k():
    """_entropy of a uniform Counter over k symbols is ln(k) nats; empty Counter is 0.0."""
    assert _entropy(Counter({"a": 2, "b": 2, "c": 2})) == pytest.approx(math.log(3))
    assert _entropy(Counter()) == 0.0


def test_mutual_information_identical_and_independent():
    """_mutual_information of identical (x, x) pairs equals H(X); an exact product distribution gives 0."""
    identical = [("a", "a"), ("a", "a"), ("b", "b"), ("c", "c")]
    h_x = _entropy(Counter(x for x, _ in identical))
    assert h_x == pytest.approx(1.5 * math.log(2))
    assert _mutual_information(identical) == pytest.approx(h_x)
    independent = [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
    assert _mutual_information(independent) == pytest.approx(0.0, abs=1e-12)
    assert _mutual_information([]) == 0.0


def test_agent_idx_parsing():
    """_agent_idx: 'agent_2'->2, int passthrough, bare digits parse, garbage/None/'all' -> -1."""
    assert _agent_idx("agent_2") == 2
    assert _agent_idx(7) == 7
    assert _agent_idx("0") == 0
    assert _agent_idx("garbage") == -1
    assert _agent_idx(None) == -1
    assert _agent_idx("all") == -1


def test_cluster_messages_hash_fallback(offline_clustering):
    """Offline fallback clustering maps identical texts to one bin in [0, 16); empty input -> []."""
    out = _cluster_messages(["alpha one", "alpha one", "beta two"])
    assert len(out) == 3
    assert out[0] == out[1]
    assert all(0 <= c < 16 for c in out)
    assert _cluster_messages([]) == []


# ── comm_eval: compute_comm_metrics over a synthetic run ─────────────────────
def test_compute_comm_metrics_synthetic_run(tmp_path, offline_clustering):
    """Directed pair counts exclude self/broadcast, token totals + tokens-per-milestone + routing breakdown match hand counts, and constant-text senders pin zero entropy/MI."""
    messages = [
        {"t": 0, "sender": "agent_0", "receiver": "agent_1",
         "text": "go north now", "tokens": 3, "valid": True,
         "routing": "model", "chamber": "ch1"},
        {"t": 1, "sender": "agent_0", "receiver": "agent_1",
         "text": "go north now", "tokens": 3, "valid": False,
         "routing": "hebbian_fallback", "chamber": "ch1"},
        {"t": 1, "sender": "agent_1", "receiver": "agent_0",
         "text": "ok coming over", "tokens": 3, "valid": True,
         "routing": "model", "chamber": "ch2"},
        # Broadcast: receiver 'all' parses to -1 -> not in the pair matrix.
        {"t": 2, "sender": "agent_0", "receiver": "all",
         "text": "go team go", "tokens": 3, "valid": True,
         "routing": "random_fallback", "chamber": "ch1"},
        # Self-message (si == ri) excluded from pair matrix; no 'routing'
        # key -> counted under the default bucket "model".
        {"t": 2, "sender": "agent_1", "receiver": "agent_1",
         "text": "ok stay here", "tokens": 2, "valid": True,
         "chamber": "ch2"},
    ]
    a1_actions = {0: "NoOp", 1: "Forward", 2: "Dig", 3: "Jump"}
    steps = [{"step": s, "agent_id": a, "chamber": "ch1",
              "action": ("Forward" if a == 0 else a1_actions[s]),
              "reward_task": 0.0, "reward_comm": 0.0}
             for s in range(4) for a in (0, 1)]
    events = [
        {"step": 3, "type": "milestone", "id": "m19_all_in_communal",
         "contributors": ["agent_0", "agent_1"], "reward": 100},
        {"step": 4, "type": "comm_milestone", "milestone": "m_comm_ch1",
         "agent": "agent_0", "reward": 2},
    ]
    _write_episode(tmp_path, 0, summary={}, events=events,
                   messages=messages, steps=steps)

    res = compute_comm_metrics(tmp_path, num_agents=2)

    assert res["total_messages"] == 5
    assert res["valid_messages"] == 4
    assert res["total_tokens"] == 14
    assert res["pair_message_count"] == [[0, 2], [1, 0]]
    assert res["routing_breakdown"] == {"model": 3, "hebbian_fallback": 1,
                                        "random_fallback": 1}
    # 14 tokens / 2 milestone events (milestone + comm_milestone).
    assert res["tokens_per_milestone"] == pytest.approx(7.0)

    # Speaker consistency: every sender always uses the same head word and
    # length bucket -> one hash bin -> entropy exactly 0, and a constant
    # cluster has zero MI with the chamber, whatever the bin id is.
    sc = res["speaker_consistency"]
    assert set(sc) == {"agent_0", "agent_1"}
    assert sc["agent_0"]["num_msgs"] == 3
    assert sc["agent_0"]["entropy"] == pytest.approx(0.0)
    assert sc["agent_0"]["mi_chamber"] == pytest.approx(0.0)
    assert sc["agent_1"]["num_msgs"] == 2
    assert sc["agent_1"]["entropy"] == pytest.approx(0.0)

    # Instantaneous coordination: constant sender cluster -> MI 0 for 0->1
    # (next actions Forward@t1, Dig@t2 exist in step_log); 1->0 has a single
    # (cluster, action) pair -> MI 0.
    assert set(res["instantaneous_coord"]) == {"0->1", "1->0"}
    assert res["instantaneous_coord"]["0->1"] == pytest.approx(0.0)
    assert res["instantaneous_coord"]["1->0"] == pytest.approx(0.0)


def test_compute_comm_metrics_no_milestones_and_empty_episode(tmp_path, offline_clustering):
    """Missing step_log/event_log files and an empty episode dir are tolerated; tokens_per_milestone is None without milestones."""
    _write_episode(tmp_path, 0, messages=[
        {"t": 0, "sender": "agent_0", "receiver": "agent_1",
         "text": "hi there", "tokens": 2, "valid": True, "chamber": "ch1"},
    ])
    _ep_dir(tmp_path, 1)  # fully empty episode dir
    res = compute_comm_metrics(tmp_path, num_agents=2)
    assert res["total_messages"] == 1
    assert res["total_tokens"] == 2
    assert res["tokens_per_milestone"] is None
    assert res["pair_message_count"] == [[0, 1], [0, 0]]
    # No step_log -> no next action -> empty pair list -> MI 0.0.
    assert res["instantaneous_coord"]["0->1"] == 0.0
