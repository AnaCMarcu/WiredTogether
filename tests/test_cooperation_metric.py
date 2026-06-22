"""Tests for mindforge.env.cooperation_metric.CooperationMetric.

Pins the observer's actual behavior with hand-computed values: proximity /
joint-dig / co-action event counting, message filtering and the rolling
10-step buffer, damage routing from infos, joint-kill credit, milestone
context (entropy + comm_before_coop), Gini, per-chamber performance/fairness
(including the fixed _CH2_ANVIL_PREFIXES == ('m8_', 'm9_')), the 5-component
cooperation score, and the episode-summary structure.

All positions are (x, y, z); chamber membership is decided by z alone.
"""

import math

import pytest

from mindforge.env.cooperation_metric import CooperationMetric


def _metric(n=2):
    return CooperationMetric(agent_ids=list(range(n)))


def _step(cm, step, positions=None, actions=None, messages=None, infos=None):
    cm.observe_step(step, positions or {}, actions or {}, messages or {},
                    task_rewards={}, infos=infos)


# ── proximity ────────────────────────────────────────────────────────────────

def test_proximity_within_four_blocks_counts_symmetric_pair():
    """Two agents < 4 blocks apart yield one proximity event and a symmetric pair count."""
    cm = _metric(3)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (3, 0, 0), 2: (100, 0, 0)})
    assert cm.proximity_events == 1
    assert cm.pair_proximity[0][1] == 1
    assert cm.pair_proximity[1][0] == 1
    assert cm.pair_proximity[0][2] == 0


def test_proximity_exactly_four_blocks_not_counted():
    """The proximity threshold is strict (< 4.0): exactly 4.0 blocks apart is no event."""
    cm = _metric(2)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (4, 0, 0)})
    assert cm.proximity_events == 0
    assert cm.pair_proximity[0][1] == 0


# ── joint dig ────────────────────────────────────────────────────────────────

def test_joint_dig_within_three_blocks():
    """Two agents both 'Dig' within 3 blocks yield a joint_dig event + symmetric pair."""
    cm = _metric(2)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (2, 0, 0)},
          actions={0: "Dig", 1: "Dig"})
    assert cm.joint_dig_events == 1
    assert cm.pair_joint_dig[0][1] == 1
    assert cm.pair_joint_dig[1][0] == 1


def test_joint_dig_too_far_apart_not_counted():
    """Both digging but 3.5 blocks apart (>= 3.0) is no joint_dig event."""
    cm = _metric(2)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (3.5, 0, 0)},
          actions={0: "Dig", 1: "Dig"})
    assert cm.joint_dig_events == 0
    # 3.5 < 4.0 still registers as proximity, just not joint dig.
    assert cm.proximity_events == 1


def test_joint_dig_requires_two_diggers():
    """A single digging agent (the other acting differently) yields no joint_dig."""
    cm = _metric(2)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (1, 0, 0)},
          actions={0: "Dig", 1: "Attack"})
    assert cm.joint_dig_events == 0


# ── co-action ────────────────────────────────────────────────────────────────

def test_co_action_counts_one_per_step():
    """2+ agents sharing an action increments co_action_events once per step, not per pair."""
    cm = _metric(4)
    # Two duplicated actions in the same step -> still a single increment.
    _step(cm, 0, actions={0: "Forward", 1: "Forward", 2: "Dig", 3: "Dig"})
    assert cm.co_action_events == 1
    # All-distinct actions -> no increment.
    _step(cm, 1, actions={0: "Forward", 1: "Dig", 2: "Attack"})
    assert cm.co_action_events == 1
    # Falsy (empty-string) actions are ignored even when duplicated.
    _step(cm, 2, actions={0: "", 1: ""})
    assert cm.co_action_events == 1


# ── messages ─────────────────────────────────────────────────────────────────

def test_message_length_filter():
    """Only messages whose stripped length is >= 5 are counted per agent."""
    cm = _metric(3)
    _step(cm, 0, messages={0: "hi", 1: "hello", 2: "  ab  "})
    assert dict(cm.messages_per_agent) == {1: 1}
    _step(cm, 1, messages={1: "hello again"})
    assert cm.messages_per_agent[1] == 2


def test_recent_message_window_is_ten_steps_inclusive():
    """recent_messages keeps a message for exactly 10 steps (step - s <= 10), then drops it."""
    cm = _metric(2)
    _step(cm, 0, messages={0: "hello world"})
    _step(cm, 10)  # 10 - 0 <= 10 -> still buffered
    assert cm.recent_messages == [(0, 0, "hello world")]
    _step(cm, 11)  # 11 - 0 > 10 -> pruned
    assert cm.recent_messages == []


# ── dwell / action histograms ────────────────────────────────────────────────

def test_dwell_and_action_histograms_accrue_per_chamber():
    """Dwell steps and action counts accrue under the chamber the agent's z falls in."""
    cm = _metric(2)
    # z=20 -> ch2 (17..30); z=55 -> ch4 (52..62).
    _step(cm, 0, positions={0: (0, 0, 20), 1: (50, 0, 55)},
          actions={0: "Dig", 1: "Attack"})
    _step(cm, 1, positions={0: (0, 0, 20), 1: (50, 0, 55)},
          actions={0: "Dig", 1: "Forward"})
    assert cm.dwell_steps[0]["ch2"] == 2
    assert cm.dwell_steps[1]["ch4"] == 2
    assert cm.action_hist[0]["ch2"]["Dig"] == 2
    assert cm.action_hist[1]["ch4"] == {"Attack": 1, "Forward": 1}
    summary = cm.episode_summary(final_step=2)
    assert summary["dwell_steps"] == {"0": {"ch2": 2}, "1": {"ch4": 2}}
    assert summary["action_hist"]["1"]["ch4"] == {"Attack": 1, "Forward": 1}
    assert summary["chamber_entry_steps"] == {"ch2": 0, "ch4": 0}


# ── damage routing ───────────────────────────────────────────────────────────

def test_damage_event_routing_from_infos():
    """damage_events route ch4_zombie->ch4_damage and boss->ch5_damage with int-normalised attackers."""
    cm = _metric(3)
    _step(cm, 5, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": "agent_1", "amount": 7.5},
        {"target": "boss", "attacker": 0, "amount": 12.0},
        {"target": "ch4_zombie", "attacker": "zombie", "amount": 5.0},  # non-agent: dropped
        {"target": "ch4_zombie", "attacker": 9, "amount": 5.0},         # unknown id: dropped
    ]})
    assert dict(cm.ch4_damage) == {1: 7.5}
    assert dict(cm.ch5_damage) == {0: 12.0}
    # Dropped attackers never reach the per-target damage log either.
    assert len(cm._damage_log) == 2
    assert cm._damage_log[0] == {"step": 5, "attacker": 1,
                                 "target": "ch4_zombie", "amount": 7.5}


# ── observe_kill ─────────────────────────────────────────────────────────────

def test_observe_kill_joint_credit_within_five_step_window():
    """Two agents that damaged the target within 5 steps of the kill share symmetric joint_kill credit."""
    cm = _metric(3)
    _step(cm, 2, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": 0, "amount": 10.0}]})
    _step(cm, 4, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": 1, "amount": 10.0}]})
    cm.observe_kill(7, killer=1, target="ch4_zombie")  # 7-2=5 and 7-4=3, both <= 5
    assert cm.pair_joint_kill[0][1] == 1
    assert cm.pair_joint_kill[1][0] == 1
    # Non-boss target: no boss-overlap credit.
    assert cm.pair_boss_overlap[0][1] == 0
    assert cm._kill_log == [{"step": 7, "killer": 1, "target": "ch4_zombie"}]


def test_observe_kill_boss_target_also_credits_boss_overlap():
    """A boss kill with two recent attackers increments joint_kill AND ch5 boss-overlap symmetrically."""
    cm = _metric(2)
    _step(cm, 0, infos={"damage_events": [
        {"target": "boss", "attacker": "agent_0", "amount": 20.0},
        {"target": "boss", "attacker": "agent_1", "amount": 20.0}]})
    cm.observe_kill(3, killer="agent_0", target="boss")
    assert cm.pair_joint_kill[0][1] == 1
    assert cm.pair_boss_overlap[0][1] == 1
    assert cm.pair_boss_overlap[1][0] == 1


def test_observe_kill_damage_outside_window_excluded():
    """Damage dealt more than 5 steps before the kill is excluded from joint credit."""
    cm = _metric(2)
    _step(cm, 0, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": 0, "amount": 10.0},
        {"target": "ch4_zombie", "attacker": 1, "amount": 10.0}]})
    cm.observe_kill(6, killer=0, target="ch4_zombie")  # 6-0=6 > 5: both excluded
    assert cm.pair_joint_kill[0][1] == 0


def test_observe_kill_single_attacker_no_pair():
    """A kill with a single recent attacker produces no joint_kill pair."""
    cm = _metric(2)
    _step(cm, 0, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": 0, "amount": 20.0}]})
    cm.observe_kill(1, killer=0, target="ch4_zombie")
    assert cm.pair_joint_kill[0][1] == 0
    assert cm.pair_joint_kill[1][0] == 0


# ── routed messages ──────────────────────────────────────────────────────────

def test_routed_messages_directed_and_self_excluded():
    """routed_messages fill the directed pair_messages matrix; self-messages and outsiders dropped."""
    cm = _metric(3)
    _step(cm, 0, infos={"routed_messages": [
        {"sender": "agent_0", "receiver": "agent_1"},
        {"sender": 0, "receiver": 1},
        {"sender": 2, "receiver": 0},
        {"sender": 1, "receiver": 1},   # self: excluded
        {"sender": 0, "receiver": 7},   # unknown receiver: excluded
    ]})
    assert cm.pair_messages[0][1] == 2  # directed, NOT mirrored
    assert cm.pair_messages[1][0] == 0
    assert cm.pair_messages[2][0] == 1
    assert cm.pair_messages[1][1] == 0


# ── observe_milestone ────────────────────────────────────────────────────────

def test_observe_milestone_comm_before_coop_and_entropy():
    """comm_before_coop is set when a contributor messaged within the rolling window; entropy is -sum p ln p."""
    cm = _metric(2)
    _step(cm, 0, messages={0: "hello world"})
    cm.observe_milestone(1, "m8_anvil_A1", ["agent_0", 1])
    rec = cm.milestone_log[0]
    assert rec["milestone"] == "m8_anvil_A1"
    assert rec["contributor_count"] == 2
    assert rec["comm_before_coop"] is True  # 'agent_0' normalised to 0, matches buffered sender
    assert rec["contribution_entropy"] == pytest.approx(math.log(2))
    # After the buffer is pruned, an identical milestone has no prior comm.
    _step(cm, 20)  # 20 - 0 > 10 -> message dropped
    cm.observe_milestone(20, "m9_anvil_B1", [0, 1])
    assert cm.milestone_log[1]["comm_before_coop"] is False
    # Non-contributor message does not set the flag.
    _step(cm, 21, messages={1: "hello world"})
    cm.observe_milestone(21, "m17_press", [0])
    assert cm.milestone_log[2]["comm_before_coop"] is False
    assert cm.milestone_log[2]["contribution_entropy"] == pytest.approx(0.0)


# ── static helpers ───────────────────────────────────────────────────────────

def test_gini_values():
    """_gini: equal split -> 0.0; one-takes-all of three -> 2/3; empty or zero-sum -> 0.0."""
    g = CooperationMetric._gini
    assert g({"a": 5, "b": 5, "c": 5}) == pytest.approx(0.0)
    assert g({"a": 9, "b": 0, "c": 0}) == pytest.approx(2.0 / 3.0)
    assert g({}) == 0.0
    assert g({"a": 0, "b": 0}) == 0.0


def test_to_int_id_normalisation():
    """_to_int_id: 'agent_2'->2, 'agent2'->2, int passthrough, non-numeric string returned unchanged."""
    f = CooperationMetric._to_int_id
    assert f("agent_2") == 2
    assert f("agent2") == 2
    assert f(3) == 3
    assert f("zombie") == "zombie"


# ── comm efficacy ────────────────────────────────────────────────────────────

def test_comm_efficacy_fraction_over_multi_contributor_milestones():
    """_comm_efficacy = fraction of >=2-contributor milestones with prior comm; solo milestones excluded."""
    cm = _metric(2)
    assert cm._comm_efficacy() == 0.0  # no milestones at all
    _step(cm, 0, messages={0: "hello team"})
    cm.observe_milestone(0, "m8_anvil_A1", [0, 1])      # multi, comm -> counted True
    _step(cm, 20)                                        # prune the buffer
    cm.observe_milestone(20, "m9_anvil_B1", [0, 1])     # multi, no comm -> counted False
    cm.observe_milestone(20, "m17_press_1", [0])        # solo -> excluded
    assert cm._comm_efficacy() == pytest.approx(0.5)


# ── per-chamber scoring ──────────────────────────────────────────────────────

def test_ch4_performance_and_fairness_hand_values():
    """Ch4: damage {0:30, 1:30} -> performance 60/60 = 1.0; fairness with idle agent 2 = 1 - 1/3."""
    cm = _metric(3)
    _step(cm, 0, infos={"damage_events": [
        {"target": "ch4_zombie", "attacker": 0, "amount": 30.0},
        {"target": "ch4_zombie", "attacker": 1, "amount": 30.0}]})
    assert cm._chamber_performance("ch4") == pytest.approx(1.0)
    # counts {0:30, 1:30, 2:0}: sorted [0,30,30], gini = 2*150/(3*60) - 4/3 = 1/3.
    assert cm._chamber_fairness("ch4") == pytest.approx(1.0 - 1.0 / 3.0)


def test_ch2_performance_counts_both_anvil_prefixes():
    """Ch2 performance counts both m8_ and m9_ anvil milestones (pins fixed _CH2_ANVIL_PREFIXES)."""
    assert CooperationMetric._CH2_ANVIL_PREFIXES == ("m8_", "m9_")
    cm = _metric(2)
    cm.observe_milestone(1, "m8_anvil_A1", [0, 1])
    assert cm._chamber_performance("ch2") == pytest.approx(0.5)
    cm.observe_milestone(2, "m9_anvil_B1", [0, 1])
    assert cm._chamber_performance("ch2") == pytest.approx(1.0)


def test_cooperation_score_is_mean_of_five_components():
    """cooperation_score == (comm_eff + 4 chamber perf*fair terms)/5; unreached chambers contribute 0."""
    cm = _metric(2)
    # Both agents in ch2 (z=20) -> ch2 reached; agent 0 messages before the milestone.
    _step(cm, 0, positions={0: (0, 0, 20), 1: (1, 0, 20)},
          messages={0: "hello world"})
    cm.observe_milestone(1, "m8_anvil_A1", [0, 1])
    # comm_eff = 1.0; ch2 = perf 0.5 * fairness 1.0 (equal contributions);
    # ch3/ch4/ch5 unreached -> 0 each.
    assert cm._cooperation_score() == pytest.approx((1.0 + 0.5) / 5.0)
    bd = cm._cooperation_breakdown()
    assert bd["comm_eff"] == pytest.approx(1.0)
    assert bd["ch2"] == {"reached": True, "performance": 0.5,
                         "fairness": 1.0, "score": 0.5}
    assert bd["ch3"] == {"reached": False, "performance": 0.0,
                         "fairness": 0.0, "score": 0.0}
    comp_sum = bd["comm_eff"] + sum(bd[ch]["score"] for ch in ("ch2", "ch3", "ch4", "ch5"))
    assert cm._cooperation_score() == pytest.approx(comp_sum / 5.0)
    assert cm.episode_summary(final_step=2)["cooperation_score"] == pytest.approx(0.3)


# ── episode summary ──────────────────────────────────────────────────────────

def test_episode_summary_pair_interaction_planes():
    """pair_interaction has exactly 5 named N×N planes ordered by agent_ids."""
    cm = _metric(3)
    _step(cm, 0, positions={0: (0, 0, 0), 1: (1, 0, 0)},  # proximity 0<->1
          infos={"routed_messages": [{"sender": 2, "receiver": 0}]})
    summary = cm.episode_summary(final_step=1, hebbian_weights=[[0.0]])
    pi = summary["pair_interaction"]
    assert set(pi.keys()) == {"messages", "joint_dig", "proximity",
                              "joint_kill", "ch5_damage_overlap"}
    for plane in pi.values():
        assert len(plane) == 3 and all(len(row) == 3 for row in plane)
    assert pi["proximity"] == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    assert pi["messages"] == [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
    assert pi["joint_dig"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert summary["final_step"] == 1
    assert summary["hebbian_W"] == [[0.0]]
