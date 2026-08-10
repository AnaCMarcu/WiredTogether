"""--comm-reward-scale (Experiment 2 noreward suite) — payout scaling.

At scale 0.0 messages still route, validity/spam rules still apply, and the
per-chamber comm milestones still FIRE as recorded events — but every payout
(base message reward + milestone reward) is zero, so talking can neither
manufacture bondable reward nor trip the milestone-success banner.

The default (1.0) must reproduce the historical tracker byte-for-byte —
that is the flag contract: without it, behavior is unchanged.
"""

import social_stubs  # noqa: F401  (sys.path bootstrap for src/mindforge)

from env.communication_rewards import (
    BASE_MSG_REWARD,
    CHAMBER_COMM_THRESHOLDS,
    CommunicationTracker,
)

CH2_POS = (0.0, 0.0, 20.0)  # inside ch2 bounds


def _spam(tracker, agent_id, n, start_step=0, gap=2):
    """Send n distinct valid messages, honoring the rate limit."""
    rewards_total = 0.0
    fired = []
    for k in range(n):
        r, m, _ = tracker.process_step(
            start_step + k * gap,
            {agent_id: f"message number {k} with plenty of length"},
            {agent_id: CH2_POS},
        )
        rewards_total += r.get(agent_id, 0.0)
        fired += m
    return rewards_total, fired


def test_default_scale_is_historical():
    t = CommunicationTracker(agent_ids=[0])
    assert t.reward_scale == 1.0
    total, fired = _spam(t, 0, 4)
    thresh, ms_reward, mid = CHAMBER_COMM_THRESHOLDS["ch2"]
    assert total == 4 * BASE_MSG_REWARD + ms_reward   # 4 msgs trip the ch2 milestone
    assert fired == [(0, mid, ms_reward)]


def test_scale_zero_pays_nothing_but_milestone_event_fires():
    t = CommunicationTracker(agent_ids=[0], reward_scale=0.0)
    total, fired = _spam(t, 0, 4)
    assert total == 0.0
    # The milestone EVENT is still recorded (metrics comparability),
    # with a zero payout attached.
    _, _, mid = CHAMBER_COMM_THRESHOLDS["ch2"]
    assert fired == [(0, mid, 0.0)]


def test_scale_zero_keeps_validity_rules():
    """Spam filtering is unchanged — only payouts are scaled."""
    t = CommunicationTracker(agent_ids=[0], reward_scale=0.0)
    # Same message twice: second is invalid (duplicate), so no speaker credit.
    r1, _, v1 = t.process_step(0, {0: "hello teammate over there"}, {0: CH2_POS})
    r2, _, v2 = t.process_step(2, {0: "hello teammate over there"}, {0: CH2_POS})
    assert v1 == {0} and v2 == set()
    assert r1.get(0, 0.0) == 0.0 and r2.get(0, 0.0) == 0.0


def test_half_scale_halves_payouts():
    t = CommunicationTracker(agent_ids=[0], reward_scale=0.5)
    total, fired = _spam(t, 0, 4)
    thresh, ms_reward, mid = CHAMBER_COMM_THRESHOLDS["ch2"]
    assert total == 4 * BASE_MSG_REWARD * 0.5 + ms_reward * 0.5
    assert fired == [(0, mid, ms_reward * 0.5)]
