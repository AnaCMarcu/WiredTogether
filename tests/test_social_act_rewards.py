"""--social-act-rewards (act-reward symmetry suite) — the obs/imit tracker.

Observation and imitation acts must be paid EXACTLY like communication:
same base reward and cap, same rate limit, same per-chamber milestone
thresholds and values (as m_obs_chN / m_imit_chN), same rescued-act
semantics (no base pay, chamber count still advances). Everything is off
unless the flag instantiates the tracker — the default reward stream is
untouched.
"""

import social_stubs  # noqa: F401  (sys.path bootstrap for src/mindforge)

from env.communication_rewards import (
    BASE_MSG_CAP,
    BASE_MSG_REWARD,
    CHAMBER_COMM_THRESHOLDS,
    RATE_LIMIT_STEPS,
)
from env.social_act_rewards import CHAMBER_ACT_THRESHOLDS, SocialActRewardTracker

CH2_POS = (0.0, 0.0, 20.0)   # inside ch2 bounds
NOWHERE = (0.0, 0.0, 999.0)  # outside every chamber


def _tick(tracker, step, agent, act, rescued=False, pos=CH2_POS):
    r, m = tracker.process_step(step, [(agent, act, rescued)], {agent: pos})
    return r.get(agent, 0.0), m


def test_mirrors_comm_constants():
    """The symmetry IS the spec: thresholds/rewards identical to comm's."""
    for act in ("obs", "imit"):
        for ch, (thr, rw, mid) in CHAMBER_COMM_THRESHOLDS.items():
            a_thr, a_rw, a_mid = CHAMBER_ACT_THRESHOLDS[act][ch]
            assert (a_thr, a_rw) == (thr, rw)
            assert a_mid == f"m_{act}_{ch}"


def test_base_pay_and_chamber_milestone():
    t = SocialActRewardTracker(agent_ids=[0])
    total, fired = 0.0, []
    for k in range(4):
        r, m = _tick(t, k * RATE_LIMIT_STEPS, 0, "obs")
        total += r
        fired += m
    thr, ms_reward, _ = CHAMBER_COMM_THRESHOLDS["ch2"]
    assert total == 4 * BASE_MSG_REWARD + ms_reward
    assert fired == [(0, "m_obs_ch2", ms_reward)]


def test_rate_limit_mirrors_comm():
    t = SocialActRewardTracker(agent_ids=[0])
    r1, _ = _tick(t, 0, 0, "obs")
    r2, _ = _tick(t, 1, 0, "obs")          # < RATE_LIMIT_STEPS later
    assert r1 == BASE_MSG_REWARD and r2 == 0.0
    # rate-limited acts are invisible: chamber count did not advance
    assert t.chamber_counts[("obs", 0, "ch2")] == 1


def test_acts_tracked_independently_per_type():
    """obs and imit have separate caps, rate limits and milestones."""
    t = SocialActRewardTracker(agent_ids=[0])
    r, m = t.process_step(0, [(0, "obs", False), (0, "imit", False)],
                          {0: CH2_POS})
    assert r[0] == 2 * BASE_MSG_REWARD     # both paid on the same step
    for k in range(1, 4):
        r, m = t.process_step(k * RATE_LIMIT_STEPS,
                              [(0, "obs", False), (0, "imit", False)],
                              {0: CH2_POS})
    fired = {mid for _, mid, _ in m}
    assert fired == {"m_obs_ch2", "m_imit_ch2"}


def test_rescued_act_no_base_pay_but_counts_to_milestone():
    t = SocialActRewardTracker(agent_ids=[0])
    total = 0.0
    fired = []
    for k in range(4):
        r, m = _tick(t, k * RATE_LIMIT_STEPS, 0, "imit", rescued=True)
        total += r
        fired += m
    thr, ms_reward, _ = CHAMBER_COMM_THRESHOLDS["ch2"]
    assert total == ms_reward              # milestone only, no base pay
    assert fired == [(0, "m_imit_ch2", ms_reward)]


def test_cap_mirrors_comm():
    t = SocialActRewardTracker(agent_ids=[0])
    paid = 0
    for k in range(BASE_MSG_CAP + 10):
        r, _ = _tick(t, k * RATE_LIMIT_STEPS, 0, "obs", pos=NOWHERE)
        paid += r > 0
    assert paid == BASE_MSG_CAP


def test_outside_chambers_no_milestone():
    t = SocialActRewardTracker(agent_ids=[0])
    for k in range(10):
        _, m = _tick(t, k * RATE_LIMIT_STEPS, 0, "obs", pos=NOWHERE)
        assert m == []


def test_unknown_act_ignored():
    t = SocialActRewardTracker(agent_ids=[0])
    r, m = t.process_step(0, [(0, "communicate", False)], {0: CH2_POS})
    assert r == {} and m == []
