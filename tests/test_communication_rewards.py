"""Tests for mindforge.env.communication_rewards.CommunicationTracker.

Pins the exact reward semantics of the Five Chambers communication shaping:
  - base reward BASE_MSG_REWARD = 0.5 per valid message, capped at
    BASE_MSG_CAP = 50 paid messages per agent (25.0 total base reward);
  - validity = len(strip) >= MIN_MSG_LEN (5), not a duplicate of the sender's
    last VALID message, and >= RATE_LIMIT_STEPS (2) steps since it;
  - chamber comm milestones from CHAMBER_COMM_THRESHOLDS, each (4 msgs,
    reward, milestone_id): ch1/ch2/ch4/ch5 pay 10.0, ch3 pays 20.0;
  - bad-target speakers stay "valid" (message routed) but earn no base reward;
  - chamber z-bands: ch1 [0,15], ch2 [17,30], ch3 [32,50], ch4 [52,62],
    ch5 [64,72]; positions in the 1-block gaps map to no chamber.
"""

import pytest

from mindforge.env.communication_rewards import (
    BASE_MSG_CAP,
    BASE_MSG_REWARD,
    CHAMBER_COMM_THRESHOLDS,
    MIN_MSG_LEN,
    RATE_LIMIT_STEPS,
    CommunicationTracker,
)

A = "agent_0"


def _tracker():
    return CommunicationTracker([A, "agent_1"])


def test_valid_message_base_reward():
    """A valid message earns exactly +0.5 and puts the speaker in valid_speakers."""
    t = _tracker()
    rewards, milestones, valid = t.process_step(
        10, {A: "hello team"}, {A: (0.0, 5.0, 5.0)}
    )
    assert rewards == {A: pytest.approx(BASE_MSG_REWARD)}
    assert rewards[A] == pytest.approx(0.5)
    assert milestones == []
    assert valid == {A}
    assert isinstance(rewards, dict) and isinstance(milestones, list) and isinstance(valid, set)
    assert t.total_valid_msgs[A] == 1


def test_short_message_invalid():
    """A message shorter than MIN_MSG_LEN=5 chars earns nothing."""
    t = _tracker()
    rewards, milestones, valid = t.process_step(10, {A: "hi"}, {A: (0, 5, 5)})
    assert rewards == {}
    assert milestones == []
    assert valid == set()
    assert t.total_valid_msgs[A] == 0


def test_whitespace_padded_short_message_invalid():
    """Length is checked on the stripped message: '  hi  ' is still too short."""
    t = _tracker()
    assert len("  hi  ") >= MIN_MSG_LEN  # raw length would pass; strip() must apply
    rewards, _, valid = t.process_step(10, {A: "  hi  "}, {A: (0, 5, 5)})
    assert rewards == {}
    assert valid == set()


def test_duplicate_of_last_message_invalid_then_new_text_valid():
    """Repeating the sender's last valid message is invalid; new text is valid again."""
    t = _tracker()
    r1, _, v1 = t.process_step(10, {A: "follow me east"}, {})
    assert v1 == {A} and r1[A] == pytest.approx(0.5)
    # Same text, well past the rate limit -> rejected as duplicate.
    r2, _, v2 = t.process_step(20, {A: "follow me east"}, {})
    assert r2 == {} and v2 == set()
    # Different text afterwards -> valid again.
    r3, _, v3 = t.process_step(30, {A: "now go north"}, {})
    assert v3 == {A} and r3[A] == pytest.approx(0.5)
    assert t.total_valid_msgs[A] == 2


def test_rate_limit_two_steps_between_valid_messages():
    """A valid message at step 10 blocks step 11 (gap 1 < 2) but not step 12."""
    t = _tracker()
    _, _, v10 = t.process_step(10, {A: "first message"}, {})
    assert v10 == {A}
    r11, _, v11 = t.process_step(11, {A: "second message"}, {})
    assert v11 == set() and r11 == {}
    # Gap of exactly RATE_LIMIT_STEPS=2 is allowed.
    r12, _, v12 = t.process_step(12, {A: "second message"}, {})
    assert v12 == {A} and r12[A] == pytest.approx(0.5)
    assert RATE_LIMIT_STEPS == 2


def test_invalid_message_does_not_update_last_msg_state():
    """An invalid (rate-limited) message leaves last_msg/last_msg_step untouched."""
    t = _tracker()
    t.process_step(10, {A: "message one"}, {})
    assert t.last_msg[A] == "message one" and t.last_msg_step[A] == 10
    # Rate-limited attempt: state writes happen AFTER the validity gate.
    _, _, v = t.process_step(11, {A: "message two"}, {})
    assert v == set()
    assert t.last_msg[A] == "message one"
    assert t.last_msg_step[A] == 10
    # Because state was not overwritten, the identical text is valid at step 13.
    r, _, v2 = t.process_step(13, {A: "message two"}, {})
    assert v2 == {A} and r[A] == pytest.approx(0.5)


def test_base_cap_50_paid_messages_chamber_counts_keep_accruing():
    """Only 50 base rewards (25.0) are paid; chamber counts pass the cap to 51."""
    t = _tracker()
    pos = {A: (0.0, 5.0, 5.0)}  # inside ch1 z-band [0, 15]
    base_total = 0.0
    milestone_total = 0.0
    last_rewards = None
    for i in range(51):  # alternate texts, 3 steps apart -> all 51 are valid
        text = f"update alpha {i}" if i % 2 == 0 else f"update bravo {i}"
        rewards, milestones, valid = t.process_step(10 + 3 * i, {A: text}, pos)
        assert valid == {A}
        milestone_total += sum(m[2] for m in milestones)
        base_total += rewards.get(A, 0.0) - sum(m[2] for m in milestones)
        last_rewards = rewards
    assert t.total_valid_msgs[A] == BASE_MSG_CAP == 50
    assert base_total == pytest.approx(50 * BASE_MSG_REWARD) == pytest.approx(25.0)
    # ch1 milestone (4 msgs, 10.0) fired exactly once along the way.
    assert milestone_total == pytest.approx(10.0)
    # Chamber accrual is NOT capped: all 51 valid messages counted.
    assert t.chamber_msg_counts[A]["ch1"] == 51
    # The 51st message paid nothing (cap reached, milestone already fired).
    assert last_rewards == {}


def test_ch2_comm_milestone_fires_once_on_fourth_message():
    """4th valid message inside ch2 (z in [17,30]) fires m_comm_ch2 for +10.0, once."""
    assert CHAMBER_COMM_THRESHOLDS["ch2"] == (4, 10.0, "m_comm_ch2")
    t = _tracker()
    pos = {A: (3.0, 8.0, 20.0)}  # 17 <= z=20 <= 30
    for i in range(3):
        rewards, milestones, _ = t.process_step(10 + 3 * i, {A: f"anvil plan {i}"}, pos)
        assert milestones == []
        assert rewards[A] == pytest.approx(0.5)
    # 4th message: base + milestone.
    rewards, milestones, _ = t.process_step(19, {A: "anvil plan 3"}, pos)
    assert milestones == [(A, "m_comm_ch2", 10.0)]
    assert rewards[A] == pytest.approx(0.5 + 10.0)
    # 5th message: milestone does not refire.
    rewards, milestones, _ = t.process_step(22, {A: "anvil plan 4"}, pos)
    assert milestones == []
    assert rewards[A] == pytest.approx(0.5)
    assert t.fired_milestones[A] == {"m_comm_ch2"}


def test_ch3_comm_milestone_pays_20():
    """ch3's comm milestone is worth 20.0 (double the other chambers)."""
    assert CHAMBER_COMM_THRESHOLDS["ch3"] == (4, 20.0, "m_comm_ch3")
    t = _tracker()
    pos = {A: (0.0, 5.0, 40.0)}  # 32 <= z=40 <= 50
    milestones = []
    for i in range(4):
        _, fired, _ = t.process_step(10 + 3 * i, {A: f"chamber three msg {i}"}, pos)
        milestones.extend(fired)
    assert milestones == [(A, "m_comm_ch3", 20.0)]
    # And every other chamber threshold for reference.
    assert CHAMBER_COMM_THRESHOLDS == {
        "ch1": (4, 10.0, "m_comm_ch1"),
        "ch2": (4, 10.0, "m_comm_ch2"),
        "ch3": (4, 20.0, "m_comm_ch3"),
        "ch4": (4, 10.0, "m_comm_ch4"),
        "ch5": (4, 10.0, "m_comm_ch5"),
    }


def test_chamber_for_z_band_mapping():
    """_chamber_for maps z into the five inclusive bands; gaps and None -> None."""
    t = _tracker()
    # One point inside each band (x and y are irrelevant; only p[2] is read).
    assert t._chamber_for((0, 0, 0)) == "ch1"
    assert t._chamber_for((9, 1, 15)) == "ch1"
    assert t._chamber_for((0, 0, 17)) == "ch2"
    assert t._chamber_for((0, 0, 30)) == "ch2"
    assert t._chamber_for((0, 0, 32)) == "ch3"
    assert t._chamber_for((0, 0, 50)) == "ch3"
    assert t._chamber_for((0, 0, 52)) == "ch4"
    assert t._chamber_for((0, 0, 62)) == "ch4"
    assert t._chamber_for((0, 0, 64)) == "ch5"
    assert t._chamber_for((0, 0, 72)) == "ch5"
    # Gap blocks between/outside the bands.
    for z in (-1, 16, 31, 51, 63, 73):
        assert t._chamber_for((0, 0, z)) is None
    assert t._chamber_for(None) is None


def test_bad_target_speaker_no_base_reward_but_chamber_accrues():
    """Bad-target speaker: valid (routed) but unpaid; chamber count still +1."""
    t = _tracker()
    rewards, milestones, valid = t.process_step(
        10, {A: "talking to myself"}, {A: (0, 5, 5)}, bad_target_speakers={A}
    )
    assert valid == {A}  # routing rescued the message
    assert rewards == {}  # no base reward for self-talk
    assert t.total_valid_msgs[A] == 0  # cap counter only tracks PAID messages
    assert t.chamber_msg_counts[A]["ch1"] == 1  # milestone progress still accrues
    assert milestones == []
    # State was updated (the message was valid), so a duplicate is rejected.
    _, _, v2 = t.process_step(20, {A: "talking to myself"}, {A: (0, 5, 5)})
    assert v2 == set()


def test_unknown_or_none_position_pays_base_without_chamber_accrual():
    """Missing or None position still pays +0.5 but accrues no chamber count."""
    t = _tracker()
    r1, m1, v1 = t.process_step(10, {A: "where are you all"}, {})  # missing key
    assert r1 == {A: pytest.approx(0.5)} and m1 == [] and v1 == {A}
    r2, m2, v2 = t.process_step(13, {A: "anyone near the gate"}, {A: None})
    assert r2 == {A: pytest.approx(0.5)} and m2 == [] and v2 == {A}
    assert dict(t.chamber_msg_counts[A]) == {}
    assert t.total_valid_msgs[A] == 2
