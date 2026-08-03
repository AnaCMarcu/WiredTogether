"""Unit tests for failure detectors and lexicons."""

from qual_fixtures import build_synth_run  # noqa: F401 (sys.path bootstrap)

from qual_lib import lexicons
from qual_lib.detectors import (action_stuck_loop, critic_verdicts,
                                degenerate_comm, hallucinated_object,
                                rl_action_collapse, task_repeat_loop)


def _row(t, agent=0, ep=1, action="Dig", chamber="ch1", reward=0.0,
         pos=None, **kw):
    r = {"exp": "e", "seed": 1, "ep": ep, "t": t, "agent": agent,
         "chamber": chamber, "action": action, "reward_task": reward,
         "pos": pos or [0.0, 11.5, 0.0], "msg": None, "beliefs": None,
         "thoughts": None, "critic": None, "task": None, "task_new": None}
    r.update(kw)
    return r


def test_task_repeat_loop_normalization_and_streaks():
    rows = []
    for i, task in enumerate(["dig tree", "Dig  Tree ", "DIG TREE",
                              "press switch"]):
        rows.append(_row(i * 10, task=task, task_new=True))
    flags = task_repeat_loop(rows)
    assert len(flags) == 1
    assert flags[0]["detail"]["n_repeats"] == 3


def test_action_stuck_loop_requires_no_reward_and_no_displacement():
    stuck = [_row(t, action="TurnRight") for t in range(15)]
    flags = action_stuck_loop(stuck)
    assert len(flags) == 1 and flags[0]["detail"]["subtype"] == "turn_circling"
    moving = [_row(t, action="MoveForward", pos=[float(t), 11.5, 0.0])
              for t in range(15)]
    assert action_stuck_loop(moving) == []
    rewarded = [_row(t, action="Dig", reward=(10.0 if t == 7 else 0.0))
                for t in range(15)]
    assert action_stuck_loop(rewarded) == []


def test_critic_false_positive_and_negative():
    rows = [_row(t) for t in range(30)]
    rows[25]["critic"] = {"success": True, "critique": "", "reasoning": ""}
    flags = critic_verdicts(rows)
    assert any(f["detector"] == "critic_false_positive" for f in flags)
    rows2 = [_row(t) for t in range(30)]
    rows2[10]["reward_task"] = 30.0
    rows2[11]["critic"] = {"success": False, "critique": "", "reasoning": ""}
    flags2 = critic_verdicts(rows2)
    assert any(f["detector"] == "critic_false_negative" for f in flags2)


def test_hallucinated_object_chamber_whitelist():
    r_bad = _row(0, chamber="ch1",
                 beliefs={"perception": ["a purple anvil ahead"]})
    r_ok = _row(1, chamber="ch2",
                beliefs={"perception": ["a purple anvil ahead"]})
    flags = hallucinated_object([r_bad, r_ok])
    assert len(flags) == 1
    assert flags[0]["detail"]["objects"] == ["anvil"]
    assert flags[0]["detail"]["source"] == "belief"


def test_rl_action_collapse_windowing():
    rows = [_row(t, action="MoveForward") for t in range(220)]
    assert rl_action_collapse(rows, is_rl=True)
    assert rl_action_collapse(rows, is_rl=False) == []
    mixed = [_row(t, action=("Dig" if t % 2 else "TurnLeft"))
             for t in range(220)]
    assert rl_action_collapse(mixed, is_rl=True) == []


def test_degenerate_comm_streak():
    rows = [_row(t, msg={"text": "hello there friend"}) for t in range(6)]
    flags = degenerate_comm(rows)
    assert len(flags) == 1 and flags[0]["detail"]["n"] == 6


def test_lexicon_categories_and_intended_actions():
    cats = lexicons.categorize_message("Please help me press the switch")
    assert "request" in cats
    assert lexicons.intended_actions("I will dig the tree") == {"Dig"}
    assert "MoveForward" in lexicons.intended_actions(
        "move toward the door then look up")
    assert lexicons.impossible_mentions("zombie ahead!", "ch2") == ["zombie"]
    assert lexicons.impossible_mentions("zombie ahead!", "ch4") == []
