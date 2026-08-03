"""Unit tests for the collaboration case-mining stage (qual_lib.collab).

All tests build synthetic in-memory ctx dicts (no filesystem) matching the
shapes load_ctx produces: timeline rows, per-episode messages/events lists,
and cooperation metrics with the REAL nesting already resolved.
"""

from types import SimpleNamespace

from qual_fixtures import REPO  # noqa: F401 (sys.path bootstrap)

from qual_lib import collab


def _run(label="LLM-9B", is_rl=False):
    return SimpleNamespace(exp="expX", seed=1, label=label, is_rl=is_rl,
                           run_id="expX/seed_1")


def _msg(t, sender, target, text, chamber="ch1", valid=True):
    return {"t": t, "sender": f"agent_{sender}", "receiver": f"agent_{target}",
            "model_target_canonical": f"agent_{target}", "text": text,
            "valid": valid, "routing": "model", "chamber": chamber}


def _row(t, agent, action="NoOp", chamber="ch1", ep=1):
    return {"exp": "expX", "seed": 1, "ep": ep, "t": t, "agent": agent,
            "chamber": chamber, "action": action}


def _ctx(messages=(), events=(), timeline=(), coop=None, label="LLM-9B",
         max_steps=1000):
    return {
        "run": _run(label),
        "timeline": list(timeline),
        "messages": {1: list(messages)},
        "events": {1: list(events)},
        "coop_by_ep": {1: coop or {}},
        "final_metrics": {"config": {"cli_args": {"max_steps": max_steps}}},
    }


def test_request_fulfilled_by_compliance():
    ctx = _ctx(
        messages=[_msg(10, 0, 1, "dig the anvil with me", chamber="ch2")],
        timeline=[_row(t, 1, action=("Dig" if t == 13 else "NoOp"),
                       chamber="ch2") for t in range(30)],
    )
    evs = collab.request_outcomes(ctx)
    assert len(evs) == 1
    e = evs[0]
    assert e["success"] and e["detail"]["status"] == "fulfilled"
    assert e["detail"]["complied"] is True
    assert e["detail"]["coop_talk"] is True
    assert e["agents"] == [0, 1]


def test_request_answered_vs_ignored_and_streak():
    # three pleas from a0, one gets a reply, the last two are ignored
    msgs = [
        _msg(10, 0, 1, "please help me break this"),
        _msg(15, 1, 0, "on my way"),          # reply (not itself a request)
        _msg(60, 0, 1, "please help me break this"),
        _msg(80, 0, 1, "can you help me please"),
    ]
    ctx = _ctx(messages=msgs, timeline=[_row(t, a) for t in range(120)
                                        for a in (0, 1)])
    evs = collab.request_outcomes(ctx)
    statuses = [e["detail"]["status"] for e in evs]
    assert statuses == ["answered", "ignored", "ignored"]
    ignored = [e for e in evs if e["detail"]["status"] == "ignored"]
    assert all(e["detail"]["ignored_streak"] == 2 for e in ignored)
    assert all(e["score"] >= 2 for e in ignored)


def test_timer_boundary_milestones_excluded():
    events = [
        {"step": 200, "type": "milestone", "id": "m17_switch_pressed",
         "contributors": ["agent1"], "reward": 20},   # timer race -> excluded
        {"step": 300, "type": "milestone", "id": "m17_switch_pressed",
         "contributors": ["agent1"], "reward": 20},
        {"step": 305, "type": "milestone", "id": "m16_enter_cell",
         "contributors": ["agent0"], "reward": 5},    # entry -> excluded
    ]
    coop = {"milestone_log": [
        {"step": 300, "milestone": "m17_switch_pressed",
         "comm_before_coop": True}]}
    ctx = _ctx(events=events, coop=coop)
    evs = collab.coop_milestone_events(ctx)
    assert len(evs) == 1
    assert evs[0]["detail"]["milestone"] == "m17_switch_pressed"
    assert evs[0]["detail"]["t_milestone"] == 300
    assert evs[0]["detail"]["comm_before_coop"] is True


def test_request_fulfilled_by_real_milestone_not_timer():
    msgs = [_msg(290, 0, 1, "press the switch to free me", chamber="ch3")]
    events = [{"step": 300, "type": "milestone", "id": "m17_switch_pressed",
               "contributors": ["agent1"], "reward": 20}]
    ctx = _ctx(messages=msgs, events=events,
               timeline=[_row(t, a, chamber="ch3") for t in range(280, 330)
                         for a in (0, 1)])
    evs = collab.request_outcomes(ctx)
    assert evs[0]["detail"]["status"] == "fulfilled"
    assert evs[0]["detail"]["milestones"] == ["m17_switch_pressed"]


def test_mutual_deadlock_requires_no_press():
    msgs = [
        _msg(100, 0, 1, "press your switch to free me", chamber="ch3"),
        _msg(110, 1, 0, "please press the switch for my door", chamber="ch3"),
    ]
    ctx = _ctx(messages=msgs,
               timeline=[_row(t, a, chamber="ch3") for t in range(90, 220)
                         for a in (0, 1)])
    evs = collab.mutual_deadlocks(ctx)
    assert len(evs) == 1
    assert set(evs[0]["agents"]) == {0, 1}

    events = [{"step": 130, "type": "milestone", "id": "m17_switch_pressed",
               "contributors": ["agent1"], "reward": 20}]
    ctx2 = _ctx(messages=msgs, events=events,
                timeline=[_row(t, a, chamber="ch3") for t in range(90, 220)
                          for a in (0, 1)])
    assert collab.mutual_deadlocks(ctx2) == []


def test_ch3_stall_and_ch2_neglect():
    coop = {"chamber_entry_steps": {"ch2": 200, "ch3": 400},
            "final_step": 900}
    msgs = [_msg(210, 0, 1, "there is an anvil here", chamber="ch2"),
            _msg(420, 1, 2, "i am locked, press the switch", chamber="ch3")]
    ctx = _ctx(messages=msgs, coop=coop,
               timeline=[_row(t, 0, action="Dig", chamber="ch2")
                         for t in range(200, 240)])
    evs = collab.chamber_failures(ctx)
    types = sorted(e["type"] for e in evs)
    assert types == ["ch2_neglect", "ch3_stall"]
    stall = next(e for e in evs if e["type"] == "ch3_stall")
    assert stall["detail"]["pressed"] is False
    assert stall["detail"]["n_switch_talk"] == 1
    neglect = next(e for e in evs if e["type"] == "ch2_neglect")
    assert neglect["detail"]["n_anvil_msgs"] == 1
    assert neglect["detail"]["n_ch2_digs"] == 40

    # a fast real press (latency < CH3_STALL_MIN) suppresses the stall
    events = [{"step": 430, "type": "milestone", "id": "m17_switch_pressed",
               "contributors": ["agent2"], "reward": 20}]
    ctx2 = _ctx(messages=msgs, events=events, coop=coop)
    assert all(e["type"] != "ch3_stall"
               for e in collab.chamber_failures(ctx2))


def test_stale_appended_records_clamped_to_final_step():
    # relaunch appended records beyond the surviving episode's end must be
    # dropped from both message and milestone streams
    msgs = [_msg(10, 0, 1, "dig the tree"),
            _msg(900, 0, 1, "dig the tree")]        # stale: t > final_step
    events = [{"step": 300, "type": "milestone", "id": "m17_switch_pressed",
               "contributors": ["agent1"], "reward": 20},
              {"step": 950, "type": "milestone", "id": "m17_switch_pressed",
               "contributors": ["agent1"], "reward": 20}]  # stale
    ctx = _ctx(messages=msgs, events=events,
               timeline=[_row(t, a) for t in range(50) for a in (0, 1)],
               coop={"final_step": 801})
    assert len(collab._directed_messages(ctx)) == 1
    assert len(collab.real_coop_events(ctx)) == 1


def test_bond_W_attached_to_directed_requests():
    W = [[0.0, 0.7, 0.1], [0.7, 0.0, 0.2], [0.1, 0.2, 0.0]]
    ctx = _ctx(messages=[_msg(10, 0, 1, "please help me dig")],
               timeline=[_row(t, a) for t in range(50) for a in (0, 1)],
               coop={"hebbian_W": W}, label="LLM-9B+Heb")
    evs = collab.request_outcomes(ctx)
    assert evs[0]["W"] == 0.7


def test_run_level_and_condition_level_aggregation():
    ctx = _ctx(messages=[
        _msg(10, 0, 1, "dig the tree"),
        _msg(60, 0, 1, "please help me"),
    ], timeline=[_row(t, 1, action=("Dig" if t == 12 else "NoOp"))
                 for t in range(100)])
    evs = collab.detect_events(ctx)
    rec = collab.run_level(evs)
    assert rec["n_requests"] == 2
    assert rec["frac_fulfilled"] == 0.5
    assert rec["frac_ignored"] == 0.5
    assert rec["comply_rate"] == 1.0        # only the directive is judgeable
    cond = collab.condition_level([rec, rec])
    assert cond["n_runs"] == 2
    assert cond["n_requests"] == 4
    assert cond["frac_fulfilled"] == 0.5


def test_within_episode_W_contrast_pairing():
    def ev(ep, status, W):
        return {"type": "request_outcome", "exp": "e", "seed": 1, "ep": ep,
                "W": W, "detail": {"status": status}}
    events = [
        ev(1, "fulfilled", 0.6), ev(1, "ignored", 0.2),   # win
        ev(2, "fulfilled", 0.1), ev(2, "ignored", 0.5),   # loss
        ev(3, "fulfilled", 0.3), ev(3, "ignored", 0.3),   # tie (static W)
        ev(4, "fulfilled", 0.9),                          # unpaired: skipped
        ev(5, "answered", 0.9), ev(5, "ignored", 0.1),    # answered != pair
    ]
    res = collab.within_episode_W_contrast(events)
    assert res == {"eps_W_fulfilled_higher": 1, "eps_W_fulfilled_lower": 1,
                   "eps_W_tie": 1}


def test_pick_exemplars_dedupes_same_window():
    base = {"exp": "expX", "seed": 1, "label": "L", "is_rl": False, "ep": 1,
            "agents": [0, 1], "W": None}
    evs = [
        {**base, "type": "request_outcome", "success": True, "t0": 100,
         "t1": 130, "score": 3.0,
         "detail": {"status": "fulfilled", "categories": ["request"],
                    "coop_talk": True, "milestones": ["m17_switch_pressed"],
                    "text": "press it", "complied": None, "replied": False}},
        {**base, "type": "coop_milestone", "success": True, "t0": 90,
         "t1": 125, "score": 3.0,
         "detail": {"milestone": "m17_switch_pressed", "reward": 20,
                    "comm_before_coop": True, "t_milestone": 120}},
        {**base, "type": "coop_milestone", "success": True, "t0": 500,
         "t1": 535, "score": 2.0,
         "detail": {"milestone": "m9_anvil_B1", "reward": 40,
                    "comm_before_coop": False, "t_milestone": 530}},
    ]
    picks = collab.pick_exemplars({"L": evs})
    slugs = [s for s, _ in picks]
    assert len(picks) == 2
    # same (run, ep, ~30-step window) -> only one of the first two survives
    assert sum("success" in s for s in slugs) == 2
