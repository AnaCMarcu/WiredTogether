"""Executor, events, progress reward and comm routing against the fake world."""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from agentworld.actions import parse_action
from agentworld.comms import CommRouter, make_policy
from agentworld.events import EventExtractor, hebbian_triples
from agentworld.executor import AgentHandle, Executor, prepare_agent
from agentworld.progress import ProgressTracker
from agentworld.tasks import compose_world, load_task
from agentworld.verify import extract_targets
from fake_kaetram import FakeGameTools, FakeWorld

TASK = Path(__file__).parent / "fixtures" / "agentworld" / "v1.3_benchmark" / "task_02_arrow_production.yaml"


def _setup(n_teams=1):
    task = load_task(TASK)
    world = compose_world([task] * n_teams)
    fake = FakeWorld()
    handles = []
    for i, name in enumerate(world.global_names):
        tools = FakeGameTools(fake)
        spec = world.team_of(i).task.agents[[a.key for a in task.agents].index(world.global_key[i])]
        assert prepare_agent(tools, spec, name, "pw", sleep=lambda s: None)
        handles.append(AgentHandle(i, name, world.global_key[i], world.global_team[i], tools))
    return world, fake, handles


def test_prepare_agent_applies_task_state():
    _, fake, handles = _setup()
    hunter = fake.players["t02_hunter_agent"]
    assert hunter["items"] == {"flask": 10, "apple": 15, "feather": 5}
    assert (hunter["x"], hunter["y"]) == (390, 3)
    assert hunter["skills"]["accuracy"] == 25 and hunter["skills"]["fishing"] == 1
    assert ("t02_hunter_agent", "clear_equipment") in fake.calls  # new_character reset


def test_executor_runs_in_parallel_and_records_traj_observation():
    world, fake, handles = _setup()
    fake.action_delay["harvest_resource"] = 0.3
    ex = Executor(handles, deadline_s=5)
    calls = {i: parse_action("harvest_resource(targetInstance='t-oak-1')") for i in range(3)}
    t0 = time.time()
    res = asyncio.run(ex.act_all(calls, round_num=1))
    assert time.time() - t0 < 0.8  # 3 × 0.3 s actions overlapped
    assert [r.agent for r in res] == [0, 1, 2]
    r = res[0]
    assert r.success and "❤️69/69" in r.status
    assert {"key": "logs", "count": 1, "name": "logs"} in r.observation["inventory"]["items"]
    ex.shutdown()


def test_long_action_leaves_agent_busy_without_blocking_round():
    world, fake, handles = _setup()
    fake.action_delay["attack_entity"] = 0.6
    ex = Executor(handles, deadline_s=0.1)

    async def scenario():
        first = await ex.act_all({0: parse_action("attack_entity(targetInstance='m-rat-1')"),
                                  1: parse_action("wait()")}, round_num=1)
        assert [r.agent for r in first] == [1] and ex.busy(0)
        # A busy agent's new call is not started.
        assert ex.submit({0: parse_action("wait()")}, 1) == []
        await asyncio.sleep(0.7)
        # Its finished attack is never dropped, even when a new call starts.
        later = await ex.act_all({0: parse_action("wait()")}, round_num=2)
        assert [(r.call.name, r.round_issued) for r in later] == [
            ("attack_entity", 1), ("wait", 2)]
        assert not ex.busy(0)

    asyncio.run(scenario())
    ex.shutdown()


def test_events_xfer_combat_kill_and_death():
    world, fake, handles = _setup()
    ex = Executor(handles, deadline_s=5)
    xt = EventExtractor(world.global_names, combat_window=2)
    r1 = asyncio.run(ex.act_all({
        1: parse_action("transfer_items(targetPlayer='t02_fletcher_agent', itemKey='feather', count=5)"),
        0: parse_action("attack_entity(targetInstance='m-rat-1')"),
    }, 1))
    ev1 = xt.from_results(1, r1)
    assert [(e.kind, e.src, e.dst) for e in ev1] == [("xfer", 1, 2)]
    r2 = asyncio.run(ex.act_all({2: parse_action("attack_entity(targetInstance='m-rat-1')")}, 2))
    ev2 = xt.from_results(2, r2)
    kinds = {e.kind for e in ev2}
    assert kinds == {"combat", "kill"}
    kill = next(e for e in ev2 if e.kind == "kill")
    assert kill.data["attackers"] == [0, 2] and kill.data["mob"] == "Rat"
    assert ("xfer" in {c for _, _, c in hebbian_triples(ev1)})
    assert (2, 0, "combat") in hebbian_triples(ev2, dms=[(0, 1)])
    assert (0, 1, "comm") in hebbian_triples(ev2, dms=[(0, 1)])

    xt.deaths(3, {0: {"playerStatus": {"hitPoints": 10}}})
    d = xt.deaths(4, {0: {"playerStatus": {"hitPoints": 0}}})
    assert [(e.kind, e.src) for e in d] == [("death", 0)]
    ex.shutdown()


def test_progress_reward_credits_gain_give_kill_and_solve():
    world, fake, handles = _setup()
    task = world.teams[0].task
    targets = {"t00": extract_targets(task.verifier_path())}
    pt = ProgressTracker(world, targets)
    ex = Executor(handles, deadline_s=5)
    obs0 = asyncio.run(ex.observe_all(1))
    pt.observe_initial(obs0)
    assert pt.relevant("t00", "arrow") and pt.relevant("t00", "logs")
    assert pt.relevant("t00", "feather") and not pt.relevant("t00", "flask")

    res = asyncio.run(ex.act_all({
        0: parse_action("harvest_resource(targetInstance='t-oak-1')"),
        1: parse_action("transfer_items(targetPlayer='t02_fletcher_agent', itemKey='feather', count=5)"),
    }, 1))
    xt = EventExtractor(world.global_names)
    ev = xt.from_results(1, res)
    obs = {r.agent: r.observation for r in res}
    obs[2] = asyncio.run(ex.observe_all(1, [2]))[2]
    bond, death, parts = pt.update(1, obs, ev, {"t00": 0.2}, {"t00": 0})
    assert parts[0]["gain"] == 2.0                     # one log harvested
    assert parts[1]["give"] == 15.0                    # 5 feathers handed over
    assert "gain" not in parts.get(2, {})              # received items are not "gains"
    assert parts[0]["progress"] == pytest.approx(4.0)  # 20 · Δp 0.2, acted recently
    assert death == [0.0, 0.0, 0.0]

    bond, _, parts = pt.update(2, {}, [], {"t00": 1.0}, {"t00": 1})
    assert all(parts[i]["solve"] == 50.0 for i in range(3))
    bond, _, parts = pt.update(3, {}, [], {"t00": 1.0}, {"t00": 1})
    assert bond == [0.0, 0.0, 0.0]                     # solve credited once
    ex.shutdown()


# ── comms ────────────────────────────────────────────────────────────────────
def _router(policy, n=12, **kw):
    return CommRouter(n, make_policy(policy, seed=1,
                                     same_team=kw.pop("same_team", None)), **kw)


def test_inbox_and_board_budgets_and_contacts_cap():
    r = _router("recency", n=200, board_budget=5, inbox_budget=4, contacts_budget=6)
    for s in range(1, 30):
        r.send_dm(1, s, 0, f"hi from {s}")
        r.post(1, s, f"post {s}")
    inbox = r.route(2)[0]
    assert len(inbox.dms) == 4 and inbox.dropped_dms == 25
    assert len(inbox.board) == 5 and inbox.unread_posts == 24
    assert all(len(ib.contacts) <= 6 for ib in r.route(3).values())


def test_hebbian_gating_prefers_bonded_authors_with_exploration():
    n = 20
    W = np.full((n, n), 0.1, dtype=np.float32)
    W[0, 7] = W[0, 9] = W[0, 11] = 0.9
    r = _router("hebbian", n=n, board_budget=5, explore_slots=2, contacts_budget=6)
    for s in range(1, n):
        r.post(1, s, f"post {s}")
    board = r.route(1, W=W)[0].board
    authors = [m.sender for m in board]
    assert authors[:3] == [7, 9, 11] or set(authors[:3]) == {7, 9, 11}
    assert len(authors) == 5 and len(set(authors[3:]) & {7, 9, 11}) == 0
    contacts = r.route(2, W=W)[0].contacts
    assert set([7, 9, 11]) <= set(contacts)


def test_gating_is_irrelevant_when_board_fits():
    n = 5
    W = np.random.default_rng(0).random((n, n)).astype(np.float32)
    reads = {}
    for name in ("recency", "hebbian", "shuffled"):
        r = _router(name, n=n, board_budget=5)
        for s in range(1, n):
            r.post(1, s, "x")
        reads[name] = sorted(m.sender for m in r.route(1, W=W)[0].board)
    assert reads["recency"] == reads["hebbian"] == reads["shuffled"] == [1, 2, 3, 4]


def test_shuffled_keeps_row_magnitudes():
    n = 8
    W = np.random.default_rng(3).random((n, n)).astype(np.float32)
    np.fill_diagonal(W, 0)
    pol = make_policy("shuffled", seed=5)
    pol.begin_round(1, W)
    row = sorted(pol.bond(0, j) for j in range(1, n))
    assert np.allclose(row, sorted(W[0, 1:]))


def test_oracle_prefers_teammates():
    world = compose_world([load_task(TASK)] * 4)
    r = _router("oracle", n=12, board_budget=2, explore_slots=0,
                same_team=world.true_team_matrix())
    for s in range(1, 12):
        r.post(1, s, "x")
    assert sorted(m.sender for m in r.route(1)[0].board) == [1, 2]
