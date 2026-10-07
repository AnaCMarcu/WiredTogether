"""Full rounds through RoundScheduler with scripted cognition and the fake world."""

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from agentworld.comms import CommRouter, make_policy
from agentworld.executor import AgentHandle, Executor, prepare_agent
from agentworld.observation import inventory_counts
from agentworld.progress import ProgressTracker
from agentworld.scheduler import Decision, RoundScheduler, SchedulerConfig
from agentworld.tasks import compose_world, load_task
from agentworld.trajectory import read_jsonl
from agentworld.verify import Verifier, extract_targets
from fake_kaetram import FakeGameTools, FakeWorld
from hebbian.multichannel import MultiChannelConfig, MultiChannelHebbianGraph

TASK = Path(__file__).parent / "fixtures" / "agentworld" / "v1.3_benchmark" / "task_02_arrow_production.yaml"


class ArrowTeam:
    """Scripted solver for task 02: logs + feathers → fletcher → arrows."""

    async def decide(self, turn):
        inv = inventory_counts(turn.observation)
        role = turn.name.split("_", 1)[1] if turn.name.count("_") > 2 else turn.name
        prefix = turn.name[: len(turn.name) - len(role)]
        fletcher = prefix + "t02_fletcher_agent" if prefix else "t02_fletcher_agent"
        if "lumberjack" in turn.name:
            if inv.get("logs", 0) + getattr(self, "_sent_logs", 0) < 3:
                return Decision("harvest_resource(targetInstance='t-oak-1')", "need logs",
                                board_post=f"I have {inv.get('logs', 0)} logs")
            if inv.get("logs", 0) >= 3:
                return Decision(f"transfer_items(targetPlayer='{fletcher}', itemKey='logs', count=3)",
                                dm_target=fletcher, dm_text="sending you 3 logs")
            return Decision("wait()")
        if "hunter" in turn.name:
            if inv.get("feather", 0) >= 5:
                return Decision(f"transfer_items(targetPlayer='{fletcher}', itemKey='feather', count=5)",
                                dm_target="fletcher_agent", dm_text="5 feathers incoming")
            return Decision("wait()")
        if inv.get("logs", 0) >= 1:
            return Decision("craft_item(skill='Fletching', itemKey='stick')")
        if inv.get("stick", 0) >= 10 and inv.get("feather", 0) >= 10:
            return Decision("craft_item(skill='Fletching', itemKey='arrow')")
        return Decision("wait()", board_post="waiting for materials", post_kind="need")


def _build(tmp_path, n_teams=1, policy="hebbian", graph=True, deadline=5.0, spacing=0):
    task = load_task(TASK)
    world = compose_world([task] * n_teams, team_spacing=spacing)
    fake = FakeWorld()
    handles = []
    for i, name in enumerate(world.global_names):
        tools = FakeGameTools(fake)
        spec = next(a for a in task.agents if a.key == world.global_key[i])
        prepare_agent(tools, spec, name, "pw", sleep=lambda s: None,
                      location=world.global_spawn[i])
        handles.append(AgentHandle(i, name, world.global_key[i], world.global_team[i], tools))
    ex = Executor(handles, deadline_s=deadline)
    router = CommRouter(world.n_agents, make_policy(policy, same_team=world.true_team_matrix()))
    targets = {t.team: extract_targets(t.task.verifier_path()) for t in world.teams}
    verifiers = {t.team: Verifier(t.task.verifier_path()) for t in world.teams}
    g = None
    if graph:
        g = MultiChannelHebbianGraph(MultiChannelConfig(
            enabled=True, num_agents=world.n_agents, eta_0=0.02, eta_plus=0.3,
            decay=0.005, eligibility_rho=0.8, interaction_radius=4.0))
    sched = RoundScheduler(world, ex, router, ArrowTeam(), ProgressTracker(world, targets),
                           verifiers, tmp_path, graph=g,
                           config=SchedulerConfig(max_rounds=20))
    return sched, world, fake, ex


def test_scripted_team_solves_task_and_writes_logs(tmp_path):
    sched, world, fake, ex = _build(tmp_path)
    summary = asyncio.run(sched.run_episode(1))
    ex.shutdown()
    team = summary["teams"]["t00"]
    assert team["success"] == 1 and team["message"] == "Arrows: 10/10"
    assert summary["rounds"] <= 8  # early stop on AgentWorld's schedule (round 5)

    ep = tmp_path / "episode_1"
    traj = json.loads(next((ep / "trajectories").glob("*.json")).read_text(encoding="utf-8"))
    actions = [a["action"] for r in traj["rounds"] for a in r["actions"]]
    assert any(a.startswith("chat(message=@t02_t02_fletcher_agent") or
               a.startswith("chat(message=@t02_fletcher_agent") for a in actions)
    assert any(a.startswith("transfer_items(") for a in actions)
    assert Verifier(world.teams[0].task.verifier_path())(traj) == (1, "Arrows: 10/10")

    msgs = read_jsonl(ep / "messages.jsonl")
    assert {m["kind"] for m in msgs} == {"dm", "post"}
    assert all(m["delivered"] for m in msgs if m["kind"] == "dm")
    events = read_jsonl(ep / "events.jsonl")
    assert {"xfer", "craft"} <= {e["kind"] for e in events}

    states = read_jsonl(ep / "replay" / "state.jsonl")
    s = states[0]
    assert {"round", "agents", "actions", "events", "messages", "reads", "teams",
            "bonds_top"} <= set(s)
    assert s["agents"][0]["x"] is not None

    # Transfers wired the givers to the fletcher (agent 2) through the xfer channel.
    W = sched.graph.W
    xfer = sched.graph.channel_growth()["xfer"]
    assert xfer[1, 2] > 0 and xfer[0, 2] > 0
    assert W[1, 2] > 0.1


def test_base_arm_runs_without_graph(tmp_path):
    sched, *_ , ex = _build(tmp_path, policy="recency", graph=False)
    summary = asyncio.run(sched.run_episode(1))
    ex.shutdown()
    assert summary["teams"]["t00"]["success"] == 1
    s = read_jsonl(tmp_path / "episode_1" / "replay" / "state.jsonl")[0]
    assert s["bonds_top"] is None


def _within_vs_across(W, same):
    off = ~np.eye(len(W), dtype=bool)
    return W[(same == 1) & off].mean(), W[(same == 0) & off].mean()


def test_replicas_on_shared_spawns_confound_proximity(tmp_path):
    """Documents the confound compose_world(team_spacing=...) exists to remove."""
    sched, world, fake, ex = _build(tmp_path, n_teams=4)
    summary = asyncio.run(sched.run_episode(1))
    ex.shutdown()
    assert world.n_agents == 12
    assert all(t["success"] == 1 for t in summary["teams"].values())
    same = world.true_team_matrix()
    within, across = _within_vs_across(sched.graph.W, same)
    assert across >= 0.5 * within  # proximity bonds strangers sharing spawn tiles
    # ...but transfer credit still only ever flows inside a team.
    xfer = sched.graph.channel_growth()["xfer"]
    assert xfer[same == 0].sum() == 0 and xfer[same == 1].sum() > 0


def test_spaced_multi_team_world_separates_teams(tmp_path):
    sched, world, fake, ex = _build(tmp_path, n_teams=4, spacing=20)
    summary = asyncio.run(sched.run_episode(1))
    ex.shutdown()
    assert all(t["success"] == 1 for t in summary["teams"].values())
    assert world.global_spawn[3] == (408, 3) and world.global_spawn[0] == (388, 3)
    within, across = _within_vs_across(sched.graph.W, world.true_team_matrix())
    assert within > 2 * across


class Chatty:
    """Every agent posts and DMs a random agent each round (prompt-size check)."""

    def __init__(self):
        self.max_contacts = 0
        self.max_board = 0
        self.max_dms = 0

    async def decide(self, turn):
        self.max_contacts = max(self.max_contacts, len(turn.contacts))
        self.max_board = max(self.max_board, len(turn.inbox.board))
        self.max_dms = max(self.max_dms, len(turn.inbox.dms))
        target = turn.names[(turn.agent * 7 + turn.round) % len(turn.names)]
        return Decision("wait()", dm_target=target, dm_text="hi", board_post="status")


def test_large_world_keeps_reading_bounded(tmp_path):
    sched, world, fake, ex = _build(tmp_path, n_teams=17, deadline=2.0, spacing=20)  # 51 agents
    chatty = Chatty()
    sched.cog = chatty
    sched.cfg.max_rounds = 4
    t0 = time.time()
    asyncio.run(sched.run_episode(1))
    ex.shutdown()
    assert time.time() - t0 < 30
    assert chatty.max_contacts <= 6 and chatty.max_board <= 5 and chatty.max_dms <= 4


def _mirrored(tmp_path, mode, mirror_flag):
    sched, world, fake, ex = _build(tmp_path)
    calls = []
    sched.cfg.mirror_dms = mirror_flag
    sched.cfg.comm_mode = mode
    sched.mirror = (lambda i, text: calls.append((i, text))) if (
        mirror_flag and mode == "side") else None
    asyncio.run(sched.run_episode(1))
    ex.shutdown()
    return calls


def test_dms_and_posts_are_mirrored_for_live_video(tmp_path):
    calls = _mirrored(tmp_path, "side", True)
    assert any(t.startswith("@t02_fletcher_agent:") for _, t in calls)
    assert any(t.startswith("[board] ") for _, t in calls)


def test_no_mirroring_unless_asked(tmp_path):
    assert _mirrored(tmp_path, "side", False) == []


def test_scheduler_refuses_mirroring_in_tool_mode(tmp_path):
    sched, world, fake, ex = _build(tmp_path)
    from agentworld.scheduler import RoundScheduler, SchedulerConfig
    s2 = RoundScheduler(world, sched.ex, sched.router, sched.cog, sched.progress,
                        sched.verifiers, tmp_path, config=SchedulerConfig(
                            comm_mode="tool", mirror_dms=True), mirror=lambda i, t: None)
    assert s2.mirror is None
    ex.shutdown()
