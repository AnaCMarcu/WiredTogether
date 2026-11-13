"""MindForge agents on AgentWorld chains (chains/mf_chain.py, chains/run_mf_chain.py).

The round loop runs on the in-memory fake server with scripted cognition, so
these tests need neither a game server nor a model.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "chains", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mf_chain  # noqa: E402
from agentworld.comms import CommRouter, make_policy  # noqa: E402
from agentworld.executor import AgentHandle, Executor, prepare_agent  # noqa: E402
from agentworld.progress import ProgressTracker  # noqa: E402
from agentworld.scheduler import Decision, SchedulerConfig  # noqa: E402
from agentworld.tasks import compose_world, load_task  # noqa: E402
from agentworld.trajectory import read_jsonl  # noqa: E402
from agentworld.verify import Verifier, extract_targets  # noqa: E402
from fake_kaetram import FakeGameTools, FakeWorld  # noqa: E402
from mf_chain import (DM_WORDING, ChainCognition, ChainRoundScheduler, StepSpec,  # noqa: E402
                      direct_only_prompts, play_step, point_at_reply_fields)
from test_aw_scheduler import ArrowTeam  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "agentworld" / "v1.3_benchmark"
TASK = FIX / "task_02_arrow_production.yaml"
VERIFIER = FIX / "task_02_success_criteria.py"
NAMES = ["t02_lumberjack_agent", "t02_hunter_agent", "t02_fletcher_agent"]


class Script:
    """Every agent plays ``fn(turn)``; records the turns it saw."""

    def __init__(self, fn):
        self.fn = fn
        self.turns = []

    async def decide(self, turn):
        self.turns.append(turn)
        return self.fn(turn)


def _sched(tmp_path, cognition, *, hold=False, cap=None, board=False, rounds=8, fake=None):
    task = load_task(TASK)
    world = compose_world([task])
    fake = fake or FakeWorld()
    handles = []
    for i, name in enumerate(world.global_names):
        tools = FakeGameTools(fake)
        spec = next(a for a in task.agents if a.key == world.global_key[i])
        prepare_agent(tools, spec, name, "pw", sleep=lambda s: None, location=world.global_spawn[i])
        handles.append(AgentHandle(i, name, world.global_key[i], world.global_team[i], tools))
    router = CommRouter(world.n_agents, make_policy("recency"), board_budget=5 if board else 0)
    team = world.teams[0].team
    return ChainRoundScheduler(
        world, Executor(handles, deadline_s=5.0), router, cognition,
        ProgressTracker(world, {team: extract_targets(VERIFIER)}), {team: Verifier(VERIFIER)},
        tmp_path, config=SchedulerConfig(max_rounds=rounds), hold_complete=hold,
        max_actions=cap, board=board)


# ── prompts and task text ───────────────────────────────────────────────────
def test_direct_only_prompts_remove_board_and_chat():
    import agentworld.agent as A
    system, instruction = direct_only_prompts(A.SYSTEM_PROMPT, A.INSTRUCTION)
    for gone in ("shared board", "board_post", "post_kind", 'chat(message="<text>")', "board posts"):
        assert gone not in system
    assert "BOARD POSTS" not in instruction and "{board}" not in instruction
    assert '"dm_target"' in system and '"dm_text"' in system and "transfer_items(" in system
    assert "There is no group chat or board" in system
    assert "{inbox}" in instruction and "{bonds}" in instruction
    # The originals are untouched (only use_direct_only_prompts patches them).
    assert "board_post" in A.SYSTEM_PROMPT


def test_direct_only_prompts_fail_loudly_when_the_prompt_changed():
    with pytest.raises(mf_chain.ChainRunError):
        direct_only_prompts("a prompt without the expected lines", "BOARD POSTS YOU CAN SEE:\n{board}\n\n")


def test_task_text_points_at_reply_fields():
    from generate_tasks import DIRECT_MESSAGES
    text = f"- Use {DIRECT_MESSAGES} to coordinate"
    assert point_at_reply_fields(text) == f"- Use {DM_WORDING} to coordinate"
    assert "direct_chat" not in point_at_reply_fields(text)


# ── the round loop ──────────────────────────────────────────────────────────
def test_hold_complete_refuses_and_counts_until_the_cap(tmp_path):
    cog = Script(lambda t: Decision("complete(response='done')"))
    sched = _sched(tmp_path, cog, hold=True, cap=3)
    summary = asyncio.run(sched.run_episode(1))
    ch = summary["chain"]
    assert set(ch["held_completes"].values()) == {3}
    assert set(ch["actions"].values()) == {3}
    assert set(ch["capped"].values()) == {3} and ch["completed"] == {}
    assert summary["rounds"] == 3 and ch["last_active_round"] == 3
    later = [t for t in cog.turns if t.round == 2]
    assert later and all(t.last_result.startswith("Not done yet") for t in later)
    events = read_jsonl(tmp_path / "episode_1" / "chain_events.jsonl")
    assert sum(e["event"] == "complete_held" for e in events) == 9
    assert sum(e["event"] == "action_cap" for e in events) == 3


def test_without_hold_complete_ends_the_agent(tmp_path):
    cog = Script(lambda t: Decision("complete(response='done')"))
    summary = asyncio.run(_sched(tmp_path, cog, hold=False).run_episode(1))
    assert set(summary["chain"]["completed"].values()) == {1}
    assert summary["rounds"] == 1


def test_cap_ignores_waits_and_refused_chats(tmp_path):
    def fn(t):
        if t.round % 2:
            return Decision("wait()")
        return Decision("chat(message='hi all')", board_post="hi board") if t.round == 2 \
            else Decision("move_character(x=250, y=181)")
    cog = Script(fn)
    summary = asyncio.run(_sched(tmp_path, cog, cap=2, rounds=12).run_episode(1))
    ch = summary["chain"]
    # Rounds 1, 3, 5 wait; round 2 chat is refused; moves at rounds 4 and 6 reach the cap.
    assert set(ch["actions"].values()) == {2}
    assert set(ch["capped"].values()) == {6}
    assert set(ch["refused_chats"].values()) == {1}
    assert ch["posts"] == 0 and sched_board_off(cog)


def sched_board_off(cog):
    after_chat = [t for t in cog.turns if t.round == 3]
    return after_chat and all("no group chat" in t.last_result for t in after_chat) \
        and all(not t.inbox.board for t in cog.turns)


def test_board_kept_when_asked(tmp_path):
    cog = Script(lambda t: Decision("wait()", board_post="note") if t.round == 1 else Decision("wait()"))
    summary = asyncio.run(_sched(tmp_path, cog, board=True, rounds=2).run_episode(1))
    assert summary["chain"]["posts"] == 3


def test_scripted_team_still_solves_with_the_chain_rules(tmp_path):
    summary = asyncio.run(_sched(tmp_path, ArrowTeam(), hold=True, cap=35, rounds=20).run_episode(1))
    team = next(iter(summary["teams"].values()))
    assert team["success"], team["message"]
    assert summary["chain"]["dms"] > 0


# ── steps: persistent agents and bonds ──────────────────────────────────────
class Member:
    """One persistent agent: delegates to a shared scripted team, counts resets."""

    def __init__(self, team):
        self.team = team
        self.name = "?"
        self.resets = 0
        self.names = []

    def on_reset(self):
        self.resets += 1

    async def decide(self, turn):
        self.names.append((turn.name, self.name))
        return await self.team.decide(turn)


def _step(index, ids):
    slots = [{"slot": f"agent_{k + 1}", "agent_id": a, "role": n.split("_")[1], "username": n}
             for k, (a, n) in enumerate(zip(ids, NAMES))]
    return StepSpec(index=index, dir_name=f"step_{index:02d}_task_02", task_key=TASK.stem,
                    task_yaml=TASK, verifier=VERIFIER, source_verifier=VERIFIER, slots=slots)


def test_play_step_carries_bonds_by_agent_id_and_renames_agents(tmp_path, monkeypatch):
    import wired_state
    hebbian = {**wired_state.hebbian_defaults(), "hebbian_decay": 0.005}
    captured = []
    real = wired_state.graph_config

    def spy(n, init, h):
        captured.append(np.array(init))
        return real(n, init, h)
    monkeypatch.setattr(wired_state, "graph_config", spy)

    team = ArrowTeam()
    agents = {a: Member(team) for a in ("a1", "a2", "a3")}
    cog = ChainCognition(agents)
    fake = FakeWorld()
    ids = ["a1", "a2", "a3"]
    W0 = np.full((3, 3), 0.1, dtype=np.float32)
    np.fill_diagonal(W0, 0.0)
    common = dict(agent_ids=ids, cognition=cog, make_tools=lambda: FakeGameTools(fake),
                  master_password="pw", arm="hebbian", seed=1, hebbian=hebbian, rounds=12,
                  max_actions=None, hold_complete=True, setup_sleep_scale=0.0)
    s1, W1, ids1 = asyncio.run(play_step(_step(1, ids), W=W0, out_dir=tmp_path / "s1", **common))
    assert ids1 == ids and not np.allclose(W1, W0)
    assert np.all(np.diag(W1) == 0)
    # Step 2 rotates the slots: agent_1 is now a2, agent_2 a3, agent_3 a1.
    rot = ["a2", "a3", "a1"]
    s2, W2, ids2 = asyncio.run(play_step(_step(2, rot), W=W1, out_dir=tmp_path / "s2", **common))
    assert ids2 == rot
    perm = [ids.index(a) for a in rot]
    assert np.allclose(captured[1], W1[np.ix_(perm, perm)])
    # Each persistent agent played the slot it was given, under that slot's name.
    assert agents["a2"].names[-1] == (NAMES[0], NAMES[0])
    assert agents["a1"].names[-1] == (NAMES[2], NAMES[2])
    assert all(a.resets == 2 for a in agents.values())
    assert (tmp_path / "s2" / "episode_1" / "summary.jsonl").is_file()


def test_play_step_refuses_a_slot_map_that_does_not_match_the_task(tmp_path):
    step = _step(1, ["a1", "a2", "a3"])
    step.slots[0]["username"] = "a1_somebody_else"
    with pytest.raises(mf_chain.ChainRunError):
        asyncio.run(play_step(step, agent_ids=["a1", "a2", "a3"], W=np.zeros((3, 3)),
                              cognition=ChainCognition({}), make_tools=lambda: None,
                              master_password="pw", out_dir=tmp_path, arm="base", seed=0,
                              hebbian={}))


def test_cli_defaults_match_the_chain_protocol():
    import run_mf_chain
    args = run_mf_chain.build_parser().parse_args(
        ["chains/pendant_staff_4.yaml", "--server-url", "http://x", "--hebbian-decay", "0.005",
         "--hold-complete"])
    assert args.max_actions == -1 and args.board is False and args.hold_complete
    assert args.hebbian_decay == 0.005 and args.arm == "hebbian"
    assert args.min_round_seconds == 0.0


def _one_step(tmp_path, arm, W0, monkeypatch=None):
    import wired_state
    hebbian = {**wired_state.hebbian_defaults(), "hebbian_decay": 0.005}
    team = ArrowTeam()
    agents = {a: Member(team) for a in ("a1", "a2", "a3")}
    ids = ["a1", "a2", "a3"]
    return asyncio.run(play_step(_step(1, ids), W=W0, out_dir=tmp_path / arm, agent_ids=ids,
                                 cognition=ChainCognition(agents),
                                 make_tools=lambda: FakeGameTools(FakeWorld()),
                                 master_password="pw", arm=arm, seed=1, hebbian=hebbian,
                                 rounds=12, max_actions=None, hold_complete=True,
                                 setup_sleep_scale=0.0))


def test_uniform_arm_keeps_w_frozen(tmp_path):
    W0 = np.full((3, 3), 0.74, dtype=np.float32)
    np.fill_diagonal(W0, 0.0)
    _, W1, _ = _one_step(tmp_path, "uniform", W0)
    assert np.array_equal(W1, W0)
    # The same step under the learned rule does move W (so the step has interactions).
    _, Wh, _ = _one_step(tmp_path, "hebbian", W0)
    assert not np.allclose(Wh, W0)


def test_freq_arm_drops_reward_terms_and_still_learns(tmp_path):
    import wired_state
    h = wired_state.hebbian_defaults()
    r = mf_chain.arm_rule("freq", h)
    assert r["hebbian_eta_plus"] == 0.0 and r["hebbian_death_ltd"] == 0.0
    assert {k: v for k, v in r.items() if k not in mf_chain.FREQ_OVERRIDES} == \
        {k: v for k, v in h.items() if k not in mf_chain.FREQ_OVERRIDES}
    for arm in ("base", "hebbian", "uniform"):
        assert mf_chain.arm_rule(arm, h) == h
    W0 = np.full((3, 3), 0.1, dtype=np.float32)
    np.fill_diagonal(W0, 0.0)
    _, W1, _ = _one_step(tmp_path, "freq", W0)
    assert not np.allclose(W1, W0)


def test_bonded_arms_show_bonds_and_rank_messages(tmp_path, monkeypatch):
    import agentworld.comms as comms
    seen = []
    real = comms.make_policy
    monkeypatch.setattr(comms, "make_policy", lambda name, **kw: seen.append(name) or real(name, **kw))
    W0 = np.full((3, 3), 0.1, dtype=np.float32)
    np.fill_diagonal(W0, 0.0)
    for arm in mf_chain.ARMS:
        _one_step(tmp_path, arm, W0)
    assert seen == ["recency", "hebbian", "hebbian", "hebbian"]
    assert mf_chain.BONDED_ARMS == ("hebbian", "uniform", "freq")


def test_cli_uniform_and_freq_arms():
    import run_mf_chain
    p = run_mf_chain.build_parser()
    a = p.parse_args(["chains/pendant_staff_4.yaml", "--server-url", "http://x", "--arm", "uniform"])
    assert a.arm == "uniform" and a.uniform_weight == 0.74
    a = p.parse_args(["chains/pendant_staff_4.yaml", "--server-url", "http://x", "--arm", "freq",
                      "--uniform-weight", "0.5"])
    assert a.arm == "freq" and a.uniform_weight == 0.5
