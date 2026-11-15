"""examples/agentworld: a user's own cognition class in the harness, fake server, no model."""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fake_kaetram import FakeGameTools, FakeWorld

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agentworld"
sys.path.insert(0, str(EXAMPLES))

import my_agent  # noqa: E402
import run_my_agent  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "agentworld" / "v1.3_benchmark"


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / "agentworld"
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "game_tools.py").write_text("# placeholder\n")
    shutil.copytree(FIX, root / "data_v0.1_multi" / "v1.3_benchmark")
    world = FakeWorld()
    vendor = SimpleNamespace(root=root, master_password="pw",
                             make_tools=lambda url: FakeGameTools(world, url))
    monkeypatch.setattr("agentworld.vendor.load_game_tools", lambda root=None: vendor)
    return root, world


def _args(root, out, *extra):
    return run_my_agent.parser().parse_args([
        "--agentworld-root", str(root), "--tasks", "task_02_arrow_production",
        "--server-url", "http://fake", "--rounds", "6", "--setup-sleep-scale", "0",
        "--video", "none", "--out", str(out), *extra])


class FakeChat:
    """Stands in for AsyncOpenAI: harvests the oak, messages the first teammate."""

    def __init__(self, reply=None):
        self.requests = []
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, model, messages, **kw):
        self.requests.append(messages)
        user = messages[-1]["content"]
        mate = user.split("TEAMMATES YOU CAN MESSAGE: ")[1].split(",")[0].split("\n")[0].strip()
        content = self.reply if self.reply is not None else json.dumps(
            {"thoughts": "get logs", "action": "harvest_resource(targetInstance='t-oak-1')",
             "dm_target": mate, "dm_text": "harvesting the oak"})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def _llm_class(fake):
    class Agent(my_agent.MyLLMAgent):
        def __init__(self, names):
            super().__init__(names, client=fake)
    return Agent


def _episode(out):
    ep = out / "episode_1"
    summary = json.loads((ep / "summary.json").read_text(encoding="utf-8"))
    msgs = [json.loads(l) for l in (ep / "messages.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    trajs = sorted((ep / "trajectories").glob("*_trajectory.json"))
    return summary, msgs, trajs


def test_scripted_agent_runs_end_to_end(fake_root, tmp_path):
    root, world = fake_root
    out = tmp_path / "run"
    summaries = asyncio.run(run_my_agent.run(_args(root, out, "--agent", "my_agent:ScriptedAgent")))
    assert len(summaries) == 1 and summaries[0]["agent"] == "my_agent:ScriptedAgent"
    summary, msgs, trajs = _episode(out)
    assert summary["rounds"] == 6 and summary["tokens"] is None
    assert trajs, "AgentWorld-schema trajectory written for the verifier"
    assert any(m["kind"] == "dm" and m["delivered"] for m in msgs)
    assert (out / "episode_1" / "hebbian_W.npy").is_file()
    traj = json.loads(trajs[0].read_text(encoding="utf-8"))
    actions = [a["action"] for r in traj["rounds"] for a in r["actions"]]
    assert any(a.startswith("move_character") for a in actions)


@pytest.mark.parametrize("arm,bonds_shown", [("base", False), ("hebbian", True)])
def test_llm_agent_gets_the_turn_and_its_decisions_run(fake_root, tmp_path, monkeypatch, arm, bonds_shown):
    root, world = fake_root
    fake = FakeChat()
    monkeypatch.setattr(my_agent, "FakeAgent", _llm_class(fake), raising=False)
    out = tmp_path / arm
    asyncio.run(run_my_agent.run(_args(root, out, "--agent", "my_agent:FakeAgent", "--arm", arm)))
    summary, msgs, trajs = _episode(out)
    users = [r[-1]["content"] for r in fake.requests]
    assert users and all("OBJECTIVE:" in u and "YOUR OBSERVATION:" in u for u in users)
    assert any("YOUR STRONGEST BONDS" in u for u in users) == bonds_shown
    assert sum(summary["tokens"]["calls"].values()) == len(fake.requests)
    assert any(m["kind"] == "dm" and m["text"] == "harvesting the oak" for m in msgs)
    traj = json.loads(trajs[0].read_text(encoding="utf-8"))
    actions = [a["action"] for r in traj["rounds"] for a in r["actions"]]
    assert any(a.startswith("harvest_resource") for a in actions)


def test_unusable_reply_becomes_wait(fake_root):
    fake = FakeChat(reply="I think I will go north.")
    agent = _llm_class(fake)(["a", "b"])
    turn = SimpleNamespace(agent=0, name="a", round=1, objective="o", game_context="g", obs_text="obs",
                           last_result="none", progress_text="p", contacts=["b"], bonds=[],
                           names=["a", "b"], inbox=SimpleNamespace(dms=[]))
    d = asyncio.run(agent.decide(turn))
    assert d.action == "wait()" and d.dm_text == "" and "unusable" in d.thoughts
