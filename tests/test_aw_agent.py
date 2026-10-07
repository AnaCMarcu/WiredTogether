"""The MindForge stack as AgentWorld cognition, with a scripted fake LLM."""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

import social_stubs  # noqa: F401  (autogen / chromadb stand-ins + sys.path)

from agentworld.agent import AgentConfig, MindForgeCognition, ModelClients
from agentworld.comms import Inbox, Message
from agentworld.scheduler import Turn
from test_aw_scheduler import _build


class FakeLLM:
    """Answers each MindForge module; the action module plays task 02."""

    def __init__(self):
        self.prompts = []

    async def create(self, messages, cancellation_token=None, **kw):
        system = messages[0].content if len(messages) > 1 else ""
        user = messages[-1].content[0]
        self.prompts.append((system, user))
        if "Is the sub-task done?" in user:
            out = {"reasoning": "", "success": "Inventory: " in user and "arrow" in user,
                   "critique": "keep going"}
        elif "What should" in user:
            out = {"reasoning": "", "task": "Contribute materials toward 10 arrows"}
        elif '"beliefs"' in user:
            out = {"beliefs": "noted"}
        else:
            out = self._act(user)
        return SimpleNamespace(content=json.dumps(out),
                               usage=SimpleNamespace(prompt_tokens=len(user) // 4,
                                                     completion_tokens=20))

    @staticmethod
    def _act(user):
        me = re.search(r"You are (\S+)\.", user).group(1)
        inv = dict((k, int(c)) for c, k in re.findall(r"(\d+)x (\w+)", user.split("Inventory:")[1].split("\n")[0]))
        fletcher = me.replace("lumberjack", "fletcher").replace("hunter", "fletcher")
        if "lumberjack" in me:
            if inv.get("logs", 0) >= 3:
                return {"thoughts": "send logs", "action":
                        f"transfer_items(targetPlayer='{fletcher}', itemKey='logs', count=3)",
                        "dm_target": fletcher, "dm_text": "3 logs {sent}", "board_post": ""}
            if "sent" in user.split("Your last action:")[1].split("\n")[0]:
                return {"thoughts": "done", "action": "wait()"}
            return {"thoughts": "need logs", "action": "harvest_resource(targetInstance='t-oak-1')",
                    "board_post": f"have {inv.get('logs', 0)} logs", "post_kind": "status"}
        if "hunter" in me:
            if inv.get("feather", 0) >= 5:
                return {"thoughts": "give feathers", "action":
                        f"transfer_items(targetPlayer='{fletcher}', itemKey='feather', count=5)",
                        "dm_target": "fletcher_agent", "dm_text": "feathers {incoming}"}
            return {"thoughts": "idle", "action": "wait()"}
        if inv.get("logs", 0) >= 1:
            return {"thoughts": "sticks", "action": "craft_item(skill='Fletching', itemKey='stick')"}
        if inv.get("stick", 0) >= 10 and inv.get("feather", 0) >= 10:
            return {"thoughts": "arrows", "action": "craft_item(skill='Fletching', itemKey='arrow')"}
        return {"thoughts": "wait", "action": "wait()", "board_post": "need logs + feathers",
                "post_kind": "need"}


def _clients(llm):
    return ModelClients(action=llm, belief=llm, critic=llm, task=llm)


def test_mindforge_cognition_solves_task_end_to_end(tmp_path):
    llm = FakeLLM()
    sched, world, fake, ex = _build(tmp_path)
    cog = MindForgeCognition(world.global_names, _clients(llm))
    sched.cog = cog
    summary = asyncio.run(sched.run_episode(1))
    ex.shutdown()
    assert summary["teams"]["t00"]["success"] == 1
    # No template placeholder survives into any prompt, braces in DMs included.
    for system, user in llm.prompts:
        assert "{observation}" not in user and "{inbox}" not in user
    acting = [u for s, u in llm.prompts if "Decide now" in u]
    assert any("feathers (incoming)" in u for u in acting)  # DM braces neutralised, delivered
    led = cog.ledger.per_agent()
    assert set(led) == set(world.global_names)
    assert all(v["calls"] >= summary["rounds"] for v in led.values())
    assert cog.ledger.per_module()["action"]["calls"] >= 3 * 5


def _turn(n_names=40, dms=4, bonds=()):
    names = [f"p{i:02d}" for i in range(n_names)]
    inbox = Inbox(agent=0, dms=[Message(1, s, f"hello {s}", "dm", 0, "none", s) for s in range(1, dms + 1)],
                  board=[Message(1, 9, "need wood", "post", None, "need", 99)],
                  contacts=list(range(1, 7)))
    return Turn(round=1, agent=0, name=names[0], team="t00", objective="Make 10 arrows",
                game_context="logs -> sticks", observation={},
                obs_text="You are at (1, 2); HP 69/69.\nInventory: 2x logs.",
                last_result="Episode start", inbox=inbox,
                contacts=[names[j] for j in range(1, 7)], bonds=list(bonds),
                progress_text="Arrows: 0/10", names=names)


def test_prompt_is_bounded_and_partner_beliefs_capped():
    llm = FakeLLM()
    cog = MindForgeCognition([f"p{i:02d}" for i in range(40)], _clients(llm),
                             config=AgentConfig(partner_updates=2))
    turn = _turn(dms=4, bonds=[("p03", 0.9), ("p01", 0.2)])
    asyncio.run(cog.decide(turn))
    partner_calls = [u for _, u in llm.prompts if "Update your belief about player" in u]
    assert len(partner_calls) == 2
    # Hebbian ordering: the strongest bond's sender first.
    assert "player p03" in partner_calls[0] or "player p03" in partner_calls[1]
    action_prompt = next(u for _, u in llm.prompts if "Decide now" in u)
    listed = action_prompt.split("PLAYERS YOU KNOW OF: ")[1].split("\n")[0].split(", ")
    assert len(listed) <= 6
    assert "Your strongest bonds" in action_prompt and "p03 (0.90)" in action_prompt
    agent = cog.agents[0]
    assert set(agent.partner_beliefs) <= {"p01", "p02", "p03", "p04"}


def test_base_arm_prompt_has_no_bonds():
    llm = FakeLLM()
    cog = MindForgeCognition([f"p{i:02d}" for i in range(40)], _clients(llm))
    asyncio.run(cog.decide(_turn(bonds=())))
    action_prompt = next(u for _, u in llm.prompts if "Decide now" in u)
    assert "strongest bonds" not in action_prompt


def test_call_budget_per_round():
    llm = FakeLLM()
    cog = MindForgeCognition([f"p{i:02d}" for i in range(10)], _clients(llm),
                             config=AgentConfig(belief_interval=5, critic_interval=3))
    agent = cog.agents[0]
    for r in range(1, 11):
        t = _turn(n_names=10, dms=0)
        t.round = r
        asyncio.run(agent.decide(t))
    calls = sum(cog.ledger.calls.values())
    # 10 actions + 1 curriculum + critic every 3rd round + 2 belief calls at rounds 1/5/10
    assert calls <= 10 + 1 + 3 + 6 + 3


def test_on_reset_keeps_skills():
    llm = FakeLLM()
    cog = MindForgeCognition(["a", "b"], _clients(llm))
    agent = cog.agents[0]
    agent.skills.add("make sticks -> craft_item(skill=Fletching, itemKey=stick)")
    agent.current_task = "x"
    cog.on_reset()
    assert agent.current_task is None and len(agent.skills) == 1
