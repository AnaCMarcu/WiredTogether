"""Templates for your own agent in the Wired Together AgentWorld harness.

The harness (``src/agentworld/scheduler.py``) runs rounds. In each round every
agent that is free gets a :class:`Turn` and must return a :class:`Decision`.
All agents decide at the same time (``asyncio.gather``), so ``decide`` must be
``async`` and must not block the event loop for long (wrap blocking calls in
``asyncio.to_thread``).

The contract ``run_my_agent.py`` expects from the class you name with
``--agent module:Class``:

    cls(names: list[str])                 one object for the whole team;
                                          names[i] is agent i's in-game username
    async decide(turn) -> Decision        one call per agent per round
    on_reset()                            optional: called before every episode
    ledger                                optional: an object with .snapshot() -> dict
                                          (token counts), saved in summary.json

``turn.agent`` says which agent the call is for, so keep per-agent state in a
dict or list keyed by it (see MyLLMAgent).

Two classes here:

* ``ScriptedAgent``: no model. Use it to check the setup (server, logins,
  actions, messages, output files) before spending any tokens.
* ``MyLLMAgent``: one call to an OpenAI-compatible endpoint per turn. Copy it
  and replace the parts marked YOUR CODE with your own modules (memory,
  planner, beliefs, tools ...).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Dict, List, Optional

from agentworld.actions import safe_text, tool_reference
from agentworld.observation import position
from agentworld.scheduler import Decision, Turn


class ScriptedAgent:
    """No model: says hello once, then steps left and right. For plumbing checks."""

    def __init__(self, names: List[str]):
        self.names = list(names)

    def on_reset(self) -> None:
        pass

    async def decide(self, turn: Turn) -> Decision:
        if turn.round == 1 and turn.contacts:
            return Decision(action="wait()", thoughts="greet a teammate",
                            dm_target=turn.contacts[0], dm_text=f"hello from {turn.name}")
        pos = position(turn.observation)
        if pos is None:
            return Decision(action="wait()", thoughts="no observation yet")
        x, y = int(pos[0]), int(pos[1])
        dx = 1 if turn.round % 2 == 0 else -1
        return Decision(action=f"move_character(x={x + dx}, y={y})", thoughts="pace")


SYSTEM = """You are {name}, one player in a team in a 2D online role-playing game.
Each round you choose exactly ONE game action. You may also send ONE private
message to ONE teammate in the same reply; only that teammate reads it.

Game actions (write the call exactly like this):
{tools}

Rules:
- Use only instance ids that appear in your CURRENT observation.
- complete() ends your participation; call it only when the team objective is met.
- wait() does nothing this round.

Reply with one JSON object and nothing else:
{{"thoughts": "<short reasoning>", "action": "<one call from the list>",
  "dm_target": "<teammate username or null>", "dm_text": "<message or empty>"}}"""

USER = """Round {round}.

OBJECTIVE:
{objective}

GAME CONTEXT:
{game_context}

YOUR OBSERVATION:
{obs}

RESULT OF YOUR LAST ACTION:
{last_result}

TEAM PROGRESS:
{progress}

MESSAGES TO YOU:
{dms}

TEAMMATES YOU CAN MESSAGE: {contacts}
{bonds}{notes}"""


def _fmt_dms(turn: Turn) -> str:
    if not turn.inbox.dms:
        return "(none)"
    return "\n".join(f"- from {turn.names[m.sender]}: {safe_text(m.text, 400)}"
                     for m in turn.inbox.dms)


def _first_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Ledger:
    """Token counts per agent, saved into each episode's summary.json."""

    def __init__(self):
        self.calls: Dict[str, int] = {}
        self.prompt: Dict[str, int] = {}
        self.completion: Dict[str, int] = {}

    def record(self, agent: str, usage) -> None:
        self.calls[agent] = self.calls.get(agent, 0) + 1
        if usage is not None:
            self.prompt[agent] = self.prompt.get(agent, 0) + int(usage.prompt_tokens or 0)
            self.completion[agent] = self.completion.get(agent, 0) + int(usage.completion_tokens or 0)

    def snapshot(self) -> dict:
        return {"calls": dict(self.calls), "prompt": dict(self.prompt),
                "completion": dict(self.completion)}


class MyLLMAgent:
    """One model call per turn, JSON reply. Copy it and add your own modules.

    Model settings (any OpenAI-compatible server: vLLM, OpenRouter, OpenAI ...):
        MY_LLM_BASE_URL   e.g. http://localhost:8000/v1
        MY_LLM_MODEL      the model id the server expects
        MY_LLM_API_KEY    leave unset for a local server without a key
        MY_LLM_MAX_CONCURRENCY   requests in flight at once (default 8)
    """

    def __init__(self, names: List[str], client=None):
        self.names = list(names)
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=os.environ.get("MY_LLM_BASE_URL"),
                                 api_key=os.environ.get("MY_LLM_API_KEY", "none"))
        self.client = client
        self.model = os.environ.get("MY_LLM_MODEL", "")
        self.sem = asyncio.Semaphore(int(os.environ.get("MY_LLM_MAX_CONCURRENCY", "8")))
        self.ledger = Ledger()
        self.notes: Dict[int, List[str]] = {}       # YOUR CODE: per-agent state lives here

    def on_reset(self) -> None:
        """Called before every episode. Decide what your agents keep between episodes."""
        self.notes = {}

    # ── YOUR CODE: what the agent remembers and how it builds its prompt ──────
    def remember(self, turn: Turn, decision: Decision) -> None:
        """Keep the last few actions per agent (a stand-in for your memory module)."""
        log = self.notes.setdefault(turn.agent, [])
        log.append(f"round {turn.round}: {decision.action} -> {safe_text(turn.last_result, 120)}")
        del log[:-5]

    def build_messages(self, turn: Turn) -> List[dict]:
        bonds = ""
        if turn.bonds:      # empty in the base arm; see AGENTWORLD.md, "Arms"
            bonds = "YOUR STRONGEST BONDS: " + ", ".join(f"{n} ({w:.2f})" for n, w in turn.bonds) + "\n"
        notes = self.notes.get(turn.agent)
        notes = ("YOUR RECENT ACTIONS:\n" + "\n".join(notes) + "\n") if notes else ""
        system = SYSTEM.format(name=turn.name, tools=tool_reference())
        user = USER.format(round=turn.round, objective=safe_text(turn.objective),
                           game_context=safe_text(turn.game_context), obs=turn.obs_text,
                           last_result=turn.last_result, progress=turn.progress_text,
                           dms=_fmt_dms(turn), contacts=", ".join(turn.contacts) or "(none)",
                           bonds=bonds, notes=notes)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ── the call the harness makes ───────────────────────────────────────────
    async def decide(self, turn: Turn) -> Decision:
        messages = self.build_messages(turn)
        async with self.sem:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=512)
        self.ledger.record(turn.name, getattr(resp, "usage", None))
        reply = _first_json(resp.choices[0].message.content or "")
        if reply is None:
            decision = Decision(action="wait()", thoughts="unusable reply (no JSON)")
        else:
            # A malformed call is passed on as written: the harness rejects it and the
            # error comes back next round in turn.last_result, so the model sees it.
            action = str(reply.get("action") or "wait()")
            target = reply.get("dm_target")
            decision = Decision(action=action, thoughts=str(reply.get("thoughts") or "")[:500],
                                dm_target=str(target) if target else None,
                                dm_text=str(reply.get("dm_text") or "") if target else "")
        self.remember(turn, decision)
        return decision
