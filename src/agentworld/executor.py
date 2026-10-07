"""Run every agent's tool call concurrently against the game server.

Each agent owns one AgentWorld ``KaetramGameTools`` object (see
:mod:`agentworld.vendor`). A tool call is that object's own method, run in a
worker thread, so the game logic (auto-approach, combat polling, loot pickup,
the two-step transfer) is exactly the benchmark's. After the action, the
executor takes the same radius-1 observation AgentWorld's runner records in
the trajectory (``console.get_player_status`` → ``get_last_observation_data``).

Long actions never stall a round: :meth:`Executor.act_all` waits at most
``deadline_s``. An agent whose call is still running is *busy* — the scheduler
skips its cognition — and its result is returned by whichever later
``act_all`` call sees it finish. All agents act at once, so a round costs the
slowest action (capped by the deadline), not the sum of all actions as in the
serial round-robin runner.
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentworld.actions import ToolCall
from agentworld.tasks import AgentSpec
from agentworld.trajectory import status_line

OBSERVE_RADIUS_TRAJ = 1   # what AgentWorld's runner logs after each action

_XFER = re.compile(r"Transfer completed: (\d+)x (.+?) transferred from .+? to (\S+)")
_CRAFT = re.compile(r"Successfully crafted (\d+)x (.+?)!")
_KILL = re.compile(r"VICTORY")


@dataclass
class ActionResult:
    agent: int
    call: ToolCall
    text: str
    observation: Optional[Dict[str, Any]]
    status: str
    started: float
    ended: float
    round_issued: int

    @property
    def success(self) -> bool:
        t = self.text or ""
        if self.call.name == "transfer_items":
            return bool(_XFER.search(t))
        if self.call.name == "craft_item":
            return bool(_CRAFT.search(t))
        if self.call.name == "attack_entity":
            return bool(_KILL.search(t))
        return not (t.startswith("Error") or t.startswith("Failed") or "Error:" in t[:40])

    def transfer(self) -> Optional[Dict[str, Any]]:
        m = _XFER.search(self.text or "")
        if not m:
            return None
        return {"count": int(m.group(1)), "item": m.group(2), "receiver": m.group(3),
                "itemKey": self.call.args.get("itemKey")}

    def crafted(self) -> Optional[Dict[str, Any]]:
        m = _CRAFT.search(self.text or "")
        if not m:
            return None
        return {"count": int(m.group(1)), "item": m.group(2),
                "itemKey": self.call.args.get("itemKey")}

    @property
    def killed(self) -> bool:
        return self.call.name == "attack_entity" and bool(_KILL.search(self.text or ""))


@dataclass
class AgentHandle:
    index: int
    username: str
    key: str                  # agent key within its task ("agent_2")
    team: str
    tools: Any                # KaetramGameTools (or a test double)
    future: Optional[cf.Future] = field(default=None, repr=False)
    last_observation: Optional[Dict[str, Any]] = None
    last_status: str = ""


def observe(tools: Any, radius: int) -> Optional[Dict[str, Any]]:
    """AgentWorld's post-processed observation at ``radius`` (None on failure)."""
    tools.observe_environment({"radius": radius})
    return tools.get_last_observation_data()


def run_tool(tools: Any, call: ToolCall) -> str:
    """Dispatch one call to the tools object (``wait`` is a harness no-op)."""
    if call.name == "wait" or not call.ok:
        return f"No action taken ({call.error})" if call.error else "Waited."
    method = getattr(tools, call.name, None)
    if method is None:
        return f"Error: tool {call.name} not available"
    return str(method(dict(call.args)))


class Executor:
    def __init__(self, handles: List[AgentHandle], deadline_s: float = 30.0,
                 clock: Callable[[], float] = time.time):
        self.handles = handles
        self.deadline_s = deadline_s
        self.clock = clock
        self._pool = cf.ThreadPoolExecutor(max_workers=max(1, len(handles)),
                                           thread_name_prefix="aw-act")
        self._ready: List[ActionResult] = []

    def busy(self, index: int) -> bool:
        f = self.handles[index].future
        return f is not None and not f.done()

    def _work(self, h: AgentHandle, call: ToolCall, round_num: int) -> ActionResult:
        started = self.clock()
        try:
            text = run_tool(h.tools, call)
        except Exception as exc:  # a crashing tool must not kill the round
            text = f"Error: {type(exc).__name__}: {exc}"
        try:
            obs = observe(h.tools, OBSERVE_RADIUS_TRAJ)
        except Exception:
            obs = None
        return ActionResult(agent=h.index, call=call, text=text, observation=obs,
                            status=status_line(obs), started=started, ended=self.clock(),
                            round_issued=round_num)

    def _harvest(self) -> None:
        """Move finished futures into the ready buffer (never drop a result)."""
        for h in self.handles:
            f = h.future
            if f is not None and f.done():
                res = f.result()
                h.future = None
                h.last_observation = res.observation or h.last_observation
                h.last_status = res.status
                self._ready.append(res)

    def submit(self, calls: Dict[int, ToolCall], round_num: int) -> List[int]:
        """Start the calls of idle agents; returns the indices actually started."""
        self._harvest()
        started = []
        for idx, call in calls.items():
            if self.busy(idx):
                continue
            h = self.handles[idx]
            h.future = self._pool.submit(self._work, h, call, round_num)
            started.append(idx)
        return started

    async def collect(self, deadline_s: Optional[float] = None) -> List[ActionResult]:
        """Wait (≤ deadline) for in-flight calls; return every finished result.

        Ordered by (round issued, agent). An agent can appear twice when a
        long action from an earlier round finished alongside a new one.
        """
        pending = [h.future for h in self.handles if h.future is not None]
        if pending:
            wrapped = [asyncio.wrap_future(f) for f in pending]
            await asyncio.wait(wrapped, timeout=self.deadline_s if deadline_s is None else deadline_s)
        self._harvest()
        out = sorted(self._ready, key=lambda r: (r.round_issued, r.agent, r.started))
        self._ready = []
        return out

    async def act_all(self, calls: Dict[int, ToolCall], round_num: int,
                      deadline_s: Optional[float] = None) -> List[ActionResult]:
        self.submit(calls, round_num)
        return await self.collect(deadline_s)

    async def observe_all(self, radius: int, indices: Optional[List[int]] = None
                          ) -> Dict[int, Optional[Dict[str, Any]]]:
        """Decision-time observations of idle agents, in parallel."""
        loop = asyncio.get_running_loop()
        idx = [i for i in (indices if indices is not None else range(len(self.handles)))
               if not self.busy(i)]
        futs = [loop.run_in_executor(self._pool, observe, self.handles[i].tools, radius)
                for i in idx]
        results = await asyncio.gather(*futs, return_exceptions=True)
        return {i: (None if isinstance(r, BaseException) else r) for i, r in zip(idx, results)}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# ── task setup (sequence of AgentWorld's console, agents/console.py) ────────
def _ok(text: str) -> bool:
    low = (text or "").lower()
    return "successfully" in low and "token obtained" in low


def login_or_create(tools: Any, username: str, master_password: str) -> bool:
    """``handle_new_character_creation``: master-password login, else create."""
    if _ok(tools.login_character({"username": username, "password": master_password})):
        return True
    return _ok(tools.create_character({"username": username, "password": master_password}))


def reset_character(tools: Any) -> None:
    """``reset_character_to_default``: clear gear, skills to 1, base HP/MP, spawn."""
    tools.clear_equipment()
    tools._make_request("POST", "/ai/setInventory",
                        {"token": tools.token, "items": [], "clearFirst": True})
    tools.set_combat_level({"level": 1})
    for skill in ("lumberjacking", "mining", "fishing", "cooking", "smithing",
                  "crafting", "fletching", "foraging"):
        tools.set_individual_skill_level({"skill": skill, "level": 1})
    tools._make_request("POST", "/ai/setPlayerStatus",
                        {"token": tools.token, "hitPoints": 69, "mana": 44,
                         "maxHitPoints": 69, "maxMana": 44})
    tools.teleport_character({"x": 250, "y": 180, "withAnimation": False})


def apply_initial_state(tools: Any, spec: AgentSpec, sleep: Callable[[float], None] = time.sleep,
                        location: Optional[tuple] = None) -> None:
    """``apply_initial_state``: position (≤10 tiles), skills, HP/MP, gear, items."""
    target = location if location is not None else spec.location
    if target:
        tx, ty = target
        for _ in range(5):
            obs = tools._make_request("GET", "/ai/observe",
                                      params={"token": tools.token, "radius": 1})
            if obs.get("status") != "success":
                sleep(0.5)
                continue
            loc = obs.get("location", {})
            if abs(loc.get("x", 0) - tx) + abs(loc.get("y", 0) - ty) <= 10:
                break
            tools.teleport_character({"x": tx, "y": ty, "withAnimation": False})
            sleep(1.5)
    if spec.skill_levels:
        for skill, level in spec.skill_levels.items():
            tools.set_individual_skill_level({"skill": skill, "level": level})
        sleep(1.0)
        tools.restore_hp_mp()
    for item in spec.equipped_items:
        parts = item.split(":")
        tools.give_and_equip_item({
            "itemKey": parts[0],
            "count": int(parts[1]) if len(parts) > 1 else 1,
            "enchantmentLevel": int(parts[2]) if len(parts) > 2 else 0,
        })
    if spec.inventory_items:
        items = []
        for item in spec.inventory_items:
            parts = item.split(":")
            entry: Dict[str, Any] = {"key": parts[0], "count": int(parts[1]) if len(parts) > 1 else 1}
            enchant = int(parts[2]) if len(parts) > 2 else 0
            if enchant > 0:
                entry["enchantments"] = {"damage": enchant, "accuracy": enchant, "defense": enchant}
            items.append(entry)
        tools.set_inventory({"items": items, "clearFirst": False})


def prepare_agent(tools: Any, spec: AgentSpec, username: str, master_password: str,
                  sleep: Callable[[float], None] = time.sleep,
                  location: Optional[tuple] = None) -> bool:
    """Log in (creating if needed), reset, and apply the task's initial state."""
    if not login_or_create(tools, username, master_password):
        return False
    if spec.new_character:
        reset_character(tools)
    apply_initial_state(tools, spec, sleep=sleep, location=location)
    return True
