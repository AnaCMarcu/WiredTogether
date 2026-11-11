"""Persistent relational state for chained AgentWorld tasks.

AgentWorld's runner builds new agent objects for every task, so an agent's
relational state lives on disk, one file per agent id::

    runs/<chain>/<run_id>/agent_state/a1.json
    {
      "agent_id": "a1",
      "bonds":   {"a2": 0.31, "a3": 0.12},      # W[a1 -> partner], keyed by agent id
      "memory":  [{"step": 1, "task": "task_03_silver_ring", "round": 4,
                   "kind": "xfer_received", "partner": "a2", "partner_role": "smelter",
                   "own_role": "miner", "text": "2x Iron Bar"}, ...],
      "history": [{"step": 1, "task": ..., "role": "miner", "success": true, ...}],
      "skills":  [{"key": "craft_item(skill=Smelting, itemKey=ironbar)", "ok": true,
                   "by_step": {"3": 3}, "gained": {"ironbar": 3}, "uses": [...]}, ...],
      "saved_after_step": 1
    }

``skills`` is the agent's procedural memory: crafts and harvests that produced
items (read from its own inventory, not counting items handed to it) and
crafts that failed, with the error. Earlier tasks' entries are shown in the
``memory`` and ``hebbian`` arms, those naming an item of the current task
first, so a lesson like "smelting ironbar fails for me as miner" carries over.

It is loaded when a task starts and saved when it ends. Bonds are keyed by the
partner's agent id, never by username, so they survive role changes.

Plasticity is the Wired Together rule, unchanged: one
``MultiChannelHebbianGraph`` (three-factor, multi-channel; calibrated rates
from ``agentworld/cli.py``) per task, warm-started from the saved bonds and
updated once per round from the same inputs as our own harness
(``scheduler.py``): ``EventExtractor`` events (transfers, joint combat,
deaths), communication pairs, positions, and the dense third factor from
``ProgressTracker`` plus the task verifier. The rule's short-term traces
(eligibility, salience) start fresh with each task; only W carries over.

Communication is direct only: ``direct_chat(target, message)`` (see
wired_agent.py) replaces AgentWorld's global chat, so every message has one
sender and one receiver. That pair is the communication event the rule sees,
and the message reaches only the receiver's prompt (:class:`Mailbox`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for _p in (HERE, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chain_spec import parse_username, validate_agent_id  # noqa: E402

STATE_VERSION = 1
MEMORY_CAP = 400          # entries kept per agent (oldest dropped)
TEXT_LIMIT = 200          # characters per remembered message
PROMPT_MEMORY_PER_PARTNER = 4
PROMPT_HISTORY = 5
PROMPT_SKILLS = 6
ARMS = ("reference", "memory", "hebbian")
DM_SENT = "Direct message sent to"       # direct_chat's result: "<DM_SENT> a2_smelter: <text>"
PROMPT_INBOX = 8                          # direct messages shown per agent

# Flags of agentworld/cli.py whose defaults are the calibrated rule.
HEBBIAN_FLAGS = ("hebbian_eta_0", "hebbian_eta_plus", "hebbian_decay",
                 "hebbian_eligibility_rho", "hebbian_coact_floor", "hebbian_radius",
                 "hebbian_reward_norm", "hebbian_death_ltd", "hebbian_init_weight",
                 "hebbian_salience", "cofiring_channels", "xfer_directed")


class ChainRunError(RuntimeError):
    """The chain cannot continue (bad agent class, failed login, wrong names)."""


@dataclass(frozen=True)
class Teammate:
    agent_id: str
    role: str
    username: str


class Mailbox:
    """One task's direct messages; each is visible only to its receiver."""

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def post(self, sender: str, sender_name: str, target: str, text: str) -> None:
        self.messages.append({"seq": len(self.messages) + 1, "sender": sender,
                              "sender_name": sender_name, "target": target,
                              "text": str(text)[:TEXT_LIMIT * 2]})

    def to(self, agent_id: str) -> List[Dict[str, Any]]:
        return [m for m in self.messages if m["target"] == agent_id]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── store ────────────────────────────────────────────────────────────────────
def new_state(agent_id: str) -> Dict[str, Any]:
    return {"version": STATE_VERSION, "agent_id": agent_id, "bonds": {}, "memory": [],
            "history": [], "skills": [], "saved_after_step": None, "updated": None}


def inventory(obs: Optional[Dict[str, Any]]) -> Optional[Counter]:
    """Item key -> count from an AgentWorld observation (None if it has no inventory)."""
    items = ((obs or {}).get("inventory") or {}).get("items")
    if items is None:
        return None
    c: Counter = Counter()
    for it in items:
        if isinstance(it, dict) and it.get("key"):
            c[str(it["key"]).lower()] += int(it.get("count", 0) or 0)
    return c


class AgentStateStore:
    """One JSON file per agent id under ``root``."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, agent_id: str) -> Path:
        return self.root / f"{validate_agent_id(agent_id)}.json"

    def load(self, agent_id: str) -> Dict[str, Any]:
        p = self.path(agent_id)
        if not p.is_file():
            return new_state(agent_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("agent_id") != agent_id or data.get("version") != STATE_VERSION:
            raise ChainRunError(f"{p} is not a version-{STATE_VERSION} state file for {agent_id}")
        return data

    def save(self, agent_id: str, state: Dict[str, Any]) -> Path:
        p = self.path(agent_id)
        state = dict(state, updated=_now())
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{agent_id}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, p)
        return p


# ── per-agent session (what the adapter's hooks act on) ──────────────────────
@dataclass
class AgentSession:
    """One agent's view of one chain step; the adapter's hooks write through it."""

    agent_id: str
    role: str
    username: str
    step: int
    task_key: str
    arm: str
    store: AgentStateStore
    state: Dict[str, Any]
    teammates: List[Teammate] = field(default_factory=list)
    bonds: Dict[str, float] = field(default_factory=dict)     # live W row, by agent id
    bonds_start: Dict[str, float] = field(default_factory=dict)
    loaded_from_step: Optional[int] = None
    turns: int = 0
    task_text: str = ""                                        # lower-cased task YAML text
    mailbox: Optional[Mailbox] = None
    # --hold-complete: () -> (verified, verifier message); None = AgentWorld's complete()
    check_done: Optional[Callable[[], Tuple[bool, str]]] = None

    def start(self, teammates: List[Teammate]) -> None:
        self.teammates = list(teammates)
        self.bonds_start = dict(self.bonds)

    def resolve(self, target: str) -> Optional[Teammate]:
        """A teammate by username or agent id (case-insensitive); never this agent."""
        t = str(target or "").strip().lstrip("@").lower()
        return next((m for m in self.teammates if t in (m.username.lower(), m.agent_id)), None)

    def inbox_block(self) -> str:
        """Direct messages this agent received in the current task (every arm)."""
        if self.mailbox is None:
            return ""
        mine = self.mailbox.to(self.agent_id)[-PROMPT_INBOX:]
        body = "\n".join(f"- from {m['sender_name']}: {m['text']}" for m in mine) or "(none yet)"
        return ("\n\n=== DIRECT MESSAGES TO YOU (only you see these; reply with direct_chat) ===\n"
                f"{body}\n=== END DIRECT MESSAGES ===")

    def learn(self, round_num: int, item: Dict[str, Any]) -> None:
        """Add one craft/harvest outcome to the agent's skill memory (aggregated by action)."""
        skills = self.state.setdefault("skills", [])
        key, ok = str(item["key"]), bool(item["ok"])
        entry = next((s for s in skills if s["key"] == key and s["ok"] == ok), None)
        if entry is None:
            entry = {"key": key, "ok": ok, "by_step": {}, "gained": {}, "items": [],
                     "error": "", "uses": []}
            skills.append(entry)
        step = str(self.step)
        entry["by_step"][step] = entry["by_step"].get(step, 0) + 1
        for k, v in (item.get("gained") or {}).items():
            entry["gained"][k] = entry["gained"].get(k, 0) + int(v)
        for k in item.get("items") or []:
            if k and k not in entry["items"]:
                entry["items"].append(k)
        if not ok:
            entry["error"] = str(item.get("error", ""))[:TEXT_LIMIT]
        use = {"step": self.step, "task": self.task_key, "role": self.role}
        if use not in entry["uses"]:
            entry["uses"].append(use)
        entry["last_round"] = round_num

    def _skill_lines(self) -> List[str]:
        """Earlier tasks' skill entries, those naming an item of this task first."""
        rows = []
        for s in self.state.get("skills") or []:
            earlier = {int(k): v for k, v in (s.get("by_step") or {}).items() if int(k) != self.step}
            if not earlier:
                continue
            relevant = any(re.search(rf"\b{re.escape(i)}\b", self.task_text)
                           for i in s.get("items") or [])
            rows.append((not relevant, -max(earlier), -sum(earlier.values()), s, sum(earlier.values())))
        lines = []
        for *_, s, n in sorted(rows, key=lambda r: r[:3])[:PROMPT_SKILLS]:
            uses = [u for u in s.get("uses") or [] if u.get("step") != self.step]
            where = ", ".join(sorted({f"{u['role']} in {u['task']}" for u in uses}))
            if s["ok"]:
                got = ", ".join(f"{v}x {k}" for k, v in sorted(s.get("gained", {}).items()))
                lines.append(f"- {s['key']}: worked {n}x, got {got} (as {where})")
            else:
                lines.append(f"- {s['key']}: failed {n}x (as {where}): \"{s.get('error', '')}\"")
        return lines

    def remember(self, round_num: int, item: Dict[str, Any]) -> None:
        entry = {"step": self.step, "task": self.task_key, "round": round_num,
                 "kind": item["kind"], "partner": item["partner"],
                 "partner_role": item["partner_role"], "own_role": self.role,
                 "text": str(item.get("text", ""))[:TEXT_LIMIT]}
        mem = self.state.setdefault("memory", [])
        mem.append(entry)
        if len(mem) > MEMORY_CAP:
            del mem[: len(mem) - MEMORY_CAP]

    def finish(self, outcome: Dict[str, Any]) -> Path:
        bonds = dict(self.state.get("bonds") or {})
        bonds.update({k: round(float(v), 6) for k, v in self.bonds.items()})
        self.state["bonds"] = bonds
        self.state.setdefault("history", []).append({
            "step": self.step, "task": self.task_key, "role": self.role,
            "username": self.username, "arm": self.arm,
            "teammates": [t.__dict__ for t in self.teammates],
            "loaded_from_step": self.loaded_from_step,
            "success": outcome.get("success"), "message": outcome.get("message"),
            "psr": outcome.get("psr"), "rounds": outcome.get("rounds"),
            "chats": outcome.get("chats"), "turns": self.turns,
            "bonds_start": {k: round(float(v), 6) for k, v in self.bonds_start.items()},
            "bonds_end": {k: round(float(v), 6) for k, v in self.bonds.items()},
        })
        self.state["saved_after_step"] = self.step
        return self.store.save(self.agent_id, self.state)

    # ── prompt ─────────────────────────────────────────────────────────────
    def prompt_block(self) -> str:
        """Relational context appended to the system prompt ('' in the reference arm)."""
        if self.arm == "reference":
            return ""
        lines = ["", "", "=== YOUR TEAM HISTORY (persists across tasks) ===",
                 f"You are agent {self.agent_id}; this task your role is {self.role} "
                 f"(username {self.username})."]
        if self.teammates:
            lines.append("Teammates: " + "; ".join(
                f"{t.agent_id} = {t.username} ({t.role})" for t in self.teammates))
        past = [h for h in self.state.get("history") or [] if h.get("step") != self.step]
        if past:
            lines.append("Earlier tasks with this team:")
            for h in past[-PROMPT_HISTORY:]:
                result = "success" if h.get("success") else "failed"
                lines.append(f"- step {h.get('step')}, {h.get('task')}, you were "
                             f"{h.get('role')}: {result} ({h.get('message')})")
        mem = [m for m in self.state.get("memory") or [] if m.get("step") != self.step]
        shown = []
        for t in self.teammates:
            mine = [m for m in mem if m.get("partner") == t.agent_id][-PROMPT_MEMORY_PER_PARTNER:]
            for m in mine:
                shown.append(f"- with {t.agent_id} (then {m.get('partner_role')}, now {t.username}), "
                             f"step {m.get('step')} {m.get('task')} round {m.get('round')}: "
                             f"{describe(m)}")
        if shown:
            lines.append("Past interactions from earlier tasks (most recent last):")
            lines.extend(shown)
        skills = self._skill_lines()
        if skills:
            lines.append("What you did yourself in earlier tasks:")
            lines.extend(skills)
        if self.arm == "hebbian" and self.teammates:
            ranked = sorted(self.teammates, key=lambda t: -self.bonds.get(t.agent_id, 0.0))
            lines.append("Your strongest bonds (higher = you worked well together recently): "
                         + ", ".join(f"{t.username} ({self.bonds.get(t.agent_id, 0.0):.2f})"
                                     for t in ranked))
        lines.append("=== END TEAM HISTORY ===")
        return "\n".join(lines)


def describe(m: Dict[str, Any]) -> str:
    text = m.get("text", "")
    return {
        "msg_sent": f'you told them: "{text}"',
        "msg_received": f'they told you: "{text}"',
        "xfer_given": f"you gave them {text}",
        "xfer_received": f"they gave you {text}",
    }.get(m.get("kind"), f"{m.get('kind')}: {text}")


# ── parsing one runner response ──────────────────────────────────────────────
_INFO = re.compile(r"\[TOOL_CALL_INFO\] ([^\n]+)")       # the pattern agents/run.py uses
_CALL = re.compile(r"^(\w+)\((.*)\)$", re.S)
_ARG_SPLIT = re.compile(r",\s*(?=[A-Za-z_]\w*=)")


def parse_response(response: str):
    """(ToolCall, full tool-result text) from a ``[TOOL_CALL_INFO]`` response, or (None, '')."""
    from agentworld.actions import ToolCall
    m = _INFO.search(response or "")
    if not m:
        return None, ""
    info = m.group(1).strip()
    cm = _CALL.match(info)
    name, argstr = (cm.group(1), cm.group(2)) if cm else (info, "")
    args: Dict[str, Any] = {}
    for part in _ARG_SPLIT.split(argstr) if argstr.strip() else []:
        key, sep, value = part.partition("=")
        if sep:
            args[key.strip()] = value.strip()
    parts = (response or "").split("[TOOL_RESULT] ", 1)
    return ToolCall(name, args), (parts[1].strip() if len(parts) > 1 else "")


_DM = re.compile(rf"^{re.escape(DM_SENT)} (\S+): (.*)$", re.S)


@dataclass
class TurnInteractions:
    dms: List[Tuple[int, int]] = field(default_factory=list)            # comm pairs for the rule
    by_agent: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def add(self, i: int, item: Dict[str, Any]) -> None:
        self.by_agent.setdefault(i, []).append(item)


def interactions_for_turn(actor: int, call, result: str, team: List[Teammate],
                          names_index: Dict[str, int]) -> TurnInteractions:
    """Messages and transfers in one turn, as memory items for both partners."""
    from agentworld.executor import ActionResult
    out = TurnInteractions()
    if call is None:
        return out
    me = team[actor]
    m = _DM.match(result.strip()) if call.name == "direct_chat" else None
    if m:
        j = names_index.get(m.group(1))
        if j is not None and j != actor:
            message = m.group(2).strip()
            other = team[j]
            out.dms.append((actor, j))
            out.add(actor, {"kind": "msg_sent", "partner": other.agent_id,
                            "partner_role": other.role, "text": message})
            out.add(j, {"kind": "msg_received", "partner": me.agent_id,
                        "partner_role": me.role, "text": message})
    elif call.name == "transfer_items":
        res = ActionResult(agent=actor, call=call, text=result, observation=None, status="",
                           started=0.0, ended=0.0, round_issued=0)
        xfer = res.transfer()
        if xfer:
            # Same receiver resolution as agentworld.events.EventExtractor.
            j = names_index.get(xfer["receiver"], names_index.get(call.args.get("targetPlayer")))
            if j is not None and j != actor:
                other = team[j]
                what = f"{xfer['count']}x {xfer['item']}"
                out.add(actor, {"kind": "xfer_given", "partner": other.agent_id,
                                "partner_role": other.role, "text": what})
                out.add(j, {"kind": "xfer_received", "partner": me.agent_id,
                            "partner_role": me.role, "text": what})
    return out


# ── plasticity: the Wired Together rule over one task ────────────────────────
def hebbian_defaults() -> Dict[str, Any]:
    """The calibrated defaults, read from agentworld/cli.py so they never drift."""
    from agentworld.cli import build_parser
    p = build_parser()
    return {k: p.get_default(k) for k in HEBBIAN_FLAGS}


def graph_config(n: int, init: np.ndarray, h: Dict[str, Any]):
    """MultiChannelConfig exactly as agentworld/cli.py builds it, plus the warm start."""
    from hebbian.multichannel import MultiChannelConfig
    return MultiChannelConfig(
        enabled=True, num_agents=n, eta_0=h["hebbian_eta_0"], eta_plus=h["hebbian_eta_plus"],
        decay=h["hebbian_decay"], eligibility_rho=h["hebbian_eligibility_rho"],
        coact_floor=h["hebbian_coact_floor"], interaction_radius=h["hebbian_radius"],
        reward_norm_R=h["hebbian_reward_norm"], eta_minus_death=h["hebbian_death_ltd"],
        init_weight=h["hebbian_init_weight"], salience_mode=h["hebbian_salience"],
        social_act_channels=tuple(c.strip() for c in h["cofiring_channels"].split(",") if c.strip()),
        channel_symmetric={"xfer": not h["xfer_directed"]},
        init_matrix=init.tolist())


class TeamPlasticity:
    """One task's run of the rule over the team, warm-started from saved bonds."""

    def __init__(self, team: List[Teammate], bonds: Dict[str, Dict[str, float]],
                 task_yaml: Path, verifier: Path, hebbian: Optional[Dict[str, Any]] = None):
        from agentworld.events import EventExtractor
        from agentworld.progress import ProgressTracker
        from agentworld.tasks import compose_world, load_task
        from agentworld.verify import extract_targets
        from hebbian.multichannel import MultiChannelHebbianGraph

        self.team = list(team)
        self.ids = [t.agent_id for t in team]
        self.names = [t.username for t in team]
        n = len(team)
        self.h = {**hebbian_defaults(), **(hebbian or {})}
        w0 = float(self.h["hebbian_init_weight"])
        init = np.full((n, n), w0, dtype=np.float32)
        for i, a in enumerate(self.ids):
            for j, b in enumerate(self.ids):
                if i != j and b in (bonds.get(a) or {}):
                    init[i, j] = float(bonds[a][b])
        np.fill_diagonal(init, 0.0)
        self.graph = MultiChannelHebbianGraph(graph_config(n, init, self.h))
        world = compose_world([load_task(task_yaml)])
        if world.global_names != self.names:
            raise ChainRunError(f"task {task_yaml} names {world.global_names}, runner has {self.names}")
        self.team_id = world.teams[0].team
        self.progress = ProgressTracker(world, {self.team_id: extract_targets(verifier)})
        self.extractor = EventExtractor(self.names)
        self.solved = 0
        self.last_obs: Dict[int, Dict[str, Any]] = {}

    def bonds_of(self, i: int) -> Dict[str, float]:
        W = self.graph.W
        return {self.ids[j]: float(W[i, j]) for j in range(len(self.ids)) if j != i}

    def observe_initial(self, observations: Dict[int, Optional[Dict[str, Any]]]) -> None:
        self.last_obs.update({i: o for i, o in observations.items() if o})
        self.progress.observe_initial(observations)

    def step(self, round_num: int, results: list, observations: Dict[int, Dict[str, Any]],
             dms: List[Tuple[int, int]], verdict: Tuple[bool, str]) -> Dict[str, Any]:
        """One plasticity update, as in RoundScheduler.run_episode."""
        from agentworld.events import hebbian_triples
        from agentworld.observation import position
        from agentworld.verify import progress_from_msg

        n = len(self.team)
        self.last_obs.update({i: o for i, o in observations.items() if o})
        events = self.extractor.from_results(round_num, results)
        events += self.extractor.deaths(round_num, {i: self.last_obs.get(i) for i in range(n)})
        ok, msg = verdict
        if ok:
            self.solved = 1
        p = 1.0 if ok else progress_from_msg(msg)
        bond, death, parts = self.progress.update(round_num, observations, events,
                                                  {self.team_id: p}, {self.team_id: self.solved})
        triples = hebbian_triples(events, dms)
        positions = [position(self.last_obs.get(i)) for i in range(n)]
        self.graph.update(positions, chambers=None, bond_rewards=bond,
                          total_rewards=[b + d for b, d in zip(bond, death)],
                          social_events=triples, death_rewards=death)
        W = self.graph.W
        return {"round": round_num, "verified": bool(ok), "message": msg, "progress": p,
                "events": [e.to_dict() for e in events],
                "comm": [[self.ids[s], self.ids[r]] for s, r in dms],
                "triples": [[self.ids[s], self.ids[r], ch] for s, r, ch in triples],
                "bond_rewards": {self.ids[i]: round(float(b), 4) for i, b in enumerate(bond)},
                "death": {self.ids[i]: d for i, d in enumerate(death) if d},
                "W": {self.ids[i]: {self.ids[j]: round(float(W[i, j]), 5)
                                    for j in range(n) if j != i} for i in range(n)}}


# ── one step, driven by the runner ───────────────────────────────────────────
@dataclass
class _PendingRound:
    round: int
    results: list = field(default_factory=list)
    obs: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    dms: List[Tuple[int, int]] = field(default_factory=list)


class StepContext:
    """Connects one task run to the agents' hooks, the store and the rule.

    The runner calls :meth:`on_agents_ready` after login, :meth:`on_action`
    after every agent turn, :meth:`flush` when the task ends, and run_chain
    calls :meth:`end` with the verified outcome.
    """

    def __init__(self, *, step: int, task_key: str, task_yaml: Path, verifier: Path,
                 slots: Dict[str, Dict[str, str]], store: AgentStateStore, arm: str,
                 verify: Callable[[Dict[str, Any]], Tuple[bool, str]],
                 hebbian: Optional[Dict[str, Any]] = None, log_path: Optional[Path] = None,
                 hold_complete: bool = False):
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        self.step, self.task_key, self.arm = step, task_key, arm
        # --hold-complete: complete() ends an agent only once the step's verifier passes;
        # the runner sets current_traj so the verifier can read the trajectory so far.
        self.hold_complete = bool(hold_complete)
        self.current_traj: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
        self.held_completes = 0
        self.task_yaml, self.verifier = Path(task_yaml), Path(verifier)
        self.slots = slots                      # slot -> {"agent_id", "role", "username"}
        self.slot_order = list(slots)
        self.store, self.verify, self.hebbian = store, verify, hebbian
        self.log_path = log_path
        self.adapters: Dict[str, Any] = {}
        self.sessions: Dict[str, AgentSession] = {}
        self.team: List[Teammate] = []
        self.index: Dict[str, int] = {}
        self.names_index: Dict[str, int] = {}
        self.loaded: Dict[str, Dict[str, Any]] = {}
        self.plasticity: Optional[TeamPlasticity] = None
        self.pending: Optional[_PendingRound] = None
        self.rounds_learned = 0
        self.task_text = self.task_yaml.read_text(encoding="utf-8").lower()
        self.inv: Dict[int, Optional[Counter]] = {}          # inventory after the agent's last turn
        self.received: Dict[int, Counter] = {}               # handed to it since then
        self.harvesting: Dict[int, bool] = {}                # last turn was a harvest without a gain

    def check_done(self) -> Tuple[bool, str]:
        """The step's verifier on the trajectory so far (rounds already recorded)."""
        traj = self.current_traj() if self.current_traj else None
        if not traj or not traj.get("rounds"):
            ok, msg = False, "nothing has been verified yet"
        else:
            try:
                ok, msg = self.verify(traj)
            except Exception as exc:                  # never let a check end the run
                ok, msg = False, f"verification error: {exc}"
        if not ok:
            self.held_completes += 1
        return bool(ok), str(msg)

    # ── start ──────────────────────────────────────────────────────────────
    def on_agents_ready(self, agent_states: Dict[str, Any]) -> None:
        try:
            self._ready(agent_states)
        except Exception:
            for st in agent_states.values():   # do not leave characters logged in
                try:
                    if st.console is not None and st.console.agent.game_tools.token:
                        st.console.agent.game_tools.logout_character()
                except Exception:
                    pass
            raise

    def _ready(self, agent_states: Dict[str, Any]) -> None:
        team = []
        for slot in self.slot_order:
            st = agent_states.get(slot)
            if st is None or st.console is None or getattr(st.state, "value", "") == "failed":
                why = getattr(st, "error_message", None) if st is not None else "not created"
                raise ChainRunError(f"step {self.step}: {slot} failed to initialise: {why}")
            agent = st.console.agent
            uname = st.console.username
            aid, role = parse_username(uname)
            want = self.slots[slot]
            if (aid, uname) != (want["agent_id"], want["username"]):
                raise ChainRunError(f"step {self.step}: {slot} is {uname}, expected {want['username']}")
            for hook in ("attach_chain", "on_task_start", "on_turn", "on_skill", "on_task_end"):
                if not callable(getattr(agent, hook, None)):
                    raise ChainRunError(
                        "the agent config must use provider: custom with a chains/wired_agent.py "
                        f"class (llm.agent_class: wired_agent:WiredOpenAIAgent); {slot} has no {hook}()")
            self.adapters[slot] = agent
            team.append(Teammate(aid, role, uname))
        self.team = team
        self.index = {slot: i for i, slot in enumerate(self.slot_order)}
        self.names_index = {t.username: i for i, t in enumerate(team)}
        self.mailbox = Mailbox()

        states = {t.agent_id: self.store.load(t.agent_id) for t in team}
        for t in team:
            s = states[t.agent_id]
            self.loaded[t.agent_id] = {
                "file_existed": self.store.path(t.agent_id).is_file(),
                "saved_after_step": s.get("saved_after_step"),
                "memory_entries": len(s.get("memory") or []),
                "history_steps": [h.get("step") for h in s.get("history") or []],
                "bonds": dict(s.get("bonds") or {})}
        self.plasticity = TeamPlasticity(team, {a: s.get("bonds") or {} for a, s in states.items()},
                                         self.task_yaml, self.verifier, self.hebbian)
        obs0 = {}
        for slot, i in self.index.items():
            t = team[i]
            session = AgentSession(agent_id=t.agent_id, role=t.role, username=t.username,
                                   step=self.step, task_key=self.task_key, arm=self.arm,
                                   store=self.store, state=states[t.agent_id],
                                   loaded_from_step=states[t.agent_id].get("saved_after_step"),
                                   bonds=self.plasticity.bonds_of(i), task_text=self.task_text,
                                   mailbox=self.mailbox,
                                   check_done=self.check_done if self.hold_complete else None)
            self.sessions[slot] = session
            agent = self.adapters[slot]
            agent.attach_chain(session)
            agent.on_task_start(t.agent_id, t.role,
                                [(o.agent_id, o.role, o.username) for o in team if o is not t])
            try:
                obs0[i] = agent.game_tools.get_last_observation_data()
            except Exception:
                obs0[i] = None
            self.inv[i] = inventory(obs0[i])
            self.received[i] = Counter()
        self.plasticity.observe_initial(obs0)

    # ── turns ──────────────────────────────────────────────────────────────
    def on_action(self, round_num: int, slot: str, response: str,
                  observation: Any, trajectory: Dict[str, Any]) -> None:
        if slot not in self.index or self.plasticity is None:
            return
        if self.pending is not None and self.pending.round != round_num:
            self.flush(trajectory)
        if self.pending is None:
            self.pending = _PendingRound(round_num)
        from agentworld.executor import ActionResult
        i = self.index[slot]
        obs = observation if isinstance(observation, dict) and observation else None
        if obs:
            self.pending.obs[i] = obs
        call, result = parse_response(response)
        self.sessions[slot].turns += 1
        if call is not None:
            res = ActionResult(agent=i, call=call, text=result, observation=obs, status="",
                               started=0.0, ended=0.0, round_issued=round_num)
            self.pending.results.append(res)
            xfer = res.transfer() if call.name == "transfer_items" else None
            if xfer:     # the receiver's next inventory gain is not its own work
                j = self.names_index.get(xfer["receiver"],
                                         self.names_index.get(call.args.get("targetPlayer")))
                key = str(call.args.get("itemKey") or "").lower()
                if j is not None and key:
                    self.received.setdefault(j, Counter())[key] += int(xfer["count"])
        for item in self._skill_outcomes(i, call, result, obs):
            self.adapters[slot].on_skill(round_num, item)
        turn = interactions_for_turn(i, call, result, self.team, self.names_index)
        self.pending.dms.extend(turn.dms)
        self.adapters[slot].on_turn(round_num, turn.by_agent.get(i, []))
        for j, items in turn.by_agent.items():
            if j != i:
                self.adapters[self.slot_order[j]].on_turn(round_num, items)

    def _skill_outcomes(self, i: int, call, result: str,
                        obs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crafts/harvests that produced items for agent i this turn, and failed crafts.

        Gains are read from i's own inventory since its previous turn, minus
        what teammates handed it. A harvest's yield can show up a turn late,
        so a gain after a fruitless harvest is credited to that harvest.
        """
        new = inventory(obs)
        if new is None:
            return []
        prev, self.inv[i] = self.inv.get(i), new
        recv = self.received.pop(i, Counter())
        if prev is None or call is None:
            return []
        gained = (new - prev) - recv
        name = call.name
        if name == "craft_item":
            self.harvesting[i] = False
            item_key = str(call.args.get("itemKey") or "").lower()
            key = f"craft_item(skill={call.args.get('skill', '')}, itemKey={call.args.get('itemKey', '')})"
            if gained:
                return [{"key": key, "ok": True, "gained": dict(gained),
                         "items": [item_key, *gained]}]
            error = (result or "").strip().splitlines()[0] if (result or "").strip() else "no item made"
            return [{"key": key, "ok": False, "error": error, "items": [item_key]}]
        if name == "harvest_resource":
            if gained:
                self.harvesting[i] = False
                return [{"key": "harvest_resource -> " + ", ".join(sorted(gained)), "ok": True,
                         "gained": dict(gained), "items": sorted(gained)}]
            self.harvesting[i] = True
            return []
        if gained and self.harvesting.get(i):
            self.harvesting[i] = False
            return [{"key": "harvest_resource -> " + ", ".join(sorted(gained)), "ok": True,
                     "gained": dict(gained), "items": sorted(gained)}]
        self.harvesting[i] = False
        return []

    def flush(self, trajectory: Dict[str, Any]) -> None:
        """Apply the pending round's plasticity update (its trajectory round is complete)."""
        p, self.pending = self.pending, None
        if p is None or self.plasticity is None:
            return
        try:
            verdict = self.verify(trajectory)
        except Exception as exc:      # a verifier crash must not stop learning
            verdict = (False, f"verifier error: {type(exc).__name__}: {exc}")
        rec = self.plasticity.step(p.round, p.results, p.obs, p.dms, verdict)
        for slot, i in self.index.items():
            self.sessions[slot].bonds = self.plasticity.bonds_of(i)
        self.rounds_learned += 1
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    # ── end ────────────────────────────────────────────────────────────────
    def end(self, outcome: Dict[str, Any]) -> Dict[str, str]:
        saved = {}
        for slot, agent in self.adapters.items():
            aid = self.sessions[slot].agent_id
            agent.on_task_end(aid, outcome)
            saved[aid] = str(self.store.path(aid))
        return saved
