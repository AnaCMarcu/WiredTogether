"""MindForge agents on a chain of AgentWorld tasks (our harness, ``src/agentworld/``).

``run_chain.py`` plays a chain with AgentWorld's own runner and LLM agent. This
module plays the same chains with the MindForge cognitive stack
(``agentworld.agent``: action selection, beliefs, critic, curriculum, skill and
episode memory) through the round loop of ``agentworld.scheduler``, so a chain
can be compared between MindForge (``base`` arm) and MindForge with social
plasticity (``hebbian`` arm). ``run_mf_chain.py`` is the command line.

Carried from one step to the next, per agent id (a1, a2, ...):

* the MindForge agent itself, with its skill and episode memory. Working memory
  (sub-task, critique, beliefs, partner beliefs, conversation log) is reset at
  every step, exactly as between MindForge episodes (``AgentWorldAgent.on_reset``);
* the bonds W. Each step builds the rule with ``wired_state.graph_config`` (the
  configuration the chains/ runs use), warm-started from the saved W, so the
  eligibility and salience traces start fresh with every task, as in
  ``wired_state.TeamPlasticity``. W is kept in agent-id order and permuted into
  the step's slot order, so chains that rotate roles keep each bond with its pair.

The protocol follows the chains/ runs where this harness allows it:

* direct messages only (``board=False``): the board's read budget is 0, the board
  is removed from the prompt (:func:`direct_only_prompts`), a ``chat()`` call is
  refused with a note, and the task text points at the reply's ``dm_target`` and
  ``dm_text`` fields instead of AgentWorld's ``direct_chat`` tool;
* ``hold_complete``: ``complete()`` before the step's verifier has passed is
  refused, the agent is told the objective is not met and keeps playing, and the
  call counts as an action (``run_chain.py --hold-complete``). The verdict is the
  previous round's, as in chains/;
* ``max_actions`` (the task's ``max_action_steps``, 25 in the v1.3 tasks): an
  agent stops after that many game actions. ``wait()`` and refused or unparsable
  calls do not count, as AgentWorld's runner does not count turns without a tool
  call.

What stays this harness's own: all agents act at once in each round
(AgentWorld's runner is serial), rounds are padded to ``min_round_s``, and a
message travels with an action instead of taking a turn of its own.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for _p in (HERE, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agentworld.actions import ToolCall, parse_action, safe_text  # noqa: E402
from agentworld.events import EventExtractor, InteractionEvent, hebbian_triples  # noqa: E402
from agentworld.observation import position, render_obs_text  # noqa: E402
from agentworld.scheduler import Decision, RoundScheduler, Turn  # noqa: E402
from agentworld.trajectory import RunLog, TeamTrajectory  # noqa: E402
from agentworld.trajectory import status_line  # noqa: E402
from agentworld.verify import progress_from_msg  # noqa: E402

ARMS = ("base", "hebbian")
# generate_tasks.DIRECT_MESSAGES names AgentWorld's direct_chat tool, which this
# harness does not have: here a message is two fields of the agent's reply.
DM_WORDING = "direct messages (dm_target and dm_text in your reply, one teammate per message)"
HELD = ("Not done yet: the team's objective is not met ({msg}). Keep working on it; "
        "the task ends only once the objective is verified.")
NO_CHAT = ("Error: there is no group chat or board. Send a direct message to one teammate "
           "with dm_target and dm_text in your reply.")


class ChainRunError(RuntimeError):
    pass


# ── prompts ──────────────────────────────────────────────────────────────────
_SYSTEM_EDITS = (
    ("- may post ONE short note on the shared board (some players will see it next round).\n", ""),
    ("You only see a few direct messages, a few board posts and a short list of known players "
     "each round. You cannot message everyone, so choose who to talk to.",
     "There is no group chat or board: every message goes to one named player. You only see a "
     "few direct messages and a short list of known players each round, so choose who to talk to."),
    ('- chat(message="<text>")\n', ""),
    (', "board_post": "<short note, or empty>", "post_kind": "<need | offer | status | done | none>"',
     ""),
)
_INSTRUCTION_EDITS = (("BOARD POSTS YOU CAN SEE:\n{board}\n\n", ""),)


def _edit(text: str, edits, what: str) -> str:
    for old, new in edits:
        if old not in text:
            raise ChainRunError(f"{what} no longer contains {old[:70]!r}: update mf_chain.py")
        text = text.replace(old, new)
    return text


def direct_only_prompts(system: str, instruction: str) -> Tuple[str, str]:
    """The MindForge AgentWorld prompts without the board and the chat tool."""
    return (_edit(system, _SYSTEM_EDITS, "agentworld/prompts/aw_system.txt"),
            _edit(instruction, _INSTRUCTION_EDITS, "agentworld/prompts/aw_instruction.txt"))


def use_direct_only_prompts() -> None:
    """Patch agentworld.agent's prompts; call before any MindForge agent is built."""
    import agentworld.agent as A
    A.SYSTEM_PROMPT, A.INSTRUCTION = direct_only_prompts(A.SYSTEM_PROMPT, A.INSTRUCTION)


def point_at_reply_fields(text: str) -> str:
    from generate_tasks import DIRECT_MESSAGES
    return text.replace(DIRECT_MESSAGES, DM_WORDING)


# ── the round loop ───────────────────────────────────────────────────────────
class ChainRoundScheduler(RoundScheduler):
    """``RoundScheduler.run_episode`` plus hold-complete, the action cap and no board.

    The loop is a copy of the parent's; every added or changed line is marked
    ``# [chain]``. Everything else (observation, routing, the batched decision
    wave, concurrent actions, verification, the dense reward, the plasticity
    update, logs and trajectories) is the parent's code unchanged.
    """

    def __init__(self, *args, hold_complete: bool = False, max_actions: Optional[int] = None,
                 board: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.hold_complete = hold_complete
        self.max_actions = max_actions or None
        self.board = board

    async def run_episode(self, episode: int = 1) -> Dict[str, Any]:
        ep_dir = self.out_dir / f"episode_{episode}"
        log = RunLog(ep_dir)
        trajs = {t.team: TeamTrajectory(t) for t in self.world.teams}
        extractor = EventExtractor(self.names)
        n = self.world.n_agents
        max_rounds = self.cfg.max_rounds or self.world.rounds
        done = [False] * n
        solved: Dict[str, int] = {t.team: 0 for t in self.world.teams}
        stopped: Dict[str, bool] = {t.team: False for t in self.world.teams}
        last_text = ["Episode start: no action yet."] * n
        last_obs: Dict[int, Optional[Dict[str, Any]]] = {}
        team_msg: Dict[str, str] = {t.team: "" for t in self.world.teams}
        t0 = time.time()
        actions = [0] * n                      # [chain] game actions per agent
        held = [0] * n                         # [chain] complete() refused before verification
        refused_chat = [0] * n                 # [chain] chat() refused (no group chat)
        capped: Dict[int, int] = {}            # [chain] agent -> round it reached max_actions
        completed: Dict[int, int] = {}         # [chain] agent -> round its complete() went through
        last_active = 0                        # [chain]

        obs0 = await self.ex.observe_all(self.cfg.obs_radius)
        last_obs.update(obs0)
        self.progress.observe_initial(obs0)

        rounds_run = 0
        for rnd in range(1, max_rounds + 1):
            rounds_run = rnd
            round_start = time.time()
            active = [i for i in range(n)
                      if not done[i] and not stopped[self.world.global_team[i]]]
            idle = [i for i in active if not self.ex.busy(i)]
            if not active:
                break
            last_active = rnd                  # [chain]
            fresh = await self.ex.observe_all(self.cfg.obs_radius, idle)
            for i, o in fresh.items():
                if o:
                    last_obs[i] = o
            positions = [position(last_obs.get(i)) for i in range(n)]
            W = self._W()
            inboxes = self.router.route(rnd, W, positions, readers=idle)

            turns: Dict[int, Turn] = {}
            for i in idle:
                slot = self.world.team_of(i)
                ib = inboxes[i]
                turns[i] = Turn(
                    round=rnd, agent=i, name=self.names[i], team=slot.team,
                    objective=slot.task.objective, game_context=slot.task.game_context,
                    observation=last_obs.get(i),
                    obs_text=render_obs_text(last_obs.get(i), self.cfg.obs_k, self.hide),
                    last_result=safe_text(last_text[i], 600), inbox=ib,
                    contacts=[self.names[j] for j in ib.contacts],
                    bonds=self._bonds(i, ib.contacts),
                    progress_text=safe_text(team_msg[slot.team] or "no progress reported yet"),
                    names=self.names,
                )
            decided = await asyncio.gather(*(self.cog.decide(turns[i]) for i in idle),
                                           return_exceptions=True)
            decisions: Dict[int, Decision] = {}
            for i, d in zip(idle, decided):
                decisions[i] = d if isinstance(d, Decision) else Decision(
                    action="wait()", thoughts=f"cognition error: {d!r}"[:300])

            for tr in trajs.values():
                tr.begin_round(rnd)

            # ── communication ───────────────────────────────────────────────
            dms: List[Tuple[int, int]] = []
            calls: Dict[int, ToolCall] = {}
            notes: Dict[int, str] = {}         # [chain] harness replies to refused calls
            for i, d in decisions.items():
                key = self.world.global_key[i]
                tr = trajs[self.world.global_team[i]]
                obs_i, status_i = last_obs.get(i), status_line(last_obs.get(i))
                call = parse_action(d.action)
                if not self.board:                                          # [chain]
                    d.board_post = ""                                       # [chain]
                    if call.name == "chat":                                 # [chain]
                        refused_chat[i] += 1                                # [chain]
                        notes[i] = NO_CHAT                                  # [chain]
                        call = ToolCall("wait")                             # [chain]
                if (call.name == "complete" and self.hold_complete           # [chain]
                        and not solved[self.world.global_team[i]]):         # [chain]
                    held[i] += 1                                            # [chain]
                    actions[i] += 1      # a held complete() is an action   # [chain]
                    msg = team_msg[self.world.global_team[i]] or "not checked yet"  # [chain]
                    notes[i] = HELD.format(msg=msg)                         # [chain]
                    log.write("chain_events.jsonl", {"round": rnd, "agent": i,  # [chain]
                              "event": "complete_held", "verifier": msg})  # [chain]
                    call = ToolCall("wait")                                 # [chain]
                if self.cfg.comm_mode == "side" and call.name == "chat":
                    d.board_post = d.board_post or str(call.args.get("message", ""))
                    call = ToolCall("wait")
                if self.cfg.comm_mode == "side":
                    if d.dm_text.strip():
                        j = self._resolve(d.dm_target, i)
                        msg = self.router.send_dm(rnd, i, j, d.dm_text.strip()) if j is not None else None
                        log.message({"round": rnd, "kind": "dm", "sender": i, "receiver": j,
                                     "target_raw": d.dm_target, "text": d.dm_text.strip(),
                                     "delivered": msg is not None})
                        if msg is not None:
                            dms.append((i, j))
                            tr.add_dm(key, self.names[j], msg.text, status_i, obs_i)
                            if self.mirror is not None:
                                self.mirror(i, f"@{self.names[j]}: {msg.text}")
                    if d.board_post.strip():
                        m = self.router.post(rnd, i, d.board_post.strip(), d.post_kind)
                        log.message({"round": rnd, "kind": "post", "sender": i,
                                     "post_kind": m.post_kind, "text": m.text, "msg_id": m.msg_id})
                        tr.add_board_post(key, m.text, status_i, obs_i)
                        if self.mirror is not None:
                            self.mirror(i, f"[board] {m.text}")
                calls[i] = call

            # ── act ──────────────────────────────────────────────────────────
            results = await self.ex.act_all(calls, rnd)
            res_obs: Dict[int, Optional[Dict[str, Any]]] = {}
            for res in results:
                i = res.agent
                tr = trajs[self.world.global_team[i]]
                thinking = decisions[i].thoughts if res.round_issued == rnd and i in decisions else ""
                tr.add_action(self.world.global_key[i], res.status, res.call.traj_string(),
                              res.observation, safe_text(thinking, 2000))
                last_text[i] = res.text
                if res.observation:
                    res_obs[i] = res.observation
                if res.call.name == "complete" and res.success:
                    done[i] = True
                    completed[i] = rnd                                      # [chain]
                if res.call.name != "wait" and res.call.ok:                 # [chain]
                    actions[i] += 1                                         # [chain]
            for i, note in notes.items():                                   # [chain]
                last_text[i] = note                                         # [chain]
            for i in range(n):                                              # [chain]
                if (self.max_actions and not done[i] and not self.ex.busy(i)  # [chain]
                        and actions[i] >= self.max_actions):                # [chain]
                    done[i] = True                                          # [chain]
                    capped[i] = rnd                                         # [chain]
                    log.write("chain_events.jsonl", {"round": rnd, "agent": i,  # [chain]
                              "event": "action_cap", "actions": actions[i]})  # [chain]

            events: List[InteractionEvent] = extractor.from_results(rnd, results)
            merged_obs = {**{i: last_obs.get(i) for i in active}, **res_obs}
            events += extractor.deaths(rnd, merged_obs)

            # ── verify / progress / reward ───────────────────────────────────
            team_p: Dict[str, Optional[float]] = {}
            for slot in self.world.teams:
                if stopped[slot.team]:
                    continue
                ok, msg = self.verifiers[slot.team](trajs[slot.team].data)
                team_msg[slot.team] = msg
                team_p[slot.team] = 1.0 if ok else progress_from_msg(msg)
                if ok:
                    solved[slot.team] = 1
                    on_schedule = (rnd >= self.cfg.verify_start
                                   and (rnd - self.cfg.verify_start) % self.cfg.verify_every == 0)
                    if self.cfg.early_stop and on_schedule:
                        stopped[slot.team] = True
            bond, death, parts = self.progress.update(rnd, res_obs, events, team_p, solved)

            triples = hebbian_triples(events, dms)
            if self.graph is not None and getattr(self.graph.config, "enabled", False):
                self.graph.update(positions, chambers=None, bond_rewards=bond,
                                  total_rewards=[b + d for b, d in zip(bond, death)],
                                  social_events=triples, death_rewards=death)

            for e in events:
                log.event(e.to_dict())
            log.state(self._state_record(rnd, round_start, merged_obs, decisions, results,
                                         events, inboxes, bond, parts, team_p, solved))
            if all(stopped.values()) or all(done[i] or stopped[self.world.global_team[i]]
                                            for i in range(n)):
                break
            spare = self.cfg.min_round_s - (time.time() - round_start)
            if spare > 0:
                await asyncio.sleep(spare)

        # Collect stragglers so their records land in the trajectory.
        tail = await self.ex.collect(deadline_s=0)
        for res in tail:
            trajs[self.world.global_team[res.agent]].add_action(
                self.world.global_key[res.agent], res.status, res.call.traj_string(),
                res.observation)
        final: Dict[str, Any] = {"episode": episode, "rounds": rounds_run,
                                 "wall_s": round(time.time() - t0, 1), "teams": {}}
        traj_dir = ep_dir / "trajectories"
        for slot in self.world.teams:
            ok, msg = self.verifiers[slot.team](trajs[slot.team].data)
            trajs[slot.team].save(traj_dir, metrics={"success": ok, "message": msg,
                                                     "rounds": rounds_run})
            final["teams"][slot.team] = {"task": slot.task.task_id, "success": ok,
                                         "message": msg, "progress": progress_from_msg(msg)}
        final["chain"] = {                                                  # [chain]
            "last_active_round": last_active, "max_actions": self.max_actions,
            "hold_complete": self.hold_complete, "board": self.board,
            "actions": {self.names[i]: actions[i] for i in range(n)},
            "held_completes": {self.names[i]: held[i] for i in range(n)},
            "refused_chats": {self.names[i]: refused_chat[i] for i in range(n)},
            "capped": {self.names[i]: r for i, r in capped.items()},
            "completed": {self.names[i]: r for i, r in completed.items()},
            "dms": sum(1 for m in self.router.history if m.kind == "dm"),
            "posts": sum(1 for m in self.router.history if m.kind == "post"),
        }
        log.write("summary.jsonl", final)
        log.close()
        return final


# ── cognition across steps ───────────────────────────────────────────────────
class ChainCognition:
    """Routes each turn to the persistent agent behind the turn's slot.

    ``agents`` maps agent id -> agent (``AgentWorldAgent`` or a test double). At
    every step :meth:`begin_step` gives each agent its new in-world name and
    calls its ``on_reset`` (working memory only; skill and episode memory stay).
    """

    def __init__(self, agents: Dict[str, Any], ledger: Any = None):
        self.agents = dict(agents)
        self.ledger = ledger
        self.order: List[Any] = []

    def begin_step(self, agent_ids: Sequence[str], usernames: Sequence[str]) -> None:
        if len(agent_ids) != len(usernames):
            raise ChainRunError("one username per agent id")
        self.order = []
        for aid, name in zip(agent_ids, usernames):
            agent = self.agents[aid]
            if hasattr(agent, "name"):
                agent.name = name
            if hasattr(agent, "on_reset"):
                agent.on_reset()
            self.order.append(agent)

    async def decide(self, turn: Turn) -> Decision:
        return await self.order[turn.agent].decide(turn)


class PlumbingAgent:
    """No model: a fixed script that touches every path (move, DM, complete, chat)."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = agent_id
        self.rounds = 0

    def on_reset(self) -> None:
        self.rounds = 0

    async def decide(self, turn: Turn) -> Decision:
        self.rounds += 1
        mates = [n for n in turn.names if n != turn.name]
        pos = position(turn.observation)
        if self.rounds == 1 and pos:
            return Decision(f"move_character(x={int(pos[0]) + 1}, y={int(pos[1])})", "plumbing",
                            dm_target=mates[0] if mates else None,
                            dm_text=f"plumbing check from {turn.name}")
        if self.rounds == 2:
            return Decision("complete(response='plumbing')", "plumbing")
        if self.rounds == 3:
            return Decision("chat(message='plumbing')", "plumbing")
        return Decision("wait()", "plumbing")


# ── one step ─────────────────────────────────────────────────────────────────
@dataclass
class StepSpec:
    index: int
    dir_name: str
    task_key: str
    task_yaml: Path
    verifier: Path
    source_verifier: Path
    slots: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_manifest(cls, entry: Dict[str, Any]) -> "StepSpec":
        return cls(index=int(entry["step"]), dir_name=entry["dir"], task_key=entry["task_key"],
                   task_yaml=Path(entry["task_yaml"]), verifier=Path(entry["verifier"]),
                   source_verifier=Path(entry["source_verifier"]), slots=list(entry["slots"]))


def load_step_task(step: StepSpec):
    """The step's renamed task, with the message wording of this harness."""
    from agentworld.tasks import load_task
    task = load_task(step.task_yaml)
    task.objective = point_at_reply_fields(task.objective)
    task.game_context = point_at_reply_fields(task.game_context)
    by_slot = {s["slot"]: s["username"] for s in step.slots}
    for spec in task.agents:
        if by_slot.get(spec.key) != spec.username:
            raise ChainRunError(f"step {step.index}: {spec.key} is {spec.username!r} in "
                                f"{step.task_yaml.name}, the chain says {by_slot.get(spec.key)!r}")
    return task


def slot_permutation(world, step: StepSpec, agent_ids: Sequence[str]) -> Tuple[List[str], List[int]]:
    """(agent id per world index, position of that id in ``agent_ids``)."""
    ids_by_slot = {s["slot"]: s["agent_id"] for s in step.slots}
    ids = [ids_by_slot[world.global_key[i]] for i in range(world.n_agents)]
    if sorted(ids) != sorted(set(ids)) or not set(ids) <= set(agent_ids):
        raise ChainRunError(f"step {step.index}: slots {ids} do not match the chain's agents")
    return ids, [list(agent_ids).index(a) for a in ids]


async def play_step(step: StepSpec, *, agent_ids: Sequence[str], W: np.ndarray,
                    cognition: ChainCognition, make_tools: Callable[[], Any],
                    master_password: str, out_dir: Path, arm: str, seed: int,
                    hebbian: Dict[str, Any], rounds: Optional[int] = None,
                    max_actions: Optional[int] = -1, hold_complete: bool = False,
                    board: bool = False, board_budget: int = 5, inbox_budget: int = 4,
                    contacts_budget: int = 6, explore_slots: int = 2, obs_radius: int = 24,
                    deadline_s: float = 30.0, min_round_s: float = 0.0,
                    setup_concurrency: int = 16, setup_sleep_scale: float = 1.0
                    ) -> Tuple[Dict[str, Any], np.ndarray, List[str]]:
    """Play one step; returns (scheduler summary, W after the step in agent-id order, ids).

    ``max_actions``: -1 = the task's ``max_action_steps``; 0 or None = no cap.
    """
    from agentworld.comms import CommRouter, make_policy
    from agentworld.executor import AgentHandle, Executor, prepare_agent
    from agentworld.progress import ProgressTracker
    from agentworld.scheduler import SchedulerConfig
    from agentworld.tasks import compose_world
    from agentworld.verify import Verifier, extract_targets
    from hebbian.multichannel import MultiChannelHebbianGraph
    from wired_state import graph_config

    if arm not in ARMS:
        raise ChainRunError(f"arm must be one of {ARMS}")
    task = load_step_task(step)
    world = compose_world([task])
    n = world.n_agents
    ids, perm = slot_permutation(world, step, agent_ids)
    init = np.asarray(W, dtype=np.float32)[np.ix_(perm, perm)].copy()
    np.fill_diagonal(init, 0.0)
    graph = MultiChannelHebbianGraph(graph_config(n, init, hebbian))

    gate = asyncio.Semaphore(max(1, setup_concurrency))

    async def setup(i: int, name: str):
        tools = make_tools()
        spec = next(a for a in task.agents if a.key == world.global_key[i])
        async with gate:
            ok = await asyncio.to_thread(prepare_agent, tools, spec, name, master_password,
                                         sleep=lambda s: time.sleep(s * setup_sleep_scale),
                                         location=world.global_spawn[i])
        if not ok:
            raise ChainRunError(f"could not log in {name}")
        return AgentHandle(i, name, world.global_key[i], world.global_team[i], tools)

    handles = list(await asyncio.gather(*(setup(i, nm) for i, nm in enumerate(world.global_names))))
    executor = Executor(handles, deadline_s=deadline_s)
    policy = make_policy("hebbian" if arm == "hebbian" else "recency", seed=seed + step.index)
    router = CommRouter(n, policy, board_budget=board_budget if board else 0,
                        inbox_budget=inbox_budget, contacts_budget=contacts_budget,
                        explore_slots=explore_slots if board else 0, seed=seed + step.index)
    cognition.begin_step(ids, world.global_names)
    team = world.teams[0].team
    verifier = Verifier(step.verifier)
    cap = task.max_action_steps if max_actions == -1 else (max_actions or None)
    sched = ChainRoundScheduler(
        world, executor, router, cognition,
        ProgressTracker(world, {team: extract_targets(step.verifier)}), {team: verifier}, out_dir,
        graph=graph,
        config=SchedulerConfig(max_rounds=rounds, obs_radius=obs_radius, comm_mode="side",
                               bonds_in_prompt=(arm == "hebbian"), bond_source="policy",
                               min_round_s=min_round_s),
        hold_complete=hold_complete, max_actions=cap, board=board)
    try:
        summary = await sched.run_episode(1)
    finally:
        executor.shutdown()
    W_new = np.asarray(W, dtype=np.float32).copy()
    W_new[np.ix_(perm, perm)] = graph.W
    np.fill_diagonal(W_new, 0.0)
    return summary, W_new, ids


def bonds_by_id(W: np.ndarray, agent_ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
    return {a: {b: round(float(W[i, j]), 4) for j, b in enumerate(agent_ids) if j != i}
            for i, a in enumerate(agent_ids)}


def memory_snapshot(agent: Any) -> Dict[str, List[str]]:
    """Skill and episode memory texts of a MindForge agent (empty for test doubles)."""
    out = {}
    for name in ("skills", "episodes"):
        store = getattr(agent, name, None)
        out[name] = list(getattr(store, "texts", []) or [])
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)
