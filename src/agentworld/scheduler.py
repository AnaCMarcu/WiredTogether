"""The round loop: observe → route → decide (all agents at once) → act → learn.

One round, for every agent that is neither busy with a long action nor done:

1. decision-time observation (radius ``obs_radius``), in parallel;
2. :class:`CommRouter` delivers ≤K DMs, ≤B board posts and ≤C contacts,
   gated by the read policy (Hebbian bonds in the Hebbian arm);
3. cognition decides for all agents concurrently (one batched LLM wave);
4. DMs and board posts are queued (read next round) and recorded;
5. all tool calls run concurrently (:class:`Executor`, deadline-capped);
6. events, verifier progress and the dense reward update the Hebbian graph;
7. trajectories (AgentWorld schema) and the replay state are written.

Cognition is any object with ``async decide(turn) -> Decision``; the MindForge
agent lives in :mod:`agentworld.agent`, scripted policies in tests and
:mod:`agentworld.generate.oracle_policy`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from agentworld.actions import ToolCall, parse_action, safe_text
from agentworld.comms import CommRouter, Inbox
from agentworld.events import EventExtractor, InteractionEvent, hebbian_triples
from agentworld.executor import ActionResult, Executor
from agentworld.observation import hp, inventory_counts, position, render_obs_text
from agentworld.progress import ProgressTracker
from agentworld.tasks import WorldSpec
from agentworld.trajectory import RunLog, TeamTrajectory, status_line
from agentworld.verify import Verifier, progress_from_msg


@dataclass
class Decision:
    action: str = "wait()"
    thoughts: str = ""
    dm_target: Optional[str] = None     # username, "agent_7"-style index, or None
    dm_text: str = ""
    board_post: str = ""
    post_kind: str = "status"


@dataclass
class Turn:
    round: int
    agent: int
    name: str
    team: str
    objective: str
    game_context: str
    observation: Optional[Dict[str, Any]]
    obs_text: str
    last_result: str
    inbox: Inbox
    contacts: List[str]
    bonds: List[Tuple[str, float]]       # top-K (name, W) — empty in the base arm
    progress_text: str
    names: List[str] = field(repr=False, default_factory=list)


class Cognition(Protocol):
    async def decide(self, turn: Turn) -> Decision: ...


@dataclass
class SchedulerConfig:
    max_rounds: Optional[int] = None    # default: the world's task rounds
    obs_radius: int = 24
    obs_k: int = 6
    comm_mode: str = "side"             # "side" | "tool" (anchor parity)
    bonds_in_prompt: bool = True
    bond_source: str = "policy"         # "policy": the read policy's bond (W, permuted W,
                                        # or true team) | "graph": raw W (prompt_only arm)
    bond_k: int = 6
    mirror_dms: bool = False            # re-post DMs/board as local game chat (live video)
    verify_start: int = 5               # AgentWorld early-stop schedule
    verify_every: int = 3
    early_stop: bool = True
    # The game runs in real time (trees regrow, mobs respawn on timers) and our
    # simultaneous rounds are fast (~6 s at N=3 on vLLM), so a round covers far
    # less game time than in AgentWorld's serial runner, where every agent's
    # turn waits for the previous one. A minimum round length restores that.
    min_round_s: float = 0.0


class RoundScheduler:
    def __init__(self, world: WorldSpec, executor: Executor, router: CommRouter,
                 cognition: Cognition, progress: ProgressTracker,
                 verifiers: Dict[str, Verifier], out_dir: Path,
                 graph: Any = None, config: Optional[SchedulerConfig] = None,
                 hide_players: Optional[set] = None, mirror: Any = None):
        self.world = world
        self.ex = executor
        self.router = router
        self.cog = cognition
        self.progress = progress
        self.verifiers = verifiers
        self.out_dir = Path(out_dir)
        self.graph = graph
        self.cfg = config or SchedulerConfig()
        self.hide = hide_players or set()
        self.names = world.global_names
        self.index = {n: i for i, n in enumerate(self.names)}
        # mirror(sender_index, text): re-posts a DM / board post as the
        # sender's LOCAL game chat so the live game client draws a native
        # speech bubble. Agents never read game chat in comm_mode "side".
        self.mirror = mirror if (self.cfg.mirror_dms and self.cfg.comm_mode == "side") else None

    # ── helpers ────────────────────────────────────────────────────────────
    def _W(self) -> Optional[np.ndarray]:
        if self.graph is None or not getattr(self.graph.config, "enabled", False):
            return None
        return self.graph.W

    def _resolve(self, target: Optional[str], sender: int) -> Optional[int]:
        if target is None:
            return None
        t = str(target).strip().lstrip("@")
        if t in self.index:
            return self.index[t]
        low = t.lower()
        exact = [i for n, i in self.index.items() if n.lower() == low]
        if exact:
            return exact[0]
        # A unique suffix ("hunter_agent" for "t02_hunter_agent"), preferring
        # the sender's own team when replicas share role names.
        hits = [i for n, i in self.index.items() if n.lower().endswith(low) and i != sender]
        team = self.world.global_team[sender]
        same = [i for i in hits if self.world.global_team[i] == team]
        if len(same) == 1:
            return same[0]
        return hits[0] if len(hits) == 1 else None

    def _bond_value(self, i: int, j: int) -> Optional[float]:
        if self.cfg.bond_source == "graph":
            W = self._W()
            return None if W is None else float(W[i, j])
        if self.router.policy.name == "recency":
            return None
        return float(self.router.policy.bond(i, j))

    def _bonds(self, i: int, contacts: List[int]) -> List[Tuple[str, float]]:
        """Top-K bonds shown in the prompt — the SAME values the read gating uses,
        so the shuffled control cannot leak true identities through text."""
        if not self.cfg.bonds_in_prompt:
            return []
        vals = [(j, self._bond_value(i, j)) for j in contacts]
        vals = [(j, v) for j, v in vals if v is not None]
        vals.sort(key=lambda jv: -jv[1])
        return [(self.names[j], round(v, 2)) for j, v in vals[: self.cfg.bond_k]]

    # ── episode ────────────────────────────────────────────────────────────
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
            for i, d in decisions.items():
                key = self.world.global_key[i]
                tr = trajs[self.world.global_team[i]]
                obs_i, status_i = last_obs.get(i), status_line(last_obs.get(i))
                call = parse_action(d.action)
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
            results: List[ActionResult] = await self.ex.act_all(calls, rnd)
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
        log.write("summary.jsonl", final)
        log.close()
        return final

    def _state_record(self, rnd, round_start, obs, decisions, results, events, inboxes,
                      bond, parts, team_p, solved) -> Dict[str, Any]:
        n = self.world.n_agents
        agents = []
        for i in range(n):
            o = obs.get(i)
            pos = position(o)
            agents.append({
                "i": i, "name": self.names[i], "team": self.world.global_team[i],
                "x": pos[0] if pos else None, "y": pos[1] if pos else None,
                "hp": hp(o), "maxhp": (o or {}).get("playerStatus", {}).get("maxHitPoints") if o else None,
                "inv": inventory_counts(o),
                "busy": self.ex.busy(i),
                "reward": round(bond[i], 3), "reward_parts": parts.get(i, {}),
                "thoughts": safe_text(decisions[i].thoughts, 300) if i in decisions else "",
            })
        mobs: Dict[str, Dict[str, Any]] = {}
        for o in obs.values():
            for m in (o or {}).get("mobs") or []:
                inst = m.get("instance")
                if inst and inst not in mobs:
                    mobs[inst] = {k: m.get(k) for k in ("instance", "name", "x", "y",
                                                        "hitPoints", "maxHitPoints")}
        reads = {i: {"dms": [m.msg_id for m in ib.dms], "board": [m.msg_id for m in ib.board],
                     "contacts": ib.contacts}
                 for i, ib in inboxes.items()}
        W = self._W()
        top = None
        if W is not None:
            k = min(4, max(0, n - 1))
            top = {i: [[int(j), round(float(W[i, j]), 3)]
                       for j in np.argsort(-W[i])[:k] if j != i] for i in range(n)}
        return {
            "round": rnd, "t_wall": round_start,
            "agents": agents,
            "actions": [{"i": r.agent, "call": r.call.traj_string(), "ok": r.success,
                         "issued": r.round_issued, "text": safe_text(r.text, 240)}
                        for r in results],
            "events": [e.to_dict() for e in events],
            "messages": [asdict(m) for m in self.router.history if m.round == rnd],
            "reads": reads,
            "mobs": list(mobs.values()),
            "teams": {t: {"progress": team_p.get(t), "solved": solved.get(t, 0)}
                      for t in solved},
            "bonds_top": top,
        }
