"""VillagerAgent-style centralized DAG orchestration (variant "villager").

Port of VillagerAgent's orchestration core — central state → dynamic task
DAG → deterministic ready-task detection → central LLM task allocation →
parallel workers → feedback → replanning — onto WIRE's MindForge agents.
Faithful boundaries: the controller decides WHO does WHAT (hard assignment,
workers are never asked to accept); the worker decides HOW (auto-curriculum
chooses concrete subtasks constrained by the assignment; the action LLM
picks primitives). NO central communication routing — the published
VillagerAgent does not orchestrate messaging, and neither does this port.

Documented adaptations (user-approved):
- Node feedback is DETERMINISTIC: a task succeeds when any of its listed
  MILESTONE_TRACK ids fires (ground-truth signals the original lacked) and
  fails on timeout or an assigned agent's death — replacing per-worker LLM
  self-reflection.
- The StateManager is a deterministic snapshot (collect_env_state + task
  table + milestone catalog), replacing recursive LLM state summaries.
- "Parallel workers" needs no thread pool here: WIRE agents already act
  every env step; assignments are standing constraints, and the env loop is
  the scheduler.
- No cross-episode memory (fresh DAG each episode, like the paper). Not
  checkpointed: on a mid-episode resume the first tick replans from the
  restored milestone sets and curriculum histories.

Heavy imports (craftium_metric pulls matplotlib; chamber_facts is light but
kept symmetric) are lazy, so this module stays light-importable for tests.
"""

from __future__ import annotations

import logging as _stdlog
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

from orchestrator import prompt as _prompt
from orchestrator.config import OrchestratorConfig
from orchestrator.core import (
    _normalize_agent,
    collect_env_state,
    collect_task_table,
)
from orchestrator.dag import TaskDAG, ingest_decomposition
from orchestrator.state import OrchestratorState

logger = _stdlog.getLogger(__name__)

#: Milestone tracks that represent game progress; the comm/obs/imit reward
#: milestones are shaping chatter, not objectives a task can target.
GAME_TRACKS = ("ch1_solo", "ch2_anvils", "ch3_switches", "ch4_combat",
               "ch5_boss")


class VillagerDecomposeResponse(BaseModel):
    """Schema handed to the shared client factory for the decomposer."""
    tasks: list
    why: str = ""


class VillagerAllocateResponse(BaseModel):
    """Schema handed to the shared client factory for the allocator."""
    assignments: list
    why: str = ""


# ── Deterministic StateManager pieces ────────────────────────────────────

def known_milestone_ids(milestone_track: Optional[dict] = None) -> set:
    """Game-progress milestone ids (the valid completion signals)."""
    if milestone_track is None:
        from agent_modules.craftium_metric import MILESTONE_TRACK
        milestone_track = MILESTONE_TRACK
    return {mid for mid, track in milestone_track.items()
            if track in GAME_TRACKS}


def build_milestone_catalog(agent_milestones: dict,
                            milestone_track: Optional[dict] = None,
                            tracks: Optional[dict] = None) -> str:
    """Per-track catalog lines with rewards and completion state.

    ``agent_milestones`` is metric._agent_milestones ("agent_N" -> set of
    fired ids). NOTE: completion is the coarse team-level union — a few
    per-agent milestones (e.g. m14_sword_equipped) can validly re-fire for
    another agent but are shown completed; the timeout/re-decompose loop
    absorbs that (documented trade-off)."""
    if milestone_track is None or tracks is None:
        from agent_modules.craftium_metric import MILESTONE_TRACK, TRACKS
        milestone_track = milestone_track or MILESTONE_TRACK
        tracks = tracks or TRACKS
    fired_by: dict = {}
    for agent, ids in (agent_milestones or {}).items():
        for mid in ids:
            fired_by.setdefault(mid, []).append(str(agent))
    lines = []
    for track in GAME_TRACKS:
        for mid, reward in tracks.get(track, []):
            who = sorted(fired_by.get(mid, []))
            state = f"completed by {', '.join(who)}" if who else "OPEN"
            lines.append(f"  {mid} ({track}, +{reward:g}): {state}")
    return "\n".join(lines) or "  (no milestone catalog available)"


def build_chamber_facts(num_agents: int, describe=None) -> str:
    if describe is None:
        from agent_modules.chamber_facts import describe_chamber as describe
    parts = []
    for ch in ("ch1", "ch2", "ch3", "ch3_communal", "ch4", "ch5"):
        try:
            parts.append(describe(ch, num_agents))
        except Exception:
            continue
    return "\n".join(parts) or "(no chamber facts available)"


def build_ready_tasks_block(dag: TaskDAG) -> str:
    ready = dag.ready_tasks()
    if not ready:
        return "  (none)"
    lines = []
    for t in ready:
        cands = ", ".join(t.candidates) if t.candidates else "any agent"
        lines.append(
            f"  {t.id}: \"{t.description}\" | done when any of "
            f"[{', '.join(t.milestones)}] fires | needs {t.min_agents} "
            f"agent(s) | candidates: {cands}"
        )
    return "\n".join(lines)


def build_free_agents_block(env_state: dict, task_table: str,
                            free_agents: list) -> str:
    """Per free agent: live state + its recent task record (the paper's
    'agent experience' input, sourced from the auto-curriculum lists)."""
    table_lines = {}
    for line in (task_table or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("agent_"):
            table_lines[stripped.split(":", 1)[0]] = stripped
    agents_state = (env_state or {}).get("agents") or {}
    lines = []
    for name in free_agents:
        info = agents_state.get(name) or {}
        hp = info.get("hp")
        state = (f"chamber={info.get('chamber') or '?'}"
                 + (f", hp={hp:.0f}" if hp is not None else ""))
        record = table_lines.get(name, f"{name}: (no task record)")
        lines.append(f"  {name}: {state}\n    {record}")
    return "\n".join(lines) or "  (none)"


# ── Coupling surfaces ────────────────────────────────────────────────────

_UNASSIGNED_DIRECTIVE = (
    "Team assignment (from the non-embodied coordinator): (none right now — "
    "the coordinator will assign you shortly; keep making safe progress in "
    "your current chamber.)"
)


def render_villager_directive(agent_name: str, dag: TaskDAG) -> str:
    """Text for the {social_directive} action-prompt slot."""
    name = _normalize_agent(agent_name) or agent_name
    task = dag.assignment_of(name)
    if task is None:
        return _UNASSIGNED_DIRECTIVE
    teammates = [a for a in task.assigned if a != name]
    role = task.roles.get(name) or "work on this objective"
    return (
        "Team assignment (from the non-embodied coordinator; it sees the "
        "whole team):\n"
        f"  Objective: {task.description}\n"
        f"  Your role: {role}\n"
        f"  Teammates on this task: "
        f"{', '.join(teammates) if teammates else '(you alone)'}\n"
        f"  Done when milestone(s) [{', '.join(task.milestones)}] fire. "
        f"Your actions this step MUST advance this objective."
    )


def assigned_objective_note(agent_name: str, dag: TaskDAG) -> str:
    """Text for the curriculum's {assigned_objective} placeholder."""
    name = _normalize_agent(agent_name) or agent_name
    task = dag.assignment_of(name)
    if task is None:
        return "(none — use your own judgment this round)"
    role = task.roles.get(name)
    return task.description + (f" (your role: {role})" if role else "")


# ── Allocation validation ────────────────────────────────────────────────

def validate_allocation(parsed: dict, *, dag: TaskDAG, ready_ids: set,
                        free_agents: list, living_agents: list) -> dict:
    """Salvage-per-entry validation of one allocator response.

    Strippable problems (busy/dead/unknown/non-candidate/double-booked
    agents, unknown tasks, under-crewed multi-agent groups) drop the entry
    or group with a warning; the response only FAILS (retry trigger) when
    it is structurally unusable or nothing survives while work and workers
    both exist."""
    result = {"ok": False, "error": None, "assignments": {}, "warnings": []}
    if not isinstance(parsed, dict):
        result["error"] = "response is not a JSON object"
        return result
    raw = parsed.get("assignments")
    if not isinstance(raw, list):
        result["error"] = "missing/invalid 'assignments' list"
        return result

    living = [_normalize_agent(a) or a for a in living_agents]
    free = {_normalize_agent(a) or a for a in free_agents}
    taken: set = set()
    groups: dict = {}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            result["warnings"].append(f"entry {i}: not an object — dropped")
            continue
        agent = _normalize_agent(entry.get("agent")
                                 or entry.get("agent_name"))
        tid_raw = entry.get("task_id") or entry.get("task")
        tid = str(tid_raw).strip() if tid_raw is not None else ""
        if agent is None or agent not in living:
            result["warnings"].append(
                f"entry {i}: unknown/dead agent {entry.get('agent')!r} "
                f"— dropped")
            continue
        if agent not in free:
            result["warnings"].append(f"entry {i}: {agent} is busy — dropped")
            continue
        if agent in taken:
            result["warnings"].append(
                f"entry {i}: {agent} already assigned this round "
                f"(first wins) — dropped")
            continue
        if tid not in ready_ids:
            result["warnings"].append(
                f"entry {i}: task {tid!r} is not a ready task — dropped")
            continue
        task = dag.get(tid)
        if task is not None and task.candidates and agent not in task.candidates:
            result["warnings"].append(
                f"entry {i}: {agent} is not a candidate for {tid} — dropped")
            continue
        taken.add(agent)
        group = groups.setdefault(tid, {"agents": [], "roles": {}})
        group["agents"].append(agent)
        group["roles"][agent] = str(entry.get("role") or "")

    # Multi-agent tasks start fully crewed or not at all.
    for tid in list(groups.keys()):
        task = dag.get(tid)
        need = task.min_agents if task is not None else 1
        if len(groups[tid]["agents"]) < need:
            result["warnings"].append(
                f"task {tid}: only {len(groups[tid]['agents'])} of "
                f"{need} required agents assigned — group dropped")
            del groups[tid]

    if not groups and ready_ids and free:
        result["error"] = "no valid assignments survived"
        return result
    result["ok"] = True
    result["assignments"] = groups
    return result


# ── The controller ───────────────────────────────────────────────────────

@dataclass
class TickResult:
    reassigned: list = field(default_factory=list)  # agents whose task CHANGED
    decomposed: bool = False
    allocated: bool = False
    completed: list = field(default_factory=list)   # task ids
    failed: list = field(default_factory=list)      # (task_id, reason)


class VillagerController:
    def __init__(self, cfg: OrchestratorConfig, num_agents: int,
                 decompose_client, allocate_client, orch_logger=None,
                 milestone_track: Optional[dict] = None,
                 tracks: Optional[dict] = None,
                 chamber_describe=None):
        self.cfg = cfg
        self.num_agents = num_agents
        self.decompose_client = decompose_client
        self.allocate_client = allocate_client
        self.orch_logger = orch_logger
        self.max_open = cfg.max_open_tasks or 2 * num_agents
        # Test-injection overrides; None (runtime) → lazy imports of the
        # real MILESTONE_TRACK / TRACKS / describe_chamber.
        self._milestone_track = milestone_track
        self._tracks = tracks
        self._chamber_describe = chamber_describe
        self.reset()

    def reset(self) -> None:
        """Episode start: fresh DAG + counters (no cross-episode memory)."""
        self.dag = TaskDAG()
        self.last_decompose_step: Optional[int] = None
        self.alloc_cooldown_from: Optional[int] = None
        self._changes_since_decompose = 0
        self.failed_calls = 0

    # ── Coupling surface delegates ───────────────────────────────────────

    def directive_text(self, agent_name: str) -> str:
        return render_villager_directive(agent_name, self.dag)

    def assigned_objective(self, agent_name: str) -> str:
        return assigned_objective_note(agent_name, self.dag)

    # ── Internals ────────────────────────────────────────────────────────

    def _log_call(self, record: dict) -> None:
        if self.orch_logger is not None:
            self.orch_logger.log_call(record)

    def _log_freed(self, episode: int, t: int, task, freed: list,
                   reason: str) -> None:
        if self.orch_logger is None:
            return
        for agent in freed:
            self.orch_logger.log_assignment({
                "episode": episode, "t": t, "agent": agent,
                "task_id": task.id, "description": task.description,
                "reason": reason,
            })

    async def _call_llm(self, client, filled: str, parse_json) -> dict:
        """One attempt against a client; mirrors orchestrate()'s call shape.
        Returns {"parsed", "prompt_tokens", "completion_tokens", "raw"}."""
        from autogen_core import CancellationToken
        from autogen_core.models import UserMessage

        out = {"parsed": None, "prompt_tokens": 0, "completion_tokens": 0,
               "raw": None}
        try:
            response = await client.create(
                [UserMessage(content=[filled], source="user")],
                cancellation_token=CancellationToken(),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            logger.error("Villager: LLM call failed: %s", str(exc)[:300])
            return out
        usage = getattr(response, "usage", None)
        if usage is not None:
            out["prompt_tokens"] = int(getattr(usage, "prompt_tokens", 0) or 0)
            out["completion_tokens"] = int(
                getattr(usage, "completion_tokens", 0) or 0)
        raw = response.content if isinstance(response.content, str) \
            else str(response.content)
        out["raw"] = raw
        try:
            out["parsed"] = parse_json(raw)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Villager: JSON parse raised: %s", exc)
        return out

    # ── The per-step tick ────────────────────────────────────────────────

    async def tick(self, *, state: OrchestratorState, living_agents: list,
                   episode: int, t: int,
                   environment=None, agents=None, metric=None,
                   env_state: Optional[dict] = None,
                   task_table: Optional[str] = None,
                   agent_milestones: Optional[dict] = None,
                   parse_json=None) -> TickResult:
        """One deterministic scheduling pass; LLM calls only when needed.

        ``env_state`` / ``task_table`` / ``agent_milestones`` are
        test-injection overrides; when None they are collected from the
        live ``environment`` / ``agents`` / ``metric`` objects.
        """
        if parse_json is None:
            from orchestrator.core import _default_parse_json
            parse_json = _default_parse_json()
        result = TickResult()
        living = [_normalize_agent(a) or a for a in living_agents]

        # 1. Drain events (load-bearing: villager never calls apply_success,
        #    so nothing else bounds the buffer). Success before timeouts so
        #    a same-tick tie goes to success.
        events = list(state.event_buffer)
        state.event_buffer = []
        for ev in events:
            kind = ev.get("type")
            if kind == "milestone":
                for task in self.dag.tasks_watching(ev.get("id", "")):
                    freed = self.dag.mark_success(task.id, t)
                    result.completed.append(task.id)
                    self._changes_since_decompose += 1
                    self._log_freed(episode, t, task, freed, "freed_success")
            elif kind == "death":
                dead = _normalize_agent(ev.get("agent")) or ev.get("agent")
                task = self.dag.assignment_of(dead)
                if task is not None:
                    freed = self.dag.mark_failure(task.id, "death", t)
                    result.failed.append((task.id, "death"))
                    self._changes_since_decompose += 1
                    self._log_freed(episode, t, task, freed, "freed_death")

        # 2. Timeouts.
        for task in self.dag.timed_out(t, self.cfg.node_timeout_steps):
            freed = self.dag.mark_failure(task.id, "timeout", t)
            result.failed.append((task.id, "timeout"))
            self._changes_since_decompose += 1
            self._log_freed(episode, t, task, freed, "freed_timeout")

        # 3. Team wipe: fail running work, no LLM calls.
        if not living:
            for task in self.dag.running_tasks():
                self.dag.mark_failure(task.id, "death", t)
                result.failed.append((task.id, "death"))
            self._sync_state_mirror(state)
            self._maybe_log_dag(episode, t, "team_wipe",
                                changed=bool(result.failed))
            return result

        free = [a for a in living if self.dag.assignment_of(a) is None]

        # 4. Decompose when due.
        interval_ok = (
            self.last_decompose_step is None
            or t - self.last_decompose_step >= self.cfg.decompose_min_interval
        )
        needs_plan = (
            self.last_decompose_step is None
            or self.dag.is_exhausted()
            or self._changes_since_decompose > 0
            or (not self.dag.ready_tasks() and bool(free))
        )
        if interval_ok and needs_plan \
                and self.dag.open_count() < self.max_open:
            await self._decompose(state, living, episode, t,
                                  environment=environment, agents=agents,
                                  metric=metric, env_state=env_state,
                                  task_table=task_table,
                                  agent_milestones=agent_milestones,
                                  parse_json=parse_json, result=result)

        # 5. Allocate when there is work and workers.
        ready = self.dag.ready_tasks()
        alloc_cooldown = (
            self.alloc_cooldown_from is not None
            and t - self.alloc_cooldown_from < self.cfg.decompose_min_interval
        )
        if ready and free and not alloc_cooldown:
            await self._allocate(living, free, episode, t,
                                 environment=environment, agents=agents,
                                 env_state=env_state, task_table=task_table,
                                 parse_json=parse_json, result=result)

        # 6-7. Mirror + snapshot.
        self._sync_state_mirror(state)
        self._maybe_log_dag(
            episode, t, "tick",
            changed=bool(result.completed or result.failed
                         or result.decomposed or result.allocated))
        return result

    async def _decompose(self, state, living, episode, t, *, environment,
                         agents, metric, env_state, task_table,
                         agent_milestones, parse_json, result) -> None:
        if env_state is None:
            env_state = collect_env_state(environment, self.num_agents, t)
        if task_table is None:
            task_table = collect_task_table(agents, self.num_agents)
        if agent_milestones is None:
            agent_milestones = (dict(getattr(metric, "_agent_milestones", {}))
                                if metric is not None else {})
        from orchestrator import map_render as _map_render
        env_text = _map_render.render_map_text(env_state,
                                               num_agents=self.num_agents)
        known = known_milestone_ids(self._milestone_track)
        team_completed = set()
        for ids in (agent_milestones or {}).values():
            team_completed.update(ids)
        open_slots = max(1, self.max_open - self.dag.open_count())
        example_ms = sorted(known - team_completed) or sorted(known)
        filled = _prompt.format_decompose_prompt(
            n_agents=len(living),
            agent_names=living,
            current_step=t,
            chamber_facts=build_chamber_facts(
                self.num_agents, describe=self._chamber_describe),
            milestone_catalog=build_milestone_catalog(
                agent_milestones, self._milestone_track, self._tracks),
            env_state_text=env_text,
            task_table=task_table,
            dag_summary=self.dag.summary_text(),
            open_slots=open_slots,
            example_milestones=example_ms[:2],
        )
        prompt_tokens = completion_tokens = 0
        raw_tail = None
        ingest = None
        for attempt in range(2):   # initial + max 1 retry
            call = await self._call_llm(self.decompose_client, filled,
                                        parse_json)
            prompt_tokens += call["prompt_tokens"]
            completion_tokens += call["completion_tokens"]
            if call["raw"] is not None:
                raw_tail = call["raw"][:6000]
            ingest = ingest_decomposition(
                self.dag, call["parsed"], t=t, known_milestones=known,
                team_completed=team_completed, living_agents=living,
                max_open=self.max_open,
            )
            if ingest["ok"]:
                break
            logger.warning("Villager decompose failed validation "
                           "(attempt %d): %s", attempt + 1, ingest["error"])
        # Cooldown either way — a failing decomposer must not be hammered
        # every step (the record_failure clock-advance lesson).
        self.last_decompose_step = t
        record = {
            "episode": episode, "t": t, "call_type": "decompose",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if ingest is not None and ingest["ok"]:
            self._changes_since_decompose = 0
            result.decomposed = True
            record.update({
                "failed": False,
                "accepted": ingest["accepted"],
                "rejected": ingest["rejected"],
                "warnings": ingest["warnings"],
            })
        else:
            self.failed_calls += 1
            record.update({
                "failed": True,
                "why": (ingest["error"] if ingest is not None
                        else "LLM call failed"),
                "raw_output": raw_tail,
            })
            logger.error("Villager: decompose kept DAG unchanged "
                         "(failed call #%d)", self.failed_calls)
        self._log_call(record)

    async def _allocate(self, living, free, episode, t, *, environment,
                        agents, env_state, task_table, parse_json,
                        result) -> None:
        if env_state is None:
            env_state = collect_env_state(environment, self.num_agents, t)
        if task_table is None:
            task_table = collect_task_table(agents, self.num_agents)
        ready = self.dag.ready_tasks()
        ready_ids = {task.id for task in ready}
        filled = _prompt.format_allocate_prompt(
            current_step=t,
            ready_tasks_block=build_ready_tasks_block(self.dag),
            free_agents_block=build_free_agents_block(env_state, task_table,
                                                      free),
            dag_summary=self.dag.summary_text(),
            free_agents=free,
            ready_task_ids=sorted(ready_ids),
        )
        prompt_tokens = completion_tokens = 0
        raw_tail = None
        verdict = None
        for attempt in range(2):
            call = await self._call_llm(self.allocate_client, filled,
                                        parse_json)
            prompt_tokens += call["prompt_tokens"]
            completion_tokens += call["completion_tokens"]
            if call["raw"] is not None:
                raw_tail = call["raw"][:6000]
            verdict = validate_allocation(
                call["parsed"], dag=self.dag, ready_ids=ready_ids,
                free_agents=free, living_agents=living,
            )
            if verdict["ok"]:
                break
            logger.warning("Villager allocate failed validation "
                           "(attempt %d): %s", attempt + 1, verdict["error"])
        record = {
            "episode": episode, "t": t, "call_type": "allocate",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if verdict is not None and verdict["ok"]:
            result.allocated = bool(verdict["assignments"])
            for tid, group in verdict["assignments"].items():
                self.dag.assign(tid, group["agents"], group["roles"], t)
                task = self.dag.get(tid)
                for agent in group["agents"]:
                    result.reassigned.append(agent)
                    if self.orch_logger is not None:
                        self.orch_logger.log_assignment({
                            "episode": episode, "t": t, "agent": agent,
                            "task_id": tid,
                            "description": task.description,
                            "role": group["roles"].get(agent, ""),
                            "reason": "allocate",
                        })
            record.update({
                "failed": False,
                "assignments": {tid: g["agents"]
                                for tid, g in verdict["assignments"].items()},
                "warnings": verdict["warnings"],
            })
        else:
            self.failed_calls += 1
            self.alloc_cooldown_from = t
            record.update({
                "failed": True,
                "why": (verdict["error"] if verdict is not None
                        else "LLM call failed"),
                "raw_output": raw_tail,
            })
            logger.error("Villager: allocation failed — agents stay free "
                         "until cooldown expires (failed call #%d)",
                         self.failed_calls)
        self._log_call(record)

    def _sync_state_mirror(self, state: OrchestratorState) -> None:
        """Mirror current assignments into state.directives (the DAG stays
        authoritative; the mirror keeps debugging uniform across variants)."""
        mirror = {}
        for task in self.dag.running_tasks():
            for agent in task.assigned:
                mirror[agent] = {
                    "task_id": task.id,
                    "description": task.description,
                    "role": task.roles.get(agent, ""),
                }
        state.directives = mirror

    def _maybe_log_dag(self, episode: int, t: int, trigger: str,
                       changed: bool) -> None:
        if changed and self.orch_logger is not None:
            self.orch_logger.log_dag({
                "episode": episode, "t": t, "trigger": trigger,
                **self.dag.to_dict(),
            })
