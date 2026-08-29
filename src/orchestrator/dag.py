"""Task DAG for the VillagerAgent-style orchestrator variant.

The graph represents SUBTASK DEPENDENCIES, not agent-agent relationships
(the deliberate contrast with the Hebbian W). Each node is a CentralTask;
an edge P -> T (stored as ``T.required = [P.id, ...]``) means P must succeed
before T becomes executable. Ready-task detection, status transitions,
timeouts and failure cascades are all deterministic — the LLM proposes
tasks and assignments, but never mutates the graph directly.

Pure stdlib, unit-testable without the runtime stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

VALID_STATUSES = ("open", "running", "success", "failure")

_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]")
_DESCRIPTION_MAX_CHARS = 200
_ROLE_MAX_CHARS = 160


@dataclass
class CentralTask:
    id: str
    description: str
    # ≥1 verifiable MILESTONE_TRACK ids; ANY of them firing completes the
    # task (the decomposer is told to split multi-signal objectives).
    milestones: list = field(default_factory=list)
    required: list = field(default_factory=list)    # predecessor task ids
    candidates: list = field(default_factory=list)  # [] = any living agent
    min_agents: int = 1
    assigned: list = field(default_factory=list)    # agent names while running
    roles: dict = field(default_factory=dict)       # agent -> one-line role
    status: str = "open"
    failure_reason: str = ""       # "timeout" | "death" | "predecessor_failed"
    created_at_step: int = -1
    assigned_at_step: int = -1
    finished_at_step: int = -1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "milestones": list(self.milestones),
            "required": list(self.required),
            "candidates": list(self.candidates),
            "min_agents": self.min_agents,
            "assigned": list(self.assigned),
            "roles": dict(self.roles),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "created_at_step": self.created_at_step,
            "assigned_at_step": self.assigned_at_step,
            "finished_at_step": self.finished_at_step,
        }


class TaskDAG:
    def __init__(self) -> None:
        self.tasks: dict = {}   # id -> CentralTask, insertion-ordered

    def get(self, task_id: str):
        return self.tasks.get(task_id)

    # ── Structure ────────────────────────────────────────────────────────

    def would_cycle(self, task_id: str, required: list) -> bool:
        """True if adding ``task_id`` with predecessor edges ``required``
        would create a cycle (including the trivial self-edge)."""
        if task_id in required:
            return True
        # A cycle needs task_id reachable from a predecessor via existing
        # required-chains.
        stack = [r for r in required if r in self.tasks]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur == task_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            node = self.tasks.get(cur)
            if node is not None:
                stack.extend(node.required)
        return False

    def add_task(self, task: CentralTask):
        """Add a node. Returns (ok, reason)."""
        if task.id in self.tasks:
            return False, "duplicate id"
        if task.status not in VALID_STATUSES:
            return False, f"invalid status {task.status!r}"
        if self.would_cycle(task.id, task.required):
            return False, "cycle"
        self.tasks[task.id] = task
        return True, ""

    # ── Queries ──────────────────────────────────────────────────────────

    def ready_tasks(self) -> list:
        """Open tasks whose every (known) predecessor has succeeded."""
        out = []
        for t in self.tasks.values():
            if t.status != "open":
                continue
            if all(
                self.tasks[r].status == "success"
                for r in t.required if r in self.tasks
            ):
                out.append(t)
        return out

    def running_tasks(self) -> list:
        return [t for t in self.tasks.values() if t.status == "running"]

    def open_count(self) -> int:
        """Open + running nodes (feeds the max-open cap)."""
        return sum(1 for t in self.tasks.values()
                   if t.status in ("open", "running"))

    def busy_agents(self) -> set:
        busy = set()
        for t in self.running_tasks():
            busy.update(t.assigned)
        return busy

    def assignment_of(self, agent_name: str):
        for t in self.running_tasks():
            if agent_name in t.assigned:
                return t
        return None

    def tasks_watching(self, milestone_id: str) -> list:
        return [t for t in self.running_tasks() if milestone_id in t.milestones]

    def timed_out(self, t: int, timeout_steps: int) -> list:
        return [task for task in self.running_tasks()
                if task.assigned_at_step >= 0
                and t - task.assigned_at_step >= timeout_steps]

    def is_exhausted(self) -> bool:
        """No open, ready or running work remains."""
        return all(t.status in ("success", "failure")
                   for t in self.tasks.values())

    # ── Transitions ──────────────────────────────────────────────────────

    def assign(self, task_id: str, agent_names: list, roles: dict,
               t: int) -> None:
        task = self.tasks[task_id]
        task.assigned = list(agent_names)
        task.roles = {a: str(roles.get(a) or "")[:_ROLE_MAX_CHARS]
                      for a in agent_names}
        task.status = "running"
        task.assigned_at_step = t

    def mark_success(self, task_id: str, t: int) -> list:
        task = self.tasks[task_id]
        freed = list(task.assigned)
        task.status = "success"
        task.finished_at_step = t
        task.assigned = []
        return freed

    def mark_failure(self, task_id: str, reason: str, t: int) -> list:
        """Fail a node and cascade ``predecessor_failed`` to every OPEN
        descendant (running descendants cannot exist: a task only runs once
        all its predecessors succeeded). Returns the freed agents of the
        primary failure."""
        task = self.tasks[task_id]
        freed = list(task.assigned)
        task.status = "failure"
        task.failure_reason = reason
        task.finished_at_step = t
        task.assigned = []
        # Transitive open descendants of the failed node.
        changed = True
        failed_ids = {task_id}
        while changed:
            changed = False
            for other in self.tasks.values():
                if other.status != "open":
                    continue
                if any(r in failed_ids for r in other.required):
                    other.status = "failure"
                    other.failure_reason = "predecessor_failed"
                    other.finished_at_step = t
                    failed_ids.add(other.id)
                    changed = True
        return freed

    # ── Rendering / serialization ────────────────────────────────────────

    def summary_text(self) -> str:
        if not self.tasks:
            return "(no tasks yet — the graph is empty)"
        lines = []
        for t in self.tasks.values():
            extra = ""
            if t.required:
                extra += f" | requires: {', '.join(t.required)}"
            if t.status == "running":
                extra += f" | assigned: {', '.join(t.assigned)}"
            if t.status == "failure":
                extra += f" | failed: {t.failure_reason}"
            lines.append(
                f"  {t.id} [{t.status}] \"{t.description}\" "
                f"(done when any of [{', '.join(t.milestones)}] fires)"
                f"{extra}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"tasks": [t.to_dict() for t in self.tasks.values()]}


# ── Decomposer-output ingestion ──────────────────────────────────────────

def _sanitize_id(raw, fallback: str) -> str:
    s = _ID_SANITIZE_RE.sub("", str(raw or ""))[:40]
    return s or fallback


def ingest_decomposition(dag: TaskDAG, parsed: dict, *, t: int,
                         known_milestones: set, team_completed: set,
                         living_agents: list, max_open: int,
                         unreachable: Optional[set] = None) -> dict:
    """Deterministically fold one decomposer response into the DAG.

    ``ok=False`` (the retry trigger) ONLY for a structurally unusable
    response — not a dict, or no non-empty ``tasks`` list; per-task problems
    reject that task with a reason and keep going. The DAG is not mutated
    on an ok=False response. An empty decomposition is a failed response,
    never "nothing to do": the run only reaches the decomposer while the
    global goal is incomplete.
    """
    # Late import to avoid a cycle (core imports dag-adjacent modules).
    from orchestrator.core import _normalize_agent

    result = {"ok": False, "error": None, "accepted": [], "rejected": [],
              "warnings": []}
    if not isinstance(parsed, dict):
        result["error"] = "response is not a JSON object"
        return result
    raw_tasks = parsed.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        result["error"] = "missing/empty 'tasks' list"
        return result

    result["ok"] = True
    unreachable_set = set(unreachable or ())
    alias = {}   # raw id string -> final sanitized id, for intra-batch refs
    for i, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            result["rejected"].append((f"item_{i}", "not an object"))
            continue
        desc = str(item.get("description") or "").strip()
        if not desc:
            result["rejected"].append(
                (str(item.get("id") or f"item_{i}"), "no description"))
            continue
        desc = desc[:_DESCRIPTION_MAX_CHARS]

        raw_id = item.get("id")
        tid = _sanitize_id(raw_id, fallback=f"t{t}_{i}")
        base = tid
        n = 2
        while tid in dag.tasks or tid in alias.values():
            tid = f"{base}_{n}"
            n += 1

        raw_ms = item.get("milestones")
        raw_ms = raw_ms if isinstance(raw_ms, list) else []
        milestones = []
        n_unreachable = 0
        for m in raw_ms:
            ms = str(m).strip()
            if ms in unreachable_set:
                n_unreachable += 1
                result["warnings"].append(
                    f"task {tid}: dropped unreachable milestone {ms!r} "
                    f"(chamber behind the team)")
            elif ms in known_milestones:
                milestones.append(ms)
            else:
                result["warnings"].append(
                    f"task {tid}: dropped unknown milestone {ms!r}")
        if not milestones:
            reason = ("unreachable chamber" if n_unreachable
                      else "no verifiable milestone")
            result["rejected"].append((tid, reason))
            continue
        if set(milestones) <= set(team_completed):
            result["rejected"].append((tid, "already satisfied"))
            continue

        required = []
        for r in (item.get("required")
                  if isinstance(item.get("required"), list) else []):
            rid = str(r).strip()
            resolved = alias.get(rid) or _sanitize_id(rid, fallback="")
            if resolved and resolved in dag.tasks:
                required.append(resolved)
            else:
                result["warnings"].append(
                    f"task {tid}: dropped unknown predecessor {rid!r}")

        candidates = []
        for c in (item.get("candidates")
                  if isinstance(item.get("candidates"), list) else []):
            cn = _normalize_agent(c)
            if cn is not None and cn in living_agents:
                if cn not in candidates:
                    candidates.append(cn)
            else:
                result["warnings"].append(
                    f"task {tid}: dropped candidate {c!r}")

        try:
            min_agents = int(item.get("min_agents", 1))
        except (TypeError, ValueError):
            min_agents = 1
        upper = max(1, len(living_agents))
        if not (1 <= min_agents <= upper):
            result["warnings"].append(
                f"task {tid}: min_agents clamped from {min_agents}")
            min_agents = min(max(min_agents, 1), upper)

        if dag.open_count() >= max_open:
            result["warnings"].append(
                f"task {tid}: dropped — open-task cap ({max_open}) reached")
            continue

        task = CentralTask(
            id=tid, description=desc, milestones=milestones,
            required=required, candidates=candidates,
            min_agents=min_agents, created_at_step=t,
        )
        ok, reason = dag.add_task(task)
        if not ok:
            result["rejected"].append((tid, reason))
            continue
        alias[str(raw_id)] = tid
        result["accepted"].append(tid)
    return result
