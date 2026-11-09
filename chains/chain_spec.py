"""Chain definitions and agent naming for chained AgentWorld tasks.

A chain is an ordered list of AgentWorld tasks played by one persistent team.
Each step names a source task YAML and maps that task's slots (``agent_1``,
``agent_2``, ...) to stable agent ids (``a1``, ``a2``, ...). In the game an
agent is called ``<agent_id>_<role>``, where the role comes from the task's
own username: ``t06_weapon_smith_agent`` -> ``weapon_smith`` ->
``a3_weapon_smith``. The agent id is the identity and the key for all
persistent state; the role may change from task to task.

Chain file (``chains/<name>.yaml``)::

    chain: smithing_3
    description: optional free text
    steps:
      - task: data_v0.1_multi/v1.3_benchmark/task_03_silver_ring.yaml
        agents: {agent_1: a1, agent_2: a2, agent_3: a3}

``task`` is absolute, or relative to the AgentWorld checkout
(``--agentworld-root`` / ``$AGENTWORLD_ROOT``), or relative to the chain file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# packages/common/util/utils.ts: isValidUsername(text) = /^[\w ]+$/.test(text)
SERVER_USERNAME = re.compile(r"^[A-Za-z0-9_ ]+$")
MAX_USERNAME = 32
# Stricter than the server: lowercase, no spaces, and no underscore in the id,
# so "<agent_id>_<role>" splits back unambiguously at the first underscore.
AGENT_ID = re.compile(r"^[a-z][a-z0-9]*$")
ROLE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
CHAIN_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")
SLOT = re.compile(r"^agent_(\d+)$")
TASK_PREFIX = re.compile(r"^t\d+_")
TASK_NUMBER = re.compile(r"^task_(\d+)_")
ROLE_SUFFIX = "_agent"


class NamingError(ValueError):
    """A username, agent id or role that the chain cannot use."""


class ChainError(ValueError):
    """A malformed chain definition."""


# ── naming ───────────────────────────────────────────────────────────────────
def validate_agent_id(agent_id: str) -> str:
    if not isinstance(agent_id, str) or not AGENT_ID.match(agent_id):
        raise NamingError(
            f"agent id {agent_id!r} must be lowercase letters/digits starting with a letter "
            "and contain no underscore (e.g. a1)")
    if re.fullmatch(r"t\d+", agent_id):
        raise NamingError(f"agent id {agent_id!r} looks like a task prefix (tNN); use e.g. a1")
    return agent_id


def _check_role(role: str) -> str:
    if not ROLE.match(role):
        raise NamingError(f"role {role!r} must be lowercase letters/digits joined by underscores")
    if role.endswith(ROLE_SUFFIX):
        raise NamingError(f"role {role!r} ends in {ROLE_SUFFIX!r}: that is an original task "
                          "username, not a chain name")
    return role


def role_from_username(original: str) -> str:
    """``t06_weapon_smith_agent`` -> ``weapon_smith`` (strip tNN_ and _agent)."""
    name = str(original).strip()
    role = TASK_PREFIX.sub("", name, count=1)
    if role.endswith(ROLE_SUFFIX):
        role = role[: -len(ROLE_SUFFIX)]
    if not ROLE.match(role):
        raise NamingError(
            f"cannot derive a role from task username {original!r}: got {role!r}, expected "
            "tNN_<role>_agent with a lowercase role of letters, digits and underscores")
    return role


def validate_username(name: str) -> str:
    """Fail loudly unless the server accepts ``name`` and it fits our format rules."""
    if not SERVER_USERNAME.match(name):
        raise NamingError(f"username {name!r} has characters the server rejects "
                          "(isValidUsername allows only letters, digits, underscores, spaces)")
    if name != name.lower():
        raise NamingError(f"username {name!r} must be lowercase (the server lowercases names)")
    if len(name) > MAX_USERNAME:
        raise NamingError(f"username {name!r} is {len(name)} characters; the limit is {MAX_USERNAME}")
    return name


def make_username(agent_id: str, role: str) -> str:
    validate_agent_id(agent_id)
    _check_role(role)
    return validate_username(f"{agent_id}_{role}")


def parse_username(name: str) -> Tuple[str, str]:
    """``a3_weapon_smith`` -> (``a3``, ``weapon_smith``); raises on anything else."""
    validate_username(name)
    agent_id, sep, role = name.partition("_")
    if not sep or not AGENT_ID.match(agent_id) or not ROLE.match(role):
        raise NamingError(f"username {name!r} is not <agent_id>_<role> (e.g. a1_miner)")
    validate_agent_id(agent_id)
    _check_role(role)
    return agent_id, role


def task_number(stem: str) -> str:
    """``task_07_pickaxe_crafting`` -> ``07`` (the stem itself if it has no number)."""
    m = TASK_NUMBER.match(stem)
    return m.group(1) if m else stem


def slot_order(keys) -> List[str]:
    return sorted((k for k in keys if SLOT.match(k)), key=lambda k: int(SLOT.match(k).group(1)))


# ── chain ────────────────────────────────────────────────────────────────────
@dataclass
class SlotAssignment:
    slot: str               # "agent_2"
    agent_id: str           # "a2"
    original_username: str  # "t06_smelter_agent"
    role: str               # "smelter"
    username: str           # "a2_smelter"

    def to_dict(self) -> Dict[str, str]:
        return {"slot": self.slot, "agent_id": self.agent_id, "role": self.role,
                "username": self.username, "original_username": self.original_username}


@dataclass
class ChainStep:
    index: int                       # 1-based position in the chain
    source: Path                     # original task YAML (never edited)
    agents: Dict[str, str]           # slot -> agent_id, as written in the chain file
    assignments: List[SlotAssignment] = field(default_factory=list)

    @property
    def task_stem(self) -> str:
        return self.source.stem

    @property
    def task_number(self) -> str:
        return task_number(self.task_stem)

    @property
    def dir_name(self) -> str:
        return f"step_{self.index:02d}_task_{self.task_number}"

    def by_slot(self) -> Dict[str, SlotAssignment]:
        return {a.slot: a for a in self.assignments}


@dataclass
class Chain:
    name: str
    description: str
    path: Path
    steps: List[ChainStep]

    @property
    def agent_ids(self) -> List[str]:
        """Every agent id in order of first appearance."""
        seen: List[str] = []
        for step in self.steps:
            for a in step.assignments:
                if a.agent_id not in seen:
                    seen.append(a.agent_id)
        return seen


def resolve_root(agentworld_root: Optional[str | os.PathLike] = None) -> Optional[Path]:
    value = agentworld_root if agentworld_root is not None else os.environ.get("AGENTWORLD_ROOT")
    return Path(value).expanduser().resolve() if value else None


def _resolve_task(raw: str, chain_path: Path, root: Optional[Path]) -> Path:
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else ([root / p] if root else []) + [chain_path.parent / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    tried = ", ".join(str(c) for c in candidates)
    hint = "" if root else " (set --agentworld-root or $AGENTWORLD_ROOT)"
    raise ChainError(f"task file {raw!r} not found; tried {tried}{hint}")


def assign_slots(task_data: Dict, agents: Dict[str, str], where: str) -> List[SlotAssignment]:
    """Check a step's slot -> agent_id map against the task and derive the names."""
    task_slots = slot_order(task_data)
    if not task_slots:
        raise ChainError(f"{where}: the task defines no agent_N slots")
    given = set(agents)
    if given != set(task_slots):
        missing = sorted(set(task_slots) - given)
        extra = sorted(given - set(task_slots))
        raise ChainError(f"{where}: map every task slot exactly once "
                         f"(task has {task_slots}; missing {missing}, unknown {extra})")
    ids = [validate_agent_id(agents[s]) for s in task_slots]
    if len(set(ids)) != len(ids):
        raise ChainError(f"{where}: an agent id appears in two slots: {ids}")
    out = []
    for slot in task_slots:
        block = task_data[slot] or {}
        original = str(block.get("username") or "")
        if not original:
            raise ChainError(f"{where}: {slot} has no username")
        role = role_from_username(original)
        out.append(SlotAssignment(slot=slot, agent_id=agents[slot], original_username=original,
                                  role=role, username=make_username(agents[slot], role)))
    names = [a.username for a in out]
    if len(set(names)) != len(names):
        raise ChainError(f"{where}: two slots get the same username: {names}")
    return out


def load_chain(path: str | os.PathLike,
               agentworld_root: Optional[str | os.PathLike] = None) -> Chain:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = data.get("chain")
    if not isinstance(name, str) or not CHAIN_NAME.match(name):
        raise ChainError(f"{path}: 'chain' must be a lowercase name of letters, digits, underscores")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ChainError(f"{path}: 'steps' must be a non-empty list")
    root = resolve_root(agentworld_root)
    steps = []
    for k, raw in enumerate(raw_steps, start=1):
        where = f"{path.name} step {k}"
        if not isinstance(raw, dict) or "task" not in raw or "agents" not in raw:
            raise ChainError(f"{where}: each step needs 'task' and 'agents'")
        agents = raw["agents"]
        if not isinstance(agents, dict) or not all(isinstance(v, str) for v in agents.values()):
            raise ChainError(f"{where}: 'agents' must map slots to agent ids, e.g. {{agent_1: a1}}")
        source = _resolve_task(str(raw["task"]), path, root)
        task_data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        step = ChainStep(index=k, source=source, agents=dict(agents))
        step.assignments = assign_slots(task_data, step.agents, where)
        steps.append(step)
    return Chain(name=name, description=str(data.get("description") or ""), path=path, steps=steps)
