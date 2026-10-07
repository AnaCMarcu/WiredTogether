"""AgentWorld task YAMLs → typed specs, and multi-team world composition.

A task YAML has ``task``, ``objectives``, ``relevant_game_context``,
``max_action_steps``, optional ``rounds``, and one ``agent_<k>`` block per
agent (username, location, skill_levels, inventory_items, equipped_items,
new_character). The item-spec strings follow AgentWorld's runner
(agents/run.py ``_create_agent_console``) exactly, because its console parses
them back with ``split(':')``.

A *world* is one or more tasks played in the same server at once. Each task
keeps its own team (trajectory, verifier), so a 10 × N=3 world is 30 agents
whose success is still judged by the original per-task verifiers.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

AGENT_KEY = re.compile(r"^agent_(\d+)$")
DEFAULT_ROUNDS = 55


@dataclass
class AgentSpec:
    key: str                      # "agent_3" — the name trajectories use
    username: str
    location: Optional[Tuple[int, int]]
    skill_levels: Dict[str, int]
    inventory_items: List[str]    # "key:count[:enchant]"
    equipped_items: List[str]     # "key[:count][:enchant]" (run.py's quirk kept)
    new_character: bool
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)


@dataclass
class TaskSpec:
    path: Path
    task_id: str                  # "task_02"
    task_key: str                 # file stem, e.g. "task_02_arrow_production"
    name: str
    description: str
    objective: str
    game_context: str
    max_action_steps: Optional[int]
    rounds: int
    agents: List[AgentSpec]
    definition: Dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def verifier_path(self) -> Path:
        num = self.task_id.split("_", 1)[1]
        candidates = [self.path.parent / f"task_{num}_success_criteria.py",
                      self.path.parent.parent / f"task_{num}_success_criteria.py"]
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(f"no success criteria for {self.task_id} near {self.path}")


def inventory_spec(item: Dict[str, Any]) -> str:
    spec = f"{item['item']}:{item['count']}"
    if item.get("enchant", 0) > 0:
        spec += f":{item['enchant']}"
    return spec


def equipped_spec(item: Dict[str, Any]) -> str:
    spec = item["item"]
    count = item.get("count", 1)
    if count > 1:
        spec += f":{count}"
    if item.get("enchant", 0) > 0:
        spec += f":{item['enchant']}" if count > 1 else f":1:{item['enchant']}"
    return spec


def _agent_spec(key: str, block: Dict[str, Any]) -> AgentSpec:
    loc = block.get("location")
    return AgentSpec(
        key=key,
        username=str(block.get("username", key)),
        location=(int(loc["x"]), int(loc["y"])) if loc else None,
        skill_levels={str(k): int(v) for k, v in (block.get("skill_levels") or {}).items()},
        inventory_items=[inventory_spec(i) for i in block.get("inventory_items") or []],
        equipped_items=[equipped_spec(i) for i in block.get("equipped_items") or []],
        new_character=bool(block.get("new_character", False)),
        raw=block,
    )


def load_task(path: str | Path) -> TaskSpec:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    stem = path.stem
    num = stem.split("_")[1] if "_" in stem else stem
    agent_keys = sorted((k for k in data if AGENT_KEY.match(k)),
                        key=lambda k: int(AGENT_KEY.match(k).group(1)))
    task = data.get("task") or {}
    return TaskSpec(
        path=path,
        task_id=f"task_{num}",
        task_key=stem,
        name=str(task.get("name", stem)),
        description=str(task.get("description", "")),
        objective=str((data.get("objectives") or {}).get("primary", "")),
        game_context=str(data.get("relevant_game_context", "")),
        max_action_steps=data.get("max_action_steps"),
        rounds=int(data.get("rounds", DEFAULT_ROUNDS)),
        agents=[_agent_spec(k, data[k]) for k in agent_keys],
        definition=data,
    )


def list_suite(directory: str | Path) -> List[Path]:
    """Task YAMLs of a suite directory (top level only), ordered by task number."""
    directory = Path(directory)
    files = [p for p in directory.glob("task_*.yaml")]

    def num(p: Path) -> int:
        m = re.match(r"task_(\d+)", p.stem)
        return int(m.group(1)) if m else 10 ** 9

    return sorted(files, key=lambda p: (num(p), p.name))


# ── worlds ──────────────────────────────────────────────────────────────────
@dataclass
class TeamSlot:
    team: str                     # "t00", "t01", ...
    task: TaskSpec
    members: List[int]            # global agent indices
    usernames: Dict[str, str]     # agent key → in-world username


@dataclass
class WorldSpec:
    teams: List[TeamSlot]
    global_names: List[str]       # in-world username per global index
    global_team: List[str]        # team id per global index
    global_key: List[str]         # agent key within its task per global index
    global_spawn: List[Optional[Tuple[int, int]]] = field(default_factory=list)

    @property
    def n_agents(self) -> int:
        return len(self.global_names)

    @property
    def rounds(self) -> int:
        return max(t.task.rounds for t in self.teams)

    def team_of(self, index: int) -> TeamSlot:
        tid = self.global_team[index]
        return next(t for t in self.teams if t.team == tid)

    def true_team_matrix(self):
        """N×N 0/1 matrix of hidden team membership (diagonal 0)."""
        import numpy as np
        n = self.n_agents
        same = np.zeros((n, n), dtype=np.float32)
        for t in self.teams:
            for i in t.members:
                for j in t.members:
                    if i != j:
                        same[i, j] = 1.0
        return same


def team_offset(k: int, spacing: int, columns: int) -> Tuple[int, int]:
    """Grid offset of team k (team 0 stays at the task's own spawn)."""
    if spacing <= 0:
        return (0, 0)
    return ((k % columns) * spacing, (k // columns) * spacing)


def compose_world(tasks: List[TaskSpec], prefix_teams: Optional[bool] = None,
                  team_spacing: int = 0, columns: int = 5) -> WorldSpec:
    """Place several tasks in one world, each as its own team.

    With more than one task, usernames get a ``t<k>_`` team prefix (unless
    ``prefix_teams=False``) so replicas of the same task never collide. A
    single task keeps its usernames, which keeps E0/E1 runs identical to the
    benchmark's own.

    ``team_spacing`` shifts team k's spawns by a grid offset. Replicas of one
    task otherwise spawn on the same tiles, and the proximity channel then
    bonds agents ACROSS teams as strongly as within them, which would make
    any bond-recovery measurement meaningless. Use a spacing larger than the
    Hebbian interaction radius; whether shifted spawns are walkable on the
    real map must be checked against the server (P0).
    """
    prefix = (len(tasks) > 1) if prefix_teams is None else prefix_teams
    teams: List[TeamSlot] = []
    names: List[str] = []
    team_of: List[str] = []
    keys: List[str] = []
    spawns: List[Optional[Tuple[int, int]]] = []
    for k, task in enumerate(tasks):
        tid = f"t{k:02d}"
        dx, dy = team_offset(k, team_spacing, columns)
        members, usernames = [], {}
        for spec in task.agents:
            uname = f"{tid}_{spec.username}" if prefix else spec.username
            usernames[spec.key] = uname
            members.append(len(names))
            names.append(uname)
            team_of.append(tid)
            keys.append(spec.key)
            spawns.append((spec.location[0] + dx, spec.location[1] + dy)
                          if spec.location else None)
        teams.append(TeamSlot(team=tid, task=task, members=members, usernames=usernames))
    if len(set(names)) != len(names):
        raise ValueError("duplicate usernames in world; compose with prefix_teams=True")
    return WorldSpec(teams=teams, global_names=names, global_team=team_of, global_key=keys,
                     global_spawn=spawns)


def anonymise_text(text: str, usernames: Dict[str, str], roles: Dict[str, str]) -> str:
    """Replace teammate usernames in task text with role descriptions.

    ``usernames`` maps agent key → original username; ``roles`` maps agent key
    → description (e.g. "the fletcher"). Used for hidden-team worlds, so agents
    must discover partners rather than read them off the objective.
    """
    out = text
    for key, uname in sorted(usernames.items(), key=lambda kv: -len(kv[1])):
        role = roles.get(key)
        if role and uname:
            out = out.replace(uname, role)
    return out


def task_definition_for(slot: TeamSlot) -> Dict[str, Any]:
    """The task definition a team's trajectory records (usernames remapped)."""
    d = copy.deepcopy(slot.task.definition)
    for key, uname in slot.usernames.items():
        if isinstance(d.get(key), dict):
            d[key]["username"] = uname
    return d
