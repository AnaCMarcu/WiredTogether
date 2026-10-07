"""Trajectories in AgentWorld's schema, plus the harness's own sidecar logs.

One :class:`TeamTrajectory` per team. Its JSON is exactly what AgentWorld's
runner writes (``task_id``, ``task_source``, ``task_key``, ``timestamp``,
``task_definition``, ``rounds[].actions[]`` with ``agent_name`` /
``status`` / ``action`` / ``observation`` / ``thinking``), so the task's own
``verify()`` and the CCE judge read it unchanged.

Communication is recorded as ``chat(message=...)`` actions because some
verifiers count chat messages: a board post as ``chat(message=<text>)``, a
direct message as ``chat(message=@<receiver>: <text>)``. They are written
before the agent's game action in the round, with the same status and
observation, so the last observation per agent stays the game action's.

:class:`RunLog` appends the JSONL sidecars (messages, events, per-round replay
state) that the analysis scripts and the video renderer read.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentworld.tasks import TeamSlot, task_definition_for


def status_line(observation: Optional[Dict[str, Any]]) -> str:
    """AgentWorld console's status string (agents/console.py get_player_status).

    The verifiers parse HP from it with ``split('❤️')`` / ``❤️(\\d+)/(\\d+)``.
    """
    if not observation or not observation.get("playerStatus"):
        return "📊 Status: Player data not available (no playerStatus field)"
    ps = observation["playerStatus"]
    loc = observation.get("location", {}) or {}
    combat = "⚔️" if ps.get("combat", False) else ""
    return (f"📊 Lv.{ps.get('level', 1)} | XP:{ps.get('experience', 0)} | "
            f"❤️{ps.get('hitPoints', '?')}/{ps.get('maxHitPoints', '?')} | "
            f"💙{ps.get('mana', '?')}/{ps.get('maxMana', '?')} | "
            f"📍({loc.get('x', '?')},{loc.get('y', '?')}) {combat}").strip()


class TeamTrajectory:
    def __init__(self, slot: TeamSlot):
        self.slot = slot
        task = slot.task
        self.data: Dict[str, Any] = {
            "task_id": task.task_id,
            "task_source": str(task.path.resolve()),
            "task_key": task.task_key,
            "timestamp": datetime.now().isoformat(),
            "task_definition": task_definition_for(slot),
            "rounds": [],
            "metrics": {},
        }
        self._round: Optional[Dict[str, Any]] = None

    def begin_round(self, round_num: int) -> None:
        self._round = {"round": round_num, "actions": []}
        self.data["rounds"].append(self._round)

    def _append(self, agent_key: str, status: str, action: str,
                observation: Any, thinking: str = "") -> None:
        if self._round is None:
            raise RuntimeError("begin_round() first")
        self._round["actions"].append({
            "agent_name": agent_key,
            "status": status,
            "action": action,
            "observation": observation if observation else "",
            "thinking": thinking,
        })

    def add_action(self, agent_key: str, status: str, action: str,
                   observation: Any, thinking: str = "") -> None:
        self._append(agent_key, status, action, observation, thinking)

    def add_board_post(self, agent_key: str, text: str, status: str, observation: Any) -> None:
        self._append(agent_key, status, f"chat(message={text})", observation)

    def add_dm(self, agent_key: str, receiver: str, text: str, status: str,
               observation: Any) -> None:
        self._append(agent_key, status, f"chat(message=@{receiver}: {text})", observation)

    def save(self, directory: Path, metrics: Optional[Dict[str, Any]] = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        if metrics is not None:
            self.data["metrics"] = metrics
        path = directory / f"{self.slot.team}_{self.slot.task.task_id}_trajectory.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path


class RunLog:
    """Append-only JSONL sidecars under one episode directory."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        (self.dir / "replay").mkdir(parents=True, exist_ok=True)
        self._files: Dict[str, Any] = {}

    def _fh(self, name: str):
        fh = self._files.get(name)
        if fh is None:
            fh = open(self.dir / name, "a", encoding="utf-8")
            self._files[name] = fh
        return fh

    def write(self, name: str, record: Dict[str, Any]) -> None:
        fh = self._fh(name)
        fh.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
        fh.flush()

    def message(self, record: Dict[str, Any]) -> None:
        self.write("messages.jsonl", record)

    def event(self, record: Dict[str, Any]) -> None:
        self.write("events.jsonl", record)

    def state(self, record: Dict[str, Any]) -> None:
        self.write("replay/state.jsonl", record)

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._files.clear()


def _jsonable(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
