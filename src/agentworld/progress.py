"""Dense per-agent "bondable" reward from task progress (the Hebbian third factor).

AgentWorld only scores a task at the end (binary ``verify``). The three-factor
rule needs a salience signal per agent per round, so this module derives one
from what the verifier checks:

    r_i = w_gain   · (own gains of relevant items: harvest / craft / pickup;
                      items received in a transfer are excluded)
        + w_give   · (relevant items i handed to a teammate)
        + w_kill   · 1[i hit a mob that died (matching the task's patterns)]
        + w_prog   · Δp_team · 1[i did something relevant in the last k rounds]
        + w_solve  · 1[the team's verify() flipped to success this round]

Relevant items are the verifier's target keys plus intermediate items named
in the task text (e.g. logs → stick → arrow). Each component is capped.
A death (HP crossing 0) is returned separately as ``death_penalty`` so it
reaches the rule's death-LTD path and never counts as positive salience.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from agentworld.events import InteractionEvent
from agentworld.tasks import WorldSpec
from agentworld.verify import Targets


@dataclass
class ProgressWeights:
    gain: float = 2.0
    give: float = 3.0
    kill: float = 10.0
    progress: float = 20.0
    solve: float = 50.0
    cap: float = 50.0            # per component per agent per round
    recent_rounds: int = 3
    death_penalty: float = -10.0


def _inventory(obs: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not obs or "inventory" not in obs:
        return None
    inv: Dict[str, int] = defaultdict(int)
    for it in (obs.get("inventory") or {}).get("items") or []:
        if isinstance(it, dict):
            inv[str(it.get("key", "")).lower()] += int(it.get("count", 0) or 0)
    return dict(inv)


class ProgressTracker:
    def __init__(self, world: WorldSpec, targets: Dict[str, Targets],
                 weights: Optional[ProgressWeights] = None):
        self.world = world
        self.targets = targets
        self.w = weights or ProgressWeights()
        self._text = {t.team: (t.task.objective + "\n" + t.task.game_context).lower()
                      for t in world.teams}
        self._relevant_cache: Dict[Tuple[str, str], bool] = {}
        self._inv: Dict[int, Dict[str, int]] = {}
        self._progress: Dict[str, float] = {t.team: 0.0 for t in world.teams}
        self._solved: Dict[str, bool] = {t.team: False for t in world.teams}
        self._recent: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=64))

    def relevant(self, team: str, key: str) -> bool:
        k = (team, key)
        if k not in self._relevant_cache:
            tg = self.targets.get(team)
            hit = bool(tg and (key in tg.items or any(p in key for p in tg.patterns)))
            if not hit and len(key) >= 3:
                hit = re.search(rf"\b{re.escape(key)}s?\b", self._text.get(team, "")) is not None
            self._relevant_cache[k] = hit
        return self._relevant_cache[k]

    def observe_initial(self, observations: Dict[int, Optional[Dict[str, Any]]]) -> None:
        for i, obs in observations.items():
            inv = _inventory(obs)
            if inv is not None:
                self._inv[i] = inv

    def update(
        self,
        round_num: int,
        observations: Dict[int, Optional[Dict[str, Any]]],
        events: List[InteractionEvent],
        team_progress: Dict[str, Optional[float]],
        team_success: Dict[str, int],
    ) -> Tuple[List[float], List[float], Dict[int, Dict[str, float]]]:
        """Returns (bond_rewards, death_rewards, per-agent component breakdown)."""
        n = self.world.n_agents
        team_of = self.world.global_team
        parts: Dict[int, Dict[str, float]] = {i: defaultdict(float) for i in range(n)}

        received: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in events:
            if e.kind == "xfer" and e.dst is not None:
                key = str(e.data.get("itemKey") or e.data.get("item") or "").lower()
                cnt = int(e.data.get("count", 0))
                received[e.dst][key] += cnt
                if self.relevant(team_of[e.src], key):
                    parts[e.src]["give"] += self.w.give * cnt
                    self._recent[e.src].append(round_num)
            elif e.kind == "kill":
                tg = self.targets.get(team_of[e.src])
                mob = str(e.data.get("mob", "")).lower()
                wanted = (not tg or not tg.patterns or any(p in mob for p in tg.patterns))
                if wanted:
                    for a in e.data.get("attackers", [e.src]):
                        parts[a]["kill"] += self.w.kill
                        self._recent[a].append(round_num)

        for i, obs in observations.items():
            inv = _inventory(obs)
            if inv is None:
                continue
            prev = self._inv.get(i)
            self._inv[i] = inv
            if prev is None:
                continue
            for key, cnt in inv.items():
                gained = cnt - prev.get(key, 0) - received[i].get(key, 0)
                if gained > 0 and self.relevant(team_of[i], key):
                    parts[i]["gain"] += self.w.gain * gained
                    self._recent[i].append(round_num)

        for slot in self.world.teams:
            p = team_progress.get(slot.team)
            if p is not None:
                dp = max(0.0, p - self._progress[slot.team])
                self._progress[slot.team] = max(self._progress[slot.team], p)
                if dp > 0:
                    for i in slot.members:
                        if any(round_num - r < self.w.recent_rounds for r in self._recent[i]):
                            parts[i]["progress"] += self.w.progress * dp
            if team_success.get(slot.team) and not self._solved[slot.team]:
                self._solved[slot.team] = True
                for i in slot.members:
                    parts[i]["solve"] += self.w.solve

        deaths = {e.src for e in events if e.kind == "death"}
        bond = []
        for i in range(n):
            total = 0.0
            for name, val in parts[i].items():
                parts[i][name] = min(val, self.w.cap)
                total += parts[i][name]
            bond.append(total)
        death = [self.w.death_penalty if i in deaths else 0.0 for i in range(n)]
        return bond, death, {i: dict(p) for i, p in parts.items() if p}

    def team_progress(self, team: str) -> float:
        return self._progress.get(team, 0.0)
