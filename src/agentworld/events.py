"""Pairwise interaction events from tool results and observations.

Events feed the multi-channel Hebbian graph (as ``(i, j, channel)`` triples),
the dense progress reward, the sidecar ``events.jsonl`` and the replay video:

    xfer    i transferred items to j (explicit, from the tool's result text)
    combat  i and j attacked the same mob instance within ``combat_window`` rounds
    kill    a mob died; ``data["attackers"]`` lists everyone who hit it recently
    craft   i crafted an item
    death   i's HP crossed from > 0 to ≤ 0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agentworld.executor import ActionResult


@dataclass
class InteractionEvent:
    round: int
    kind: str
    src: int
    dst: Optional[int] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventExtractor:
    def __init__(self, names: List[str], combat_window: int = 2):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.combat_window = combat_window
        self._attacks: Dict[str, Dict[int, int]] = {}   # target → {agent: last round}
        self._hp: Dict[int, float] = {}

    def from_results(self, round_num: int, results: Iterable[ActionResult]) -> List[InteractionEvent]:
        events: List[InteractionEvent] = []
        for res in results:
            i = res.agent
            call = res.call
            if call.name == "transfer_items":
                xfer = res.transfer()
                if xfer:
                    j = self.index.get(xfer["receiver"], self.index.get(call.args.get("targetPlayer")))
                    if j is not None and j != i:
                        events.append(InteractionEvent(round_num, "xfer", i, j, xfer))
            elif call.name == "craft_item":
                crafted = res.crafted()
                if crafted:
                    events.append(InteractionEvent(round_num, "craft", i, None, crafted))
            elif call.name == "attack_entity":
                target = str(call.args.get("targetInstance", ""))
                hitters = self._attacks.setdefault(target, {})
                for j, last in hitters.items():
                    if j != i and round_num - last <= self.combat_window:
                        events.append(InteractionEvent(round_num, "combat", i, j, {"target": target}))
                hitters[i] = round_num
                if res.killed:
                    attackers = sorted(j for j, last in hitters.items()
                                       if round_num - last <= self.combat_window)
                    name = self._mob_name(res, target)
                    events.append(InteractionEvent(round_num, "kill", i, None,
                                                   {"target": target, "mob": name,
                                                    "attackers": attackers}))
                    self._attacks.pop(target, None)
        return events

    @staticmethod
    def _mob_name(res: ActionResult, target: str) -> str:
        text = res.text or ""
        if "Defeated " in text:
            return text.split("Defeated ", 1)[1].split(" (")[0].strip()
        if "VICTORY (Team Kill):" in text:
            return text.split("VICTORY (Team Kill):", 1)[1].split(" (")[0].strip()
        return target

    def deaths(self, round_num: int, observations: Dict[int, Optional[Dict[str, Any]]]
               ) -> List[InteractionEvent]:
        out = []
        for i, obs in observations.items():
            if not obs:
                continue
            hp = (obs.get("playerStatus") or {}).get("hitPoints")
            if hp is None:
                continue
            prev = self._hp.get(i)
            if prev is not None and prev > 0 and hp <= 0:
                out.append(InteractionEvent(round_num, "death", i, None, {"hp": hp}))
            self._hp[i] = hp
        return out


def hebbian_triples(events: List[InteractionEvent], dms: List[Tuple[int, int]] = ()
                    ) -> List[Tuple[int, int, str]]:
    """``(initiator, target, channel)`` triples for MultiChannelHebbianGraph."""
    out: List[Tuple[int, int, str]] = [(s, r, "comm") for s, r in dms if s != r]
    for e in events:
        if e.kind in ("xfer", "combat") and e.dst is not None and e.dst != e.src:
            out.append((e.src, e.dst, e.kind))
    return out
