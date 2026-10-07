"""Hub-and-spoke plumbing for the centralised-communication variant (hmas2).

In a hub variant no message travels agent to agent: an agent's per-step
``communication`` is a report to the orchestrator (the main loop calls
``HubReports.receive`` instead of writing a teammate's inbox), and the
orchestrator's message to an agent is delivered by REPLACING that agent's
inbox, so it is the only message the agent sees. This module holds the
pieces that are independent of how the orchestrator decides what to say:
report intake, text sanitising, delivery (with recipient-pays budget
accounting) and the per-delivery log rows.

Stdlib only.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from orchestrator.core import _normalize_agent

#: ``TextMessage.source`` of every hub message, and the routing target name
#: agents are told to use.
HUB_SOURCE = "orchestrator"
#: One report as shown to the orchestrator (agent messages are short; this
#: only guards against a runaway generation).
REPORT_MAX_CHARS = 400


def sanitize_hub_text(text, max_words: Optional[int]) -> str:
    """Make hub text safe to deliver: braces become parentheses (the action
    prompt is ``str.format``-ed AFTER the inbox text is appended, so a stray
    brace would turn every step into a NoOp until the next replacement),
    whitespace is collapsed, and the word cap is enforced."""
    s = str(text or "").replace("{", "(").replace("}", ")")
    words = s.split()
    if max_words is not None and len(words) > int(max_words):
        words = words[:int(max_words)]
    return " ".join(words)


def clip(text, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


class HubReports:
    """Per-agent buffer of reports received since they were last consumed.

    Each agent keeps its newest ``cap`` reports; overflow is counted as
    dropped (a hub-capacity signal)."""

    def __init__(self, agent_names, cap: int = 2):
        self.agent_names = list(agent_names)
        self.cap = int(cap)
        self.reset()

    def reset(self) -> None:
        self._pending = {a: deque(maxlen=self.cap) for a in self.agent_names}
        self.dropped_since_consume = 0
        self.reports_in = 0
        self.reports_dropped = 0

    def receive(self, *, t: int, sender: str, text: str,
                chamber: Optional[str] = None) -> None:
        agent = _normalize_agent(sender) or str(sender)
        body = clip(text, REPORT_MAX_CHARS)
        if not body:
            return
        queue = self._pending.setdefault(agent, deque(maxlen=self.cap))
        if len(queue) == queue.maxlen:
            self.dropped_since_consume += 1
            self.reports_dropped += 1
        queue.append((int(t), str(chamber or "?"), body))
        self.reports_in += 1

    def pending(self) -> list:
        """[(t, agent, chamber, text)] oldest first."""
        rows = [(t, agent, ch, text)
                for agent, queue in self._pending.items()
                for (t, ch, text) in queue]
        return sorted(rows, key=lambda r: (r[0], r[1]))

    def count(self) -> int:
        return sum(len(q) for q in self._pending.values())

    def consume(self) -> None:
        for queue in self._pending.values():
            queue.clear()
        self.dropped_since_consume = 0


def deliver_hub_messages(inboxes: dict, messages: dict, *, living: list,
                         step: int, make_message: Callable[[str], object],
                         comm_budget=None) -> dict:
    """Write hub messages (``{agent: text}``) into the recipients' inboxes.

    REPLACE semantics: each recipient's inbox becomes exactly
    ``[make_message(text)]``; agents without a new message keep whatever
    they had (a stale message persists, as a peer message does in mesh).
    Dead agents are never written. With ``comm_budget`` (recipient-pays)
    each message is charged to the recipient: blocked -> not delivered (old
    inbox kept), truncated -> the cut text is delivered. Returns
    ``{agent: delivery record}``."""
    living_set = {_normalize_agent(a) or a for a in living}
    out = {}
    for agent, text in (messages or {}).items():
        name = _normalize_agent(agent) or agent
        rec = {"delivered": False, "status": None, "charged": 0,
               "budget_left": None, "text": ""}
        if name not in living_set:
            rec["status"] = "dead"
            out[name] = rec
            continue
        try:
            idx = int(str(name).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            rec["status"] = "bad_name"
            out[name] = rec
            continue
        wire = sanitize_hub_text(text, None)
        if not wire:
            continue
        rec["status"] = "sent"
        if comm_budget is not None:
            charge = comm_budget.charge_incoming(idx, wire, step)
            rec.update(status=charge.status, charged=int(charge.charged),
                       budget_left=int(charge.budget_left),
                       exhausted_now=bool(charge.exhausted_now))
            if charge.status == "blocked":
                out[name] = rec
                continue
            wire = charge.text
        inboxes[idx] = [make_message(wire)]
        rec["delivered"] = True
        rec["text"] = wire
        out[name] = rec
    return out
