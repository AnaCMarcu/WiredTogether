"""Persistent within-episode state for the task-ledger orchestrator.

The memory horizon is the experimental point of the O2 condition: the ledger
persists WITHIN an episode only and is wiped at every episode start — in
contrast with the Hebbian W(t), which persists across episodes. ``reset()``
must therefore be called from the training loop at every episode start,
right where the agents' ``on_reset`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def empty_ledger() -> dict:
    """A fresh, empty ledger.

    ``progress`` schema (filled by the orchestrator's own responses):
    ``{"current_stage_goal": str, "assignments": <copy of directives>,
    "issued_at_step": int, "expected_signal": str}``.
    """
    return {"task_facts": [], "progress": None, "stall_counter": 0}


@dataclass
class OrchestratorState:
    # {"task_facts": [str], "progress": dict|None, "stall_counter": int}
    ledger: dict = field(default_factory=empty_ledger)
    # agent_id ("agent_N") -> {"comm_target": str, "help": str}
    directives: dict = field(default_factory=dict)
    last_call_step: int = -1
    # Event dicts (see orchestrator.events) accumulated since the last call.
    event_buffer: list = field(default_factory=list)
    call_count: int = 0
    # Calls whose response failed validation even after the single retry;
    # the previous ledger/directives were kept unchanged for these.
    failed_calls: int = 0

    def reset(self, keep_ledger: bool = False) -> None:
        """Reset at episode start.

        ``keep_ledger=False`` (the task variant): fresh empty state — the
        within-episode memory horizon is the experimental point.

        ``keep_ledger=True`` (social/plan variants): the ledger AND standing
        directives survive the episode boundary, matching W(t)'s horizon
        (CustomAgent.on_reset likewise preserves the Hebbian bonds and the
        SocialModule's last_thought). The episode-clock state — event buffer,
        last_call_step, counters — still resets either way.
        """
        if not keep_ledger:
            self.ledger = empty_ledger()
            self.directives = {}
        self.last_call_step = -1
        self.event_buffer = []
        self.call_count = 0
        self.failed_calls = 0

    def add_event(self, event: dict) -> None:
        self.event_buffer.append(event)

    def apply_success(
        self,
        ledger: dict,
        directives: dict,
        t: int,
        max_task_facts: int,
    ) -> None:
        """Install a validated response: replace ledger (facts cap enforced,
        most recent kept) + directives, advance the call clock, clear the
        event buffer.

        The capped list is ``task_facts`` (task variant) or ``notes``
        (social/plan variants) — whichever the validated ledger carries."""
        list_key = "notes" if "notes" in ledger else "task_facts"
        facts = list(ledger.get(list_key) or [])
        if max_task_facts > 0 and len(facts) > max_task_facts:
            facts = facts[-max_task_facts:]  # FIFO eviction: keep most recent
        self.ledger = {
            list_key: facts,
            "progress": ledger.get("progress"),
            "stall_counter": ledger.get("stall_counter", 0),
        }
        self.directives = dict(directives)
        self.last_call_step = t
        self.event_buffer = []
        self.call_count += 1

    def record_failure(self, t: int) -> None:
        """A call whose response never validated: keep the previous ledger
        and directives untouched, and advance the call clock so the failed
        call is not re-triggered every single step. The event buffer is
        deliberately KEPT — those events were wasted on the failed call, so
        the next successful call should still get to see them."""
        self.failed_calls += 1
        self.last_call_step = t
        self.call_count += 1
