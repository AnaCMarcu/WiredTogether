"""Per-episode state shared between the training loop and the orchestrator.

The memory horizon is the experimental point of the baseline: the
orchestrator holds no state across episodes, in contrast with the Hebbian
W(t), which persists. ``reset()`` must therefore be called from the training
loop at every episode start, right where the agents' ``on_reset`` runs.

The loop appends events (see ``orchestrator.events``) to ``event_buffer``;
the villager controller drains them every tick and mirrors its current
assignments into ``directives`` for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrchestratorState:
    # agent_id ("agent_N") -> {"task_id", "description", "role"} — a mirror
    # of the DAG's running assignments; the DAG stays authoritative.
    directives: dict = field(default_factory=dict)
    # Event dicts (see orchestrator.events) accumulated since the last tick.
    event_buffer: list = field(default_factory=list)

    def reset(self) -> None:
        """Fresh state at episode start."""
        self.directives = {}
        self.event_buffer = []

    def add_event(self, event: dict) -> None:
        self.event_buffer.append(event)
