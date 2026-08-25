"""Event accumulation + plain-text digest for the orchestrator.

Events are plain dicts appended each env step by the training loop, built
from data the loop already produces (message routing metadata, the drained
milestone_events.jsonl / death_events.jsonl records, and the per-agent
chamber tracking). No new instrumentation of the environment is needed.

Event shapes:
  message        {"type": "message", "t": int, "sender": str, "target": str,
                  "text": str}                       (text truncated ~120 chars)
  milestone      {"type": "milestone", "t": int, "id": str,
                  "contributors": [str]}
  death          {"type": "death", "t": int, "agent": str}
  chamber_change {"type": "chamber_change", "t": int, "chamber": str}
"""

from __future__ import annotations

MESSAGE_TEXT_MAX_CHARS = 120

#: Event types that count as call triggers when cfg.event_triggers is on.
TRIGGER_TYPES = ("milestone", "chamber_change", "death")


def message_event(t: int, sender: str, target: str, text: str) -> dict:
    text = str(text or "")
    if len(text) > MESSAGE_TEXT_MAX_CHARS:
        text = text[: MESSAGE_TEXT_MAX_CHARS - 1] + "…"
    return {"type": "message", "t": t, "sender": sender, "target": target,
            "text": text}


def milestone_event(t: int, milestone_id: str, contributors: list) -> dict:
    return {"type": "milestone", "t": t, "id": str(milestone_id or "?"),
            "contributors": [str(c) for c in (contributors or [])]}


def death_event(t: int, agent: str) -> dict:
    return {"type": "death", "t": t, "agent": str(agent or "?")}


def chamber_change_event(t: int, chamber: str) -> dict:
    return {"type": "chamber_change", "t": t, "chamber": str(chamber or "?")}


def _format_event(ev: dict) -> str:
    kind = ev.get("type")
    t = ev.get("t", "?")
    if kind == "message":
        return (f"t={t} msg {ev.get('sender', '?')}->{ev.get('target', '?')}: "
                f"\"{ev.get('text', '')}\"")
    if kind == "milestone":
        contribs = ", ".join(ev.get("contributors") or []) or "?"
        return f"t={t} MILESTONE {ev.get('id', '?')} by [{contribs}]"
    if kind == "death":
        return f"t={t} DEATH {ev.get('agent', '?')}"
    if kind == "chamber_change":
        return f"t={t} CHAMBER -> {ev.get('chamber', '?')}"
    return f"t={t} {kind}: {ev}"


def build_digest(events: list, max_events: int) -> str:
    """Render events as a chronological plain-text block.

    Keeps the LAST ``max_events`` entries; when truncated, a
    ``(showing last K of M events)`` banner is prepended so the orchestrator
    knows its view is partial. Returns a friendly placeholder when empty.
    """
    if not events:
        return "(no events since your last call)"
    total = len(events)
    shown = events[-max_events:] if (max_events and total > max_events) else events
    lines = [_format_event(ev) for ev in shown]
    if len(shown) < total:
        lines.insert(0, f"(showing last {len(shown)} of {total} events)")
    return "\n".join(lines)


# ── Pair-activity accumulator (social/plan variants) ─────────────────────
# The information-matched variants must see exactly what the Hebbian update
# rule consumes — pairwise co-presence, directed message COUNTS (never the
# text), bondable co-reward, chambers — aggregated between calls the same way
# the rule integrates per-step signals into W. Fed once per env step from the
# loop's Phase-2 scope (positions / comm_events / _bond_rewards / _chambers
# are all in hand there); rendered + cleared at each successful call.

class PairAccumulator:
    def __init__(self, num_agents: int, radius: float = 5.0):
        self.num_agents = num_agents
        # Same default as HebbianConfig.interaction_radius; the loop passes
        # args.hebbian_radius so a radius ablation stays matched by
        # construction.
        self.radius = float(radius)
        self.clear()

    def clear(self) -> None:
        self.steps = 0
        self.copresence: dict = {}     # (i, j) i<j -> co-present step count
        self.messages: dict = {}       # (sender, receiver) -> count
        self.coreward: dict = {}       # (i, j) i<j -> bondable reward earned
                                       #   by either while co-present
        self.reward_totals = [0.0] * self.num_agents
        self.last_chambers = ["?"] * self.num_agents

    @staticmethod
    def _dist(a, b):
        try:
            return sum((float(a[k]) - float(b[k])) ** 2 for k in range(3)) ** 0.5
        except (TypeError, ValueError, IndexError):
            return None

    def note_step(self, positions, comm_events, bond_rewards, chambers) -> None:
        """Record one env step. ``positions`` is a list of (x,y,z)|None,
        ``comm_events`` a list of (sender_idx, recv_idx), ``bond_rewards`` the
        per-agent bondable reward this step (the same stream
        hebbian_graph.update receives), ``chambers`` per-agent chamber ints
        (1..5, 0 unknown) or names."""
        n = self.num_agents
        self.steps += 1
        for i in range(n):
            try:
                self.reward_totals[i] += float(bond_rewards[i])
            except (TypeError, ValueError, IndexError):
                pass
            try:
                ch = chambers[i]
                self.last_chambers[i] = (
                    f"ch{ch}" if isinstance(ch, int) and ch > 0 else str(ch or "?")
                )
            except (TypeError, IndexError):
                pass
        for i in range(n):
            for j in range(i + 1, n):
                if positions is None:
                    continue
                try:
                    pi, pj = positions[i], positions[j]
                except (TypeError, IndexError):
                    continue
                if pi is None or pj is None:
                    continue
                d = self._dist(pi, pj)
                if d is None or d > self.radius:
                    continue
                self.copresence[(i, j)] = self.copresence.get((i, j), 0) + 1
                try:
                    rw = float(bond_rewards[i]) + float(bond_rewards[j])
                except (TypeError, ValueError, IndexError):
                    rw = 0.0
                self.coreward[(i, j)] = self.coreward.get((i, j), 0.0) + rw
        for ev in comm_events or []:
            try:
                s, r = int(ev[0]), int(ev[1])
            except (TypeError, ValueError, IndexError):
                continue
            self.messages[(s, r)] = self.messages.get((s, r), 0) + 1

    def render(self) -> str:
        """The pair-activity digest block for the social/plan prompts."""
        if self.steps == 0:
            return "(no steps recorded since your last call)"
        n = self.num_agents
        lines = []
        for i in range(n):
            for j in range(i + 1, n):
                co = self.copresence.get((i, j), 0)
                m_ij = self.messages.get((i, j), 0)
                m_ji = self.messages.get((j, i), 0)
                rw = self.coreward.get((i, j), 0.0)
                lines.append(
                    f"  agent_{i}~agent_{j}: co-present {co}/{self.steps} steps"
                    f", msgs {m_ij} (agent_{i}->agent_{j})"
                    f" / {m_ji} (agent_{j}->agent_{i})"
                    f", reward-while-together {rw:+.1f}"
                )
        lines.append("  Per-agent reward: " + ", ".join(
            f"agent_{i} {self.reward_totals[i]:+.1f}" for i in range(n)))
        lines.append("  Chambers now: " + "  ".join(
            f"agent_{i}={self.last_chambers[i]}" for i in range(n)))
        return "\n".join(lines)
