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
