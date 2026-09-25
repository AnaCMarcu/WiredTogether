"""Events the training loop feeds to the orchestrator.

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
