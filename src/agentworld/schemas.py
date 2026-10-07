"""Response schema for the AgentWorld action call (flat: ``load_json`` cannot
parse nested objects reliably, see the orchestrator notes)."""

from __future__ import annotations

from pydantic import BaseModel


class AWAgentResponse(BaseModel):
    thoughts: str
    action: str
    dm_target: str = ""
    dm_text: str = ""
    board_post: str = ""
    post_kind: str = "none"
