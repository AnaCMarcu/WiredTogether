"""Prompt assembly for the orchestrator call.

One template (prompts/orchestrator.txt, loaded at import like the social
module's), filled ONCE here with ``str.format`` — the filled text goes to the
client verbatim, so JSON braces inside substituted VALUES need no escaping
(only the template file itself uses ``{{ }}`` for literals).

Deliberately contains NO chamber strategy and NO pairing heuristics: the
orchestrator must infer strategy itself, and the ledger is how its
conclusions persist within the episode.
"""

from __future__ import annotations

import json
import os

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

with open(os.path.join(_PROMPT_DIR, "orchestrator.txt"), "r",
          encoding="utf-8") as f:
    _orchestrator_prompt = f.read()


def _ledger_json(ledger: dict) -> str:
    if not ledger or (not ledger.get("task_facts")
                      and ledger.get("progress") is None
                      and not ledger.get("stall_counter")):
        return "(none yet — this is your first call this episode)"
    return json.dumps(ledger, indent=1)


def _directives_json(directives: dict) -> str:
    if not directives:
        return "(none yet)"
    return json.dumps(directives, indent=1)


def format_prompt(
    *,
    n_agents: int,
    agent_names: list,
    last_call_step: int,
    current_step: int,
    digest: str,
    ledger: dict,
    directives: dict,
    stall_threshold: int,
    map_text_fallback: str = "",
) -> str:
    """Fill the orchestrator template.

    ``map_text_fallback`` is empty when a map image is attached to the call;
    otherwise it carries the text world-state block from
    :func:`orchestrator.map_render.render_map_text`.
    """
    return _orchestrator_prompt.format(
        n_agents=n_agents,
        agent_names=", ".join(agent_names),
        map_text_fallback_block_if_no_image=map_text_fallback or "",
        last_call_step=(last_call_step if last_call_step >= 0
                        else "episode start"),
        current_step=current_step,
        digest=digest,
        ledger_json=_ledger_json(ledger),
        directives_json=_directives_json(directives),
        stall_threshold=stall_threshold,
    )
