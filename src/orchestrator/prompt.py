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

def _load(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


_orchestrator_prompt = _load("orchestrator.txt")
_orchestrator_social_prompt = _load("orchestrator_social.txt")
_orchestrator_plan_prompt = _load("orchestrator_plan.txt")
_orchestrator_decompose_prompt = _load("orchestrator_decompose.txt")
_orchestrator_allocate_prompt = _load("orchestrator_allocate.txt")


def _ledger_json(ledger: dict) -> str:
    if not ledger or (not ledger.get("task_facts")
                      and ledger.get("progress") is None
                      and not ledger.get("stall_counter")):
        return "(none yet — this is your first call this episode)"
    return json.dumps(ledger, indent=1)


def _build_directives_example(agent_names: list) -> str:
    """Render the response-format example with the REAL living agent names.

    A fixed two-agent example was copied LITERALLY by the backbone: in the
    second smoke run it emitted directives for agent_0/agent_1 only and
    omitted agent_2, failing validation on 11 of the first 11 calls (it
    recovered only once a successful 3-agent block appeared in the standing
    directives for it to copy instead). Generating the example from the
    living set makes the required arity self-evident and scales to the
    agent-count sweep (N=2..9) without another prompt edit.

    comm_target cycles to the next agent, so it is never the agent itself.
    """
    n = len(agent_names)
    if n == 0:
        return '    "agent_0": {"comm_target": "agent_1", "help": "..."}'
    lines = []
    for i, name in enumerate(agent_names):
        target = agent_names[(i + 1) % n] if n > 1 else name
        lines.append(
            f'    "{name}": {{"comm_target": "{target}", "help": "..."}}'
        )
    return ",\n".join(lines)


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
        directives_example=_build_directives_example(list(agent_names)),
        stall_threshold=stall_threshold,
    )


# ── Social / plan variants ───────────────────────────────────────────────

def _notes_ledger_json(ledger: dict) -> str:
    if not ledger or (not ledger.get("notes")
                      and not ledger.get("stall_counter")):
        return "(none yet — this is your first call)"
    view = {"notes": ledger.get("notes") or [],
            "stall_counter": ledger.get("stall_counter", 0)}
    return json.dumps(view, indent=1)


def _build_social_directives_example(agent_names: list,
                                     with_plan: bool) -> str:
    """SocialThought-shaped example generated from the REAL living agents —
    same lesson as _build_directives_example: this backbone copies example
    arity literally, so the example must always show every living agent (and
    an open-ended shape everywhere a list can grow)."""
    n = len(agent_names)
    if n == 0:
        agent_names, n = ["agent_0", "agent_1"], 2
    lines = []
    plan_part = ', "plan_note": ""' if with_plan else ""
    for i, name in enumerate(agent_names):
        target = agent_names[(i + 1) % n] if n > 1 else "null"
        if i == n - 1 and n > 1:
            # Show the stay-focused shape too, so null is visibly legal.
            entry = (f'    "{name}": {{"reasoning": "...", '
                     f'"ask_target": null, "ask_message": null, '
                     f'"respond_to": ["{agent_names[0]}"], '
                     f'"pair_notes": {{}}{plan_part}}}')
        else:
            entry = (f'    "{name}": {{"reasoning": "...", '
                     f'"ask_target": "{target}", "ask_message": "...", '
                     f'"respond_to": [], '
                     f'"pair_notes": {{"{target}": "..."}}{plan_part}}}')
        lines.append(entry)
    return ",\n".join(lines)


def format_social_prompt(
    *,
    n_agents: int,
    agent_names: list,
    last_call_step: int,
    current_step: int,
    pair_digest: str,
    ledger: dict,
    directives: dict,
    task_table: str = None,
) -> str:
    """Fill the social (task_table=None) or plan template."""
    template = (_orchestrator_plan_prompt if task_table is not None
                else _orchestrator_social_prompt)
    kwargs = dict(
        n_agents=n_agents,
        agent_names=", ".join(agent_names),
        last_call_step=(last_call_step if last_call_step >= 0
                        else "episode start"),
        current_step=current_step,
        pair_digest=pair_digest,
        ledger_json=_notes_ledger_json(ledger),
        directives_json=_directives_json(directives),
        directives_example=_build_social_directives_example(
            list(agent_names), with_plan=task_table is not None),
    )
    if task_table is not None:
        kwargs["task_table"] = task_table
    return template.format(**kwargs)


# ── Villager variant (decompose / allocate) ──────────────────────────────

def _build_tasks_example(agent_names: list, milestone_ids: list,
                         open_slots: int) -> str:
    """Decomposer response example GENERATED from the real living agents and
    real catalog milestone ids (the example-arity lesson: this backbone
    copies example structure literally, so examples must only ever show
    names/ids that are actually valid)."""
    names = list(agent_names) or ["agent_0", "agent_1"]
    ids = list(milestone_ids) or ["m1_move_5", "m4_dig_5_wood"]
    n_tasks = max(1, min(2, open_slots))
    lines = [
        ('    {{"id": "task_a", "description": "...", '
         '"milestones": ["{m}"], "required": [], '
         '"candidates": ["{c}"], "min_agents": 1}}').format(
            m=ids[0], c=names[0])
    ]
    if n_tasks > 1:
        lines.append(
            ('    {{"id": "task_b", "description": "...", '
             '"milestones": ["{m}"], "required": ["task_a"], '
             '"candidates": [], "min_agents": 1}}').format(
                m=ids[1 % len(ids)])
        )
    return ",\n".join(lines)


def _build_assignments_example(free_agents: list,
                               ready_task_ids: list) -> str:
    """Allocator response example with one entry per REAL free agent,
    cycling over REAL ready task ids (arity lesson again)."""
    names = list(free_agents) or ["agent_0"]
    ids = list(ready_task_ids) or ["task_a"]
    lines = []
    for i, name in enumerate(names):
        lines.append(
            ('    {{"agent": "{a}", "task_id": "{t}", '
             '"role": "..."}}').format(a=name, t=ids[i % len(ids)])
        )
    return ",\n".join(lines)


def format_decompose_prompt(
    *,
    n_agents: int,
    agent_names: list,
    current_step: int,
    chamber_facts: str,
    milestone_catalog: str,
    env_state_text: str,
    task_table: str,
    dag_summary: str,
    open_slots: int,
    example_milestones: list,
) -> str:
    return _orchestrator_decompose_prompt.format(
        n_agents=n_agents,
        agent_names=", ".join(agent_names),
        current_step=current_step,
        chamber_facts=chamber_facts,
        milestone_catalog=milestone_catalog,
        env_state_text=env_state_text,
        task_table=task_table,
        dag_summary=dag_summary,
        open_slots=open_slots,
        tasks_example=_build_tasks_example(
            list(agent_names), list(example_milestones), open_slots),
    )


def format_allocate_prompt(
    *,
    current_step: int,
    ready_tasks_block: str,
    free_agents_block: str,
    dag_summary: str,
    free_agents: list,
    ready_task_ids: list,
) -> str:
    return _orchestrator_allocate_prompt.format(
        current_step=current_step,
        ready_tasks_block=ready_tasks_block,
        free_agents_block=free_agents_block,
        dag_summary=dag_summary,
        assignments_example=_build_assignments_example(
            list(free_agents), list(ready_task_ids)),
    )
