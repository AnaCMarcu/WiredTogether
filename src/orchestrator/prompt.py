"""Prompt assembly for the centralised orchestrator (VillagerAgent-style).

Two templates, loaded at import: ``prompts/orchestrator_decompose.txt`` (the
decomposer, which proposes milestone-verified subtasks into the DAG) and
``prompts/orchestrator_allocate.txt`` (the allocator, which binds free agents
to ready subtasks). Each is filled ONCE here with ``str.format`` — the filled
text goes to the client verbatim, so JSON braces inside substituted VALUES
need no escaping (only the template files use ``{{ }}`` for literals).

Deliberately contains NO chamber strategy and NO pairing heuristics: the
orchestrator must infer strategy itself from the chamber facts and the
milestone catalog.
"""

from __future__ import annotations

import os

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def _load(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


_orchestrator_decompose_prompt = _load("orchestrator_decompose.txt")
_orchestrator_allocate_prompt = _load("orchestrator_allocate.txt")


# ── Decompose / allocate ─────────────────────────────────────────────────

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
