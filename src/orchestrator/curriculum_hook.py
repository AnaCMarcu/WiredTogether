"""Curriculum-prompt hook for the plan variant (O-plan).

The orchestrator's per-agent plan_note reaches an agent's auto-curriculum by
appending this suffix — which carries the ``{team_plan_note}`` placeholder —
to the curriculum's USER template (curriculum_info.txt, the one llm_call
``.format``s; the role/system prompt is passed unformatted and is left
alone). The suffix is applied at build_agents time and ONLY when the plan
variant is on, so every other configuration's curriculum prompt stays
byte-identical. Kept in its own dependency-free module so the byte-identity
property is unit-testable without importing the runtime stack.
"""

TEAM_PLAN_NOTE_PLACEHOLDER = "{team_plan_note}"

PLAN_SUFFIX = (
    "\n\nTEAM COORDINATION NOTE (from the non-embodied team coordinator — it"
    " can see every agent's current task; this is ADVICE for your task"
    " choice, weigh it against your own observations and ignore it if it"
    " conflicts with what you see):\n"
    "{team_plan_note}\n"
)


def apply_plan_suffix(task_info_prompt: str, enabled: bool) -> str:
    """Append the coordination-note block when ``enabled``; identity
    otherwise. Idempotent: never appends twice."""
    if not enabled:
        return task_info_prompt
    if TEAM_PLAN_NOTE_PLACEHOLDER in task_info_prompt:
        return task_info_prompt
    return task_info_prompt + PLAN_SUFFIX


# ── Villager variant ─────────────────────────────────────────────────────
# Unlike PLAN_SUFFIX (advice the curriculum may ignore), the villager
# assignment is HARD: VillagerAgent's controller decides WHO does WHAT and
# the worker is never asked whether it accepts — only HOW remains the
# agent's own. The curriculum keeps generating concrete tasks, but they
# must advance the assigned objective.

ASSIGNED_OBJECTIVE_PLACEHOLDER = "{assigned_objective}"

VILLAGER_SUFFIX = (
    "\n\nTEAM ASSIGNMENT (from the non-embodied team coordinator — it"
    " decomposed the team's goal into subtasks and assigned this one to"
    " YOU; this is not advice):\n"
    "{assigned_objective}\n"
    "The task you choose MUST directly advance this assigned objective."
    " Choose the smallest next task that makes progress on it in your"
    " current situation.\n"
)


def apply_villager_suffix(task_info_prompt: str, enabled: bool) -> str:
    """Append the hard-assignment block when ``enabled``; identity
    otherwise. Idempotent: never appends twice."""
    if not enabled:
        return task_info_prompt
    if ASSIGNED_OBJECTIVE_PLACEHOLDER in task_info_prompt:
        return task_info_prompt
    return task_info_prompt + VILLAGER_SUFFIX
