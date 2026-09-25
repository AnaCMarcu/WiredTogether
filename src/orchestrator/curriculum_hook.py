"""Curriculum-prompt hook for the orchestrator.

The assigned objective reaches an agent's auto-curriculum by appending this
suffix — which carries the ``{assigned_objective}`` placeholder — to the
curriculum's USER template (curriculum_info.txt, the one llm_call
``.format``s; the role/system prompt is passed unformatted and is left
alone). The suffix is applied at build_agents time and ONLY when the
orchestrator is on, so every other configuration's curriculum prompt stays
byte-identical. Kept in its own dependency-free module so the byte-identity
property is unit-testable without importing the runtime stack.
"""

# The villager assignment is HARD: VillagerAgent's controller decides WHO does WHAT and
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
