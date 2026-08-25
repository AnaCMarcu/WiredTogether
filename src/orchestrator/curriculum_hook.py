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
