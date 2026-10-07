"""Configuration for the centralised orchestration baseline.

All fields have defaults so that ``OrchestratorConfig()`` produces a disabled
no-op instance — mirroring the HebbianConfig / RLConfig pattern. The
orchestrator condition runs INSTEAD of the Hebbian coupling (the two are
mutually exclusive at startup validation).

The orchestrator is VillagerAgent-style: a decomposer LLM proposes
milestone-verified subtasks into a dependency DAG, and an allocator LLM
assigns ready subtasks to free agents. Assignments are HARD — the agent's
auto-curriculum is constrained to the objective and replans on reassignment —
while primitive control and communication stay decentralised. The DAG is
rebuilt at the start of every episode.
"""

from dataclasses import dataclass

# Variant groups. Every main-loop gate tests membership in one of these,
# never ``== "<variant>"``, so a new variant cannot silently fall into the
# wrong path (tests/test_orchestrator_hmas2.py forbids literal gates).
#: Per-step controller variants (tick() / directive_text() /
#: assigned_objective() interface).
CONTROLLER_VARIANTS = ("villager", "hmas2")
#: Hub-and-spoke communication: agent messages go to the orchestrator.
HUB_VARIANTS = ("hmas2",)
#: The agent's curriculum task is PINNED to the orchestrator's assignment.
PINNED_TASK_VARIANTS = ("hmas2",)
#: The curriculum keeps choosing tasks, constrained by an
#: {assigned_objective} suffix.
ASSIGNED_OBJECTIVE_VARIANTS = ("villager",)


@dataclass
class OrchestratorConfig:
    """All orchestrator settings.

    When ``enabled=False`` the entire module is a no-op.
    """

    # ── Master switch ──
    enabled: bool = False

    # ── Variant ──
    # "villager": the soft orchestrator described above.
    # "hmas2":    the HARD orchestrator — the HMAS-2 protocol (Chen et al.,
    #             ICRA 2024) adapted to WIRE: every step a central planner
    #             assigns subtasks, each assigned agent checks its own
    #             assignment ("I Agree" or an objection), the planner revises
    #             (<= 3 rounds). Hub-and-spoke: agents cannot message each
    #             other; their messages are reports to the planner, which
    #             sends each agent only the teammate information it chooses.
    #             The agent's curriculum task is pinned to its assignment.
    #             See orchestrator/hmas2.py.
    variant: str = "villager"

    # ── LLM override (None => reuse the agents' backbone/client) ──
    model: str | None = None

    # ── Subdirectory of the run dir for orchestrator logs ──
    log_dir_name: str = "orchestrator"

    # A running DAG node fails after this many steps without one of its
    # milestones firing (backstop for unverifiable progress).
    node_timeout_steps: int = 60
    # Cap on open+running DAG nodes; 0 = auto (2 × num_agents, resolved in
    # the controller since the config does not know N).
    max_open_tasks: int = 0
    # Minimum steps between decomposer calls — T_soc, matched to the social
    # module's deliberation interval (--social-interval, default 8). Also the
    # cooldown after a failed allocator call.
    decompose_min_interval: int = 8

    # ── hmas2 variant only (defaults = the HMAS-2 reference code) ──
    # Check -> revise rounds per step (`while ... count_round_HMAS2 < 3`).
    hmas2_max_rounds: int = 3
    # Syntactic re-prompts per plan (`while iteration_num < 6`).
    hmas2_syntax_retries: int = 6
    # State-action history budget in tokens (`input_prompt_token_limit`,
    # 3000 in BoxNet2's prompt module); the newest pairs are kept.
    hmas2_history_tokens: int = 3000
    # Word cap per coordinator message (fits the 32-token message cap that
    # comm-budget runs also apply to incoming messages).
    hmas2_message_words: int = 24
    # Reports kept per agent between steps (newest first; overflow logged
    # as dropped — a hub-capacity signal).
    hmas2_report_cap: int = 2
    # Generation cap of a local check ("I Agree" or a brief objection).
    hmas2_check_max_tokens: int = 96

    VALID_VARIANTS = ("villager", "hmas2")

    def validate(self) -> None:
        """Raise ValueError on an invalid setting. Cheap, call at startup."""
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"orchestrator.variant must be one of {self.VALID_VARIANTS}, "
                f"got {self.variant!r}"
            )
        if self.node_timeout_steps <= 0:
            raise ValueError("node_timeout_steps must be positive")
        if self.decompose_min_interval < 1:
            raise ValueError("decompose_min_interval must be >= 1")
        if self.variant == "hmas2":
            for name in ("hmas2_max_rounds", "hmas2_syntax_retries",
                         "hmas2_history_tokens", "hmas2_message_words",
                         "hmas2_report_cap", "hmas2_check_max_tokens"):
                if getattr(self, name) < (0 if name == "hmas2_syntax_retries"
                                          else 1):
                    raise ValueError(f"{name} must be positive, got "
                                     f"{getattr(self, name)!r}")
