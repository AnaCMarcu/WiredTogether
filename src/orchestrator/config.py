"""Configuration for the centralized task-ledger orchestrator (O2 baseline).

All fields have defaults so that ``OrchestratorConfig()`` produces a disabled
no-op instance — mirroring the HebbianConfig / RLConfig pattern. The
orchestrator condition runs INSTEAD of the Hebbian coupling (the two are
mutually exclusive at startup validation), but shares its cadence default
with the social module so the two conditions are matched on call frequency.
"""

from dataclasses import dataclass

# Variant groups. Every main-loop gate tests membership in one of these,
# never ``== "<variant>"``, so a new variant cannot silently fall into the
# wrong path (tests/test_orchestrator_hmas2.py forbids literal gates).
#: Per-step controller variants (tick() / directive_text() /
#: assigned_objective() interface) — as opposed to the cadence-based
#: orchestrate() path of task/social/plan.
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

    # ── Master switch for the O2 condition ──
    enabled: bool = False

    # ── Variant ──
    # "task":   the original O2 task-ledger orchestrator — world-state map +
    #           event digest in, comm_target/help directives out, relational
    #           content excluded from the ledger, ledger reset per episode.
    # "social": centralized social deliberation — the orchestrator replaces
    #           the per-agent SocialModule: it sees ONLY the signals the
    #           Hebbian rule consumes (pair co-presence, message counts,
    #           co-rewards, chambers; no map, no message text) and emits a
    #           SocialThought per agent (ask_target/ask_message/respond_to),
    #           rendered in the SocialModule's exact directive format.
    #           Relational notes allowed; ledger persists across episodes
    #           (matching W(t)'s horizon).
    # "plan":   "social" plus each agent's auto-curriculum task in view and a
    #           per-agent plan_note delivered to that agent's curriculum the
    #           next time it generates a task. Upper baseline (exceeds the
    #           Hebbian condition's information and influence by design).
    # "villager": VillagerAgent-style centralized DAG orchestration — a
    #           decomposer LLM proposes milestone-verified subtasks into a
    #           dependency DAG, an allocator LLM assigns ready tasks to free
    #           agents (HARD assignment: the agent's curriculum is
    #           constrained to the objective and replans on reassignment).
    #           No communication routing (faithful to the published method);
    #           fresh DAG every episode. Hard-orchestration upper baseline.
    # "hmas2":  the HMAS-2 protocol (Chen et al., ICRA 2024) adapted to WIRE —
    #           every step a central planner assigns subtasks, each assigned
    #           agent checks its own assignment ("I Agree" or an objection),
    #           the planner revises (<= 3 rounds). Hub-and-spoke: agents
    #           cannot message each other; their messages are reports to the
    #           planner, which sends each agent only the teammate information
    #           it chooses. The agent's curriculum task is pinned to its
    #           assignment. See orchestrator/hmas2.py.
    variant: str = "task"

    # ── Coupling mode ──
    # "advisory": directives rendered as text in the action prompt (same
    #             {social_directive} slot the social module uses); the action
    #             LLM may still ignore them.
    # "bias":     directive's comm_target additionally overrides the emitted
    #             communication_target at the message-routing site —
    #             mirroring the social-module bias coupling exactly.
    mode: str = "advisory"

    # ── Scheduled-call cadence (steps between calls) ──
    # 8 = the social module's T_soc default (--social-interval), so the
    # orchestrator and the Hebbian/social condition are matched on how often
    # their social-reasoning LLM step runs.
    cadence: int = 8

    # ── Also call on milestone / chamber-change / death events ──
    event_triggers: bool = True

    # ── Replan when the ledger's stall_counter exceeds this ──
    stall_threshold: int = 2

    # ── Ledger task-facts cap (FIFO eviction — most recent kept) ──
    max_task_facts: int = 15

    # ── Events included in the since-last-call digest ──
    max_digest_events: int = 30

    # ── Attach the schematic top-down map image (multimodal call) ──
    # Falls back to a text block when False or when the client lacks vision.
    use_map_image: bool = True

    # ── LLM override (None => reuse the agents' backbone/client) ──
    model: str | None = None

    # ── Subdirectory of the run dir for orchestrator logs/maps ──
    log_dir_name: str = "orchestrator"

    # ── Villager variant only ──
    # A running DAG node fails after this many steps without one of its
    # milestones firing (backstop for unverifiable progress).
    node_timeout_steps: int = 60
    # Cap on open+running DAG nodes; 0 = auto (2 × num_agents, resolved in
    # the controller since the config does not know N).
    max_open_tasks: int = 0
    # Minimum steps between decomposer calls (defaults to the cadence
    # default, in the matched-call-frequency spirit). Also the cooldown
    # after a failed allocator call.
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

    VALID_MODES = ("advisory", "bias")
    VALID_VARIANTS = ("task", "social", "plan", "villager", "hmas2")

    def validate(self) -> None:
        """Raise ValueError on an invalid mode/variant. Cheap, call at startup."""
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"orchestrator.mode must be one of {self.VALID_MODES}, "
                f"got {self.mode!r}"
            )
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"orchestrator.variant must be one of {self.VALID_VARIANTS}, "
                f"got {self.variant!r}"
            )
        if self.variant in CONTROLLER_VARIANTS and self.mode == "bias":
            raise ValueError(
                f"the {self.variant} variant issues task assignments, not "
                "comm directives; there is no comm_target to bias — use "
                "advisory"
            )
        if self.variant == "villager":
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
