"""Configuration for the centralized task-ledger orchestrator (O2 baseline).

All fields have defaults so that ``OrchestratorConfig()`` produces a disabled
no-op instance — mirroring the HebbianConfig / RLConfig pattern. The
orchestrator condition runs INSTEAD of the Hebbian coupling (the two are
mutually exclusive at startup validation), but shares its cadence default
with the social module so the two conditions are matched on call frequency.
"""

from dataclasses import dataclass


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

    VALID_MODES = ("advisory", "bias")
    VALID_VARIANTS = ("task", "social", "plan", "villager")

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
        if self.variant == "villager" and self.mode == "bias":
            raise ValueError(
                "the villager variant issues task assignments, not comm "
                "directives; there is no comm_target to bias — use advisory"
            )
        if self.variant == "villager":
            if self.node_timeout_steps <= 0:
                raise ValueError("node_timeout_steps must be positive")
            if self.decompose_min_interval < 1:
                raise ValueError("decompose_min_interval must be >= 1")
