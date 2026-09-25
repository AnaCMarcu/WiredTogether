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


@dataclass
class OrchestratorConfig:
    """All orchestrator settings.

    When ``enabled=False`` the entire module is a no-op.
    """

    # ── Master switch ──
    enabled: bool = False

    # ── Variant ──
    # Only "villager" is implemented; the field is kept so run configs
    # record which orchestrator produced them.
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

    VALID_VARIANTS = ("villager",)

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
