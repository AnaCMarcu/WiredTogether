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

    VALID_MODES = ("advisory", "bias")

    def validate(self) -> None:
        """Raise ValueError on an invalid mode. Cheap, call at startup."""
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"orchestrator.mode must be one of {self.VALID_MODES}, "
                f"got {self.mode!r}"
            )
