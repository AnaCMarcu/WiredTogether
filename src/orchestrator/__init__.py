"""Centralized task-ledger orchestrator (the O2 baseline condition).

A non-embodied coordinator called once every ``cadence`` steps (and on
milestone / chamber-change / death events) that observes a schematic
top-down map + a text digest of events since its last call + its own
persistent within-episode ledger, and emits per-agent standing directives
(``comm_target`` + ``help``). Directives are injected into the same
``{social_directive}`` action-prompt slot the social module uses (advisory
mode) and can optionally override ``communication_target`` at the routing
site (bias mode) — mirroring the Hebbian couplings exactly.

Runs INSTEAD of the Hebbian coupling (mutually exclusive at startup); the
ledger resets at every episode start, in deliberate contrast with W(t).

Kept import-light: the training loop imports ``orchestrator.core`` (and
``orchestrator.logging``) explicitly; this package root only exposes the
dependency-free config/state/events pieces.
"""

from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState

__all__ = ["OrchestratorConfig", "OrchestratorState"]
