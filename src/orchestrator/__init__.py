"""Centralised orchestration baseline (VillagerAgent-style).

A non-embodied coordinator that decomposes the team goal into
milestone-verified subtasks held in a dependency DAG and assigns them to
agents (``orchestrator.villager``). Its directive is injected into the same
``{social_directive}`` action-prompt slot the social module uses, and the
assigned objective constrains each agent's auto-curriculum.

Runs INSTEAD of the Hebbian coupling (mutually exclusive at startup); the DAG
resets at every episode start, in deliberate contrast with W(t).

Kept import-light: the training loop imports ``orchestrator.core`` /
``orchestrator.villager`` (and ``orchestrator.logging``) explicitly; this
package root only exposes the dependency-free config/state pieces.
"""

from orchestrator.config import OrchestratorConfig
from orchestrator.state import OrchestratorState

__all__ = ["OrchestratorConfig", "OrchestratorState"]
