"""AgentWorld (Kaetram) harness for the MindForge + Hebbian stack.

Drives an AgentWorld game server over its /ai/* HTTP API with simultaneous
decision rounds, instead of AgentWorld's serial round-robin runner. The game
logic of every tool is AgentWorld's own ``KaetramGameTools`` (loaded unmodified
from a pinned checkout, see :mod:`agentworld.vendor`), and trajectories are
written in AgentWorld's schema so its per-task verifiers score them unchanged.

Modules that need autogen/chromadb (the MindForge agent) are kept out of this
package's import path so the harness core stays importable without them.
"""
