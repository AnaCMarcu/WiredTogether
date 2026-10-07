"""Replay videos rendered from a run's logs (no browser, no game server).

Reads ``episode_*/replay/state.jsonl`` written by the scheduler and draws each
round on the real Kaetram map (when an AgentWorld checkout is available) or a
plain grid: agents, mobs, direct-message speech bubbles with an arrow to the
receiver, board posts, transfer arcs, kills, and a side panel with the
non-global chat log, team progress and the bond graph.

    python -m agentworld.replay <episode_dir> [--team t00 | --agent 3 | --world]
"""

from agentworld.replay.render import ReplayRenderer, RenderOptions, load_states

__all__ = ["ReplayRenderer", "RenderOptions", "load_states"]
