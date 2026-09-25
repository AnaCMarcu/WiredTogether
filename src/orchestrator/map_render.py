"""Text rendering of the WIRE world state for the orchestrator's prompts.

``env_state`` is a plain dict snapshot (see orchestrator.core.collect_env_state):
  {
    "step": int,
    "agents": {"agent_0": {"pos": (x,y,z)|None, "chamber": str|None,
                            "hp": float|None, "alive": bool}, ...},
    "doors": {"door1": bool, "door2": bool, "door3": bool, "door4": bool},
    "anvils": [{"kind": str, "hp": int}, ...],       # unbroken only
    "cell_doors_open": [int, ...],
    "recent_messages": [(sender, target), ...],       # optional, last few
  }
"""

from __future__ import annotations


def render_map_text(env_state: dict, num_agents: int = 3) -> str:
    """Agent positions + chamber + fixture states as a text block for the
    decomposer prompt."""
    lines = ["WORLD STATE (text map fallback):"]
    for name in sorted((env_state.get("agents") or {}).keys()):
        info = env_state["agents"][name]
        pos = info.get("pos")
        pos_str = (f"({pos[0]:.0f}, {pos[2]:.0f})" if pos is not None
                   else "(unknown)")
        hp = info.get("hp")
        hp_str = f" hp={hp:.0f}" if hp is not None else ""
        alive_str = "" if info.get("alive", True) else " DEAD"
        lines.append(f"  {name}: pos {pos_str} in "
                     f"{info.get('chamber') or '?'}{hp_str}{alive_str}")
    doors = env_state.get("doors") or {}
    lines.append("  Doors: " + ", ".join(
        f"{d.upper()}={'OPEN' if doors.get(d) else 'closed'}"
        for d in ("door1", "door2", "door3", "door4")
    ))
    anvils = env_state.get("anvils") or []
    if anvils:
        lines.append("  Anvils (unbroken): " + ", ".join(
            f"{a.get('kind')}(hp={a.get('hp')})" for a in anvils))
    cells = env_state.get("cell_doors_open") or []
    if cells:
        lines.append("  Ch3 cells open: "
                     + ", ".join(f"cell {c}" for c in sorted(cells)))
    return "\n".join(lines)
