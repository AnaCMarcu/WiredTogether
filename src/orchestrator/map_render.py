"""Schematic top-down map of the five-chambers world for the orchestrator.

Pure matplotlib (headless Agg) over LOGGED state — no game-engine rendering.
The static geometry mirrors src/marl_craftium/.../five_chambers/config.lua
(the 3-agent legacy layout; Ch3 width scales as 4*N+1 like the Lua side).
The PNG saved per call under the run's orchestrator log dir doubles as an
audit artifact of exactly what the orchestrator saw.

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Static geometry (mirrors config.lua) ────────────────────────────────
CH1 = dict(x0=0, x1=15, z0=0, z1=15)
CH2 = dict(x0=2, x1=10, z0=17, z1=25)
CH3_CELL_Z0, CH3_CELL_Z1 = 28, 30
CH3_COMMUNAL_Z0, CH3_COMMUNAL_Z1 = 32, 44
CH3_NORTH_WALL_Z = 45
CH4 = dict(x0=1, x1=11, z0=47, z1=57)
CH5 = dict(x0=2, x1=10, z0=59, z1=67)

DOOR1 = (6, 16)   # Ch1 north wall
DOOR2 = (6, 26)   # Ch2 north wall
DOOR4 = (6, 58)   # Ch4 south wall (Ch4→Ch5 door sits at z=58)
ANVIL_POSITIONS = {"sword": (6, 19), "chestplate": (6, 22)}

AGENT_COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange",
                "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def _ch3_bounds(num_agents: int) -> dict:
    # Ch3 width = 4*N+1 blocks; X: 0..4N (config.lua).
    return dict(x0=0, x1=4 * num_agents, z0=CH3_CELL_Z0 - 1,
                z1=CH3_NORTH_WALL_Z)


def _door3_x(num_agents: int) -> int:
    return min(2 * num_agents, 10)


def _draw_chamber(ax, bounds: dict, label: str) -> None:
    ax.add_patch(Rectangle(
        (bounds["x0"] - 0.5, bounds["z0"] - 0.5),
        bounds["x1"] - bounds["x0"] + 1,
        bounds["z1"] - bounds["z0"] + 1,
        fill=False, edgecolor="black", linewidth=1.2,
    ))
    ax.text(bounds["x1"] + 1.0, (bounds["z0"] + bounds["z1"]) / 2, label,
            fontsize=10, fontweight="bold", va="center")


def _draw_door(ax, x: float, z: float, is_open: bool, name: str) -> None:
    color = "limegreen" if is_open else "crimson"
    ax.plot([x - 1, x + 1], [z, z], color=color, linewidth=4,
            solid_capstyle="butt", zorder=3)
    ax.annotate(f"{name} {'OPEN' if is_open else 'closed'}", (x + 1.3, z),
                fontsize=6, va="center", color=color)


def render_map(env_state: dict, out_path: str, num_agents: int = 3):
    """Render the schematic map PNG. Returns ``out_path`` (str) on success.

    Never raises on drawing problems the caller can't act on — the caller
    falls back to the text block when this returns None.
    """
    try:
        fig, ax = plt.subplots(figsize=(4.5, 9), dpi=110)
        ch3 = _ch3_bounds(num_agents)
        for bounds, label in ((CH1, "Ch1"), (CH2, "Ch2"), (ch3, "Ch3"),
                              (CH4, "Ch4"), (CH5, "Ch5")):
            _draw_chamber(ax, bounds, label)
        # Ch3 internals: isolation cells (south strip) vs communal room.
        ax.plot([ch3["x0"] - 0.5, ch3["x1"] + 0.5],
                [CH3_CELL_Z1 + 1, CH3_CELL_Z1 + 1],
                color="gray", linewidth=0.8, linestyle="--")
        cell_open = set(env_state.get("cell_doors_open") or [])
        for i in range(num_agents):
            cx = 4 * i + 2  # cell centres spread across the 4N+1 width
            mark = "○" if i in cell_open else "●"
            ax.text(cx, CH3_CELL_Z0 + 1, f"{mark}c{i}", fontsize=6,
                    ha="center", color="dimgray")

        doors = env_state.get("doors") or {}
        _draw_door(ax, *DOOR1, bool(doors.get("door1")), "D1")
        _draw_door(ax, *DOOR2, bool(doors.get("door2")), "D2")
        _draw_door(ax, _door3_x(num_agents), CH3_NORTH_WALL_Z,
                   bool(doors.get("door3")), "D3")
        _draw_door(ax, *DOOR4, bool(doors.get("door4")), "D4")

        # Anvils — drawn only while unbroken (hp > 0 entries in the snapshot).
        for anvil in env_state.get("anvils") or []:
            pos = ANVIL_POSITIONS.get(str(anvil.get("kind", "")).lower())
            if pos is None:
                continue
            ax.scatter([pos[0]], [pos[1]], marker="s", s=60, c="purple",
                       zorder=3)
            ax.annotate(f"anvil {anvil.get('kind')} hp={anvil.get('hp')}",
                        (pos[0] + 0.6, pos[1]), fontsize=6, color="purple")

        # Agents: labeled dots, color per agent, health annotated when known.
        agent_xy = {}
        for name, info in (env_state.get("agents") or {}).items():
            pos = info.get("pos")
            if pos is None:
                continue
            try:
                idx = int(str(name).split("_")[-1])
            except (ValueError, IndexError):
                idx = 0
            color = AGENT_COLORS[idx % len(AGENT_COLORS)]
            x, z = float(pos[0]), float(pos[2])
            agent_xy[name] = (x, z)
            alive = info.get("alive", True)
            ax.scatter([x], [z], s=80, c=color, alpha=1.0 if alive else 0.3,
                       edgecolors="black", zorder=4)
            hp = info.get("hp")
            label = name + ("" if hp is None else f" hp={hp:.0f}")
            if not alive:
                label += " (dead)"
            ax.annotate(label, (x, z + 0.9), fontsize=7, ha="center",
                        color=color, zorder=4)

        # Thin arrows for the last few messages (sender → target).
        for sender, target in (env_state.get("recent_messages") or [])[-4:]:
            if sender in agent_xy and target in agent_xy:
                sx, sz = agent_xy[sender]
                tx, tz = agent_xy[target]
                ax.annotate("", xy=(tx, tz), xytext=(sx, sz),
                            arrowprops=dict(arrowstyle="->", color="gray",
                                            linewidth=0.8, alpha=0.7))

        ax.set_xlim(-2, max(20, 4 * num_agents + 6))
        ax.set_ylim(-2, 70)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("z (south → north)")
        ax.set_title(f"Five Chambers — step {env_state.get('step', '?')}",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def render_map_text(env_state: dict, num_agents: int = 3) -> str:
    """Text fallback when no image can be attached (use_map_image=False or a
    text-only client): agent positions + chamber + fixture states, formatted
    to slot into the prompt where the map note sits."""
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
