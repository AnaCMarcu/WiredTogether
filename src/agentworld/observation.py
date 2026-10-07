"""Compact, brace-free observation text for prompts.

AgentWorld's raw observation at radius 64 is several KB of JSON. Prompts get a
bounded summary instead: own status, inventory, and the k nearest players,
mobs and resources — so prompt length does not grow with the number of agents
in the world.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentworld.actions import safe_text


def _nearest(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return sorted(items or [], key=lambda d: d.get("distanceFrom", 1e9))[:k]


def position(obs: Optional[Dict[str, Any]]) -> Optional[tuple]:
    if not obs:
        return None
    loc = obs.get("location") or {}
    if "x" not in loc or "y" not in loc:
        return None
    return (float(loc["x"]), float(loc["y"]), 0.0)


def hp(obs: Optional[Dict[str, Any]]) -> Optional[float]:
    if not obs:
        return None
    return (obs.get("playerStatus") or {}).get("hitPoints")


def inventory_counts(obs: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in ((obs or {}).get("inventory") or {}).get("items") or []:
        if isinstance(it, dict):
            key = str(it.get("key", ""))
            out[key] = out.get(key, 0) + int(it.get("count", 0) or 0)
    return out


def render_obs_text(obs: Optional[Dict[str, Any]], k: int = 6,
                    hide_players: Optional[set] = None) -> str:
    if not obs:
        return "Observation unavailable this round."
    lines = []
    loc = obs.get("location") or {}
    ps = obs.get("playerStatus") or {}
    lines.append(f"You are at ({loc.get('x', '?')}, {loc.get('y', '?')}); "
                 f"HP {ps.get('hitPoints', '?')}/{ps.get('maxHitPoints', '?')}.")
    inv = inventory_counts(obs)
    lines.append("Inventory: " + (", ".join(f"{c}x {k_}" for k_, c in inv.items()) or "empty") + ".")
    hide = hide_players or set()
    players = [p for p in obs.get("players") or [] if p.get("name") not in hide]
    if players:
        lines.append("Nearby players: " + "; ".join(
            f"{p.get('name')} at ({p.get('x')}, {p.get('y')}), {p.get('distanceFrom')} tiles"
            for p in _nearest(players, k)))
    mobs = obs.get("mobs") or []
    if mobs:
        lines.append("Mobs: " + "; ".join(
            f"{m.get('name')} lvl {m.get('level')} HP {m.get('hitPoints')}/{m.get('maxHitPoints')} "
            f"[instance {m.get('instance')}] {m.get('distanceFrom')} tiles"
            for m in _nearest(mobs, k)))
    for label, key in (("Trees", "trees"), ("Rocks", "rocks"), ("Fishing spots", "fishSpots"),
                       ("Foraging", "foraging")):
        res = obs.get(key) or []
        if res:
            lines.append(f"{label}: " + "; ".join(
                f"{r.get('name')} [instance {r.get('instance')}] at ({r.get('x')}, {r.get('y')})"
                for r in _nearest(res, k)))
    return safe_text("\n".join(lines))
