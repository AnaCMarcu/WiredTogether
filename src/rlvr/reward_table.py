"""Flatten ``TRACKS`` into a single ``{milestone_id: reward}`` dict.

``TRACKS`` (in ``mindforge.agent_modules.craftium_metric``) is shaped as
``{track_name: [(milestone_id, reward), ...]}``. The verifier wants a flat
lookup. This helper centralises the flattening so future code never imports
``MILESTONE_TRACK`` by mistake — that one maps milestone_id to track-name
*strings*, not to reward floats.
"""

from __future__ import annotations

from mindforge.agent_modules.craftium_metric import TRACKS


def build_milestone_rewards() -> dict[str, float]:
    """Return ``{milestone_id: reward}`` flattened from ``TRACKS``.

    The mapping is constructed fresh on each call; callers may cache it.
    """
    out: dict[str, float] = {}
    for _track, entries in TRACKS.items():
        for mid, reward in entries:
            out[mid] = float(reward)
    return out
