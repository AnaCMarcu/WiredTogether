"""Discrete-action mapping for Craftium's MARL env.

PettingZoo expects an integer action; Craftium expects a ``{name: 1, mouse: [x,y]}``
dict. ``_discrete_to_dict`` bridges the two. Action 0 is NOP; 1-22 map to
``_DISCRETE_ACTIONS``.
"""

_DISCRETE_ACTIONS = [
    "forward", "backward", "left", "right", "jump", "sneak",
    "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
    "mouse x+", "mouse x-", "mouse y-", "mouse y+",  # y- = look down, y+ = look up (Minetest Y-axis is inverted)
    # Added actions (indices 17-21):
    "inventory",                          # RESERVED NOP — never selected (no LLM
                                          # ACTION_MAP entry and not in RLConfig.actions);
                                          # kept only to preserve index alignment so
                                          # drop/slot_6-8 stay at indices 18-21. Mapped
                                          # to NOP in _discrete_to_dict (would otherwise
                                          # open an uncloseable inventory formspec).
    "drop",                               # drop held item
    "slot_6", "slot_7", "slot_8",         # extra hotbar slots
]

# Doubled from 0.5 → ~20-30° per step, halves the steps needed for orientation.
_MOUSE_MOV = 1.0


def _discrete_to_dict(action: int) -> dict:
    """Convert a Discrete(23) integer to MarlCraftiumEnv dict format.

    Action 0 → NOP. 1-22 → named actions in ``_DISCRETE_ACTIONS``. Mouse
    actions return ``{"mouse": [x, y]}``; everything else returns
    ``{name: 1, "mouse": [0, 0]}``.
    """
    action = int(action)
    if action == 0:
        return {}  # NOP: no mouse movement, no key

    name = _DISCRETE_ACTIONS[action - 1]
    mouse = [0.0, 0.0]

    if name == "mouse x+":
        mouse[0] = _MOUSE_MOV
        return {"mouse": mouse}
    if name == "mouse x-":
        mouse[0] = -_MOUSE_MOV
        return {"mouse": mouse}
    if name == "mouse y+":
        mouse[1] = _MOUSE_MOV
        return {"mouse": mouse}
    if name == "mouse y-":
        mouse[1] = -_MOUSE_MOV
        return {"mouse": mouse}
    if name == "inventory":
        # Pressing the inventory key opens the survival inventory/crafting
        # formspec — a gray panel that occludes the headless agent's view and
        # that it can never close, so it persists across chambers and episodes.
        # Inventory contents are already surfaced in the text prompt, so this
        # action is useless; treat it as a NOP (mirrors the LLM side, where
        # ACTION_MAP maps "Inventory" -> NoOp). Kept at its index so the
        # Discrete(23) space and existing RL checkpoints don't shift.
        return {}
    return {name: 1, "mouse": mouse}
