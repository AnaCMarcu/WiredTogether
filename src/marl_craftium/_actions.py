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
    "inventory",                          # toggle inventory/crafting menu
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
    return {name: 1, "mouse": mouse}
