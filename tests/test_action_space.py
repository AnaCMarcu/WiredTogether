"""Action-space contract tests.

Pins the three layers of the action pipeline against each other:

1. ``marl_craftium._actions`` — backend Discrete(23) -> Craftium dict mapping
   (``_DISCRETE_ACTIONS``, ``_discrete_to_dict``).
2. ``mindforge.custom_environment_craftium`` — named-action layer
   (``ACTION_MAP``, ``canonicalize_action``, idle/pitch/sustain constants).
3. ``rl_layer.config`` / ``rl_layer.rl_layer`` — the RL candidate action set
   (``RLConfig.actions`` must mirror ACTION_MAP; ``mask_slot_actions`` masks
   exactly the 8 Slot* actions, leaving 14 candidates).

All expected values are hand-copied from source, not computed from it.
"""

import types

import pytest

from marl_craftium._actions import _DISCRETE_ACTIONS, _MOUSE_MOV, _discrete_to_dict
from mindforge.custom_environment_craftium import (
    ACTION_MAP,
    VALID_ACTIONS,
    _ACTION_ALIASES,
    CraftiumEnvironmentInterface,
    canonicalize_action,
)
from rl_layer.config import RLConfig
from rl_layer.rl_layer import RLLayer


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: backend discrete actions
# ─────────────────────────────────────────────────────────────────────────────

def test_discrete_actions_exact_contents():
    """_DISCRETE_ACTIONS is exactly the 22-entry backend list (Discrete(23) incl. NOP 0)."""
    assert _DISCRETE_ACTIONS == [
        "forward", "backward", "left", "right", "jump", "sneak",
        "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
        "mouse x+", "mouse x-", "mouse y-", "mouse y+",
        "inventory", "drop", "slot_6", "slot_7", "slot_8",
    ]
    assert len(_DISCRETE_ACTIONS) == 22
    # Discrete space size = 22 named + NOP index 0 = 23.
    assert len(_DISCRETE_ACTIONS) + 1 == 23


def test_discrete_to_dict_nop_is_empty():
    """Index 0 is NOP: no key, no mouse movement — the empty dict."""
    assert _discrete_to_dict(0) == {}


def test_discrete_to_dict_mouse_signs():
    """Mouse indices 14-17 give the documented sign convention (y- = look down)."""
    assert _MOUSE_MOV == 1.0
    # _DISCRETE_ACTIONS[13..16] = "mouse x+", "mouse x-", "mouse y-", "mouse y+"
    assert _discrete_to_dict(14) == {"mouse": [1.0, 0.0]}    # x+ (turn right)
    assert _discrete_to_dict(15) == {"mouse": [-1.0, 0.0]}   # x- (turn left)
    assert _discrete_to_dict(16) == {"mouse": [0.0, -1.0]}   # y- (look DOWN, inverted axis)
    assert _discrete_to_dict(17) == {"mouse": [0.0, 1.0]}    # y+ (look UP)


def test_discrete_to_dict_keyed_actions():
    """Non-mouse actions return {name: 1, 'mouse': [0, 0]}."""
    assert _discrete_to_dict(1) == {"forward": 1, "mouse": [0.0, 0.0]}
    assert _discrete_to_dict(7) == {"dig": 1, "mouse": [0.0, 0.0]}
    assert _discrete_to_dict(19) == {"drop": 1, "mouse": [0.0, 0.0]}
    assert _discrete_to_dict(22) == {"slot_8": 1, "mouse": [0.0, 0.0]}


def test_discrete_to_dict_inventory_reserved_noop():
    """Reserved 'inventory' index 18 is a no-op: returns the empty dict, not a key press."""
    assert _DISCRETE_ACTIONS[17] == "inventory"
    assert _discrete_to_dict(18) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: named ACTION_MAP
# ─────────────────────────────────────────────────────────────────────────────

def test_action_map_exact():
    """ACTION_MAP has 22 names; key anchors NoOp=0, Dig=7, Drop=19, Slot8=22 hold."""
    assert len(ACTION_MAP) == 22
    assert ACTION_MAP["NoOp"] == 0
    assert ACTION_MAP["Dig"] == 7
    assert ACTION_MAP["Drop"] == 19
    assert ACTION_MAP["Slot8"] == 22
    assert VALID_ACTIONS == list(ACTION_MAP)


def test_action_map_index_hole_is_18():
    """Mapped indices are exactly 0..22 minus 18 (the reserved 'inventory' slot)."""
    assert set(ACTION_MAP.values()) == set(range(23)) - {18}


def test_action_map_backend_cross_consistency():
    """Every named action != NoOp maps onto its expected backend string in _DISCRETE_ACTIONS."""
    expected_backend = {
        "MoveForward": "forward", "MoveBackward": "backward",
        "MoveLeft": "left", "MoveRight": "right",
        "Jump": "jump", "Sneak": "sneak", "Dig": "dig", "Place": "place",
        "Slot1": "slot_1", "Slot2": "slot_2", "Slot3": "slot_3",
        "Slot4": "slot_4", "Slot5": "slot_5",
        "TurnRight": "mouse x+", "TurnLeft": "mouse x-",
        "LookDown": "mouse y-", "LookUp": "mouse y+",
        "Drop": "drop", "Slot6": "slot_6", "Slot7": "slot_7", "Slot8": "slot_8",
    }
    assert set(expected_backend) == set(ACTION_MAP) - {"NoOp"}
    for name, backend in expected_backend.items():
        assert _DISCRETE_ACTIONS[ACTION_MAP[name] - 1] == backend, name


def test_look_actions_end_to_end_mouse_dicts():
    """LookDown/LookUp routed through ACTION_MAP produce the inverted-Y mouse dicts."""
    assert _discrete_to_dict(ACTION_MAP["LookDown"]) == {"mouse": [0.0, -1.0]}
    assert _discrete_to_dict(ACTION_MAP["LookUp"]) == {"mouse": [0.0, 1.0]}
    assert _discrete_to_dict(ACTION_MAP["TurnRight"]) == {"mouse": [1.0, 0.0]}
    assert _discrete_to_dict(ACTION_MAP["TurnLeft"]) == {"mouse": [-1.0, 0.0]}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: RLConfig action tuple + candidate masking
# ─────────────────────────────────────────────────────────────────────────────

def test_rlconfig_actions_match_action_map():
    """RLConfig().actions equals tuple(ACTION_MAP.keys()) in name AND order ('must match' contract)."""
    assert RLConfig().actions == tuple(ACTION_MAP.keys())


def test_mask_slot_actions_leaves_14_candidates():
    """mask_slot_actions=True masks the 8 Slot* actions: 22 -> 14 candidates, order preserved."""
    dummy = types.SimpleNamespace(config=RLConfig(mask_slot_actions=True))
    candidates = RLLayer._build_candidate_actions(dummy)
    assert len(candidates) == 14
    assert candidates == (
        "NoOp", "MoveForward", "MoveBackward", "MoveLeft", "MoveRight",
        "Jump", "Sneak", "Dig", "Place",
        "TurnRight", "TurnLeft", "LookDown", "LookUp", "Drop",
    )
    assert not any(a.startswith("Slot") for a in candidates)


def test_mask_slot_actions_off_keeps_all_22():
    """mask_slot_actions=False keeps the full 22-action candidate set."""
    dummy = types.SimpleNamespace(config=RLConfig(mask_slot_actions=False))
    candidates = RLLayer._build_candidate_actions(dummy)
    assert candidates == RLConfig().actions
    assert len(candidates) == 22
    assert RLConfig().mask_slot_actions is True  # default masks slots


# ─────────────────────────────────────────────────────────────────────────────
# canonicalize_action
# ─────────────────────────────────────────────────────────────────────────────

def test_canonicalize_exact_names_pass_through():
    """Exact ACTION_MAP keys are returned unchanged."""
    for name in ("NoOp", "MoveForward", "Dig", "LookDown", "Slot8", "Drop"):
        assert canonicalize_action(name) == name


def test_canonicalize_alias_samples():
    """Synonym aliases resolve case-insensitively: attack/mine/hit->Dig, turn->TurnRight, wait/stop->NoOp, ..."""
    assert canonicalize_action("attack") == "Dig"
    assert canonicalize_action("Mine") == "Dig"
    assert canonicalize_action("HIT") == "Dig"
    assert canonicalize_action("turn") == "TurnRight"
    assert canonicalize_action("rotateleft") == "TurnLeft"
    assert canonicalize_action("wait") == "NoOp"
    assert canonicalize_action("stop") == "NoOp"
    assert canonicalize_action("throw") == "Drop"
    assert canonicalize_action("crouch") == "Sneak"
    assert canonicalize_action("place block") == "Place"   # norm 'placeblock'
    assert canonicalize_action("tilt_up") == "LookUp"      # norm 'tiltup'
    # alias dict itself maps normalized keys -> canonical names only
    assert set(_ACTION_ALIASES.values()) <= set(ACTION_MAP)


def test_canonicalize_format_normalisation():
    """Case / space / underscore / hyphen variants of a valid name all canonicalize."""
    assert canonicalize_action("move_forward") == "MoveForward"
    assert canonicalize_action("Move Forward") == "MoveForward"
    assert canonicalize_action("MOVE-FORWARD") == "MoveForward"
    assert canonicalize_action("noop") == "NoOp"
    assert canonicalize_action("look down") == "LookDown"


def test_canonicalize_garbage_returns_none():
    """Unrecoverable strings return None."""
    assert canonicalize_action("fly") is None
    assert canonicalize_action("") is None
    assert canonicalize_action("xyzzy plugh") is None


def test_canonicalize_multiword_fallback_uses_first_token():
    """Multi-word strings fall back to the leading word: 'attack the zombie' -> Dig."""
    assert canonicalize_action("attack the zombie") == "Dig"
    assert canonicalize_action("MoveForward to the door") == "MoveForward"
    assert canonicalize_action("dig the stone wall") == "Dig"


def test_canonicalize_non_string_inputs():
    """Non-strings are str()-ified: None -> 'NoOp' (via the 'none' alias!); ints -> None."""
    # str(None).lower() == 'none', which IS in _ACTION_ALIASES -> NoOp.
    assert canonicalize_action(None) == "NoOp"
    assert canonicalize_action(7) is None
    assert canonicalize_action(0) is None


# ─────────────────────────────────────────────────────────────────────────────
# Class constants + idle guard
# ─────────────────────────────────────────────────────────────────────────────

def test_pitch_sustain_idle_constants():
    """Pitch limits, sustained-tick table, and idle thresholds match source."""
    assert CraftiumEnvironmentInterface._PITCH_MAX_DOWN == 2
    assert CraftiumEnvironmentInterface._PITCH_MAX_UP == 1
    assert CraftiumEnvironmentInterface._SUSTAINED_TICKS == {"Dig": 10}
    assert CraftiumEnvironmentInterface._MAX_CONSECUTIVE_IDLE == 1
    assert CraftiumEnvironmentInterface._IDLE_ACTIONS == frozenset({"NoOp"})


def _bare_env():
    """Interface instance without __init__ (no game launch); only idle-guard state."""
    env = CraftiumEnvironmentInterface.__new__(CraftiumEnvironmentInterface)
    env._consecutive_idle = {}
    env._idle_force_counts = {}
    return env


def test_idle_guard_forces_moveforward_on_first_noop():
    """With _MAX_CONSECUTIVE_IDLE==1, the very FIRST NoOp is rewritten to MoveForward."""
    env = _bare_env()
    # Non-idle action passes through and (re)sets the counter to 0.
    assert env._idle_guard("agent_0", 0, "Dig") == "Dig"
    assert env._consecutive_idle["agent_0"] == 0
    # First NoOp: counter hits 1, 1 < 1 is False -> forced.
    assert env._idle_guard("agent_0", 0, "NoOp") == "MoveForward"
    assert env._consecutive_idle["agent_0"] == 0  # reset after forcing
    assert env._idle_force_counts["agent_0"] == {
        "invalid_action": 0, "explicit_noop": 1,
    }


def test_idle_guard_tallies_invalid_action_cause_separately():
    """invalid_original!=None records cause 'invalid_action'; summary returns copies."""
    env = _bare_env()
    assert env._idle_guard("agent_1", 1, "NoOp", invalid_original="Attack") == "MoveForward"
    assert env._idle_guard("agent_1", 1, "NoOp") == "MoveForward"
    assert env._idle_force_counts["agent_1"] == {
        "invalid_action": 1, "explicit_noop": 1,
    }
    summary = env.idle_force_summary()
    assert summary == {"agent_1": {"invalid_action": 1, "explicit_noop": 1}}
    summary["agent_1"]["explicit_noop"] = 999  # mutate the copy
    assert env._idle_force_counts["agent_1"]["explicit_noop"] == 1
