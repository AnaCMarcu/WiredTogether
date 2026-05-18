"""Parse LLM JSON output into ``(action_dict, format_reward)``.

The format reward is the RLVR-style "did the model produce a parseable
output" signal. Weighting it small (≤ 0.1 × milestone-reward magnitude)
in the verifier teaches the model to emit clean JSON before any
milestone-completion signal can land.

Valid action names come from ``marl_craftium._actions._DISCRETE_ACTIONS``
(22 entries) plus the synthetic ``"nop"`` (env action 0).
"""

from __future__ import annotations

import json

# Hardcoded copy of ``marl_craftium._actions._DISCRETE_ACTIONS``. We cannot
# always import the upstream constant directly because ``marl_craftium``'s
# package ``__init__`` eagerly imports ``openworld_multi_agents``, which
# requires ``gymnasium`` — not present in minimal dev environments. The
# soft-check below catches drift whenever the full env IS available.
_DISCRETE_ACTIONS_FALLBACK: list[str] = [
    "forward", "backward", "left", "right", "jump", "sneak",
    "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
    "mouse x+", "mouse x-", "mouse y-", "mouse y+",
    "inventory", "drop", "slot_6", "slot_7", "slot_8",
]

try:
    from marl_craftium._actions import _DISCRETE_ACTIONS as _UPSTREAM  # noqa: F401
    if list(_UPSTREAM) != _DISCRETE_ACTIONS_FALLBACK:
        raise RuntimeError(
            "rlvr.action_parser._DISCRETE_ACTIONS_FALLBACK is out of sync with "
            "marl_craftium._actions._DISCRETE_ACTIONS. Update the hardcoded copy."
        )
    _DISCRETE_ACTIONS: list[str] = list(_UPSTREAM)
except ImportError:
    _DISCRETE_ACTIONS = _DISCRETE_ACTIONS_FALLBACK

VALID_ACTION_NAMES: frozenset[str] = frozenset(_DISCRETE_ACTIONS) | {"nop"}


def parse_action_json(text: str, n_agents: int) -> tuple[dict | None, float]:
    """Parse the LLM's JSON output and score its format.

    Returns ``(parsed, reward)`` where:

    - ``parsed`` is the JSON dict on success, else ``None``. A non-None
      ``parsed`` is guaranteed to have ``parsed["action"]`` in
      ``VALID_ACTION_NAMES``. Optional fields are present-or-absent;
      callers should re-check anything they rely on.
    - ``reward`` ∈ {0.0, 0.5, 1.0}:
        * ``1.0`` — JSON parses, ``action`` is valid, and both optional
          fields (``communication_target``, ``thoughts``) are present and
          well-typed.
        * ``0.5`` — ``action`` is valid but at least one optional field is
          missing or wrong-typed.
        * ``0.0`` — JSON does not parse, is not a dict, or ``action`` is
          missing / not a string / not in ``VALID_ACTION_NAMES``.

    ``n_agents`` bounds the valid range for ``communication_target``: the
    field must be ``None`` or an int in ``[0, n_agents)``. The sender's own
    id is permitted (we don't enforce target ≠ sender here; that's a
    rule for the trainer, not the parser).
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 0.0

    if not isinstance(data, dict):
        return None, 0.0

    score = score_parsed_action(data, n_agents)
    if score == 0.0:
        return None, 0.0
    return data, score


def score_parsed_action(action: object, n_agents: int) -> float:
    """Re-score an already-parsed action dict on the same rubric as
    ``parse_action_json``. Returns ``0.0`` if ``action`` is not a dict or
    its ``action`` field is invalid; ``0.5`` for valid-action-but-partial;
    ``1.0`` for fully-valid.

    Used by the verifier to score format rewards from
    ``GRPOTrajectory.actions`` (which carries already-parsed dicts).
    """
    if not isinstance(action, dict):
        return 0.0
    name = action.get("action")
    if not isinstance(name, str) or name not in VALID_ACTION_NAMES:
        return 0.0
    optionals_ok = (
        _is_valid_comm_target(action.get("communication_target", _MISSING), n_agents)
        and isinstance(action.get("thoughts"), str)
    )
    return 1.0 if optionals_ok else 0.5


# Sentinel distinguishes "field omitted" from "field is None".
_MISSING = object()


def _is_valid_comm_target(value: object, n_agents: int) -> bool:
    """``communication_target`` is valid when explicitly ``None`` or an int
    in ``[0, n_agents)``. Omitted (``_MISSING``) or wrong-typed → invalid.
    """
    if value is _MISSING:
        return False
    if value is None:
        return True
    # Reject bools (which are ints in Python) — they were not intended here.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value < n_agents
    return False
