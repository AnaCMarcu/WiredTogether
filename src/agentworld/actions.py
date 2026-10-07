"""AgentWorld tool calls: registry, parsing of model output, trajectory format.

The model writes one call string, e.g. ``craft_item(skill="Fletching",
itemKey="arrow")``. It is parsed with :mod:`ast` (literals only — nothing is
evaluated) and checked against :data:`TOOLS`. The trajectory string mirrors
AgentWorld's runner (``name(k=v, k2=v2)``, values unquoted), because the
verifiers search those strings for substrings like ``attack`` and ``craft``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: name → (required args, optional args). Mirrors agents/tool_definitions.py;
#: ``wait`` is the harness's explicit no-op (an agent with nothing to do).
TOOLS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "move_character": (("x", "y"), ()),
    "enter_portal": ((), ()),
    "equip_item": (("index",), ()),
    "harvest_resource": (("targetInstance",), ()),
    "craft_item": (("skill", "itemKey"), ("count",)),
    "attack_entity": (("targetInstance",), ()),
    "sleep": ((), ("seconds",)),
    "complete": ((), ("response",)),
    "chat": (("message",), ()),
    "transfer_items": (("targetPlayer", "itemKey", "count"), ()),
    "wait": ((), ()),
}

#: Tools that run game logic on the server (everything but the harness no-op).
GAME_TOOLS = tuple(t for t in TOOLS if t != "wait")

_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None   # set when parsing/validation failed

    @property
    def ok(self) -> bool:
        return self.error is None

    def traj_string(self) -> str:
        """AgentWorld's ``[TOOL_CALL_INFO]`` display form."""
        inner = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({inner})"


def _literal(node: ast.AST) -> Any:
    value = ast.literal_eval(node)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("only scalar literal arguments are allowed")


def parse_action(text: Any) -> ToolCall:
    """Parse a model-written call. Never raises; failures set ``error``."""
    if isinstance(text, dict):  # {"name": ..., "args": {...}}
        name = str(text.get("name", ""))
        return validate(ToolCall(name=name, args=dict(text.get("args") or {})))
    raw = str(text or "").strip().strip("`").strip()
    if not raw:
        return ToolCall(name="wait", error="empty action")
    if raw in TOOLS:
        raw = raw + "()"
    m = _CALL.match(raw.splitlines()[0]) or _CALL.match(raw)
    if not m:
        return ToolCall(name="wait", error=f"not a tool call: {raw[:80]!r}")
    name = m.group(1)
    try:
        tree = ast.parse(f"_f({m.group(2)})", mode="eval")
        call = tree.body
        assert isinstance(call, ast.Call)
        args: Dict[str, Any] = {}
        if call.args:
            required, optional = TOOLS.get(name, ((), ()))
            params = list(required) + list(optional)
            if len(call.args) > len(params):
                raise ValueError("too many positional arguments")
            for p, node in zip(params, call.args):
                args[p] = _literal(node)
        for kw in call.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs not allowed")
            args[kw.arg] = _literal(kw.value)
    except (SyntaxError, ValueError, AssertionError) as exc:
        return ToolCall(name=name if name in TOOLS else "wait", error=f"bad arguments: {exc}")
    return validate(ToolCall(name=name, args=args))


def validate(call: ToolCall) -> ToolCall:
    if call.name not in TOOLS:
        return ToolCall(name="wait", args={}, error=f"unknown tool {call.name!r}")
    required, optional = TOOLS[call.name]
    missing = [a for a in required if a not in call.args]
    extra = [a for a in call.args if a not in required and a not in optional]
    if missing:
        return ToolCall(call.name, call.args, error=f"missing {', '.join(missing)}")
    if extra:
        call = ToolCall(call.name, {k: v for k, v in call.args.items() if k not in extra})
    for coord in ("x", "y", "index", "count"):
        if coord in call.args:
            try:
                call.args[coord] = int(call.args[coord])
            except (TypeError, ValueError):
                return ToolCall(call.name, call.args, error=f"{coord} must be an integer")
    return call


def safe_text(text: Any, limit: Optional[int] = None) -> str:
    """Text that is safe to splice into a ``str.format`` prompt template.

    Braces in game text (JSON observations, model output echoed back) would
    either raise in ``.format`` or be silently treated as placeholders.
    """
    s = str(text if text is not None else "")
    s = s.replace("{", "(").replace("}", ")")
    if limit is not None and len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


def tool_reference() -> str:
    """One line per tool, for the system prompt."""
    lines = []
    for name, (req, opt) in TOOLS.items():
        params = list(req) + [f"{o}?" for o in opt]
        lines.append(f"- {name}({', '.join(params)})")
    return "\n".join(lines)
