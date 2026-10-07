"""Run AgentWorld's per-task verifiers on harness trajectories.

Each ``task_NN_success_criteria.py`` does ``from verifier_utils import ...``.
Suites ship their own ``verifier_utils.py``, so the module is loaded from the
verifier's own directory and swapped into ``sys.modules`` only for the import.

Verifier messages carry ratios such as ``"Arrows: 7/10"``; :func:`progress_from_msg`
turns them into a partial-progress score, and :func:`extract_targets` reads the
item keys and thresholds the verifier checks straight from its syntax tree
(nothing is executed), which the dense Hebbian reward uses.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

_LOCK = threading.Lock()
_UTILS: Dict[Path, ModuleType] = {}
_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")

#: verifier_utils helpers whose 2nd positional argument is an item key.
_ITEM_FUNCS = {"count_item_in_inventories", "has_item_in_any_inventory"}
#: helpers whose 2nd positional argument is a list of patterns.
_PATTERN_FUNCS = {"check_crafted_items", "count_combat_kills", "count_attack_actions",
                  "check_boss_killed_by_loot"}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Verifier:
    """``verify(traj) -> (0|1, message)`` of one task, isolated imports."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with _LOCK:
            utils_path = self.path.parent / "verifier_utils.py"
            utils = _UTILS.get(utils_path)
            if utils is None:
                utils = _load_module(f"aw_verifier_utils_{len(_UTILS)}", utils_path)
                _UTILS[utils_path] = utils
            saved = sys.modules.get("verifier_utils")
            try:
                sys.modules["verifier_utils"] = utils
                self.module = _load_module(f"aw_verifier_{self.path.stem}", self.path)
            finally:
                if saved is not None:
                    sys.modules["verifier_utils"] = saved
                else:
                    sys.modules.pop("verifier_utils", None)
        if not hasattr(self.module, "verify"):
            raise AttributeError(f"{self.path} has no verify()")

    def __call__(self, traj: Dict[str, Any]) -> Tuple[int, str]:
        try:
            ok, msg = self.module.verify(traj)
            return int(ok), str(msg)
        except Exception as exc:  # a verifier crash must never kill a run
            return 0, f"verifier error: {type(exc).__name__}: {exc}"


def progress_from_msg(msg: str) -> Optional[float]:
    """Mean of min(1, a/b) over every ``a/b`` ratio in a verifier message."""
    vals = []
    for a, b in _RATIO.findall(msg or ""):
        b_f = float(b)
        if b_f > 0:
            vals.append(min(1.0, float(a) / b_f))
    return sum(vals) / len(vals) if vals else None


@dataclass
class Targets:
    items: Dict[str, int] = field(default_factory=dict)     # item key → threshold (≥1)
    patterns: List[str] = field(default_factory=list)       # craft / kill patterns

    @property
    def keys(self) -> List[str]:
        return list(self.items)


def _main_function(tree: ast.Module) -> Optional[str]:
    """Name of the function ``verify()`` returns the result of."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "verify":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    func = sub.value.func
                    if isinstance(func, ast.Name):
                        return func.id
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    return node.value.lower() if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def extract_targets(path: str | Path) -> Targets:
    """Item keys, thresholds and patterns the task's main verifier checks."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    main = _main_function(tree)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and (main is None or n.name == main)]
    out = Targets()
    for fn in funcs:
        var_item: Dict[str, str] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in _ITEM_FUNCS and len(node.args) >= 2:
                    key = _const_str(node.args[1])
                    if key:
                        threshold = 1
                        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                            threshold = int(node.args[2].value)
                        out.items[key] = max(out.items.get(key, 1), threshold)
                elif name in _PATTERN_FUNCS and len(node.args) >= 2:
                    seq = node.args[1]
                    if isinstance(seq, (ast.List, ast.Tuple)):
                        out.patterns.extend(s for s in map(_const_str, seq.elts) if s)
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "count_item_in_inventories"
                    and len(node.value.args) >= 2):
                key = _const_str(node.value.args[1])
                if key:
                    var_item[node.targets[0].id] = key
        for node in ast.walk(fn):
            if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                    and node.left.id in var_item and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and isinstance(node.comparators[0].value, (int, float))):
                n = int(node.comparators[0].value)
                if isinstance(node.ops[0], ast.Gt):
                    n += 1
                key = var_item[node.left.id]
                out.items[key] = max(out.items.get(key, 1), n)
    out.patterns = sorted(set(out.patterns))
    return out
