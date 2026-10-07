"""Load AgentWorld's own game-tool code from a pinned checkout.

AgentWorld's ``agents/game_tools.py`` (MPL-2.0) implements every tool's game
logic: the auto-approach before a harvest, combat polling and loot pickup, the
two-step transfer, the observation post-processing the verifiers depend on.
Re-implementing that would drift from the benchmark, so the harness imports
the file unmodified from ``$AGENTWORLD_ROOT`` (cloned at the commit pinned in
hpc/agentworld/AGENTWORLD_COMMIT) and never copies it into this repository.

``game_tools.py`` does ``from config import ...`` — a bare top-level module
name. :func:`load_game_tools` imports AgentWorld's ``config`` under that name
only for the duration of the import, then restores whatever ``config`` was in
``sys.modules`` before, so nothing else in the process sees it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

ENV_ROOT = "AGENTWORLD_ROOT"
_LOCK = threading.Lock()
_CACHE: dict[Path, "Vendor"] = {}


@dataclass(frozen=True)
class Vendor:
    """Handles to the AgentWorld modules the harness uses."""

    root: Path
    game_tools: ModuleType
    config: ModuleType

    @property
    def tools_class(self):
        return self.game_tools.KaetramGameTools

    @property
    def master_password(self) -> str:
        return getattr(self.config, "MASTER_PASSWORD", "agentworld-benchmark")

    def make_tools(self, base_url: str):
        """A fresh per-agent tools object bound to one server."""
        return self.tools_class(base_url=base_url)


def find_root(root: Optional[str | os.PathLike] = None) -> Path:
    """Resolve the AgentWorld checkout (argument, else ``$AGENTWORLD_ROOT``)."""
    value = root if root is not None else os.environ.get(ENV_ROOT)
    if not value:
        raise FileNotFoundError(
            f"AgentWorld checkout not given: pass root= or set ${ENV_ROOT} "
            "(see hpc/agentworld/fetch_agentworld.sh)")
    path = Path(value).expanduser().resolve()
    if not (path / "agents" / "game_tools.py").is_file():
        raise FileNotFoundError(f"{path} is not an AgentWorld checkout (no agents/game_tools.py)")
    return path


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_game_tools(root: Optional[str | os.PathLike] = None) -> Vendor:
    """Import AgentWorld's ``config`` and ``game_tools`` without leaking ``config``."""
    path = find_root(root)
    with _LOCK:
        if path in _CACHE:
            return _CACHE[path]
        agents = path / "agents"
        saved = sys.modules.get("config")
        try:
            config = _load("agentworld_vendor_config", agents / "config.py")
            sys.modules["config"] = config
            game_tools = _load("agentworld_vendor_game_tools", agents / "game_tools.py")
        finally:
            if saved is not None:
                sys.modules["config"] = saved
            else:
                sys.modules.pop("config", None)
        vendor = Vendor(root=path, game_tools=game_tools, config=config)
        _CACHE[path] = vendor
        return vendor
