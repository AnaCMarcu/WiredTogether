"""Shared pytest configuration for the WiredTogether test suite.

Everything here runs BEFORE any test module is imported, which matters:

1. Offline/env guards must be set before transformers / sentence_transformers
   are imported anywhere.
2. ``sys.path`` must contain ``src/`` (the repo uses absolute imports with
   ``PYTHONPATH=src``: packages ``hebbian``, ``rl_layer``, ``mindforge``,
   ``marl_craftium``).
3. ``pettingzoo`` and ``craftium`` are not importable on this machine
   (pettingzoo is not installed; craftium ships a Linux-only ``mt_server``
   extension).  ``marl_craftium/__init__`` imports both at module scope, so
   lightweight stand-ins are installed in ``sys.modules`` IF the real packages
   are unavailable.  Tests never launch the game — they only need the Python
   modules to import.
"""

import os
import sys
import types
from pathlib import Path

# ── 1. Environment guards (before any heavy import) ─────────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ── 2. Import paths ──────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for p in (str(SRC), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 3. Stubs for packages that cannot import on this machine ────────────────
def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


try:  # pragma: no cover - depends on local install
    import pettingzoo  # noqa: F401
except ImportError:
    _stub_module("pettingzoo", ParallelEnv=type("ParallelEnv", (object,), {}))

try:  # pragma: no cover - craftium's mt_server is a Linux-only extension
    import craftium.multiagent_env  # noqa: F401
except ImportError:
    # The real `craftium` namespace may have been partially imported and
    # poisoned sys.modules — drop any half-imported entries first.
    for key in [k for k in list(sys.modules) if k == "craftium" or k.startswith("craftium.")]:
        del sys.modules[key]
    _stub_module("craftium")
    _stub_module(
        "craftium.multiagent_env",
        MarlCraftiumEnv=type("MarlCraftiumEnv", (object,), {}),
        ACTION_ORDER=[],
    )

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# ── 4. Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def seed_all():
    """Deterministic numpy/torch state for every test."""
    np.random.seed(0)
    import torch

    torch.manual_seed(0)
    yield


@pytest.fixture
def lua_root() -> Path:
    """Root of the five_chambers Lua mod (for spec-parsing tests)."""
    return SRC / "marl_craftium" / "craftium-envs" / "five-chambers" / "mods" / "five_chambers"


@pytest.fixture
def hcfg():
    """HebbianConfig factory with decay=0 so hand computations are exact.

    Tests that exercise the homeostatic decay pass ``decay=...`` explicitly.
    """
    from hebbian import HebbianConfig

    def _make(num_agents: int = 3, **overrides):
        defaults = dict(
            enabled=True,
            mode="reward_modulated",
            num_agents=num_agents,
            decay=0.0,
        )
        defaults.update(overrides)
        return HebbianConfig(**defaults)

    return _make


class FakeSentenceTransformer:
    """Deterministic, download-free stand-in for sentence_transformers.

    ``encode`` returns all-ones vectors of dimension ``DIM`` so the semantic
    block of the centralized-critic joint state is exactly assertable.
    """

    DIM = 8

    def __init__(self, model_name_or_path=None, device=None, **kwargs):
        self.model_name = model_name_or_path
        self.device = device

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.ones((len(sentences), self.DIM), dtype=np.float32)

    def to(self, device):  # noqa: D401 - torch-style chaining
        return self


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a fake `sentence_transformers` module and a fake top-level
    `agent_modules.util` (CentralizedCritic.__init__ imports both lazily;
    note it imports `agent_modules.util`, not `mindforge.agent_modules.util`,
    because the runtime adds src/mindforge to sys.path).
    """
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)

    am_pkg = types.ModuleType("agent_modules")
    am_pkg.__path__ = []  # mark as package
    am_util = types.ModuleType("agent_modules.util")
    am_util.ST_MODEL_NAME = "fake-model"
    monkeypatch.setitem(sys.modules, "agent_modules", am_pkg)
    monkeypatch.setitem(sys.modules, "agent_modules.util", am_util)

    return FakeSentenceTransformer


@pytest.fixture
def autogen_stubs(monkeypatch):
    """Optional stubs for autogen/chromadb/wandb — for FUTURE tests that import
    mindforge.custom_agent or the ChromaDB-backed memory modules.  Unused by
    the current suite on purpose: global stubs would mask accidental heavy
    imports.
    """
    for name in ("autogen_agentchat", "autogen_core", "autogen_ext", "chromadb", "wandb"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    yield
