"""Pytest bootstrap: put ``src/`` (and ``src/mindforge/``) on sys.path.

The package is laid out as ``src/hebbian/...`` and imported as ``hebbian.*``
throughout the codebase (not ``src.hebbian.*``), so tests need ``src/`` on the
path the same way the training entrypoints get it via PYTHONPATH. The mindforge
entrypoints additionally run from ``src/mindforge`` (so ``agent_modules`` and
``custom_environment_craftium`` resolve as top-level modules); we mirror that.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
for _rel in ("../src", "../src/mindforge"):
    _p = os.path.abspath(os.path.join(_HERE, _rel))
    if _p not in sys.path:
        sys.path.insert(0, _p)
