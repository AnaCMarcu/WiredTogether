"""Pytest bootstrap: put ``src/`` on sys.path so ``import hebbian`` resolves.

The package is laid out as ``src/hebbian/...`` and imported as ``hebbian.*``
throughout the codebase (not ``src.hebbian.*``), so tests need ``src/`` on the
path the same way the training entrypoints get it via PYTHONPATH.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
