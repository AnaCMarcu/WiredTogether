"""Pytest configuration: add ``src/`` to ``sys.path`` so tests can use the
same bare-package imports the runtime expects (e.g. ``from rlvr.reward_table
import ...``, ``from hebbian.config import ...``).

This matches the launch convention documented in ``README.md`` and used in
``scripts/mindforge_slurm.sh`` (``PYTHONPATH=src`` / ``cd src/mindforge``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
