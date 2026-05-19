"""pytest conftest — make hebbian-marl's `epymarl/src/` importable.

EPyMARL is invoked as `python epymarl/src/main.py`, so its modules live
on the `epymarl/src/` import path. Tests need the same import path or
they'd have to know about the vendor layout.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EPYMARL_SRC = REPO_ROOT / "epymarl" / "src"

if str(EPYMARL_SRC) not in sys.path:
    sys.path.insert(0, str(EPYMARL_SRC))
