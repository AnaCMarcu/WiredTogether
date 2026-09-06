"""Repository paths shared by the analysis scripts.

Importing this module puts ``analysis/``, the repo root, ``src/`` and the
qualitative pipeline on ``sys.path``, so a script can import its siblings
(``make_results`` and friends), the ``hebbian``/``mindforge`` packages and
``qual_lib`` no matter which directory it was launched from.

Anchoring every default to :data:`REPO` also means the scripts no longer
have to be run from the repo root to find their inputs and outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
SRC = REPO / "src"
QUALITATIVE = ANALYSIS / "qualitative"

#: Run artifacts synced off the cluster (git-ignored; see docs/experiments.md).
RUNS = REPO / "runs_from_daic"
#: Generated tables and figures, one sub-directory per experiment family.
ASSETS = REPO / "paper_assets"

for _p in (ANALYSIS, REPO, SRC, QUALITATIVE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
