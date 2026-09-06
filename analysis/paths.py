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


def group(name: str, runs: Path | None = None) -> Path:
    """Locate one run group by name, whatever it is nested under.

    Run groups are filed by research question (``rq1_social_plasticity/``,
    ``rq2_cofiring/``, ...; see ``docs/dataset.md``), but a group name is unique
    across the dataset, so scripts ask for it by name and stay independent of
    the grouping. Falls back to ``<runs>/<name>`` when nothing matches, so the
    caller reports a missing run root rather than this raising.
    """
    root = runs or RUNS
    direct = root / name
    if direct.is_dir():
        return direct
    for rq in sorted(p for p in root.glob("*") if p.is_dir()):
        if (rq / name).is_dir():
            return rq / name
    return direct
