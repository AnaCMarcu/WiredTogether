#!/usr/bin/env python3
"""run_3f.py - the qualitative pipeline over the Hebbian 2.0 arms.

qual_lib.registry discovers runs by walking make_results.CONDITIONS, which
predates the three-factor arms: new_exp_0_gemma_si3f*, exp34 and exp35 are
not in it, so `run.py parse --runs-root <3f root>` reports "no runs found"
and silently does nothing. The pareto_*_hebbian dirs DO parse, because those
directory names are shared with the reward-modulated sweep.

This registers the missing conditions and delegates to run.py untouched.
registry derives DIR_TO_LABEL / LABEL_TO_DIR / HEBBIAN_DIRS from CONDITIONS
at import time, so all four have to be rebuilt, not just the list.

Every argument is passed through, so usage mirrors run.py:

  python analysis/qualitative/run_3f.py parse \\
      --runs-root runs_from_daic/compute/pareto_social_3f --out out_3f_social
  python analysis/qualitative/run_3f.py metrics \\
      --runs-root runs_from_daic/compute/pareto_social_3f --out out_3f_social
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qual_lib import registry  # noqa: E402

# (label, dir, group, hebbian?) - the CONDITIONS tuple shape. All three are
# three_factor + signed death LTD (hebbian_death_ltd=0.05), i.e. Hebbian 2.0;
# the interval-8 Gemma point doubles as the social Pareto's si3f8 arm.
EXTRA = [
    ("Gemma-E4B+Heb2.0", "new_exp_0_gemma_si3f8", "main", True),
    # The six-seed three-factor Gemma arm the results tables point at
    # (Gemma-E4B+Heb2.0 in make_final_table_extended). It differs from
    # si3f8 only in hebbian_death_ltd (0 vs 0.05); registered under its own
    # label so both can be parsed without either overwriting the other.
    ("Gemma-E4B+Heb3f", "new_exp_0_gemma_hebbian3f", "main", True),
    ("LLM-2B+Heb2.0", "exp34_llm_2b_three_factor_fdecay", "main", True),
    ("LLM-9B+Heb2.0", "exp35_llm_9b_three_factor_fdecay", "main", True),
    # The rest of the interval sweep, so the same driver can produce beliefs
    # for the frontier figure as those arms fill out.
    ("Gemma-E4B+Heb2.0 si2", "new_exp_0_gemma_si3f2", "main", True),
    ("Gemma-E4B+Heb2.0 si20", "new_exp_0_gemma_si3f20", "main", True),
    ("Gemma-E4B+Heb2.0 si50", "new_exp_0_gemma_si3f50", "main", True),
    ("Gemma-E4B+Heb2.0 si100", "new_exp_0_gemma_si3f100", "main", True),
    ("Gemma-E4B+Heb2.0 si200", "new_exp_0_gemma_si3f200", "main", True),
    ("Gemma-E4B+Heb2.0 si500", "new_exp_0_gemma_si3f500", "main", True),
]

_known = {d for _, d, _, _ in registry.CONDITIONS}
registry.CONDITIONS = list(registry.CONDITIONS) + [
    c for c in EXTRA if c[1] not in _known
]
registry.DIR_TO_LABEL = {d: lab for lab, d, _, _ in registry.CONDITIONS}
registry.LABEL_TO_DIR = {lab: d for lab, d, _, _ in registry.CONDITIONS}
registry.HEBBIAN_DIRS = {d for _, d, _, h in registry.CONDITIONS if h}

import run  # noqa: E402  (must follow the registry patch)

if __name__ == "__main__":
    raise SystemExit(run.main())
