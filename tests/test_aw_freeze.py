"""The AgentWorld work must not change the WIRE stack (paper reproducibility).

Fails if any commit since this branch forked from ``world-scaling`` (the merge
base, so new commits on world-scaling itself do not trip it) touches a WIRE-side
file, unless that commit is a WIRE change this branch carries on purpose
(``WIRE_COMMITS``). Uncommitted or untracked edits inside WIRE paths fail too.
Skipped where that branch is not available (e.g. a cluster checkout that only
has this branch).
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = "world-scaling"
FROZEN = [
    "src/hebbian/graph.py",
    "src/hebbian/config.py",
    "src/mindforge/custom_agent.py",
    "src/mindforge/agent_modules",
    "src/mindforge/multi_agent_craftium.py",
    "src/mindforge/cli.py",
    "src/mindforge/env",
    "src/marl_craftium",
    "src/mindforge/prompts",
]
# WIRE changes carried on purpose, matched by commit-subject prefix (subjects
# survive cherry-picks and rebases, hashes do not). The HMAS-2 hard orchestrator
# is opt-in (--orchestrator-variant hmas2) and is developed on
# hard-orchestrator-snellius; its default paths are pinned byte-identical by
# tests/test_orchestrator_hmas2.py and tests/test_hub_prompts.py.
WIRE_COMMITS = (
    "Add the HMAS-2 hard-orchestrator baseline",
)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_wire_stack_unchanged():
    if _git("rev-parse", "--verify", "--quiet", BASE).returncode != 0:
        pytest.skip(f"branch {BASE} not available")
    fork = _git("merge-base", "HEAD", BASE).stdout.strip()
    if not fork:
        pytest.skip(f"no common history with {BASE}")
    log = _git("log", "--format=%h %s", f"{fork}..HEAD", "--", *FROZEN)
    assert log.returncode == 0, log.stderr
    offending = [line for line in log.stdout.splitlines()
                 if not line.split(" ", 1)[-1].startswith(WIRE_COMMITS)]
    assert not offending, (f"commits since {BASE} fork {fork[:8]} change WIRE "
                           "files:\n" + "\n".join(offending))
    diff = _git("diff", "--stat", "HEAD", "--", *FROZEN)
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.strip() == "", f"uncommitted WIRE edits:\n{diff.stdout}"
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN)
    assert untracked.stdout.strip() == "", f"new files inside WIRE paths:\n{untracked.stdout}"
