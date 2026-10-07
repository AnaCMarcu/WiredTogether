"""The AgentWorld work must not change the WIRE stack (paper reproducibility).

Fails if any WIRE-side file differs from the commit this branch forked from
``world-scaling`` (the merge base, so new commits on world-scaling itself do
not trip it). Skipped where that branch is not available (e.g. a cluster
checkout that only has this branch).
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


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_wire_stack_unchanged():
    if _git("rev-parse", "--verify", "--quiet", BASE).returncode != 0:
        pytest.skip(f"branch {BASE} not available")
    fork = _git("merge-base", "HEAD", BASE).stdout.strip()
    if not fork:
        pytest.skip(f"no common history with {BASE}")
    diff = _git("diff", "--stat", fork, "--", *FROZEN)
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout.strip() == "", f"WIRE files changed vs {BASE} fork {fork[:8]}:\n{diff.stdout}"
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN)
    assert untracked.stdout.strip() == "", f"new files inside WIRE paths:\n{untracked.stdout}"
