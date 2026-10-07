"""AgentWorld analysis scripts on synthetic matrices and on real CLI output."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis" / "agentworld"))
import make_aw_progress_curves as curves  # noqa: E402
import make_bond_recovery as rec  # noqa: E402

from agentworld import cli  # noqa: E402
from test_aw_cli import _args, fake_root  # noqa: E402,F401


def test_auc_basics():
    assert rec.auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([1, 1, 0, 0])) == 1.0
    assert rec.auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0])) == 0.0
    assert rec.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 0, 0])) == 0.5
    assert np.isnan(rec.auc(np.array([1.0]), np.array([1])))


def test_precision_at_k():
    same = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    perfect = same * 0.9 + 0.1 * (1 - same)
    assert rec.precision_at_k(perfect, same) == 1.0
    assert rec.precision_at_k(1 - perfect, same) == 0.0


@pytest.fixture
def runs(fake_root, tmp_path):
    root, _ = fake_root
    out = tmp_path / "group"
    for arm in ("base", "hebbian"):
        args = _args(root, out / arm, "--replicas", "3", "--team-spacing", "20",
                     "--arm", arm, "--video", "none")
        asyncio.run(cli.run(args))
    return out


def test_bond_recovery_on_cli_output(runs, capsys):
    rec.main([str(runs), "--out", str(runs / "rec.csv")])
    text = (runs / "rec.csv").read_text()
    assert text.count("\n") == 3              # header + base + hebbian
    row = next(r for r in __import__("csv").DictReader(open(runs / "rec.csv"))
               if r["arm"] == "hebbian")
    assert int(row["n_agents"]) == 9
    assert float(row["auc_W"]) > 0.9            # bonds recover the hidden teams...
    # ...but so does proximity alone: each team spawns as a cluster. This is
    # the confound the spatial baseline exists to expose (jitter spawns or use
    # specialisation worlds before claiming W adds anything).
    assert float(row["auc_spatial"]) > 0.9
    # Raw counts miss teammates who never interact directly (lumberjack–hunter).
    assert 0.5 < float(row["auc_counts"]) < 1.0


def test_progress_curves_on_cli_output(runs):
    out = runs / "fig.png"
    curves.main([str(runs), "--rounds", "12", "--out", str(out)])
    assert out.is_file() and out.with_suffix(".pdf").is_file()
    got = curves.collect([runs], "progress", 1, 12)
    assert set(got) == {(9, "base"), (9, "hebbian")}
    assert all(c[-1] == 1.0 for runs_ in got.values() for c in runs_)   # all teams solve
