"""Tests for ``RunPaths.create_tagged`` — the Phase B++ tagged-and-seeded
run-directory factory that aligns legacy output with the GRPO
``runs/<tag>/seed_<N>/`` pattern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mindforge.run_layout import RunPaths


def test_create_tagged_uses_default_legacy_root(tmp_path: Path, monkeypatch):
    """Default ``root`` resolves to ``./runs/legacy/<tag>/seed_<seed>/``."""
    monkeypatch.chdir(tmp_path)
    rp = RunPaths.create_tagged(tag="M1", seed=0)
    assert rp.root == Path("runs") / "legacy" / "M1" / "seed_0"
    assert rp.root.exists()
    assert rp.run_id == "M1/seed_0"


def test_create_tagged_with_custom_root(tmp_path: Path):
    rp = RunPaths.create_tagged(
        tag="L2", seed=42, root=tmp_path / "outputs",
    )
    assert rp.root == tmp_path / "outputs" / "L2" / "seed_42"
    assert rp.root.exists()


def test_create_tagged_creates_skeleton(tmp_path: Path):
    """``episodes/``, ``checkpoints/``, ``plots/`` all auto-created."""
    rp = RunPaths.create_tagged(tag="G4", seed=1, root=tmp_path)
    assert (rp.root / "episodes").is_dir()
    assert (rp.root / "checkpoints").is_dir()
    assert (rp.root / "plots").is_dir()


def test_create_tagged_idempotent(tmp_path: Path):
    """Calling twice with the same (tag, seed) returns a working RunPaths
    without raising on the already-existing dirs."""
    rp1 = RunPaths.create_tagged(tag="M3", seed=7, root=tmp_path)
    rp2 = RunPaths.create_tagged(tag="M3", seed=7, root=tmp_path)
    assert rp1.root == rp2.root
    assert rp1.run_id == rp2.run_id == "M3/seed_7"


def test_tagged_runs_isolated_per_seed(tmp_path: Path):
    """Different seeds produce sibling dirs under the same tag."""
    rp0 = RunPaths.create_tagged(tag="L1", seed=0, root=tmp_path)
    rp1 = RunPaths.create_tagged(tag="L1", seed=1, root=tmp_path)
    assert rp0.root.parent == rp1.root.parent     # same tag dir
    assert rp0.root != rp1.root                   # different seeds


def test_tagged_isolated_per_tag(tmp_path: Path):
    """Different tags produce sibling dirs under the root."""
    rp_m1 = RunPaths.create_tagged(tag="M1", seed=0, root=tmp_path)
    rp_l1 = RunPaths.create_tagged(tag="L1", seed=0, root=tmp_path)
    assert rp_m1.root.parent.parent == rp_l1.root.parent.parent
    assert rp_m1.root.parent != rp_l1.root.parent


def test_tagged_run_paths_subdirs_resolve():
    """``RunPaths`` accessors (episode_dir, hebbian_snapshots, etc.) still
    work after tagged construction — the dataclass surface is unchanged."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rp = RunPaths.create_tagged(tag="M1", seed=0, root=Path(tmp))
        assert rp.hebbian_snapshots == rp.root / "hebbian_snapshots.jsonl"
        assert rp.final_metrics_json == rp.root / "final_metrics.json"
        assert rp.episode_dir(3) == rp.root / "episodes" / "ep_0003"
        assert rp.metrics_dir == rp.root   # back-compat alias


def test_create_untagged_still_works(tmp_path: Path, monkeypatch):
    """Backwards compat: ``RunPaths.create(run_id, root)`` unchanged."""
    monkeypatch.chdir(tmp_path)
    rp = RunPaths.create("2026-01-01_test", root="custom_runs")
    assert rp.root == Path("custom_runs") / "2026-01-01_test"
    assert rp.run_id == "2026-01-01_test"
