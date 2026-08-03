"""Run-group plumbing: runs/<group>/<tag>/seed_<N>/ + W&B id namespacing.

Both halves used to be missing, with one shared symptom: every suite wrote to
runs/legacy/ and reused the same W&B id, so a new suite overwrote the previous
one on disk and silently resumed its W&B run instead of creating a new one.
"""

import sys
from pathlib import Path

import pytest

SRC_MINDFORGE = Path(__file__).resolve().parents[1] / "src" / "mindforge"
if str(SRC_MINDFORGE) not in sys.path:
    sys.path.insert(0, str(SRC_MINDFORGE))

from run_layout import DEFAULT_RUN_GROUP, RunPaths, resolve_run_group  # noqa: E402
import wandb_logger  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No ambient run-group/root config leaking in from the shell."""
    monkeypatch.delenv("WIREDTOGETHER_RUN_GROUP", raising=False)
    monkeypatch.delenv("WIREDTOGETHER_RUNS_ROOT", raising=False)


# ── group resolution ────────────────────────────────────────────────────────
def test_group_defaults_to_legacy():
    assert resolve_run_group() == "legacy"
    assert DEFAULT_RUN_GROUP == "legacy"


def test_group_from_env(monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUN_GROUP", "gemma4")
    assert resolve_run_group() == "gemma4"


def test_explicit_group_beats_env(monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUN_GROUP", "gemma4")
    assert resolve_run_group("final") == "final"


@pytest.mark.parametrize("raw", ["", "   ", "/", None])
def test_blank_group_falls_back_to_default(raw, monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUN_GROUP", "")
    assert resolve_run_group(raw) == DEFAULT_RUN_GROUP


def test_group_strips_surrounding_slashes():
    assert resolve_run_group("/medium2k/") == "medium2k"


# ── on-disk layout ──────────────────────────────────────────────────────────
def test_tagged_layout_defaults_to_legacy_subtree(tmp_path, monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUNS_ROOT", str(tmp_path))
    rp = RunPaths.create_tagged(tag="exp04_ippo", seed=42)
    assert rp.root == tmp_path / "legacy" / "exp04_ippo" / "seed_42"
    assert rp.group == "legacy"
    assert rp.run_id == "exp04_ippo/seed_42"
    assert rp.episodes_dir.is_dir()


def test_tagged_layout_honours_group(tmp_path, monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUNS_ROOT", str(tmp_path))
    rp = RunPaths.create_tagged(tag="exp04_ippo", seed=42, group="gemma4")
    assert rp.root == tmp_path / "gemma4" / "exp04_ippo" / "seed_42"
    assert rp.group == "gemma4"
    # run_id stays tag/seed — downstream tooling keys on it.
    assert rp.run_id == "exp04_ippo/seed_42"


def test_tagged_layout_honours_group_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WIREDTOGETHER_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("WIREDTOGETHER_RUN_GROUP", "medium2k")
    rp = RunPaths.create_tagged(tag="exp05_mappo_hebbian", seed=123)
    assert rp.root == tmp_path / "medium2k" / "exp05_mappo_hebbian" / "seed_123"


def test_groups_do_not_collide_on_disk(tmp_path, monkeypatch):
    """The regression: same exp+seed in two suites must be two directories."""
    monkeypatch.setenv("WIREDTOGETHER_RUNS_ROOT", str(tmp_path))
    a = RunPaths.create_tagged(tag="exp04_ippo", seed=42, group="medium2k")
    b = RunPaths.create_tagged(tag="exp04_ippo", seed=42, group="gemma4")
    assert a.root != b.root


def test_explicit_root_is_used_verbatim(tmp_path):
    """An explicit root says exactly where the tag tree goes; the group is
    still recorded because it also namespaces the W&B id."""
    rp = RunPaths.create_tagged(
        tag="exp04_ippo", seed=42, root=tmp_path, group="gemma4",
    )
    assert rp.root == tmp_path / "exp04_ippo" / "seed_42"
    assert rp.group == "gemma4"


# ── W&B id namespacing ──────────────────────────────────────────────────────
def test_legacy_group_keeps_the_historical_id():
    """Runs recorded before groups existed must still resume into themselves."""
    for group in (None, "", "legacy"):
        assert wandb_logger._scoped_id("exp04_ippo/seed_42", group) == "exp04_ippo_seed_42"


def test_other_groups_are_prefixed():
    assert (wandb_logger._scoped_id("exp04_ippo/seed_42", "gemma4")
            == "gemma4_exp04_ippo_seed_42")


def test_ids_differ_across_groups():
    """Two suites, same exp+seed, same project -> two distinct W&B runs."""
    a = wandb_logger._scoped_id("exp11_llm_9b_allied_none/seed_456", "medium")
    b = wandb_logger._scoped_id("exp11_llm_9b_allied_none/seed_456", "medium2k")
    assert a != b


def test_scoped_id_stays_within_wandb_charset_and_length():
    wb_id = wandb_logger._scoped_id("exp09_llm_9b_allied_all/seed_1213", "gemma4")
    assert len(wb_id) <= 64
    assert all(c.isalnum() or c in "_-" for c in wb_id)


def test_scoped_name_prefixes_only_non_legacy():
    assert wandb_logger._scoped_name("exp04_ippo/seed_42", "legacy") == "exp04_ippo/seed_42"
    assert (wandb_logger._scoped_name("exp04_ippo/seed_42", "gemma4")
            == "gemma4/exp04_ippo/seed_42")
