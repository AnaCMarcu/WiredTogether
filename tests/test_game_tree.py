"""Shared game tree (marl_craftium._game_tree).

craftium copies the env dir into every Luanti run dir; the wrapper turns the
``games/`` copy into a symlink and leaves every other copy alone. Pins:

  * games/ is linked, bin/ client/ worlds/ mods/ and files are still copied;
  * WT_COPY_GAMES=1 and a missing sync_dir fall through to upstream verbatim;
  * shutil.copytree is restored after the call, even when upstream raises;
  * install() wraps the two classes once and skips a module without them.
"""

import os
import shutil
import types

import pytest

from marl_craftium import _game_tree as gt


class FakeMT:
    """Upstream's ``_create_mt_dirs`` body, verbatim (craftium/minetest.py)."""

    def _create_mt_dirs(self, root_dir, target_dir, sync_dir=None):
        def link_dir(name):
            os.symlink(os.path.join(root_dir, name), os.path.join(target_dir, name))

        def copy_dir(name):
            shutil.copytree(os.path.join(root_dir, name),
                            os.path.join(target_dir, name), dirs_exist_ok=True)

        for name in ("builtin", "fonts", "locale", "textures"):
            link_dir(name)
        copy_dir("bin")
        copy_dir("client")
        if sync_dir is not None:
            for item in os.listdir(sync_dir):
                src = os.path.join(sync_dir, item)
                tgt = os.path.join(target_dir, item)
                if os.path.isfile(src):
                    shutil.copy(src, tgt)
                else:
                    shutil.copytree(src, tgt, dirs_exist_ok=True)
        else:
            copy_dir("worlds")
            copy_dir("games")


UPSTREAM = FakeMT.__dict__["_create_mt_dirs"]


def _tree(tmp_path):
    root = tmp_path / "luanti"
    for name in ("builtin", "fonts", "locale", "textures", "bin", "client", "worlds", "games"):
        (root / name).mkdir(parents=True)
        (root / name / "x.txt").write_text(name)
    sync = tmp_path / "wire"
    for name in ("games", "worlds", "mods", "clientmods"):
        (sync / name).mkdir(parents=True)
        (sync / name / "a.txt").write_text(name)
    (sync / "games" / "VoxeLibre").mkdir()
    (sync / "games" / "VoxeLibre" / "game.conf").write_text("name = VoxeLibre")
    (sync / "minetest.conf").write_text("port = 1")
    run = tmp_path / "run"
    run.mkdir()
    return root, sync, run


@pytest.fixture
def symlinks(monkeypatch):
    """Record symlinks instead of making them (Windows needs a privilege)."""
    made = []

    def fake_symlink(src, dst, *a, **k):
        made.append((os.path.realpath(src), str(dst)))
        os.makedirs(dst, exist_ok=True)      # so a later listdir sees something there

    monkeypatch.setattr(os, "symlink", fake_symlink)
    return made


def test_games_linked_everything_else_copied(tmp_path, symlinks):
    root, sync, run = _tree(tmp_path)
    gt.wrap_create_mt_dirs(UPSTREAM)(FakeMT(), str(root), str(run), str(sync))
    linked = {dst for _, dst in symlinks}
    assert (os.path.realpath(sync / "games"), str(run / "games")) in symlinks
    # the upstream engine links are untouched
    assert {str(run / n) for n in ("builtin", "fonts", "locale", "textures")} <= linked
    # real copies for everything that is written at runtime or per process
    assert (run / "bin" / "x.txt").read_text() == "bin"
    assert (run / "client" / "x.txt").read_text() == "client"
    assert (run / "worlds" / "a.txt").read_text() == "worlds"
    assert (run / "mods" / "a.txt").read_text() == "mods"
    assert (run / "clientmods" / "a.txt").read_text() == "clientmods"
    assert (run / "minetest.conf").read_text() == "port = 1"
    assert not (run / "games" / "VoxeLibre").exists()      # not copied


def test_opt_out_and_no_sync_dir_use_upstream(tmp_path, symlinks, monkeypatch):
    root, sync, run = _tree(tmp_path)
    wrapped = gt.wrap_create_mt_dirs(UPSTREAM)
    monkeypatch.setenv("WT_COPY_GAMES", "1")
    wrapped(FakeMT(), str(root), str(run), str(sync))
    assert (run / "games" / "VoxeLibre" / "game.conf").exists()   # a real copy
    assert all(dst != str(run / "games") for _, dst in symlinks)

    monkeypatch.delenv("WT_COPY_GAMES")
    run2 = tmp_path / "run2"
    run2.mkdir()
    wrapped(FakeMT(), str(root), str(run2), None)
    assert (run2 / "games" / "x.txt").read_text() == "games"      # upstream fallback copy


def test_copytree_restored_even_on_error(tmp_path, symlinks):
    root, sync, run = _tree(tmp_path)
    real = shutil.copytree

    def boom(self, root_dir, target_dir, sync_dir=None):
        assert shutil.copytree is not real          # swapped inside the call
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError):
        gt.wrap_create_mt_dirs(boom)(FakeMT(), str(root), str(run), str(sync))
    assert shutil.copytree is real


def test_install_wraps_once_and_tolerates_missing_classes():
    class Srv:
        def _create_mt_dirs(self, root_dir, target_dir, sync_dir=None):
            return "srv"

    class Cli:
        def _create_mt_dirs(self, root_dir, target_dir, sync_dir=None):
            return "cli"

    mod = types.SimpleNamespace(MTServerOnly=Srv, MTClientOnly=Cli)
    assert gt.install(mod) == ["MTServerOnly", "MTClientOnly"]
    assert gt.install(mod) == []                                  # idempotent
    assert Srv._create_mt_dirs._wt_links_games
    assert Cli()._create_mt_dirs("r", "t", None) == "cli"         # still calls upstream
    assert gt.install(types.SimpleNamespace()) == []
