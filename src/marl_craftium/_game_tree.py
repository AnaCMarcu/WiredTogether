"""Share one read-only game tree per job instead of copying it per Luanti process.

craftium's ``_create_mt_dirs`` copies every entry of the env dir (``sync_dir``)
into the server's and each client's run dir. ``games/VoxeLibre`` alone is
6,234 files (85 MB), so a job with N agents creates (N+1) x 6k files before
its first step: ~70k at N=10, ~130k at N=20. Luanti never writes under
``games/`` (the game Lua, VoxeLibre's included, writes only under the world
path) and clients never read it (media come from the server), so a symlink
serves. Everything else (bin/, client/, worlds/, mods/, minetest.conf) is
still copied as upstream does.

Why it matters: on Snellius the scratch quota (3M soft / 4M hard inodes) is a
single pool over /scratch-shared AND the node-local $TMPDIR, so these copies
count even in a per-job work dir, and the end-of-job salvage rsync used to
mirror them into run_artifacts/ for good. That is how the 2026-10-09 inode
overflow killed every new N>=10 job with "[Errno 122] Disk quota exceeded"
in copytree.

``WT_COPY_GAMES=1`` restores the upstream copy.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from functools import wraps

LINKED = ("games",)


def copy_games() -> bool:
    """True when the operator asked for upstream's per-process copy."""
    return os.environ.get("WT_COPY_GAMES", "0") == "1"


@contextlib.contextmanager
def linking_copytree(sync_dir):
    """Inside the block, ``shutil.copytree`` of ``<sync_dir>/<LINKED>`` makes a symlink.

    craftium looks ``shutil.copytree`` up on the module at call time, so
    swapping the attribute for the duration of one ``_create_mt_dirs`` call
    redirects exactly the copies made there and nothing else.
    """
    real = shutil.copytree
    linked = {os.path.realpath(os.path.join(sync_dir, name)) for name in LINKED}

    def copytree(src, dst, *args, **kwargs):
        target = os.path.realpath(src)
        if target not in linked:
            return real(src, dst, *args, **kwargs)
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):          # a stale copy from dirs_exist_ok reuse
            shutil.rmtree(dst)
        os.symlink(target, dst)
        return dst

    shutil.copytree = copytree
    try:
        yield
    finally:
        shutil.copytree = real


def wrap_create_mt_dirs(orig):
    """``_create_mt_dirs`` that links the game tree; upstream body untouched."""

    @wraps(orig)
    def _create_mt_dirs(self, root_dir, target_dir, sync_dir=None):
        if sync_dir is None or copy_games():
            return orig(self, root_dir, target_dir, sync_dir)
        with linking_copytree(sync_dir):
            return orig(self, root_dir, target_dir, sync_dir)

    _create_mt_dirs._wt_links_games = True
    return _create_mt_dirs


def install(minetest_module) -> list[str]:
    """Patch the server and client classes of ``craftium.minetest`` (idempotent)."""
    done = []
    for name in ("MTServerOnly", "MTClientOnly"):
        cls = getattr(minetest_module, name, None)
        orig = getattr(cls, "_create_mt_dirs", None)
        if orig is None or getattr(orig, "_wt_links_games", False):
            continue
        cls._create_mt_dirs = wrap_create_mt_dirs(orig)
        done.append(name)
    return done
