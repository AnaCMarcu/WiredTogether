"""Ensure the in-tree craftium submodule is importable as the top-level
``craftium`` package.

Import this module **before** any ``from craftium...`` import. With ``pip
install -e ./craftium`` Python finds the real package via site-packages, but
when pip-install hasn't been run (or is broken on a compute node) Python
binds ``craftium`` to the empty namespace package at ``WiredTogether/craftium/``
and ``root_path`` (defined only in the real ``craftium/craftium/__init__.py``)
goes missing — exactly the
"cannot import name 'root_path' from 'craftium' (unknown location)" failure.
"""

import os
import sys

_this_file = os.path.abspath(__file__)
# src/marl_craftium/_bootstrap.py → project root is three dirs up.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))
_real_craftium_parent = os.path.join(_project_root, "craftium")

if (
    os.path.isfile(os.path.join(_real_craftium_parent, "craftium", "__init__.py"))
    and _real_craftium_parent not in sys.path
):
    sys.path.insert(0, _real_craftium_parent)
