"""Multi-agent Craftium env wrapper + HPC patches.

Public API:
    from marl_craftium import OpenWorldMultiAgentEnv

The package also re-exports ``_PatchedMarlCraftiumEnv`` and the action mapping
helpers for tests / advanced consumers, but in normal use you only need the
``OpenWorldMultiAgentEnv`` class.

Imports trigger ``_bootstrap``'s side effect of putting the in-tree
``WiredTogether/craftium/`` parent directory onto ``sys.path`` so that
``import craftium`` resolves to the real package even when ``pip install -e
./craftium`` hasn't been run on the current node.
"""

from . import _bootstrap  # noqa: F401  (must run before any `from craftium...`)

from marl_craftium.openworld_multi_agents import OpenWorldMultiAgentEnv
from marl_craftium._actions import _DISCRETE_ACTIONS, _discrete_to_dict
from marl_craftium._patched_env import _PatchedMarlCraftiumEnv

__all__ = [
    "OpenWorldMultiAgentEnv",
    "_PatchedMarlCraftiumEnv",
    "_DISCRETE_ACTIONS",
    "_discrete_to_dict",
]
