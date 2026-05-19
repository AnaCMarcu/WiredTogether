"""Module-level singleton holding the active HebbianSocialGraph instance.

EPyMARL constructs the runner and the learner separately, with no
built-in handle for one to reach the other. The Hebbian graph lives in
the runner (advanced after every env step) but is needed by the learner
(for ``W̄[i,j]`` weighting in shared-experience sampling). This singleton
gives both sides a stable reference.

Use:
    from hebbian_module.runtime import set_graph, get_graph

    # In HebbianRunner.__init__:
    set_graph(self.hebbian)

    # In HebbianSEACLearner.train():
    graph = get_graph()
    w_bar_i = graph.get_normalized_weights(i)  if graph is not None else None
"""

from typing import Optional

from .graph import HebbianSocialGraph

_GRAPH: Optional[HebbianSocialGraph] = None


def set_graph(graph: HebbianSocialGraph) -> None:
    """Register the active Hebbian graph (typically called by the runner)."""
    global _GRAPH
    _GRAPH = graph


def get_graph() -> Optional[HebbianSocialGraph]:
    """Return the active Hebbian graph, or None if no runner has registered one."""
    return _GRAPH


def clear_graph() -> None:
    """Drop the registered graph (mostly for tests)."""
    global _GRAPH
    _GRAPH = None
