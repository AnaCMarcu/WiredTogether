"""Weight-gated experience sharing (social replay, Eq. 7) — RL-side tests.

The mechanism has three parts, each pinned here:

1. ``HebbianSocialGraph.get_social_replay_indices`` turns bond weights into
   per-neighbour sample counts implementing the mixture
   (1-ρ)·D_i + ρ·Σ w̄ij·D_j  as counts over a pool that already holds all
   of D_i (m = n·ρ/(1-ρ) neighbour samples → social fraction exactly ρ).

2. ``ppo_update._collect_social_replay`` maps those indices onto neighbour
   RolloutBuffers, computing GAE on deep COPIES — at sampling time a
   neighbour's own update has not run yet, so its live buffer holds
   dataclass-default advantage=0.0 and must not be mutated.

3. Regression: before 2026-08-20 the sizes list passed to the graph put 0
   at the agent's own slot, so int(ρ·own_size)=0 and social replay never
   fired in ANY run (156 PPO updates in exp05/seed_42, zero replay lines).
"""

import numpy as np
import pytest

from hebbian.config import HebbianConfig
from hebbian.graph import HebbianSocialGraph
from rl_layer.ppo_update import _collect_social_replay
from rl_layer.trajectory_buffer import RolloutBuffer


# ── Fixtures ────────────────────────────────────────────────────────────────

def _graph(rho, W=None, n=3):
    cfg = HebbianConfig(enabled=True, num_agents=n, social_replay_rho=rho)
    g = HebbianSocialGraph(cfg)
    if W is not None:
        g.W = np.asarray(W, dtype=np.float32)
    return g


def _filled_buffer(n_transitions, reward=1.0, agent_tag="j"):
    """A rollout buffer with n stored transitions and NO GAE computed."""
    buf = RolloutBuffer(max_size=64)
    for t in range(n_transitions):
        buf.store_action(
            prompt_text=f"prompt {agent_tag}{t}", action_idx=0,
            log_prob=-1.0, value=0.5,
        )
        buf.store_reward(reward, done=False)
    return buf


class _FakeRL:
    """Just the attributes _collect_social_replay/_gae_on_copy touch."""

    def __init__(self, agent_id, own_size, centralized=False):
        from rl_layer.config import RLConfig
        self.agent_id = agent_id
        self.buffer = _filled_buffer(own_size, agent_tag="i")
        self.config = RLConfig()
        self._use_centralized = centralized
        self.centralized_critic = None


# ── 1. Graph-side mixture counts ────────────────────────────────────────────

def test_social_fraction_of_pool_is_rho():
    """With uniform bonds, the sampled count makes the pool exactly ρ social."""
    W = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    for rho in (0.1, 0.25, 0.3, 0.5):
        pairs = _graph(rho, W).get_social_replay_indices(0, [40, 40, 40])
        m, n = len(pairs), 40
        assert m / (n + m) == pytest.approx(rho, abs=0.05), f"rho={rho}"


def test_allocation_proportional_to_bond_weight():
    """Twice the bond → about twice the samples."""
    W = [[0.0, 0.2, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = _graph(0.3, W).get_social_replay_indices(0, [64, 64, 64])
    per_j = {j: sum(1 for _, jj in pairs if jj == j) for j in (1, 2)}
    assert per_j[2] == pytest.approx(2 * per_j[1], abs=2)


def test_rho_at_or_above_one_is_clamped_not_unbounded():
    W = [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    pairs = _graph(1.0, W).get_social_replay_indices(0, [10, 50, 50])
    # clamp to 0.9 → m = round(10·0.9/0.1) = 90, capped by buffer sizes.
    assert 0 < len(pairs) <= 100


def test_zero_own_buffer_yields_no_samples():
    W = [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert _graph(0.3, W).get_social_replay_indices(0, [0, 40, 40]) == []


# ── 2. Buffer-side collection with GAE-on-copy ──────────────────────────────

def _neighbourhood(rho=0.3, own=16, sizes=(16, 16)):
    rl = _FakeRL(agent_id=0, own_size=own)
    graph = _graph(rho, [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    neighbours = {
        1: _filled_buffer(sizes[0], reward=2.0, agent_tag="a"),
        2: _filled_buffer(sizes[1], reward=2.0, agent_tag="b"),
    }
    return rl, graph, neighbours


def test_collection_fires_with_own_size_at_own_slot():
    """The 2026-08-20 regression: neighbour-only sizes made own_size 0."""
    rl, graph, neighbours = _neighbourhood()
    out = _collect_social_replay(rl, neighbours, graph)
    assert len(out) > 0
    m, n = len(out), len(rl.buffer)
    assert m / (n + m) == pytest.approx(0.3, abs=0.07)


def test_sampled_transitions_have_gae_but_originals_are_untouched():
    rl, graph, neighbours = _neighbourhood()
    out = _collect_social_replay(rl, neighbours, graph)
    # Copies got real GAE: constant positive rewards on a 0.5 baseline give
    # strictly nonzero advantages/returns after per-rollout normalisation.
    assert any(tr.advantage != 0.0 for tr in out)
    assert all(tr.returns != 0.0 for tr in out)
    # The neighbours' live buffers still await their own compute_gae.
    for buf in neighbours.values():
        assert all(tr.advantage == 0.0 and tr.returns == 0.0
                   for tr in buf.get_all())


def test_collection_skips_empty_and_missing_neighbours():
    rl, graph, neighbours = _neighbourhood(sizes=(16, 16))
    neighbours[1] = RolloutBuffer()          # cleared by its earlier update
    out = _collect_social_replay(rl, neighbours, graph)
    assert len(out) > 0
    assert all(tr.prompt_text.startswith("prompt b") for tr in out)


def test_rho_zero_collects_nothing():
    rl, graph, neighbours = _neighbourhood(rho=0.0)
    assert _collect_social_replay(rl, neighbours, graph) == []


def test_no_graph_or_no_neighbours_is_a_noop():
    rl, graph, neighbours = _neighbourhood()
    assert _collect_social_replay(rl, neighbours, None) == []
    assert _collect_social_replay(rl, {}, graph) == []


def test_disabled_graph_collects_nothing():
    rl, _, neighbours = _neighbourhood()
    g = HebbianSocialGraph(HebbianConfig(
        enabled=False, num_agents=3, social_replay_rho=0.3))
    assert _collect_social_replay(rl, neighbours, g) == []


# ── 3. The pool the PPO loop actually sees ──────────────────────────────────

def test_extra_transitions_enter_the_minibatch_pool():
    rl, graph, neighbours = _neighbourhood()
    social = _collect_social_replay(rl, neighbours, graph)
    seen = [tr for batch in rl.buffer.sample_batches(4, extra_transitions=social)
            for tr in batch]
    assert len(seen) == len(rl.buffer) + len(social)
    assert any(not tr.prompt_text.startswith("prompt i") for tr in seen)


# ── 4. Step-scoped snapshots (update-order asymmetry regression) ────────────

def test_snapshot_survives_clear():
    buf = _filled_buffer(8)
    snap = buf.snapshot()
    buf.clear()
    assert len(buf) == 0
    assert len(snap) == 8
    assert all(tr.prompt_text.startswith("prompt j") for tr in snap.get_all())


def test_snapshot_shares_transitions_without_copy():
    """Shallow by design: the deep copy happens in _gae_on_copy, once, only
    for neighbours that were actually selected."""
    buf = _filled_buffer(4)
    snap = buf.snapshot()
    assert all(a is b for a, b in zip(buf.get_all(), snap.get_all()))
    assert snap.max_size == buf.max_size


def test_last_agent_still_gets_replay_from_snapshots():
    """Simulates the call-site sequence that used to starve agent N-1:
    agents update in index order and each update clears its live buffer.
    With per-step snapshots, the last agent still samples full neighbours."""
    graph = _graph(0.3, [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    live = {aid: _filled_buffer(16, agent_tag=f"a{aid}-") for aid in range(3)}

    # First updater takes the snapshot of everyone …
    snapshot = {aid: buf.snapshot() for aid, buf in live.items()}

    collected = {}
    for agent_id in range(3):  # … then updates run (and clear) in order.
        rl = _FakeRL(agent_id=agent_id, own_size=16)
        neighbours = {aid: s for aid, s in snapshot.items() if aid != agent_id}
        collected[agent_id] = _collect_social_replay(rl, neighbours, graph)
        live[agent_id].clear()  # what update() does to the live buffer

    for agent_id, out in collected.items():
        assert len(out) > 0, f"agent {agent_id} starved of social replay"
        tags = {tr.prompt_text.split("-")[0] for tr in out}
        assert f"prompt a{agent_id}" not in tags  # never samples itself


# ── 5. Staggered updates (Gemma env-idle hang mitigation) ───────────────────

def _stagger_rl(agent_id, n, stagger=True):
    from rl_layer.config import RLConfig
    from rl_layer.rl_layer import RLLayer
    rl = RLLayer.__new__(RLLayer)  # skip model loading; should_update only
    rl.config = RLConfig(enabled=True, update_interval=64,
                         update_stagger=stagger)
    rl.agent_id = agent_id
    rl.buffer = _filled_buffer(0)
    rl.buffer._buf = [None] * n  # only len() is consulted
    return rl


def test_stagger_shifts_threshold_by_agent_id():
    """Agent i fires at interval+i, so updates land on consecutive steps
    and the env never idles for the whole 3-agent round."""
    for aid in range(3):
        assert not _stagger_rl(aid, 64 + aid - 1).should_update()
        assert _stagger_rl(aid, 64 + aid).should_update()


def test_stagger_off_keeps_synchronous_updates():
    for aid in range(3):
        assert _stagger_rl(aid, 64, stagger=False).should_update()


def test_cycle_snapshot_survives_staggered_consumption():
    """Call-site protocol: first updater of a cycle takes the snapshot; it is
    retired only after every expected agent consumed it — so under stagger,
    agents updating on LATER steps still see the same pre-round view even
    though earlier updaters already cleared their live buffers."""
    graph = _graph(0.3, [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    live = {aid: _filled_buffer(16, agent_tag=f"a{aid}-") for aid in range(3)}

    snapshot, pending = None, set()
    collected = {}
    for agent_id in range(3):  # three consecutive steps, one updater each
        if snapshot is None:
            snapshot = {aid: buf.snapshot() for aid, buf in live.items()}
            pending = set(snapshot)
        rl = _FakeRL(agent_id=agent_id, own_size=16)
        neighbours = {a: s for a, s in snapshot.items() if a != agent_id}
        collected[agent_id] = _collect_social_replay(rl, neighbours, graph)
        live[agent_id].clear()
        live[agent_id].store_action(prompt_text="fresh", action_idx=0,
                                    log_prob=-1.0, value=0.5)
        pending.discard(agent_id)
        if not pending:
            snapshot = None

    assert snapshot is None  # cycle retired after the last consumer
    for agent_id, out in collected.items():
        assert len(out) > 0
        # Everyone sampled the pre-round view: 16-transition buffers, never
        # the post-clear "fresh" refills of earlier updaters.
        assert all(tr.prompt_text.startswith("prompt a") for tr in out)
