"""Comm-augmented Level-Based Foraging.

Wraps any standard ``lbforaging:Foraging-*-v3`` env with a discrete,
targeted signal channel. The wrapper:

- Expands each agent's action space from ``Discrete(6)`` to
  ``Discrete(6 + (N − 1))``. Action indices ``6..6 + N − 2`` mean
  "signal the k-th non-self teammate (0-indexed)".
- Augments each agent's observation with ``(N − 1)`` binary flags
  indicating "did teammate j signal me on the *previous* step?"
- Returns ``info['comm_events']`` — list of ``(sender, receiver)`` index
  pairs emitted this step — and ``info['positions']`` — list of
  ``(x, y, 0.0)`` triples per agent (z=0 because LBF is 2D, but the
  Hebbian spatial gate is 3D-shaped).

Signal actions are mutually exclusive with movement: when an agent picks
a signal action, the underlying env receives ``NoOp`` (action 0) for
that agent's movement. This is the "opportunity cost" design — see
HEBBIAN_MARL_PLAN.md §1.

Registration
------------
For each standard lbforaging size/players/foods/version combination, this
module registers a bare env id of the form
``"Foraging-Comm-WxH-Np-Mf-vV"`` that yields
``LBFCommWrapper(lbforaging_env)``. To trigger registration, this module
must be imported before ``gym.make`` is called on one of those ids —
``epymarl/src/envs/__init__.py`` does that import.

Usage from EPyMARL CLI:

    --env-config=gymma \\
        with env_args.key="Foraging-Comm-10x10-3p-3f-v3"
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LBFCommWrapper(gym.Wrapper):
    """Discrete targeted-signal comm channel for Level-Based Foraging."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        N = env.unwrapped.n_agents
        self.n_agents = N

        # Expanded per-agent action space
        new_n = 6 + (N - 1)
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(new_n) for _ in range(N))
        )

        # Extended per-agent observation: append (N-1) binary signal-received flags
        new_obs_spaces = []
        for s in env.observation_space:
            assert isinstance(s, spaces.Box), (
                f"LBFCommWrapper expects each per-agent obs to be Box, got {type(s)}"
            )
            low = np.concatenate(
                [s.low.astype(np.float32), np.zeros(N - 1, dtype=np.float32)]
            )
            high = np.concatenate(
                [s.high.astype(np.float32), np.ones(N - 1, dtype=np.float32)]
            )
            new_obs_spaces.append(
                spaces.Box(low=low, high=high, dtype=np.float32)
            )
        self.observation_space = spaces.Tuple(tuple(new_obs_spaces))

        # _next_signal_flags[i] holds the (N-1) flags that agent i will see on
        # its *next* observation. Updated at the end of step() based on this
        # step's comm_events.
        self._next_signal_flags = np.zeros((N, N - 1), dtype=np.float32)

    # ── action-index helpers ──

    def _teammate_idx(self, sender: int, offset: int) -> int:
        """Map (sender, signal_offset) → absolute teammate index.

        ``offset == 0`` selects the first non-self agent in index order.
        """
        teammates = [k for k in range(self.n_agents) if k != sender]
        return teammates[offset]

    def _signal_flag_slot(self, receiver: int, sender: int) -> int:
        """Where does sender's signal land in the receiver's (N-1) flags?"""
        teammates = [k for k in range(self.n_agents) if k != receiver]
        return teammates.index(sender)

    # ── obs augmentation ──

    def _augment_obs(self, base_obs):
        N = self.n_agents
        new_obs = []
        for i in range(N):
            o = np.asarray(base_obs[i], dtype=np.float32).ravel()
            flags = self._next_signal_flags[i].astype(np.float32)
            new_obs.append(np.concatenate([o, flags]))
        return tuple(new_obs)

    # ── position read ──

    def _read_positions(self):
        """Read (x, y, 0.0) positions per agent from lbforaging player objects.

        lbforaging stores ``player.position`` as a 2-tuple (row, col). We
        widen to 3D (z=0.0) to match the Hebbian spatial-gate API.
        """
        players = self.env.unwrapped.players
        out = []
        for p in players:
            pos = p.position
            out.append((float(pos[0]), float(pos[1]), 0.0))
        return out

    # ── gym API ──

    def reset(self, *, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        # Gymnasium API: (obs, info)
        base_obs, info = out
        N = self.n_agents
        self._next_signal_flags = np.zeros((N, N - 1), dtype=np.float32)
        info = dict(info) if info else {}
        info["comm_events"] = []
        info["positions"] = self._read_positions()
        return self._augment_obs(base_obs), info

    def step(self, actions):
        N = self.n_agents

        underlying_actions: List[int] = []
        comm_events: List[Tuple[int, int]] = []
        for i, a in enumerate(actions):
            a_int = int(a)
            if a_int < 6:
                underlying_actions.append(a_int)
            else:
                offset = a_int - 6
                receiver = self._teammate_idx(i, offset)
                comm_events.append((i, receiver))
                underlying_actions.append(0)  # NoOp for movement (opportunity cost)

        out = self.env.step(tuple(underlying_actions))
        # Gymnasium API: (obs, reward, terminated, truncated, info)
        base_obs, reward, terminated, truncated, info = out
        info = dict(info) if info else {}

        # Build the signal flags the NEXT obs will see.
        next_flags = np.zeros((N, N - 1), dtype=np.float32)
        for sender, receiver in comm_events:
            slot = self._signal_flag_slot(receiver, sender)
            next_flags[receiver, slot] = 1.0
        self._next_signal_flags = next_flags

        info["comm_events"] = comm_events
        info["positions"] = self._read_positions()
        return self._augment_obs(base_obs), reward, terminated, truncated, info


# ───────────────────────────────────────────────────────────────────────
# Gym env registration
# ───────────────────────────────────────────────────────────────────────


def _make_comm_lbf(
    size: int,
    players: int,
    foods: int,
    version: int = 3,
    coop: bool = False,
    **kwargs,
):
    """Factory used as a gym entry_point: build lbforaging env, then wrap.

    When ``coop=True`` we pass ``force_coop=True`` through to the underlying
    ``ForagingEnv`` constructor. This sets every food's level to N (number
    of players) in every episode, making cooperation strongly necessary
    (a level-N agent could in principle solo-load, but rarely happens to
    be both present AND adjacent in practice).
    """
    inner_id = f"Foraging-{size}x{size}-{players}p-{foods}f-v{version}"
    if coop:
        kwargs = {"force_coop": True, **kwargs}
    inner = gym.make(f"lbforaging:{inner_id}", **kwargs)
    return LBFCommWrapper(inner)


def _register_comm_envs():
    """Register comm-augmented variants as bare env ids.

    Registers both standard and ``-coop`` variants for each size/players/
    foods combination. The coop variant passes ``force_coop=True`` to the
    underlying lbforaging env via gym.make kwargs, independent of whether
    lbforaging itself registers a coop env for that specific combination.
    """
    import lbforaging  # noqa: F401  ensure lbforaging's own register() ran first

    for size, players, foods, coop in itertools.product(
        range(5, 21), range(2, 5), range(1, 7), (False, True)
    ):
        suffix = "-coop" if coop else ""
        full_id = f"Foraging-Comm-{size}x{size}-{players}p-{foods}f{suffix}-v3"
        # Skip if already registered (e.g. on re-import)
        if full_id in gym.envs.registry:
            continue
        gym.register(
            id=full_id,
            entry_point=(
                lambda s=size, p=players, f=foods, c=coop, **kw:
                _make_comm_lbf(s, p, f, coop=c, **kw)
            ),
        )


_register_comm_envs()
