"""PettingZoo ParallelEnv wrapper around the patched Craftium MARL env."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Side-effect: ensure the in-tree craftium submodule is importable as the
# top-level `craftium` package before we try to import from it.
from . import _bootstrap  # noqa: F401

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from marl_craftium._actions import _discrete_to_dict
from marl_craftium._patched_env import _PatchedMarlCraftiumEnv


class OpenWorldMultiAgentEnv(ParallelEnv):
    """Multi-agent OpenWorld environment with the PettingZoo ParallelEnv API.

    Adapts Craftium's ``MarlCraftiumEnv`` to PettingZoo's ParallelEnv interface
    so it works with multi-agent RL libraries like RLlib, CleanRL, etc.

    Args:
        num_agents: Number of agents in the environment.
        obs_width / obs_height: Observation size in pixels.
        max_steps: Episode horizon (in main-loop rounds).
        task_focus: Optional task tag for reward shaping (``"mining"``,
                    ``"crafting"``, ``"building"``, ``"exploration"``).
        render_mode: ``"rgb_array"`` or None.
        seed: ``fixed_map_seed`` for the underlying world generator.
        frameskip: Frames per Craftium step.
        pmul: Physics multiplier (movement speed scale).
    """

    metadata = {"render_modes": ["rgb_array"], "name": "openworld_multi_agent_v0"}

    def __init__(
        self,
        num_agents: int = 2,
        obs_width: int = 480,
        obs_height: int = 480,
        max_steps: int = 10000,
        task_focus: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        frameskip: int = 3,
        pmul: int = 20,
    ):
        super().__init__()

        # Use _num_agents to avoid conflict with PettingZoo's num_agents property.
        self._num_agents = num_agents
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.max_steps = max_steps
        self.task_focus = task_focus
        self.render_mode = render_mode
        self.frameskip = frameskip
        self.pmul = pmul

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        self.env = _PatchedMarlCraftiumEnv(
            **self._build_marl_kwargs(num_agents, obs_width, obs_height,
                                      max_steps, seed, frameskip, pmul),
        )

        self._observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8,
        )
        # Craftium discrete actions: 0=NOP + 22 named actions.
        self._action_space = spaces.Discrete(23)
        self._step_count = 0

    # ─── Construction helpers ─────────────────────────────────────────

    @staticmethod
    def _build_marl_kwargs(num_agents, obs_width, obs_height, max_steps,
                           seed, frameskip, pmul) -> Dict[str, Any]:
        """Resolve env_dir + minetest_dir and pack the kwargs for the patched env."""
        from craftium import root_path as craftium_root

        minetest_dir = os.environ.get(
            "CRAFTIUM_LUANTI_DIR", os.path.join(craftium_root, "luanti"),
        )

        # env_dir resolution order:
        #   1. $CRAFTIUM_ENV_DIR (set by the SLURM scripts to five-chambers)
        #   2. installed package's craftium-envs/voxel-libre2
        #   3. local repo's craftium submodule fallback
        _this_file = os.path.abspath(__file__)
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_this_file)))
        _pkg_env = os.path.join(craftium_root, "craftium-envs", "voxel-libre2")
        _local_env = os.path.join(_project_root, "craftium", "craftium-envs", "voxel-libre2")
        env_dir = (
            os.environ.get("CRAFTIUM_ENV_DIR")
            or (_pkg_env if os.path.isdir(_pkg_env) else _local_env)
        )

        return dict(
            env_dir=env_dir,
            game_id="VoxeLibre",
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            # CraftiumEnvironmentInterface calls env.step() once per agent per
            # round; each env.step() runs one step_agent round which (after our
            # fix) increments timesteps once. So timesteps = num_agents * rounds.
            max_timesteps=max_steps * num_agents,
            minetest_dir=minetest_dir,
            mt_listen_timeout=300_000,  # 5 min per client; VoxeLibre loads slowly on HPC
            seed=seed,
            frameskip=frameskip,
            pmul=pmul,
            # Smallest engine-allowed HUD scale so hearts/food bar don't eat half
            # the 320×180 observation frame.
            mt_clients_conf={"hud_scaling": 0.5},
        )

    # ─── PettingZoo properties ────────────────────────────────────────

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        return {agent: self._action_space for agent in self.possible_agents}

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    # ─── Lifecycle ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            np.random.seed(seed)

        obs_array, _ = self.env.reset()
        self.agents = self.possible_agents.copy()
        self._step_count = 0

        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_array)}
        infos = {f"agent_{i}": {} for i in range(self._num_agents)}
        return observations, infos

    def step(
        self, actions: Dict[str, int],
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self._step_count += 1

        # Always send actions for ALL agents — MarlCraftiumEnv expects exactly
        # num_agents actions every step. NoOp for terminated agents.
        action_list = [
            _discrete_to_dict(actions.get(agent, 0))
            for agent in self.possible_agents
        ]
        obs_dict, reward_dict, done_dict, truncated_dict, _ = self.env.step(action_list)

        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_dict)}
        rewards      = {f"agent_{i}": float(r) for i, r in enumerate(reward_dict)}
        terminations = {f"agent_{i}": bool(d) for i, d in enumerate(done_dict)}
        truncations  = {f"agent_{i}": bool(t) for i, t in enumerate(truncated_dict)}
        infos        = {f"agent_{i}": {} for i in range(self._num_agents)}

        rewards = self._add_exploration_bonus(rewards)
        if self.task_focus:
            rewards = self._apply_task_focus(rewards, infos)

        # Drop terminated agents from the active list (PettingZoo convention).
        self.agents = [a for a in self.agents if not terminations.get(a, False)]
        return observations, rewards, terminations, truncations, infos

    def warmup_noop(self):
        """NoOp every channel without advancing the step counter; returns obs list."""
        return self.env.warmup_noop()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def close(self):
        self.env.close()

    # ─── Reward shaping ───────────────────────────────────────────────

    def _add_exploration_bonus(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Small per-step XZ-distance bonus (~0.1/node) — discourages standing still
        without overwhelming milestone rewards (dig=1.0, stage=128+)."""
        for i in range(self._num_agents):
            prev = self.env._prev_pos[i]
            curr = self.env._positions[i]
            if prev is None or curr is None:
                continue
            dx = curr[0] - prev[0]
            dz = curr[2] - prev[2]
            rewards[f"agent_{i}"] += 0.1 * float(np.sqrt(dx * dx + dz * dz))
        return rewards

    _TASK_BONUSES = {
        "mining":      {"mined_blocks":      1.0},
        "crafting":    {"items_crafted":     2.0},
        "building":    {"blocks_placed":     1.0},
        "exploration": {"distance_traveled": 0.1},
    }

    def _apply_task_focus(
        self, rewards: Dict[str, float], infos: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Boost rewards based on task-specific metrics in info dicts."""
        bonuses = self._TASK_BONUSES.get(self.task_focus)
        if bonuses is None:
            return rewards
        shaped = rewards.copy()
        for agent, info in infos.items():
            bonus = sum(info[m] * w for m, w in bonuses.items() if m in info)
            shaped[agent] = rewards[agent] + bonus
        return shaped
