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
        voxel_obs: bool = False,
        voxel_obs_rx: int = 10,
        voxel_obs_ry: int = 5,
        voxel_obs_rz: int = 10,
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
        # T2.1 voxel observation opt-in. Default OFF so existing runs are
        # unaffected. Radii chosen for the five-chambers world: 10/5/10
        # covers a 21×11×21 grid (~5k cells) — plenty of context for
        # the LLM-grounding use case without blowing TCP throughput
        # (default Craftium radii are 20/10/20, ~4× more data).
        self.voxel_obs = voxel_obs
        self.voxel_obs_rx = voxel_obs_rx
        self.voxel_obs_ry = voxel_obs_ry
        self.voxel_obs_rz = voxel_obs_rz

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        _marl_kwargs = self._build_marl_kwargs(
            num_agents, obs_width, obs_height, max_steps, seed, frameskip, pmul,
        )
        if voxel_obs:
            _marl_kwargs.update(
                voxel_obs=True,
                voxel_obs_rx=voxel_obs_rx,
                voxel_obs_ry=voxel_obs_ry,
                voxel_obs_rz=voxel_obs_rz,
            )
        self.env = _PatchedMarlCraftiumEnv(**_marl_kwargs)

        self._observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8,
        )
        # Craftium discrete actions: 0=NOP + 22 named actions.
        self._action_space = spaces.Discrete(23)

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

        # Unique MT server port per job, INDEPENDENT of the reproducibility
        # seed. craftium falls back to random.randint(49152, 65535) when
        # mt_server_port is None, but multi_agent_craftium.py calls
        # random.seed(seed) for reproducibility — so two jobs with the SAME
        # seed draw the SAME "random" port (e.g. 52800 for seed 42) and collide
        # ("Server socket listen timeout") when they land on the same node.
        # Derive the port from SLURM_JOB_ID (unique per job), falling back to an
        # UNSEEDED SystemRandom draw for local/non-SLURM runs.
        _jobid = os.environ.get("SLURM_JOB_ID", "")
        if _jobid.isdigit():
            _mt_port = 49152 + int(_jobid) % 16000
        else:
            import random as _random
            _mt_port = _random.SystemRandom().randint(49152, 65535)

        return dict(
            mt_server_port=_mt_port,
            env_dir=env_dir,
            game_id="VoxeLibre",
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            # Disable env-side step-count truncation. The old setting was
            # `max_steps * num_agents`, written before the _patched_env fix
            # that made `self.timesteps` advance once per ROUND instead of
            # once per agent. Combined with sustained actions (Dig calls
            # env.step() 5× per Python step in custom_environment_craftium.py),
            # the env truncates AT episodes well before `max_steps` is reached
            # (~750 Python steps for max_steps=1500 in dig-heavy rollouts,
            # ~26 for max_steps=50). Python's `for step in range(max_steps)`
            # in multi_agent_craftium.py is the authoritative episode-length
            # cap; with max_timesteps=None the env's truncated flag stays
            # False for the whole episode and only termination (an agent
            # dying) ends the episode early.
            max_timesteps=None,
            minetest_dir=minetest_dir,
            mt_listen_timeout=300_000,  # 5 min per client; VoxeLibre loads slowly on HPC
            seed=seed,
            frameskip=frameskip,
            pmul=pmul,
            # T1.4 — Luanti server tuning for our small world.
            # Defaults in craftium/craftium/minetest.py target an open
            # Minecraft-scale world with a 100-block view distance and a
            # 200 fps server tick. Five-chambers spans ~70 blocks in z
            # total (Ch1 z=0..15 through Ch5 z=59..67), all pre-generated
            # by world_gen.lua at server start, so almost all of the
            # default server CPU goes to renderable-chunk bookkeeping the
            # agents never see.
            #
            # fps_max: 200 → 30 — Lua tick rate is 20 Hz; server tick +
            #   screenshot capture happen once per env step, so 30 fps
            #   gives ~50% safety margin without burning idle CPU on a
            #   software renderer. Passed at the top level so it applies
            #   to BOTH server and client (the constructor threads it
            #   into both, see craftium/craftium/multiagent_env.py:103,
            #   126).
            fps_max=30,
            mt_server_conf={
                # max_block_send_distance: 100 → 8 — 8 mapblocks ≈ 128
                # blocks, more than enough for the agent's camera
                # frustum at obs_width=320.
                "max_block_send_distance": 8,
                "max_block_generate_distance": 8,
            },
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
