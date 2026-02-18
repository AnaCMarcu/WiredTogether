"""PettingZoo ParallelEnv wrapper for Craftium's OpenWorld environment."""

import os
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from craftium.multiagent_env import MarlCraftiumEnv

_DISCRETE_ACTIONS = [
    "forward", "backward", "left", "right", "jump", "sneak",
    "dig", "place", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
    "mouse x+", "mouse x-", "mouse y+",
]
_MOUSE_MOV = 0.5


def _discrete_to_dict(action: int) -> dict:
    """Convert a Discrete(17) integer to MarlCraftiumEnv dict format.

    Action 0 is NOP. Actions 1-16 map to _DISCRETE_ACTIONS.
    PettingZoo action_space.sample() return an integer, but Craftium expects {action, mouse} format
    """
    action = int(action)
    if action == 0:
        return {}  # NOP: no mouse movement

    name = _DISCRETE_ACTIONS[action - 1]
    mouse = [0.0, 0.0]

    if name == "mouse x+":
        mouse[0] = _MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse x-":
        mouse[0] = -_MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse y+":
        mouse[1] = _MOUSE_MOV
        return {"mouse": mouse}
    elif name == "mouse y-":
        mouse[1] = -_MOUSE_MOV
        return {"mouse": mouse}
    else:
        return {name: 1, "mouse": mouse}


class OpenWorldMultiAgentEnv(ParallelEnv):
    """Multi-agent OpenWorld environment with PettingZoo ParallelEnv API.

    This wrapper adapts Craftium's MarlCraftiumEnv to PettingZoo's ParallelEnv interface,
    allowing use with multi-agent RL libraries like RLlib, CleanRL, etc.

    Args:
        num_agents: Number of agents in the environment
        obs_width: Observation width in pixels (default: 320)
        obs_height: Observation height in pixels (default: 180)
        max_steps: Maximum steps per episode (default: 10000)
        task_focus: Optional task to focus agents on (e.g., "mining", "crafting")
        render_mode: Rendering mode ("rgb_array" or None)
    """

    metadata = {"render_modes": ["rgb_array"], "name": "openworld_multi_agent_v0"}

    def __init__(
        self,
        num_agents: int = 2,
        obs_width: int = 320,
        obs_height: int = 180,
        max_steps: int = 10000,
        task_focus: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store config (use _num_agents to avoid conflict with PettingZoo property)
        self._num_agents = num_agents
        self.obs_width = obs_width
        self.obs_height = obs_height
        self.max_steps = max_steps
        self.task_focus = task_focus
        self.render_mode = render_mode

        # Define agent names BEFORE creating env (needed for num_agents property)
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

        from craftium import root_path as craftium_root
        minetest_dir = os.environ.get(
            "CRAFTIUM_LUANTI_DIR",
            os.path.join(craftium_root, "luanti")
        )
        env_dir = os.environ.get(
            "CRAFTIUM_ENV_DIR",
            os.path.join(craftium_root, "craftium-envs", "voxel-libre2")
        )

        self.env = MarlCraftiumEnv(
            env_dir=env_dir,
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            max_timesteps=max_steps,
            minetest_dir=minetest_dir,
        )

        # Define observation and action spaces
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )
        # Craftium uses 17 discrete actions (DiscreteActionWrapper)
        self._action_space = spaces.Discrete(17)

        self._step_count = 0

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Return observation spaces for all agents."""
        return {agent: self._observation_space for agent in self.possible_agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Return action spaces for all agents."""
        return {agent: self._action_space for agent in self.possible_agents}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment.

        Returns:
            observations: Dictionary of observations per agent
            infos: Dictionary of info dicts per agent
        """
        if seed is not None:
            np.random.seed(seed)

        obs_array, _ = self.env.reset()

        # Reset agents and step counter
        self.agents = self.possible_agents.copy()
        self._step_count = 0

        # Convert to PettingZoo format.
        # obs_array is a numpy array of shape (num_agents, obs_width, obs_height, 3).
        # info_dict is a plain dict (not per-agent), so we give each agent an empty dict.
        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_array)}
        infos = {f"agent_{i}": {} for i in range(self._num_agents)}

        return observations, infos

    def step(
        self, actions: Dict[str, int]
    ) -> tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment with actions from all agents.

        Args:
            actions: Dictionary mapping agent names to actions

        Returns:
            observations: Dictionary of observations per agent
            rewards: Dictionary of rewards per agent
            terminations: Dictionary of termination flags per agent
            truncations: Dictionary of truncation flags per agent
            infos: Dictionary of info dicts per agent
        """
        self._step_count += 1

        # Convert Discrete(17) integers to dict format expected by MarlCraftiumEnv
        action_list = [_discrete_to_dict(actions[agent]) for agent in self.agents]

        # Step the underlying environment
        obs_dict, reward_dict, done_dict, truncated_dict, _ = self.env.step(
            action_list
        )

        # Convert to PettingZoo format.
        # obs_dict, reward_dict, done_dict, truncated_dict are all numpy arrays.
        # info_dict is a plain merged dict, not per-agent.
        observations = {f"agent_{i}": obs for i, obs in enumerate(obs_dict)}
        rewards = {f"agent_{i}": float(rew) for i, rew in enumerate(reward_dict)}
        terminations = {f"agent_{i}": bool(done) for i, done in enumerate(done_dict)}
        truncations = {f"agent_{i}": bool(trunc) for i, trunc in enumerate(truncated_dict)}
        infos = {f"agent_{i}": {} for i in range(self._num_agents)}

        # Apply task-focused reward shaping if specified
        if self.task_focus:
            rewards = self._apply_task_focus(rewards, infos)

        # Remove agents that are done
        self.agents = [
            agent for agent in self.agents if not terminations.get(agent, False)
        ]

        return observations, rewards, terminations, truncations, infos

    def _apply_task_focus(
        self, rewards: Dict[str, float], infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply task-focused reward shaping.

        This encourages agents to focus on specific tasks by modifying rewards
        based on the task_focus parameter.

        Args:
            rewards: Original rewards
            infos: Info dictionaries containing task progress

        Returns:
            Modified rewards dictionary
        """
        # Task-specific reward bonuses
        task_bonuses = {
            "mining": {"mined_blocks": 1.0},
            "crafting": {"items_crafted": 2.0},
            "building": {"blocks_placed": 1.0},
            "exploration": {"distance_traveled": 0.1},
        }

        if self.task_focus not in task_bonuses:
            return rewards

        bonuses = task_bonuses[self.task_focus]
        shaped_rewards = rewards.copy()

        for agent, info in infos.items():
            bonus = 0.0
            for metric, weight in bonuses.items():
                if metric in info:
                    bonus += info[metric] * weight
            shaped_rewards[agent] = rewards[agent] + bonus

        return shaped_rewards

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            return self.env.render()
        return None

    def close(self):
        """Close the environment and cleanup resources."""
        self.env.close()

    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for a specific agent."""
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for a specific agent."""
        return self._action_space
