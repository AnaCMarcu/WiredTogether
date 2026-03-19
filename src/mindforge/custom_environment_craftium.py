"""Environment adapter bridging Craftium OpenWorld multi-agent env to CausalForge's interface."""

import sys
import os
import numpy as np
import PIL.Image

# Import the PettingZoo env from src/craftium/
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)  # src/
_craftium_dir = os.path.join(_src_dir, "craftium")
if _craftium_dir not in sys.path:
    sys.path.insert(0, _craftium_dir)

from openworld_multi_agents import OpenWorldMultiAgentEnv

# Load environment prompt
with open(os.path.join(_this_dir, "prompts", "environment_prompt.txt"), "r") as f:
    environment_prompt = f.read()

# Map from human-readable action names (used by the LLM) to Discrete(17) integers.
# The integer mapping matches _DISCRETE_ACTIONS in openworld_multi_agents.py:
#   0 = NOP
#   1-16 = forward, backward, left, right, jump, sneak, dig, place,
#           slot_1..slot_5, mouse_x+, mouse_x-, mouse_y+
ACTION_MAP = {
    "NoOp": 0,
    "MoveForward": 1,
    "MoveBackward": 2,
    "MoveLeft": 3,
    "MoveRight": 4,
    "Jump": 5,
    "Sneak": 6,
    "Dig": 7,
    "Place": 8,
    "Slot1": 9,
    "Slot2": 10,
    "Slot3": 11,
    "Slot4": 12,
    "Slot5": 13,
    "TurnRight": 14,
    "TurnLeft": 15,
    "LookDown": 16,
    "LookUp": 17,
}

VALID_ACTIONS = list(ACTION_MAP.keys())


class CraftiumEnvironmentInterface:
    """Wraps OpenWorldMultiAgentEnv to match CausalForge's Environment_Interface contract.

    CausalForge expects:
      - step(action_str, agentId) -> (event, action_str)
      - environment_prompt (str)
      - pickedup_object(agentId) -> str | None
    """

    def __init__(self, num_agents=3, obs_width=320, obs_height=180, max_steps=10000, seed=None):
        self.num_agents = num_agents
        self.seed = seed
        self.env = OpenWorldMultiAgentEnv(
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            max_steps=max_steps,
            seed=seed,
        )
        self.environment_prompt = environment_prompt

        # Per-agent state
        self._observations = {}   # agent_name -> np.ndarray (H, W, 3)
        self._rewards = {}        # agent_name -> float (cumulative since last query)
        self._step_rewards = {}   # agent_name -> float (raw reward from last step)
        self._terminations = {}   # agent_name -> bool
        self._truncations = {}    # agent_name -> bool
        self._infos = {}          # agent_name -> dict
        self._last_actions = {}   # agent_name -> str

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the environment and return initial observations."""
        observations, infos = self.env.reset()
        self._observations = observations
        self._infos = infos
        self._rewards = {a: 0.0 for a in self.env.possible_agents}
        self._terminations = {a: False for a in self.env.possible_agents}
        self._truncations = {a: False for a in self.env.possible_agents}
        self._last_actions = {a: "NoOp" for a in self.env.possible_agents}
        return self._observations

    def step(self, action_str: str, agentId: int):
        """Execute one action for one agent, NoOp for the rest, then step the env.

        Args:
            action_str: Action name (e.g. "Dig", "MoveForward")
            agentId: Integer index of the acting agent

        Returns:
            (observations_dict, action_str) — matching CausalForge's contract

        Raises:
            ValueError: If action_str is not in ACTION_MAP
        """
        if action_str not in ACTION_MAP:
            raise ValueError(
                f"Invalid action: '{action_str}'. Valid actions: {VALID_ACTIONS}"
            )

        # Build action dict: acting agent gets the requested action, others NoOp
        actions = {}
        for i in range(self.num_agents):
            agent_name = f"agent_{i}"
            if agent_name not in self.env.agents:
                continue  # skip terminated agents
            if i == agentId:
                actions[agent_name] = ACTION_MAP[action_str]
            else:
                actions[agent_name] = ACTION_MAP["NoOp"]

        # Step the PettingZoo env
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        self._observations = observations
        self._terminations = terminations
        self._truncations = truncations
        self._infos = infos

        # Store raw per-step rewards and accumulate for summary
        self._step_rewards = dict(rewards)
        for agent_name, rew in rewards.items():
            self._rewards[agent_name] = self._rewards.get(agent_name, 0.0) + rew

        agent_name = f"agent_{agentId}"
        self._last_actions[agent_name] = action_str

        return self._observations, action_str

    # ------------------------------------------------------------------
    # Observation helpers (match CausalForge's usage patterns)
    # ------------------------------------------------------------------
    def get_agent_frame(self, agentId: int) -> np.ndarray:
        """Return the raw observation array (H, W, 3) for a specific agent."""
        agent_name = f"agent_{agentId}"
        return self._observations.get(agent_name)

    def get_pil_image(self, agentId: int) -> PIL.Image.Image:
        """Return a PIL Image for the given agent's current view."""
        frame = self.get_agent_frame(agentId)
        if frame is None:
            # Return a blank image if no observation available
            return PIL.Image.new("RGB", (320, 180), (0, 0, 0))
        return PIL.Image.fromarray(frame)

    def get_reward_summary(self, agentId: int) -> str:
        """Return a text summary of the agent's reward and status.

        This is injected into the instruction prompt so the LLM sees its reward.
        Reading the summary resets the accumulated reward for that agent.
        """
        agent_name = f"agent_{agentId}"
        reward = self._rewards.get(agent_name, 0.0)
        terminated = self._terminations.get(agent_name, False)
        truncated = self._truncations.get(agent_name, False)

        if terminated:
            status = "Dead"
        elif truncated:
            status = "Episode ended"
        else:
            status = "Alive"

        # Reset accumulated reward after reading
        self._rewards[agent_name] = 0.0

        return f"Reward: {reward:.2f} / Status: {status}"

    def any_done(self) -> bool:
        """Check if any agent is terminated or truncated."""
        for agent_name in self.env.possible_agents:
            if self._terminations.get(agent_name, False):
                return True
            if self._truncations.get(agent_name, False):
                return True
        return False

    def all_done(self) -> bool:
        """Check if all agents are terminated or truncated."""
        for agent_name in self.env.possible_agents:
            if not self._terminations.get(agent_name, False) and not self._truncations.get(agent_name, False):
                return False
        return True

    def get_step_reward(self, agentId: int) -> float:
        """Return the raw reward from the last env.step() for this agent."""
        agent_name = f"agent_{agentId}"
        return self._step_rewards.get(agent_name, 0.0)

    def pickedup_object(self, agentId: int = 0):
        """Return held item name (Craftium doesn't expose this, so always None)."""
        return None

    def warmup_noop(self):
        """Send NoOps to keep channels alive without incrementing step counters.

        Use this during media-loading warm-up instead of step().
        Returns list of observations (one per agent).
        """
        return self.env.warmup_noop()

    def close(self):
        """Clean up the underlying environment."""
        self.env.close()
