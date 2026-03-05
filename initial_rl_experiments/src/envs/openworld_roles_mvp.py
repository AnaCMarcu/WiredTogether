"""Role-based OpenWorld multi-agent."""

from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from .openworld_multi_agents import OpenWorldMultiAgentEnv


class RoleBasedOpenWorld(OpenWorldMultiAgentEnv):
    """Open-world multi-agent role-based environment.

    Changes from base OpenWorldParallelEnv:
    1. Assigns fixed roles at reset (engineer, hunter, guardian)
    2. Applies role-based reward multipliers:
       - Actions in agent's track: FULL reward (1.0x)
       - Actions outside agent's track: HALF reward (0.5x)
    3. Uses OpenWorld's base milestone rewards for tools, hunt, defend

    Agents CAN do any action, but get reduced rewards for out-of-track actions.
    """

    def __init__(
        self,
        num_agents: int = 3,
        roles: Optional[List[str]] = None,
        obs_width: int = 320,
        obs_height: int = 180,
        max_steps: int = 10000,
        render_mode: Optional[str] = None,
    ):
        """Initialize role-based environment.

        Args:
            num_agents: Number of agents
            roles: List of role names. Can be:
                - None: defaults to cycling ["engineer", "hunter", "guardian"]
                - Fewer than num_agents: cycles through provided roles
                - Equal to num_agents: assigns roles directly
                - More than num_agents: truncates to num_agents
                Examples:
                - 4 agents, roles=["engineer", "hunter"]: ["engineer", "hunter", "engineer", "hunter"]
                - 6 agents, roles=None: ["engineer", "hunter", "guardian", "engineer", "hunter", "guardian"]
            obs_width: Observation width
            obs_height: Observation height
            max_steps: Max timesteps per episode
            render_mode: Render mode
        """
        super().__init__(
            num_agents=num_agents,
            obs_width=obs_width,
            obs_height=obs_height,
            max_steps=max_steps,
            task_focus=None,  # rewards are handled in this class
            render_mode=render_mode,
        )

        # Handle role assignment
        if roles is None:
            # Default: cycle through engineer, hunter, guardian
            base_roles = ["engineer", "hunter", "guardian"]
            roles = [base_roles[i % len(base_roles)] for i in range(num_agents)]
        elif len(roles) < num_agents:
            # If fewer roles than agents, cycle through provided roles
            roles = [roles[i % len(roles)] for i in range(num_agents)]
        elif len(roles) > num_agents:
            # If more roles than agents, truncate
            roles = roles[:num_agents]
        # else: len(roles) == num_agents, use as-is

        self.role_names = roles
        self.agent_roles = {}  # Will be set in reset()

        # Map roles to their tracks
        self.role_tracks = {
            "engineer": "tools",   # Crafting, building, mining
            "hunter": "hunt",      # Hunting animals
            "guardian": "defend",  # Killing monsters, defense
        }

        # Simple multipliers: full reward in-track, half reward out-of-track
        self.in_track_multiplier = 1.0   # Full reward for actions in your track
        self.out_track_multiplier = 0.5  # Half reward for actions outside your track

        # Track team milestones (shared across all agents)
        self.team_milestones = {
            "has_wooden_pickaxe": False,
            "has_stone_pickaxe": False,
            "has_iron_pickaxe": False,
            "has_diamond_pickaxe": False,
            "killed_5_monsters": False,
            "killed_10_monsters": False,
            "hunted_5_animals": False,
            "hunted_10_animals": False,
        }

        self.team_stats = {
            "tools_crafted": 0,
            "ore_mined": 0,
            "monsters_killed": 0,
            "animals_hunted": 0,
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset environment and assign roles."""
        observations, infos = super().reset(seed=seed, options=options)

        # Assign roles
        self.agent_roles = {
            f"agent_{i}": role for i, role in enumerate(self.role_names)
        }

        # Reset team tracking
        self.team_milestones = {k: False for k in self.team_milestones}
        self.team_stats = {k: 0 for k in self.team_stats}

        # Add role info to agent infos
        for agent, info in infos.items():
            info["role"] = self.agent_roles[agent]
            info["role_stats"] = {
                "tools_crafted": 0,
                "ore_mined": 0,
                "monsters_killed": 0,
                "animals_hunted": 0,
            }

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
        """Step with role-based reward shaping."""
        # Get base step from parent
        observations, base_rewards, terminations, truncations, infos = super().step(actions)

        # Apply role-based reward shaping
        shaped_rewards = self._apply_role_rewards(base_rewards, infos)

        # Compute team reward (shared)
        team_reward = self._compute_team_reward(shaped_rewards, infos)

        # All agents get same team reward
        final_rewards = {agent: team_reward for agent in self.agents}

        # Add role info to infos
        for agent in self.agents:
            infos[agent]["role"] = self.agent_roles[agent]

        return observations, final_rewards, terminations, truncations, infos

    def _apply_role_rewards(
        self, base_rewards: Dict[str, float], infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply role-specific reward multipliers.

        Uses OpenWorld's base rewards and applies track-based multipliers:
        - Full reward (1.0x) for actions in agent's track
        - Half reward (0.5x) for actions outside agent's track
        """
        shaped_rewards = {}

        for agent, base_reward in base_rewards.items():
            role = self.agent_roles[agent]
            agent_track = self.role_tracks[role]
            info = infos[agent]

            # Detect which track this reward belongs to
            reward_track = self._detect_reward_track(info)

            # Apply multiplier based on whether reward is in agent's track
            if reward_track == agent_track:
                multiplier = self.in_track_multiplier  # 1.0x - full reward
            elif reward_track is not None:
                multiplier = self.out_track_multiplier  # 0.5x - half reward
            else:
                multiplier = 1.0  # No track detected, keep base reward

            shaped_rewards[agent] = base_reward * multiplier

        return shaped_rewards

    def _detect_reward_track(self, info: Dict[str, Any]) -> Optional[str]:
        """Detect which track a reward belongs to based on info dict.

        Returns:
            "tools" for crafting/mining actions
            "hunt" for hunting animals
            "defend" for killing monsters
            None if no specific track detected
        """
        # Tools track: crafting, mining, building
        if info.get("crafted_item") or info.get("mined_ore") or info.get("placed_block"):
            if info.get("crafted_item"):
                self.team_stats["tools_crafted"] += 1
                info["role_stats"]["tools_crafted"] += 1
            elif info.get("mined_ore"):
                self.team_stats["ore_mined"] += 1
                info["role_stats"]["ore_mined"] += 1
            return "tools"

        # Hunt track: killing animals
        if info.get("hunted_animal") or info.get("killed_animal"):
            self.team_stats["animals_hunted"] += 1
            info["role_stats"]["animals_hunted"] += 1
            return "hunt"

        # Defend track: killing monsters
        if info.get("killed_monster") or info.get("killed_mob"):
            self.team_stats["monsters_killed"] += 1
            info["role_stats"]["monsters_killed"] += 1
            return "defend"

        return None

    def _compute_team_reward(
        self, shaped_rewards: Dict[str, float], infos: Dict[str, Dict[str, Any]]
    ) -> float:
        """Compute shared team reward.

        Combines individual shaped rewards + milestone bonuses.
        """
        # Sum individual shaped rewards
        team_reward = sum(shaped_rewards.values())

        # Add milestone bonuses (sparse, high-value)
        milestone_bonus = self._check_milestones()
        team_reward += milestone_bonus

        # Add cooperation bonus (simple heuristic)
        cooperation_bonus = self._detect_cooperation(infos)
        team_reward += cooperation_bonus

        return team_reward

    def _check_milestones(self) -> float:
        """Check and reward team milestones."""
        bonus = 0.0

        # Check tool progression (simplified - would need actual Craftium state)
        if self.team_stats["tools_crafted"] >= 1 and not self.team_milestones["has_wooden_pickaxe"]:
            self.team_milestones["has_wooden_pickaxe"] = True
            bonus += 10.0

        if self.team_stats["tools_crafted"] >= 3 and not self.team_milestones["has_stone_pickaxe"]:
            self.team_milestones["has_stone_pickaxe"] = True
            bonus += 20.0

        if self.team_stats["tools_crafted"] >= 5 and not self.team_milestones["has_iron_pickaxe"]:
            self.team_milestones["has_iron_pickaxe"] = True
            bonus += 50.0

        # Check combat milestones
        if self.team_stats["monsters_killed"] >= 5 and not self.team_milestones["killed_5_monsters"]:
            self.team_milestones["killed_5_monsters"] = True
            bonus += 15.0

        if self.team_stats["monsters_killed"] >= 10 and not self.team_milestones["killed_10_monsters"]:
            self.team_milestones["killed_10_monsters"] = True
            bonus += 30.0

        # Check hunting milestones
        if self.team_stats["animals_hunted"] >= 5 and not self.team_milestones["hunted_5_animals"]:
            self.team_milestones["hunted_5_animals"] = True
            bonus += 10.0

        if self.team_stats["animals_hunted"] >= 10 and not self.team_milestones["hunted_10_animals"]:
            self.team_milestones["hunted_10_animals"] = True
            bonus += 20.0

        return bonus

    def _detect_cooperation(self, infos: Dict[str, Dict[str, Any]]) -> float:
        """Detect simple cooperation patterns.

        MVP version: Just check if agents are near each other.
        """
        bonus = 0.0

        # Get agent positions
        positions = {}
        for agent, info in infos.items():
            if "position" in info:
                positions[agent] = np.array(info["position"])

        # If we have at least 2 agents with positions
        if len(positions) >= 2:
            # Check pairwise distances
            agents_list = list(positions.keys())
            for i in range(len(agents_list)):
                for j in range(i + 1, len(agents_list)):
                    dist = np.linalg.norm(positions[agents_list[i]] - positions[agents_list[j]])

                    # Small bonus for being close (encourages staying together)
                    if dist < 10.0:
                        bonus += 0.01

        return bonus

    def get_team_stats(self) -> Dict[str, Any]:
        """Get current team statistics."""
        return {
            "milestones": self.team_milestones.copy(),
            "stats": self.team_stats.copy(),
            "roles": self.agent_roles.copy(),
        }
