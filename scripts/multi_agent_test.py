"""Test script for multi-agent OpenWorld environment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.envs import OpenWorldMultiAgentEnv


def main():
    """Run a simple multi-agent test with task-focused agents."""
    # Create environment with 4 agents focused on mining
    env = OpenWorldMultiAgentEnv(
        num_agents=4,
        obs_width=320,
        obs_height=180,
        max_steps=1000,
        task_focus="mining",  # Focus agents on mining tasks
        render_mode="rgb_array",
    )

    print(f"Created multi-agent environment with {env.num_agents} agents")
    print(f"Agents: {env.possible_agents}")
    print(f"Task focus: {env.task_focus}")
    print(f"Observation space: {env.observation_spaces}")
    print(f"Action space: {env.action_spaces}")
    print()

    # Reset environment
    observations, infos = env.reset(seed=42)
    print(f"Environment reset. Active agents: {env.agents}")
    print()

    # Run for 50 steps
    episode_rewards = {agent: 0.0 for agent in env.possible_agents}

    for step in range(50):
        # Sample random actions for all active agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            episode_rewards[agent] += rewards[agent]

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}:")
            print(f"  Active agents: {len(env.agents)}")
            for agent in env.agents:
                print(
                    f"  {agent}: reward={rewards[agent]:.3f}, "
                    f"total={episode_rewards[agent]:.3f}"
                )
            print()

        # Check if all agents are done
        if not env.agents:
            print("All agents terminated!")
            break

        # Reset if any agent is done (for demonstration)
        if any(terminations.values()) or any(truncations.values()):
            print("Episode ended, resetting...")
            observations, infos = env.reset()
            episode_rewards = {agent: 0.0 for agent in env.possible_agents}
            print()

    # Final results
    print("\nFinal episode rewards:")
    for agent, total_reward in episode_rewards.items():
        print(f"  {agent}: {total_reward:.3f}")

    env.close()
    print("\nEnvironment closed successfully!")


if __name__ == "__main__":
    main()
