"""Test script for MVP role-based environment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.envs import RoleBasedOpenWorld


def main():
    """Test MVP role-based environment."""
    print("=" * 60)
    print("Testing MVP Role-Based OpenWorld Environment")
    print("=" * 60)

    # Create environment with 3 agents (engineer, hunter, guardian)
    env = RoleBasedOpenWorld(
        num_agents=3,
        roles=["engineer", "hunter", "guardian"],
        obs_width=320,
        obs_height=180,
        max_steps=1000,
        render_mode="rgb_array",
    )

    print(f"\n✓ Environment created")
    print(f"  Agents: {env.possible_agents}")
    print(f"  Roles: {env.role_names}")

    # Reset environment
    observations, infos = env.reset(seed=42)

    print(f"\n✓ Environment reset")
    print(f"  Active agents: {env.agents}")
    print(f"  Role assignments:")
    for agent, role in env.agent_roles.items():
        print(f"    {agent}: {role}")

    # Run for 100 steps
    print(f"\n{'='*60}")
    print("Running 100 steps with random actions...")
    print(f"{'='*60}\n")

    episode_rewards = {agent: 0.0 for agent in env.possible_agents}
    step_count = 0

    for step in range(100):
        # Random actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Accumulate rewards
        for agent in env.agents:
            episode_rewards[agent] += rewards[agent]

        step_count += 1

        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"\n--- Step {step + 1} ---")
            print(f"Active agents: {len(env.agents)}")

            # Show individual role stats
            for agent in env.agents:
                role = env.agent_roles[agent]
                stats = infos[agent].get("role_stats", {})
                print(f"  {agent} ({role}):")
                print(f"    Reward: {rewards[agent]:.3f} (Total: {episode_rewards[agent]:.3f})")
                print(f"    Stats: {stats}")

            # Show team stats
            team_stats = env.get_team_stats()
            print(f"\n  Team Stats:")
            print(f"    Tools crafted: {team_stats['stats']['tools_crafted']}")
            print(f"    Ore mined: {team_stats['stats']['ore_mined']}")
            print(f"    Monsters killed: {team_stats['stats']['monsters_killed']}")
            print(f"    Animals hunted: {team_stats['stats']['animals_hunted']}")
            print(f"\n  Milestones unlocked:")
            for milestone, unlocked in team_stats['milestones'].items():
                if unlocked:
                    print(f"    ✓ {milestone}")

        # Check if all agents are done
        if not env.agents:
            print("\nAll agents terminated!")
            break

        # Reset if episode ends
        if any(terminations.values()) or any(truncations.values()):
            print("\nEpisode ended, resetting...")
            observations, infos = env.reset()
            episode_rewards = {agent: 0.0 for agent in env.possible_agents}
            step_count = 0

    # Final summary
    print(f"\n{'='*60}")
    print("Final Episode Summary")
    print(f"{'='*60}")

    team_stats = env.get_team_stats()

    print(f"\nTeam Statistics:")
    print(f"  Total steps: {step_count}")
    print(f"  Tools crafted: {team_stats['stats']['tools_crafted']}")
    print(f"  Ore mined: {team_stats['stats']['ore_mined']}")
    print(f"  Monsters killed: {team_stats['stats']['monsters_killed']}")
    print(f"  Animals hunted: {team_stats['stats']['animals_hunted']}")

    print(f"\nMilestones Achieved:")
    unlocked_count = sum(team_stats['milestones'].values())
    total_count = len(team_stats['milestones'])
    print(f"  {unlocked_count}/{total_count} milestones")
    for milestone, unlocked in team_stats['milestones'].items():
        status = "✓" if unlocked else "✗"
        print(f"  {status} {milestone}")

    print(f"\nFinal Episode Rewards:")
    for agent, total_reward in episode_rewards.items():
        role = env.agent_roles[agent]
        print(f"  {agent} ({role}): {total_reward:.3f}")

    env.close()
    print(f"\n✓ Environment closed successfully!")


if __name__ == "__main__":
    main()
