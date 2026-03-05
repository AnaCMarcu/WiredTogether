"""Test basic multi-agent OpenWorld environment."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs import OpenWorldMultiAgentEnv

NUM_AGENTS = 2

def main():
    """Test basic multi-agent OpenWorld environment."""
    print("=" * 60)
    print("Testing Multi-Agent OpenWorld Environment")
    print("=" * 60)

    env = None
    step_count = 0

    try:
        # Create environment with 4 agents
        print(f"\nCreating environment with {NUM_AGENTS} agents...")
        env = OpenWorldMultiAgentEnv(
            num_agents=NUM_AGENTS,
            obs_width=320,
            obs_height=180,
            max_steps=1000,
            render_mode="rgb_array",
        )

        print(f"\n✓ Environment created successfully!")
        print(f"  Agents: {env.possible_agents}")
        print(f"  Action space: {env.action_space('agent_0')}")
        print(f"  Observation space: {env.observation_space('agent_0')}")

        # Reset
        print("\nResetting environment...")
        observations, infos = env.reset(seed=42)

        print(f"\n✓ Environment reset successfully!")
        print(f"  Active agents: {env.agents}")
        print(f"  Observation shapes:")
        for agent, obs in observations.items():
            print(f"    {agent}: {obs.shape}")

        # Verify all observations are correct
        assert len(observations) == len(env.agents), f"Expected {len(env.agents)} observations, got {len(observations)}"
        for agent in env.agents:
            assert agent in observations, f"Missing observation for {agent}"
            print(f"    {agent}: shape={observations[agent].shape}")

        print("\n✓ All agents have correct observations!")

        # Run 100 steps
        print(f"\n{'='*60}")
        print("Running 100 steps with random actions...")
        print(f"{'='*60}\n")

        episode_rewards = {agent: 0.0 for agent in env.possible_agents}

        for step in range(100):
            # Random actions for all agents
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Accumulate rewards
            for agent in env.agents:
                episode_rewards[agent] += rewards[agent]

            step_count = step + 1

            # Print progress every 20 steps
            if step_count % 20 == 0:
                print(f"Step {step_count}/100:")
                for agent in env.agents:
                    print(
                        f"  {agent}: reward={rewards[agent]:.3f}, total={episode_rewards[agent]:.3f}"
                    )
                print()

            # Check if all agents are done
            if not env.agents:
                print(f"All agents terminated at step {step_count}")
                break

        # Final summary
        print(f"\n{'='*60}")
        print("Final Summary")
        print(f"{'='*60}")
        print(f"\nCompleted {step_count} steps")
        print("\nTotal Episode Rewards:")
        for agent, total in episode_rewards.items():
            print(f"  {agent}: {total:.3f}")

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        print(f"Completed {step_count} steps before interruption")

    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always close environment
        if env is not None:
            print("\nClosing environment...")
            env.close()
            print("✓ Environment closed successfully!")
            print("\nNote: Luanti-run directories remain in the project root.")


if __name__ == "__main__":
    main()
