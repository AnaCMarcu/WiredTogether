"""Scripted heuristic agent to verify the Craftium environment is functional.

This script bypasses ALL LLM machinery and sends hardcoded actions directly
to the environment. The goal is to confirm:
  1. The environment starts up correctly
  2. Actions actually move the agent
  3. Dig produces rewards (small per-block rewards)
  4. Ideally: the wood stage reward (128.0) is reachable

Run from src/mindforge/:
    python test_scripted_agent.py

Or with more episodes:
    python test_scripted_agent.py --episodes 3 --max-steps 300
"""

import argparse
import os
import sys
import time
import numpy as np

# ── Path setup ──
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
_craftium_dir = os.path.join(_src_dir, "craftium")
for p in [_this_dir, _craftium_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from custom_environment_craftium import CraftiumEnvironmentInterface, ACTION_MAP


# ──────────────────────────────────────────────
# Heuristic policy
# ──────────────────────────────────────────────

class ScriptedPolicy:
    """Dead-simple rule-based policy.

    Strategy:
      - Rotate right for a few steps to scan for trees (center pixel heuristic)
      - Move forward to close distance
      - Dig repeatedly when facing something (any block gives small reward)
      - Occasionally jump to get unstuck
      - Repeat

    No vision model. No LLM. This is purely to test that the env works.
    """

    def __init__(self):
        self._step = 0
        self._dig_streak = 0
        self._no_reward_streak = 0

    def act(self, obs: np.ndarray, last_reward: float) -> str:
        """Return an action name string given the current observation frame."""
        t = self._step
        self._step += 1

        # Track reward streaks
        if last_reward > 0.01:
            self._no_reward_streak = 0
            self._dig_streak += 1
        else:
            self._no_reward_streak += 1
            self._dig_streak = 0

        # If we've been digging and getting rewards, keep digging
        if self._dig_streak > 0 and self._dig_streak < 30:
            return "Dig"

        # If stuck (no reward for a long time), try to get unstuck
        if self._no_reward_streak > 20:
            self._no_reward_streak = 0
            # Cycle through unstuck moves
            unstuck = t % 40
            if unstuck < 5:
                return "Jump"
            elif unstuck < 10:
                return "MoveForward"
            elif unstuck < 15:
                return "TurnRight"
            elif unstuck < 20:
                return "TurnRight"
            elif unstuck < 25:
                return "MoveForward"
            else:
                return "Dig"

        # Default scanning pattern: turn + move + dig
        phase = t % 60
        if phase < 10:
            return "TurnRight"      # scan right
        elif phase < 15:
            return "MoveForward"    # close distance
        elif phase < 25:
            return "Dig"            # try to hit block
        elif phase < 30:
            return "TurnLeft"       # scan left
        elif phase < 35:
            return "MoveForward"
        elif phase < 45:
            return "Dig"
        elif phase < 50:
            return "TurnRight"
        else:
            return "MoveForward"


# ──────────────────────────────────────────────
# Main test loop
# ──────────────────────────────────────────────

def run_test(num_agents: int, num_episodes: int, max_steps: int,
             warmup_time: int, verbose: bool):

    print(f"\n{'='*60}")
    print(f"  Scripted Agent Environment Test")
    print(f"  agents={num_agents}, episodes={num_episodes}, steps={max_steps}")
    print(f"{'='*60}\n")

    env = CraftiumEnvironmentInterface(
        num_agents=num_agents,
        obs_width=320,
        obs_height=180,
        max_steps=max_steps,
    )

    all_episode_rewards = []
    milestone_hits = []  # (episode, step, agent, reward_value)

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")

        env.reset()

        # ── Warm-up: wait for media to load ──
        print(f"  Waiting for media ({warmup_time}s min)...")
        warmup_start = time.time()
        max_warmup = 900
        all_loaded = False
        last_log = 0.0
        while time.time() - warmup_start < max_warmup:
            observations = env.warmup_noop()
            elapsed = time.time() - warmup_start

            stds = []
            if observations:
                for obs in observations:
                    stds.append(float(np.std(obs.astype(np.float32))) if obs is not None else 0.0)

            if elapsed - last_log >= 15.0 and stds:
                std_str = ", ".join(f"a{i}={s:.1f}" for i, s in enumerate(stds))
                print(f"    [{elapsed:.0f}s] std: {std_str}  (>30 = loaded)")
                last_log = elapsed

            if elapsed >= warmup_time and stds and all(s > 30.0 for s in stds):
                all_loaded = True
                break
            time.sleep(2)

        elapsed = time.time() - warmup_start
        status = "loaded" if all_loaded else "timeout"
        print(f"  Warm-up done ({elapsed:.0f}s, {status})")

        # ── Per-agent policies and state ──
        policies = [ScriptedPolicy() for _ in range(num_agents)]
        episode_rewards = [0.0] * num_agents
        last_rewards = [0.0] * num_agents

        for step in range(max_steps):
            if env.all_done():
                print(f"  All agents done at step {step}")
                break

            for agent_id in range(num_agents):
                agent_name = f"agent_{agent_id}"
                if env._terminations.get(agent_name, False):
                    continue

                obs = env.get_agent_frame(agent_id)
                if obs is None:
                    obs = np.zeros((180, 320, 3), dtype=np.uint8)

                action = policies[agent_id].act(obs, last_rewards[agent_id])
                env.step(action, agentId=agent_id)

                raw_reward = env.get_step_reward(agent_id)
                last_rewards[agent_id] = raw_reward
                episode_rewards[agent_id] += raw_reward

                # ── Log anything interesting ──
                if raw_reward > 0.05:
                    if verbose or raw_reward >= 10.0:
                        print(f"  [step {step:4d}] agent_{agent_id} action={action:<15s}"
                              f"  reward={raw_reward:8.2f}"
                              f"  cumulative={episode_rewards[agent_id]:8.2f}")

                # Milestone detection: large reward spikes
                if raw_reward >= 100.0:
                    milestone_hits.append((ep+1, step, agent_id, raw_reward))
                    print(f"\n  *** MILESTONE HIT! ***")
                    print(f"      episode={ep+1}, step={step}, agent={agent_id}, reward={raw_reward:.1f}")
                    print(f"      {'WOOD STAGE!' if abs(raw_reward-128)<5 else ''}")
                    print(f"      {'STONE STAGE!' if abs(raw_reward-256)<5 else ''}\n")

                # Log every 50 steps even with no rewards (proof of life)
                if step % 50 == 0 and agent_id == 0:
                    inv = env.pickedup_object(0)
                    pos = env.get_position_text(0)
                    print(f"  [step {step:4d}] a0 action={action:<15s}"
                          f"  cum_reward={episode_rewards[0]:6.2f}"
                          f"  pos={pos}")
                    if inv and verbose:
                        print(f"             inventory: {inv.strip()}")

        all_episode_rewards.append(episode_rewards)
        print(f"\n  Episode {ep+1} summary:")
        for i, r in enumerate(episode_rewards):
            print(f"    agent_{i}: total_reward={r:.2f}")

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({num_episodes} episodes)")
    print(f"{'='*60}")
    for ep_idx, ep_rewards in enumerate(all_episode_rewards):
        print(f"  Episode {ep_idx+1}: {[f'{r:.1f}' for r in ep_rewards]}")

    if milestone_hits:
        print(f"\n  Milestones reached:")
        for ep, step, agent, reward in milestone_hits:
            print(f"    ep={ep} step={step} agent={agent} reward={reward:.1f}")
    else:
        print(f"\n  No milestones reached (reward >= 100).")
        print(f"  Check if small rewards (0.01-1.0) appeared — those confirm Dig works.")

    # ── Diagnosis ──
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS")
    print(f"{'='*60}")
    total_reward = sum(sum(ep) for ep in all_episode_rewards)
    if total_reward < 0.1:
        print("  PROBLEM: Zero total reward. Likely causes:")
        print("    1. Agents are not moving (check warmup / connection)")
        print("    2. Dig is not hitting blocks (orientation issue)")
        print("    3. Reward signal not reaching Python (Lua mod issue)")
        print("    Action: check stderr logs in the server run_dir")
    elif total_reward < 5.0:
        print(f"  MARGINAL: Very low reward ({total_reward:.2f}). Dig is working but")
        print(f"    agents rarely face blocks. Orientation/navigation is the bottleneck.")
        print(f"    This confirms the core problem with discrete LLM action selection.")
    elif milestone_hits:
        print(f"  SUCCESS: Milestones reached! The environment is functional.")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    The RL training loop should work.")
    else:
        print(f"  PARTIAL: Dig works (reward={total_reward:.2f}) but no milestone yet.")
        print(f"    This is expected for a random-ish policy in limited steps.")
        print(f"    Increase --max-steps to 1000+ to see milestone rewards.")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scripted agent environment test")
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Number of agents (minimum 2, required by MarlCraftiumEnv)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Steps per episode. Use 500+ to have a chance at reward 128")
    parser.add_argument("--warmup-time", type=int, default=60,
                        help="Minimum warmup seconds before checking if media loaded")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every non-zero reward, not just large ones")
    args = parser.parse_args()

    run_test(
        num_agents=args.num_agents,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        warmup_time=args.warmup_time,
        verbose=args.verbose,
    )
