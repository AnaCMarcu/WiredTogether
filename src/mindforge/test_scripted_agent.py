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
    """Rule-based policy designed to maximise movement and block-breaking.

    Strategy (repeating cycle):
      1. Walk forward for many steps (actually changes X/Z position)
      2. Look down and dig ground-level blocks (guaranteed to face a block)
      3. Look up to reset camera pitch
      4. Turn to a new direction to avoid getting stuck on one wall
      5. Repeat

    If no reward for a long stretch, execute a full escape sequence:
    jump + turn 180 + walk forward.

    No vision model. No LLM.
    """

    # Cycle definition: list of (duration, action)
    # Design rules:
    #   - NO LookUp: after LookDown the camera overshoots when looking back up,
    #     leaving the crosshair aimed at sky → Dig hits nothing → escape fires.
    #   - MoveForward AFTER Dig: dropped items lie on the ground where the block
    #     was; walking over them auto-collects them (Minetest pickup radius ~1 block).
    #   - LookDown before each Dig block: re-aims camera regardless of current pitch.
    _CYCLE = [
        (3,  "LookDown"),      # aim at ground
        (12, "Dig"),           # break ground blocks
        (6,  "MoveForward"),   # walk over drops → auto-collect items
        (3,  "TurnRight"),     # new heading
        (3,  "LookDown"),      # re-aim
        (12, "Dig"),
        (6,  "MoveForward"),   # collect
        (3,  "TurnRight"),
        (15, "MoveForward"),   # explore — find new terrain / trees
        (3,  "Jump"),          # clear obstacles
        (10, "MoveForward"),
        (3,  "TurnRight"),
    ]
    _CYCLE_LEN = sum(d for d, _ in _CYCLE)

    def __init__(self):
        self._step = 0
        self._no_reward_streak = 0
        self._escape_remaining = 0

    def act(self, obs: np.ndarray, last_reward: float) -> str:
        t = self._step
        self._step += 1

        if last_reward > 0.001:
            self._no_reward_streak = 0
        else:
            self._no_reward_streak += 1

        # Escape sequence: stuck for too long → jump + turn + sprint
        if self._escape_remaining > 0:
            self._escape_remaining -= 1
            seq_pos = self._escape_remaining
            if seq_pos >= 25:
                return "Jump"
            elif seq_pos >= 15:
                return "TurnRight"
            else:
                return "MoveForward"

        if self._no_reward_streak > 40:
            self._no_reward_streak = 0
            self._escape_remaining = 35
            return "Jump"

        # Normal cycle
        pos = t % self._CYCLE_LEN
        acc = 0
        for duration, action in self._CYCLE:
            acc += duration
            if pos < acc:
                return action
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
        consecutive_loaded = 0  # require sustained signal, not a single fluke
        while time.time() - warmup_start < max_warmup:
            observations = env.warmup_noop()
            elapsed = time.time() - warmup_start

            stds = []
            if observations:
                for obs in observations:
                    stds.append(float(np.std(obs.astype(np.float32))) if obs is not None else 0.0)

            if elapsed - last_log >= 15.0 and stds:
                std_str = ", ".join(f"a{i}={s:.1f}" for i, s in enumerate(stds))
                print(f"    [{elapsed:.0f}s] std: {std_str}  (>25 = loaded)")
                last_log = elapsed

            # Lower threshold to 25 (was 30) and require 3 consecutive checks
            if elapsed >= warmup_time and stds and all(s > 25.0 for s in stds):
                consecutive_loaded += 1
                if consecutive_loaded >= 3:
                    all_loaded = True
                    break
            else:
                consecutive_loaded = 0
            time.sleep(2)

        elapsed = time.time() - warmup_start
        status = "loaded" if all_loaded else "timeout"
        print(f"  Warm-up done ({elapsed:.0f}s, {status})")

        # ── Inventory sanity check ──
        # Verify the Lua mod wrote inventory files and that init_tools were given.
        # Expected: slot 1 = stone axe, slot 2 = torches (from init_tools in Lua mod).
        print(f"\n  [INVENTORY CHECK]")
        inv_ok = True
        world_path = env._get_world_path()
        for i in range(num_agents):
            inv_file = os.path.join(world_path, f"inv_agent{i}.txt")
            # Read raw file
            try:
                with open(inv_file, "r") as f:
                    raw = f.read().strip()
                print(f"    agent_{i} raw inventory file: '{raw}'")
                if not raw or all(s == "" for s in raw.split("|")[1:]):
                    print(f"    agent_{i} WARNING: inventory is empty — init_tools not applied?")
                    inv_ok = False
                else:
                    # Check for expected tools
                    has_axe   = "axe_stone" in raw
                    has_torch = "torch" in raw
                    status_str = []
                    if has_axe:   status_str.append("stone axe OK")
                    else:         status_str.append("stone axe MISSING")
                    if has_torch: status_str.append("torches OK")
                    else:         status_str.append("torches MISSING")
                    print(f"    agent_{i}: {', '.join(status_str)}")
                    if not has_axe or not has_torch:
                        inv_ok = False
            except FileNotFoundError:
                print(f"    agent_{i} ERROR: inventory file not found at {inv_file}")
                print(f"      -> Lua globalstep may not have run yet, or world path is wrong")
                inv_ok = False

        if inv_ok:
            print(f"  [INVENTORY CHECK] PASS — all agents have expected tools")
        else:
            print(f"  [INVENTORY CHECK] FAIL — check Lua mod on_joinplayer and world path")
            print(f"    World path used: {world_path}")
        print()

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
        print(f"  Episode {ep_idx+1}: {[f'{r:.3f}' for r in ep_rewards]}")

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
    if total_reward < 0.001:
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


def run_dig_test(num_agents: int, warmup_time: int):
    """Focused dig test: Dig at eye level (no LookDown) then collect drops.

    Strategy:
      - Agents spawn with random yaw facing the environment.
      - Eye-level dig hits TREE TRUNKS, which the stone axe can break quickly.
      - LookDown was removed: axe on dirt/grass (wrong tool) takes 50+ ticks
        to break — far more than our sustained-tick budget.
      - After each dig burst, MoveForward collects the dropped items.

    What to look for:
      - PASS: reward spikes on individual Dig steps, inventory fills with wood/items
      - FAIL: reward stays 0.0 → Dig not registering (check connection / server log)
      - PARTIAL: reward spikes but inventory empty → blocks break, need more MoveForward
    """
    print(f"\n{'='*60}")
    print(f"  DIG TEST — eye-level Dig x60 + MoveForward to collect per agent")
    print(f"  Agents face random directions at spawn; some will face trees.")
    print(f"{'='*60}\n")

    env = CraftiumEnvironmentInterface(num_agents=num_agents, obs_width=320, obs_height=180, max_steps=200)
    env.reset()

    # Proper polling warmup — must keep reading the socket or the TCP buffer fills up
    # and the connection breaks after ~30s. Same logic as run_test().
    print(f"  Waiting for warmup ({warmup_time}s min, polling socket)...")
    warmup_start = time.time()
    all_loaded = False
    last_log = 0.0
    consecutive_loaded = 0
    while time.time() - warmup_start < 900:
        observations = env.warmup_noop()
        elapsed = time.time() - warmup_start

        stds = [float(np.std(obs.astype(np.float32))) if obs is not None else 0.0
                for obs in (observations or [])]

        if elapsed - last_log >= 15.0 and stds:
            std_str = ", ".join(f"a{i}={s:.1f}" for i, s in enumerate(stds))
            print(f"    [{elapsed:.0f}s] std: {std_str}  (>25 = loaded)")
            last_log = elapsed

        if elapsed >= warmup_time and stds and all(s > 25.0 for s in stds):
            consecutive_loaded += 1
            if consecutive_loaded >= 3:
                all_loaded = True
                break
        else:
            consecutive_loaded = 0
        time.sleep(2)

    elapsed = time.time() - warmup_start
    print(f"  Warmup done ({elapsed:.0f}s, {'loaded' if all_loaded else 'timeout'})")

    # No LookDown: stone axe on dirt/grass (wrong tool) takes 50+ physics ticks
    # to break. Eye-level (horizontal) aims at tree trunks — correct tool match.
    print(f"\n  [PHASE 1] Dig x60 at eye level — aim at whatever is directly ahead")
    print(f"  (agents spawned with random yaw; some will face trees)")
    step_rewards = {i: [] for i in range(num_agents)}
    inv_before = {i: env.pickedup_object(i) for i in range(num_agents)}

    for step in range(60):
        for agent_id in range(num_agents):
            if env._terminations.get(f"agent_{agent_id}", False):
                continue
            env.step("Dig", agentId=agent_id)
            r = env.get_step_reward(agent_id)
            step_rewards[agent_id].append(r)
            if r > 0.001:
                print(f"    step {step:3d} agent_{agent_id}: reward={r:.4f}  ← block hit")

    # Walk forward to collect dropped items (items drop on the ground where the
    # block was; Minetest requires stepping on them to pick them up).
    print(f"\n  [PHASE 2] MoveForward x10 — collect dropped items")
    for _ in range(10):
        for agent_id in range(num_agents):
            env.step("MoveForward", agentId=agent_id)

    inv_after = {i: env.pickedup_object(i) for i in range(num_agents)}

    print(f"\n  [RESULTS]")
    for i in range(num_agents):
        rewards = step_rewards[i]
        nonzero = [r for r in rewards if r > 0.001]
        total = sum(rewards)
        print(f"  agent_{i}:")
        print(f"    total reward over 60 Dig steps: {total:.4f}")
        print(f"    steps with nonzero reward: {len(nonzero)}/{len(rewards)}")
        print(f"    reward per hit: {(total/len(nonzero)):.4f}" if nonzero else "    reward per hit: N/A (no hits)")
        print(f"    inventory before: {inv_before[i]}")
        print(f"    inventory after:  {inv_after[i]}")

        if total < 0.001:
            print(f"    FAIL: Dig produced no reward — crosshair may be misaimed or Dig action not registering")
        elif not nonzero:
            print(f"    FAIL: Total reward nonzero but no per-step spikes — reward may be passive/continuous")
        elif inv_before[i] == inv_after[i]:
            print(f"    PARTIAL: Reward OK but inventory unchanged — blocks breaking but items not being picked up")
        else:
            print(f"    PASS: Dig works, reward spikes on hits, inventory changed")

    # The tail_server_log() calls inside env.step("Dig") already printed matching lines
    # in real-time. Do a final sweep to catch anything buffered after the last step.
    print(f"\n  [SERVER LOG — final sweep]")
    remaining = env.tail_server_log()
    if not remaining:
        print(f"  (no new lines — all events were printed in real-time above)")
        print(f"  Note: dirt/grass digs don't emit [TOOLS]; only tree/stone/iron/diamond do.")

    env.close()
    print("\nDig test done.")


def _wait_for_warmup(env, warmup_time):
    """Shared warmup loop: poll NoOps until all agents show std>25 for 3 checks."""
    import time as _time
    print(f"  Waiting for media ({warmup_time}s min)...")
    warmup_start = _time.time()
    consecutive = 0
    last_log = 0.0
    while _time.time() - warmup_start < 900:
        observations = env.warmup_noop()
        elapsed = _time.time() - warmup_start
        stds = [float(np.std(obs.astype(np.float32))) if obs is not None else 0.0
                for obs in (observations or [])]
        if elapsed - last_log >= 15.0 and stds:
            print(f"    [{elapsed:.0f}s] std: {', '.join(f'a{i}={s:.1f}' for i,s in enumerate(stds))}")
            last_log = elapsed
        if elapsed >= warmup_time and stds and all(s > 25.0 for s in stds):
            consecutive += 1
            if consecutive >= 3:
                break
        else:
            consecutive = 0
        _time.sleep(2)
    print(f"  Warmup done ({_time.time()-warmup_start:.0f}s)")


def run_milestones_test(num_agents: int, max_steps: int, warmup_time: int):
    """D2 end-to-end test: scripted agents + milestone event polling.

    Runs for max_steps steps, polls env.poll_milestone_events() every step,
    and prints each new event. At the end, reports which of M1-M7 fired for
    which agents.

    PASS criteria:
      - M1 (move_5) fired for ALL agents
      - M2 (dig_3_any) fired for >= 2 agents
      - At least one M5 or M6 event appeared (animal kill; luck-dependent)
      - milestone_events.jsonl exists and every line is valid JSON with the
        expected {step, milestone, contributors, reward} schema
    """
    import collections

    print(f"\n{'='*60}")
    print(f"  MILESTONE TEST (D2) — {num_agents} agents, {max_steps} steps")
    print(f"{'='*60}\n")

    env = CraftiumEnvironmentInterface(
        num_agents=num_agents,
        obs_width=320,
        obs_height=180,
        max_steps=max_steps,
    )
    env.reset()
    _wait_for_warmup(env, warmup_time)
    env.reset_milestone_offset()

    # Per-agent scripted policies
    policies      = [ScriptedPolicy() for _ in range(num_agents)]
    last_rewards  = [0.0] * num_agents

    # Accumulated milestone events across the run
    all_events    = []
    # fired_for[milestone_id] = set of agent names that earned it
    fired_for     = collections.defaultdict(set)

    for step in range(max_steps):
        if env.all_done():
            print(f"  All agents done at step {step}")
            break

        for agent_id in range(num_agents):
            agent_name = f"agent_{agent_id}"
            if env._terminations.get(agent_name, False):
                continue
            obs    = env.get_agent_frame(agent_id)
            if obs is None:
                obs = np.zeros((180, 320, 3), dtype=np.uint8)
            action = policies[agent_id].act(obs, last_rewards[agent_id])
            env.step(action, agentId=agent_id)
            last_rewards[agent_id] = env.get_step_reward(agent_id)

        # Poll new milestone events
        new_events = env.poll_milestone_events()
        for ev in new_events:
            all_events.append(ev)
            mid = ev.get("milestone", "?")
            for name in ev.get("contributors", []):
                fired_for[mid].add(name)
            print(f"  [step {step:4d}] MILESTONE {mid}"
                  f"  contributors={ev.get('contributors')}"
                  f"  reward={ev.get('reward')}")

        # Print server log lines with [MILESTONE] tag
        env.tail_server_log(tags=("[MILESTONE]",), print_lines=False)

        if step % 100 == 0 and step > 0:
            chamber_str = ", ".join(
                f"a{i}={env.get_chamber(i) or '?'}" for i in range(num_agents)
            )
            print(f"  [step {step:4d}] chambers: {chamber_str}"
                  f"  events so far: {len(all_events)}")

    # ── Final report ──────────────────────────────────────────────

    ch1_milestones = ["m1_move_5","m2_dig_3_any","m3_pickup_3",
                      "m4_dig_5_wood","m5_kill_1_animal","m6_kill_2_animals","m7_dig_3_stone"]

    print(f"\n{'='*60}")
    print(f"  MILESTONE REPORT — {len(all_events)} total events")
    print(f"{'='*60}")
    agent_names = [f"agent_{i}" for i in range(num_agents)]

    print(f"\n  Ch1 milestone coverage:")
    for mid in ch1_milestones:
        who = fired_for.get(mid, set())
        fired_all = agent_names and all(n in who for n in agent_names)
        flag = "ALL" if fired_all else f"{len(who)}/{num_agents}"
        print(f"    {mid:<30s}  [{flag}]  {sorted(who)}")

    # ── Schema check ──────────────────────────────────────────────
    print(f"\n  JSONL schema check ({len(all_events)} lines):")
    schema_ok = True
    for ev in all_events:
        for key in ("step", "milestone", "contributors", "reward"):
            if key not in ev:
                print(f"    FAIL: missing key '{key}' in {ev}")
                schema_ok = False
                break
    if schema_ok and all_events:
        print(f"    PASS — all {len(all_events)} events have required keys")
    elif not all_events:
        print(f"    WARNING — no events recorded; check Lua emit_milestone()")

    # ── World path (show user where JSONL lives) ──────────────────
    try:
        wpath = env._get_world_path()
        print(f"\n  milestone_events.jsonl at: {wpath}/milestone_events.jsonl")
    except Exception:
        pass

    # ── Pass/fail verdict ─────────────────────────────────────────
    print(f"\n  VERDICT:")
    m1_all = all(n in fired_for.get("m1_move_5", set()) for n in agent_names)
    m2_ok  = len(fired_for.get("m2_dig_3_any", set())) >= max(1, num_agents - 1)
    kill_ok = bool(fired_for.get("m5_kill_1_animal") or fired_for.get("m6_kill_2_animals"))

    if m1_all and m2_ok:
        print(f"  PASS — M1 fired for all {num_agents} agents, M2 fired for >={num_agents-1}.")
        if kill_ok:
            print(f"  BONUS — at least one animal kill milestone fired.")
        else:
            print(f"  (M5/M6 not seen — agents may not have encountered animals; acceptable.)")
    else:
        if not m1_all:
            print(f"  FAIL — M1 (move_5) not fired for all agents. Fired for: {fired_for.get('m1_move_5')}")
            print(f"    → Check if agents are actually moving (warmup long enough?)")
        if not m2_ok:
            print(f"  FAIL — M2 (dig_3_any) fired for <{num_agents-1} agents. Fired for: {fired_for.get('m2_dig_3_any')}")
            print(f"    → Check on_dignode hook in milestones.lua and Ch1 geometry.")

    env.close()
    print("\nMilestone test done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scripted agent environment test")
    parser.add_argument("--mode", choices=["normal", "dig", "milestones"], default="normal",
                        help="normal=full scripted test; dig=focused dig; milestones=D2 end-to-end")
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Number of agents (minimum 2, required by MarlCraftiumEnv)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Steps per episode. Use 400 for milestones mode.")
    parser.add_argument("--warmup-time", type=int, default=60,
                        help="Minimum warmup seconds before checking if media loaded")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every non-zero reward, not just large ones")
    args = parser.parse_args()

    if args.mode == "dig":
        run_dig_test(
            num_agents=args.num_agents,
            warmup_time=args.warmup_time,
        )
    elif args.mode == "milestones":
        run_milestones_test(
            num_agents=max(args.num_agents, 3),
            max_steps=max(args.max_steps, 400),
            warmup_time=args.warmup_time,
        )
    else:
        run_test(
            num_agents=args.num_agents,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            warmup_time=args.warmup_time,
            verbose=args.verbose,
        )
