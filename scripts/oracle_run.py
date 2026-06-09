"""Oracle baseline for the five-chambers environment (rooms 1-3).

No LLM, no belief system, no critic — just scripted action cycles per chamber
plus an if/else over Hebbian weights for the ask/answer social decision. The
purpose is to isolate the Hebbian routing contribution from LLM noise so the
thesis claim ("Hebbian provides social intelligence") can be tested cleanly.

Four routing conditions, swapped via --routing:
    none         agents never ask, never respond     (solo upper baseline)
    random       ask random teammate when stuck      (noise floor)
    roundrobin   ask (i+1) % N when stuck            (no learning, no randomness)
    hebbian      ask argmax_j W[i, j] iff above tau   (the thesis condition)

Same seed across all four → milestone-timeline diffs are purely attributable
to the routing function.

Usage (Windows / WSL / Linux; needs the luanti/minetest binary on PATH):

    python scripts/oracle_run.py --routing hebbian --seed 42
    python scripts/oracle_run.py --routing none --seed 42
    # ... then diff the runs/oracle/<tag>/seed_42/episodes/ep_0001/event_log.jsonl

Outputs follow the standard run layout under runs/oracle/<tag>/seed_<N>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── PYTHONPATH bootstrap (matches multi_agent_craftium.py expectations) ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "src", "mindforge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from custom_environment_craftium import CraftiumEnvironmentInterface  # noqa: E402
from hebbian import HebbianConfig                                     # noqa: E402
from hebbian.graph import HebbianSocialGraph                          # noqa: E402
from mindforge.env.episode_logger import EpisodeLogger                # noqa: E402
from mindforge.run_layout import RunPaths                             # noqa: E402

logger = logging.getLogger("oracle_run")


# ────────────────────────────────────────────────────────────────────────────
# Map constants — mirrored from
# src/marl_craftium/craftium-envs/five-chambers/mods/five_chambers/config.lua
# Keep these in sync if config.lua changes.
# ────────────────────────────────────────────────────────────────────────────

# Ch1 spawns for N=3 (config.lua:60-64)
CH1_SPAWNS = {
    0: (1, 12, 1),
    1: (10, 12, 1),
    2: (5, 12, 10),
}

# Ch2 fallback spawns when the Ch1 timeout fires (config.lua:102-106)
CH2_FALLBACK_SPAWNS = {
    0: (3, 11, 19),
    1: (7, 11, 19),
    2: (11, 11, 19),
}

# Anvil coordination: PAIR_DIG_RATE - DECAY = 4 - 1 = +3 per tick;
# SOLO_DIG_RATE - DECAY = 0 per tick — solo can't break the anvil.
CH2_ANVIL_HP = 30
CH2_DIGGER_RADIUS = 3   # config.lua:153


# ────────────────────────────────────────────────────────────────────────────
# Chamber action cycles — deterministic primitives per chamber.
# The cycles are tuned conservatively: each chamber's cycle should
# (1) make agents move around their local area, (2) repeatedly attempt the
# chamber-specific interaction (dig wood/stone/anvil/switch), (3) not get
# permanently stuck. Tune by reading the diagnostic log of the first run.
# ────────────────────────────────────────────────────────────────────────────

CHAMBER_CYCLES = {
    # Ch1: solo tasks. Hit trees / stones / animals scattered around the chamber.
    # The TurnRight every few steps gives angular coverage of the spawn area.
    "ch1": [
        "MoveForward", "MoveForward", "MoveForward",
        "Dig", "Dig",
        "TurnRight", "TurnRight",
        "MoveForward", "MoveForward",
        "Dig",
        "TurnRight",
        "LookDown",
        "Dig",
        "LookUp",
        "TurnRight",
    ],
    # Ch2: anvil coordination. Long Dig runs (each Dig = 15 physics ticks) so
    # pair-formation has a wide coordination window (ACTIVE_WINDOW=18 ticks).
    "ch2": [
        "MoveForward", "MoveForward",
        "Dig", "Dig", "Dig",          # ~45 physics ticks of punching
        "TurnRight",
        "Slot1",                       # try to equip whatever dropped
        "Dig", "Dig",
        "TurnRight",
    ],
    # Ch3: switch puzzle. Each agent walks into its cell, digs the switch,
    # then heads toward the communal area (m19).
    "ch3": [
        "MoveForward", "MoveForward", "MoveForward",
        "Dig", "Dig",                  # press switch
        "TurnRight",
        "MoveForward", "MoveForward",
        "TurnLeft",
    ],
    # Fallback when chamber lookup returns None (e.g., between chambers or
    # mid-teleport). ScanArea finds a heading; MoveForward exits the no-mans-land.
    None: ["ScanArea", "MoveForward", "MoveForward", "MoveForward"],
}


# ────────────────────────────────────────────────────────────────────────────
# Routing functions — the four conditions. Each returns:
#   ask_target : int or None   — who to ask for help this step (None = don't ask)
#   accept_from: set[int]      — set of inbox senders whose ask to accept
# ────────────────────────────────────────────────────────────────────────────

def route_none(W, agent_i, stuck, inbox, *, tau_ask, tau_help, rng):
    return None, set()


def route_random(W, agent_i, stuck, inbox, *, tau_ask, tau_help, rng):
    N = W.shape[0]
    candidates = [j for j in range(N) if j != agent_i]
    ask = None
    if stuck and candidates:
        ask = rng.choice(candidates)
    accept = set(inbox)
    return ask, accept


def route_roundrobin(W, agent_i, stuck, inbox, *, tau_ask, tau_help, rng):
    N = W.shape[0]
    ask = ((agent_i + 1) % N) if stuck else None
    accept = set(inbox)
    return ask, accept


def route_hebbian(W, agent_i, stuck, inbox, *, tau_ask, tau_help, rng):
    """The thesis condition. Asymmetric: ask uses my outgoing W[i, :],
    accept uses my outgoing W[i, sender] (MY representation of bond to sender).
    """
    N = W.shape[0]
    ask = None
    if stuck:
        row = W[agent_i].copy()
        row[agent_i] = -np.inf
        j = int(np.argmax(row))
        if row[j] >= tau_ask:
            ask = j
    accept = {s for s in inbox if W[agent_i, s] >= tau_help}
    return ask, accept


ROUTING_FUNCS = {
    "none":       route_none,
    "random":     route_random,
    "roundrobin": route_roundrobin,
    "hebbian":    route_hebbian,
}


# ────────────────────────────────────────────────────────────────────────────
# Oracle policy
# ────────────────────────────────────────────────────────────────────────────

class OraclePolicy:
    """Per-agent deterministic action selector with help-coupling override.

    Each agent maintains a step counter into the current chamber's cycle. When
    the routing function flags this agent as "helping" someone, we override
    the cycle for HELP_WINDOW_STEPS with an ApproachTarget macro pointing at
    the partner's last-known position. After the window expires, we resume
    the cycle exactly where we left off.
    """

    HELP_WINDOW_STEPS = 20

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self._chamber_steps = {i: 0 for i in range(num_agents)}
        self._help_target = {i: None for i in range(num_agents)}
        self._help_window = {i: 0 for i in range(num_agents)}

    def set_help_target(self, agent_i: int, target_j: int) -> None:
        self._help_target[agent_i] = target_j
        self._help_window[agent_i] = self.HELP_WINDOW_STEPS

    def pick_action(self, agent_i: int, chamber: str | None,
                    my_pos, partner_pos) -> str:
        """Return the action string for agent_i this step.

        my_pos / partner_pos are (x, y, z) tuples or None.
        """
        # Help-window override: drift toward partner via ApproachTarget macro.
        if self._help_window[agent_i] > 0 and self._help_target[agent_i] is not None:
            self._help_window[agent_i] -= 1
            # ApproachTarget is an 11-step macro that does TurnRight×3 then
            # MoveForward×8. The env's macro queue handles the unroll; we
            # only need to emit the macro name once per window. Subsequent
            # steps inside the window emit MoveForward as a cheap default —
            # the macro queue is already chewing through the original.
            if self._help_window[agent_i] == self.HELP_WINDOW_STEPS - 1:
                return "ApproachTarget"
            return "MoveForward"

        # Default: pull from the chamber cycle.
        cycle = CHAMBER_CYCLES.get(chamber, CHAMBER_CYCLES[None])
        idx = self._chamber_steps[agent_i] % len(cycle)
        self._chamber_steps[agent_i] += 1
        return cycle[idx]


# ────────────────────────────────────────────────────────────────────────────
# Main run
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--routing", choices=list(ROUTING_FUNCS.keys()),
                   default="hebbian",
                   help="Social-routing condition. The if/else under test.")
    p.add_argument("--tau-stuck", type=int, default=30,
                   help="Steps without reward before agent is 'stuck' and may ask.")
    p.add_argument("--tau-ask", type=float, default=0.25,
                   help="Minimum W[i,j] required for agent i to ask j (hebbian only).")
    p.add_argument("--tau-help", type=float, default=0.20,
                   help="Minimum W[i,sender] required for i to accept sender's ask.")
    p.add_argument("--tag", type=str, default=None,
                   help="Run tag; output lands at runs/oracle/<tag>/seed_<seed>/. "
                        "Defaults to the routing name so all four conditions "
                        "produce separable trees.")
    p.add_argument("--warmup-secs", type=int, default=60,
                   help="Seconds to wait after env reset before stepping. The "
                        "VoxeLibre media load needs 30–120s on first launch.")
    p.add_argument("--ch1-timeout-steps", type=int, default=400,
                   help="Mirrors multi_agent_craftium.py — Ch1→Ch2 timeout.")
    p.add_argument("--no-hebbian-update", action="store_true",
                   help="Skip HebbianSocialGraph.update() per step. Useful with "
                        "--routing none to confirm the env-only baseline.")
    return p.parse_args()


def setup_run(args) -> tuple[RunPaths, str]:
    tag = args.tag or f"oracle_{args.routing}"
    rp = RunPaths.create_tagged(
        tag=tag, seed=args.seed,
        root=os.path.join("runs", "oracle"),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[oracle] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Mirror config snapshot.
    with open(rp.config_json, "w") as f:
        json.dump({
            "run_id": rp.run_id,
            "start_ts": datetime.now().isoformat(),
            "cli_args": vars(args),
            "map_constants": {
                "CH1_SPAWNS": CH1_SPAWNS,
                "CH2_FALLBACK_SPAWNS": CH2_FALLBACK_SPAWNS,
                "CH2_ANVIL_HP": CH2_ANVIL_HP,
                "CH2_DIGGER_RADIUS": CH2_DIGGER_RADIUS,
            },
        }, f, indent=2, default=str)
    return rp, tag


def wait_for_media(env, num_agents: int, warmup_secs: int) -> None:
    """Same pattern as multi_agent_craftium.py — poll std-dev of obs frames
    until they look like rendered game world rather than the loading bar.
    """
    start = time.time()
    consec = 0
    while time.time() - start < max(warmup_secs * 3, 240):
        observations = env.warmup_noop()
        stds = []
        if observations:
            for obs in observations:
                stds.append(float(np.std(obs.astype(np.float32))) if obs is not None else 0.0)
        elapsed = time.time() - start
        if elapsed >= warmup_secs and stds and all(s > 25.0 for s in stds):
            consec += 1
            if consec >= 3:
                logger.info("media loaded after %.0fs  std=%s", elapsed, stds)
                return
        else:
            consec = 0
        time.sleep(2)
    logger.warning("media load timeout — proceeding anyway")


def run() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rp, tag = setup_run(args)
    logger.info("run_id=%s  routing=%s", rp.run_id, args.routing)

    # Plumb the Ch1 timeout into the Lua mod so the teleport fires at a known step.
    os.environ["CH1_TIMEOUT_TICKS"] = str(args.ch1_timeout_steps * args.num_agents * 3)

    env = CraftiumEnvironmentInterface(
        num_agents=args.num_agents,
        obs_width=160, obs_height=90,    # tiny frames — we don't render any vision
        max_steps=args.max_steps,
        seed=args.seed,
    )

    hebb_cfg = HebbianConfig(
        enabled=True,                # always built; --routing none just doesn't read it
        num_agents=args.num_agents,
        init_weight=0.1,
    )
    hebb = HebbianSocialGraph(hebb_cfg, agent_roles=[0] * args.num_agents)
    route_fn = ROUTING_FUNCS[args.routing]
    rng = random.Random(args.seed)

    # Aggregate run-level counters (filled in below as each episode finalizes).
    run_milestone_counts: dict[str, int] = {}
    run_steps_to_first: dict[str, int] = {}

    try:
        env.reset()
        env.reset_milestone_offset()
        try:
            env.signal_warmup_complete    # method exists on patched env class
        except AttributeError:
            pass
        wait_for_media(env, args.num_agents, args.warmup_secs)
        try:
            env.signal_warmup_complete()
        except AttributeError:
            pass

        global_step = 0
        for ep in range(args.episodes):
            ep_idx = ep + 1
            logger.info("===== episode %d / %d =====", ep_idx, args.episodes)
            if ep > 0:
                env.reset()
                env.reset_milestone_offset()

            ep_log = EpisodeLogger(run_dir=rp.root, episode=ep_idx)
            policy = OraclePolicy(num_agents=args.num_agents)
            stuck_counter = {i: 0 for i in range(args.num_agents)}
            inboxes: dict[int, list[int]] = {i: [] for i in range(args.num_agents)}
            decisions_path = ep_log.dir / "oracle_decisions.jsonl"
            decisions_f = open(decisions_path, "w", encoding="utf-8")
            hebbian_path = ep_log.dir / "hebbian_snapshots.jsonl"
            hebbian_f = open(hebbian_path, "w", encoding="utf-8")

            for step in range(args.max_steps):
                global_step += 1

                # ── 1. Snapshot pre-step state ──
                positions = {i: env.get_agent_position(i) for i in range(args.num_agents)}
                chambers  = {i: env.get_chamber(i) for i in range(args.num_agents)}

                # ── 2. Action selection: deterministic cycle with help-window override ──
                actions: dict[int, str] = {}
                for i in range(args.num_agents):
                    target = policy._help_target[i]
                    partner_pos = positions.get(target) if target is not None else None
                    actions[i] = policy.pick_action(i, chambers[i], positions[i], partner_pos)

                # ── 3. Step the env one agent at a time (matches multi_agent_craftium) ──
                step_rewards = {i: 0.0 for i in range(args.num_agents)}
                for i in range(args.num_agents):
                    try:
                        env.step(actions[i], agentId=i)
                    except Exception as exc:
                        logger.warning("step failed agent=%d action=%s: %s",
                                       i, actions[i], exc)
                    step_rewards[i] = env.get_step_reward(i)

                # ── 4. Drain milestone events into per-agent rewards & log ──
                new_events = env.poll_milestone_events()
                for ev in new_events:
                    ep_log.log_event(ev)
                    mid = ev.get("milestone", "")
                    run_milestone_counts[mid] = run_milestone_counts.get(mid, 0) + 1
                    if mid not in run_steps_to_first:
                        run_steps_to_first[mid] = global_step
                    r = ev.get("reward", 0.0)
                    for contrib_name in ev.get("contributors", []):
                        # Lua emits 'agent0' (no underscore); Python uses
                        # 'agent_0'. Strip both prefixes so either form parses.
                        _id_str = contrib_name.removeprefix("agent_").removeprefix("agent")
                        try:
                            cid = int(_id_str)
                        except ValueError:
                            continue
                        step_rewards[cid] = step_rewards.get(cid, 0.0) + r

                # ── 5. Update stuck counters (reward this step resets it) ──
                for i in range(args.num_agents):
                    if step_rewards[i] > 0:
                        stuck_counter[i] = 0
                    else:
                        stuck_counter[i] += 1

                # ── 6. Routing decisions FOR THIS STEP's outbound comms ──
                comm_events: list[tuple[int, int]] = []
                messages: dict[int, str] = {i: "" for i in range(args.num_agents)}
                for i in range(args.num_agents):
                    is_stuck = stuck_counter[i] >= args.tau_stuck
                    ask, accept = route_fn(
                        hebb.W, i, is_stuck, inboxes[i],
                        tau_ask=args.tau_ask, tau_help=args.tau_help, rng=rng,
                    )
                    if ask is not None:
                        # i asks j for help
                        comm_events.append((i, ask))
                        messages[i] = f"HELP→agent_{ask}"
                        stuck_counter[i] = 0          # cooldown — don't ask every step
                        # write to j's inbox for NEXT step
                        inboxes[ask].append(i)
                        decisions_f.write(json.dumps({
                            "step": step, "agent": i, "decided": "ask",
                            "target": ask, "routing": args.routing,
                            "W_used": float(hebb.W[i, ask]),
                            "stuck_counter": stuck_counter[i],
                        }) + "\n")
                    for sender in accept:
                        # i decides to help sender → couple motion, ACK comm
                        policy.set_help_target(i, sender)
                        comm_events.append((i, sender))
                        if not messages[i]:
                            messages[i] = f"ACK→agent_{sender}"
                        else:
                            messages[i] += f" ACK→agent_{sender}"
                        decisions_f.write(json.dumps({
                            "step": step, "agent": i, "decided": "help",
                            "target": sender, "routing": args.routing,
                            "W_used": float(hebb.W[i, sender]),
                        }) + "\n")

                # Inbox is single-step (a request that wasn't accepted this step is gone).
                # Matches the "sliding window" semantics in social_module.txt:43.
                inboxes = {i: [] for i in range(args.num_agents)}

                # ── 7. Hebbian update ──
                pos_list = [positions[i] for i in range(args.num_agents)]
                reward_list = [step_rewards[i] for i in range(args.num_agents)]
                if not args.no_hebbian_update:
                    hebb.update(
                        positions=pos_list,
                        step_rewards=reward_list,
                        advantages=None,
                        comm_events=comm_events,
                    )

                # ── 8. Step-level logging ──
                ep_log.log_step(
                    step=step,
                    positions=positions,
                    actions=actions,
                    messages=messages,
                    task_rewards=step_rewards,
                    comm_rewards={i: 0.0 for i in range(args.num_agents)},
                    infos={"chambers": chambers},
                )
                # Per-message records (so messages.jsonl matches the LLM-run layout)
                for i, txt in messages.items():
                    if not txt:
                        continue
                    ep_log.log_message({
                        "t": step,
                        "sender": f"agent_{i}",
                        "receiver": txt.split("→")[-1].split()[0] if "→" in txt else "all",
                        "text": txt,
                        "tokens": len(txt.split()),
                        "valid": True,
                        "rewarded_base": 0.0,
                        "rewarded_milestone": 0.0,
                        "chamber": chambers[i] or "",
                        "routing": args.routing,
                    })
                # Hebbian snapshot per step (smaller than per-episode — full bond
                # trajectory for the bond-evolution plot)
                if step % 5 == 0:
                    hebbian_f.write(json.dumps({
                        "step": step, "episode": ep_idx,
                        "W": hebb.W.tolist(),
                    }) + "\n")

                if step % 50 == 0:
                    w_mean = float(hebb.W[hebb.W > 0].mean()) if (hebb.W > 0).any() else 0.0
                    logger.info("ep=%d step=%d  chambers=%s  W_mean=%.3f  ms=%d",
                                ep_idx, step, [chambers[i] for i in range(args.num_agents)],
                                w_mean, len(run_milestone_counts))

            # ── Episode finalize ──
            decisions_f.close()
            hebbian_f.close()
            ep_log.finalize({
                "episode": ep_idx,
                "routing": args.routing,
                "milestones_counts_so_far": dict(run_milestone_counts),
                "steps_to_first_so_far": dict(run_steps_to_first),
                "final_W": hebb.W.tolist(),
            })
            logger.info("ep %d done — milestones so far: %s", ep_idx, run_milestone_counts)

    finally:
        try:
            env.close()
        except Exception:
            pass

    # ── Run-level summary ──
    summary = {
        "run_id": rp.run_id,
        "routing": args.routing,
        "seed": args.seed,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "milestones_counts": run_milestone_counts,
        "steps_to_first": run_steps_to_first,
        "final_W": hebb.W.tolist(),
        # Key metrics for H1 comparison:
        "m8_first_anvil_break_step": run_steps_to_first.get("m8_anvil_A1"),
        "ch2_milestone_count": sum(
            c for m, c in run_milestone_counts.items()
            if m.startswith("m8_") or m.startswith("m9_") or m.startswith("m10_")
            or m.startswith("m11_") or m.startswith("m12_") or m.startswith("m13_")
        ),
        "ch3_door_open_step": run_steps_to_first.get("m18_door_opened"),
    }
    with open(rp.final_metrics_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("==== summary ====")
    logger.info("m8 first anvil break:  step=%s", summary["m8_first_anvil_break_step"])
    logger.info("Ch2 milestones (m8-m13): %d", summary["ch2_milestone_count"])
    logger.info("m18 door opened:        step=%s", summary["ch3_door_open_step"])
    logger.info("output → %s", rp.root)


if __name__ == "__main__":
    run()
