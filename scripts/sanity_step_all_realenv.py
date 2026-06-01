#!/usr/bin/env python3
"""Real-Luanti sanity check for CraftiumEnvironmentInterface.step_all().

Runs INSIDE the apptainer container (via daic/sanity_step_all.sbatch) because
it needs a live Luanti server. Needs NO LLM and NO GPU — step_all is pure env.

This is a SMOKE + STRUCTURAL check, deliberately NOT a byte-identity check:
the Lua step_counter ticks in wall-clock, so the real env is non-deterministic
across runs even at a fixed seed. Exact step==step_all identity is already
proven on the deterministic mock in scripts/parity_test_stepping.py. Here we
only confirm step_all drives real Luanti correctly:

  Phase 0  baseline      the untouched turn-based step() still works (the
                         _resolve_action_for_agent extraction didn't regress it)
  Phase A  step_all      runs N steps without crashing; every call returns
                         well-formed obs (all live agents, right shape), finite
                         rewards, valid resolved primitives; the world actually
                         advances (an agent physically moves on MoveForward)

Exit 0 = all checks passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

# custom_environment_craftium lives in src/mindforge and imports its siblings
# bare; add that dir to the path (PYTHONPATH=src already covers marl_craftium).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src", "mindforge")))

from custom_environment_craftium import (  # noqa: E402
    CraftiumEnvironmentInterface,
    VALID_ACTIONS,
)

_FAILS = []


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)
    if not cond:
        _FAILS.append(msg)


def _dist(a, b):
    if not a or not b:
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-time", type=int, default=120,
                    help="Seconds of warmup_noop() while VoxeLibre media loads.")
    ap.add_argument("--steps", type=int, default=20,
                    help="step_all steps to drive in Phase A.")
    args = ap.parse_args()

    n = args.num_agents
    print("=" * 64, flush=True)
    print(f"step_all real-Luanti sanity — N={n} seed={args.seed}", flush=True)
    print("=" * 64, flush=True)

    env = CraftiumEnvironmentInterface(num_agents=n, seed=args.seed)
    env.reset()
    env.reset_milestone_offset()

    # ── Warmup: keep channels alive while media loads ──
    print(f"Warmup {args.warmup_time}s (warmup_noop) ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < args.warmup_time:
        env.warmup_noop()
        time.sleep(2)
    try:
        env.signal_warmup_complete()
    except Exception as exc:
        print(f"  (signal_warmup_complete: {exc})", flush=True)

    def _live(i):
        return not env._terminations.get(f"agent_{i}", False)

    # ── Phase 0: turn-based step() baseline (regression guard) ──
    print("Phase 0 — turn-based step() baseline", flush=True)
    base_ok = True
    for i in range(n):
        if not _live(i):
            continue
        try:
            env.step("MoveForward", agentId=i)
        except Exception as exc:
            base_ok = False
            check(False, f"step() raised for agent {i}: {exc}")
            break
    if base_ok:
        check(True, "untouched step() still drives the real env")

    # ── Phase A: step_all smoke ──
    print(f"Phase A — step_all smoke ({args.steps} steps)", flush=True)
    start_pos = {i: env.get_agent_position(i) for i in range(n)}
    seq = ["MoveForward", "TurnRight", "Jump", "MoveForward", "Dig"]
    crashed = False
    malformed = 0
    for s in range(args.steps):
        actions = {i: seq[(s + i) % len(seq)] for i in range(n) if _live(i)}
        if not actions:
            break
        try:
            obs, resolved = env.step_all(actions)
        except Exception as exc:
            check(False, f"step_all raised at step {s}: {exc!r}")
            crashed = True
            break
        ok_obs = all(env.get_agent_frame(i) is not None for i in range(n) if _live(i))
        ok_rew = all(np.isfinite(v) for v in env._step_rewards.values())
        ok_act = all(a in VALID_ACTIONS for a in resolved.values())
        if not (ok_obs and ok_rew and ok_act):
            malformed += 1
            if malformed <= 3:
                print(f"    step {s}: obs={ok_obs} rew={ok_rew} act={ok_act} "
                      f"resolved={resolved} rewards={env._step_rewards}", flush=True)

    if not crashed:
        check(True, f"step_all ran {args.steps} steps without crashing")
        check(malformed == 0,
              f"obs/rewards/resolved well-formed every step "
              f"({malformed} malformed)")

    # World responsive? At least one agent should have physically moved.
    displacements = []
    for i in range(n):
        d = _dist(start_pos.get(i), env.get_agent_position(i))
        if d is not None:
            displacements.append(round(d, 2))
    check(any(d > 0.5 for d in displacements),
          f"≥1 agent physically moved under step_all "
          f"(displacements={displacements})")

    print("-" * 64, flush=True)
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} check(s)", flush=True)
        for m in _FAILS:
            print(f"  - {m}", flush=True)
        return 1
    print("ALL CHECKS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
