#!/usr/bin/env python3
"""Parity harness for CraftiumEnvironmentInterface.step_all (simultaneous-move)
vs .step (turn-based).

WHY THIS EXISTS
---------------
step_all() advances the underlying env ONCE for all agents; step() advances it
once *per agent* (acting agent moves, the rest NoOp). At N>1 the two are
deliberately different dynamics, so exact reward parity is impossible — that is
the whole point of the change. What we CAN prove, and what licenses migrating
step() onto the shared _resolve_action_for_agent helper later, is:

  Tier 1  N=1 EXACT IDENTITY    with one agent the two modes are the same
                                physical process, so step and step_all must
                                produce byte-identical observations + rewards
                                over a fixed action sequence.
  Tier 2  step_all MECHANICS    correct reward accumulation, NoOp-fill of
                                finished agents, and the N×->1 underlying-tick
                                reduction that motivates the change.
  Tier 3  TICK-EXPANSION        Dig sustains for its tick budget; Jump expands
                                to [Jump, MoveForward].

It runs WITHOUT Luanti / a GPU by injecting a deterministic MockUnderlyingEnv
in place of marl_craftium.OpenWorldMultiAgentEnv (same trick as
scripts/test_ch1_timeout.py's MockEnv).

Run:
    python scripts/parity_test_stepping.py
Exit 0 = all tiers passed, 1 = at least one assertion failed.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

# ── Locate src/mindforge and stub marl_craftium BEFORE importing the wrapper ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_MINDFORGE = os.path.join(_HERE, "..", "src", "mindforge")
sys.path.insert(0, os.path.abspath(_MINDFORGE))


class MockUnderlyingEnv:
    """Deterministic stand-in for OpenWorldMultiAgentEnv (PettingZoo parallel).

    reward[agent] == the integer action id it was sent on that tick, so every
    test can audit *exactly* which primitives reached the env and how many
    ticks fired. Observations encode the running tick count so trajectory
    divergence is visible. No terminations unless ``kill_on_tick`` is armed.
    """

    def __init__(self, num_agents=3, **_kw):
        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = list(self.possible_agents)
        self.tick = 0
        self.step_calls = 0          # number of underlying env.step() invocations
        self.action_log = []         # one dict per underlying tick
        self.kill_on_tick = None     # (agent_name, tick) -> drop that agent
        # Stuck-detection reads self.env.env._positions[i]; provide a static one.
        self.env = types.SimpleNamespace(
            _positions={i: [0.0, 64.0, 0.0] for i in range(num_agents)}
        )

    def _obs(self):
        return {a: np.full((4, 4, 3), self.tick % 256, dtype=np.uint8)
                for a in self.agents}

    def reset(self):
        self.tick = 0
        self.step_calls = 0
        self.action_log = []
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        self.tick += 1
        self.step_calls += 1
        self.action_log.append(dict(actions))
        rewards = {a: float(actions.get(a, 0)) for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        if self.kill_on_tick and self.kill_on_tick[1] == self.tick:
            dead = self.kill_on_tick[0]
            terminations[dead] = True
            if dead in self.agents:
                self.agents.remove(dead)
        obs = self._obs()
        return obs, rewards, terminations, truncations, {a: {} for a in self.agents}


# Inject the stub package so `from marl_craftium import OpenWorldMultiAgentEnv`
# resolves to our mock instead of running the real bootstrap.
_fake = types.ModuleType("marl_craftium")
_fake.OpenWorldMultiAgentEnv = MockUnderlyingEnv
sys.modules["marl_craftium"] = _fake

from custom_environment_craftium import (  # noqa: E402
    ACTION_MAP,
    CraftiumEnvironmentInterface,
)


# ── Test helpers ──────────────────────────────────────────────────────────

def _make(num_agents):
    """Build an interface backed by a fresh MockUnderlyingEnv, with the
    file-touching helpers stubbed so it runs offline."""
    env = CraftiumEnvironmentInterface(num_agents=num_agents)
    env.reset_log_offset = lambda *a, **k: None
    env.tail_server_log = lambda *a, **k: None
    env._find_best_tool = lambda agentId: None  # no auto-equip pre-tick
    env.reset()
    return env


_FAILS = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        _FAILS.append(msg)


# ── Tier 1: N=1 exact identity ─────────────────────────────────────────────

def tier1_identity():
    print("Tier 1 — N=1 exact identity (step vs step_all)")
    # Single-tick + Dig actions only. Jump is excluded here: step() drops the
    # vault (jump) tick's reward by initialising total_rewards AFTER it, whereas
    # step_all counts every scheduled tick. That divergence is intentional
    # (step_all is the more correct accounting) and is exercised in Tier 3.
    sequence = ["MoveForward", "TurnRight", "NoOp", "Dig", "MoveBackward", "LookDown"]

    turn = _make(1)
    sim = _make(1)
    for a in sequence:
        _, ta = turn.step(a, 0)
        _, sres = sim.step_all({0: a})
        sa = sres[0]
        check(ta == sa, f"resolved action match on {a!r}: step={ta!r} step_all={sa!r}")
        check(turn._step_rewards == sim._step_rewards,
              f"_step_rewards match on {a!r}: {turn._step_rewards} == {sim._step_rewards}")
        check(np.array_equal(turn.get_agent_frame(0), sim.get_agent_frame(0)),
              f"observation match on {a!r}")
    check(turn.env.step_calls == sim.env.step_calls,
          f"identical underlying tick count at N=1 "
          f"(step={turn.env.step_calls}, step_all={sim.env.step_calls})")


# ── Tier 2: step_all mechanics at N=3 ──────────────────────────────────────

def tier2_mechanics():
    print("Tier 2 — step_all mechanics at N=3")
    sim = _make(3)
    base = sim.env.step_calls
    _, resolved = sim.step_all({0: "MoveForward", 1: "TurnRight", 2: "NoOp"})
    check(sim.env.step_calls - base == 1,
          f"all 3 agents share ONE underlying tick (delta=={sim.env.step_calls - base})")
    # Mock contract: reward[ag] == the action id actually sent, which is the
    # RESOLVED action (after macro/idle/pitch guards), not the raw input. NoOp
    # is deliberately included so the test confirms the idle-guard fires inside
    # step_all exactly as it does in step() (NoOp -> MoveForward on the 1st idle).
    for aid in range(3):
        sent = resolved[aid]
        check(sim._step_rewards[f"agent_{aid}"] == ACTION_MAP[sent],
              f"agent_{aid} reward == id of its resolved action {sent!r}")

    # The motivating contrast: turn-based costs N ticks for the same outer step.
    turn = _make(3)
    base = turn.env.step_calls
    for aid, a in {0: "MoveForward", 1: "TurnRight", 2: "NoOp"}.items():
        turn.step(a, aid)
    check(turn.env.step_calls - base == 3,
          f"turn-based costs N=3 underlying ticks for the same outer step "
          f"(delta=={turn.env.step_calls - base}) -> step_all is the 3x reduction")


# ── Tier 3: tick-expansion (Dig sustain, Jump pairing, NoOp-fill) ──────────

def tier3_expansion():
    print("Tier 3 — Dig sustain / Jump pairing / NoOp-fill")
    # Dig sustains for _SUSTAINED_TICKS["Dig"] ticks; a co-acting MoveForward
    # agent NoOp-fills the remaining ticks.
    sim = _make(2)
    base = sim.env.step_calls
    sim.step_all({0: "Dig", 1: "MoveForward"})
    dig_ticks = sim._SUSTAINED_TICKS["Dig"]
    check(sim.env.step_calls - base == dig_ticks,
          f"Dig drives {dig_ticks} underlying ticks (delta=={sim.env.step_calls - base})")
    # agent_0 dug every tick; agent_1 moved on tick 0 then NoOp-filled.
    log = sim.env.action_log[-dig_ticks:]
    check(all(d["agent_0"] == ACTION_MAP["Dig"] for d in log),
          "agent_0 sent Dig on every sustained tick")
    check(d := (log[0]["agent_1"] == ACTION_MAP["MoveForward"]),
          "agent_1 moved on the first tick")
    check(all(d["agent_1"] == ACTION_MAP["NoOp"] for d in log[1:]),
          "agent_1 NoOp-filled the remaining Dig ticks")

    # Jump expands to [Jump, MoveForward] = exactly 2 ticks.
    sim2 = _make(1)
    base = sim2.env.step_calls
    sim2.step_all({0: "Jump"})
    check(sim2.env.step_calls - base == 2,
          f"Jump expands to 2 ticks (delta=={sim2.env.step_calls - base})")
    j = sim2.env.action_log[-2:]
    check(j[0]["agent_0"] == ACTION_MAP["Jump"]
          and j[1]["agent_0"] == ACTION_MAP["MoveForward"],
          "Jump tick then MoveForward tick (vault pairing)")

    # NoOp-fill must also cover a dying agent without KeyError.
    sim3 = _make(2)
    sim3.env.kill_on_tick = ("agent_1", 3)  # agent_1 dies mid-Dig
    sim3.step_all({0: "Dig", 1: "MoveForward"})
    check("agent_0" in sim3._step_rewards,
          "surviving agent still accounted after a teammate dies mid-step")


def main():
    print("=" * 64)
    print("step_all parity harness (mock underlying env — no Luanti/GPU)")
    print("=" * 64)
    tier1_identity()
    tier2_mechanics()
    tier3_expansion()
    print("-" * 64)
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} assertion(s)")
        for m in _FAILS:
            print(f"  - {m}")
        return 1
    print("ALL TIERS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
