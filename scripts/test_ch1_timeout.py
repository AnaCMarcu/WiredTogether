#!/usr/bin/env python3
"""Standalone unit test for the Python-side Ch1 timeout decision logic.

The decision tree lives in multi_agent_craftium.py around lines 1277-1303 - at
each step we check if step+1 >= --ch1-timeout-steps, count how many agents are
past Ch1, and branch:
    >= 2 advanced  ->  skipped (anvil-coop pair already in Ch2)
    1 advanced     ->  REGROUP (pull lone leader + stragglers to fallback spawns)
    0 advanced     ->  RESCUE  (classic all-agents teleport)

This script mirrors that exact code path and runs it against a controllable
``MockEnv`` so every branch can be verified without Luanti, an LLM, or a GPU.

Run:
    python scripts/test_ch1_timeout.py

Exit code 0 = every assertion passed, 1 = at least one failed (with a stack
trace pointing at the broken case).
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field


# --- Mock environment ------------------------------------------------------


@dataclass
class MockEnv:
    """Minimal stand-in for CraftiumEnvironmentInterface that exposes the two
    methods the decision tree touches: ``get_chamber`` and
    ``force_ch1_teleport``. The current per-agent chamber list is settable and
    every force-flag write is recorded so tests can assert it fired (or not).
    """
    chambers: list      # list[str | None] - index = agent_id, value = chamber name
    force_calls: int = field(default=0)
    raise_on_get_chamber: bool = False

    def get_chamber(self, agent_id: int):
        if self.raise_on_get_chamber:
            raise RuntimeError("simulated get_chamber failure")
        return self.chambers[agent_id]

    def force_ch1_teleport(self) -> bool:
        self.force_calls += 1
        return True


# --- The decision logic - copied verbatim from multi_agent_craftium.py -----
#
# This is intentionally byte-for-byte equivalent to lines 1277-1303 of
# multi_agent_craftium.py so we test the actual production semantics rather
# than a rewrite that might drift. The only change is the surrounding context:
# arguments come from the caller instead of locals captured by the outer loop.


def run_decision_step(
    *,
    episode: int,
    step: int,                       # 0-indexed episode step number
    ch1_timeout_steps: int,
    ch1_force_teleport_fired: bool,
    environment: MockEnv,
    num_agents: int,
) -> tuple[bool, str | None]:
    """Run one iteration of the per-step Ch1 timeout check.

    Returns the new value of ``ch1_force_teleport_fired`` plus the decision
    tag (``"skipped"`` / ``"RESCUE"`` / ``"REGROUP"`` / ``None``).
    """
    decision = None

    # Begin verbatim copy ---------------------------------------------------
    if (not ch1_force_teleport_fired
            and step + 1 >= ch1_timeout_steps):
        try:
            _trigger_chambers = [
                environment.get_chamber(_i) for _i in range(num_agents)
            ]
        except Exception:
            _trigger_chambers = []
        _n_advanced = sum(
            1 for _c in _trigger_chambers
            if _c is not None and _c != "ch1"
        )
        if _n_advanced >= 2:
            ch1_force_teleport_fired = True
            print(f"[CH1_TIMEOUT] skipped at ep={episode+1} "
                  f"step={step+1}: {_n_advanced} agents past Ch1 "
                  f"(chambers={_trigger_chambers})")
            decision = "skipped"
        elif environment.force_ch1_teleport():
            ch1_force_teleport_fired = True
            _verb = "RESCUE" if _n_advanced == 0 else "REGROUP"
            print(f"[CH1_TIMEOUT] {_verb} at ep={episode+1} "
                  f"step={step+1} "
                  f"(threshold={ch1_timeout_steps}, "
                  f"n_advanced={_n_advanced}, "
                  f"chambers={_trigger_chambers})")
            decision = _verb
    # End verbatim copy -----------------------------------------------------

    return ch1_force_teleport_fired, decision


# --- Test cases ------------------------------------------------------------


def t(name: str, condition: bool, detail: str = "") -> bool:
    """One-line assert. Prints PASS/FAIL with optional detail."""
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


def main() -> int:
    results: list[bool] = []
    # Each test case redirects the verbose [CH1_TIMEOUT] log lines into a
    # buffer so the test output stays readable. Buffer contents are still
    # included on FAIL for diagnostics.

    # 1. Pre-threshold: step counter hasn't reached the trigger yet.
    print("\n-- 1. Pre-threshold - no fire expected --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=10, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision is None", dec is None, repr(dec)))
    results.append(t("fired stays False", fired is False))
    results.append(t("force_ch1_teleport NOT called", env.force_calls == 0))

    # 2. At threshold, all in Ch1 -> RESCUE.
    print("\n-- 2. Threshold, all in Ch1 - RESCUE expected --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'RESCUE'", dec == "RESCUE", repr(dec)))
    results.append(t("fired becomes True", fired is True))
    results.append(t("force_ch1_teleport called once", env.force_calls == 1))
    if not all(results[-3:]):
        print(buf.getvalue())

    # 3. At threshold, 1 agent past Ch1 -> REGROUP.
    print("\n-- 3. Threshold, 1 in Ch2 + 2 in Ch1 - REGROUP expected --")
    env = MockEnv(chambers=["ch2", "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'REGROUP'", dec == "REGROUP", repr(dec)))
    results.append(t("fired becomes True", fired is True))
    results.append(t("force_ch1_teleport called once", env.force_calls == 1))

    # 4. At threshold, 2 agents past Ch1 -> skipped.
    print("\n-- 4. Threshold, 2 in Ch2 + 1 in Ch1 - skipped expected --")
    env = MockEnv(chambers=["ch2", "ch2", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'skipped'", dec == "skipped", repr(dec)))
    results.append(t("fired becomes True", fired is True))
    results.append(t("force_ch1_teleport NOT called", env.force_calls == 0,
                     "skip branch must not write the flag file"))

    # 5. At threshold, all 3 past Ch1 (different chambers) -> skipped.
    print("\n-- 5. Threshold, all 3 past Ch1 (ch2/ch3/ch2) - skipped --")
    env = MockEnv(chambers=["ch2", "ch3", "ch2"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'skipped'", dec == "skipped", repr(dec)))
    results.append(t("force_ch1_teleport NOT called", env.force_calls == 0))

    # 6. Already fired earlier this episode -> don't fire again.
    print("\n-- 6. Already fired this episode - no second fire --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2999, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=True,   # already fired earlier
            environment=env, num_agents=3,
        )
    results.append(t("decision is None", dec is None, repr(dec)))
    results.append(t("fired stays True", fired is True))
    results.append(t("force_ch1_teleport NOT called again", env.force_calls == 0))

    # 7. New episode (caller reset fired=False) -> ready to fire again.
    print("\n-- 7. New episode (fired reset) - ready to fire again --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=1, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False,   # caller reset at episode start
            environment=env, num_agents=3,
        )
    results.append(t("decision == 'RESCUE'", dec == "RESCUE", repr(dec)))
    results.append(t("force_ch1_teleport called", env.force_calls == 1))

    # 8. get_chamber returns None for an agent -> treated as Ch1 (0 advanced).
    print("\n-- 8. None-chamber edge case - counted as Ch1 --")
    env = MockEnv(chambers=[None, "ch1", "ch1"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'RESCUE'", dec == "RESCUE", repr(dec)))
    results.append(t("force_ch1_teleport called", env.force_calls == 1))

    # 9. get_chamber raises -> graceful fallback, 0 advanced -> RESCUE.
    print("\n-- 9. get_chamber raises - safe fallback to RESCUE --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"], raise_on_get_chamber=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=2499, ch1_timeout_steps=2500,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("decision == 'RESCUE'", dec == "RESCUE", repr(dec)))
    results.append(t("force_ch1_teleport called", env.force_calls == 1))

    # 10. exp0_long config: max_steps=4000, ch1_timeout_steps=2000 (50%).
    #     Verifies the threshold fires AT step 1999 (the moment step+1 == 2000).
    print("\n-- 10. exp0_long config (max=4000, threshold=2000) --")
    env = MockEnv(chambers=["ch1", "ch1", "ch1"])
    # one step before threshold - should not fire
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=1998, ch1_timeout_steps=2000,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("step=1998 (step+1=1999 < 2000) - no fire", dec is None))
    # exactly at threshold
    buf = io.StringIO()
    with redirect_stdout(buf):
        fired, dec = run_decision_step(
            episode=0, step=1999, ch1_timeout_steps=2000,
            ch1_force_teleport_fired=False, environment=env, num_agents=3,
        )
    results.append(t("step=1999 (step+1=2000 == 2000) - fires", dec == "RESCUE"))

    # -- Summary -----------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print(f"\n----------------------------------------")
    print(f"  {passed} / {total} assertions passed")
    print(f"----------------------------------------")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
