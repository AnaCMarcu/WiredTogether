"""Pure schedule logic for the per-chamber force-teleport timers.

Extracted from multi_agent_craftium.py so it can be unit-tested without the
runner's heavy imports (autogen, torch, craftium).
"""


def compute_chamber_schedule(max_steps, start_chamber=None, max_chamber=None):
    """Trigger steps for the per-chamber force-teleport timers.

    Returns ``{from_ch: step or None}`` for ``from_ch`` in (1, 2, 3, 4) —
    the env step at which the Ch{from_ch}→Ch{from_ch+1} force-teleport
    fires, or None when that timer is suppressed.

    Default (both overrides None): each chamber gets 20% of the episode,
    i.e. triggers at 20/40/60/80% of max_steps — the historical behavior.

    max_chamber=M (Phase A pair runs): timers with from_ch >= M are
    suppressed, so agents are never force-moved past chamber M (organic
    progression past it stays possible — doors are not locked by this).

    start_chamber=S (Phase B merged runs): timers with from_ch < S are
    suppressed (those chambers are skipped via the start teleport), and
    chambers S..5 split the episode evenly: with S=3 the Ch3→Ch4 timer
    fires at 33% and Ch4→Ch5 at 67%.
    """
    schedule = {n: max(1, int(max_steps * 0.2 * n)) for n in (1, 2, 3, 4)}
    if max_chamber is not None:
        for n in schedule:
            if n >= max_chamber:
                schedule[n] = None
    if start_chamber is not None:
        n_active = 5 - start_chamber + 1
        for n in schedule:
            if n < start_chamber:
                schedule[n] = None
            else:
                schedule[n] = max(
                    1, int(max_steps * (n - start_chamber + 1) / n_active)
                )
    return schedule
