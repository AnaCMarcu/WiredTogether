"""compute_chamber_schedule: the per-chamber force-teleport timer schedule.

The default must reproduce the historical inline computation
(20/40/60/80% of max_steps) exactly; --max-chamber and --start-chamber
suppress/redistribute timers for the pair-bonding transplant experiment.
"""

import pytest

from mindforge.chamber_schedule import compute_chamber_schedule


@pytest.mark.parametrize("max_steps", [150, 1000, 1500, 2000])
def test_default_matches_historical_20pct_schedule(max_steps):
    schedule = compute_chamber_schedule(max_steps)
    expected = {n: max(1, int(max_steps * 0.2 * n)) for n in (1, 2, 3, 4)}
    assert schedule == expected


def test_default_1000_exact_values():
    assert compute_chamber_schedule(1000) == {1: 200, 2: 400, 3: 600, 4: 800}


def test_max_chamber_2_keeps_only_ch1_timer():
    schedule = compute_chamber_schedule(1000, max_chamber=2)
    assert schedule == {1: 200, 2: None, 3: None, 4: None}


def test_max_chamber_1_suppresses_everything():
    schedule = compute_chamber_schedule(1000, max_chamber=1)
    assert schedule == {1: None, 2: None, 3: None, 4: None}


def test_start_chamber_3_splits_remaining_evenly():
    schedule = compute_chamber_schedule(1000, start_chamber=3)
    # Chambers 3/4/5 split the episode: Ch3→Ch4 at 33%, Ch4→Ch5 at 67%.
    assert schedule == {1: None, 2: None, 3: 333, 4: 666}


def test_start_chamber_5_has_no_timers():
    schedule = compute_chamber_schedule(1000, start_chamber=5)
    assert schedule == {1: None, 2: None, 3: None, 4: None}


def test_start_chamber_2_gives_quarter_splits():
    schedule = compute_chamber_schedule(1000, start_chamber=2)
    assert schedule == {1: None, 2: 250, 3: 500, 4: 750}


def test_triggers_never_below_1():
    schedule = compute_chamber_schedule(1)
    assert all(s >= 1 for s in schedule.values())
    schedule = compute_chamber_schedule(2, start_chamber=3)
    assert schedule[3] >= 1 and schedule[4] >= 1
