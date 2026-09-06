"""Wall-clock -> (episode, step) table extracted from log.txt step markers.

log.txt is denylisted for wholesale reading (60-150 MB), but a single
streaming regex pass over it is cheap and gives the ONLY reliable clock for
runs whose llm_log unit streams cannot be text-anchored to step_log (RL runs:
driver-level retries produce extra rl_thoughts units, and the routed message
text does not equal the unit's communication field).

Marker line format (multi_agent_craftium step loop):
    [exp03_mappo/seed_42] 2026-06-26 09:59:40 INFO ep=1 step=1/1000 global_step=1

Real-world messiness this must survive (all observed in exp03/seed_42):
- relaunched jobs APPEND to the same log.txt -> multiple attempts;
- a CONCURRENT job with a different max_steps config (step=N/2500) can
  interleave lines with the medium run's (step=N/1000) in the same file.

Strategy: bucket markers by their step DENOMINATOR, split each bucket into
restart segments (sequence going backward = new attempt), then pick the
segment whose (ep, t) set best covers the run's actual step rows.
``step`` is 1-based at loop start; step_log.csv t is 0-based -> t = step - 1.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict

_MARKER = re.compile(
    r"^\[[^\]]*\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) INFO "
    r"ep=(?P<ep>\d+) step=(?P<step>\d+)/(?P<den>\d+)"
)

MIN_SEGMENT = 30  # markers; drop noise slivers


def read_step_segments(run) -> list:
    """All candidate marker segments: [{den, markers: [(ts, ep, t), ...]}]."""
    path = run.path / "log.txt"
    if not path.exists():
        return []
    by_den = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _MARKER.match(line)
            if not m:
                continue
            by_den[int(m.group("den"))].append(
                (m.group("ts"), int(m.group("ep")), int(m.group("step")))
            )
    segments = []
    for den, marks in by_den.items():
        cur = []
        last = (0, 0)
        for ts, ep, step in marks:
            if (ep, step) <= last and cur:
                if len(cur) >= MIN_SEGMENT:
                    segments.append({"den": den, "markers": cur})
                cur = []
            cur.append((ts, ep, step - 1))
            last = (ep, step)
        if len(cur) >= MIN_SEGMENT:
            segments.append({"den": den, "markers": cur})
    return segments


def _assign(units, markers) -> None:
    keys = [m[0] for m in markers]
    for u in units:
        i = bisect_right(keys, u["ts"]) - 1
        u["clock"] = list(markers[i][1:]) if i >= 0 else None


def _coverage(per_agent_units, per_agent_rows) -> float:
    """Mean over agents of |clocked (ep,t) ∩ row (ep,t)| / |rows|."""
    covs = []
    for a, rows in per_agent_rows.items():
        if not rows:
            continue
        row_keys = {(ep, t) for ep, t, _r in rows}
        clocked = {tuple(u["clock"]) for u in per_agent_units.get(a, [])
                   if u.get("clock")}
        covs.append(len(clocked & row_keys) / len(row_keys))
    return sum(covs) / len(covs) if covs else 0.0


def stamp_best_clock(run, units, per_agent_units, per_agent_rows):
    """Assign u["clock"] from the best-covering marker segment.

    Evaluates every candidate segment by action-unit coverage of the actual
    step rows; stamps ALL units with the winner. Returns {"segments": n,
    "chosen": {den, span, n_markers} | None, "coverage": float}.
    """
    segments = read_step_segments(run)
    if not segments:
        for u in units:
            u["clock"] = None
        return {"segments": 0, "chosen": None, "coverage": 0.0}
    action_units = [u for us in per_agent_units.values() for u in us]
    best, best_cov = None, -1.0
    for seg in segments:
        _assign(action_units, seg["markers"])
        cov = _coverage(per_agent_units, per_agent_rows)
        if cov > best_cov:
            best, best_cov = seg, cov
    _assign(units, best["markers"])
    return {
        "segments": len(segments),
        "chosen": {
            "den": best["den"],
            "span": [best["markers"][0][0], best["markers"][-1][0]],
            "n_markers": len(best["markers"]),
        },
        "coverage": round(best_cov, 4),
    }
