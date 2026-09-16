"""Memory x bond (2x2) report: what actually carries partner preference?

The transplant experiment as first run varies ONE factor. Both of its arms
transplant memories AND bonds; Transplant-vs-Shuffled only changes whether the
memories are true. That leaves the obvious objection open: the agents remember
who they met, so why is the Hebbian graph needed at all? These cells close it
by crossing the two factors:

    cell   memory      bond        arm
    A      retained    retained    expB_merged_transplant
           fabricated  retained    expB_merged_shuffled   (A's truth control)
    B      retained    reset       expB_memory_only
    C      reset       retained    expB_bond_only
    D      reset       reset       expB_neither

"Bond reset" is the flat matrix from `merge_pair_runs.py uniform-w`, whose mean
off-diagonal equals the merged matrix's exactly: bond MASS is constant across
every cell and only its distribution changes. "Memory reset" is fresh agents,
with no --agent-state-init at all.

PRIMARY METRIC, fixed before the B/C/D runs existed: ring-conditioned partner
preference, P(target = seatmate | target is a ring neighbour), chance 0.50.
Raw seatmate preference is reported beside it as the secondary, because WIRE's
Chamber-3 switch ring and its index-ordered spawn rows make an agent's seatmate
a task partner, so the nominal 1/(N-1) = 0.20 chance line understates what a
team with no relationships scores. Cell D measures that floor directly, and
every contrast below is read against D rather than against 0.20.

Usage:
    PYTHONPATH=src python src/mindforge/tools/memory_bond_report.py
Writes paper_assets/transplant/MEMORY_BOND_REPORT.md.
"""

import argparse
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mindforge.tools.transplant_report import (  # noqa: E402
    phase_b_runs,
    phase_b_wiring,
)

# Display order and the factor levels each cell holds. The tag is what the
# contrasts below refer to; "A2" is cell A's fabricated-memory control (the
# published Shuffled arm), not a fifth cell of the factorial.
CELLS = [
    ("A", "transplant", "retained", "retained", "real partner, learned graph"),
    ("A2", "shuffled", "fabricated", "retained", "stranger, learned graph"),
    ("B", "memory_only", "retained", "reset", "real partner, flat graph"),
    ("C", "bond_only", "reset", "retained", "no memory, learned graph"),
    ("D", "neither", "reset", "reset", "no memory, flat graph (floor)"),
]

OUT = Path("paper_assets/transplant/MEMORY_BOND_REPORT.md")


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def _ms(values, nd=2):
    """mean ± sample sd; just the mean when n < 2."""
    vals = [v for v in values if v is not None]
    if not vals:
        return "—"
    if len(vals) < 2:
        return f"{vals[0]:.{nd}f}"
    return f"{st.mean(vals):.{nd}f} ± {st.stdev(vals):.{nd}f}"


def _arm_values(wiring, arm, key):
    """Per-seed values of `key` for one arm; seeds with no run are skipped."""
    runs = wiring.get(arm, {}).get("runs", {})
    return [d.get(key) for d in runs.values() if d.get(key) is not None]


def _mean_or_none(vals):
    return st.mean(vals) if vals else None


def _delta(a, b):
    return None if (a is None or b is None) else a - b


def build(args):
    base = Path(args.phaseb_base)
    seeds = args.seeds
    arms = [c[1] for c in CELLS]
    wiring = phase_b_wiring(base, arms, seeds)
    present = {arm: len(phase_b_runs(base, arm, seeds)) for arm in arms}

    L = []
    L.append("# Memory x Bond (2x2) — what carries partner preference?")
    L.append("")
    L.append(f"*Generated {datetime.now():%Y-%m-%d %H:%M} by "
             f"`src/mindforge/tools/memory_bond_report.py` — every number is "
             f"computed from the run artifacts; regenerate rather than "
             f"hand-edit.*")
    L.append("")
    L.append("## Design")
    L.append("")
    L.append("| cell | arm | memory | bond | seating | runs found |")
    L.append("|---|---|---|---|---|---|")
    for tag, arm, mem, bond, desc in CELLS:
        L.append(f"| {tag} | `{arm}` | {mem} | {bond} | {desc} | "
                 f"{present[arm]}/{len(seeds)} |")
    L.append("")
    L.append("Bond mass is held constant across every cell: the flat matrix "
             "has the same mean off-diagonal as the merged one (0.133, versus "
             "0.265 within-seat and 0.100 cross-seat), so a difference cannot "
             "come from one arm simply starting with stronger bonds. "
             "Everything else is pinned to cell A: 6 agents, "
             "`--start-chamber 3`, 3 episodes x 1000 steps, the "
             "`reward_modulated` rule at the exp08 settings (the "
             "`--hebbian-mode` default, and what cell A recorded), "
             "`--social-module prompt`.")
    L.append("")

    missing = [t for t, a, *_ in CELLS if present[a] == 0]
    if missing:
        L.append(f"> **Incomplete:** no runs yet for cell(s) "
                 f"{', '.join(missing)}. Their rows stay blank and any "
                 f"contrast that needs them is not computed.")
        L.append("")

    # ── Partner preference ───────────────────────────────────────────────
    L.append("## Partner preference")
    L.append("")
    L.append("The ring-conditioned column is the primary metric (chance "
             "0.50). The raw column (nominal chance 0.20) is inflated in "
             "every arm by the Chamber-3 switch ring and the index-ordered "
             "spawn rows, which is the thing cell D quantifies. `interior` "
             "drops the two end seats, whose two ring neighbours are not "
             "spatially symmetric.")
    L.append("")
    L.append("| cell | arm | ring-cond. (primary) | interior only | raw | "
             "vs D (ring) |")
    L.append("|---|---|---|---|---|---|")
    floor = _mean_or_none(_arm_values(wiring, "neither", "ring_pref_mean"))
    for tag, arm, *_ in CELLS:
        ring = _arm_values(wiring, arm, "ring_pref_mean")
        inter = _arm_values(wiring, arm, "ring_pref_interior")
        raw = _arm_values(wiring, arm, "pref_mean")
        d = _delta(_mean_or_none(ring), floor)
        L.append(f"| {tag} | `{arm}` | {_ms(ring)} | {_ms(inter)} | "
                 f"{_ms(raw)} | {_f(d)} |")
    L.append("")
    L.append(f"(mean ± sample sd over up to {len(seeds)} seeds.)")
    L.append("")

    # ── The contrasts the design exists to produce ───────────────────────
    L.append("## Contrasts")
    L.append("")
    m = {tag: _mean_or_none(_arm_values(wiring, arm, "ring_pref_mean"))
         for tag, arm, *_ in CELLS}
    contrasts = [
        ("A − B", _delta(m["A"], m["B"]),
         "does the relational graph add anything beyond episodic memory?"),
        ("A − C", _delta(m["A"], m["C"]),
         "does episodic memory add anything beyond the graph?"),
        ("C − D", _delta(m["C"], m["D"]),
         "does the graph alone carry partner preference?"),
        ("B − D", _delta(m["B"], m["D"]),
         "does memory alone carry partner preference?"),
        ("A − A2", _delta(m["A"], m["A2"]),
         "the published effect: does the memory have to be TRUE?"),
    ]
    L.append("| contrast | Δ ring-cond. | question |")
    L.append("|---|---|---|")
    for name, val, q in contrasts:
        L.append(f"| {name} | {_f(val)} | {q} |")
    L.append("")
    inter_term = None
    if None not in (m["A"], m["B"], m["C"], m["D"]):
        inter_term = (m["A"] - m["B"]) - (m["C"] - m["D"])
    L.append(f"**Interaction (A−B)−(C−D): {_f(inter_term)}.** Positive means "
             "memory and structure are worth more together than the sum of "
             "their separate contributions, which is social plasticity as a "
             "coupling rather than as externalised social memory. Near zero "
             "means the two factors are additive and separable.")
    L.append("")

    # ── Per-episode trend ────────────────────────────────────────────────
    L.append("## Per-episode trend (ring-conditioned)")
    L.append("")
    L.append("W is a fast variable: within-pair weights decay toward the "
             "rule's fixed point over an episode, so a structure-only effect "
             "should be strongest in episode 1 and fade after. Read cell C "
             "down this table before concluding the graph carries nothing.")
    L.append("")
    L.append("| cell | arm | ep1 | ep2 | ep3 |")
    L.append("|---|---|---|---|---|")
    for tag, arm, *_ in CELLS:
        by_ep = {}
        for d in wiring.get(arm, {}).get("runs", {}).values():
            for ep, v in (d.get("ep_ring_trend") or {}).items():
                by_ep.setdefault(ep, []).append(v)
        cols = [_ms([v for v in by_ep[ep] if v is not None])
                for ep in sorted(by_ep)][:3]
        cols += ["—"] * (3 - len(cols))
        L.append(f"| {tag} | `{arm}` | " + " | ".join(cols) + " |")
    L.append("")

    # ── Bond evolution ───────────────────────────────────────────────────
    L.append("## Bond evolution (mean within-seat W / mean cross-seat W)")
    L.append("")
    L.append("Cells A and C start at 0.265 within-seat and 0.100 cross-seat; "
             "cells B and D start flat at 0.133. The two columns converging "
             "is the graph erasing its own initialisation.")
    L.append("")
    L.append("| cell | arm | ep1 | ep2 | ep3 |")
    L.append("|---|---|---|---|---|")
    for tag, arm, *_ in CELLS:
        by_ep = {}
        for d in wiring.get(arm, {}).get("runs", {}).values():
            for ep, pair in (d.get("w_by_ep") or {}).items():
                by_ep.setdefault(ep, []).append(pair)
        cols = []
        for ep in sorted(by_ep)[:3]:
            ins = [a for a, _ in by_ep[ep]]
            crs = [b for _, b in by_ep[ep]]
            cols.append(f"{_f(_mean_or_none(ins), 3)} / "
                        f"{_f(_mean_or_none(crs), 3)}")
        cols += ["—"] * (3 - len(cols))
        L.append(f"| {tag} | `{arm}` | " + " | ".join(cols) + " |")
    L.append("")

    L.append("## Caveats")
    L.append("")
    L.append("- Cells C and D use fresh agents, so they also lack the task "
             "competence the transplanted agents carry. That does not "
             "confound partner preference, which is a within-run choice among "
             "six equally competent teammates, but it does confound any "
             "cross-cell comparison of task performance.")
    L.append("- Seat-pair labels (GENUINE / CONTROL) are meaningful only in "
             "cells A and B. After block-mean normalisation every dyad in C "
             "and D is identical by construction, so those runs are read "
             "unlabelled.")
    L.append("- 3 seeds per cell, one model (Gemma 4 E4B).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phaseb-base",
                    default="runs_from_daic/rq3_topology_transfer/pair_bonding")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    report = build(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
