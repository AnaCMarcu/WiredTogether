#!/usr/bin/env python3
r"""make_counterfactual_story.py - cooperation arising under bonds; deferral under the plan.

Two figures in the make_final_figures WIDE grammar (the wide_gemma_hebbian3f
layout: mutual-message strip, mechanism lane with chamber bands, shaded focus
windows with corner connectors down to three bordered storyboard panels of
interleaved chat lines and frame pairs), built for the same seed so they
stack as (a)/(b) at one width. Panel k of (a) is the counterpart of panel k
of (b): same episode, and for k = 1, 3 the same milestone on the same map.

  k  episode  (a) central orchestrator                (b) Hebbian 2.0
  1  ep 1 Ch4 all three on attack/engage, 7 slots      a1 volunteers a0's flank,
              time out, "which zombie do you mean?"    spots first, kills t=715
  2  ep 3     a0+a2 assigned Anvil A: both say         a0 asks a2 by name, a2
     Ch2/Ch3  "centered and ready", neither digs,      presses, a0's door opens,
              slot times out                           joint fire, W(a0-a2) +0.15
  3  ep 3 Ch4 all three on clear_zombies, 4 slots,     a1 announces it sees a0
              "which zombie should I focus on?" x3    and converges; a0 kills t=663

Pair 2 is a mechanism contrast, not a matched milestone: no arm ever breaks
an anvil, so the orchestrator's co-assignment to an anvil is set against the
bonds' request-driven switch press in the same episode.

Lane (a) is the team_tenure_a grammar (one row per agent, task tenure on the
blue team-size ramp, events on the credited agent's row), spliced in by
monkeypatching make_final_figures.lane_coassign onto make_team_tenure.lane_orch
with sizes scaled from the 5.5 in FS_PRINT to this 13.6 in canvas. Lane (b)
is the RECORDED 50-step bond snapshots ("bonds_snap"): the per-step three-
factor replay omits the death-LTD term and runs 0.01-0.04 above the recorded
W here. Neither working script is edited.

Focus windows are the span of the frames + lines shown. The renderer pads
each shaded box by 75% of the window on either side, so a 50-step exchange
shades ~125 steps; the orchestrator windows are longer than the Hebbian ones
because its failures are 60-step assignment slots and the Hebbian successes
are 10-30-step exchanges.

Cells were picked by make_counterfactual_scan.py (seed 42 ep 1 M17 is its
top-ranked HEB_ONLY cell: 420 orchestrator agent-steps, 7 slot timeouts).

Usage:
  python analysis/make_counterfactual_story.py            # both panels
  python analysis/make_counterfactual_story.py a          # orchestrator only
Out:  paper_assets/timelines/counterfactual/counterfactual_{a,b}.{pdf,png,svg}
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paths import ASSETS, group  # noqa: E402  (also puts siblings on sys.path)

import make_final_figures as mff  # noqa: E402   (the WIDE layout)
import make_team_tenure as mtt  # noqa: E402    (the team_tenure_a lane)
from make_directive_timelines import load_run  # noqa: E402
from make_team_tenure import events  # noqa: E402

SEED = 42
ORCH_RUN = group("orchestrator") / "new_exp_0_gemma_orch_villager_advisory" / f"seed_{SEED}"
HEB_RUN = group("pareto_social_3f") / "new_exp_0_gemma_si3f8" / f"seed_{SEED}"
OUT = ASSETS / "timelines" / "counterfactual"

# Panel titles state what happened and why it succeeded or failed, so they
# run longer than the family's; one notch smaller keeps them on one line.
# Titles are 2-4 word behavioural noun phrases, so the family size fits.
# Panel k of (a) pairs with panel k of (b):
#   1  Coordinator-directed combat      <-> Joint task execution
#   2  Assigned joint subtask           <-> Partner help request
#   3  Task reassignment after failure  <-> Partner-specific support
# Pair 3 is the one enlarged by make_counterfactual_compact.py.

# Episode offsets (cumulative step of each episode's first step). Frame and
# message times inside the examples are WITHIN-episode, as the renderer
# expects; the focus window `x` is cumulative.
O_OFF = {1: 0, 2: 802, 3: 1610}
H_OFF = {1: 0, 2: 802, 3: 1604}


# ─── orchestrator lane in the team_tenure_a grammar ─────────────────────
def _tenure_lane(ax, arm, run, spans, xlim):
    """team_tenure_a's per-agent tenure rows, at this canvas's type sizes,
    plus the family's event key on the right (as lane_coassign has)."""
    fs = dict(mtt.FS_PRINT)
    fs.update(axis=mff.FS["axis"], tick=mff.FS["tick"], leg=8.5,
              chamber=mff.FS["chamber"], joint=58, star=160, line=1.3, bar=0.30)
    mtt.lane_orch(ax, run, spans, arm["joints"], arm["kills"], fs, xlim[1])
    team_key = ax.get_legend()
    if team_key is not None:
        ax.add_artist(team_key)
    ax.legend(handles=mff._event_handles(), fontsize=8, ncol=2, frameon=False,
              loc="upper right", borderaxespad=0.2, handletextpad=0.4)
    ax.tick_params(axis="y", labelsize=mff.FS["raster"] + 1)


mff.lane_coassign = _tenure_lane          # build_figure looks this name up


# ─── chamber delimiters: dotted rules instead of alternating fills ──────
def _chamber_rules(ax, spans):
    """A light dotted rule at each chamber start. The episode boundaries the
    renderer draws are dashed and darker, so the two stay distinguishable.

    Only spans that chamber_labels will actually label get a rule (longer
    than 40 steps, and 3% of the axis apart). Agents re-enter chambers
    constantly, so ruling every span start puts four dotted lines in a
    hundred steps with no label against any of them."""
    if not spans:
        return
    total = spans[-1][2] - spans[0][1]
    last = -1e9
    for _lab, lo, hi in spans:
        if hi - lo <= 40 or lo - last < total * 0.03 or lo <= 0:
            continue
        ax.axvline(lo, color="#b3bcc7", lw=0.7, ls=(0, (1.2, 2.4)), zorder=0)
        last = lo


# make_team_tenure bound the name at import, so both copies must be replaced
mff.shade_chambers = _chamber_rules
mtt.shade_chambers = _chamber_rules


# ─── merge adjacent same-chamber spans ─────────────────────────────────
# chamber_spans_from_steps absorbs a visit shorter than min_len into the
# PRECEDING span but keeps the preceding span's label, so Ch2 -> brief Ch1
# -> Ch2 emerges as two adjacent spans both labelled Ch2 and the lane prints
# "Ch2   Ch2". Join same-label neighbours, but never across an episode
# boundary: Ch1 at the end of one episode and Ch1 at the start of the next
# are different visits.
_spans_of = mff.chamber_spans_from_steps


def _merged_spans(arm, run, min_len=12):
    bounds = {b for b, _ in run["_ep_bounds"]}
    out = []
    for lab, lo, hi in _spans_of(arm, run, min_len):
        if out and out[-1][0] == lab and lo <= out[-1][2] and lo not in bounds:
            out[-1] = (lab, out[-1][1], hi)
        else:
            out.append((lab, lo, hi))
    return out


mff.chamber_spans_from_steps = _merged_spans


def _ex(key, ep, off, lo, hi, agents, title, frames):
    return dict(key=key, x=(off[ep] + lo, off[ep] + hi), ep=ep, agents=agents,
                dpairs=[], title=title, frames=frames, outcome="")


ARMS = {
    # ── (a) central orchestrator: mutual deferral ──────────────────────
    "counterfactual_orch_s42": dict(
        ph=0.50,
        run=ORCH_RUN, exp="new_exp_0_gemma_orch_villager_advisory", seed=SEED,
        lane="coassign",
        examples=[
            _ex("combat1", 1, O_OFF, 668, 715, (1, 0),
                "Coordinator-directed combat",
                [("M", 688, 1, 0, "I'm moving forward to join the fight, let me know if you need me to target a specific zombie."),
                 ("F", (1, 668, "a1 sees a0 beside a zombie"),
                  (0, 700, "a0: 'still dark, moving forward'")),
                 ("M", 694, 1, 0, "I'm turning right to try and spot the zombies Agent 0 is targeting. Can you confirm which zombie you mean?"),
                 ("M", 705, 0, 1, "I'm still moving forward into the dark area; keep pushing towards the fight."),
                 ("F", (1, 715, "a1 sees a0 in the lit room again"),
                  (0, 715, "a0 at the red door")),
                 ("M", 709, 1, 0, "I'm moving forward to locate the zombies. Keep pressure on the center one.")]),
            _ex("anvil3", 3, O_OFF, 198, 250, (0, 2),
                "Assigned joint subtask",
                [("M", 198, 0, 2, "Ready to dig Anvil A. Need you near me, agent_2."),
                 ("M", 200, 2, 0, "I am centered on Anvil A. Ready to dig with you now."),
                 ("F", (0, 200, "a0: still in the Ch1 doorway"),
                  (2, 220, "a2: the anvil at its shoulder")),
                 ("M", 205, 0, 2, "I'm centered and ready to dig Anvil A with you. Let's break this sword anvil!"),
                 ("M", 206, 0, 2, "Moving toward Anvil A now. Ready to dig when you are."),
                 ("F", (0, 250, "a0 reaches the anvil room as the slot ends"),
                  (2, 240, "a2: wall")),
                 ("M", 224, 0, 2, "I'm moving to meet you at Anvil A1. Ready to punch when you are.")]),
            _ex("combat3", 3, O_OFF, 612, 668, (0, 1),
                "Task reassignment after failure",
                [("M", 657, 0, 2, "I'm turning right to center the zombie for attack. Let's focus fire on this one."),
                 ("F", (0, 612, "a0: zombie at its shoulder as slot 1 starts"),
                  (1, 640, "a1: a0 and a2 beside a zombie at the red door")),
                 ("M", 664, 1, 0, "I am moving forward to join the zombie fight. Which zombie should I focus on?"),
                 ("M", 668, 0, 1, "I am moving forward to join the combat zone, let me know when you are ready to focus fire on a target."),
                 ("F", (0, 668, "a0: wall; slot 2 timed out at 660"),
                  (1, 668, "a1: wall")),
                 ("M", 669, 1, 0, "I'm moving forward to join the fight, let me know which zombie to focus on.")]),
        ],
        figures=[dict(name="counterfactual_a", xlim=None, keys=None, delib=False)],
        filmstrips=[]),

    # ── (b) Hebbian 2.0, same seed: cooperation arising ────────────────
    "counterfactual_heb2_s42": dict(
        ph=0.50,
        run=HEB_RUN, exp="new_exp_0_gemma_si3f8", seed=SEED,
        lane="bonds_snap",
        examples=[
            _ex("kill1", 1, H_OFF, 704, 730, (1, 0),
                "Joint task execution",
                [("M", 704, 1, 0, "I'll move forward now to cover the flanks for you."),
                 ("M", 706, 1, 0, "Turning left now to check the flank while you watch the right."),
                 ("F", (1, 708, "a1: zombie at the room entrance"),
                  (0, 724, "a0 arrives: a1 and two zombies in view")),
                 ("M", 707, 1, 0, "Zombies spotted up ahead. Engaging now."),
                 ("M", 710, 0, 1, "Still moving forward, watch flanks for zombies."),
                 ("F", (1, 715, "kill lands (t=715)"),
                  (0, 730, "a0 face to face with a zombie, a2 beside"))]),
            _ex("switch3", 3, H_OFF, 415, 425, (2, 0),
                "Partner help request",
                [("M", 415, 0, 2, "Agent 2, please confirm when the switch is pressed; I'm waiting for my door to open."),
                 ("M", 415, 2, 0, "I am aiming for the switch now; I will press it for you."),
                 ("F", (2, 418, "a2: a0 and a1 in view, red door, blue switch"),
                  (0, 418, "a0's door: red = locked")),
                 ("M", 420, 2, 0, "I see the switch. I will press it now to help open your door, agent_0."),
                 ("M", 423, 2, 0, "Door should be open now. Walking north."),
                 ("F", (2, 425, "a2 pressing the switch"),
                  (0, 425, "a0 through the open door, moving north")),
                 ("M", 425, 0, 1, "Door 0 is open; moving north now.")]),
            _ex("kill3", 3, H_OFF, 653, 664, (0, 1),
                "Partner-specific support",
                [("M", 653, 1, 0, "I see agent_0. I'll start targeting the zombies now."),
                 ("F", (0, 655, "a0: both teammates and the zombies in one room"),
                  (1, 657, "a1 sees a0 and a2, moves in")),
                 ("M", 657, 0, 1, "Zombies here. Targeting one now."),
                 ("M", 662, 1, 0, "I'm turning right to find a zombie to help Agent 0 clear the path."),
                 ("F", (0, 663, "kill lands (t=663), a1 and a2 beside"),
                  (1, 659, "a1 alongside a2 at the zombie")),
                 ("M", 664, 1, 0, "I'm turning right to move toward the combat area, let me know if you need backup on the zombies!")]),
        ],
        figures=[dict(name="counterfactual_b", xlim=None, keys=None, delib=False)],
        filmstrips=[]),
}


def main(which: str | None = None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    for name, arm in ARMS.items():
        tag = "a" if "orch" in name else "b"
        if which and which not in (tag, name):
            continue
        print(f"== {name} ==")
        for ex in arm["examples"]:                 # column purity, as in mff.main
            aL, aR = ex["agents"]
            for row in ex["frames"]:
                if row[0] != "F":
                    continue
                for slot, want in ((row[1], aL), (row[2], aR)):
                    if slot is not None and slot[0] != want:
                        raise AssertionError(f"{name}/{ex['key']}: frame a{slot[0]}@{slot[1]} "
                                             f"sits in the a{want} column")
        run = load_run(arm["run"])
        # joints / kills straight from the run's milestone events, so the lane
        # marks every cooperative fire of the run, not just the storied ones
        arm["joints"], arm["kills"] = events(run)
        messages = mff.load_messages(run)
        spans = mff.chamber_spans_from_steps(arm, run)
        Wr = mff.load_bond_series(arm)
        wanted = [(s[0], ex["ep"], s[1]) for ex in arm["examples"]
                  for r in ex["frames"] if r[0] == "F" for s in (r[1], r[2]) if s]
        frames = mff.grab_frames(arm, wanted)
        missing = len(set(wanted)) - len(frames)
        print(f"  frames {len(frames)}/{len(set(wanted))}"
              + (f"  [{missing} missing - pull this seed's gifs/ to fill]" if missing else "")
              + f", joints {arm['joints']}, kills {arm['kills']}")
        with plt.rc_context(mff.RC):
            for spec in arm["figures"]:
                mff.build_figure(name, arm, spec, run, Wr, messages, spans, frames, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
