"""Final ICLR qualitative figures, all arms.

Per arm: overview figure (global timeline + curated example panels),
optional zoom-horizon figures (restricted xlim, fewer/larger panels), and
filmstrip panels (dense frame sequences of one agent's view). Outputs to
paper_assets/timelines/<arm>/final/ as png(300dpi)+pdf+svg.

Design: Times serif, mutual-message strip, chamber bands (no lines),
events on the graph (circle=joint fire, star=kill), full-height zoom bands
with corner connectors, panels = two-column frame storyboard with
Minecraft-style chat lines, directed bond sparklines where bonds exist.
All content verbatim from run logs; gemma bonds from the validated replay.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)
from make_directive_timelines import load_run, load_orch  # noqa: E402
from replay_hebbian_terms import load_inputs  # noqa: E402
from prototype_three_factor_rule import replay_three_factor  # noqa: E402

# ── design constants ────────────────────────────────────────────────────
INK, MUTED = "#31373f", "#8b95a1"
AGENT_C = {0: "#0072B2", 1: "#D55E00", 2: "#009E73"}
PAIR_C = {(0, 1): "#5B4EA4", (0, 2): "#B2652C", (1, 2): "#2E8B6E"}
CHAMBER_TINT = ["#f2f5f8", "#ffffff"]
# ICLR is a single-column 5.5 inch format.  Build at final print size so
# typography is not silently shrunk by LaTeX (the old 13.6 in figure made
# 11 pt labels render at roughly 4.5 pt when included at \textwidth).
FIG_W = 5.50
FS = dict(axis=7.8, tick=6.5, panel_title=8.4, caption=6.2, body=6.35,
          header=6.8, raster=6.0, chamber=6.0, chat=5.7, badge=5.7)
PANEL_EDGE = "#c3c9d0"
SHOW_BOND_EFFECT = False   # bond deltas live in the caption, not the panel
DPI = 400
MAXLEN = 66
RC = {"font.family": "serif",
      # TeX Gyre Termes is a metrically compatible Times face and tends to
      # render more consistently than platform-specific Times New Roman.
      "font.serif": ["TeX Gyre Termes", "Times New Roman", "STIXGeneral",
                     "DejaVu Serif"],
      "font.size": 7.0,
      "axes.edgecolor": "#9aa3ad",
      "axes.linewidth": 0.55,
      "xtick.major.width": 0.45, "ytick.major.width": 0.45,
      "xtick.major.size": 2.2, "ytick.major.size": 2.2,
      "svg.fonttype": "none", "pdf.fonttype": 42,
      "mathtext.fontset": "stix"}

ROOT = ASSETS / "timelines"


def W_(t):  # wrap helper placeholder (captions are pre-fit)
    return t


# ── arm configurations ──────────────────────────────────────────────────
GEMMA_DELIB = {
    "anvil": [(1, 254, "Both agents have communicated their movement towards the anvils, suggesting shared task progress. Since no direct request for help was made, I will focus on my own movement [...]"),
              (2, 257, "Agent 1 is actively moving and communicating — I should stay focused on my own movement [...] rather than interrupting the established coordination.")],
    "kill": [(1, 692, "I will offer assistance if they indicate a need [...] their request is informational rather than a direct ask for help."),
             (0, 700, "Agent_1 has communicated that they are starting combat against a zombie [...] since they haven't explicitly asked for help, I will focus on my own task for now.")],
    "switch": [(0, 2497, "Agent 1 [...] has requested me to 'Keep an eye on the door state' — I should respond to offer help / acknowledge the request to maintain the alliance.")],
}

ARMS = {
    "gemma3f_seed123": dict(
        run=RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f/seed_123",
        exp="new_exp_0_gemma_hebbian3f", seed=123, lane="bonds_replay",
        joints=[(442, (0, 1)), (1407, (0, 1)), (2401, (0, 1))],
        kills=[(722, 1), (1683, 2), (2642, 2)],
        examples=[
            dict(key="anvil", x=(2099, 2125), ep=3, agents=(1, 0),
                 dpairs=[], title="Failure in anvil dig",
                 frames=[("F", (1, 99, ""), (0, 101, "")),
                         ("M", 100, 1, 0, "I'm centered on the left anvil; ready to punch with Agent 0."),
                         ("M", 101, 0, 1, "I am moving forward to meet you at the anvils. Ready to punch the left one when you are."),
                         ("F", (1, 101, ""), (0, 111, "")),
                         ("M", 109, 0, 1, "I'm moving toward the center now to help break the anvils. Let me know when you're ready to coordinate."),
                         ("M", 113, 1, 0, "I'm moving toward the center now, ready to dig the left anvil with you."),
                         ("F", (1, 115, ""), (0, 123, "")),
                         ("M", 124, 0, 1, "I am moving toward the center now. Ready to coordinate on an anvil when I get closer."),
                         ],
                 outcome="both stay permanently ready and permanently approaching; "
                         "the anvil is never struck"),
            dict(key="kill", x=(710, 730), ep=1, agents=(1, 0),
                 dpairs=[(1, 0)], title="Cooperative zombie kill",
                 frames=[("M", 710, 1, 0, "Zombie here, starting combat."),
                         ("M", 714, 0, 1, "Turning right to reorient towards the combat zone."),
                         ("F", (1, 716, ""), (0, 712, "")),
                         ("M", 717, 1, 0, "Zombie here, focusing fire now."),
                         ("M", 718, 0, 1, "Orientation should be right now; ready to engage zombies."),
                         ("F", (1, 719, ""), (0, 720, "")),
                         ("M", 720, 1, 0, "Boss is centered; I am attacking it now."),
                         ("F", (1, 723, ""), (0, 728, "")),
                         ("M", 728, 0, 1, "I'm turning right to try and get oriented towards the fight, let me know if you see the zombies from your side."),
                         ],
                 outcome="a0 keeps reorienting toward the fight and never arrives; "
                         "a1 kills alone"),
            dict(key="switch", x=(426, 450), ep=1, agents=(0, 1),
                 dpairs=[(0, 1)], title="Switch puzzle solved",
                 frames=[("M", 439, 0, 1, "Still stuck in cell. Check door state if you see any change."),
                         ("M", 440, 1, 0, "I am turning to find the south wall switch, let me know if you see any door changes."),
                         ("F", (0, 442, ""), (1, 428, "")),
                         ("M", 442, 0, 1, "Blue switch is centered. I will press it now."),
                         ("M", 443, 0, 1, "I am digging the blue cube now; check if your door state changed."),
                         ("F", (0, 443, ""), (1, 441, "")),
                         ("M", 443, 1, 0, "Door is open, moving north now"),
                         ("M", 447, 1, 0, "Door is open, moving out now."),
                         ],
                 outcome="a0's press opens a1's door; a0 itself stays locked in, "
                         "still waiting for a press of its own"),
        ],
        figures=[
            dict(name="overview", xlim=None, keys=None, delib=False),
            dict(name="overview_with_deliberation", xlim=None, keys=None,
                 delib=True),
            dict(name="zoom_ep1", xlim=(400, 840), keys=["switch", "kill"],
                 delib=False),
            dict(name="zoom_ep3", xlim=(2060, 2200), keys=["anvil"],
                 delib=False),
        ],
        filmstrips=[
            dict(name="filmstrip_zombie_kill", agent=1, ep=1,
                 t0=710, t1=725,
                 title="Zombie kill, agent 1's view (episode 1, steps 710–725)"),
            dict(name="filmstrip_switch_chain", agent=0, ep=3,
                 t0=399, t1=414,
                 title="Switch chain, agent 0's view (episode 3, steps 399–414)"),
        ]),
    "orch789": dict(
        ph=0.50,
        run=RUNS / "orchestrator/new_exp_0_gemma_orch_villager_advisory/seed_789",
        exp="new_exp_0_gemma_orch_villager_advisory", seed=789,
        lane="coassign",
        joints=[(1230, (0, 1)), (1387, (0, 2)), (2122, (0, 1)),
                (2213, (0, 2))],
        kills=[(690, 2), (1511, 1), (2435, 1)],
        examples=[
            dict(key="kill", x=(683, 696), ep=1, agents=(2, 1), dpairs=[],
                 title="Cooperative zombie kill",
                 frames=[("M", 688, 2, 0, "Zombie centered, ready to attack now."),
                         ("F", (2, 689, "a2 engages; both teammates in view"),
                          (1, 690, "a1 arrives at the fight")),
                         ("M", 689, 1, 0, "Zombie spotted right, turning to engage."),
                         ("F", (2, 690, "two zombies, tags visible"),
                          (1, 691, "a1: zombie beside agent2")),
                         ("M", 692, 1, 0, "Zombie centered, attacking now. Focus fire!"),
                         ("F", (2, 694, ""), (1, 693, "")),
                         ],
                 outcome="a2 kills with a1 alongside — coordination routed via a0"),
            dict(key="switch", x=(2119, 2132), ep=3, agents=(0, 1),
                 dpairs=[],
                 title="Switch opened on request",
                 frames=[("M", 402, 1, 0, "Door is still locked, waiting for you to press switch."),
                         ("F", (0, 404, "a0 approaches the blue switch"),
                          (1, 405, "a1 waits at its door")),
                         ("M", 403, 0, 1, "I see the blue switch now. I will attempt to dig it to open the door."),
                         ("M", 405, 0, 1, "Switch is centered, I am pressing it now to try and open my door."),
                         ("F", (0, 406, "a0 presses the switch"),
                          (1, 409, "a1's door opens; a1 exits")),
                         ("M", 407, 1, 0, "Door is open, I am moving north now to meet you."),
                         ],
                 outcome="explicit wait-and-press exchange; the door opens"),
            dict(key="solokill", x=(2423, 2440), ep=3, agents=(1, 0),
                 dpairs=[],
                 title="Failure: focus-fire talk, solo kill",
                 frames=[("M", 706, 1, 0, "I see two zombies centered. Equipping sword now to start attacking."),
                         ("F", (1, 706, "a1 alone with two zombies"),
                          (0, 708, "a0: face-first into a villager")),
                         ("M", 708, 0, 1, "Moving forward to join the fight; I'll target the left zombie if it's available."),
                         ("F", (1, 713, "a1 takes damage (red hearts)"),
                          (0, 710, "a0 still at the villager")),
                         ("M", 712, 2, 0, "I am moving forward to support the fight near Agent 0."),
                         ("F", (1, 715, "a1 fights on, point-blank"), None),
                         ],
                 outcome="everyone talks support; nobody arrives — a1 fights alone"),
        ],
        figures=[
            dict(name="overview", xlim=None, keys=None, delib=False),
            dict(name="zoom_ep3", xlim=(2080, 2470),
                 keys=["switch", "solokill"], delib=False),
        ],
        filmstrips=[
            dict(name="filmstrip_zombie_kill", agent=2, ep=1, t0=684, t1=695,
                 title="Zombie kill, agent 2's view (episode 1, steps 684–695)"),
        ]),
    "qwen9b_base789": dict(
        ph=0.52,
        run=RUNS / "medium_runs/exp02_llm_9b/seed_789",
        exp="exp02_llm_9b", seed=789, lane="messages",
        joints=[(200, (0, 1)), (414, (1, 2)), (1009, (1, 2)),
                (1201, (0, 2)), (1297, (1, 2)), (2070, (0, 1)),
                (2082, (0, 2))],
        kills=[(633, 0), (1451, 1), (1458, 0), (2409, 0)],
        examples=[
            dict(key="anvil", x=(1003, 1013), ep=2, agents=(1, 2),
                 dpairs=[],
                 title="Cooperative anvil break",
                 frames=[("M", 207, 2, 1, "I'm here near the anvils. Remember, we need TWO agents Digging the same anvil simultaneously to break it."),
                         ("F", (1, 208, ""), (2, 208, "")),
                         ("M", 209, 1, 2, "I am digging the right anvil. Please ensure you are also Digging it simultaneously to break it."),
                         ("M", 209, 2, 1, "I am digging the anvil now, please dig the same one with me to break it cooperatively."),
                         ("F", (1, 216, ""), (2, 209, "")),
                         ("M", 215, 1, 2, "I'm digging the anvil now, keep digging with me!"),
                         ],
                 outcome="explicit two-agent plan, simultaneous dig — anvil breaks"),
            dict(key="combat", x=(622, 637), ep=1, agents=(0, 1),
                 dpairs=[],
                 title="Division of labour in combat",
                 frames=[("M", 625, 0, 1, "I'm hitting the zombie now, help me out!"),
                         ("F", (0, 624, "a0 fights, sword drawn"),
                          (1, 630, "a1 joins: agent0 in view")),
                         ("M", 630, 1, 0, "I am attacking the zombie you see. Keep attacking the other one."),
                         ("F", (0, 634, "a0's zombie flashes red"),
                          (1, 635, "a1's zombie, point-blank")),
                         ("M", 633, 0, 1, "I am attacking a zombie. agent_1, please attack the other zombie to help clear the room faster."),
                         ("M", 634, 1, 0, "I am hitting the left zombie. Keep attacking yours."),
                         ],
                 outcome="zombies split by agreement: 'the left one is mine, keep yours'"),
            dict(key="wipe", x=(788, 801), ep=1, agents=(0, 2), dpairs=[],
                 title="Failure: lost before the boss",
                 frames=[("M", 792, 0, 1, "I see two zombies ahead, attacking them now. I'll report when one is down."),
                         ("F", (0, 790, "a0 sees mobs with health bars"),
                          (2, 793, "a2: dark walls, teammates' tags only")),
                         ("M", 795, 2, 1, "I am still facing a wall in Ch4. Need to keep turning right to find the exit to Chamber 2."),
                         ("F", (0, 798, "a0 takes damage (red hearts)"),
                          (2, 799, "a2 still lost at a wall")),
                         ("M", 799, 2, 1, "I am facing a wall, turning left to find the open doorway or zombies."),
                         ("F", (0, 800, "episode ends at Ch5 entry"), None),
                         ],
                 outcome="the team scatters in Ch4; the episode ends 3 steps into Ch5"),
        ],
        figures=[
            dict(name="overview", xlim=None, keys=None, delib=False),
        ],
        filmstrips=[
            dict(name="filmstrip_anvil_break", agent=1, ep=2, t0=202, t1=217,
                 title="Anvil break, agent 1's view (episode 2, steps 202–217)"),
        ]),
    "qwen9b_heb42": dict(
        ph=0.50,
        run=RUNS / "medium_runs/exp08_llm_9b_social_prompt/seed_42",
        exp="exp08_llm_9b_social_prompt", seed=42, lane="bonds_snap",
        joints=[(400, (0, 1)), (518, (0, 2)), (1105, (0, 2)),
                (1204, (0, 1)), (2036, (0, 2)), (2046, (1, 2)),
                (2048, (0, 1))],
        kills=[(687, 2), (728, 1), (1500, 2)],
        examples=[
            dict(key="anvil", x=(1099, 1111), ep=2, agents=(0, 2),
                 dpairs=[(0, 2)],
                 title="Cooperative anvil break",
                 frames=[("M", 295, 0, 1, "Hit the anvil now!"),
                         ("F", (0, 296, "a0 targets the anvil"),
                          (2, 298, "a2: agent0 beside the anvil")),
                         ("M", 297, 2, 1, "I am digging the anvil with you, agent_1! Keep digging to break it together."),
                         ("M", 301, 2, 0, "I see you on the anvil! Let's dig together to break it."),
                         ("F", (0, 301, ""), (2, 303, "")),
                         ("F", (0, 306, ""), (2, 305, "")),
                         ("M", 306, 0, 1, "I see a dropped sword on the ground, collecting it now before we dig the anvils."),
                         ],
                 outcome="mutual 'dig with me' — the break credits a0 and a2"),
            dict(key="kill3", x=(1489, 1502), ep=2, agents=(2, 0),
                 dpairs=[(2, 0)],
                 title="Three agents on one zombie",
                 frames=[("M", 687, 0, 1, "I see a zombie centered on my crosshair and I am attacking it with my diamond sword."),
                         ("F", (2, 688, ""), (0, 687, "")),
                         ("M", 689, 2, 1, "I see you engaging the zombie, I will move forward to assist."),
                         ("F", (2, 697, ""), (0, 691, "")),
                         ("M", 691, 1, 2, "I see a zombie centered and am attacking it with the diamond sword to help clear the room."),
                         ("M", 697, 2, 1, "I am attacking the zombie. Keep focusing on yours!"),
                         ],
                 outcome="all three engage: a0 at 0.1 blocks, a1 and a2 both landing hits"),
            dict(key="phantom", x=(1192, 1203), ep=2, agents=(0, 1),
                 dpairs=[(0, 1)],
                 title="Failure: digging a phantom anvil",
                 frames=[("M", 390, 0, 1, "Digging the anvil now to break it with you."),
                         ("F", (0, 390, "a0 face-first into the anvil top"),
                          (1, 391, "a1 at the villager, sword out")),
                         ("M", 391, 1, 0, "I am centering the anvil and will Dig now to break it with you."),
                         ("M", 394, 0, 1, "Anvil HP is 0, digging again to trigger the break and open Door 2."),
                         ("F", (0, 394, ""), (1, 395, "")),
                         ("M", 395, 2, 1, "I'm backing up to find the anvils. Let's locate them together."),
                         ],
                 outcome="twelve steps of mutual confirmation on a spent anvil — no break fires"),
        ],
        figures=[
            dict(name="overview", xlim=None, keys=None, delib=False),
        ],
        filmstrips=[
            dict(name="filmstrip_anvil_break", agent=2, ep=2, t0=293, t1=308,
                 title="Anvil break, agent 2's view (episode 2, steps 293–308)"),
        ]),
}


# ── data loading ────────────────────────────────────────────────────────
def load_messages(run):
    out = []
    for e, (s0, _) in enumerate(run["_ep_bounds"]):
        p = run["_run_dir"] / "episodes" / f"ep_{e + 1:04d}" / "messages.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            m = json.loads(line)
            snd = int(re.search(r"(\d+)", m["sender"]).group(1))
            rcv = int(re.search(r"(\d+)", m["receiver"]).group(1))
            out.append((m["t"] + s0, snd, rcv))
    return out


def grab_frames(arm, wanted):
    import cv2
    run_dir, exp, seed = arm["run"], arm["exp"], arm["seed"]
    out, caps = {}, {}
    for a, ep, t in wanted:
        key = (a, ep)
        if key not in caps:
            for p in (run_dir / f"seed_{seed}_agent_{a}_ep{ep}.mp4",
                      run_dir / "gifs" / exp / f"seed_{seed}_agent_{a}_ep{ep}.mp4"):
                if p.exists():
                    caps[key] = cv2.VideoCapture(str(p))
                    break
            else:
                caps[key] = None
        cap = caps[key]
        if cap is None:
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(t - 1, 0), n - 1))
        ok, fr = cap.read()
        if ok:
            fr = cv2.convertScaleAbs(fr, alpha=1.45, beta=12)
            out[(a, ep, t)] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return out


def chamber_spans_from_steps(arm, run, min_len=12):
    try:
        inp = load_inputs(arm["run"])
        ch = inp["chamber"]
    except Exception:
        return []
    T = len(ch)
    maj = np.zeros(T, dtype=int)
    for t in range(T):
        known = ch[t][ch[t] > 0]
        maj[t] = int(np.median(known)) if len(known) else (maj[t - 1] if t else 0)
    bounds = {b for b, _ in run["_ep_bounds"]} | {T}
    spans, t0 = [], 0
    for t in range(1, T + 1):
        if t == T or maj[t] != maj[t0] or t in bounds:
            if maj[t0] > 0 and t - t0 >= min_len:
                spans.append((f"Ch{maj[t0]}", t0, t))
            elif spans and maj[t0] > 0 and t0 not in bounds:
                spans[-1] = (spans[-1][0], spans[-1][1], t)
            t0 = t
    return spans


def load_bond_series(arm):
    """(T,3,3) directed bond array: replay for gemma, snapshot interp else."""
    if arm["lane"] == "bonds_replay":
        return replay_three_factor(load_inputs(arm["run"]))["W"]
    if arm["lane"] == "bonds_snap":
        fm = json.loads((arm["run"] / "final_metrics.json")
                        .read_text(encoding="utf-8"))
        snaps = sorted((s["step"], np.array(s["W"])) for s in
                       fm.get("graph_snapshots", []) if s.get("W"))
        T = snaps[-1][0] + 1
        W = np.zeros((T, 3, 3))
        ss = [s for s, _ in snaps]
        for i in range(3):
            for j in range(3):
                W[:, i, j] = np.interp(np.arange(T), ss,
                                       [m[i, j] for _, m in snaps])
        return W
    return None


# ── timeline lanes ──────────────────────────────────────────────────────
def shade_chambers(ax, spans):
    for k, (_, lo, hi) in enumerate(spans):
        ax.axvspan(lo, hi, color=CHAMBER_TINT[k % 2], zorder=0, linewidth=0)


def chamber_labels(ax, spans, y, xlim):
    last = -1e9
    for lab, lo, hi in spans:
        cx = (lo + hi) / 2
        if xlim and not (xlim[0] < cx < xlim[1]):
            continue
        if cx - last > (xlim[1] - xlim[0]) * 0.03 if xlim else 100:
            if hi - lo > 40:
                ax.text(cx, y, lab, fontsize=FS["chamber"], color=MUTED,
                        ha="center", va="bottom", zorder=2)
                last = cx


def mutual_strip(ax, messages, spans, main=False):
    shade_chambers(ax, spans)
    per_step = {}
    for t, snd, rcv in messages:
        per_step.setdefault(t, set()).add((snd, rcv))
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_of = {pr: len(pairs) - 1 - i for i, pr in enumerate(pairs)}
    for t, dirs in per_step.items():
        for i, j in pairs:
            if (i, j) in dirs and (j, i) in dirs:
                y = y_of[(i, j)]
                ax.vlines(t, y + 0.14, y + 0.86, color=PAIR_C[(i, j)],
                          lw=0.6 if main else 0.5, alpha=0.85, zorder=2)
    ax.set_yticks([y_of[pr] + 0.5 for pr in pairs])
    ax.set_yticklabels([f"a{i}↔a{j}" for i, j in pairs],
                       fontsize=FS["raster"])
    for tick, pr in zip(ax.get_yticklabels(), pairs):
        tick.set_color(PAIR_C[pr])
    ax.set_ylim(-0.4 if main else 0, len(pairs) + (0.6 if main else 0))
    ax.set_ylabel("mutual\nmessages", fontsize=FS["axis"], color=INK)
    ax.tick_params(length=0)
    return y_of


def lane_bonds(ax, arm, Wr, spans, xlim):
    shade_chambers(ax, spans)
    steps = np.arange(len(Wr))
    for q in PAIR_C:
        ys = (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2
        ax.plot(steps, ys, color=PAIR_C[q], lw=1.25,
                label=f"$\\bar{{W}}$(a{q[0]}–a{q[1]})",
                solid_capstyle="round", zorder=3)
    for t, q in arm["joints"]:
        y = (Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2
        ax.scatter(t, y, marker="o", s=24, facecolor="white",
                   edgecolor=PAIR_C[q], linewidths=1.1, zorder=6)
    for t, killer in arm["kills"]:
        ys = [(Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2
              for q in PAIR_C if killer in q]
        ax.scatter(t, max(ys), marker="*", s=58, facecolor=AGENT_C[killer],
                   edgecolor="white", linewidths=0.6, zorder=6)
    ymax = float(np.nanmax(Wr))
    chamber_labels(ax, spans, 0.075, xlim)
    ax.grid(axis="y", color="#eef1f4", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("bond strength", fontsize=FS["axis"], color=INK)
    ax.tick_params(labelsize=FS["tick"])
    ax.legend(fontsize=5.9, ncol=3, frameon=False, loc="upper left",
              borderaxespad=0.2)
    ax.set_ylim(0.05, ymax * 1.21)


def lane_coassign(ax, arm, run, spans, xlim):
    shade_chambers(ax, spans)
    orch = load_orch(run)
    pair_y = {(0, 1): 2, (0, 2): 1, (1, 2): 0}
    team_by_pair = {q: [] for q in PAIR_C}
    for tid, lo, hi, q in orch["team_iv"]:
        team_by_pair[tuple(sorted(q))].append((lo, hi))
    for q, y in pair_y.items():
        for lo, hi in orch["pair_iv"][q]:
            ax.broken_barh([(lo, hi - lo)], (y - 0.22, 0.44),
                           facecolors=PAIR_C[q], alpha=0.28, linewidth=0,
                           zorder=2)
        for lo, hi in team_by_pair[q]:
            ax.broken_barh([(lo, hi - lo)], (y - 0.22, 0.44),
                           facecolors=PAIR_C[q], alpha=0.95,
                           edgecolors="white", linewidth=0.3, zorder=3)
    for t, q in arm["joints"]:
        ax.scatter(t, pair_y[q] + 0.38, marker="o", s=58, facecolor="white",
                   edgecolor=PAIR_C[q], linewidths=1.1, zorder=6)
    for t, killer in arm["kills"]:
        ax.scatter(t, 2.72, marker="*", s=160, facecolor=AGENT_C[killer],
                   edgecolor="white", linewidths=0.6, zorder=6)
    chamber_labels(ax, spans, -0.45, xlim)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0–a1", "a0–a2", "a1–a2"], fontsize=FS["raster"])
    for tick, q in zip(ax.get_yticklabels(), PAIR_C):
        tick.set_color(PAIR_C[q])
    ax.set_ylim(-0.6, 3.05)
    ax.set_ylabel("orchestrator\nco-assigned pairs", fontsize=FS["axis"],
                  color=INK)


# ── panels ──────────────────────────────────────────────────────────────
def truncate(txt, n=MAXLEN):
    return txt if len(txt) <= n else txt[:n - 1].rstrip() + "…"


def panel(fig, arm, ex, frames, Wr, x0, w, y0, h, show_delib):
    fig.patches.append(Rectangle((x0, y0), w, h, transform=fig.transFigure,
                                 facecolor="white", edgecolor=PANEL_EDGE,
                                 lw=1.0, zorder=1))
    pad = 0.008
    aL, aR = ex["agents"]
    y = y0 + h - 0.018
    fig.text(x0 + pad, y, ex["title"], fontsize=FS["panel_title"],
             color=INK, va="center", fontweight="bold", zorder=4)
    y -= 0.024
    colw = (w - 3 * pad) / 2
    for a, cx in ((aL, x0 + pad + colw / 2),
                  (aR, x0 + 2 * pad + colw * 1.5)):
        fig.text(cx, y, f"agent {a}", fontsize=9.5, color=AGENT_C[a],
                 ha="center", va="center", fontweight="bold", zorder=4)
    y -= 0.010

    fr_h, cap_h, msg_h = 0.079, 0.008, 0.0150
    for row in ex["frames"]:
        if row[0] == "F":
            y -= fr_h
            for slot, xleft in ((row[1], x0 + pad),
                                (row[2], x0 + 2 * pad + colw)):
                if slot is None:
                    continue
                a, t, cap = slot
                ax = fig.add_axes([xleft, y, colw, fr_h])
                ax.axis("off")
                ax.set_zorder(3)
                fr = frames.get((a, ex["ep"], t))
                if fr is not None:
                    ax.imshow(fr)
                ax.text(0.02, 0.95, f"t={t}", transform=ax.transAxes,
                        fontsize=6.4, color="white", va="top", zorder=5,
                        bbox=dict(facecolor="#00000099", edgecolor="none",
                                  pad=1.1))
            y -= cap_h
        else:                                # Minecraft-style chat line
            import textwrap
            _, t, src, dst, txt = row
            lines = textwrap.wrap(txt, 46) or [""]
            y -= msg_h
            fig.text(x0 + pad + 0.002, y + 0.004, f"<a{src} → a{dst}>",
                     fontsize=7.2, color=AGENT_C[src], va="bottom",
                     family="monospace", fontweight="bold", zorder=4)
            fig.text(x0 + pad + 0.047, y + 0.004, lines[0],
                     fontsize=7.2, color=INK, va="bottom",
                     family="monospace", zorder=4)
            for extra in lines[1:]:
                y -= 0.0125
                fig.text(x0 + pad + 0.047, y + 0.004, extra,
                         fontsize=7.2, color=INK, va="bottom",
                         family="monospace", zorder=4)
            y -= 0.0032
    y -= 0.012

    if show_delib and arm.get("delib", {}).get(ex["key"]):
        import textwrap
        fig.text(x0 + pad, y, "Social deliberation", fontsize=FS["header"],
                 color=MUTED, va="top", fontweight="bold", zorder=4)
        y -= 0.016
        for a, step, reason in arm["delib"][ex["key"]]:
            fig.text(x0 + pad + 0.002, y, f"a{a} (t≈{step}):",
                     fontsize=FS["body"], color=AGENT_C[a], va="top",
                     fontweight="bold", zorder=4)
            y -= 0.0135
            for ln in textwrap.wrap(reason, 70)[:3]:
                fig.text(x0 + pad + 0.006, y, ln, fontsize=FS["body"],
                         color=INK, va="top", zorder=4)
                y -= 0.0122
            y -= 0.005
        y -= 0.004

    if SHOW_BOND_EFFECT and Wr is not None and ex["dpairs"]:
        y -= 0.012
        spark_w, spark_h = 0.105, 0.038
        xcur = x0 + pad + 0.004
        for i, j in ex["dpairs"]:
            t0 = max(ex["x"][0] - 60, 0)
            t1 = min(ex["x"][1] + 80, len(Wr) - 1)
            col = PAIR_C[tuple(sorted((i, j)))]
            wb = float(Wr[max(ex["x"][0] - 1, 0), i, j])
            wa = float(Wr[min(ex["x"][1] + 25, len(Wr) - 1), i, j])
            d = wa - wb
            fig.text(xcur, y,
                     f"$W_{{a{i}\\rightarrow a{j}}}$"
                     f"  {wb:.2f}→{wa:.2f} ({'+' if d >= 0 else '−'}{abs(d):.2f})",
                     fontsize=8.4, color=col, va="top", fontweight="bold",
                     zorder=4)
            axS = fig.add_axes([xcur, y - 0.013 - spark_h, spark_w, spark_h])
            axS.set_zorder(3)
            ts = np.arange(t0, t1)
            axS.plot(ts, Wr[t0:t1, i, j], color=col, lw=1.5)
            axS.plot(ts, Wr[t0:t1, j, i], color=col, lw=0.8,
                     ls=(0, (2, 1.5)), alpha=0.6)
            axS.axvspan(ex["x"][0], ex["x"][1], color="#5b6472", alpha=0.12,
                        linewidth=0)
            for s_ in axS.spines.values():
                s_.set_visible(False)
            axS.set_xticks([])
            axS.set_yticks([])
            xcur += spark_w + 0.040
        y -= 0.013 + spark_h + 0.010
    fig.text(x0 + pad, y, ex["outcome"], fontsize=FS["body"], color=INK,
             va="top", zorder=4)


# ── ICLR story callouts ─────────────────────────────────────────────────
def _status(ex):
    """Short, evidence-facing badge; intentionally avoids generic card prose."""
    txt = (ex.get("title", "") + " " + ex.get("outcome", "")).lower()
    if "never" in txt or "failure" in txt or "stuck" in txt:
        return "FAILURE"
    if "alone" in txt or "nobody arrives" in txt:
        return "MISCOORDINATION"
    return "SUCCESS"


def _pick_story_rows(ex):
    """Keep only the evidence a reader can actually parse at 5.5 in.

    The full transcript remains in the logs / appendix filmstrip.  The main
    figure shows two visual states plus one short exchange.
    """
    frows = [r for r in ex["frames"] if r[0] == "F"]
    if len(frows) <= 2:
        chosen_f = frows
    else:
        chosen_f = [frows[0], frows[-1]]

    mrows = [r for r in ex["frames"] if r[0] == "M"]
    chosen_m = []
    if mrows:
        chosen_m.append(mrows[0])
        # Prefer a reply from the other speaker; otherwise keep the last line.
        other = next((r for r in mrows[1:] if r[2] != mrows[0][2]), None)
        if other is not None:
            chosen_m.append(other)
        elif len(mrows) > 1:
            chosen_m.append(mrows[-1])
    return chosen_f, chosen_m


def _story_layout(n):
    """Staggered callouts: intentionally not three equal bottom cards."""
    if n == 1:
        return [(0.12, 0.055, 0.76, 0.48)]
    if n == 2:
        return [(0.045, 0.12, 0.44, 0.39),
                (0.515, 0.235, 0.44, 0.35)]
    if n == 3:
        return [(0.025, 0.285, 0.29, 0.315),
                (0.325, 0.055, 0.35, 0.405),
                (0.695, 0.245, 0.28, 0.34)]
    # Conservative fallback for appendix variants.
    gap = 0.018
    w = (0.95 - gap * (n - 1)) / n
    return [(0.025 + i * (w + gap), 0.09 + 0.025 * (i % 2), w, 0.42)
            for i in range(n)]


def _short(txt, width):
    import textwrap
    return textwrap.shorten(txt.replace("\n", " "), width=width,
                            placeholder=" …")


def story_callout(fig, arm, ex, frames, idx, rect, show_delib=False):
    """A borderless, editorial-style zoom callout.

    Visual hierarchy is title -> frames -> one exchange -> takeaway.  This is
    closer to the qualitative callouts used in embodied-agent papers than a
    transcript-heavy dashboard.
    """
    x0, y0, w, h = rect
    fig_h = fig.get_figheight()
    fig_w = fig.get_figwidth()
    aL, aR = ex["agents"]
    pair = tuple(sorted((aL, aR)))
    accent = PAIR_C.get(pair, INK)
    pad = 0.010

    # A short rule gives grouping without the 'three cards' look.
    fig.add_artist(plt.Line2D([x0, x0 + min(w, 0.14)],
                              [y0 + h, y0 + h],
                              transform=fig.transFigure, color=accent,
                              lw=2.1, solid_capstyle="round", zorder=5))

    # Number = the same number used on the timeline focus window.
    fig.text(x0, y0 + h - 0.018, f"{idx}", fontsize=FS["badge"],
             color="white", ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.22", facecolor=accent,
                       edgecolor="none"), zorder=6)
    fig.text(x0 + 0.020, y0 + h - 0.018, ex["title"],
             fontsize=FS["panel_title"], color=INK, va="center",
             fontweight="bold", zorder=5)
    fig.text(x0 + w, y0 + h - 0.018, _status(ex), fontsize=FS["badge"],
             color=accent, ha="right", va="center", fontweight="bold",
             zorder=5)

    frows, mrows = _pick_story_rows(ex)
    top = y0 + h - 0.052
    col_gap = 0.006
    colw = (w - col_gap) / 2
    # Maintain 16:9 physically, then cap so smaller cards remain readable.
    im_h = min((colw * fig_w) * (9 / 16) / fig_h, 0.092)

    # Agent labels act as the only column headers.
    fig.text(x0 + colw / 2, top + 0.004, f"agent {aL}",
             fontsize=FS["caption"], color=AGENT_C[aL], ha="center",
             va="bottom", fontweight="bold")
    fig.text(x0 + colw + col_gap + colw / 2, top + 0.004, f"agent {aR}",
             fontsize=FS["caption"], color=AGENT_C[aR], ha="center",
             va="bottom", fontweight="bold")

    y = top - im_h
    for rr, row in enumerate(frows[:2]):
        for slot, xleft in ((row[1], x0),
                            (row[2], x0 + colw + col_gap)):
            if slot is None:
                continue
            a, t, _cap = slot
            ax = fig.add_axes([xleft, y, colw, im_h], zorder=3)
            ax.axis("off")
            fr = frames.get((a, ex["ep"], t))
            if fr is not None:
                ax.imshow(fr)
            else:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="#f3f4f6", edgecolor="none"))
            ax.text(0.025, 0.94, f"t={t}", transform=ax.transAxes,
                    fontsize=4.9, color="white", va="top",
                    bbox=dict(facecolor="#0000009a", edgecolor="none",
                              pad=0.7))
        y -= im_h + 0.007

    # One compact exchange, not a wall of chat text.
    for row in mrows[:2]:
        _, _t, src, dst, txt = row
        fig.text(x0, y, f"a{src}→a{dst}", fontsize=FS["chat"],
                 color=AGENT_C[src], va="top", fontweight="bold",
                 family="DejaVu Sans Mono")
        fig.text(x0 + 0.045, y, _short(txt, 58), fontsize=FS["chat"],
                 color=INK, va="top", family="DejaVu Sans Mono")
        y -= 0.020

    if show_delib and arm.get("delib", {}).get(ex["key"]):
        a, _step, reason = arm["delib"][ex["key"]][0]
        y -= 0.002
        fig.text(x0, y, "deliberation", fontsize=FS["badge"], color=MUTED,
                 va="top", fontstyle="italic")
        fig.text(x0 + 0.058, y, _short(reason, 72), fontsize=FS["body"],
                 color=MUTED, va="top", fontstyle="italic")
        y -= 0.025

    # The takeaway is the visual punch line and therefore gets more weight.
    y_out = max(y0 + 0.008, y - 0.005)
    fig.text(x0, y_out, "Outcome", fontsize=FS["body"], color=accent,
             va="top", fontweight="bold")
    import textwrap
    lines = textwrap.wrap(ex["outcome"], max(28, int(62 * w / 0.30)))[:3]
    fig.text(x0 + 0.050, y_out, "\n".join(lines), fontsize=FS["body"],
             color=INK, va="top", linespacing=1.18)


# ── figure builder ──────────────────────────────────────────────────────
def build_figure(arm_name, arm, spec, run, Wr, messages, spans, frames,
                 outdir):
    """Main-paper layout: a compact analytic timeline plus staggered evidence.

    The old version used three equal bordered transcript panels.  This version
    treats the timeline as the visual spine, assigns each selected episode a
    numbered focus window, and lets the evidence callouts form an asymmetric
    story below it.
    """
    examples = sorted((e for e in arm["examples"]
                       if spec["keys"] is None or e["key"] in spec["keys"]),
                      key=lambda e: e["x"][0])
    xlim = spec["xlim"] or (0, run["_ep_bounds"][-1][1])
    show_delib = spec["delib"]
    n = len(examples)

    # Build at the actual ICLR text width (5.5 in).  The slight extra height
    # accommodates the staggered callouts without forcing tiny text.
    H = 5.15 + (0.32 if show_delib else 0.0)
    fig = plt.figure(figsize=(FIG_W, H))

    two_lanes = arm["lane"] in ("bonds_replay", "bonds_snap", "coassign")
    timeline_bottom = 0.655
    if two_lanes:
        gs = fig.add_gridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.055,
                              left=0.105, right=0.992, top=0.982,
                              bottom=timeline_bottom)
        axR = fig.add_subplot(gs[0])
        axB = fig.add_subplot(gs[1], sharex=axR)
    else:
        gs = fig.add_gridspec(1, 1, left=0.105, right=0.992, top=0.972,
                              bottom=timeline_bottom)
        axB = fig.add_subplot(gs[0])
        axR = None

    if axR is not None:
        mutual_strip(axR, messages, spans)
        plt.setp(axR.get_xticklabels(), visible=False)

    if arm["lane"] in ("bonds_replay", "bonds_snap"):
        lane_bonds(axB, arm, Wr, spans, xlim)
    elif arm["lane"] == "coassign":
        lane_coassign(axB, arm, run, spans, xlim)
    else:
        y_of = mutual_strip(axB, messages, spans, main=True)
        for t, q in arm["joints"]:
            axB.scatter(t, y_of[q] + 0.5, marker="o", s=24,
                        facecolor="white", edgecolor=PAIR_C[q],
                        linewidths=1.05, zorder=6)
        for t, killer in arm["kills"]:
            axB.scatter(t, 3.25, marker="*", s=58,
                        facecolor=AGENT_C[killer], edgecolor="white",
                        linewidths=0.45, zorder=6, clip_on=False)
        chamber_labels(axB, spans, -0.36, xlim)

    axB.set_xlabel("environment step (cumulative)", fontsize=FS["axis"],
                   labelpad=1.5)
    axB.tick_params(labelsize=FS["tick"], pad=1.5)

    # Highlight only the local windows.  Numbered badges replace the long
    # dashed corner connectors and make the timeline + story relationship
    # legible even when the figure is printed small.
    for idx, ex in enumerate(examples, 1):
        lo, hi = ex["x"]
        accent = PAIR_C.get(tuple(sorted(ex["agents"])), INK)
        for ax in filter(None, (axR, axB)):
            ax.axvspan(lo, hi, color=accent, alpha=0.075, zorder=1,
                       linewidth=0)
        mid = 0.5 * (lo + hi)
        axB.annotate(str(idx), xy=(mid, 0.965),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="center", fontsize=FS["badge"],
                     color="white", fontweight="bold", zorder=9,
                     bbox=dict(boxstyle="circle,pad=0.19",
                               facecolor=accent, edgecolor="white",
                               linewidth=0.55))

    for ax in filter(None, (axR, axB)):
        for e, (o, _) in enumerate(run["_ep_bounds"]):
            if e and xlim[0] < o < xlim[1]:
                ax.axvline(o, color=INK, lw=0.6, ls=(0, (3, 2)), alpha=0.28)
        ax.set_xlim(xlim[0] - (xlim[1] - xlim[0]) * 0.005,
                    xlim[1] + (xlim[1] - xlim[0]) * 0.005)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Minimal symbol key.  It can move to the caption later if space is tight.
    axB.text(0.995, 1.035, "○ joint co-firing   ★ kill",
             transform=axB.transAxes, ha="right", va="bottom",
             fontsize=FS["caption"], color=MUTED)

    # The intentionally uneven rhythm is the main visual change.
    for idx, (ex, rect) in enumerate(zip(examples, _story_layout(n)), 1):
        story_callout(fig, arm, ex, frames, idx, rect,
                      show_delib=show_delib)

    # Save with tight bounds disabled: figure coordinates are already composed
    # for 5.5 in and should not change between PDF/SVG/PNG.
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"{spec['name']}.{ext}",
                    dpi=DPI if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)
    print(f"  wrote {outdir.name}/final/{spec['name']}.*")


def build_filmstrip(arm, fsp, outdir):
    import cv2
    ts = list(range(fsp["t0"], fsp["t1"] + 1))
    wanted = [(fsp["agent"], fsp["ep"], t) for t in ts]
    frames = grab_frames(arm, wanted)
    cols = 8
    rows = int(np.ceil(len(ts) / cols))
    fw_in = 13.6 * 0.94 / cols
    fh_in = fw_in * 9 / 16
    row_in = fh_in + 0.26
    H = 0.52 + rows * row_in + 0.06
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.6, H))
        fig.text(0.03, 1 - 0.16 / H, fsp["title"], fontsize=12.5,
                 color=INK, va="top", fontweight="bold")
        for k, t in enumerate(ts):
            r, c = divmod(k, cols)
            bot_in = H - 0.52 - (r + 1) * row_in + 0.20
            ax = fig.add_axes([0.03 + c * (0.94 / cols), bot_in / H,
                               (0.94 / cols) * 0.965, fh_in / H])
            ax.axis("off")
            fr = frames.get((fsp["agent"], fsp["ep"], t))
            if fr is not None:
                ax.imshow(fr)
            ax.set_title(f"t={t}", fontsize=7.5, color=MUTED, pad=1.5)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{fsp['name']}.{ext}",
                        dpi=DPI if ext == "png" else None,
                        facecolor="white")
        plt.close(fig)
    print(f"  wrote {outdir.name}/final/{fsp['name']}.*")


def main(only=None):
    ARMS["gemma3f_seed123"]["delib"] = GEMMA_DELIB
    for arm_name, arm in ARMS.items():
        if only and arm_name not in only:
            continue
        print(f"== {arm_name} ==")
        outdir = ROOT / arm_name / "final"
        outdir.mkdir(parents=True, exist_ok=True)
        for ex in arm["examples"]:                 # column purity guard
            aL, aR = ex["agents"]
            for row in ex["frames"]:
                if row[0] != "F":
                    continue
                for slot, want in ((row[1], aL), (row[2], aR)):
                    if slot is not None and slot[0] != want:
                        raise AssertionError(
                            f"{arm_name}/{ex['key']}: frame a{slot[0]}@{slot[1]}"
                            f" sits in the a{want} column")
        run = load_run(arm["run"])
        messages = load_messages(run)
        spans = chamber_spans_from_steps(arm, run)
        Wr = load_bond_series(arm)
        wanted = [(s[0], ex["ep"], s[1]) for ex in arm["examples"]
                  for r in ex["frames"] if r[0] == "F"
                  for s in (r[1], r[2]) if s]
        frames = grab_frames(arm, wanted)
        print(f"  frames {len(frames)}/{len(set(wanted))}, "
              f"spans {len(spans)}, msgs {len(messages)}")
        with plt.rc_context(RC):
            for spec in arm["figures"]:
                build_figure(arm_name, arm, spec, run, Wr, messages, spans,
                             frames, outdir)
        for fsp in arm["filmstrips"]:
            build_filmstrip(arm, fsp, outdir)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
