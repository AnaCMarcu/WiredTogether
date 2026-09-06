"""Story timelines, all four arms: mechanism top lane + paired-communication
lane + two mob-kill/anvil story panels (game frames + the contributing
agents' conversation). Arms: Gemma+3f (seed 123, replayed bond lane),
orchestrator (seed 789, assignment Gantt), Qwen3.5-9B base (seed 789,
event lane), Qwen3.5-9B+Hebbian (seed 42, bond-snapshot lane).

Frames from <run>/seed_<S>_agent_<A>_ep<E>.mp4 (1 frame per env step; also
looked up under gifs/<exp>/); missing files render as placeholders."""
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from make_directive_timelines import (CHAMBER_FILL, load_chamber_bands,
                                      load_msg_share, load_orch, load_run)

from paths import ASSETS, RUNS  # noqa: E402  (also puts siblings on sys.path)

OUT = ASSETS / "timelines"
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_C = {(0, 1): "#4a3aa7", (0, 2): "#008300", (1, 2): "#d55181"}
AGENT_C = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
INK, MUTED = "#31373f", "#98a1ad"


def W(t):
    return "\n".join(textwrap.wrap(t, 54))


ARMS = {
    "gemma3f_seed123": dict(
        title="Gemma-E4B + three-factor Hebbian — seed 123",
        run=RUNS / "new_exp_0_gemma/new_exp_0_gemma_hebbian3f/seed_123",
        exp="new_exp_0_gemma_hebbian3f", seed=123, lane="bonds3f",
        cards=[
            dict(ep=1, x=260, tag="A1", pair=(1, 2),
                 title="A1. Failure: anvil never reached",
                 inter="20 steps of approach talk — nobody digs",
                 frames=[(1, 258, "a1: wall, not anvils"),
                         (2, 262, "a2 among animals"),
                         (0, 266, "a0 alone at an anvil")],
                 chat=[]),
            dict(ep=1, x=722, tag="A2", pair=(0, 1),
                 title="A2. Mob kill with support",
                 inter="sight \u2192 focus-fire call \u2192 kill; a0 adjacent",
                 frames=[(1, 716, "a1 sights the zombie"),
                         (0, 720, "a0 stands by the fight"),
                         (1, 723, "final hits land (+60)")],
                 chat=[]),
            dict(ep=3, x=2401, tag="A3", pair=(0, 1),
                 title="A3. Switch chain (reciprocal help)",
                 inter="a0 presses \u2192 a1 freed \u2192 a2 presses back \u2192 a0 freed",
                 frames=[(0, 403, "a0 digs its switch"),
                         (2, 405, "a2 turns to its own switch"),
                         (0, 413, "a0 freed in turn")],
                 chat=[]),
        ],
        joints=[(442, (0, 1), ""), (1407, (0, 1), ""), (2401, (0, 1), "")],
        kills=[(722, 1, "a0 near"), (1683, 2, "solo"), (2642, 2, "a1 near")],
        panels=[
            dict(tag="", ep=1, slug="kill_ep1_t722",
                 title="Mob kill — a1 attacks, a0 stands by, a2 closes in (ep 1, t=722)",
                 frames=[(1, 716), (0, 716), (1, 718), (1, 719), (0, 720),
                         (1, 720), (1, 722), (1, 723)],
                 chat=[(716, 1, W("Zombie sighted in Ch4. Focusing fire now.")),
                       (718, 0, W("Orientation should be right now; ready to engage zombies.")),
                       (719, 2, W("Zombies are visible right; I'll join the fight.")),
                       (720, 1, W("Boss is centered; I am attacking it now.")),
                       (721, 1, W("I am joining the fight now. Focus on the boss.")),
                       (722, 0, W("I am turning right to orient toward the fight, Agent 2.")),
                       (722, 1, W("I see the boss centered. I'm attacking now to help the fight."))]),
        ],
        extras=[
            dict(ep=1, slug="switch_ep1_t442",
                 title="Switch room — a0 presses, a1 passes (ep 1, t=442)",
                 frames=[(1, 441, "a1 locked in; blue cube +\ndoor frame visible"),
                         (0, 442, "a0: switch centered"),
                         (0, 443, "a0 digs the switch"),
                         (0, 444, "switch up close, red door edge"),
                         (1, 444, "a1's doorway opens"),
                         (0, 447, "a0 presses again for its own door")],
                 chat=[(437, 0, W("I am waiting for the switch press. Let me know when you see it.")),
                       (439, 0, W("Still stuck in cell. Check door state if you see any change.")),
                       (440, 1, W("I am turning to find the south wall switch, let me know if you see any door changes.")),
                       (441, 0, W("Blue cube is right; I'm turning to face it now.")),
                       (442, 0, W("Blue switch is centered. I will press it now.")),
                       (443, 0, W("I am digging the blue cube now; check if your door state changed.")),
                       (443, 1, W("Door is open, moving north now")),
                       (444, 0, W("Digging switch now, need door open for me")),
                       (447, 1, W("Door is open, moving out now."))]),
            dict(ep=3, slug="switch_chain_ep3_t401",
                 title="Switch chain — a0 frees a1, a2 answers by pressing its own (ep 3, t=401)",
                 frames=[(1, 401, "a1's door: locked (red)"),
                         (0, 403, "a0 digs its switch"),
                         (1, 404, "a1 freed, moving north"),
                         (2, 405, "a2 sees the team; its own\nswitch at bottom of view"),
                         (2, 407, "a2 turns to the locked door,\ngoes to press its switch"),
                         (0, 413, "a0 freed, follows the others")],
                 chat=[(400, 0, W("Need agent_2 to press switch C to open my door.")),
                       (401, 0, W("Switch is centered, ready to press. Need agent_2 to press theirs.")),
                       (401, 2, W("Agent 0, I am still in the cell. I will look for the switch C now.")),
                       (403, 1, W("Cell door is open now, moving north to communal area.")),
                       (403, 2, W("Agent 0 is pressing switch now, I will turn to press mine.")),
                       (405, 2, W("Switch is on the south wall, I need to face it to press it.")),
                       (406, 2, W("Switch is on the south wall, turning right to face it now.")),
                       (413, 0, W("Door is open now, following you both."))]),
            dict(ep=3, slug="kill_ep3_t642",
                 title="Mob kill — a2 attacks, a1 closes in (ep 3, t=642)",
                 frames=[(2, 635), (2, 638), (2, 640), (1, 641), (2, 642),
                         (1, 643)],
                 chat=[(637, 2, W("Zombie centered. I'm attacking now.")),
                       (640, 2, W("Focusing on the boss now, Agent 0. Let me know if the door state changes.")),
                       (641, 2, W("Boss is centered, I'm engaging now.")),
                       (643, 1, W("Zombies are present, moving forward to engage them."))]),
            dict(ep=1, slug="failure_bosswipe_ep1_t799",
                 title="Failure · Ch5 boss — a1 and a2 die, a0 never arrives (ep 1, t≈799–807)",
                 frames=[(1, 801), (0, 801), (0, 804), (0, 805), (2, 807),
                         (1, 809), (0, 810), (1, 810)],
                 chat=[(800, 1, W("Moving forward now to engage boss.")),
                       (804, 0, W("Still turning right as instructed")),
                       (806, 2, W("Boss is here, moving in to assist combat.")),
                       (807, 1, W("Focusing on boss now, let's maximize damage output.")),
                       (809, 1, W("Boss centered, ready to attack."))]),
        ]),
    "orch789": dict(
        title="Central orchestrator (villager-advisory) — seed 789",
        run=RUNS / "orchestrator/new_exp_0_gemma_orch_villager_advisory/seed_789",
        exp="new_exp_0_gemma_orch_villager_advisory", seed=789, lane="orch",
        joints=[(1230, (0, 1), ""), (1387, (0, 2), ""),
                (2122, (0, 1), ""), (2213, (0, 2), "")],
        kills=[(690, 2, "a1 near"), (1511, 1, "a0 near"), (2435, 1, "solo")],
        panels=[
            dict(tag="", ep=1, slug="kill_ep1_t690",
                 title="Mob kill — a2 attacks, a1 arrives (ep 1, t=690)",
                 frames=[(2, 689), (1, 690), (2, 690), (1, 691), (2, 692),
                         (1, 693), (2, 693), (2, 694)],
                 chat=[(688, 2, W("Zombie centered, ready to attack now.")),
                       (688, 1, W("Zombies are visible; I'm turning right to orient for combat.")),
                       (689, 2, W("Zombie is centered, ready to strike. Focus fire on this target.")),
                       (689, 1, W("Zombie spotted right, turning to engage.")),
                       (690, 2, W("Zombie is centered, continuing assault.")),
                       (692, 1, W("Zombie centered, attacking now. Focus fire!")),
                       (692, 2, W("Zombie centered, continuing attack."))]),
        ],
        extras=[
            dict(ep=3, slug="switch_ep3_t403",
                 title="Switch room — a0 presses, a1 exits (ep 3, t=403)",
                 frames=[(0, 404), (1, 405), (2, 406), (0, 406), (1, 409),
                         (2, 412)],
                 chat=[(398, 1, W("Moving forward to meet you at Anvil A. Ready to coordinate the hit.")),
                       (402, 1, W("Door is still locked, waiting for you to press switch.")),
                       (403, 0, W("I see the blue switch now. I will attempt to dig it to open the door.")),
                       (405, 0, W("Switch is centered, I am pressing it now to try and open my door.")),
                       (407, 1, W("Door is open, I am moving north now to meet you."))]),
            dict(ep=3, slug="failure_solokill_ep3_t716",
                 title="Failure · Focus-fire talk, solo kill — teammates 7–12 blocks away (ep 3, t=716)",
                 frames=[(1, 706), (0, 708), (1, 709), (0, 710), (1, 713),
                         (1, 715)],
                 chat=[(706, 1, W("I see two zombies centered. Equipping sword now to start attacking.")),
                       (707, 1, W("I'm targeting the right zombie now; let me know when you focus on the left one.")),
                       (708, 0, W("Moving forward to join the fight; I'll target the left zombie if it's available.")),
                       (710, 1, W("Focusing on this center zombie; let's coordinate fire on the next one.")),
                       (712, 2, W("I am moving forward to support the fight near Agent 0."))]),
        ]),
    "qwen9b_base789": dict(
        title="Qwen3.5-9B base — seed 789",
        run=RUNS / "medium_runs/exp02_llm_9b/seed_789",
        exp="exp02_llm_9b", seed=789, lane="events",
        joints=[(200, (0, 1), "anvil A1"), (414, (1, 2), ""),
                (1009, (1, 2), "anvil B1"), (1201, (0, 2), ""),
                (1297, (1, 2), ""), (2070, (0, 1), ""), (2082, (0, 2), "")],
        kills=[(633, 0, "a1 engaged"), (1451, 1, "a2 near"),
               (1458, 0, "a2 near"), (2409, 0, "a1 engaged")],
        panels=[
            dict(tag="A", ep=2, slug="anvilB1_ep2_t208",
                 title="A · Anvil B1 — a1+a2 break it together (ep 2, t=208)",
                 frames=[(2, 208), (1, 208), (2, 209), (1, 209), (2, 215),
                         (1, 216)],
                 chat=[(207, 2, W("I'm here near the anvils. Remember, we need TWO agents Digging the same anvil simultaneously to break it. Let's coordinate.")),
                       (209, 1, W("I am digging the right anvil. Please ensure you are also Digging it simultaneously to break it.")),
                       (209, 2, W("I am digging the anvil now, please dig the same one with me to break it cooperatively.")),
                       (215, 1, W("I'm digging the anvil now, keep digging with me!"))]),
            dict(tag="B", ep=1, slug="kill_ep1_t633",
                 title="B · Mob kill — a0 and a1 split the zombies (ep 1, t=633)",
                 frames=[(0, 624), (0, 627), (1, 630), (0, 634), (1, 635),
                         (1, 636)],
                 chat=[(625, 0, W("I'm hitting the zombie now, help me out!")),
                       (630, 1, W("I am attacking the zombie you see. Keep attacking the other one.")),
                       (633, 0, W("I am attacking a zombie. agent_1, please attack the other zombie to help clear the room faster.")),
                       (634, 1, W("I am hitting the left zombie. Keep attacking yours."))]),
        ],
        extras=[
            dict(ep=1, slug="anvilA1_ep1_t200",
                 title="Anvil A1 — a0+a1 break it together (ep 1, t=200)",
                 frames=[(0, 199), (1, 201), (0, 201), (0, 202), (1, 205),
                         (0, 206)],
                 chat=[(200, 1, W("I am on the purple anvil but need 2 agents to break it. Who is helping me right now?")),
                       (200, 0, W("I am digging the front anvil now to help break it with you. Keep attacking the same one!")),
                       (201, 1, W("I'm turning right to center the front anvil so we can dig it together.")),
                       (203, 1, W("I see you on the pedestal, I need to LookUp to hit the anvil top."))]),
            dict(ep=2, slug="switch_ep2_t400",
                 title="Switch room — a2 presses, a0 exits (ep 2, t=400)",
                 frames=[(2, 400), (2, 401), (0, 402), (0, 403)],
                 chat=[(400, 2, W("I am in Cell C, pressing the blue switch to free you (Cell A).")),
                       (401, 2, W("I see the blue switch in my cell. Please check your cell for a switch and press it to open my door.")),
                       (402, 0, W("My door is open, I'm exiting now.")),
                       (403, 0, W("I am exiting my cell now to join you in the communal room."))]),
            dict(ep=1, slug="failure_bosswipe_ep1_t799",
                 title="Failure · Lost in Ch4, wiped on Ch5 entry (ep 1, t≈799)",
                 frames=[(0, 790), (2, 793), (0, 794), (0, 798), (2, 799),
                         (0, 800)],
                 chat=[(792, 0, W("I see two zombies ahead, attacking them now. I'll report when one is down.")),
                       (795, 2, W("I am still facing a wall in Ch4. Need to keep turning right to find the exit to Chamber 2.")),
                       (798, 1, W("I see you on the left, the red door is on the right. We need to find the zombies first before opening the door.")),
                       (799, 2, W("I am facing a wall, turning left to find the open doorway or zombies."))]),
        ]),
    "qwen9b_heb42": dict(
        title="Qwen3.5-9B + Hebbian — seed 42",
        run=RUNS / "medium_runs/exp08_llm_9b_social_prompt/seed_42",
        exp="exp08_llm_9b_social_prompt", seed=42, lane="bonds",
        joints=[(400, (0, 1), ""), (518, (0, 2), ""), (1105, (0, 2), "anvil A1"),
                (1204, (0, 1), ""), (2036, (0, 2), ""), (2046, (1, 2), ""),
                (2048, (0, 1), "")],
        kills=[(687, 2, "a1 engaged"), (728, 1, "a2 engaged"),
               (1500, 2, "a0+a1 engaged")],
        panels=[
            dict(tag="A", ep=2, slug="anvilA1_ep2_t301",
                 title="A · Anvil A1 — a0+a2 break it together (ep 2, t=301)",
                 frames=[(0, 296), (2, 298), (0, 301), (2, 303), (2, 305),
                         (0, 306)],
                 chat=[(295, 0, "Hit the anvil now!"),
                       (297, 2, W("I am digging the anvil with you, agent_1! Keep digging to break it together.")),
                       (301, 2, W("I see you on the anvil! Let's dig together to break it.")),
                       (306, 0, W("I see a dropped sword on the ground, collecting it now before we dig the anvils."))]),
            dict(tag="B", ep=2, slug="kill_ep2_t696",
                 title="B · Mob kill — three agents engage together (ep 2, t=696)",
                 frames=[(0, 687), (2, 688), (0, 688), (0, 691), (2, 697),
                         (2, 698)],
                 chat=[(687, 0, W("I see a zombie centered on my crosshair and I am attacking it with my diamond sword.")),
                       (689, 2, W("I see you engaging the zombie, I will move forward to assist.")),
                       (691, 1, W("I see a zombie centered and am attacking it with the diamond sword to help clear the room.")),
                       (697, 2, W("I am attacking the zombie. Keep focusing on yours!"))]),
        ],
        extras=[
            dict(ep=2, slug="switch_ep2_t400",
                 title="Switch room — a0 presses, a1 freed (ep 2, t=400)",
                 frames=[(1, 400), (0, 401), (0, 402), (1, 403)],
                 chat=[(400, 1, W("I am in Chamber 3. I need to find the blue switch to help you break the anvil in Chamber 2.")),
                       (401, 0, W("I am pressing the switch to open your door.")),
                       (402, 0, W("Switch pressed for your cell! You should be free to move now.")),
                       (403, 1, W("I see agent_2, turning right to face the anvil now to help agent_0 break it."))]),
            dict(ep=2, slug="failure_phantom_anvil_ep2_t394",
                 title="Failure · Digging a phantom anvil — 12 steps of mutual confirmation, no break (ep 2, t≈388–399)",
                 frames=[(0, 390), (1, 391), (2, 390), (0, 394), (1, 395),
                         (2, 399)],
                 chat=[(390, 0, W("Digging the anvil now to break it with you.")),
                       (391, 1, W("I am centering the anvil and will Dig now to break it with you.")),
                       (394, 0, W("Anvil HP is 0, digging again to trigger the break and open Door 2.")),
                       (395, 2, W("I'm backing up to find the anvils. Let's locate them together.")),
                       (398, 2, W("I see the anvil centered. I will dig now."))]),
        ]),
}


def grab_frames(cfg, wanted):
    """(agent, ep, step) -> RGB frame; brightened for the dark MT scenes."""
    try:
        import cv2
    except ImportError:
        return {}
    out, caps = {}, {}
    for a, ep, t in wanted:
        key = (a, ep)
        if key not in caps:
            for p in (cfg["run"] / f"seed_{cfg['seed']}_agent_{a}_ep{ep}.mp4",
                      cfg["run"] / "gifs" / cfg["exp"]
                      / f"seed_{cfg['seed']}_agent_{a}_ep{ep}.mp4"):
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


def draw_markers(ax, cfg, ys):
    """Joint diamonds (tie-lined, labeled) and kill X's with assist notes."""
    for t, (i, j), note in cfg["joints"]:
        yi, yj = (ys(i, t), ys(j, t)) if callable(ys) else (ys[i], ys[j])
        ax.plot([t, t], [yi, yj], color=PAIR_C[(i, j)], lw=1.0, zorder=5)
        for y in (yi, yj):
            ax.scatter(t, y, marker="D", s=46, facecolor="white",
                       edgecolor=PAIR_C[(i, j)], linewidths=1.4, zorder=6)
        if note:
            ax.annotate(note, (t, max(yi, yj)), xytext=(0, 9),
                        textcoords="offset points", fontsize=6.5,
                        color=PAIR_C[(i, j)], ha="center", va="bottom",
                        fontweight="bold")
    for t, a, note in cfg["kills"]:
        y = ys(a, t) if callable(ys) else ys[a]
        ax.scatter(t, y, marker="X", s=46, facecolor=AGENT_C[a],
                   edgecolor="white", linewidths=0.6, zorder=6)
        ax.annotate(f"kill a{a}\n({note})", (t, y), xytext=(0, 7),
                    textcoords="offset points", fontsize=6, color=AGENT_C[a],
                    ha="center", va="bottom", linespacing=1.1)


def lane_orch(ax, run, cfg):
    """Groups, not tasks: a row per PAIR, filled while the orchestrator has
    that pair co-assigned (solid = on the SAME task, light = both busy on
    concurrent tasks). Chamber progress as background bands."""
    orch = load_orch(run)
    for ch, lo, hi in load_chamber_bands(run):
        ax.axvspan(lo, hi, color=CHAMBER_FILL.get(ch, "#ccc"), alpha=0.10,
                   linewidth=0)
    pair_y = {(0, 1): 2, (0, 2): 1, (1, 2): 0}
    team_by_pair = {q: [] for q in PAIRS}
    for tid, lo, hi, q in orch["team_iv"]:
        team_by_pair[tuple(sorted(q))].append((lo, hi))
    for q, y in pair_y.items():
        ax.axhline(y, color="#e3e7ec", lw=0.7, zorder=1)
        for lo, hi in orch["pair_iv"][q]:
            ax.broken_barh([(lo, hi - lo)], (y - 0.24, 0.48),
                           facecolors=PAIR_C[q], alpha=0.30, linewidth=0,
                           zorder=2)
        for lo, hi in team_by_pair[q]:
            ax.broken_barh([(lo, hi - lo)], (y - 0.24, 0.48),
                           facecolors=PAIR_C[q], alpha=0.95,
                           edgecolors="white", linewidth=0.3, zorder=3)
    for t, q, note in cfg["joints"]:
        y = pair_y[tuple(sorted(q))]
        ax.scatter(t, y, marker="D", s=52, facecolor="white",
                   edgecolor=PAIR_C[tuple(sorted(q))], linewidths=1.5,
                   zorder=6)
        if note:
            ax.annotate(note, (t, y), xytext=(0, 9),
                        textcoords="offset points", fontsize=6.5,
                        color=PAIR_C[tuple(sorted(q))], ha="center",
                        va="bottom", fontweight="bold")
    for t, a, note in cfg["kills"]:
        ax.scatter(t, 2.78, marker="X", s=46, facecolor=AGENT_C[a],
                   edgecolor="white", linewidths=0.6, zorder=6,
                   clip_on=False)
        ax.annotate(f"kill a{a}\n({note})", (t, 2.78), xytext=(0, 7),
                    textcoords="offset points", fontsize=6,
                    color=AGENT_C[a], ha="center", va="bottom",
                    linespacing=1.1)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0–a1", "a0–a2", "a1–a2"], fontsize=8)
    for tick, q in zip(ax.get_yticklabels(), PAIRS):
        tick.set_color(PAIR_C[q])
    ax.set_ylim(-0.55, 3.15)
    ax.set_ylabel("orchestrator\nco-assigned pairs", fontsize=9, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#9aa3ad", alpha=0.95, label="same task"),
        Patch(facecolor="#9aa3ad", alpha=0.30, label="both assigned")],
        fontsize=7, ncol=2, frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, 1.06))


def lane_events(ax, run, cfg):
    for ch, lo, hi in load_chamber_bands(run):
        ax.axvspan(lo, hi, color=CHAMBER_FILL.get(ch, "#ccc"), alpha=0.13,
                   linewidth=0)
    agent_y = {0: 2, 1: 1, 2: 0}
    for y in agent_y.values():
        ax.axhline(y, color="#d7dce3", lw=0.7, zorder=1)
    draw_markers(ax, cfg, lambda a, t: agent_y[a])
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["a0", "a1", "a2"], fontsize=8)
    ax.set_ylim(-0.8, 3.0)
    ax.set_ylabel("cooperative-track\nmilestone fires", fontsize=9, color=INK)


def _bond_marker_fn(series):
    def on_line(a, t):
        return max(float(np.interp(t, *series[q])) for q in PAIRS if a in q)
    return on_line


def lane_bonds(ax, run, cfg):
    snaps = sorted((s["step"], np.array(s["W"]))
                   for s in run.get("graph_snapshots", []) if s.get("W"))
    series = {}
    for q in PAIRS:
        xs = [s for s, _ in snaps]
        ys = [(Wm[q[0], q[1]] + Wm[q[1], q[0]]) / 2 for _, Wm in snaps]
        series[q] = (xs, ys)
        ax.plot(xs, ys, color=PAIR_C[q], lw=1.5, marker="o", markersize=2.4,
                label=f"a{q[0]}–a{q[1]}", zorder=3)
    draw_markers(ax, cfg, _bond_marker_fn(series))
    ax.set_ylabel("bond strength W̄\n(50-step snapshots)", fontsize=9,
                  color=INK)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")


def lane_bonds3f(ax, run, cfg):
    from replay_hebbian_terms import load_inputs
    from prototype_three_factor_rule import replay_three_factor
    inp = load_inputs(cfg["run"])
    Wr = replay_three_factor(inp)["W"]
    steps = np.arange(len(Wr))
    series = {}
    for q in PAIRS:
        ys = (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2
        series[q] = (steps, ys)
        ax.plot(steps, ys, color=PAIR_C[q], lw=1.5,
                label=f"a{q[0]}–a{q[1]}", zorder=3)
    for G, Wt in inp["snaps"]:
        if G - 1 < len(Wr):
            for q in PAIRS:
                ax.scatter(G - 1, (Wt[q[0], q[1]] + Wt[q[1], q[0]]) / 2, s=8,
                           color=PAIR_C[q], edgecolors="white",
                           linewidths=0.4, zorder=4)
    draw_markers(ax, cfg, _bond_marker_fn(series))
    ax.set_ylabel("bond strength W̄\n(replayed, per step)", fontsize=9,
                  color=INK)
    ax.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")


LANES = {"orch": lane_orch, "events": lane_events, "bonds": lane_bonds,
         "bonds3f": lane_bonds3f}


def lane_comm(ax, run):
    """Rolling per-pair message share (50-step window, episode-local)."""
    for q, (xs, ys) in load_msg_share(run, 50).items():
        ax.plot(xs, ys, color=PAIR_C[q], lw=1.1, zorder=3)
    ax.axhline(1 / 3, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.998, 1 / 3, " uniform", fontsize=6.5, color=MUTED,
            transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("pair message\nshare (50-step)", fontsize=9, color=INK)


def build(name, cfg):
    run = load_run(cfg["run"])
    xmax = run["_ep_bounds"][-1][1]
    allp = cfg["panels"] + cfg.get("extras", [])
    wanted = [(f[0], p["ep"], f[1]) for p in allp for f in p["frames"]]
    frames = grab_frames(cfg, wanted)
    print(f"{name}: frames {len(frames)}/{len(wanted)}")

    fig = plt.figure(figsize=(12.0, 10.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.00, 0.48, 0.78, 1.28],
                          hspace=0.30, wspace=0.12, left=0.075, right=0.985,
                          top=0.95, bottom=0.02)
    axT = fig.add_subplot(gs[0, :])
    axT.set_title(cfg["title"], loc="left", fontsize=12, color=INK, pad=8)
    LANES[cfg["lane"]](axT, run, cfg)
    axC = fig.add_subplot(gs[1, :], sharex=axT)
    lane_comm(axC, run)
    axC.set_xlabel("environment step (cumulative)", fontsize=9, labelpad=1)
    for ax in (axT, axC):
        for e, (o, _) in enumerate(run["_ep_bounds"]):
            if e:
                ax.axvline(o, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.3)
        ax.set_xlim(-15, xmax + 15)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    plt.setp(axT.get_xticklabels(), visible=False)

    single = len(cfg["panels"]) == 1
    for col, p in enumerate(cfg["panels"]):
        axF = fig.add_subplot(gs[2, :] if single else gs[2, col])
        axF.set_title(p["title"], fontsize=10, loc="left", color=INK, pad=4)
        axF.axis("off")
        n = len(p["frames"])
        cap_fs = 7.5 if (single or n <= 4) else 6.4
        for k, f in enumerate(p["frames"]):
            a, t = f[0], f[1]
            x0 = k / n
            ims = axF.inset_axes([x0 + 0.006, 0.16, 1 / n - 0.012, 0.82])
            ims.axis("off")
            fr = frames.get((a, p["ep"], t))
            if fr is not None:
                ims.imshow(fr)
            else:
                ims.set_facecolor("#eceff3")
                ims.text(0.5, 0.5, "frame\npending", fontsize=7, color=MUTED,
                         ha="center", va="center", transform=ims.transAxes)
            axF.text(x0 + 0.5 / n, 0.03, f"a{a} view · t={t}", fontsize=cap_fs,
                     color=AGENT_C[a], ha="center")
        axF.set_xlim(0, 1)
        axF.set_ylim(0, 1)

        chat = p["chat"]
        first = chat[0][1]
        if single:
            half = (len(chat) + 1) // 2
            chat_cols = [(0, chat[:half]), (1, chat[half:])]
        else:
            chat_cols = [(col, chat)]
        for ci, entries in chat_cols:
            axB = fig.add_subplot(gs[3, ci])
            axB.axis("off")
            axB.set_xlim(0, 1)
            axB.set_ylim(0, 1)
            y = 0.99
            for t, a, txt in entries:
                lines = txt.count("\n") + 1
                h = 0.075 + 0.062 * lines
                side = 0.02 if a == first else 0.14
                axB.add_patch(FancyBboxPatch((side, y - h + 0.01), 0.84,
                                             h - 0.018,
                                             boxstyle="round,pad=0.010",
                                             facecolor=AGENT_C[a] + "22",
                                             edgecolor=AGENT_C[a], lw=0.9))
                axB.text(side + 0.015, y - 0.019, f"agent {a} · t={t}",
                         fontsize=6.6, color=AGENT_C[a], va="top",
                         fontweight="bold")
                axB.text(side + 0.015, y - 0.072, txt, fontsize=7.4, color=INK,
                         va="top", linespacing=1.25)
                y -= h + 0.013
    armdir = OUT / name
    armdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(armdir / "timeline_full.png", dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {armdir}/timeline_full.png")
    export_lanes(cfg, run, armdir)
    export_assets(cfg, run, frames, armdir)


# ─── per-arm asset export ───────────────────────────────────────────────
def export_lanes(cfg, run, armdir):
    """The two graphs as standalone files."""
    xmax = run["_ep_bounds"][-1][1]
    for fname, height, draw in (
            ("lane_mechanism.png", 3.0,
             lambda ax: LANES[cfg["lane"]](ax, run, cfg)),
            ("lane_communication.png", 2.2, lambda ax: lane_comm(ax, run))):
        fig, ax = plt.subplots(figsize=(11.0, height))
        draw(ax)
        for e, (o, _) in enumerate(run["_ep_bounds"]):
            if e:
                ax.axvline(o, color=INK, lw=0.7, ls=(0, (4, 3)), alpha=0.3)
        ax.set_xlim(-15, xmax + 15)
        ax.set_xlabel("environment step (cumulative)", fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.tight_layout()
        fig.savefig(armdir / fname, dpi=150, facecolor="white")
        plt.close(fig)


def _chat_inches(entries):
    return [0.30 + 0.17 * (txt.count("\n") + 1) for _, _, txt in entries]


def story_card(cfg, p, frames, path):
    """Standalone storyline for one example: frames (large) + conversation."""
    keys = p["frames"]
    rows = [keys[i:i + 4] for i in range(0, len(keys), 4)]
    hs = _chat_inches(p["chat"])
    has_lab = any(len(f) > 2 for f in p["frames"])
    fr_h = 2.05 if has_lab else 1.72
    ch_h, ttl = sum(hs) + 0.15 * len(hs), 0.42
    fig_h = ttl + len(rows) * fr_h + ch_h + 0.25
    fig = plt.figure(figsize=(9.8, fig_h))
    fig.text(0.03, 1 - 0.28 / fig_h, re.sub(r"^[AB] · ", "", p["title"]), fontsize=12,
             color=INK, va="top")
    y = 1 - ttl / fig_h
    for row in rows:
        y -= fr_h / fig_h
        for k, f in enumerate(row):
            a, t = f[0], f[1]
            ax = fig.add_axes([0.03 + k * 0.2375, y + 0.44 / fig_h, 0.227,
                               (fr_h - 0.54) / fig_h])
            ax.axis("off")
            fr = frames.get((a, p["ep"], t))
            if fr is not None:
                ax.imshow(fr)
            else:
                ax.set_facecolor("#eceff3")
            ax.set_title(f"a{a} view · t={t}", fontsize=8.5,
                         color=AGENT_C[a], pad=2)
            if len(f) > 2:
                ax.text(0.5, -0.06, f[2], transform=ax.transAxes,
                        fontsize=7.8, color=INK, ha="center", va="top",
                        linespacing=1.2)
    axC = fig.add_axes([0.03, 0.10 / fig_h, 0.94,
                        y - 0.16 / fig_h])
    axC.axis("off")
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1)
    first = p["chat"][0][1]
    ax_h = y * fig_h - 0.26
    yc = 0.99
    for (t, a, txt), h_in in zip(p["chat"], hs):
        h = (h_in) / ax_h
        side = 0.02 if a == first else 0.12
        axC.add_patch(FancyBboxPatch((side, yc - h + 0.01), 0.86, h - 0.02,
                                     boxstyle="round,pad=0.008",
                                     facecolor=AGENT_C[a] + "22",
                                     edgecolor=AGENT_C[a], lw=1.0))
        axC.text(side + 0.013, yc - 0.05 / ax_h, f"agent {a} · t={t}",
                 fontsize=7.6, color=AGENT_C[a], va="top", fontweight="bold")
        axC.text(side + 0.013, yc - 0.21 / ax_h, txt, fontsize=8.6, color=INK,
                 va="top", linespacing=1.3)
        yc -= h + 0.12 / ax_h
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def export_assets(cfg, run, frames, armdir):
    import json
    conv = [f"# Conversations — {cfg['title']}", ""]
    for p in cfg["panels"] + cfg.get("extras", []):
        slug = p["slug"]
        fdir = armdir / "frames" / slug
        fdir.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(p["frames"], 1):
            a, t = f[0], f[1]
            fr = frames.get((a, p["ep"], t))
            if fr is not None:
                plt.imsave(fdir / f"{k:02d}_agent{a}_t{t}.png", fr)
        story_card(cfg, p, frames, armdir / f"story_{slug}.png")
        # full conversation window, curated lines marked
        curated = {(t, f"agent_{a}") for t, a, _ in p["chat"]}
        ts = [t for t, _, _ in p["chat"]] + [f[1] for f in p["frames"]]
        lo, hi = min(ts) - 5, max(ts) + 5
        conv += [f"## {re.sub(r'^[AB] · ', '', p['title'])}",
                 f"(episode {p['ep']}, steps {lo}–{hi}; lines shown in the "
                 f"figure are marked with `>`)", ""]
        mp = cfg["run"] / "episodes" / f"ep_{p['ep']:04d}" / "messages.jsonl"
        for line in open(mp, encoding="utf-8"):
            m = json.loads(line)
            if lo <= m["t"] <= hi:
                mark = ">" if (m["t"], m["sender"]) in curated else " "
                conv.append(f"{mark} t={m['t']:>3} {m['sender']} -> "
                            f"{m['receiver']}: {m['text']}")
        conv.append("")
    (armdir / "conversations.md").write_text("\n".join(conv),
                                             encoding="utf-8")
    print(f"  + lanes, {sum(len(p['frames']) for p in cfg['panels'] + cfg.get('extras', []))} frames, "
          f"story cards, conversations.md")


def lane_chambers(ax, cfg):
    """Outline lane: agent chamber trajectories as step curves (AgentLens
    'agent curve' analogue). Rows = chambers, lines = agents."""
    from replay_hebbian_terms import load_inputs
    inp = load_inputs(cfg["run"])
    T = inp["total"]
    ch = inp["chamber"].astype(float)          # (T, 3), 0 = unknown
    ch[ch <= 0] = np.nan
    for c in range(1, 6):
        ax.axhspan(c - 0.5, c + 0.5, color=CHAMBER_FILL.get(f"ch{c}", "#ccc"),
                   alpha=0.07, linewidth=0)
    for a in range(3):
        ax.plot(np.arange(T), ch[:, a] + (a - 1) * 0.12, color=AGENT_C[a],
                lw=1.2, drawstyle="steps-post", alpha=0.9,
                label=f"agent {a}")
    for t, q, note in cfg["joints"]:
        y = np.nanmax(ch[min(t, T - 1)]) if not np.isnan(ch[min(t, T - 1)]).all() else 3
        ax.scatter(t, y, marker="D", s=44, facecolor="white",
                   edgecolor=PAIR_C[tuple(sorted(q))], linewidths=1.4,
                   zorder=6)
    for t, a, note in cfg["kills"]:
        y = ch[min(t, T - 1), a]
        ax.scatter(t, y if not np.isnan(y) else 4, marker="X", s=44,
                   facecolor=AGENT_C[a], edgecolor="white", linewidths=0.6,
                   zorder=6)
    ax.set_yticks(range(1, 6))
    ax.set_yticklabels([f"Ch{c}" for c in range(1, 6)], fontsize=8)
    ax.set_ylim(0.4, 5.9)
    ax.set_ylabel("agent position\n(chamber)", fontsize=9, color=INK)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")


PAIR_CB = {(0, 1): "#0072B2", (0, 2): "#D55E00", (1, 2): "#009E73"}


def draw_card_paper(fig, card, frames, x0, w, y0, h, dW):
    """Paper-style case panel: title, 3 large snapshots with one-line
    captions, interaction summary and local bond effect."""
    from matplotlib.patches import Rectangle
    fig.patches.append(Rectangle((x0, y0), w, h, transform=fig.transFigure,
                                 facecolor="white", edgecolor="#c3c9d0",
                                 lw=1.0, zorder=1))
    fig.text(x0 + 0.008, y0 + h - 0.020, card["title"], fontsize=11,
             color=INK, va="center", fontweight="bold", zorder=4)
    n = len(card["frames"])
    fw = (w - 0.010 * (n + 1)) / n
    fr_h = 0.125
    top = y0 + h - 0.048
    for k, f in enumerate(card["frames"]):
        a, t = f[0], f[1]
        ax = fig.add_axes([x0 + 0.010 + k * (fw + 0.010), top - fr_h,
                           fw, fr_h])
        ax.axis("off")
        ax.set_zorder(3)
        fr = frames.get((a, card["ep"], t))
        if fr is not None:
            ax.imshow(fr)
        ax.text(0.5, 1.05, f"({chr(97 + k)}) t={t}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, ha="center", va="bottom")
        ax.text(0.5, -0.09, f[2], transform=ax.transAxes, fontsize=7.6,
                color=INK, ha="center", va="top", linespacing=1.15)
    yl = top - fr_h - 0.058
    fig.text(x0 + 0.008, yl, "Interaction:", fontsize=8, color=MUTED,
             va="top", zorder=4)
    fig.text(x0 + 0.052, yl, card["inter"], fontsize=8, color=INK,
             va="top", zorder=4)
    q = card["pair"]
    sign = "+" if dW >= 0.005 else ("\u2212" if dW <= -0.005 else "\u2248")
    val = f"{abs(dW):.2f}" if abs(dW) >= 0.005 else "0"
    fig.text(x0 + 0.008, yl - 0.026, "Effect:", fontsize=8, color=MUTED,
             va="top", zorder=4)
    fig.text(x0 + 0.052, yl - 0.026,
             f"\u0394W(a{q[0]}\u2013a{q[1]}) {sign}{val} over 25 steps",
             fontsize=8, color=PAIR_CB[q], va="top", fontweight="bold",
             zorder=4)


def build_agentlens(name, cfg):
    """Paper-style page: dominant-pair communication strip, one main bond
    plot, compact event strip with chamber starts, and three case panels."""
    import matplotlib as mpl
    run = load_run(cfg["run"])
    xmax = run["_ep_bounds"][-1][1]
    allp = cfg["panels"] + cfg.get("extras", []) + cfg["cards"]
    wanted = [(f[0], p["ep"], f[1]) for p in allp for f in p["frames"]]
    frames = grab_frames(cfg, wanted)
    print(f"{name}: frames {len(set(wanted))} wanted, {len(frames)} loaded")

    from replay_hebbian_terms import load_inputs
    from prototype_three_factor_rule import replay_three_factor
    inp = load_inputs(cfg["run"])
    Wr = replay_three_factor(inp)["W"]
    T = len(Wr)

    def sym(t, q):
        t = min(max(t, 0), T - 1)
        return (Wr[t, q[0], q[1]] + Wr[t, q[1], q[0]]) / 2

    with mpl.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "axes.edgecolor": "#9aa3ad",
                         "mathtext.fontset": "dejavusans"}):
        fig = plt.figure(figsize=(13.2, 7.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.16, 1.0, 0.14],
                              hspace=0.10, left=0.06, right=0.985,
                              top=0.91, bottom=0.44)

        # communication strip: dominant pair as colored bands
        axS = fig.add_subplot(gs[0])
        share = load_msg_share(run, 50)
        xs = share[PAIRS[0]][0]
        mat = np.stack([np.interp(np.arange(xmax), share[q][0], share[q][1],
                                  left=np.nan, right=np.nan) for q in PAIRS])
        dom = np.nanargmax(np.where(np.isnan(mat), -1, mat), axis=0)
        mx = np.nanmax(np.where(np.isnan(mat), 0, mat), axis=0)
        t0 = 0
        for t in range(1, xmax + 1):
            if t == xmax or dom[t] != dom[t0] or (mx[t] > 0.40) != (mx[t0] > 0.40):
                if mx[t0] > 0.40:
                    axS.axvspan(t0, t, color=PAIR_CB[PAIRS[dom[t0]]],
                                alpha=0.75, linewidth=0)
                t0 = t
        axS.set_yticks([])
        axS.set_ylabel("dominant\nmsg pair", fontsize=9, color=INK,
                       rotation=0, ha="right", va="center", labelpad=6)
        axS.set_title(cfg["title"], loc="left", fontsize=14, color=INK,
                      pad=10)

        # main bond plot
        axB = fig.add_subplot(gs[1], sharex=axS)
        steps = np.arange(T)
        for q in PAIRS:
            ys = (Wr[:, q[0], q[1]] + Wr[:, q[1], q[0]]) / 2
            axB.plot(steps, ys, color=PAIR_CB[q], lw=2.2,
                     label=f"a{q[0]}\u2013a{q[1]}", solid_capstyle="round")
        for t, q, note in cfg["joints"]:
            axB.scatter(t, sym(t, tuple(sorted(q))), marker="D", s=64,
                        facecolor="white", edgecolor=PAIR_CB[tuple(sorted(q))],
                        linewidths=1.8, zorder=6)
        ymin = float(np.nanmin(Wr[Wr > 0])) if (Wr > 0).any() else 0.05
        axB.set_ylim(max(0.05, ymin - 0.03), float(np.nanmax(Wr)) * 1.06)
        axB.grid(axis="y", color="#eef0f3", lw=0.8)
        axB.set_axisbelow(True)
        axB.set_ylabel("bond strength $\\bar{W}$", fontsize=12, color=INK)
        axB.tick_params(labelsize=10)
        axB.legend(fontsize=10, ncol=3, frameon=False, loc="upper left")

        # event strip: joints, kills, chamber starts, card anchors
        axE = fig.add_subplot(gs[2], sharex=axS)
        axE.set_yticks([])
        axE.set_ylim(0, 1)
        axE.set_ylabel("events", fontsize=9, color=INK, rotation=0,
                       ha="right", va="center", labelpad=6)
        last_lab = -1e9
        for ch, lo, hi in sorted(load_chamber_bands(run), key=lambda b: b[1]):
            n_ch = ch.replace("ch", "")
            for ax in (axS, axB, axE):
                ax.axvline(lo, color="#c9ced6", lw=0.8, zorder=0)
            if lo - last_lab > 80:
                axE.text(lo + 8, 0.80, f"Ch{n_ch}", fontsize=7, color=MUTED,
                         va="center")
                last_lab = lo
        for t, q, note in cfg["joints"]:
            axE.scatter(t, 0.42, marker="D", s=46, facecolor="white",
                        edgecolor=PAIR_CB[tuple(sorted(q))], linewidths=1.5,
                        zorder=6)
        for t, a, note in cfg["kills"]:
            axE.scatter(t, 0.42, marker="X", s=46, facecolor=AGENT_C[a],
                        edgecolor="white", linewidths=0.5, zorder=6)
        axE.set_xlabel("environment step (cumulative)", fontsize=11,
                       labelpad=2)
        axE.tick_params(labelsize=10)
        for ax in (axS, axB, axE):
            for e, (o, _) in enumerate(run["_ep_bounds"]):
                if e:
                    ax.axvline(o, color=INK, lw=0.9, ls=(0, (4, 3)),
                               alpha=0.35)
            ax.set_xlim(-15, xmax + 15)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        plt.setp(axS.get_xticklabels(), visible=False)
        plt.setp(axB.get_xticklabels(), visible=False)

        # case panels
        from matplotlib.patches import ConnectionPatch
        n = len(cfg["cards"])
        cw, gap = 0.295, 0.014
        total = n * cw + (n - 1) * gap
        left = 0.5 - total / 2
        for k, card in enumerate(sorted(cfg["cards"], key=lambda c: c["x"])):
            x0 = left + k * (cw + gap)
            t_ev = card["x"]
            dW = sym(t_ev + 25, card["pair"]) - sym(t_ev - 1, card["pair"])
            draw_card_paper(fig, card, frames, x0, cw, 0.016, 0.33, dW)
            con = ConnectionPatch(xyA=(t_ev, 0.0), coordsA=axE.transData,
                                  xyB=(x0 + cw / 2, 0.35),
                                  coordsB=fig.transFigure, color="#aab3bd",
                                  lw=1.0, ls=(0, (3, 2)), zorder=0)
            fig.add_artist(con)
            axE.scatter(t_ev, 0.10, marker="v", s=42, color="#5a6472",
                        zorder=7, clip_on=False)

        armdir = OUT / name
        armdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(armdir / "timeline_full.png", dpi=150, facecolor="white")
        plt.close(fig)
        print(f"wrote {armdir}/timeline_full.png")
    export_lanes(cfg, run, armdir)
    export_assets(cfg, run, frames, armdir)


if __name__ == "__main__":
    for name, cfg in ARMS.items():
        if "cards" in cfg:
            build_agentlens(name, cfg)
        else:
            build(name, cfg)
