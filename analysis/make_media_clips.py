#!/usr/bin/env python
"""Cut the project-page and supplementary video clips from the run recordings.

Every run writes one `.mp4` per agent per episode at one frame per environment
step (2 fps on the timeline, so step `t` is second `t/2`). Those recordings are
6-16 minutes long and 320x180; neither length nor size suits a web page, so this
cuts the moments the paper already documents into short loops.

The scenes are the ones defined in `analysis/make_final_figures.py` — the same
windows, agents and episodes that the qualitative figures are built from — plus
a chamber tour through one episode that reaches all five chambers. Windows here
are episode-local steps, as in the filmstrip specs.

Outputs per scene:

  site/assets/videos/<name>.mp4       web loop: upscaled 3x (nearest, so the
                                      voxels stay crisp), replayed at `--fps`
                                      steps/second; the agents the scene names
  site/assets/videos/<name>.jpg       poster frame
  site/assets/videos/<name>_all.mp4   every agent in the episode, side by side
                                      (up to three) or in a grid, at 2x — the
                                      page shows it behind an "all views" button

`--full` additionally copies the uncut episode recordings of the four scene
runs into `dist/supplementary/videos/episodes/`, where
`analysis/make_supplementary.py` picks them up.

Usage:
  python analysis/make_media_clips.py                 # all clips
  python analysis/make_media_clips.py --only tour     # one section
  python analysis/make_media_clips.py --list          # what would be cut
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from paths import REPO, group  # noqa: E402  (also puts siblings on sys.path)

SITE_VIDEOS = REPO / "site" / "assets" / "videos"
SUPP_MEDIA = REPO / "dist" / "supplementary" / "videos"

#: Recordings hold one frame per environment step, written at this rate.
SRC_FPS = 2


# ── the four runs the qualitative figures are built from ────────────────
RUNS = {
    "gemma_3f": dict(
        run=group("new_exp_0_gemma") / "new_exp_0_gemma_hebbian3f" / "seed_123",
        exp="new_exp_0_gemma_hebbian3f", seed=123,
        label="Gemma-E4B + Hebbian (three-factor)"),
    "orch": dict(
        run=group("orchestrator") / "new_exp_0_gemma_orch_villager_advisory" / "seed_789",
        exp="new_exp_0_gemma_orch_villager_advisory", seed=789,
        label="Gemma-E4B + centralised orchestrator"),
    "qwen9b_base": dict(
        run=group("medium_runs") / "exp02_llm_9b" / "seed_789",
        exp="exp02_llm_9b", seed=789,
        label="Qwen3.5-9B (no social layer)"),
    "qwen9b_heb": dict(
        run=group("medium_runs") / "exp08_llm_9b_social_prompt" / "seed_42",
        exp="exp08_llm_9b_social_prompt", seed=42,
        label="Qwen3.5-9B + Hebbian social module"),
    # Two more arms the candidate scanner draws on; not scene runs.
    "gemma_base": dict(
        run=group("new_exp_0_gemma") / "new_exp_0_gemma_base" / "seed_42",
        exp="new_exp_0_gemma_base", seed=42,
        label="Gemma-E4B (no social layer)"),
    "gemma_heb": dict(
        run=group("new_exp_0_gemma") / "new_exp_0_gemma_hebbian" / "seed_42",
        exp="new_exp_0_gemma_hebbian", seed=42,
        label="Gemma-E4B + Hebbian (instantaneous)"),
    # RL fine-tuned arms. "+inst." is the single-timescale rule with experience
    # sharing (social_replay); "+trace" the eligibility-trace rule (social_replay_3f).
    "qwen2b_mappo": dict(
        run=group("medium_runs") / "exp03_mappo" / "seed_42", exp="exp03_mappo", seed=42,
        label="Qwen3.5-2B MAPPO"),
    "qwen2b_ippo": dict(
        run=group("medium_runs") / "exp04_ippo" / "seed_42", exp="exp04_ippo", seed=42,
        label="Qwen3.5-2B IPPO"),
    "qwen2b_mappo_inst": dict(
        run=group("social_replay_qwen") / "exp30_mappo_hebbian_replay" / "seed_42",
        exp="exp30_mappo_hebbian_replay", seed=42, label="Qwen3.5-2B MAPPO +inst."),
    "qwen2b_ippo_inst": dict(
        run=group("social_replay_qwen") / "exp31_ippo_hebbian_replay" / "seed_42",
        exp="exp31_ippo_hebbian_replay", seed=42, label="Qwen3.5-2B IPPO +inst."),
    "qwen2b_mappo_trace": dict(
        run=group("social_replay_3f_qwen") / "exp36_mappo_hebbian_replay_3f" / "seed_42",
        exp="exp36_mappo_hebbian_replay_3f", seed=42, label="Qwen3.5-2B MAPPO +trace"),
    "qwen2b_ippo_trace": dict(
        run=group("social_replay_3f_qwen") / "exp37_ippo_hebbian_replay_3f" / "seed_42",
        exp="exp37_ippo_hebbian_replay_3f", seed=42, label="Qwen3.5-2B IPPO +trace"),
    "gemma_mappo": dict(
        run=group("gemma4") / "exp03_mappo" / "seed_42", exp="exp03_mappo", seed=42,
        label="Gemma-E4B MAPPO"),
    "gemma_ippo": dict(
        run=group("gemma4") / "exp04_ippo" / "seed_42", exp="exp04_ippo", seed=42,
        label="Gemma-E4B IPPO"),
    "gemma_mappo_inst": dict(
        run=group("social_replay_gemma4") / "exp30_mappo_hebbian_replay" / "seed_42",
        exp="exp30_mappo_hebbian_replay", seed=42, label="Gemma-E4B MAPPO +inst."),
    "gemma_ippo_inst": dict(
        run=group("social_replay_gemma4") / "exp31_ippo_hebbian_replay" / "seed_42",
        exp="exp31_ippo_hebbian_replay", seed=42, label="Gemma-E4B IPPO +inst."),
    "gemma_ippo_trace": dict(
        run=group("social_replay_3f_gemma4") / "exp37_ippo_hebbian_replay_3f" / "seed_42",
        exp="exp37_ippo_hebbian_replay_3f", seed=42, label="Gemma-E4B IPPO +trace"),
}



def _first_seed(arm_dir: Path) -> Path:
    """The lowest-numbered seed dir of an arm, or seed_42 when none is present."""
    seeds = sorted((d for d in arm_dir.glob("seed_*")
                    if d.is_dir() and d.name.split("_", 1)[1].isdigit()),
                   key=lambda d: int(d.name.split("_", 1)[1]))
    return seeds[0] if seeds else arm_dir / "seed_42"


def _register(key: str, grp: str, exp: str, label: str) -> None:
    run = _first_seed(group(grp) / exp)
    RUNS[key] = dict(run=run, exp=exp, seed=int(run.name.split("_", 1)[1]), label=label)


# Six-agent teams: the orchestrator and its matched no-graph / +inst. teams
# from the team-size sweep, and the RQ3 Phase-B transplant with its re-paired
# control. Same recorder, same layout; six views instead of three.
_register("base_n6", "agent_scaling", "scale_gemma_base_n6", "Gemma-E4B, six agents")
_register("heb_n6", "agent_scaling", "scale_gemma_hebbian_n6", "Gemma-E4B +inst., six agents")
_register("orch_n6", "agent_scaling_orch", "scale_gemma_orch_villager_n6",
          "Gemma-E4B +Central Orch., six agents")
_register("transplant", "pair_bonding", "expB_merged_transplant",
          "Transplant — co-fired partners, six agents")
_register("transplant_shuffled", "pair_bonding", "expB_merged_shuffled",
          "Transplant — re-paired, six agents")

#: The runs the documented scenes come from; the bundle ships their recordings.
SCENE_RUNS = ("gemma_3f", "orch", "qwen9b_base", "qwen9b_heb")


# ── chamber tour: one episode that reaches all five chambers ────────────
# gemma_3f episode 1 runs the full 1000 steps and enters Ch5 at step 799.
TOUR = [
    dict(name="tour_ch1", run="gemma_3f", ep=1, agents=(0,), t=(40, 100),
         chamber="Chamber 1 — Skill acquisition",
         caption="Move, dig, pick things up, kill a passive animal. Nothing "
                 "here is shared, so the team enters Chamber 2 with the "
                 "primitives already learned — and whatever the social "
                 "layer changes later, it does not change this."),
    dict(name="tour_ch2", run="gemma_3f", ep=1, agents=(0,), t=(230, 290),
         chamber="Chamber 2 — Joint action",
         caption="An anvil decays exactly as fast as one agent can damage "
                 "it, so digging alone is net zero. Two agents digging the "
                 "same anvil inside the same short window break it. The "
                 "first place where talking is not enough and the timing "
                 "has to line up."),
    dict(name="tour_ch3", run="gemma_3f", ep=1, agents=(0, 1), t=(430, 470),
         chamber="Chamber 3 — Communication under partial observability",
         caption="Agents are teleported into isolated cells wired in a "
                 "cycle: each agent's switch opens the next agent's door. "
                 "Nobody can free themselves, and nobody can see the cell "
                 "their own switch governs, so release propagates only "
                 "through targeted messages."),
    dict(name="tour_ch4", run="gemma_3f", ep=1, agents=(0,), t=(620, 680),
         chamber="Chamber 4 — Team combat",
         caption="One zombie per agent. A fatal hit costs −10 instead of "
                 "killing, which makes the chamber a rehearsal: the team "
                 "can practise focusing fire where dying is not yet final."),
    dict(name="tour_ch5", run="gemma_3f", ep=1, agents=(0, 1), t=(830, 890),
         chamber="Chamber 5 — Cooperative boss fight",
         caption="A powered zombie with three times an ordinary mob's "
                 "health — more than one agent can burn through inside the "
                 "time budget. Here death is real and permanent, and the "
                 "episode ends when the boss falls or the whole team is "
                 "down."),
]


# ── arm comparison: the scenes the paper's qualitative figures analyse ──
# `t` is the episode-local step window; `agents` is the view(s) to show,
# the first being the agent the caption follows.
SCENES = [
    # Qwen3.5-9B, no social layer
    dict(name="qwen9b_base_anvil", run="qwen9b_base", ep=2, agents=(1, 2),
         t=(198, 222), outcome="success", title="Anvil broken by agreement",
         caption="a2 states the rule — two agents, same anvil, same step — a1 "
                 "names the right-hand anvil, and both dig it together. The "
                 "anvil breaks."),
    dict(name="qwen9b_base_combat", run="qwen9b_base", ep=1, agents=(0, 1),
         t=(618, 640), outcome="success", title="Combat split by agreement",
         caption="“Keep attacking the other one.” The two zombies are "
                 "divided in one exchange, and each agent stays on the one it "
                 "claimed."),
    # (The "team scatters in Ch4" scene of the paper's figure is deliberately
    # not cut: the mobs are distant specks in a0's view, and a combat clip
    # without a visible mob shows nothing. The page uses a gated candidate
    # instead — see scan_clip_candidates.py and mob_visibility.py.)

    # Qwen3.5-9B + Hebbian social module
    dict(name="qwen9b_heb_anvil", run="qwen9b_heb", ep=2, agents=(0, 2),
         t=(289, 312), outcome="success", title="Anvil broken with a bonded partner",
         caption="Mutual “dig with me” between a0 and a2; the break "
                 "credits exactly that pair, and their bond is the one that grows."),
    dict(name="qwen9b_heb_kill3", run="qwen9b_heb", ep=2, agents=(2, 0),
         t=(683, 701), outcome="success", title="Three agents on one mob",
         caption="a0 engages, a2 announces it is moving in to assist, and all "
                 "three land hits on the same target."),
    dict(name="qwen9b_heb_phantom", run="qwen9b_heb", ep=2, agents=(0, 1),
         t=(386, 400), outcome="failure", title="Twelve steps on a spent anvil",
         caption="Both agents confirm the joint dig to each other on an anvil "
                 "that is already broken. The confirmations are perfect; nothing "
                 "fires."),

    # Gemma-E4B + Hebbian (three-factor rule)
    dict(name="gemma_3f_switch", run="gemma_3f", ep=1, agents=(0, 1),
         t=(433, 452), outcome="success", title="Switch pressed on request",
         caption="a0 reports it is stuck and asks a1 to watch the door; a0's own "
                 "press opens a1's door. The release is one-way — a0 stays locked in."),
    dict(name="gemma_3f_kill", run="gemma_3f", ep=1, agents=(1, 0),
         t=(706, 730), outcome="partial", title="Announced, not joined",
         caption="a1 calls the fight and keeps calling it. a0 reorients toward the "
                 "combat zone for twenty steps and never arrives; a1 kills alone."),
    dict(name="gemma_3f_anvil", run="gemma_3f", ep=3, agents=(1, 0),
         t=(95, 128), outcome="failure", title="Permanently ready, permanently approaching",
         caption="Every message is a correct statement of intent: ready to punch, "
                 "moving to the centre, ready to coordinate. The anvil is never struck."),

    # Centralised orchestrator
    dict(name="orch_kill", run="orch", ep=1, agents=(2, 1),
         t=(682, 698), outcome="success", title="Kill routed through the orchestrator",
         caption="a2 engages with a1 alongside. The coordination is real, and it "
                 "travels through a0, the orchestrator, rather than between them."),
    dict(name="orch_switch", run="orch", ep=3, agents=(0, 1),
         t=(399, 412), outcome="success", title="Wait, press, exit",
         caption="An explicit wait-and-press exchange: a1 holds at its door, a0 "
                 "finds and digs the switch, the door opens."),
    dict(name="orch_solokill", run="orch", ep=3, agents=(1, 0),
         t=(702, 718), outcome="failure", title="Everyone reports support; nobody arrives",
         caption="a1 takes two zombies point-blank while a0 walks into a villager "
                 "and a2 announces it is moving up. a1 fights alone and takes the damage."),
]


# ── ffmpeg ──────────────────────────────────────────────────────────────
def ffmpeg_exe() -> str:
    """The bundled imageio-ffmpeg binary, or whatever is on PATH."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(exe).exists():
            return exe
    except Exception:  # noqa: BLE001 — fall through to PATH
        pass
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("no ffmpeg: pip install imageio-ffmpeg, or put ffmpeg on PATH")
    return exe


def recording(run_key: str, agent: int, ep: int, seed: int | None = None) -> Path:
    """The mp4 for one agent-episode. Runs write it in one of two places.

    `seed` picks a sibling seed of the registered run; the default is the
    registered one.
    """
    cfg = RUNS[run_key]
    run = cfg["run"] if seed is None else cfg["run"].parent / f"seed_{seed}"
    seed = cfg["seed"] if seed is None else seed
    stem = f"seed_{seed}_agent_{agent}_ep{ep}.mp4"
    for p in (run / stem, run / "gifs" / cfg["exp"] / stem):
        if p.exists():
            return p
    raise FileNotFoundError(f"{run_key} seed {seed} a{agent} ep{ep}: no recording under {run}")


def episode_agents(run_key: str, seed: int | None, ep: int) -> list[int]:
    """Every agent in the episode, from the step log (or the recordings on disk)."""
    cfg = RUNS[run_key]
    run = cfg["run"] if seed is None else cfg["run"].parent / f"seed_{seed}"
    f = run / "episodes" / f"ep_{ep:04d}" / "step_log.csv"
    ids: set[int] = set()
    if f.exists():
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                a = (r.get("agent_id") or "").strip()
                if a.isdigit():
                    ids.add(int(a))
                if i > 600:
                    break
    if not ids:
        seed_n = cfg["seed"] if seed is None else seed
        for p in run.rglob(f"seed_{seed_n}_agent_*_ep{ep}.mp4"):
            ids.add(int(p.stem.split("_agent_")[1].split("_")[0]))
    return sorted(ids)


def grid_for(n: int) -> tuple[int, int]:
    """(columns, rows) for n views: a strip up to three, then a grid."""
    if n <= 3:
        return n, 1
    if n == 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    return 3, (n + 2) // 3


def _aid(s) -> int | None:
    """'agent_1', 'agent1' and '1' all mean agent 1."""
    if s is None:
        return None
    digits = "".join(ch for ch in str(s) if ch.isdigit())
    return int(digits) if digits else None


def messages_for(run_key: str, seed: int | None, ep: int, agents, t0: int, t1: int,
                 context: int = 3) -> list[dict]:
    """The messages the clip's agents sent or received over its window.

    Read from the episode's `messages.jsonl`; a few steps of context before
    the window are included so the exchange that led into it is visible.
    Sorted by step, then sender.
    """
    cfg = RUNS[run_key]
    run = cfg["run"] if seed is None else cfg["run"].parent / f"seed_{seed}"
    f = run / "episodes" / f"ep_{ep:04d}" / "messages.jsonl"
    agents = set(agents)
    out = []
    if not f.exists():
        return out
    for line in open(f, encoding="utf-8", errors="ignore"):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = int(m.get("t", -1))
        if not (t0 - context <= t <= t1):
            continue
        snd, rcv = _aid(m.get("sender")), _aid(m.get("receiver"))
        if snd not in agents and rcv not in agents:
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        out.append(dict(t=t, sender=snd, receiver=rcv, text=text))
    out.sort(key=lambda m: (m["t"], m["sender"] if m["sender"] is not None else -1))
    return out


def cut(exe: str, spec: dict, outdir: Path, fps: int, scale: int,
        crf: int, dry: bool, all_views: bool = True) -> dict:
    """Cut one scene to `<outdir>/<name>.mp4` plus a poster, and describe it."""
    t0, t1 = spec["t"]
    srcs = [recording(spec["run"], a, spec["ep"], spec.get("seed")) for a in spec["agents"]]
    # Recordings are one frame per step at SRC_FPS, so step t sits at t/SRC_FPS.
    start, end = t0 / SRC_FPS, (t1 + 1) / SRC_FPS
    # Replay `fps` steps per second: stretch or squeeze presentation timestamps.
    k = SRC_FPS / fps

    chains, labels = [], []
    for i in range(len(srcs)):
        chains.append(f"[{i}:v]trim=start={start:.3f}:end={end:.3f},"
                      f"setpts=(PTS-STARTPTS)*{k:.6f}[t{i}]")
        labels.append(f"[t{i}]")
    joined = "".join(labels)
    stack = f"{joined}hstack=inputs={len(srcs)}[s]" if len(srcs) > 1 else f"{joined}null[s]"
    chains.append(stack)
    chains.append(f"[s]scale=iw*{scale}:ih*{scale}:flags=neighbor[v]")
    graph = ";".join(chains)

    out = outdir / f"{spec['name']}.mp4"
    poster = outdir / f"{spec['name']}.jpg"
    # The all-agents variant: whoever was in the episode, in id order.
    agents_all, srcs_all = [], []
    if all_views:
        for a in episode_agents(spec["run"], spec.get("seed"), spec["ep"]):
            try:
                srcs_all.append(recording(spec["run"], a, spec["ep"], spec.get("seed")))
                agents_all.append(a)
            except FileNotFoundError:
                continue
        if len(agents_all) <= len(spec["agents"]):
            agents_all, srcs_all = [], []       # nothing more to show
    out_all = outdir / f"{spec['name']}_all.mp4"
    poster_all = outdir / f"{spec['name']}_all.jpg"
    meta = dict(
        name=spec["name"], run=spec["run"], arm=RUNS[spec["run"]]["label"],
        seed=spec.get("seed") or RUNS[spec["run"]]["seed"],
        episode=spec["ep"], agents=list(spec["agents"]), steps=[t0, t1],
        seconds=round((end - start) * k, 2), fps=fps,
        video=f"assets/videos/{out.name}", poster=f"assets/videos/{poster.name}",
        source=str(srcs[0].relative_to(REPO)),
        **{key: spec[key] for key in ("title", "caption", "outcome", "chamber")
           if key in spec},
    )
    meta["messages"] = messages_for(spec["run"], spec.get("seed"), spec["ep"],
                                    spec["agents"], t0, t1)
    if agents_all:
        cols, rows = grid_for(len(agents_all))
        meta.update(agents_all=agents_all, layout_all=[cols, rows],
                    video_all=f"assets/videos/{out_all.name}",
                    poster_all=f"assets/videos/{poster_all.name}")
    if dry:
        if out.exists():
            meta["bytes"] = out.stat().st_size
        return meta

    cmd = [exe, "-y", "-loglevel", "error"]
    for s in srcs:
        cmd += ["-i", str(s)]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-r", str(fps), str(out)]
    subprocess.run(cmd, check=True)
    # Poster from the middle of the window: the first frame of a scene is
    # often an agent still facing a wall.
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(out),
                    "-ss", f"{meta['seconds'] / 2:.2f}",
                    "-frames:v", "1", "-q:v", "4", str(poster)], check=True)
    meta["bytes"] = out.stat().st_size

    if agents_all:
        cols, rows = grid_for(len(agents_all))
        chains = []
        for i in range(len(srcs_all)):
            chains.append(f"[{i}:v]trim=start={start:.3f}:end={end:.3f},"
                          f"setpts=(PTS-STARTPTS)*{k:.6f}[t{i}]")
        joined = "".join(f"[t{i}]" for i in range(len(srcs_all)))
        if rows == 1:
            stack = f"{joined}hstack=inputs={len(srcs_all)}[s]" if len(srcs_all) > 1 else f"{joined}null[s]"
        else:
            # Recordings are 320x180; place each view on that grid, black where
            # a row is short.
            pos = "|".join(f"{(i % cols) * 320}_{(i // cols) * 180}" for i in range(len(srcs_all)))
            stack = f"{joined}xstack=inputs={len(srcs_all)}:layout={pos}:fill=black[s]"
        chains.append(stack)
        chains.append("[s]scale=iw*2:ih*2:flags=neighbor[v]")
        cmd = [exe, "-y", "-loglevel", "error"]
        for src in srcs_all:
            cmd += ["-i", str(src)]
        cmd += ["-filter_complex", ";".join(chains), "-map", "[v]", "-an",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-r", str(fps), str(out_all)]
        subprocess.run(cmd, check=True)
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(out_all),
                        "-ss", f"{meta['seconds'] / 2:.2f}",
                        "-frames:v", "1", "-q:v", "4", str(poster_all)], check=True)
        meta["bytes_all"] = out_all.stat().st_size
    return meta


def copy_episodes(dest: Path, keys: tuple[str, ...] = SCENE_RUNS) -> int:
    """Copy the uncut recordings of the scene runs, for the bundle."""
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for key in keys:
        cfg = RUNS[key]
        out = dest / key
        out.mkdir(exist_ok=True)
        for src in sorted(cfg["run"].rglob("*.mp4")):
            target = out / src.name
            if not target.exists() or target.stat().st_size != src.stat().st_size:
                shutil.copy2(src, target)
            total += target.stat().st_size
        (out / "SOURCE.txt").write_text(
            f"{cfg['label']}\n{cfg['run'].relative_to(REPO)}\n"
            "One frame per environment step, recorded at 2 fps.\n", encoding="utf-8")
    return total


def human(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("tour", "scenes"), help="one section only")
    ap.add_argument("--outdir", type=Path, default=SITE_VIDEOS)
    ap.add_argument("--fps", type=int, default=6, help="steps replayed per second")
    ap.add_argument("--scale", type=int, default=3, help="integer upscale factor")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--full", action="store_true",
                    help="also copy the uncut episode recordings for the bundle")
    ap.add_argument("--no-all", action="store_true", help="skip the all-agents variants")
    ap.add_argument("--list", action="store_true", help="print the plan and stop")
    args = ap.parse_args()

    specs = ([*TOUR, *SCENES] if not args.only
             else TOUR if args.only == "tour" else SCENES)
    missing = {s["run"] for s in specs if not RUNS[s["run"]]["run"].is_dir()}
    if missing:
        sys.exit("missing run directories: " + ", ".join(
            f"{k} -> {RUNS[k]['run']}" for k in sorted(missing)))

    exe = ffmpeg_exe()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = {"tour": [], "scenes": []}
    for spec in specs:
        section = "tour" if spec in TOUR else "scenes"
        meta = cut(exe, spec, args.outdir, args.fps, args.scale, args.crf, args.list,
                   all_views=not args.no_all)
        out[section].append(meta)
        size = human(meta["bytes"]) if "bytes" in meta else "—"
        print(f"  {meta['name']:<24} ep{meta['episode']} "
              f"a{'+'.join(map(str, meta['agents']))} "
              f"steps {meta['steps'][0]}–{meta['steps'][1]}  "
              f"{meta['seconds']}s  {size}")

    if args.list:
        return

    manifest = args.outdir / "manifest.json"
    prior = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    for section, items in out.items():
        if items:
            prior[section] = items
    prior["arms"] = {k: v["label"] for k, v in RUNS.items()}
    manifest.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest  {manifest.relative_to(REPO)}")

    if args.full:
        total = copy_episodes(SUPP_MEDIA / "episodes")
        print(f"episodes  {(SUPP_MEDIA / 'episodes').relative_to(REPO)}  {human(total)}")


if __name__ == "__main__":
    main()
