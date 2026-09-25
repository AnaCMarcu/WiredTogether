#!/usr/bin/env python
"""Mine the run logs for moments worth a video clip: clear successes and clear failures.

The project page needs more than the handful of scenes the qualitative figures
annotate. This walks every episode of the zero-shot arms and lists, with a
score, the moments where cooperation visibly worked or visibly did not:

  success
    anvil      a Ch2 anvil broke (m8/m9) — two contributors, always
    door       a Ch3 door opened after a *different* agent pressed a switch
    kill       a Ch4 first-mob kill with a teammate close by and talking
    ch5        the team reached Chamber 5 on merit (m24)
  failure
    ch2_stall  the chamber timed out with no anvil, while agents kept digging
               and talking about the anvils
    ch3_stall  a door never opened while agents kept talking about switches
    ch4_stall  no kill, while agents kept announcing combat
    lost       one agent turned in place for 20+ steps without moving

Chamber 4/5 candidates are kept only if a hostile mob is actually on screen in
at least one of the two views (analysis/mob_visibility.py): a combat clip with
no zombie in it shows nothing, however the messages read.

Writes paper_assets/clip_candidates/candidates.{csv,md}; `--cut` also renders
every candidate as a side-by-side clip through analysis/make_media_clips.py,
into site/assets/videos/candidates/, so they can be reviewed and thinned.

Usage:
  python analysis/scan_clip_candidates.py              # list
  python analysis/scan_clip_candidates.py --cut        # list + render
  python analysis/scan_clip_candidates.py --top 40     # keep the 40 best
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from paths import ASSETS, REPO, group  # noqa: E402  (also puts siblings on sys.path)
import mob_visibility  # noqa: E402

OUT = ASSETS / "clip_candidates"
CANDIDATE_VIDEOS = REPO / "site" / "assets" / "videos" / "candidates"

#: The zero-shot arms the page compares. Keys match make_media_clips.RUNS.
ARMS = {
    "qwen9b_base": ("medium_runs", "exp02_llm_9b", "Qwen3.5-9B"),
    "qwen9b_heb":  ("medium_runs", "exp08_llm_9b_social_prompt", "Qwen3.5-9B +inst."),
    "gemma_base":  ("new_exp_0_gemma", "new_exp_0_gemma_base", "Gemma-E4B"),
    "gemma_heb":   ("new_exp_0_gemma", "new_exp_0_gemma_hebbian", "Gemma-E4B +inst."),
    "gemma_3f":    ("new_exp_0_gemma", "new_exp_0_gemma_hebbian3f", "Gemma-E4B +trace"),
    "orch":        ("orchestrator", "new_exp_0_gemma_orch_villager_advisory",
                    "Gemma-E4B +Central Orch."),
    # RL fine-tuned arms (table 1b). Seeds without recordings are skipped.
    "qwen2b_mappo":       ("medium_runs", "exp03_mappo", "Qwen3.5-2B MAPPO"),
    "qwen2b_ippo":        ("medium_runs", "exp04_ippo", "Qwen3.5-2B IPPO"),
    "qwen2b_mappo_inst":  ("social_replay_qwen", "exp30_mappo_hebbian_replay", "Qwen3.5-2B MAPPO +inst."),
    "qwen2b_ippo_inst":   ("social_replay_qwen", "exp31_ippo_hebbian_replay", "Qwen3.5-2B IPPO +inst."),
    "qwen2b_mappo_trace": ("social_replay_3f_qwen", "exp36_mappo_hebbian_replay_3f", "Qwen3.5-2B MAPPO +trace"),
    "qwen2b_ippo_trace":  ("social_replay_3f_qwen", "exp37_ippo_hebbian_replay_3f", "Qwen3.5-2B IPPO +trace"),
    "gemma_mappo":        ("gemma4", "exp03_mappo", "Gemma-E4B MAPPO"),
    "gemma_ippo":         ("gemma4", "exp04_ippo", "Gemma-E4B IPPO"),
    "gemma_mappo_inst":   ("social_replay_gemma4", "exp30_mappo_hebbian_replay", "Gemma-E4B MAPPO +inst."),
    "gemma_ippo_inst":    ("social_replay_gemma4", "exp31_ippo_hebbian_replay", "Gemma-E4B IPPO +inst."),
    "gemma_ippo_trace":   ("social_replay_3f_gemma4", "exp37_ippo_hebbian_replay_3f", "Gemma-E4B IPPO +trace"),
    # Six-agent teams: orchestrator vs matched teams from the size sweep, and
    # the RQ3 Phase-B transplant (starts in Chamber 3) with its re-paired control.
    "base_n6":            ("agent_scaling", "scale_gemma_base_n6", "Gemma-E4B, six agents"),
    "heb_n6":             ("agent_scaling", "scale_gemma_hebbian_n6", "Gemma-E4B +inst., six agents"),
    "orch_n6":            ("agent_scaling_orch", "scale_gemma_orch_villager_n6", "Gemma-E4B +Central Orch., six agents"),
    "transplant":         ("pair_bonding", "expB_merged_transplant", "Transplant — co-fired partners, six agents"),
    "transplant_shuffled": ("pair_bonding", "expB_merged_shuffled", "Transplant — re-paired, six agents"),
}
ZERO_SHOT = ("qwen9b_base", "qwen9b_heb", "gemma_base", "gemma_heb", "gemma_3f", "orch")
SIX = ("base_n6", "heb_n6", "orch_n6", "transplant", "transplant_shuffled")


def family(arm: str) -> str:
    return "zs" if arm in ZERO_SHOT else "six" if arm in SIX else "rl"

ANVIL = {"m8_anvil_A1", "m9_anvil_B1"}
PRESS, DOOR, KILL, CH5 = "m17_switch_pressed", "m18_door_opened", "m21_first_mob_kill", "m24_enter_ch5"

WORDS = {
    "ch2": re.compile(r"anvil|dig", re.I),
    "ch3": re.compile(r"switch|door|cell|lock", re.I),
    "ch4": re.compile(r"zombie|attack|sword|fight|mob|hit", re.I),
}
NEAR = 6.0        # blocks: a teammate this close to a killer counts as present
PAD_BEFORE, PAD_AFTER = 14, 6


# ── loading ─────────────────────────────────────────────────────────────
def agent_id(s) -> int:
    """'agent1', 'agent_1' and '1' all mean agent 1."""
    return int(re.sub(r"\D", "", str(s)))


def load_episode(ep: Path) -> dict | None:
    steps = ep / "step_log.csv"
    if not steps.exists():
        return None
    rows = defaultdict(dict)       # step -> agent -> row
    for r in csv.DictReader(open(steps, encoding="utf-8", errors="ignore")):
        # RL logs occasionally carry a malformed or repeated-header row.
        if not (r.get("step") or "").strip().isdigit() or not (r.get("agent_id") or "").strip().isdigit():
            continue
        rows[int(r["step"])][int(r["agent_id"])] = r
    events = []
    ev = ep / "event_log.jsonl"
    if ev.exists():
        for line in open(ev, encoding="utf-8", errors="ignore"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "milestone":
                events.append((int(e["step"]), e["id"],
                               [agent_id(c) for c in e.get("contributors", [])]))
    msgs = []
    mf = ep / "messages.jsonl"
    if mf.exists():
        for line in open(mf, encoding="utf-8", errors="ignore"):
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs.append((int(m["t"]), agent_id(m["sender"]),
                         agent_id(m["receiver"]) if m.get("receiver") else None,
                         m.get("text", "")))
    return dict(rows=rows, events=events, msgs=msgs,
                n=max(rows) + 1 if rows else 0)


def chamber_span(data: dict, ch: str, agent: int) -> tuple[int, int] | None:
    t = [s for s, agents in data["rows"].items()
         if agent in agents and agents[agent].get("chamber") == ch]
    return (min(t), max(t)) if t else None


def msgs_in(data: dict, t0: int, t1: int, pair: tuple[int, ...] | None = None):
    out = []
    for t, snd, rcv, text in data["msgs"]:
        if t0 <= t <= t1 and (pair is None or snd in pair):
            out.append((t, snd, rcv, text))
    return out


def pos(data: dict, t: int, a: int):
    r = data["rows"].get(t, {}).get(a)
    if not r or not r.get("pos_x"):
        return None
    return float(r["pos_x"]), float(r["pos_z"])


def dist(p, q) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def densest_window(data: dict, agents: list[int], t0: int, t1: int,
                   pattern: re.Pattern, width: int = 30):
    """The `width`-step window in [t0, t1] with the most matching messages + Dig actions."""
    hits = defaultdict(int)
    for t, snd, _, text in data["msgs"]:
        if t0 <= t <= t1 and pattern.search(text):
            hits[t] += 1
    for t in range(t0, t1 + 1):
        for a in agents:
            r = data["rows"].get(t, {}).get(a)
            if r and r.get("action") == "Dig":
                hits[t] += 1
    best, best_score = None, 0
    for s in range(t0, max(t0, t1 - width) + 1):
        score = sum(hits[t] for t in range(s, s + width))
        if score > best_score:
            best, best_score = (s, min(s + width, t1)), score
    return best, best_score


# ── candidates ──────────────────────────────────────────────────────────
def scan_episode(arm: str, seed: int, ep_no: int, data: dict) -> list[dict]:
    out = []
    agents = sorted({a for r in data["rows"].values() for a in r})
    ev = data["events"]

    def cand(kind, outcome, t0, t1, who, score, note, chamber):
        who = tuple(who)
        window = msgs_in(data, t0, t1, who)
        out.append(dict(
            arm=arm, seed=seed, ep=ep_no, kind=kind, outcome=outcome, chamber=chamber,
            t0=max(0, t0), t1=min(data["n"] - 1, t1), agents=who, score=round(score, 1),
            note=note, n_msgs=len(window),
            chat=" | ".join(f"a{s}→a{r if r is not None else '?'} t{t}: {x[:70]}"
                            for t, s, r, x in window[:4]),
        ))

    # ── successes ──────────────────────────────────────────────────────
    for t, mid, who in ev:
        if mid in ANVIL and len(who) >= 2:
            cand("anvil", "success", t - 18, t + 6, who[:2], 100 + len(msgs_in(data, t - 18, t, tuple(who))),
                 f"{mid} broken by a{who[0]}+a{who[1]} at t{t}", "ch2")
        elif mid == CH5 and who and t >= 30:
            # m24 at an episode's first steps is carried-over bookkeeping, not an arrival.
            cand("ch5", "success", t - 20, t + 20, who[:2] if len(who) > 1 else (who[0],), 90,
                 f"reached Chamber 5 on merit at t{t}", "ch5")

    presses = [(t, who[0]) for t, mid, who in ev if mid == PRESS and who]
    for t, mid, who in ev:
        if mid != DOOR or not who:
            continue
        opened = who[0]
        # The press that freed this agent came from someone else, shortly before.
        cause = [(pt, pa) for pt, pa in presses if pa != opened and t - 15 <= pt <= t]
        if not cause:
            continue
        pt, presser = cause[-1]
        talk = msgs_in(data, pt - 10, t + 4, (presser, opened))
        between = [m for m in talk if m[2] in (presser, opened)]
        cand("door", "success", pt - 10, t + 6, (presser, opened),
             60 + 4 * len(between), f"a{presser} pressed at t{pt}; a{opened}'s door opened at t{t}", "ch3")

    for t, mid, who in ev:
        if mid != KILL or not who:
            continue
        killer = who[0]
        pk = pos(data, t, killer)
        near = [a for a in agents if a != killer and pk and pos(data, t, a)
                and dist(pk, pos(data, t, a)) <= NEAR]
        talk = [m for m in msgs_in(data, t - 14, t + 2) if WORDS["ch4"].search(m[3])]
        partner = near[0] if near else None
        if partner is None:
            # Nobody close: only worth it if a teammate was talking combat.
            talkers = [m[1] for m in talk if m[1] != killer]
            if not talkers:
                continue
            partner = talkers[0]
        score = 40 + 25 * bool(near) + 3 * len(talk)
        cand("kill", "success", t - 14, t + 4, (killer, partner), score,
             f"a{killer} first kill at t{t}; " + (f"a{partner} within {NEAR:.0f} blocks" if near
                                                   else f"a{partner} talking combat, not close"),
             "ch4")

    # ── failures ───────────────────────────────────────────────────────
    got = {mid for _, mid, _ in ev}
    for ch, kinds in (("ch2", ANVIL), ("ch3", {DOOR}), ("ch4", {KILL})):
        if got & kinds:
            continue
        spans = [chamber_span(data, ch, a) for a in agents]
        spans = [s for s in spans if s]
        if not spans:
            continue
        t0, t1 = min(s[0] for s in spans), max(s[1] for s in spans)
        if t1 - t0 < 40:
            continue
        win, hits = densest_window(data, agents, t0, t1, WORDS[ch])
        if not win or hits < 6:
            continue
        talkers = defaultdict(int)
        for _, snd, _, text in msgs_in(data, *win):
            if WORDS[ch].search(text):
                talkers[snd] += 1
        who = sorted(talkers, key=talkers.get, reverse=True)[:2] or agents[:2]
        if len(who) < 2:
            who = (who + [a for a in agents if a not in who])[:2]
        label = {"ch2": "no anvil broken", "ch3": "no door opened", "ch4": "no mob killed"}[ch]
        acts = {"ch2": "anvil messages and digs", "ch3": "switch/door messages and digs",
                "ch4": "combat messages and swings"}[ch]
        cand(f"{ch}_stall", "failure", win[0], win[1], who, 30 + hits,
             f"{label} in {t1 - t0 + 1} steps; {hits} {acts} in the window", ch)

    # An agent spinning in place: 20+ consecutive turns with no displacement.
    for a in agents:
        run_start, run_len, p0 = None, 0, None
        for t in sorted(data["rows"]):
            r = data["rows"][t].get(a)
            if not r:
                continue
            p = pos(data, t, a)
            turning = str(r.get("action", "")).startswith(("Turn", "Look"))
            still = p0 is not None and p is not None and dist(p, p0) < 0.5
            if turning and (run_start is None or still):
                if run_start is None:
                    run_start, p0 = t, p
                run_len += 1
            else:
                if run_len >= 20:
                    ch = data["rows"][run_start][a].get("chamber") or "?"
                    other = [b for b in agents if b != a][0]
                    cand("lost", "failure", run_start, run_start + min(run_len, 36), (a, other),
                         10 + run_len, f"a{a} turned in place for {run_len} steps", ch)
                run_start, run_len, p0 = None, 0, None
    return out


def gate_combat(cands: list[dict]) -> list[dict]:
    """Drop Ch4/Ch5 candidates in which no mob is visible in either agent's view."""
    import make_media_clips as clips  # noqa: E402

    exe = clips.ffmpeg_exe()
    kept, dropped = [], 0
    for c in cands:
        if c["chamber"] not in ("ch4", "ch5"):
            kept.append(c)
            continue
        seen, stats = False, {}
        for a in c["agents"]:
            try:
                src = clips.recording(c["arm"], a, c["ep"], c["seed"])
                ok, st = mob_visibility.check_recording(exe, src, c["t0"], c["t1"])
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            stats[f"a{a}"] = st
            seen = seen or ok
        c["mob_visible"] = seen
        c["mob_stats"] = stats
        if seen:
            kept.append(c)
        else:
            dropped += 1
    print(f"  combat gate: dropped {dropped} Ch4/Ch5 candidates with no mob in frame")
    return kept


def scan(arms: dict) -> list[dict]:
    found = []
    for arm, (grp, exp, _) in arms.items():
        root = group(grp) / exp
        if not root.is_dir():
            print(f"  {arm:<12} missing: {root}")
            continue
        n = 0
        for seed_dir in sorted(root.glob("seed_*")):
            # seed_42.oom / seed_123.failed are aborted siblings, not runs.
            if not seed_dir.name.split("_", 1)[1].isdigit():
                continue
            seed = int(seed_dir.name.split("_")[1])
            for ep in sorted(seed_dir.glob("episodes/ep_*")):
                ep_no = int(ep.name.split("_")[1])
                data = load_episode(ep)
                if data:
                    found += scan_episode(arm, seed, ep_no, data)
                    n += 1
        print(f"  {arm:<12} {n} episodes")
    return found


# ── output ──────────────────────────────────────────────────────────────
def name_of(c: dict) -> str:
    return f"{c['arm']}_s{c['seed']}_e{c['ep']}_{c['kind']}_t{c['t0']}"


def write(cands: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["name", "arm", "seed", "ep", "kind", "outcome", "chamber", "t0", "t1",
            "agents", "score", "n_msgs", "mob_visible", "note", "chat"]
    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cands:
            w.writerow({**{k: c.get(k, "") for k in cols}, "name": name_of(c),
                        "agents": "+".join(map(str, c["agents"]))})
    lines = ["# Clip candidates\n",
             "Scored moments from the zero-shot arms. Higher is clearer. "
             "`agents` is the pair whose views a side-by-side clip would show.\n"]
    for arm, (_, _, label) in ARMS.items():
        rows = [c for c in cands if c["arm"] == arm]
        if not rows:
            continue
        lines.append(f"\n## {label} (`{arm}`) — {len(rows)} candidates\n")
        lines.append("| score | kind | seed/ep | steps | agents | note | chat |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in rows:
            lines.append(f"| {c['score']} | {c['outcome']}:{c['kind']} | {c['seed']}/{c['ep']} "
                         f"| {c['t0']}–{c['t1']} | a{'+a'.join(map(str, c['agents']))} "
                         f"| {c['note']} | {c['chat'].replace('|', '/')} |")
    (OUT / "candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cut_all(cands: list[dict], fps: int, skip_existing: bool = True) -> None:
    import make_media_clips as clips  # noqa: E402

    exe = clips.ffmpeg_exe()
    CANDIDATE_VIDEOS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for c in cands:
        spec = dict(name=name_of(c), run=c["arm"], seed=c["seed"], ep=c["ep"],
                    agents=tuple(c["agents"]),
                    t=(c["t0"], c["t1"]), title=f"{c['outcome']}: {c['kind']}",
                    caption=c["note"], outcome=c["outcome"])
        done = CANDIDATE_VIDEOS / f"{spec['name']}.mp4"
        done_all = CANDIDATE_VIDEOS / f"{spec['name']}_all.mp4"
        try:
            if skip_existing and done.exists() and done.stat().st_size > 0 and done_all.exists():
                meta = clips.cut(exe, spec, CANDIDATE_VIDEOS, fps, 3, 26, True)   # describe only
            else:
                meta = clips.cut(exe, spec, CANDIDATE_VIDEOS, fps, 3, 26, False)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            print(f"  skip {spec['name']}: {str(err)[:90]}")
            for stale in (done, done.with_suffix(".jpg"), done_all, done_all.with_suffix(".jpg")):
                stale.unlink(missing_ok=True)
            continue
        meta["video"] = f"assets/videos/candidates/{done.name}"
        meta["poster"] = f"assets/videos/candidates/{spec['name']}.jpg"
        if meta.get("video_all"):
            meta["video_all"] = f"assets/videos/candidates/{done_all.name}"
            meta["poster_all"] = f"assets/videos/candidates/{spec['name']}_all.jpg"
        meta["kind"] = c["kind"]
        meta["messages"] = clips.messages_for(c["arm"], c["seed"], c["ep"], c["agents"],
                                              c["t0"], c["t1"])
        if "mob_visible" in c:
            meta["mob_visible"] = c["mob_visible"]
            meta["mob_stats"] = c["mob_stats"]
        meta.update(score=c["score"], chat=c["chat"], seed=c["seed"], chamber=c["chamber"])
        manifest.append(meta)
        print(f"  {spec['name']:<44} {meta['seconds']:>5}s  {meta['bytes'] / 1e3:>5.0f} kB")
    (CANDIDATE_VIDEOS / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=0, help="keep only the N best per outcome")
    ap.add_argument("--per-kind", type=int, default=0, help="keep only the N best per kind")
    ap.add_argument("--kinds", nargs="*", help="restrict to these kinds")
    ap.add_argument("--cut", action="store_true", help="render every candidate")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--recut", action="store_true", help="re-render clips that already exist")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the Ch4/Ch5 mob-visibility gate (fast, for listing only)")
    args = ap.parse_args()

    cands = scan(ARMS)
    # Two events in the same window (both contributors of one milestone)
    # would share a clip name; keep the first.
    seen_names, unique = set(), []
    for c in cands:
        if name_of(c) not in seen_names:
            seen_names.add(name_of(c))
            unique.append(c)
    cands = unique
    if args.kinds:
        cands = [c for c in cands if c["kind"] in args.kinds]
    cands.sort(key=lambda c: (-c["score"], c["arm"], c["seed"], c["ep"], c["t0"]))
    if not args.no_gate:
        # Gate only what could be published: the best 3x per kind, so the
        # frame sampling stays minutes, not hours.
        pool = defaultdict(int)
        head, tail = [], []
        for c in cands:
            fam = family(c["arm"])
            if c["chamber"] in ("ch4", "ch5") and args.per_kind and pool[(fam, c["kind"])] >= 3 * args.per_kind:
                tail.append(c)
            else:
                pool[(fam, c["kind"])] += 1
                head.append(c)
        cands = gate_combat(head) + [c for c in tail if c["chamber"] not in ("ch4", "ch5")]
    if args.per_kind:
        # Caps apply per family (zero-shot / RL), so the RL arms are not
        # crowded out by the zero-shot ones, which talk more and score higher.
        seen = defaultdict(int)
        kept = []
        for c in cands:
            fam = family(c["arm"])
            if seen[(fam, c["kind"])] < args.per_kind:
                kept.append(c)
                seen[(fam, c["kind"])] += 1
        cands = kept
    if args.top:
        keep = []
        for outcome in ("success", "failure"):
            keep += [c for c in cands if c["outcome"] == outcome][:args.top]
        cands = keep

    write(cands)
    by = defaultdict(int)
    for c in cands:
        by[(c["outcome"], c["kind"])] += 1
    print("\n" + "  ".join(f"{o}:{k}={n}" for (o, k), n in sorted(by.items())))
    print(f"-> {OUT.relative_to(REPO) / 'candidates.md'}  ({len(cands)} candidates)")
    if args.cut:
        cut_all(cands, args.fps, skip_existing=not args.recut)


if __name__ == "__main__":
    main()
