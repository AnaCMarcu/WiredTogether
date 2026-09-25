#!/usr/bin/env python3
r"""make_counterfactual_scan.py - where the bonds succeed and the plan fails.

Enumerates every (seed, episode, cooperative milestone) cell that BOTH the
central-orchestrator arm and a Hebbian arm ran (same seeds -> same maps), and
classifies it:

  HEB_ONLY   the Hebbian team fired the milestone in that episode, the
             orchestrator team did not.  Sub-flag PLANNED when the
             orchestrator had actually assigned agents to a task targeting
             that milestone in that episode - the counterfactual proper:
             same map, same episode, one team told to do it and failing,
             the other doing it with no instruction.
  ORCH_ONLY / BOTH / NEITHER   the other three quadrants, kept so the
             appendix table is the full grid rather than a highlight reel.

Only Ch3 (switch/door) and Ch4 (first mob kill) ever fire for either arm.
The anvil milestones the orchestrator proposes thousands of times are never
completed by ANY arm, so they are reported separately as shared failures,
not as counterfactuals.

Per HEB_ONLY+PLANNED cell the scan records what the figure will need:
  orchestrator  agent-steps spent on tasks targeting the milestone, how
                many agents, whether two were ever co-assigned, and how the
                tasks ended (freed_timeout / freed_unreachable / ...)
  hebbian       within-episode fire step, contributing agents, whether it
                was a joint fire, the contributing pair's bond W at the fire
                and its change over +-25 steps, and the message traffic
                between the contributors in the 15 steps before.
and a legibility score that prefers a clear orchestrator failure (many
agent-steps, nothing to show), a clean Hebbian success (joint fire, bond
consolidation, early in the episode) and frames on disk for both runs.

Outputs (under --out):
  cells.csv      every cell, all fields
  cases.md       HEB_ONLY cells ranked by score (+ the ORCH_ONLY ones)
  summary.md     quadrant counts per milestone and the seed x episode grid

Usage:
  python analysis/make_counterfactual_scan.py                 # Hebbian 2.0 (si3f8)
  python analysis/make_counterfactual_scan.py --heb heb1      # reward-modulated
  python analysis/make_counterfactual_scan.py --heb heb3f     # three-factor, no LTD
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from paths import ASSETS, group  # noqa: F401  (also puts siblings on sys.path)

import make_directive_timelines as mdt
import make_coordination_timelines as mct
from make_results import COOP_TRACKS, MILESTONE_TRACK

ORCH = group("orchestrator") / "new_exp_0_gemma_orch_villager_advisory"
HEB_ARMS = {
    "heb2": (group("pareto_social_3f") / "new_exp_0_gemma_si3f8",
             "Gemma-E4B + Hebbian 2.0 (si3f8)"),
    "heb1": (group("new_exp_0_gemma") / "new_exp_0_gemma_hebbian",
             "Gemma-E4B + Hebbian 1.0 (reward-modulated)"),
    "heb3f": (group("new_exp_0_gemma") / "new_exp_0_gemma_hebbian3f",
              "Gemma-E4B + three-factor, no death LTD"),
}
COOP_IDS = sorted(m for m, t in MILESTONE_TRACK.items() if t in COOP_TRACKS)
PRE_WINDOW = 15      # steps of chat before the fire that count as run-up
DW_HALF = 25         # +-steps for the local bond change


# ─── per-run loaders ────────────────────────────────────────────────────
def seeds_of(run_root: Path) -> list[int]:
    return sorted(int(p.parent.name[5:]) for p in run_root.glob("seed_*/final_metrics.json")
                  if p.parent.name[5:].isdigit())


def has_frames(run_dir: Path, seed: int, ep: int) -> bool:
    exp = run_dir.parent.name
    for a in range(3):
        f = f"seed_{seed}_agent_{a}_ep{ep}.mp4"
        if not ((run_dir / f).exists() or (run_dir / "gifs" / exp / f).exists()):
            return False
    return True


def fires_by_ep(run: dict) -> dict:
    """{(ep, mid): {"step": cum, "t": within-ep, "agents": [..], "joint": bool}}
    for the FIRST fire of each coop milestone in each episode, grouping
    contributors that fired on the same lua_step (a joint fire)."""
    bounds = run["_ep_bounds"]

    def ep_of(step):
        for e, (lo, hi) in enumerate(bounds, 1):
            if lo <= step < hi:
                return e, step - lo
        return None, None

    by_lua = defaultdict(lambda: {"agents": set(), "step": 10 ** 9})
    for ev in run.get("milestone_events", []):
        mid = ev["milestone_id"]
        if MILESTONE_TRACK.get(mid) not in COOP_TRACKS:
            continue
        k = (mid, ev["lua_step"])
        by_lua[k]["agents"].add(mdt.agent_id(ev["contributor"]))
        by_lua[k]["step"] = min(by_lua[k]["step"], int(ev["step"]))
    out = {}
    for (mid, _), d in by_lua.items():
        ep, t = ep_of(d["step"])
        if ep is None:
            continue
        key = (ep, mid)
        if key not in out or d["step"] < out[key]["step"]:
            out[key] = {"step": d["step"], "t": t,
                        "agents": sorted(a for a in d["agents"] if a is not None),
                        "joint": len(d["agents"]) >= 2}
    return out


def orch_plan(run: dict) -> dict:
    """{ep: {mid: {agent_steps, agents, coassigned, ends, n_alloc}}} - what
    the orchestrator actually assigned toward each coop milestone."""
    od = run["_run_dir"] / "orchestrator"
    bounds = run["_ep_bounds"]
    targets = defaultdict(lambda: defaultdict(set))       # ep -> tid -> {mid}
    for line in (od / "dag.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        for t in d.get("tasks", []):
            for m in (t.get("milestones") or []):
                if MILESTONE_TRACK.get(m) in COOP_TRACKS:
                    targets[d["episode"]][t["id"]].add(m)
    orch = mdt.load_orch(run)
    plan = defaultdict(lambda: defaultdict(lambda: {
        "agent_steps": 0, "agents": set(), "coassigned": False,
        "ends": Counter(), "n_alloc": 0, "first_t": None}))
    for aid, tid, t0, t1 in orch["intervals"]:
        for ep, (lo, hi) in enumerate(bounds, 1):
            if lo <= t0 < hi:
                for m in targets[ep].get(tid, ()):
                    p = plan[ep][m]
                    p["agent_steps"] += t1 - t0
                    p["agents"].add(aid)
                    p["n_alloc"] += 1
                    ft = t0 - lo
                    p["first_t"] = ft if p["first_t"] is None else min(p["first_t"], ft)
    for tid, lo, hi, q in orch["team_iv"]:
        for ep, (b0, b1) in enumerate(bounds, 1):
            if b0 <= lo < b1:
                for m in targets[ep].get(tid, ()):
                    plan[ep][m]["coassigned"] = True
    for line in (od / "assignments.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r["reason"].startswith("freed"):
            continue
        for m in targets[r["episode"]].get(r["task_id"], ()):
            plan[r["episode"]][m]["ends"][r["reason"]] += 1
    return plan, targets


def orch_proposals_by_ep(run: dict) -> dict:
    """{(ep, mid): [outcome, ...]} for multi-agent proposals targeting mid."""
    out = defaultdict(list)
    for p in mct.load_orch_proposals(run)["proposals"]:
        for m in p["milestones"]:
            if MILESTONE_TRACK.get(m) in COOP_TRACKS:
                out[(p["episode"], m)].append(p["outcome"])
    return out


def bond_context(run: dict, step: int, agents: list[int]) -> dict:
    """Contributing pair's W at the fire and its change over +-DW_HALF."""
    series = mdt.load_bonds(run)
    if not series:
        return {"pair": None, "W": float("nan"), "dW": float("nan")}
    if len(agents) >= 2:
        pair = tuple(sorted(agents[:2]))
    else:
        a = agents[0] if agents else 0
        cands = [q for q in mdt.PAIRS if a in q]
        pair = max(cands, key=lambda q: float(np.interp(step, *series[q])))
    xs, ys = series[pair]
    w = float(np.interp(step, xs, ys))
    dw = float(np.interp(step + DW_HALF, xs, ys) - np.interp(step - DW_HALF, xs, ys))
    return {"pair": pair, "W": w, "dW": dw}


def chat_runup(run_dir: Path, step: int, agents: list[int]) -> int:
    """Messages exchanged AMONG the contributing agents in the run-up."""
    log = json.loads((run_dir / "communication_log.json").read_text(encoding="utf-8"))
    who = set(agents)
    n = 0
    for t, snd, _txt, rcv in log:
        if step - PRE_WINDOW <= t <= step:
            s, r = mdt.agent_id(snd), mdt.agent_id(rcv)
            if len(who) >= 2 and s in who and r in who:
                n += 1
            elif len(who) < 2 and (s in who or r in who):
                n += 1
    return n


# ─── scoring ────────────────────────────────────────────────────────────
def legibility(cell: dict) -> float:
    if cell["class"] != "HEB_ONLY":
        return 0.0
    s = 0.0
    s += math.log1p(cell["orch_agent_steps"]) * (1.0 if cell["orch_planned"] else 0.0)
    s += 1.0 if cell["orch_coassigned"] else 0.0
    s += 2.0 if cell["heb_joint"] else 0.0
    s += 1.5 if (cell["heb_dW"] == cell["heb_dW"] and cell["heb_dW"] > 0.01) else 0.0
    s += 1.0 - cell["heb_t"] / max(cell["ep_len"], 1)
    s += 0.5 * min(cell["heb_chat"], 6) / 6
    s += 1.0 if (cell["frames_orch"] and cell["frames_heb"]) else 0.0
    return round(s, 3)


# ─── main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heb", default="heb2", choices=sorted(HEB_ARMS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    heb_root, heb_title = HEB_ARMS[args.heb]
    out = args.out or (ASSETS / "counterfactual" / args.heb)
    out.mkdir(parents=True, exist_ok=True)

    seeds = sorted(set(seeds_of(ORCH)) & set(seeds_of(heb_root)))
    print(f"orchestrator vs {heb_title}: matched seeds {seeds}")

    cells, shared_fail = [], Counter()
    for seed in seeds:
        o_dir, h_dir = ORCH / f"seed_{seed}", heb_root / f"seed_{seed}"
        o_run, h_run = mdt.load_run(o_dir), mdt.load_run(h_dir)
        o_f, h_f = fires_by_ep(o_run), fires_by_ep(h_run)
        plan, targets = orch_plan(o_run)
        props = orch_proposals_by_ep(o_run)
        n_ep = min(len(o_run["_ep_bounds"]), len(h_run["_ep_bounds"]))
        for ep in range(1, n_ep + 1):
            planned_mids = set(plan[ep]) | {m for tid in targets[ep] for m in targets[ep][tid]}
            for mid in COOP_IDS:
                of, hf = o_f.get((ep, mid)), h_f.get((ep, mid))
                p = plan[ep].get(mid)
                if of is None and hf is None:
                    if mid in planned_mids:
                        shared_fail[mid] += 1
                    cls = "NEITHER"
                elif of is None:
                    cls = "HEB_ONLY"
                elif hf is None:
                    cls = "ORCH_ONLY"
                else:
                    cls = "BOTH"
                if cls == "NEITHER" and mid not in planned_mids:
                    continue          # nothing happened, nothing planned: skip row
                h_ep_len = h_run["_ep_bounds"][ep - 1][1] - h_run["_ep_bounds"][ep - 1][0]
                cell = {
                    "seed": seed, "ep": ep, "mid": mid,
                    "milestone": mdt.milestone_label(mid), "paper_id": mdt.paper_id(mid),
                    "class": cls, "ep_len": h_ep_len,
                    "orch_fired_t": of["t"] if of else "",
                    "orch_agents": "".join(str(a) for a in of["agents"]) if of else "",
                    "orch_planned": bool(p and p["agent_steps"] > 0) or (mid in planned_mids),
                    "orch_agent_steps": p["agent_steps"] if p else 0,
                    "orch_n_agents": len(p["agents"]) if p else 0,
                    "orch_coassigned": bool(p and p["coassigned"]),
                    "orch_first_t": p["first_t"] if p and p["first_t"] is not None else "",
                    "orch_ends": ";".join(f"{k}:{v}" for k, v in sorted(p["ends"].items())) if p else "",
                    "orch_proposals": len(props.get((ep, mid), [])),
                    "orch_prop_outcomes": ";".join(sorted(props.get((ep, mid), []))),
                    "heb_fired_t": hf["t"] if hf else "",
                    "heb_step": hf["step"] if hf else "",
                    "heb_agents": "".join(str(a) for a in hf["agents"]) if hf else "",
                    "heb_joint": bool(hf and hf["joint"]),
                    "heb_t": hf["t"] if hf else h_ep_len,
                    "heb_pair": "", "heb_W": float("nan"), "heb_dW": float("nan"),
                    "heb_chat": 0,
                    "frames_orch": has_frames(o_dir, seed, ep),
                    "frames_heb": has_frames(h_dir, seed, ep),
                }
                if hf:
                    bc = bond_context(h_run, hf["step"], hf["agents"])
                    cell["heb_pair"] = f"a{bc['pair'][0]}-a{bc['pair'][1]}" if bc["pair"] else ""
                    cell["heb_W"], cell["heb_dW"] = bc["W"], bc["dW"]
                    cell["heb_chat"] = chat_runup(h_dir, hf["step"], hf["agents"])
                cell["score"] = legibility(cell)
                cells.append(cell)

    # ---- write ---------------------------------------------------------
    fields = list(cells[0].keys()) if cells else []
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in c.items()})

    quad = defaultdict(Counter)
    for c in cells:
        quad[c["mid"]][c["class"]] += 1
    L = [f"# Counterfactual scan: orchestrator vs {heb_title}", "",
         f"Matched seeds: {', '.join(map(str, seeds))}  ({len(seeds)} seeds x 3 episodes).",
         "", "## Quadrants per milestone (cells = seed x episode)", "",
         "| milestone | HEB_ONLY | BOTH | ORCH_ONLY | NEITHER (planned) |",
         "|---|---|---|---|---|"]
    for mid in COOP_IDS:
        q = quad.get(mid)
        if not q:
            continue
        L.append(f"| {mdt.milestone_label(mid)} | **{q['HEB_ONLY']}** | {q['BOTH']} | "
                 f"{q['ORCH_ONLY']} | {q['NEITHER']} |")
    L += ["", "## Shared failures (orchestrator planned it, nobody fired it)", ""]
    for mid, n in shared_fail.most_common():
        L.append(f"- {mdt.milestone_label(mid)}: planned in {n} episodes, fired by neither arm")
    L += ["", "## Seed x episode grid", "",
          "Per cell: one letter per milestone that fired in that episode - "
          "`H` Hebbian only, `O` orchestrator only, `B` both; `.` = neither. "
          "Order: " + ", ".join(mdt.paper_id(m) for m in COOP_IDS
                                if any(c["mid"] == m for c in cells)), ""]
    used = [m for m in COOP_IDS if any(c["mid"] == m for c in cells)]
    L.append("| seed | ep 1 | ep 2 | ep 3 |")
    L.append("|---|---|---|---|")
    sym = {"HEB_ONLY": "H", "ORCH_ONLY": "O", "BOTH": "B", "NEITHER": "."}
    idx = {(c["seed"], c["ep"], c["mid"]): c for c in cells}
    for seed in seeds:
        row = [str(seed)]
        for ep in (1, 2, 3):
            row.append("`" + "".join(sym[idx[(seed, ep, m)]["class"]] if (seed, ep, m) in idx else "."
                                      for m in used) + "`")
        L.append("| " + " | ".join(row) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    ranked = sorted((c for c in cells if c["class"] == "HEB_ONLY"),
                    key=lambda c: -c["score"])
    C = [f"# HEB_ONLY cells, ranked by legibility - {heb_title}", "",
         "| # | seed | ep | milestone | score | orch: agent-steps / agents / co-assigned / ends "
         "| heb: t / agents / joint / W / dW / chat | frames |",
         "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(ranked, 1):
        C.append(
            f"| {i} | {c['seed']} | {c['ep']} | {c['milestone']} | {c['score']:.2f} | "
            f"{c['orch_agent_steps']} / {c['orch_n_agents']} / {'y' if c['orch_coassigned'] else 'n'} / "
            f"{c['orch_ends'] or '-'} | "
            f"t={c['heb_fired_t']} / a{c['heb_agents']} / {'joint' if c['heb_joint'] else 'solo'} / "
            f"{c['heb_pair']} W={c['heb_W']:.2f} dW={c['heb_dW']:+.2f} / {c['heb_chat']} msgs | "
            f"{'both' if c['frames_orch'] and c['frames_heb'] else ('orch' if c['frames_orch'] else ('heb' if c['frames_heb'] else 'none'))} |")
    orch_only = [c for c in cells if c["class"] == "ORCH_ONLY"]
    C += ["", f"## ORCH_ONLY cells ({len(orch_only)}) - the other direction, for balance", ""]
    for c in orch_only:
        C.append(f"- seed {c['seed']} ep {c['ep']} {c['milestone']}: orchestrator fired at "
                 f"t={c['orch_fired_t']} (a{c['orch_agents']}); Hebbian did not")
    (out / "cases.md").write_text("\n".join(C) + "\n", encoding="utf-8")

    # ---- console -------------------------------------------------------
    tot = Counter(c["class"] for c in cells)
    print(f"cells: {len(cells)}  HEB_ONLY={tot['HEB_ONLY']}  BOTH={tot['BOTH']}  "
          f"ORCH_ONLY={tot['ORCH_ONLY']}  NEITHER(planned)={tot['NEITHER']}")
    print(f"shared failures (planned, never fired by either): "
          + ", ".join(f"{mdt.paper_id(m)}x{n}" for m, n in shared_fail.most_common()))
    print(f"\ntop {args.top} HEB_ONLY by legibility:")
    print("  %-5s %-3s %-22s %-6s %-30s %-38s %s" % (
        "seed", "ep", "milestone", "score", "orch (steps/agents/co/ends)", "heb (t/agents/joint/W/dW/chat)", "frames"))
    for c in ranked[:args.top]:
        print("  %-5d %-3d %-22s %-6.2f %-30s %-38s %s" % (
            c["seed"], c["ep"], c["milestone"][:22], c["score"],
            f"{c['orch_agent_steps']}/{c['orch_n_agents']}/{'y' if c['orch_coassigned'] else 'n'}/{c['orch_ends'] or '-'}"[:30],
            f"t={c['heb_fired_t']} a{c['heb_agents']} {'J' if c['heb_joint'] else 's'} W={c['heb_W']:.2f} dW={c['heb_dW']:+.2f} c={c['heb_chat']}",
            "both" if c["frames_orch"] and c["frames_heb"] else ("orch" if c["frames_orch"] else ("heb" if c["frames_heb"] else "none"))))
    print(f"\nwrote {out}/cells.csv, cases.md, summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
