#!/usr/bin/env python3
"""wt_status.py - which runs finished, which died, and whether the finished
ones are healthy. Read-only; run on a login node (no budget used):

    python3 wt_status.py rebuttal comm_budget_vllm pair_bonding_3f_vllm
    python3 wt_status.py --root $WT_WORKSPACE/WiredTogether-agentworld/runs_aw aw_rebuttal
    python3 wt_status.py -v rebuttal          # one line per seed, not only problems

A run is DONE when it wrote final_metrics.json (AgentWorld: an episode_*/
summary.json). Otherwise it is QUEUED/RUNNING when squeue holds its job name
(<group>-<arm>_s<seed>, as the submit scripts name them) or its log moved in
the last --idle-hours, and DEAD after that. Finished runs are also checked for
the silent failures seen before: missing episodes, a NoOp share far above the
healthy <=8 % (an OOM turns every action into NoOp), and experience sharing
that never fired. Only runs started on/after --since are listed.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

ERRORS = [
    ("is MT down", "Luanti client/server died"),
    ("CUDA out of memory", "CUDA OOM"),
    ("Disk quota exceeded", "quota (inodes?)"),
    ("vLLM did not come up", "vLLM did not start"),
    ("cannot run an --rl arm", "vLLM + --rl refused"),
    ("Bad file descriptor", "Luanti reconnect EBADF"),
    ("Traceback", "Python traceback"),
]
FLAG_KEYS = [
    ("num_agents", "N"), ("rl_critic_mode", "critic"), ("hebbian_mode", "rule"),
    ("hebbian_gamma", "gamma"), ("hebbian_rho", "rho"), ("hebbian_eta_0", "eta0"),
    ("hebbian_eta_plus", "eta+"), ("hebbian_death_ltd", "deathLTD"),
    ("hebbian_freeze", "freeze"), ("hebbian_bond_strong", "strong"),
    ("hebbian_radius", "radius"), ("comm_reward_scale", "msgpay"),
    ("comm_budget_tokens", "budget"), ("social_module", "social"),
    ("orchestrator_variant", "orch"), ("max_steps", "steps"), ("episodes", "eps"),
]


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tail(path, nbytes=200000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def last_error(seed_dir):
    text = tail(os.path.join(seed_dir, "run.log")) + tail(os.path.join(seed_dir, "log.txt"))
    best = None
    for pat, label in ERRORS:
        i = text.rfind(pat)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, label)
    return best[1] if best else "no error in the log tail"


def noop_share(seed_dir):
    n = noop = 0
    for p in glob.glob(os.path.join(seed_dir, "episodes", "ep_*", "step_log.csv")):
        try:
            with open(p, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    n += 1
                    noop += (row.get("action") or "").strip() == "NoOp"
        except (OSError, csv.Error):
            pass
    return 100.0 * noop / n if n else None


def squeue_names():
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split("|", 1) for line in out.splitlines() if "|" in line)


def check_seed(group, arm, seed_dir, queue, idle_h):
    name = os.path.basename(seed_dir)
    cfg = load(os.path.join(seed_dir, "config.json")) or {}
    args = cfg.get("cli_args") or {}
    rec = {"seed": name.replace("seed_", ""), "args": args, "warn": []}
    if not re.fullmatch(r"seed_\d+", name):
        rec["state"] = "ASIDE"
        return rec
    fm_path = os.path.join(seed_dir, "final_metrics.json")
    aw_eps = sorted(glob.glob(os.path.join(seed_dir, "episode_*", "summary.json")))
    if os.path.exists(fm_path):
        rec["state"] = "DONE"
        fm = load(fm_path) or {}
        lens = fm.get("episode_lengths") or []
        want = args.get("episodes")
        if want and len(lens) < int(want):
            rec["warn"].append("episodes %d/%s" % (len(lens), want))
        ups = fm.get("rl_updates") or []
        if args.get("rl"):
            n_agents = int(args.get("num_agents") or cfg.get("num_agents") or 3)
            rec["warn"] += [] if ups else ["no PPO updates"]
            shared = sum(int((u.get("info") or {}).get("social_replay_n") or 0) for u in ups)
            if args.get("hebbian") and float(args.get("hebbian_rho") or 0) > 0 and shared == 0:
                rec["warn"].append("experience sharing never fired")
            rec["upd"] = len(ups) / max(n_agents, 1)
        ns = noop_share(seed_dir)
        if ns is not None:
            rec["noop"] = ns
            if ns > 15:
                rec["warn"].append("NoOp %.0f%% (healthy <=8%%)" % ns)
        return rec
    if aw_eps:
        rec["state"] = "DONE"
        rec["aw_eps"] = len(aw_eps)
        return rec
    job = "%s-%s_s%s" % (group, arm, rec["seed"])
    logs = [os.path.join(seed_dir, f) for f in ("log.txt", "run.log")]
    mtimes = [os.path.getmtime(p) for p in logs if os.path.exists(p)]
    age_h = (time.time() - max(mtimes)) / 3600 if mtimes else None
    rec["age_h"] = age_h
    if job in queue:
        rec["state"] = queue[job]
    elif age_h is not None and age_h < idle_h:
        rec["state"] = "ACTIVE?"
    else:
        rec["state"] = "DEAD"
        rec["why"] = last_error(seed_dir)
    return rec


def started(seed_dir, since):
    cfg = load(os.path.join(seed_dir, "config.json"))
    ts = str((cfg or {}).get("start_ts") or "")
    if ts:
        return ts[:10] >= since
    return time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(seed_dir))) >= since


def flag_line(args):
    parts = []
    for key, short in FLAG_KEYS:
        if key in args and args[key] not in (None, False, "none"):
            if key == "rl_critic_mode" and not args.get("rl"):
                continue
            if key.startswith("hebbian_") and not args.get("hebbian"):
                continue
            if key == "hebbian_bond_strong" and not args.get("hebbian_freeze"):
                continue
            if key == "hebbian_radius" and float(args[key]) == 5.0:
                continue
            if key == "orchestrator_variant" and not args.get("orchestrator"):
                continue
            parts.append("%s=%s" % (short, args[key]))
    if not args.get("hebbian"):
        parts.append("no-graph")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", help="run groups (default: every group under --root)")
    ap.add_argument("--root", default=os.path.join(os.environ.get("WT_WORKSPACE", os.path.expanduser("~/wt")),
                                                   "WiredTogether", "runs"))
    ap.add_argument("--since", default="2026-11-07", help="only runs started on/after this date")
    ap.add_argument("--idle-hours", type=float, default=2.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    groups = a.groups or sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))
    queue = squeue_names()
    if queue:
        print("squeue: %d of your jobs still queued/running" % len(queue))
    totals = {}
    dead = []
    for g in groups:
        gdir = os.path.join(a.root, g)
        if not os.path.isdir(gdir):
            print("\n== %s: no such group under %s" % (g, a.root))
            continue
        arms = sorted(d for d in os.listdir(gdir) if os.path.isdir(os.path.join(gdir, d)))
        rows = []
        for arm in arms:
            seeds = [s for s in glob.glob(os.path.join(gdir, arm, "seed_*"))
                     if os.path.isdir(s) and started(s, a.since)]
            if seeds:
                recs = [check_seed(g, arm, s, queue, a.idle_hours) for s in seeds]
                recs.sort(key=lambda r: int(r["seed"]) if r["seed"].isdigit() else 10 ** 9)
                rows.append((arm, recs))
        if not rows:
            continue
        print("\n== %s  (%s)" % (g, gdir))
        for arm, recs in rows:
            live = [r for r in recs if r["state"] != "ASIDE"]
            done = [r for r in live if r["state"] == "DONE"]
            for r in recs:
                totals[r["state"]] = totals.get(r["state"], 0) + 1
            other = ["%s:%s" % (r["seed"], r["state"]) for r in live if r["state"] != "DONE"]
            print("  %-34s %d/%d done  [%s]%s" % (arm, len(done), len(live), " ".join(r["seed"] for r in done),
                                                 ("  not done: " + " ".join(other)) if other else ""))
            args = next((r["args"] for r in done if r["args"]), next((r["args"] for r in recs if r["args"]), {}))
            print("  %-34s %s" % ("", flag_line(args)))
            for r in recs:
                noop = " NoOp %.1f%%" % r["noop"] if "noop" in r else ""
                upd = " %.0f PPO updates/agent" % r["upd"] if "upd" in r else ""
                if r["state"] == "DEAD":
                    age = "idle %.0f h" % r["age_h"] if r.get("age_h") is not None else "no log"
                    print("      seed_%-6s DEAD  %s, last: %s" % (r["seed"], age, r["why"]))
                    dead.append(os.path.join(gdir, arm, "seed_" + r["seed"]))
                elif r["warn"]:
                    print("      seed_%-6s %s  CHECK: %s%s%s" % (r["seed"], r["state"], "; ".join(r["warn"]), upd, noop))
                elif a.verbose:
                    print("      seed_%-6s %s%s%s%s" % (r["seed"], r["state"], upd, noop,
                                                       " %d AW episodes" % r["aw_eps"] if "aw_eps" in r else ""))
    print("\nTotals: " + ", ".join("%s %d" % kv for kv in sorted(totals.items())))
    if dead:
        stamp = time.strftime("%m%d")
        print("\nDEAD runs: move each aside (log.txt appends), then rerun the SAME submit command;")
        print("the submit scripts skip finished runs. Check the job is really gone first (squeue).")
        for d in dead:
            print("  mv %s %s.failed_%s" % (d, d, stamp))


if __name__ == "__main__":
    sys.exit(main())
