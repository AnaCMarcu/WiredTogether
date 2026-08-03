#!/usr/bin/env python3
"""Compute-budget table from wandb. Run where wandb is installed + logged in:
    pip install wandb
    wandb login
    python wandb_compute_budget.py
Pulls per-run _runtime (clean process wall-clock) + GPU model from run.metadata
for project anamarcu/medium_wired_together, aggregates into the paper's groups,
and prints LaTeX-ready rows. Over-budget (non-finished) runs are reported
separately and excluded from the means, matching the table caption.
"""
import statistics
from collections import defaultdict
import wandb

# 1000-step suite (main + topology) and 2000-step horizon ablation live in
# separate wandb projects. NOTE: the 2000-step exp11 runs collided into the
# 1000-step project (same run-id), so the horizon row is sourced ONLY from
# medium2k_wired_together (exp09/exp10, seeds 42+123); exp11-2000 is reported
# separately in the response if needed.
MAIN_PROJECT = "anamarcu/medium_wired_together"
HORIZON_PROJECT = "anamarcu/medium2k_wired_together"
WALL_BUDGET_H = 168.0

# exp-dir prefix -> table group. Order = table row order. The horizon group is
# matched by PROJECT, not prefix (see group_of).
GROUPS = [
    ("LLM-only (2B)",            ["exp01_llm_2b"]),
    ("LLM-only (9B)",            ["exp02_llm_9b"]),
    ("LLM + Hebbian (2B)",       ["exp07_llm_2b_social_prompt"]),
    ("LLM + Hebbian (9B)",       ["exp08_llm_9b_social_prompt"]),
    ("IPPO / MAPPO",             ["exp03_mappo", "exp04_ippo"]),
    ("IPPO / MAPPO + Hebbian",   ["exp05_mappo_hebbian", "exp06_ippo_hebbian"]),
    ("Frozen topologies (9B)",   ["exp09_llm_9b_allied_all",
                                  "exp10_llm_9b_allied_pair",
                                  "exp11_llm_9b_allied_none"]),
    ("Horizon ablation (2000)",  []),   # matched by project below
]


def group_of(run_name, project):
    if project == HORIZON_PROJECT:
        return "Horizon ablation (2000)"
    for label, prefixes in GROUPS:
        if any(run_name.startswith(p) or p in run_name for p in prefixes):
            return label
    return None


def main():
    api = wandb.Api()
    runs = []
    for proj in (MAIN_PROJECT, HORIZON_PROJECT):
        try:
            got = list(api.runs(proj, per_page=500))
            print(f"fetched {len(got)} runs from {proj}")
            runs += [(r, proj) for r in got]
        except Exception as e:
            print(f"  [warn] could not fetch {proj}: {e}")
    print()

    agg = defaultdict(lambda: {"h": [], "gpu": set(), "over": 0, "n": 0})
    for r, proj in runs:
        g = group_of(r.name, proj)
        if g is None:
            print(f"  [ungrouped] {r.name}")
            continue
        meta = r.metadata or {}
        gpus = meta.get("gpu_devices") or ([meta["gpu"]] if meta.get("gpu") else [])
        for gd in gpus:
            name = gd.get("name") if isinstance(gd, dict) else gd
            if name:
                agg[g]["gpu"].add(str(name).replace("NVIDIA ", "").strip())
        rt = (r.summary.get("_runtime") if r.summary else None)
        h = (rt / 3600.0) if rt else None
        finished = (r.state == "finished")
        agg[g]["n"] += 1
        if h is not None and finished and h <= WALL_BUDGET_H:
            agg[g]["h"].append(h)
        elif not finished or (h is not None and h > WALL_BUDGET_H):
            agg[g]["over"] += 1

    print("% group & GPU & Runs & Wall-clock/run (h) & Total (GPU-h)")
    grand = 0.0
    for label, _ in GROUPS:
        a = agg[label]
        hs = a["h"]
        if hs:
            m, s = statistics.fmean(hs), (statistics.pstdev(hs) if len(hs) > 1 else 0.0)
            total = sum(hs)
            grand += total
            cell = f"${m:.1f} \\pm {s:.1f}$"
        else:
            m = s = total = 0.0
            cell = "$\\cdot$"
        gpu = ", ".join(sorted(a["gpu"])) or "?"
        over = f"  (+{a['over']} over-budget, excluded)" if a["over"] else ""
        print(f"{label:26s} & {gpu} & {len(hs)} & {cell} & {total:.0f}{over}")
    print(f"\\midrule\nTotal & & & & ${grand:.0f}$ GPU-h")
    print("\n% NOTE: Horizon row = exp09/exp10 2000-step runs only "
          "(medium2k_wired_together). The 2000-step exp11 runs collided into "
          "the 1000-step project's run-ids and are not counted here.")


if __name__ == "__main__":
    main()
