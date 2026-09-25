#!/usr/bin/env python3
r"""make_final_table_credits.py - tab:final_comparison without the agent union.

The paper's Milest. % and Coop. % count DISTINCT team milestones per episode
(union over agents, make_results.episode_milestone_sets): three agents each
killing a mob count the same as one agent killing one. This script recomputes
the same rows with per-agent CREDITS instead - every (agent, milestone) pair
that fired counts - and adds a Chamber-1 column for solo exploration.

For each episode (entry-honesty filter applied, exactly as in the table):

  union   milest  = 100 * |team non-comm milestones| / 25
          coop    = 100 * |team Ch2-Ch5 milestones|  / 17
          ch1     = 100 * |team Ch1 milestones|      / 8
  credits milest  = 100 * sum_agents |non-comm milestones of agent| / (24 N + 1)
          coop    = 100 * sum_agents |Ch2-Ch5 milestones of agent|  / (17 N)
          ch1     = 100 * sum_agents |Ch1 milestones of agent|      / (7 N + 1)

Credit caps: every milestone can be credited to every agent once per episode
(Lua once=true per agent) EXCEPT m_door1_open, which fires for the single
agent whose Ch1 milestone unlocked Door 1 - hence 7N+1 for Ch1 and 24N+1 for
the 25 non-comm milestones. Anvil, gear, arena and boss milestones are
credited to every contributor, so their cap is N.

Mean +/- population SD over pooled episodes, like the table. Task return is
repeated unchanged (decomposed task + comm streams) so the rows can be
matched against tab:final_comparison. Rows are the fifteen conditions of the
pasted table, (a)-(o), taken from the make_final_table registries.

Usage:
  python analysis/make_final_table_credits.py
  python analysis/make_final_table_credits.py --out paper_assets/final_ext/credits
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

from paths import ASSETS  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
import make_results as MR
from make_final_table_extended import NEW_ROWS
from make_final_table_latex import EXTRA_ROWS

# (paper tag, printed label, registry key) in tab:final_comparison order.
TABLE_ROWS = [
    ("(a)", "Qwen3.5-2B",                    "LLM-2B"),
    ("(b)", "  +Social plasticity",          "LLM-2B+Heb2.0"),
    ("(c)", "Qwen3.5-9B",                    "LLM-9B"),
    ("(d)", "  +Social plasticity",          "LLM-9B+Heb2.0"),
    ("(e)", "Gemma-E4B",                     "Gemma-E4B"),
    ("(f)", "  +Central orch.",              "Gemma-E4B+Central Orch."),
    ("(g)", "  +Social plasticity",          "Gemma-E4B+Heb2.0"),
    ("(h)", "Qwen3.5-2B IPPO",               "IPPO"),
    ("(i)", "  +Social plasticity",          "IPPO+plast"),
    ("(j)", "Qwen3.5-2B MAPPO",              "MAPPO"),
    ("(k)", "  +Social plasticity",          "MAPPO+plast"),
    ("(l)", "Gemma-E4B IPPO",                "Gemma IPPO"),
    ("(m)", "  +Social plasticity",          "Gemma IPPO+plast"),
    ("(n)", "Gemma-E4B MAPPO",               "Gemma MAPPO"),
    ("(o)", "  +Social plasticity",          "Gemma MAPPO+plast"),
]

CH1 = frozenset(m for m, t in MR.MILESTONE_TRACK.items() if t == "ch1_solo")
COOP = frozenset(m for m, t in MR.MILESTONE_TRACK.items() if t in MR.COOP_TRACKS)
NONCOMM = frozenset(m for m, t in MR.MILESTONE_TRACK.items()
                    if t not in MR.SOCIAL_ACT_TRACKS)
CH1_MAX = len(CH1)                     # 8
SINGLE_CREDIT = {"m_door1_open"}       # fires for exactly one agent per episode
assert CH1_MAX == 8 and MR.NONCOMM_MAX == 25 and MR.COOP_MAX == 17


def caps(n_agents: int) -> dict:
    """Maximum credits per episode for N agents."""
    def cap(ms):
        return sum(1 if m in SINGLE_CREDIT else n_agents for m in ms)
    return {"milest": cap(NONCOMM), "coop": cap(COOP), "ch1": cap(CH1)}


def episode_metrics(run: dict) -> list[dict]:
    """One dict per episode with union and credit percentages."""
    per_agent = run.get("milestones_per_episode", [])
    n_agents = int(run["config"]["num_agents"])
    cp = caps(n_agents)
    n_eps = max((len(a) for a in per_agent), default=0)
    out = []
    for e in range(n_eps):
        agent_sets = [set(a[e]) for a in per_agent if e < len(a)]
        team = set().union(*agent_sets) if agent_sets else set()
        u_ms = len(team & NONCOMM)
        u_co = len(team & COOP)
        u_c1 = len(team & CH1)
        c_ms = sum(len(s & NONCOMM) for s in agent_sets)
        c_co = sum(len(s & COOP) for s in agent_sets)
        c_c1 = sum(len(s & CH1) for s in agent_sets)
        # per-agent view: each agent's own completion fraction over what ONE
        # agent can complete (25 / 17 / 8), then the mean and the spread
        # across the N agents of this episode
        a_ms = [100.0 * len(s & NONCOMM) / MR.NONCOMM_MAX for s in agent_sets]
        a_co = [100.0 * len(s & COOP) / MR.COOP_MAX for s in agent_sets]
        a_c1 = [100.0 * len(s & CH1) / CH1_MAX for s in agent_sets]
        out.append({
            "a_milest": st.fmean(a_ms) if a_ms else None,
            "a_coop": st.fmean(a_co) if a_co else None,
            "a_ch1": st.fmean(a_c1) if a_c1 else None,
            "a_milest_sd": st.pstdev(a_ms) if len(a_ms) > 1 else 0.0,
            "a_coop_sd": st.pstdev(a_co) if len(a_co) > 1 else 0.0,
            "a_ch1_sd": st.pstdev(a_c1) if len(a_c1) > 1 else 0.0,
            "a_coop_min": min(a_co) if a_co else None, "a_coop_max": max(a_co) if a_co else None,
            "a_milest_min": min(a_ms) if a_ms else None, "a_milest_max": max(a_ms) if a_ms else None,
            "n_agents": n_agents,
            "u_milest": 100.0 * u_ms / MR.NONCOMM_MAX, "u_coop": 100.0 * u_co / MR.COOP_MAX,
            "u_ch1": 100.0 * u_c1 / CH1_MAX,
            "c_milest": 100.0 * c_ms / cp["milest"], "c_coop": 100.0 * c_co / cp["coop"],
            "c_ch1": 100.0 * c_c1 / cp["ch1"],
            "credits_milest": c_ms, "credits_coop": c_co, "credits_ch1": c_c1,
            "union_milest": u_ms, "union_coop": u_co, "union_ch1": u_c1,
        })
    return out


def ms(vals, nd):
    if not vals:
        return "--"
    return "%.*f ± %.*f" % (nd, st.fmean(vals), nd, st.pstdev(vals))


def ms_seed(vals, nd):
    """mean ± SAMPLE SD (ddof=1) across seeds; the independent unit."""
    if not vals:
        return "--"
    sd = st.stdev(vals) if len(vals) >= 2 else 0.0
    return "%.*f ± %.*f" % (nd, st.fmean(vals), nd, sd)


METRICS = ["task_return", "u_milest", "c_milest", "u_coop", "c_coop", "u_ch1", "c_ch1",
           "credits_milest", "credits_coop", "credits_ch1",
           "a_milest", "a_coop", "a_ch1", "a_milest_sd", "a_coop_sd", "a_ch1_sd",
           "a_milest_min", "a_milest_max", "a_coop_min", "a_coop_max"]


def per_seed_means(eps: list[dict]) -> list[dict]:
    """One row per seed: the mean over that seed's episodes of every metric."""
    by_seed = {}
    for e in eps:
        by_seed.setdefault(e["seed"], []).append(e)
    out = []
    for seed, rows_ in sorted(by_seed.items()):
        r = {"tag": rows_[0]["tag"], "label": rows_[0]["label"], "arm": rows_[0]["arm"],
             "seed": seed, "n_eps": len(rows_)}
        for k in METRICS:
            vals = [x[k] for x in rows_ if x.get(k) is not None]
            r[k] = st.fmean(vals) if vals else None
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ASSETS / "final_ext" / "credits")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    registry = {n: (d, r) for n, d, r, *_ in list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)}
    rows, per_ep_rows, per_seed_rows = [], [], []
    for tag, label, key in TABLE_ROWS:
        if key not in registry:
            print("  [skip] %s %s: %s not in registry" % (tag, label, key), file=sys.stderr)
            continue
        dirname, root = registry[key]
        runs = MR.load_runs(Path(root), dirname)
        if not runs:
            print("  [skip] %s %s: no runs under %s/%s" % (tag, label, root, dirname),
                  file=sys.stderr)
            continue
        eps = []
        task = []
        for run in runs:
            seed = Path(run["_path"]).parent.name
            t, _ = MR.episode_task_returns(run)          # decomposed task + comm
            task.extend(t)
            for i, m in enumerate(episode_metrics(run)):
                m.update(tag=tag, label=label.strip(), arm=dirname, seed=seed, episode=i + 1,
                         task_return=t[i] if i < len(t) else None)
                eps.append(m)
        per_ep_rows.extend(eps)
        seeds = per_seed_means(eps)
        per_seed_rows.extend(seeds)
        n = eps[0]["n_agents"]
        cp = caps(n)
        seed_stats = {("s_" + k): ms_seed([s[k] for s in seeds if s[k] is not None],
                                          0 if k == "task_return" else 1)
                      for k in METRICS}
        rows.append({
            "tag": tag, "label": label.strip(), "arm": dirname, "group": Path(root).name,
            "n_seeds": len(runs), "n_eps": len(eps), "n_agents": n,
            "task": ms(task, 0),
            "u_milest": ms([e["u_milest"] for e in eps], 1),
            "c_milest": ms([e["c_milest"] for e in eps], 1),
            "u_coop": ms([e["u_coop"] for e in eps], 1),
            "c_coop": ms([e["c_coop"] for e in eps], 1),
            "u_ch1": ms([e["u_ch1"] for e in eps], 1),
            "c_ch1": ms([e["c_ch1"] for e in eps], 1),
            "credits_milest": ms([e["credits_milest"] for e in eps], 2),
            "credits_coop": ms([e["credits_coop"] for e in eps], 2),
            "credits_ch1": ms([e["credits_ch1"] for e in eps], 2),
            "cap_milest": cp["milest"], "cap_coop": cp["coop"], "cap_ch1": cp["ch1"],
            **seed_stats,
        })

    # ── report ──
    L = ["# tab:final_comparison recomputed with per-agent credits", "",
         "union = distinct team milestones (the printed table); credits = every (agent, milestone) "
         "that fired. Denominators: union 25 / 17 / 8; credits 24N+1 / 17N / 7N+1 "
         "(N=3 → 73 / 51 / 22). Mean ± population SD over pooled episodes; entry-honesty filter "
         "applied. Task return = decomposed task + comm, unchanged.", ""]
    hdr = ["", "Condition", "seeds", "eps", "Task return",
           "Milest.% union", "Milest.% credits", "Coop.% union", "Coop.% credits",
           "Ch1% union", "Ch1% credits"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join("---" for _ in hdr) + "|")
    for r in rows:
        L.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["label"], r["n_seeds"], r["n_eps"], r["task"],
            r["u_milest"], r["c_milest"], r["u_coop"], r["c_coop"], r["u_ch1"], r["c_ch1"]))
    L += ["", "## Raw credit counts per episode (mean ± SD), with caps", ""]
    hdr2 = ["", "Condition", "non-comm credits (cap)", "coop credits (cap)", "Ch1 credits (cap)"]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "|".join("---" for _ in hdr2) + "|")
    for r in rows:
        L.append("| %s | %s | %s (%d) | %s (%d) | %s (%d) |" % (
            r["tag"], r["label"], r["credits_milest"], r["cap_milest"],
            r["credits_coop"], r["cap_coop"], r["credits_ch1"], r["cap_ch1"]))
    L += ["", "## Seed-level: each run's 3 episodes averaged first, then mean ± SAMPLE SD across seeds",
          "",
          "Same metrics; the unit is the seed (the independent draw), so the SD is the spread "
          "between runs, not between episodes.", ""]
    hdr3 = ["", "Condition", "seeds", "Task return",
            "Milest.% union", "Milest.% credits", "Coop.% union", "Coop.% credits",
            "Ch1% union", "Ch1% credits"]
    L.append("| " + " | ".join(hdr3) + " |")
    L.append("|" + "|".join("---" for _ in hdr3) + "|")
    for r in rows:
        L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["label"], r["n_seeds"], r["s_task_return"],
            r["s_u_milest"], r["s_c_milest"], r["s_u_coop"], r["s_c_coop"],
            r["s_u_ch1"], r["s_c_ch1"]))
    L += ["", "## Per-agent (no union): each agent's own completion %, averaged over agents, "
          "then episodes, then mean ± sample SD across seeds", "",
          "Per-agent denominators are what ONE agent can complete: 25 / 17 / 8. "
          "`agent SD` = population SD across the N agents within an episode (how unevenly the "
          "milestones are spread), averaged over episodes and seeds. `min/max agent` = the "
          "least and most productive agent's % in an episode, averaged the same way.", ""]
    hdr5 = ["", "Condition", "seeds", "Milest.% per agent", "agent SD", "min/max agent",
            "Coop.% per agent", "agent SD", "min/max agent", "Ch1% per agent", "agent SD"]
    L.append("| " + " | ".join(hdr5) + " |")
    L.append("|" + "|".join("---" for _ in hdr5) + "|")
    for r in rows:
        L.append("| %s | %s | %d | %s | %s | %s / %s | %s | %s | %s / %s | %s | %s |" % (
            r["tag"], r["label"], r["n_seeds"],
            r["s_a_milest"], r["s_a_milest_sd"].split(" ± ")[0],
            r["s_a_milest_min"].split(" ± ")[0], r["s_a_milest_max"].split(" ± ")[0],
            r["s_a_coop"], r["s_a_coop_sd"].split(" ± ")[0],
            r["s_a_coop_min"].split(" ± ")[0], r["s_a_coop_max"].split(" ± ")[0],
            r["s_a_ch1"], r["s_a_ch1_sd"].split(" ± ")[0]))
    L += ["", "## Per-seed means (one row per run)", ""]
    hdr4 = ["", "Condition", "seed", "eps", "Task", "Milest.% u", "Milest.% c",
            "Coop.% u", "Coop.% c", "Ch1% u", "Ch1% c", "coop credits/ep"]
    L.append("| " + " | ".join(hdr4) + " |")
    L.append("|" + "|".join("---" for _ in hdr4) + "|")
    for s in per_seed_rows:
        L.append("| %s | %s | %s | %d | %.0f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f |" % (
            s["tag"], s["label"], s["seed"], s["n_eps"], s["task_return"] or 0,
            s["u_milest"], s["c_milest"], s["u_coop"], s["c_coop"], s["u_ch1"], s["c_ch1"],
            s["credits_coop"]))
    report = "\n".join(L) + "\n"
    print(report)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "final_table_credits_per_seed.csv").open("w", newline="",
                                                             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        w.writeheader(); w.writerows(per_seed_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_table_credits.md").write_text(report, encoding="utf-8")
    with (args.out / "final_table_credits_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (args.out / "final_table_credits_per_episode.csv").open("w", newline="",
                                                                encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep_rows[0].keys()))
        w.writeheader(); w.writerows(per_ep_rows)
    print("wrote %s/{final_table_credits.md, final_table_credits_rows.csv, "
          "final_table_credits_per_episode.csv}" % args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
