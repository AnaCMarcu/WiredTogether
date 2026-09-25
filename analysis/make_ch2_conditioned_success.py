#!/usr/bin/env python3
r"""make_ch2_conditioned_success.py - Ch3/Ch4 milestone success GIVEN Ch2 progress.

Question: in the Hebbian 2.0 "+trace" runs (three-factor eligibility-trace
rule, ``hebbian_mode == three_factor``), how often do the Chamber 3 and
Chamber 4 milestones fire in the episodes where the team completed at least
one Chamber 2 milestone?

Why condition on Ch2: the Ch2->Ch3 transition is time-gated. When the Ch2
timeout fires, Python writes ``ch2_force_teleport.txt`` and Lua relocates
every agent into a Ch3 isolation cell WITHOUT opening Door 2
(doors.lua, ``_per_chamber_timeout_specs``). Ch3/Ch4 milestones can then be
earned by teams that did nothing in Ch2, so an unconditioned Ch3/Ch4
completion rate mixes "progressed by playing" with "was dropped into Ch3".

"Ch2 completed" (``--ch2-def``):
  any   (default) at least one Ch2-track milestone fired in the episode
        (m8_anvil_A1, m9_anvil_B1, m14_sword_equipped, m15_chestplate_equipped).
  door  both anvils broken - m8_anvil_A1 AND m9_anvil_B1 - which is exactly
        the condition under which anvil.lua starts the Door 2 countdown.
  all   all four Ch2-track milestones.

Unit of analysis: the EPISODE. Every episode re-locks the doors and resets
the milestone state (init.lua reset -> init_doors / reset_milestone_state),
and the timeout teleport is one-shot per episode, so Ch2 progress is an
episode property. A run-level view (a run qualifies when at least one of
its episodes qualifies; a milestone counts when it fired in one of that
run's qualifying episodes) is emitted alongside.

Milestone accounting is make_results': the entry-honesty filter is applied
(m20_enter_ch4 needs m18_door_opened in the same episode), team milestone
sets are the union over agents, and only ``seed_<digits>`` directories are
read (``seed_42.failed_copy`` and ``seed_<N>.oom`` parkings are skipped).

Arms are DISCOVERED, not listed: every ``<rq>/<group>/<arm>/seed_<N>`` under
runs_from_daic/ (smoke/ and the ``_*`` symlink views excluded) whose
final_metrics.json reports ``hebbian_mode == "three_factor"``. The report
carries ``hebbian_death_ltd`` per arm because the checklist distinguishes
Hebbian 2.0 proper (death LTD 0.05) from the death-LTD-free ``*_three_factor``
/ ``hebbian3f`` arms; ``--require-death-ltd`` keeps only the former.

Outputs (``--out``, default paper_assets/ch2_conditioned/):
  ch2_conditioned_episodes.csv   per arm: n runs/episodes, qualifying
                                 episodes, per-milestone % (Ch3, Ch4) among
                                 qualifying episodes and among all episodes
  ch2_conditioned_runs.csv       the run-level view
  ch2_conditioned_per_episode.csv  one row per (arm, seed, episode): the raw
                                 flags behind both tables
  ch2_conditioned_report.md      the same, readable

Usage:
  python analysis/make_ch2_conditioned_success.py
  python analysis/make_ch2_conditioned_success.py --ch2-def door --require-death-ltd
  python analysis/make_ch2_conditioned_success.py --family zero_shot --family rl_social_replay
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from paths import ASSETS, RUNS  # noqa: F401  (also puts siblings on sys.path)

import make_results as MR

SEED_RE = re.compile(r"^seed_(\d+)$")

CH2 = [m for m, t in MR.MILESTONE_TRACK.items() if t == "ch2_anvils"]
CH3 = [m for m, t in MR.MILESTONE_TRACK.items() if t == "ch3_switches"]
CH4 = [m for m, t in MR.MILESTONE_TRACK.items() if t == "ch4_combat"]
DOOR2_MILESTONES = ("m8_anvil_A1", "m9_anvil_B1")   # anvil.lua Door 2 trigger

CH2_DEFS = {
    "any": lambda s: any(m in s for m in CH2),
    "door": lambda s: all(m in s for m in DOOR2_MILESTONES),
    "all": lambda s: all(m in s for m in CH2),
}
CH2_DEF_TEXT = {
    "any": "at least one Ch2-track milestone fired in the episode "
           "(m8_anvil_A1 / m9_anvil_B1 / m14_sword_equipped / m15_chestplate_equipped)",
    "door": "both anvils broken in the episode (m8_anvil_A1 AND m9_anvil_B1) - "
            "the Door 2 trigger in anvil.lua",
    "all": "all four Ch2-track milestones fired in the episode",
}

# Group -> experiment family, for pooling. Anything not listed pools under
# its own group name.
FAMILY = {
    "medium_runs": "zero_shot",
    "new_exp_0_gemma": "zero_shot",
    "pareto_gemma4_3f": "zero_shot_pareto_size",
    "pareto_social_3f": "zero_shot_pareto_interval",
    "agent_scaling_3f": "zero_shot_agent_scaling",
    "social_replay_3f_qwen": "rl_social_replay",
    "social_replay_3f_gemma4": "rl_social_replay",
    "cofiring_bidi_3f": "cofiring",
    "pair_bonding_3f": "transplant_phaseB",
}


def short_label(mid: str) -> str:
    pid, name = MR.PAPER_LABEL.get(mid, (mid, mid))
    return "%s %s" % (pid.replace("\\", ""), name.replace("$", "").replace("\\", ""))


# --- Discovery ------------------------------------------------------------
def _cli_args(fm_path: Path, cfg_path: Path) -> dict:
    try:
        d = json.loads(fm_path.read_text(encoding="utf-8"))
        ca = (d.get("config") or {}).get("cli_args")
        if isinstance(ca, dict) and ca:
            return ca
    except (OSError, json.JSONDecodeError):
        pass
    try:
        d = json.loads(cfg_path.read_text(encoding="utf-8"))
        ca = d.get("cli_args")
        if isinstance(ca, dict):
            return ca
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _hebbian_mode(fm_path: Path, cfg_path: Path) -> str:
    ca = _cli_args(fm_path, cfg_path)
    if not ca.get("hebbian", True) and "hebbian_mode" not in ca:
        return "none"
    return str(ca.get("hebbian_mode", "unknown"))


def discover_arms(runs_root: Path):
    """[(rq, group, arm, group_path)] for every arm holding a three_factor run."""
    out = []
    for rq in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if rq.name in ("smoke", "cluster_logs"):
            continue
        for grp in sorted(p for p in rq.iterdir() if p.is_dir()):
            if grp.name.startswith("_") or grp.is_symlink():
                continue
            for arm in sorted(p for p in grp.iterdir() if p.is_dir()):
                if arm.is_symlink():
                    continue
                modes = set()
                for seed in arm.iterdir():
                    if not SEED_RE.match(seed.name):
                        continue
                    fm = seed / "final_metrics.json"
                    if not fm.is_file():
                        continue
                    modes.add(_hebbian_mode(fm, seed / "config.json"))
                if "three_factor" in modes:
                    if len(modes) > 1:
                        print("  [warn] %s/%s mixes hebbian modes %s - keeping the "
                              "three_factor seeds only" % (grp.name, arm.name, sorted(modes)),
                              file=sys.stderr)
                    out.append((rq.name, grp.name, arm.name, grp))
    return out


def load_arm(group_path: Path, arm: str) -> list[dict]:
    """make_results.load_runs restricted to seed_<digits> and three_factor."""
    runs = []
    for run in MR.load_runs(group_path, arm, frozenset()):
        seed_dir = Path(run["_path"]).parent
        m = SEED_RE.match(seed_dir.name)
        if not m:
            continue
        ca = (run.get("config") or {}).get("cli_args") or {}
        if ca.get("hebbian_mode") != "three_factor":
            continue
        run["_seed"] = int(m.group(1))
        run["_cli"] = ca
        runs.append(run)
    return runs


# --- Per-episode flags ----------------------------------------------------
def episode_rows(rq, group, arm, runs, ch2_ok):
    rows = []
    for run in runs:
        sets_ = MR.episode_milestone_sets(run)
        for e, s in enumerate(sets_):
            row = {
                "rq": rq, "group": group, "arm": arm, "seed": run["_seed"],
                "episode": e + 1, "ch2_done": int(ch2_ok(s)),
                "n_ch2": sum(1 for m in CH2 if m in s),
                "n_ch3": sum(1 for m in CH3 if m in s),
                "n_ch4": sum(1 for m in CH4 if m in s),
            }
            for m in CH2 + CH3 + CH4:
                row[m] = int(m in s)
            rows.append(row)
    return rows


def pct(num, den):
    return (100.0 * num / den) if den else float("nan")


def fmt_pct(v):
    return "--" if v != v else "%.0f" % v


def summarise_episodes(rows):
    """Per-milestone and per-chamber percentages over one set of episode rows."""
    n = len(rows)
    cond = [r for r in rows if r["ch2_done"]]
    k = len(cond)
    out = {"n_eps": n, "n_eps_ch2": k, "pct_eps_ch2": pct(k, n)}
    for tag, mids in (("ch3", CH3), ("ch4", CH4)):
        for m in mids:
            out["%s|ch2" % m] = pct(sum(r[m] for r in cond), k)
            out["%s|all" % m] = pct(sum(r[m] for r in rows), n)
        nk = "n_%s" % tag
        out["%s_mean_pct|ch2" % tag] = (st.fmean(100.0 * r[nk] / len(mids) for r in cond)
                                        if k else float("nan"))
        out["%s_any|ch2" % tag] = pct(sum(1 for r in cond if r[nk] > 0), k)
        out["%s_all|ch2" % tag] = pct(sum(1 for r in cond if r[nk] == len(mids)), k)
        out["%s_mean_pct|all" % tag] = (st.fmean(100.0 * r[nk] / len(mids) for r in rows)
                                        if n else float("nan"))
        out["%s_any|all" % tag] = pct(sum(1 for r in rows if r[nk] > 0), n)
        out["%s_all|all" % tag] = pct(sum(1 for r in rows if r[nk] == len(mids)), n)
    return out


def summarise_runs(rows):
    """Run-level view: a run qualifies if any episode qualifies."""
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["group"], r["arm"], r["seed"])].append(r)
    n = len(by_run)
    qual = {key: [r for r in eps if r["ch2_done"]] for key, eps in by_run.items()}
    qual = {key: eps for key, eps in qual.items() if eps}
    k = len(qual)
    out = {"n_runs": n, "n_runs_ch2": k, "pct_runs_ch2": pct(k, n)}
    for tag, mids in (("ch3", CH3), ("ch4", CH4)):
        nk = "n_%s" % tag
        for m in mids:
            out["%s|ch2" % m] = pct(sum(1 for eps in qual.values() if any(r[m] for r in eps)), k)
            out["%s|all" % m] = pct(sum(1 for eps in by_run.values() if any(r[m] for r in eps)), n)
        out["%s_all|ch2" % tag] = pct(
            sum(1 for eps in qual.values() if any(r[nk] == len(mids) for r in eps)), k)
        out["%s_any|ch2" % tag] = pct(
            sum(1 for eps in qual.values() if any(r[nk] > 0 for r in eps)), k)
    return out


# --- Emission -------------------------------------------------------------
def arm_meta(runs):
    ca = runs[0]["_cli"] if runs else {}
    return {
        "n_agents": ca.get("num_agents"),
        "max_steps": ca.get("max_steps"),
        "death_ltd": ca.get("hebbian_death_ltd"),
        "rl": bool(ca.get("rl")) if "rl" in ca else None,
    }


def md_table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--out", type=Path, default=ASSETS / "ch2_conditioned")
    ap.add_argument("--ch2-def", choices=list(CH2_DEFS), default="any")
    ap.add_argument("--require-death-ltd", action="store_true",
                    help="keep only arms with hebbian_death_ltd > 0 (Hebbian 2.0 proper)")
    ap.add_argument("--family", action="append", default=None,
                    help="restrict to these families (repeatable); see FAMILY in the source")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ch2_ok = CH2_DEFS[args.ch2_def]
    arms = discover_arms(args.runs)
    if not arms:
        print("no three_factor arms found under %s" % args.runs, file=sys.stderr)
        return 1

    all_rows, arm_tables, run_tables = [], [], []
    for rq, group, arm, gpath in arms:
        fam = FAMILY.get(group, group)
        if args.family and fam not in args.family:
            continue
        runs = load_arm(gpath, arm)
        if not runs:
            continue
        meta = arm_meta(runs)
        if args.require_death_ltd and not (meta["death_ltd"] or 0) > 0:
            print("  [skip] %s/%s: death_ltd=%s" % (group, arm, meta["death_ltd"]),
                  file=sys.stderr)
            continue
        rows = episode_rows(rq, group, arm, runs, ch2_ok)
        for r in rows:
            r["family"] = fam
            r["death_ltd"] = meta["death_ltd"]
        all_rows.extend(rows)
        e = summarise_episodes(rows)
        e.update(group=group, arm=arm, family=fam, **meta)
        r_ = summarise_runs(rows)
        r_.update(group=group, arm=arm, family=fam, **meta)
        arm_tables.append(e)
        run_tables.append(r_)

    if not all_rows:
        print("no runs matched the filters", file=sys.stderr)
        return 1

    # Pooled rows: per family, and overall.
    fams = sorted({r["family"] for r in all_rows})
    pooled_e, pooled_r = [], []
    for fam in fams + ["ALL"]:
        sub = all_rows if fam == "ALL" else [r for r in all_rows if r["family"] == fam]
        e = summarise_episodes(sub)
        e.update(group="(pooled)", arm=fam, family=fam)
        r_ = summarise_runs(sub)
        r_.update(group="(pooled)", arm=fam, family=fam)
        pooled_e.append(e)
        pooled_r.append(r_)

    # -- CSV --
    args.out.mkdir(parents=True, exist_ok=True)
    ep_fields = ["family", "rq", "group", "arm", "seed", "episode", "death_ltd", "ch2_done",
                 "n_ch2", "n_ch3", "n_ch4"] + CH2 + CH3 + CH4
    with (args.out / "ch2_conditioned_per_episode.csv").open("w", newline="",
                                                             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ep_fields)
        w.writeheader()
        w.writerows(all_rows)

    def write_summary(name, tables, lead):
        keys = lead + [k for k in tables[0] if k not in lead]
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for t in tables:
                w.writerow({k: ("" if (isinstance(v, float) and v != v) else
                                (round(v, 2) if isinstance(v, float) else v))
                            for k, v in t.items()})

    lead_e = ["family", "group", "arm", "n_agents", "max_steps", "death_ltd", "rl",
              "n_eps", "n_eps_ch2", "pct_eps_ch2"]
    lead_r = ["family", "group", "arm", "n_agents", "max_steps", "death_ltd", "rl",
              "n_runs", "n_runs_ch2", "pct_runs_ch2"]
    write_summary("ch2_conditioned_episodes.csv", arm_tables + pooled_e, lead_e)
    write_summary("ch2_conditioned_runs.csv", run_tables + pooled_r, lead_r)

    # -- Markdown report --
    L = ["# Ch3/Ch4 milestone success conditioned on Ch2 progress",
         "",
         "Runs: every arm with `hebbian_mode == three_factor` (Hebbian 2.0 / +trace) under "
         "`%s`%s." % (args.runs, " - death_ltd > 0 only" if args.require_death_ltd else ""),
         "",
         "Qualifying episode (`--ch2-def %s`): %s." % (args.ch2_def, CH2_DEF_TEXT[args.ch2_def]),
         "",
         "Unit: episode (doors and milestone state reset per episode; the Ch2 timeout teleport "
         "is one-shot per episode). Percentages are over qualifying episodes unless a column "
         "says `all`. Entry-honesty filter from make_results applied.",
         ""]

    nr = {(t["group"], t["arm"]): t["n_runs"] for t in run_tables}

    L += ["## A. Ch2 progress per arm", ""]
    hdr = ["family", "arm", "N", "steps", "death_ltd", "runs", "episodes",
           "qualifying eps", "%"]
    body = []
    for t in arm_tables:
        body.append([t["family"], t["arm"], t["n_agents"], t["max_steps"], t["death_ltd"],
                     nr[(t["group"], t["arm"])], t["n_eps"], t["n_eps_ch2"],
                     fmt_pct(t["pct_eps_ch2"])])
    for t in pooled_e:
        body.append(["**%s**" % t["family"], "(pooled)", "", "", "", "", t["n_eps"],
                     t["n_eps_ch2"], fmt_pct(t["pct_eps_ch2"])])
    L += [md_table(hdr, body), ""]

    for tag, mids, title in (("ch3", CH3, "Chamber 3 - switches"),
                             ("ch4", CH4, "Chamber 4 - combat")):
        L += ["## B%s. %s: %% of qualifying episodes in which each milestone fired"
              % (tag[-1], title), ""]
        hdr = ["family", "arm", "qual. eps"] + [short_label(m) for m in mids] + \
              ["mean %% of %s milestones" % tag.upper(), "any", "all 4"]
        body = []
        for t in arm_tables + pooled_e:
            if t["n_eps_ch2"] == 0:
                continue
            name = t["arm"] if t["group"] != "(pooled)" else "**%s (pooled)**" % t["family"]
            body.append([t["family"], name, t["n_eps_ch2"]]
                        + [fmt_pct(t["%s|ch2" % m]) for m in mids]
                        + [fmt_pct(t["%s_mean_pct|ch2" % tag]), fmt_pct(t["%s_any|ch2" % tag]),
                           fmt_pct(t["%s_all|ch2" % tag])])
        L += [md_table(hdr, body), ""]

    L += ["## C. Conditioning effect: chamber-level completion, qualifying vs all episodes", "",
          "`mean %` = mean over episodes of (milestones fired / 4); `any` = % of episodes "
          "with at least one milestone of that chamber.", ""]
    hdr = ["family", "arm", "qual. / all eps",
           "Ch3 mean % (qual.)", "Ch3 mean % (all)", "Ch3 any (qual.)", "Ch3 any (all)",
           "Ch4 mean % (qual.)", "Ch4 mean % (all)", "Ch4 any (qual.)", "Ch4 any (all)"]
    body = []
    for t in arm_tables + pooled_e:
        name = t["arm"] if t["group"] != "(pooled)" else "**%s (pooled)**" % t["family"]
        body.append([t["family"], name, "%d / %d" % (t["n_eps_ch2"], t["n_eps"]),
                     fmt_pct(t["ch3_mean_pct|ch2"]), fmt_pct(t["ch3_mean_pct|all"]),
                     fmt_pct(t["ch3_any|ch2"]), fmt_pct(t["ch3_any|all"]),
                     fmt_pct(t["ch4_mean_pct|ch2"]), fmt_pct(t["ch4_mean_pct|all"]),
                     fmt_pct(t["ch4_any|ch2"]), fmt_pct(t["ch4_any|all"])])
    L += [md_table(hdr, body), ""]

    L += ["## D. Run-level view", "",
          "A run qualifies when at least one of its episodes qualifies; a milestone counts "
          "when it fired in one of that run's qualifying episodes.", ""]
    hdr = ["family", "arm", "runs", "qual. runs"] + [short_label(m) for m in CH3 + CH4]
    body = []
    for t in run_tables + pooled_r:
        if t["n_runs_ch2"] == 0:
            continue
        name = t["arm"] if t["group"] != "(pooled)" else "**%s (pooled)**" % t["family"]
        body.append([t["family"], name, t["n_runs"], t["n_runs_ch2"]]
                    + [fmt_pct(t["%s|ch2" % m]) for m in CH3 + CH4])
    L += [md_table(hdr, body), ""]

    L += ["## E. The qualifying episodes", ""]
    hdr = ["family", "arm", "seed", "ep", "Ch2 milestones", "Ch3 milestones", "Ch4 milestones"]
    body = []
    for r in all_rows:
        if not r["ch2_done"]:
            continue
        body.append([r["family"], r["arm"], r["seed"], r["episode"],
                     ",".join(m.split("_")[0] for m in CH2 if r[m]),
                     ",".join(m.split("_")[0] for m in CH3 if r[m]) or "-",
                     ",".join(m.split("_")[0] for m in CH4 if r[m]) or "-"])
    L += [md_table(hdr, body), ""]

    report = "\n".join(L)
    (args.out / "ch2_conditioned_report.md").write_text(report, encoding="utf-8")
    print(report)
    print("\nwrote %s/{ch2_conditioned_report.md, ch2_conditioned_episodes.csv, "
          "ch2_conditioned_runs.csv, ch2_conditioned_per_episode.csv}" % args.out,
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
