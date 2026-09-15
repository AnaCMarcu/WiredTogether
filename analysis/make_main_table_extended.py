#!/usr/bin/env python3
"""make_main_table_extended.py — tab:main_results with the newer run groups.

make_results.py emits the main table for ONE --runs-root, and its CONDITIONS
registry is keyed by directory name alone. The arms added after the medium
suite live in different run groups (gemma4, social_replay_qwen,
social_replay_gemma4) and two of them reuse the SAME directory names in two
groups (exp30/exp31 exist in both social-replay lanes), so they cannot all be
expressed in one root. This driver keeps make_results.py untouched and calls
its loader/aggregator per (group, arm), then prints one combined table.

Numbers are computed by make_results.aggregate, so the conventions are the
paper's by construction:
  * Coop. milestones = distinct team milestones in Ch2-Ch5 per episode
  * All milestones   = distinct milestones per episode (Ch1 included)
  * Task return      = team-summed task + comm_base + comm_milestone,
                       EXCLUDING hebbian_diffuse
  * mean +/- SD pooled over all episodes of all seeds

Usage:
  python analysis/make_main_table_extended.py
  python analysis/make_main_table_extended.py --tex      # LaTeX rows only
  python analysis/make_main_table_extended.py --out paper_assets/main_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import ASSETS, group  # noqa: F401  (also puts siblings on sys.path)
from make_results import aggregate, fmt, load_runs

# (label, run group, directory name). Order = row order.
# Section boundaries are drawn where `label` is None.
ROWS = [
    # ── the paper's Table 2, Qwen medium suite (runs/medium_runs) ────────
    ("LLM-2B",        "medium_runs", "exp01_llm_2b"),
    ("LLM-9B",        "medium_runs", "exp02_llm_9b"),
    ("LLM-2B+Heb",    "medium_runs", "exp07_llm_2b_social_prompt"),
    ("LLM-9B+Heb",    "medium_runs", "exp08_llm_9b_social_prompt"),
    ("IPPO",          "medium_runs", "exp04_ippo"),
    ("MAPPO",         "medium_runs", "exp03_mappo"),
    ("IPPO+Heb",      "medium_runs", "exp06_ippo_hebbian"),
    ("MAPPO+Heb",     "medium_runs", "exp05_mappo_hebbian"),
    (None, "Gemma-4 E4B RL baselines (no Hebbian)", None),
    ("Gemma IPPO",    "gemma4",      "exp04_ippo"),
    ("Gemma MAPPO",   "gemma4",      "exp03_mappo"),
    (None, "Social replay (Eq. 7 experience sharing, rho=0.3)", None),
    ("Qwen MAPPO+Heb+SR",  "social_replay_qwen",   "exp30_mappo_hebbian_replay"),
    ("Qwen IPPO+Heb+SR",   "social_replay_qwen",   "exp31_ippo_hebbian_replay"),
    ("Gemma MAPPO+Heb+SR", "social_replay_gemma4", "exp30_mappo_hebbian_replay"),
    ("Gemma IPPO+Heb+SR",  "social_replay_gemma4", "exp31_ippo_hebbian_replay"),
    (None, "Three-factor rule + signed death LTD (Hebbian 2.0)", None),
    ("LLM-2B+Heb2.0", "medium_runs", "exp34_llm_2b_three_factor_fdecay"),
    ("LLM-9B+Heb2.0", "medium_runs", "exp35_llm_9b_three_factor_fdecay"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=None,
                    help="dataset root (default: runs_from_daic)")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write table_rows.tex + summary here")
    ap.add_argument("--tex", action="store_true", help="print LaTeX only")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="EXP/seed_N")
    args = ap.parse_args()

    # Windows consoles default to cp1252, which cannot encode "±".
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    excl = frozenset(args.exclude)
    text, tex = [], ["% ── tab:main_results rows (extended) ──────────────────"]
    text.append(f"{'condition':<22} {'coop ms':>13} {'all ms':>14} "
                f"{'task return':>15} {'seeds':>6} {'eps':>5}")
    text.append("-" * 80)

    for label, grp, dir_name in ROWS:
        if label is None:
            text.append("")
            text.append(f"-- {grp} --")
            tex.append(f"\\addlinespace")
            tex.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{grp}}}}}\\\\")
            continue
        root = group(grp, runs=args.runs) if args.runs else group(grp)
        runs = load_runs(root, dir_name, excl)
        if not runs:
            text.append(f"{label:<22} {'(no runs found)':>13}   {root/dir_name}")
            continue
        a = aggregate(runs)
        if not a["task_is_decomposed"]:
            print(f"  [warn] {label}: reward decomposition missing, "
                  f"task return fell back to per_episode_returns",
                  file=sys.stderr)
        text.append(
            f"{label:<22} {a['coop'][0]:>6.1f} ± {a['coop'][1]:<4.1f} "
            f"{a['allms'][0]:>6.1f} ± {a['allms'][1]:<4.1f} "
            f"{a['task'][0]:>7.0f} ± {a['task'][1]:<5.0f} "
            f"{a['n_runs']:>5} {a['n_eps']:>5}")
        tex.append(f"{label:<22} & {fmt(a['coop'])} & {fmt(a['allms'])} & "
                   f"{fmt(a['task'], 0)} \\\\")

    out_text = "\n".join(text)
    out_tex = "\n".join(tex)
    print(out_tex if args.tex else out_text + "\n\n" + out_tex)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "table_rows.tex").write_text(out_tex + "\n",
                                                 encoding="utf-8")
        (args.out / "main_table.txt").write_text(out_text + "\n",
                                                 encoding="utf-8")
        print(f"\nwrote {args.out}/table_rows.tex", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
