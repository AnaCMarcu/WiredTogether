#!/usr/bin/env python3
r"""make_steps_table_latex.py - tab:steps_to_milestone, generated.

Median environment step of FIRST completion per milestone, per condition,
with the number of completing episodes as a subscript. "--" means never
completed.

Columns are exactly the zero-shot VLM panel of tab:final_comparison, read
from make_final_table_latex.PANELS, so the two tables can never disagree
about which conditions exist or what they are called. The previous
hand-written version of this table predated the Heb2.0 runs: it carried a
"Gemma-E4B trace precursor" column (new_exp_0_gemma_hebbian3f, the
three-factor arm with eta_minus_d = 0, i.e. NOT Heb2.0) in place of the
real +plast. arms, and pooled 3 seeds there against 6 elsewhere.

make_final_table.collect() keeps only the four headline step columns, so
this reads make_results.aggregate() directly, which returns all 30.

Usage:
  python analysis/make_steps_table_latex.py --out paper_assets/final_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import ASSETS  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft
import make_results as MR
from make_final_table_extended import NEW_ROWS
from make_final_table_latex import EXTRA_ROWS, PANELS

# (section heading, [milestone ids]) - the paper's chamber grouping.
SECTIONS = [
    ("Chamber 1 --- solo skill acquisition",
     ["m1_move_5", "m2_dig_3_any", "m3_pickup_3", "m4_dig_5_wood",
      "m5_kill_1_animal", "m6_kill_2_animals", "m7_dig_3_stone",
      "m_door1_open"]),
    ("Chamber 2 --- cooperative resource acquisition",
     ["m8_anvil_A1", "m9_anvil_B1", "m14_sword_equipped",
      "m15_chestplate_equipped"]),
    ("Chamber 3 --- communication under partial obs.",
     ["m16_enter_cell", "m17_switch_pressed", "m18_door_opened",
      "m19_all_in_communal"]),
    ("Chamber 4 --- team combat",
     ["m20_enter_ch4", "m21_first_mob_kill", "m22_all_mobs_killed",
      "m23_all_alive_ch4"]),
    ("Chamber 5 --- cooperative boss fight",
     ["m24_enter_ch5", "m25_first_boss_dmg", "m26_boss_half_hp",
      "m27_boss_defeated", "m28_all_alive_bonus"]),
    ("Communication track",
     ["m_comm_ch1", "m_comm_ch2", "m_comm_ch3", "m_comm_ch4", "m_comm_ch5"]),
]

# Paper ID and short name per milestone. Kept here rather than derived so the
# printed IDs match the environment appendix (tab:milestone_schedule) exactly.
ROWS_META = {
    "m1_move_5": ("M1", r"Move $>$5 from spawn"),
    "m2_dig_3_any": ("M2", "Dig 3 any"),
    "m3_pickup_3": ("M3", "Pick up 3 items"),
    "m4_dig_5_wood": ("M4", "Dig 5 wood"),
    "m5_kill_1_animal": ("M5", "Kill 1 animal"),
    "m6_kill_2_animals": ("M6", "Kill 2 animals"),
    "m7_dig_3_stone": ("M7", "Dig 3 stone"),
    "m_door1_open": (r"M\_door1", "Door 1 unlocked"),
    "m8_anvil_A1": ("M8", "First anvil break"),
    "m9_anvil_B1": ("M9", "Second anvil break"),
    "m14_sword_equipped": ("M10", "Sword equipped"),
    "m15_chestplate_equipped": ("M11", "Chestplate equipped"),
    "m16_enter_cell": ("M12", "Enter isolation cell"),
    "m17_switch_pressed": ("M13", "Press switch"),
    "m18_door_opened": ("M14", "Door opened"),
    "m19_all_in_communal": ("M15", "All in communal room"),
    "m20_enter_ch4": ("M16", "Enter Chamber 4"),
    "m21_first_mob_kill": ("M17", "First mob kill"),
    "m22_all_mobs_killed": ("M18", "All mobs killed"),
    "m23_all_alive_ch4": ("M19", "All survive (damage)"),
    "m24_enter_ch5": ("M20", "Enter Chamber 5"),
    "m25_first_boss_dmg": ("M21", "First boss damage"),
    "m26_boss_half_hp": (r"M22", r"Boss at 50\% HP"),
    "m27_boss_defeated": ("M23", "Boss defeated"),
    "m28_all_alive_bonus": ("M24", "All alive at kill"),
    "m_comm_ch1": (r"M\_comm\_ch1", "Valid messaging, Ch1"),
    "m_comm_ch2": (r"M\_comm\_ch2", "Valid messaging, Ch2"),
    "m_comm_ch3": (r"M\_comm\_ch3", "Valid messaging, Ch3"),
    "m_comm_ch4": (r"M\_comm\_ch4", "Valid messaging, Ch4"),
    "m_comm_ch5": (r"M\_comm\_ch5", "Valid messaging, Ch5"),
}

# Column headers: the printed label of each VLM row, de-indented.
HEAD = {
    "LLM-2B": "Qwen3.5-2B", "LLM-2B+Heb2.0": r"Qwen3.5-2B$+$plast.",
    "LLM-2B+Heb": r"Qwen3.5-2B$+$plast.\ (inst.)",
    "LLM-9B": "Qwen3.5-9B", "LLM-9B+Heb2.0": r"Qwen3.5-9B$+$plast.",
    "LLM-9B+Heb": r"Qwen3.5-9B$+$plast.\ (inst.)",
    "Gemma-E4B": "Gemma-E4B",
    "Gemma-E4B+Central Orch.": r"Gemma-E4B$+$Central Orch.",
    "Gemma-E4B+Heb2.0": r"Gemma-E4B$+$plast.",
    "Gemma-E4B+Heb": r"Gemma-E4B$+$plast.\ (inst.)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    registry = {n: (d, r) for n, d, r, *_ in
                list(mft.ROWS) + list(NEW_ROWS) + list(EXTRA_ROWS)}
    # the zero-shot VLM panel, in the order tab:final_comparison prints it
    keys = [key for blk in PANELS[0][1] for _tag, _lab, key in blk if key]

    cols, seeds = [], {}
    for key in keys:
        d, root = registry[key]
        runs = MR.load_runs(Path(root), d, frozenset())
        if not runs:
            print(f"  [skip] {key}: no runs", file=sys.stderr)
            continue
        a = MR.aggregate(runs)
        cols.append((key, a["steps"]))
        seeds[key] = (a["n_runs"], a["n_eps"])

    L = [r"\begin{tabular}{l l *{%d}{c}}" % len(cols), r"\toprule",
         "ID & Milestone"]
    for key, _ in cols:
        L.append(r" & \rotatebox{90}{%s~}" % HEAD.get(key, key))
    L.append(r" \\")
    L.append(r"\midrule")
    ncol = len(cols) + 2
    for heading, mids in SECTIONS:
        L.append(r"\multicolumn{%d}{l}{\emph{%s}}\\" % (ncol, heading))
        for mid in mids:
            pid, name = ROWS_META[mid]
            cells = []
            for _key, steps in cols:
                med, n = steps.get(mid, (None, 0))
                cells.append("--" if med is None else "$%d_{%d}$" % (round(med), n))
            L.append("%-12s & %-22s & %s \\\\" % (pid, name, " & ".join(cells)))
        if (heading, mids) != SECTIONS[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(L) + "\n"
    print(tex)

    print("%% seeds x episodes per column:", file=sys.stderr)
    for key, _ in cols:
        print("%%   %-28s %d seeds, %d episodes" % (key, *seeds[key]),
              file=sys.stderr)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "steps_to_milestone_rows.tex").write_text(tex,
                                                              encoding="utf-8")
        print("wrote %s/steps_to_milestone_rows.tex" % args.out,
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
