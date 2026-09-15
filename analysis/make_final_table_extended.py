#!/usr/bin/env python3
"""make_final_table_extended.py — the paper's Table 2 plus the newer arms.

make_final_table.py's ROWS registry is the published eleven conditions. The
arms that landed after it (the Gemma-4 RL baselines, both social-replay lanes,
and the three-factor "Hebbian 2.0" rule on all three backbones) live in other
run groups, and two of them reuse the SAME directory names in two groups
(exp30/exp31 exist in both replay lanes), so they cannot be expressed by
editing a dir name.

This driver appends them to make_final_table.ROWS and calls its own collect(),
so every published row is recomputed by the published code path — same
percentage denominators, same decomposed task return, same steps-to-first-
completion medians — and the new rows are computed identically.

  milestone completion = non-comm milestones fired / 25, per episode
  cooperative          = Ch2-Ch5 milestones fired / 17, per episode
  task return          = decomposed team return (task + comm), excludes
                         hebbian_diffuse
  steps                = within-episode step of first completion, median over
                         completing episodes (count in parentheses)

Usage:
  python analysis/make_final_table_extended.py
  python analysis/make_final_table_extended.py --out paper_assets/final_ext
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paths import ASSETS, group  # noqa: F401  (also puts siblings on sys.path)

import make_final_table as mft

GEMMA4_RL = group("gemma4")
SR_QWEN = group("social_replay_qwen")
SR_GEMMA = group("social_replay_gemma4")

# (name, dir, runs-root, base model, learning, social coupling) — the tuple
# shape make_final_table.ROWS uses.
NEW_ROWS = [
    # Hebbian 2.0 (three-factor + signed death LTD) on the frozen-LLM arms.
    ("Gemma-E4B+Heb2.0", "new_exp_0_gemma_hebbian3f", mft.GEMMA,
     "Gemma-4-E4B", "none (frozen LLM)", "Hebbian 2.0 + prompt"),
    ("LLM-2B+Heb2.0", "exp34_llm_2b_three_factor_fdecay", mft.MEDIUM,
     "Qwen3.5-2B", "none (frozen LLM)", "Hebbian 2.0 + prompt"),
    ("LLM-9B+Heb2.0", "exp35_llm_9b_three_factor_fdecay", mft.MEDIUM,
     "Qwen3.5-9B", "none (frozen LLM)", "Hebbian 2.0 + prompt"),
    # Gemma-4 RL baselines (no Hebbian) — the RL block's Gemma counterpart.
    ("Gemma IPPO", "exp04_ippo", GEMMA4_RL,
     "Gemma-4-E4B", "IPPO (LoRA)", "—"),
    ("Gemma MAPPO", "exp03_mappo", GEMMA4_RL,
     "Gemma-4-E4B", "MAPPO (shared critic)", "—"),
    # Social replay (Eq. 7 experience sharing, rho=0.3), both lanes.
    ("IPPO+Heb+SR", "exp31_ippo_hebbian_replay", SR_QWEN,
     "Qwen3.5-2B", "IPPO (LoRA)", "Hebbian + diff. + replay"),
    ("MAPPO+Heb+SR", "exp30_mappo_hebbian_replay", SR_QWEN,
     "Qwen3.5-2B", "MAPPO (shared critic)", "Hebbian + diff. + replay"),
    ("Gemma IPPO+Heb+SR", "exp31_ippo_hebbian_replay", SR_GEMMA,
     "Gemma-4-E4B", "IPPO (LoRA)", "Hebbian + diff. + replay"),
    ("Gemma MAPPO+Heb+SR", "exp30_mappo_hebbian_replay", SR_GEMMA,
     "Gemma-4-E4B", "MAPPO (shared critic)", "Hebbian + diff. + replay"),
]

# Paper display names (tab:final_comparison uses the model name, not the
# internal label) and the block each row belongs to.
DISPLAY = {
    "LLM-2B": "Qwen3.5-2B", "LLM-2B+Heb": "Qwen3.5-2B+Heb",
    "LLM-2B+Heb2.0": "Qwen3.5-2B+Heb2.0",
    "LLM-9B": "Qwen3.5-9B", "LLM-9B+Heb": "Qwen3.5-9B+Heb",
    "LLM-9B+Heb2.0": "Qwen3.5-9B+Heb2.0",
    "Gemma-E4B": "Gemma-E4B", "Gemma-E4B+Heb": "Gemma-E4B+Heb",
    "Gemma-E4B+Heb2.0": "Gemma-E4B+Heb2.0",
    "Gemma-E4B+Central Orch.": "Gemma-E4B+Central Orch.",
    "IPPO": "Qwen3.5-2B IPPO", "IPPO+Heb": "Qwen3.5-2B IPPO+Heb",
    "IPPO+Heb+SR": "Qwen3.5-2B IPPO+Heb+SR",
    "MAPPO": "Qwen3.5-2B MAPPO", "MAPPO+Heb": "Qwen3.5-2B MAPPO+Heb",
    "MAPPO+Heb+SR": "Qwen3.5-2B MAPPO+Heb+SR",
    "Gemma IPPO": "Gemma-E4B IPPO", "Gemma MAPPO": "Gemma-E4B MAPPO",
    "Gemma IPPO+Heb+SR": "Gemma-E4B IPPO+Heb+SR",
    "Gemma MAPPO+Heb+SR": "Gemma-E4B MAPPO+Heb+SR",
}

# Row order = the paper's, with each new arm filed into its own block.
ORDER = [
    ("Qwen3.5 frozen LLM", ["LLM-2B", "LLM-2B+Heb", "LLM-2B+Heb2.0",
                            "LLM-9B", "LLM-9B+Heb", "LLM-9B+Heb2.0"]),
    ("Gemma-4 frozen LLM", ["Gemma-E4B", "Gemma-E4B+Central Orch.",
                            "Gemma-E4B+Heb", "Gemma-E4B+Heb2.0"]),
    ("Qwen3.5-2B IPPO", ["IPPO", "IPPO+Heb", "IPPO+Heb+SR"]),
    ("Qwen3.5-2B MAPPO", ["MAPPO", "MAPPO+Heb", "MAPPO+Heb+SR"]),
    ("Gemma-4 E4B RL", ["Gemma IPPO", "Gemma IPPO+Heb+SR",
                        "Gemma MAPPO", "Gemma MAPPO+Heb+SR"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    mft.ROWS = list(mft.ROWS) + NEW_ROWS
    rows, _perception = mft.collect()
    by_name = {r["name"]: r for r in rows}

    hdr = (f"| Condition | Task return | Milest. % | Coop. % | "
           + " | ".join(lbl for _, lbl in mft.STEP_COLS)
           + " | seeds | eps |")
    L = [f"## Table 2 — results "
         f"(denominators: {mft.NONCOMM_MAX} non-comm, {mft.COOP_MAX} coop)",
         "", hdr, "|" + "---|" * (5 + len(mft.STEP_COLS) + 2)]

    for block, names in ORDER:
        L.append(f"| **{block}** |" + " |" * (6 + len(mft.STEP_COLS)))
        for name in names:
            r = by_name.get(name)
            if r is None:
                L.append(f"| {DISPLAY.get(name, name)} | *(no runs)* |"
                         + " |" * (5 + len(mft.STEP_COLS)))
                continue
            if not r["decomposed"]:
                print(f"  [warn] {name}: task return not decomposed",
                      file=sys.stderr)
            cells = [DISPLAY.get(name, name),
                     f"{r['task'][0]:.0f} ± {r['task'][1]:.0f}",
                     mft.f2(r["ms_pct"]), mft.f2(r["coop_pct"])]
            cells += [mft.f_steps(r["steps"][mid]) for mid, _ in mft.STEP_COLS]
            cells += [str(r["n_runs"]), str(r["n_eps"])]
            L.append("| " + " | ".join(cells) + " |")

    text = "\n".join(L)
    print(text)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "final_table_extended.md").write_text(text + "\n",
                                                          encoding="utf-8")
        (args.out / "table_rows.tex").write_text(mft.latex(rows) + "\n",
                                                 encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
