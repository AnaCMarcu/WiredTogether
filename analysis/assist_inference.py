"""Assisted-kill inference for ch4_combat milestone fires in the 3f runs.
present = within 5 blocks of the killer in the 10 steps before the kill;
engaged = present AND >=1 Dig (attack) action in that window."""
import csv
import json
import math
import re
from pathlib import Path

from paths import RUNS  # noqa: E402  (also puts siblings on sys.path)
from make_results import MILESTONE_TRACK  # noqa: E402

ROOT = RUNS / "new_exp_0_gemma" / "new_exp_0_gemma_hebbian3f"
RADIUS, WIN = 5.0, 10


def aid(name):
    return int(re.match(r"agent_?(\d+)", name).group(1))


def main():
    for seed in (42, 123, 456):
        rd = ROOT / f"seed_{seed}"
        fm = json.loads((rd / "final_metrics.json").read_text(encoding="utf-8"))
        lens = fm["episode_lengths"]
        offs = [0]
        for L in lens[:-1]:
            offs.append(offs[-1] + L)
        kills = [(e["step"], aid(e["contributor"]), e["milestone_id"])
                 for e in fm["milestone_events"]
                 if MILESTONE_TRACK.get(e["milestone_id"]) == "ch4_combat"]
        print(f"--- seed {seed} ---")
        for gstep, k, mid in sorted(kills):
            ep = max(i for i, o in enumerate(offs) if o <= gstep)
            t = gstep - offs[ep]
            rows = list(csv.DictReader(open(rd / f"episodes/ep_{ep + 1:04d}/step_log.csv",
                                            encoding="utf-8")))
            pos, dig = {}, {a: 0 for a in range(3)}
            for r in rows:
                s = int(r["step"])
                if t - WIN <= s <= t:
                    a = int(r["agent_id"])
                    if r["pos_x"]:
                        pos.setdefault(s, {})[a] = (float(r["pos_x"]), float(r["pos_z"]))
                    if r["action"] == "Dig":
                        dig[a] += 1
            verdicts = []
            for j in range(3):
                if j == k:
                    continue
                dmin = min((math.dist(p[j], p[k]) for p in pos.values()
                            if j in p and k in p), default=float("inf"))
                tag = ("ENGAGED" if dig[j] else "present") if dmin <= RADIUS else "-"
                verdicts.append(f"a{j}: {tag} (min d={dmin:.1f}, digs={dig[j]})")
            print(f"  {mid} step {gstep} (ep{ep + 1} t={t}) killer a{k} "
                  f"(digs={dig[k]}) | " + " | ".join(verdicts))


if __name__ == "__main__":
    main()
