"""Experiment-2 main-table metrics for one cofiring suite.

Act use % is pooled over 3 seeds x 3 episodes (population SD), matching the
table caption. dW % and rho are one value per run, so they are reported as
mean +- population SD across the 3 seeds. The pooled columns (task return,
milestone %, coop %, mean W) come straight from make_results.py's
cofire_summary.csv so the two never drift.

rho = Spearman over the 6 directed pairs x 3 episodes of (count of cue acts
i->j in episode e) vs (W[i][j] at the end of episode e), per seed.

Usage:  python analysis/cofire_table.py <runs-root> <assets-dir> [--csv]
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import pstdev

ARMS = [
    ("null (pr)", "exp28_cofire_null", "None", []),
    ("pro", "exp21_cofire_pro", "Obs", ["obs"]),
    ("pri", "exp22_cofire_pri", "Imit", ["imit"]),
    ("prc", "exp20_cofire_prc", "Comm", ["comm"]),
    ("prco", "exp29_cofire_prco", "Comm+Obs", ["comm", "obs"]),
    ("prcoi", "exp23_cofire_prcoi", "Comm+Obs+Imit", ["comm", "obs", "imit"]),
]
ACT_OF_CUE = {"comm": "communicate", "obs": "observe", "imit": "imitate"}


def _total(m):
    """Sum a co-firing attribution entry: matrix, vector or scalar."""
    if m is None:
        return 0.0
    if isinstance(m, (int, float)):
        return float(m)
    return sum(_total(x) for x in m)


def spearman(xs, ys):
    """Spearman rho with average ranks for ties; None if either side is flat."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def seed_cue_stats(run_dir: Path, cues):
    """{cue: {act_use, dW, rho}} for one seed."""
    fm = json.loads((run_dir / "final_metrics.json").read_text())
    sam = fm.get("social_act_metrics") or {}

    counts = sam.get("act_counts") or {}
    total_slots = sum(counts.values())

    attr = sam.get("cofire_attribution") or {}
    # "total" is a sibling key holding the pre-computed sum -- summing every
    # value would double-count it (make_results uses attr["total"] directly).
    tot_attr = (_total(attr.get("total"))
                or sum(_total(m) for k, m in attr.items() if k != "total"))

    # directed act counts per (episode, i, j) from the sidecar
    per_ep = {}
    slots = {}  # episode -> Counter of decision slots by act
    sa = run_dir / "social_acts.jsonl"
    if sa.exists():
        for line in sa.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Adoption echoes (rows carrying 'adopted_step') report an
            # earlier decision being re-enacted, not a new choice -- act_counts
            # excludes them, so we must too or imitation is counted twice.
            if "adopted_step" in r:
                continue
            act = r.get("act")
            slots.setdefault(int(r.get("ep", 0)), Counter())["_all"] += 1
            if act and act != "none":
                slots[int(r.get("ep", 0))][act] += 1
            tgt = r.get("target")
            if act in (None, "none") or not tgt:
                continue
            try:
                i = int(str(r.get("agent", "")).rsplit("_", 1)[-1])
                j = int(str(tgt).rsplit("_", 1)[-1])
            except ValueError:
                continue
            per_ep.setdefault(int(r.get("ep", 0)), Counter())[(act, i, j)] += 1

    # end-of-episode W
    W = {}
    snap = run_dir / "hebbian_snapshots.jsonl"
    if snap.exists():
        for line in snap.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("W"):
                W[int(r.get("episode", 0))] = r["W"]

    out = {}
    for cue in cues:
        act_name = ACT_OF_CUE[cue]
        use = counts.get(act_name, 0) / total_slots if total_slots else float("nan")
        ep_use = [c[act_name] / c["_all"] for c in slots.values() if c["_all"]]

        m = attr.get(cue)
        dW = (_total(m) / tot_attr) if (m is not None and tot_attr) else 0.0

        xs, ys = [], []
        for ep, mat in sorted(W.items()):
            acts = per_ep.get(ep, Counter())
            n = len(mat)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    xs.append(acts.get((act_name, i, j), 0))
                    ys.append(mat[i][j])
        out[cue] = {"act_use": use, "ep_use": ep_use,
                    "dW": dW, "rho": spearman(xs, ys)}
    return out


def agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return None, None, 0
    return sum(vals) / len(vals), pstdev(vals), len(vals)


def main():
    root = Path(sys.argv[1])
    assets = Path(sys.argv[2])
    as_csv = "--csv" in sys.argv

    pooled = {r["condition"]: r for r in
              csv.DictReader(open(assets / "cofire_summary.csv"))}

    rows = []
    for cond, exp, label, cues in ARMS:
        if cond not in pooled:
            continue
        p = pooled[cond]
        seeds = sorted((root / exp).glob("seed_*"))
        seeds = [s for s in seeds if (s / "final_metrics.json").exists()]
        per_cue = {}
        for cue in cues:
            stats = [seed_cue_stats(s, [cue])[cue] for s in seeds]
            d = {k: agg([st[k] for st in stats]) for k in ("dW", "rho")}
            # Act use is pooled over the 3 seeds x 3 episodes, matching the
            # table caption; dW and rho are per-seed (one value per run).
            d["act_use"] = agg([v for st in stats for v in st["ep_use"]])
            per_cue[cue] = d
        rows.append({
            "label": label, "cond": cond, "n_seeds": len(seeds),
            "task": (float(p["task_nocomm_mean"]), float(p["task_nocomm_std"])),
            "allms": (float(p["allms_nocomm_pct_mean"]),
                      float(p["allms_nocomm_pct_std"])),
            "coop": (float(p["coop_pct_mean"]), float(p["coop_pct_std"])),
            "W": float(p["bond_mean"]),
            "cues": per_cue,
        })

    if as_csv:
        w = csv.writer(sys.stdout, lineterminator="\n")
        w.writerow(["condition", "cue", "task", "task_sd", "allms_pct",
                    "allms_pct_sd", "coop_pct", "coop_pct_sd", "meanW",
                    "act_use_pct", "act_use_sd", "dW_pct", "dW_sd",
                    "rho", "rho_sd", "rho_n"])
        for r in rows:
            base = [r["label"], "", f"{r['task'][0]:.1f}", f"{r['task'][1]:.1f}",
                    f"{r['allms'][0]:.1f}", f"{r['allms'][1]:.1f}",
                    f"{r['coop'][0]:.1f}", f"{r['coop'][1]:.1f}",
                    f"{r['W']:.3f}"]
            if not r["cues"]:
                w.writerow(base + [""] * 7)
            for cue, s in r["cues"].items():
                (u, us, _), (d, ds, _), (rh, rs, rn) = (
                    s["act_use"], s["dW"], s["rho"])
                w.writerow(base[:1] + [cue] + base[2:] + [
                    f"{100*u:.1f}", f"{100*us:.1f}",
                    f"{100*d:.1f}", f"{100*ds:.1f}",
                    "" if rh is None else f"{rh:.2f}",
                    "" if rs is None else f"{rs:.2f}", rn])
        return

    for r in rows:
        print(f"{r['label']}  (n_seeds={r['n_seeds']})")
        print(f"   task {r['task'][0]:6.0f} +-{r['task'][1]:<5.0f}"
              f"  allms% {r['allms'][0]:5.1f} +-{r['allms'][1]:<4.1f}"
              f"  coop% {r['coop'][0]:5.1f} +-{r['coop'][1]:<4.1f}"
              f"  W {r['W']:.2f}")
        for cue, s in r["cues"].items():
            (u, us, _), (d, ds, _), (rh, rs, rn) = (
                s["act_use"], s["dW"], s["rho"])
            rho = "--" if rh is None else f"{rh:.2f} +-{rs:.2f} (n={rn})"
            print(f"      {cue:5s} act {100*u:5.1f} +-{100*us:<4.1f}"
                  f"  dW {100*d:5.1f} +-{100*ds:<4.1f}  rho {rho}")
        print()


if __name__ == "__main__":
    main()
