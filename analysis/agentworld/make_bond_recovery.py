"""Do learned bonds recover the hidden teams? (AgentWorld multi-team worlds)

For every episode directory (``.../episode_k`` with ``hebbian_W.npy`` and
``true_team.npy``) this scores three N×N matrices against true team membership:

    W        the Hebbian bonds at the end of the episode
    counts   raw interaction counts (DMs both ways + transfers + joint attacks)
    spatial  rounds spent within ``--radius`` tiles of each other

A score is the ROC AUC over all ordered off-diagonal pairs (is i's bond to j
higher for teammates than for strangers?) plus precision@k per agent, with k =
the agent's number of true teammates. W only says something about Hebbian
learning where it beats ``counts`` and ``spatial``; with replicas on shared
spawn tiles ``spatial`` is trivially high (compose worlds with --team-spacing).

    python analysis/agentworld/make_bond_recovery.py runs_aw/<group> [--out csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUC (ties count half); NaN if a class is empty."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    r_pos = r[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def precision_at_k(M: np.ndarray, same: np.ndarray) -> float:
    vals = []
    for i in range(len(M)):
        k = int(same[i].sum())
        if k == 0:
            continue
        row = M[i].astype(float).copy()
        row[i] = -np.inf
        top = np.argsort(-row, kind="mergesort")[:k]
        vals.append(same[i, top].mean())
    return float(np.mean(vals)) if vals else float("nan")


def _jsonl(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def interaction_counts(ep: Path, n: int) -> np.ndarray:
    C = np.zeros((n, n))
    for m in _jsonl(ep / "messages.jsonl"):
        if m.get("kind") == "dm" and m.get("delivered") and m.get("receiver") is not None:
            C[m["sender"], m["receiver"]] += 1
            C[m["receiver"], m["sender"]] += 1
    for e in _jsonl(ep / "events.jsonl"):
        if e["kind"] in ("xfer", "combat") and e.get("dst") is not None:
            C[e["src"], e["dst"]] += 1
            C[e["dst"], e["src"]] += 1
    return C


def spatial_counts(ep: Path, n: int, radius: float) -> np.ndarray:
    S = np.zeros((n, n))
    for s in _jsonl(ep / "replay" / "state.jsonl"):
        pts = np.full((n, 2), np.nan)
        for a in s["agents"]:
            if a.get("x") is not None:
                pts[a["i"]] = (a["x"], a["y"])
        d = np.abs(pts[:, None, :] - pts[None, :, :]).sum(axis=2)
        S += np.nan_to_num((d <= radius).astype(float))
    np.fill_diagonal(S, 0)
    return S


def score_episode(ep: Path, radius: float = 4.0) -> Dict[str, float]:
    W = np.load(ep / "hebbian_W.npy").astype(float)
    same = np.load(ep / "true_team.npy").astype(int)
    n = len(W)
    off = ~np.eye(n, dtype=bool)
    out: Dict[str, float] = {"n_agents": n, "n_teams": int(round(n / max(1, same[0].sum() + 1)))}
    for name, M in (("W", W), ("counts", interaction_counts(ep, n)),
                    ("spatial", spatial_counts(ep, n, radius))):
        out[f"auc_{name}"] = auc(M[off], same[off])
        out[f"p@k_{name}"] = precision_at_k(M, same)
    return out


def run_meta(ep: Path) -> Dict[str, str]:
    cfg = ep.parent / "config.json"
    meta = {"run": str(ep.parent), "episode": ep.name.split("_")[-1]}
    if cfg.is_file():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        meta.update(arm=c.get("arm", "?"), replicas=str(c.get("replicas", "?")),
                    seed=str(c.get("seed", "?")))
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = []
    for root in args.roots:
        for ep in sorted(root.rglob("episode_*")):
            if (ep / "hebbian_W.npy").is_file() and (ep / "true_team.npy").is_file():
                same = np.load(ep / "true_team.npy")
                if same.sum() == len(same) * (len(same) - 1):
                    continue  # a single team: nothing to recover
                rows.append({**run_meta(ep), **score_episode(ep, args.radius)})
    if not rows:
        sys.exit("no multi-team episodes with hebbian_W.npy + true_team.npy found")
    keys = list(rows[0])
    out = args.out or args.roots[0] / "bond_recovery.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} episodes → {out}")
    for r in rows:
        print(f"{r.get('arm', '?'):12s} N={r['n_agents']:4d} ep{r['episode']}  "
              f"AUC W {r['auc_W']:.2f} | counts {r['auc_counts']:.2f} | spatial {r['auc_spatial']:.2f}")


if __name__ == "__main__":
    main()
