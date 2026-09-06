"""Cross-dimension — do learned bonds track joint behaviour?

Backs the analysis promised in paper/appendix/appendix_evaluation.tex (the
5-plane pair-interaction tensor "for qualitative analysis of whether learned
bonds track actual joint behaviour"). Reads the tensor from
episode_summary.json["cooperation_metrics"] via episode_io (NEVER from
final_metrics.coop_metrics — zeroed by the coop_eval nesting bug).
"""

from __future__ import annotations

import numpy as np

PLANES = ("messages", "joint_dig", "proximity", "joint_kill",
          "ch5_damage_overlap")


def _spearman(x, y):
    """Spearman rho with average ranks (no scipy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None

    def rank(v):
        order = np.argsort(v)
        ranks = np.empty(len(v), float)
        ranks[order] = np.arange(len(v), dtype=float)
        # average ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                ranks[m] = ranks[m].mean()
        return ranks

    rx, ry = rank(x), rank(y)
    c = np.corrcoef(rx, ry)[0, 1]
    return None if np.isnan(c) else float(c)


def pairs_per_episode(ctx) -> list:
    """[(ep, i, j, W_ij, {plane: count})] for all ordered pairs, all eps."""
    out = []
    for ep_i, coop in ctx["coop_by_ep"].items():
        W = coop.get("hebbian_W")
        tensor = coop.get("pair_interaction") or {}
        if not W:
            continue
        n = len(W)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                planes = {}
                for p in PLANES:
                    mat = tensor.get(p)
                    if mat and i < len(mat) and j < len(mat[i]):
                        planes[p] = float(mat[i][j])
                out.append((ep_i, i, j, float(W[i][j]), planes))
    return out


def analyze_run(ctx) -> dict:
    """Per-run pair records + per-plane Spearman (pooled across episodes)."""
    recs = pairs_per_episode(ctx)
    if not recs:
        return {}
    res = {"n_pair_episodes": len(recs)}
    for p in PLANES:
        xs = [w for (_e, _i, _j, w, pl) in recs if p in pl]
        ys = [pl[p] for (_e, _i, _j, _w, pl) in recs if p in pl]
        rho = _spearman(xs, ys)
        res[f"rho_{p}"] = round(rho, 4) if rho is not None else None
        res[f"n_{p}"] = len(xs)
    # excluded-agent contrast (meaningful for exp10 allied-pair; harmless
    # elsewhere): agent_2's share of messages + proximity involvement.
    tot_msgs = a2_msgs = tot_prox = a2_prox = 0.0
    for (_e, i, j, _w, pl) in recs:
        m = pl.get("messages", 0.0)
        pr = pl.get("proximity", 0.0)
        tot_msgs += m
        tot_prox += pr
        if 2 in (i, j):
            a2_msgs += m
            a2_prox += pr
    res["agent2_msg_share"] = (round(a2_msgs / tot_msgs, 4)
                               if tot_msgs else None)
    res["agent2_prox_share"] = (round(a2_prox / tot_prox, 4)
                                if tot_prox else None)
    res["_pair_records"] = [
        {"ep": e, "i": i, "j": j, "W": w, **pl}
        for (e, i, j, w, pl) in recs
    ]
    return res


def pooled_condition(records: list) -> dict:
    """Condition-level Spearman over pooled pair records from all runs."""
    out = {}
    for p in PLANES:
        xs = [r["W"] for r in records if p in r]
        ys = [r[p] for r in records if p in r]
        rho = _spearman(xs, ys)
        out[f"rho_{p}"] = round(rho, 4) if rho is not None else None
        out[f"n_{p}"] = len(xs)
    return out
