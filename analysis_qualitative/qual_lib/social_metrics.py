"""Dimension 2 — social-module interpretability metrics.

Works on non-stale social_module units from llm_calls plus the run's W
sources: final_metrics graph_snapshots (per-50-steps, no full W — top_3_pairs
+ per_agent_out_strength allow partial reconstruction for N=3) and
hebbian_snapshots.jsonl (episode-end full W).
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

_AGENT = re.compile(r"agent[_\s]?(\d+)", re.IGNORECASE)


def _aidx(name):
    m = _AGENT.search(str(name or ""))
    return int(m.group(1)) if m else None


def _ep_bounds(final_metrics):
    lens = [int(x) for x in (final_metrics or {}).get("episode_lengths", [])]
    bounds, c = [], 0
    for L in lens:
        bounds.append((c, c + L))
        c += L
    return bounds


def _global_step(ep, t, bounds):
    if 0 < ep <= len(bounds):
        return bounds[ep - 1][0] + t
    return None


def _w_from_snapshot(snap, i, j, n_agents=3):
    """W[i][j] from a graph_snapshot: direct via top_3_pairs, else
    out_strength reconstruction (needs the OTHER outgoing pair present)."""
    pairs = {(p["i"], p["j"]): p["w"] for p in snap.get("top_3_pairs") or []}
    if (i, j) in pairs:
        return pairs[(i, j)], "direct"
    outs = snap.get("per_agent_out_strength")
    if outs and i < len(outs):
        others = [k for k in range(n_agents) if k not in (i, j)]
        if len(others) == 1 and (i, others[0]) in pairs:
            return outs[i] - pairs[(i, others[0])], "reconstructed"
    return None, None


def analyze_run(ctx) -> dict:
    """All social-interpretability measures for one run (or {} if no units)."""
    units = [u for u in ctx["units"]
             if u["module"] == "social_module" and not u.get("stale")
             and u.get("parse_ok")]
    if not units:
        return {}

    fm = ctx["final_metrics"] or {}
    bounds = _ep_bounds(fm)
    snaps = fm.get("graph_snapshots") or []
    snap_steps = [int(s.get("step", -1)) for s in snaps]
    ep_end_W = {int(s.get("episode", i + 1)): s.get("W")
                for i, s in enumerate(ctx["hebbian_snapshots"] or [])}
    n_agents = ctx["n_agents"]

    # message lookup: (ep, t, sender_idx) -> record ; marginal target shares
    msg_map = {}
    target_marg = defaultdict(lambda: defaultdict(int))
    for ep_i, ep_msgs in ctx["messages"].items():
        for m in ep_msgs:
            s = _aidx(m.get("sender"))
            msg_map[(ep_i, int(m.get("t", -1)), s)] = m
            tgt = _aidx(m.get("model_target_canonical") or m.get("model_target"))
            if tgt is not None:
                target_marg[s][tgt] += 1

    res = {
        "n_deliberations": len(units), "n_with_bond_refs": 0,
        "ask_rate": 0, "faith": [], "faith_resolvable": 0, "faith_total": 0,
        "follow_same": [0, 0], "follow_window": [0, 0], "follow_baseline": [],
        "adoption": [], "argmax_ok": [0, 0], "anti_bond": [0, 0],
        "respond_follow": [0, 0],
    }
    si = ctx["intervals"].get("social_interval", 8)

    for u in units:
        resp = u.get("response") or {}
        a = _aidx(u.get("agent"))
        ep_t = u.get("clock")  # stamped by metrics.load_ctx for aligned runs
        refs = resp.get("referenced_bonds") or {}
        if isinstance(refs, dict) and refs:
            res["n_with_bond_refs"] += 1

        # bond-value faithfulness
        gstep = ep_t and _global_step(ep_t[0], ep_t[1], bounds)
        for mate, wclaim in (refs or {}).items():
            j = _aidx(mate)
            if j is None or a is None or not isinstance(wclaim, (int, float)):
                continue
            res["faith_total"] += 1
            actual = None
            if gstep is not None and snaps:
                k = min(range(len(snaps)),
                        key=lambda x: abs(snap_steps[x] - gstep))
                if abs(snap_steps[k] - gstep) <= 50:
                    actual, _how = _w_from_snapshot(snaps[k], a, j, n_agents)
            if actual is None and ep_t and ep_end_W.get(ep_t[0]):
                W = ep_end_W[ep_t[0]]
                if a < len(W) and j < len(W[a]):
                    actual = W[a][j]
            if actual is not None:
                res["faith_resolvable"] += 1
                res["faith"].append(abs(float(wclaim) - float(actual)))

        ask = _aidx(resp.get("ask_target"))
        if ask is not None:
            res["ask_rate"] += 1
            if refs:
                vals = {_aidx(k): v for k, v in refs.items()
                        if _aidx(k) is not None}
                if vals:
                    mx = max(vals, key=vals.get)
                    mn = min(vals, key=vals.get)
                    res["argmax_ok"][1] += 1
                    res["argmax_ok"][0] += int(ask == mx)
                    res["anti_bond"][1] += 1
                    res["anti_bond"][0] += int(ask == mn and mn != mx)
            if ep_t and a is not None:
                ep_i, t = ep_t
                m = msg_map.get((ep_i, t, a))
                if m:
                    tgt = _aidx(m.get("model_target_canonical")
                                or m.get("model_target"))
                    res["follow_same"][1] += 1
                    res["follow_same"][0] += int(tgt == ask)
                followed = False
                for dt in range(0, si):
                    m2 = msg_map.get((ep_i, t + dt, a))
                    if m2 and _aidx(m2.get("model_target_canonical")
                                    or m2.get("model_target")) == ask:
                        followed = True
                        break
                res["follow_window"][1] += 1
                res["follow_window"][0] += int(followed)
                tm = target_marg.get(a) or {}
                tot = sum(tm.values())
                if tot:
                    res["follow_baseline"].append(tm.get(ask, 0) / tot)
                if m and resp.get("ask_message"):
                    ratio = difflib.SequenceMatcher(
                        None, str(resp["ask_message"]).lower(),
                        str(m.get("text") or "").lower()).ratio()
                    res["adoption"].append(ratio)

        rto = [x for x in (resp.get("respond_to") or []) if _aidx(x) is not None]
        if rto and ep_t and a is not None:
            ep_i, t = ep_t
            ok = False
            for dt in range(0, si):
                m2 = msg_map.get((ep_i, t + dt, a))
                if m2 and _aidx(m2.get("model_target_canonical")
                                or m2.get("model_target")) in {
                                    _aidx(x) for x in rto}:
                    ok = True
                    break
            res["respond_follow"][1] += 1
            res["respond_follow"][0] += int(ok)

    def _rate(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else None

    def _mean(v):
        return round(sum(v) / len(v), 4) if v else None

    return {
        "n_deliberations": res["n_deliberations"],
        "bond_ref_coverage": round(res["n_with_bond_refs"]
                                   / res["n_deliberations"], 4),
        "ask_rate": round(res["ask_rate"] / res["n_deliberations"], 4),
        "faith_mae": _mean(res["faith"]),
        "faith_within_005": (round(sum(1 for d in res["faith"] if d <= 0.05)
                                   / len(res["faith"]), 4)
                             if res["faith"] else None),
        "faith_resolvable": (round(res["faith_resolvable"]
                                   / res["faith_total"], 4)
                             if res["faith_total"] else None),
        "follow_same_step": _rate(res["follow_same"]),
        "follow_window": _rate(res["follow_window"]),
        "follow_baseline": _mean(res["follow_baseline"]),
        "ask_msg_adoption": _mean(res["adoption"]),
        "argmax_consistency": _rate(res["argmax_ok"]),
        "anti_bond_ask_rate": _rate(res["anti_bond"]),
        "respond_follow": _rate(res["respond_follow"]),
    }
