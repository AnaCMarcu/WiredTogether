"""Dimension 3 — failure-mode detectors over timeline rows.

Every detector yields Flag dicts:
    {detector, exp, seed, ep, agent, t0, t1, detail}
Counts are aggregated per condition by the metrics orchestrator.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from qual_lib import lexicons

_WS = re.compile(r"\s+")


def _norm(s) -> str:
    return _WS.sub(" ", str(s or "")).strip().lower()


def _flag(det, r, t0, t1, **detail):
    return {"detector": det, "exp": r["exp"], "seed": r["seed"], "ep": r["ep"],
            "agent": r["agent"], "t0": t0, "t1": t1, "detail": detail}


def _per_agent_series(rows):
    series = defaultdict(list)
    for r in rows:
        series[(r["ep"], r["agent"])].append(r)
    for k in series:
        series[k].sort(key=lambda r: r["t"])
    return series


def task_repeat_loop(rows, min_repeat: int = 3):
    """Same normalized curriculum task adopted >= min_repeat times in a row."""
    out = []
    new_tasks = defaultdict(list)  # (ep, agent) -> [(t, norm_task, row)]
    for r in rows:
        if r.get("task_new") and r.get("task"):
            new_tasks[(r["ep"], r["agent"])].append((r["t"], _norm(r["task"]), r))
    for (ep, a), seq in new_tasks.items():
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1]:
                j += 1
            n = j - i + 1
            if n >= min_repeat:
                out.append(_flag("task_repeat_loop", seq[i][2], seq[i][0],
                                 seq[j][0], task=seq[i][1][:120], n_repeats=n))
            i = j + 1
    return out


def action_stuck_loop(rows, min_len: int = 12):
    """Same action >= min_len consecutive steps, zero task reward, < 1 block
    net displacement. Subtype turn_circling for pure camera spins."""
    out = []
    for (ep, a), seq in _per_agent_series(rows).items():
        i = 0
        while i < len(seq):
            j = i
            act = seq[i].get("action")
            while (j + 1 < len(seq) and seq[j + 1].get("action") == act):
                j += 1
            n = j - i + 1
            if n >= min_len and act and act != "NoOp":
                seg = seq[i:j + 1]
                rew = sum(r.get("reward_task") or 0 for r in seg)
                p0, p1 = seg[0].get("pos") or [], seg[-1].get("pos") or []
                disp = None
                if len(p0) == 3 and len(p1) == 3 and p0[0] is not None and p1[0] is not None:
                    disp = math.dist((p0[0], p0[2]), (p1[0], p1[2]))
                if rew == 0 and (disp is None or disp < 1.0):
                    sub = ("turn_circling" if str(act).startswith(("Turn", "Look"))
                           else "same_action")
                    out.append(_flag("action_stuck_loop", seg[0], seg[0]["t"],
                                     seg[-1]["t"], action=act, n=n,
                                     displacement=disp, subtype=sub))
            i = j + 1
    return out


def idle_pattern(rows, min_len: int = 10):
    out = []
    for (ep, a), seq in _per_agent_series(rows).items():
        i = 0
        while i < len(seq):
            j = i
            while (j + 1 < len(seq) and seq[j + 1].get("action") == "NoOp"
                   and seq[i].get("action") == "NoOp"):
                j += 1
            if seq[i].get("action") == "NoOp" and j - i + 1 >= min_len:
                out.append(_flag("idle_pattern", seq[i], seq[i]["t"],
                                 seq[j]["t"], n=j - i + 1))
            i = j + 1
    return out


def critic_verdicts(rows, lookback: int = 20, milestone_reward: float = 10.0):
    """critic_false_positive / critic_false_negative (fuzzy prefilters)."""
    out = []
    for (ep, a), seq in _per_agent_series(rows).items():
        rewards = {r["t"]: (r.get("reward_task") or 0) for r in seq}
        for r in seq:
            c = r.get("critic")
            if not c or c.get("success") is None:
                continue
            recent = [rewards.get(t, 0) for t in range(max(0, r["t"] - lookback),
                                                       r["t"] + 1)]
            hit_recent = any(v >= milestone_reward for v in recent)
            if c["success"] and not hit_recent:
                out.append(_flag("critic_false_positive", r, r["t"], r["t"],
                                 critique=(c.get("critique") or "")[:120]))
            near = [rewards.get(t, 0) for t in (r["t"] - 1, r["t"], r["t"] + 1,
                                                r["t"] + 2)]
            if c["success"] is False and any(v >= milestone_reward for v in near):
                out.append(_flag("critic_false_negative", r, r["t"], r["t"],
                                 critique=(c.get("critique") or "")[:120]))
    return out


def hallucinated_object(rows):
    """Perception beliefs / thoughts / messages naming chamber-impossible
    objects. One flag per (row, source)."""
    out = []
    for r in rows:
        ch = r.get("chamber")
        sources = {
            "belief": None, "thought": r.get("thoughts"),
            "message": (r.get("msg") or {}).get("text"),
        }
        b = r.get("beliefs") or {}
        pv = b.get("perception")
        if pv:
            sources["belief"] = " ".join(pv) if isinstance(pv, list) else str(pv)
        for src, text in sources.items():
            if not text:
                continue
            hits = lexicons.impossible_mentions(str(text), ch)
            if hits:
                out.append(_flag("hallucinated_object", r, r["t"], r["t"],
                                 source=src, objects=hits, chamber=ch,
                                 excerpt=str(text)[:120]))
    return out


def door_confusion(rows):
    out = []
    for r in rows:
        if not str(r.get("chamber") or "").startswith("ch3"):
            continue
        for src in ("thoughts", None):
            text = r.get("thoughts") if src else (r.get("msg") or {}).get("text")
            if text and lexicons.DOOR_CONFUSION.search(str(text)):
                out.append(_flag("door_confusion", r, r["t"], r["t"],
                                 excerpt=str(text)[:120]))
                break
    return out


def rl_action_collapse(rows, is_rl: bool, window: int = 200,
                       top1_thresh: float = 0.7):
    if not is_rl:
        return []
    out = []
    for (ep, a), seq in _per_agent_series(rows).items():
        i = 0
        while i + window <= len(seq):
            acts = Counter(r.get("action") for r in seq[i:i + window])
            act, n = acts.most_common(1)[0]
            if act and n / window > top1_thresh:
                out.append(_flag("rl_action_collapse", seq[i], seq[i]["t"],
                                 seq[i + window - 1]["t"],
                                 action=act, share=round(n / window, 3)))
                i += window  # don't re-flag the same stretch
            else:
                i += window // 4
    return out


def degenerate_comm(rows, min_streak: int = 5):
    out = []
    for (ep, a), seq in _per_agent_series(rows).items():
        texts = [(r["t"], _norm((r.get("msg") or {}).get("text")), r)
                 for r in seq if (r.get("msg") or {}).get("text")]
        i = 0
        while i < len(texts):
            j = i
            while j + 1 < len(texts) and texts[j + 1][1] == texts[i][1]:
                j += 1
            n = j - i + 1
            if n >= min_streak and texts[i][1]:
                out.append(_flag("degenerate_comm", texts[i][2], texts[i][0],
                                 texts[j][0], n=n, text=texts[i][1][:80]))
            i = j + 1
    return out


ALL_DETECTORS = [
    task_repeat_loop, action_stuck_loop, idle_pattern, critic_verdicts,
    hallucinated_object, door_confusion, degenerate_comm,
]


def run_detectors(rows, is_rl: bool) -> list:
    flags = []
    for det in ALL_DETECTORS:
        flags.extend(det(rows))
    flags.extend(rl_action_collapse(rows, is_rl))
    return flags
