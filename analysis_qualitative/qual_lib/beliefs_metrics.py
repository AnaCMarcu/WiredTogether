"""Dimension 4 — belief quality vs ground truth."""

from __future__ import annotations

import re
from collections import defaultdict

from qual_lib import lexicons

_AGENT = re.compile(r"agent[_\s]?(\d+)", re.IGNORECASE)
# "agent_1 ... in chamber 2 / ch2 / cell / communal"
_LOC_CLAIM = re.compile(
    r"agent[_\s]?(?P<agent>\d+)(?P<mid>[^.;\n]{0,60}?)"
    r"(?P<loc>ch(?:amber)?[\s_]?(?P<num>[1-5])|communal|cell\b)",
    re.IGNORECASE)


def _belief_text(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return " | ".join(str(x) for x in val)
    return str(val)


def _true_chamber(row):
    return str(row.get("chamber") or "").lower()


def _claim_matches(claimed_loc, claimed_num, true_ch):
    if claimed_num:
        return true_ch.startswith(f"ch{claimed_num}")
    loc = claimed_loc.lower()
    if "communal" in loc:
        return true_ch == "ch3_communal"
    if "cell" in loc:
        return true_ch == "ch3"
    return None


def analyze_run(ctx) -> dict:
    rows = ctx["timeline"]
    # ground truth: (ep, t, agent) -> row
    truth = {(r["ep"], r["t"], r["agent"]): r for r in rows}

    n_percep = grounded = 0
    strict_grounded = diagnostic = 0
    halluc_rows = []
    loc_claims = loc_correct = 0
    resp_checked = resp_hit = 0
    contagion_cand = contagion_hit = 0
    per_agent = defaultdict(list)
    for r in rows:
        per_agent[(r["ep"], r["agent"])].append(r)
    for k in per_agent:
        per_agent[k].sort(key=lambda r: r["t"])

    for r in rows:
        b = r.get("beliefs") or {}
        # perception grounding
        pv = _belief_text(b.get("perception"))
        if pv:
            n_percep += 1
            bad = lexicons.impossible_mentions(pv, r.get("chamber"))
            diag = lexicons.diagnostic_mentions(pv, r.get("chamber"))
            if diag:
                diagnostic += 1
            if bad:
                halluc_rows.append(r)
            else:
                grounded += 1
                # STRICT grounding: nothing impossible AND at least one
                # chamber-diagnostic object named. The permissive rate above
                # has a degenerate optimum (an empty/vague statement passes);
                # this one requires positive, checkable evidence.
                if diag:
                    strict_grounded += 1
        # partner-location claims
        for part in (b.get("partner") or []):
            text = _belief_text(part)
            for m in _LOC_CLAIM.finditer(text):
                j = int(m.group("agent"))
                if j == r["agent"]:
                    continue
                other = truth.get((r["ep"], r["t"], j))
                if not other:
                    continue
                ok = _claim_matches(m.group("loc"), m.group("num"),
                                    _true_chamber(other))
                if ok is None:
                    continue
                loc_claims += 1
                loc_correct += int(ok)
        # responsiveness: received message content reaches partner beliefs
        msg = r.get("msg")
        if msg and msg.get("text") and b.get("partner"):
            pass  # handled below via receiver side

    # responsiveness (receiver side): after agent a RECEIVES a message at t,
    # does a partner-belief update within 2*belief_interval mention its tokens?
    bi = ctx["intervals"].get("belief_interval", 5)
    recv = defaultdict(list)  # (ep, receiver) -> [(t, text)]
    for ep_i, ep_msgs in ctx["messages"].items():
        for m in ep_msgs:
            am = _AGENT.search(str(m.get("receiver") or ""))
            if am and m.get("text"):
                recv[(ep_i, int(am.group(1)))].append((int(m["t"]), m["text"]))
    for (ep_i, a), lst in recv.items():
        seq = per_agent.get((ep_i, a), [])
        upd = [(r["t"], _belief_text((r.get("beliefs") or {}).get("partner")))
               for r in seq if (r.get("beliefs") or {}).get("partner")]
        for t, text in lst[:200]:  # cap per episode for cost
            words = {w for w in re.findall(r"[a-z]{4,}", text.lower())
                     if w not in ("agent",)}
            if len(words) < 2:
                continue
            after = [u for u in upd if t < u[0] <= t + 2 * bi]
            if not after:
                continue
            resp_checked += 1
            hit = any(len(words & set(re.findall(r"[a-z]{4,}", u[1].lower())))
                      >= max(2, len(words) // 3) for u in after)
            resp_hit += int(hit)

    # hallucination -> futile action contagion
    for r in halluc_rows:
        seq = per_agent.get((r["ep"], r["agent"]), [])
        idx = next((i for i, x in enumerate(seq) if x["t"] == r["t"]), None)
        if idx is None:
            continue
        contagion_cand += 1
        nxt = seq[idx + 1: idx + 6]
        if any(x.get("action") == "Dig" and not (x.get("reward_task") or 0)
               for x in nxt):
            contagion_hit += 1

    def _rate(n, d):
        return round(n / d, 4) if d else None

    return {
        "n_perception_updates": n_percep,
        "perception_grounding_rate": _rate(grounded, n_percep),
        "perception_grounding_strict": _rate(strict_grounded, n_percep),
        "perception_specificity": _rate(diagnostic, n_percep),
        # NET grounding = strict - hallucination: +1 per grounded-and-diagnostic
        # statement, -1 per statement naming an impossible object, 0 for vague.
        # The parameter-free (lambda = 1) member of the family
        # strict - lambda * wrong; reported as a robustness check beside strict
        # (lambda = 0), never blended with a tuned lambda.
        "perception_net_grounding": _rate(strict_grounded - len(halluc_rows), n_percep),
        "partner_loc_claims": loc_claims,
        "partner_loc_accuracy": _rate(loc_correct, loc_claims),
        "belief_responsiveness": _rate(resp_hit, resp_checked),
        "responsiveness_n": resp_checked,
        "hallucination_rows": len(halluc_rows),
        "contagion_rate": _rate(contagion_hit, contagion_cand),
    }
