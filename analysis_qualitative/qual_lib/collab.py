"""Collaboration case mining — success/failure interaction episodes +
Hebbian-vs-baseline contrast.

Detects, per run, concrete agent-interaction episodes where collaboration
succeeds or fails, then aggregates them into run/condition tables and a
matched-pair (baseline vs +Heb) contrast, and renders exemplar transcripts.

Event grammar (all events carry provenance {exp, seed, ep, t0, t1, agents}):
    request_outcome   directed request/directive message; outcome within
                      HORIZON steps: fulfilled (target complies and/or a real
                      coop milestone fires) / answered (reply only) / ignored
    coop_milestone    non-timer coop-track milestone (success by definition),
                      annotated with comm_before_coop from milestone_log
    mutual_deadlock   Ch3: both agents of a pair ask each other to press
                      within a window, nobody presses -> failure
    ch3_stall         locked in Ch3 >= CH3_STALL_MIN steps before the first
                      real (non-timer) switch press -> coordination latency
    ch2_neglect       episode reaches Ch2 but never breaks the first anvil
                      despite anvil talk -> failure

Timer discipline: chamber teleports at k*max_steps/5 force milestone races
(m17/m18 fire on the boundary without coordination); every milestone within
TIMER_SLACK of a boundary is excluded from success events (same rule as
cases.py). Entry milestones (m16/m20/m24) are never counted as successes.

Hebbian contrast: for Hebbian runs each directed event carries the
end-of-episode bond W[sender][target] (episode_summary cooperation_metrics —
via episode_io, never final_metrics), so request outcomes can be conditioned
on bond strength.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from qual_lib import registry, lexicons, provenance
from qual_lib import metrics as metrics_mod
from qual_lib.cases import render_transcript

HORIZON = 30        # steps to observe a request's outcome
COMPLY_WIN = 8      # steps for the target to perform the asked action
REPLY_WIN = 30      # steps for a directed reply
TIMER_SLACK = 5     # +- steps around chamber-timer boundaries
CH3_STALL_MIN = 60  # locked steps before first real press => stall
DEADLOCK_WIN = 40   # both-ask window for mutual deadlock
CLUSTER_GAP = 40    # ignored requests by same sender within gap => one streak

# Entry/teleport milestones are positioning, not collaboration acts.
SUCCESS_EXCLUDE = {"m16_enter_cell", "m20_enter_ch4", "m24_enter_ch5"}

MATCHED_PAIRS = [("LLM-2B", "LLM-2B+Heb"), ("LLM-9B", "LLM-9B+Heb"),
                 ("IPPO", "IPPO+Heb"), ("MAPPO", "MAPPO+Heb"),
                 # Experiment 2 (cofire): enforced vs chosen communication.
                 # Pairs are skipped when a side has no runs under the
                 # aggregated root.
                 ("anchor", "prc")]

COOP_TALK = re.compile(
    r"\b(switch(es)?|press|cell|unlock|locked|free (me|you)|anvils?|"
    r"together|help|zombies?|mobs?|boss|open (the |my |your )?door)\b", re.I)

_DIGITS = re.compile(r"(\d+)")


def _aidx(x):
    """Agent index from 'agent_1' / 'agent1' / 1 / '1' (None if unparseable)."""
    if isinstance(x, int):
        return x
    m = _DIGITS.search(str(x or ""))
    return int(m.group(1)) if m else None


def _timer_bounds(ctx):
    ms = int(((ctx.get("final_metrics") or {}).get("config", {})
              .get("cli_args", {}) or {}).get("max_steps", 1000))
    return {k * ms // 5 for k in (1, 2, 3, 4)}


def _prov(ctx, ep, t0, t1, agents):
    run = ctx["run"]
    return {"exp": run.exp, "seed": run.seed, "label": run.label,
            "is_rl": run.is_rl, "ep": ep, "t0": int(t0), "t1": int(t1),
            "agents": agents}


def _final_step(ctx, ep):
    """Surviving attempt's last step for an episode (None if unknown).

    Relaunches APPEND to messages.jsonl / event_log.jsonl; stale records at
    t beyond the surviving episode's end have no legit (t, sender) twin to
    overwrite them in the dedupe, so they must be clamped out here.
    """
    fs = (ctx["coop_by_ep"].get(ep) or {}).get("final_step")
    if fs:
        return int(fs)
    lens = (ctx.get("final_metrics") or {}).get("episode_lengths") or []
    if 0 < ep <= len(lens):
        return int(lens[ep - 1])
    return None


def _bond_W(ctx, ep, i, j):
    W = (ctx["coop_by_ep"].get(ep) or {}).get("hebbian_W")
    try:
        return round(float(W[i][j]), 4)
    except (TypeError, IndexError, KeyError):
        return None


def real_coop_events(ctx):
    """[(ep, step, mid, contributors, reward, comm_before)] for coop-track
    milestones, timer-boundary races and entry milestones excluded."""
    bounds = _timer_bounds(ctx)
    out = []
    for ep, evs in ctx["events"].items():
        fs = _final_step(ctx, ep)
        ml = {(m.get("milestone"), int(m.get("step", -1))): m
              for m in ((ctx["coop_by_ep"].get(ep) or {})
                        .get("milestone_log") or [])}
        for e in evs:
            mid = e.get("id") or e.get("milestone") or ""
            if (registry.MILESTONE_TRACK.get(mid) not in registry.COOP_TRACKS
                    or mid in SUCCESS_EXCLUDE):
                continue
            t = int(e.get("step", -1))
            if fs is not None and t > fs:
                continue  # stale appended-attempt record
            if any(abs(t - b) <= TIMER_SLACK for b in bounds):
                continue
            contribs = [a for a in map(_aidx, e.get("contributors") or [])
                        if a is not None]
            cbc = (ml.get((mid, t)) or {}).get("comm_before_coop")
            out.append((ep, t, mid, contribs, float(e.get("reward") or 0),
                        cbc))
    return out


def _series_by_agent(ctx):
    ser = defaultdict(list)
    for r in ctx["timeline"]:
        ser[(r["ep"], r["agent"])].append(r)
    for k in ser:
        ser[k].sort(key=lambda r: r["t"])
    return ser


def _directed_messages(ctx):
    """Valid directed messages [(ep, t, sender, target, text, routing, ch)]."""
    chamber_of = {(r["ep"], r["t"], r["agent"]): r.get("chamber")
                  for r in ctx["timeline"]}
    out = []
    for ep, msgs in ctx["messages"].items():
        fs = _final_step(ctx, ep)
        for m in msgs:
            if not m.get("valid") or not m.get("text"):
                continue
            s = _aidx(m.get("sender"))
            tgt = _aidx(m.get("model_target_canonical") or m.get("receiver"))
            if s is None or tgt is None or s == tgt:
                continue
            t = int(m.get("t", -1))
            if fs is not None and t > fs:
                continue  # stale appended-attempt record
            ch = m.get("chamber") or chamber_of.get((ep, t, s))
            out.append((ep, t, s, tgt, str(m["text"]),
                        m.get("routing") or "?", ch))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def request_outcomes(ctx):
    """One event per request/directive message, with its observed outcome."""
    coop_evs = real_coop_events(ctx)
    msgs = _directed_messages(ctx)
    series = _series_by_agent(ctx)
    events = []
    for (ep, t, s, tgt, text, routing, ch) in msgs:
        cats = lexicons.categorize_message(text)
        if not ({"request", "directive"} & set(cats)):
            continue
        replied = any(ep2 == ep and t < t2 <= t + REPLY_WIN
                      and s2 == tgt and tgt2 == s
                      for (ep2, t2, s2, tgt2, *_r) in msgs)
        intended = lexicons.intended_actions(text)
        complied = None
        if intended:
            acts = {r.get("action") for r in series.get((ep, tgt), [])
                    if t < r["t"] <= t + COMPLY_WIN}
            complied = bool(intended & acts)
        # milestone credit only when the sender or the asked target is an
        # actual contributor — an unrelated third-agent milestone inside the
        # horizon is not this request's outcome.
        hit = [(mid, te, rew, cbc) for (ep2, te, mid, contribs, rew, cbc)
               in coop_evs if ep2 == ep and t < te <= t + HORIZON
               and (not contribs or {s, tgt} & set(contribs))]
        if complied or hit:
            status = "fulfilled"
        elif replied:
            status = "answered"
        else:
            status = "ignored"
        coop_talk = bool(COOP_TALK.search(text))
        score = ((2.0 if hit else 0.0) + (1.0 if complied else 0.0)
                 + (0.5 if replied else 0.0) + (0.5 if coop_talk else 0.0))
        events.append({
            "type": "request_outcome", "success": status == "fulfilled",
            **_prov(ctx, ep, t, t + HORIZON, [s, tgt]),
            "score": score, "W": _bond_W(ctx, ep, s, tgt),
            "detail": {"status": status, "categories": cats,
                       "coop_talk": coop_talk, "chamber": ch,
                       "routing": routing, "replied": replied,
                       "complied": complied,
                       "milestones": [m[0] for m in hit],
                       "text": text[:160]},
        })
    _cluster_ignored(events)
    return events


def _cluster_ignored(events):
    """Streaks of ignored requests by the same sender boost failure scores."""
    ign = defaultdict(list)
    for e in events:
        if e["detail"]["status"] == "ignored":
            ign[(e["ep"], e["agents"][0])].append(e)
    for seq in ign.values():
        seq.sort(key=lambda e: e["t0"])
        i = 0
        while i < len(seq):
            j = i
            while (j + 1 < len(seq)
                   and seq[j + 1]["t0"] - seq[j]["t0"] <= CLUSTER_GAP):
                j += 1
            n = j - i + 1
            for e in seq[i:j + 1]:
                e["detail"]["ignored_streak"] = n
                e["score"] = n + (0.5 if e["detail"]["coop_talk"] else 0.0)
            i = j + 1


def coop_milestone_events(ctx):
    events = []
    for (ep, t, mid, contribs, rew, cbc) in real_coop_events(ctx):
        events.append({
            "type": "coop_milestone", "success": True,
            **_prov(ctx, ep, max(0, t - HORIZON), t + 5, contribs),
            "score": rew / 10.0 + (5.0 if cbc else 0.0) + len(contribs),
            "W": (_bond_W(ctx, ep, contribs[0], contribs[1])
                  if len(contribs) >= 2 else None),
            "detail": {"milestone": mid, "reward": rew,
                       "comm_before_coop": cbc, "t_milestone": t},
        })
    return events


def mutual_deadlocks(ctx):
    """Ch3 pairs asking each other to press with no real press following."""
    msgs = [m for m in _directed_messages(ctx)
            if str(m[6] or "").startswith("ch3")
            and lexicons.SWITCH_TALK.search(m[4])]
    presses = {(ep, t) for (ep, t, mid, *_r) in real_coop_events(ctx)
               if mid.startswith("m17")}
    events = []
    seen = set()
    for (ep, t1, a, b, text1, *_x) in msgs:
        for (ep2, t2, a2, b2, text2, *_y) in msgs:
            if (ep2 != ep or (a2, b2) != (b, a)
                    or not 0 <= t2 - t1 <= DEADLOCK_WIN):
                continue
            key = (ep, frozenset((a, b)), t1 // (DEADLOCK_WIN * 2))
            if key in seen:
                continue
            if any(pe == ep and t1 <= pt <= t2 + DEADLOCK_WIN + 20
                   for (pe, pt) in presses):
                continue
            seen.add(key)
            events.append({
                "type": "mutual_deadlock", "success": False,
                **_prov(ctx, ep, max(0, t1 - 5), t2 + DEADLOCK_WIN, [a, b]),
                "score": 4.0, "W": _bond_W(ctx, ep, a, b),
                "detail": {"ask_ab": text1[:120], "ask_ba": text2[:120],
                           "t_ask_ab": t1, "t_ask_ba": t2},
            })
            break
    return events


def _densest_window(ts, width):
    """(t0, count) of the width-step window covering most timestamps."""
    if not ts:
        return None, 0
    ts = sorted(ts)
    best_t, best_n = ts[0], 1
    for i, t in enumerate(ts):
        n = sum(1 for u in ts[i:] if u <= t + width)
        if n > best_n:
            best_t, best_n = t, n
    return best_t, best_n


def chamber_failures(ctx):
    """ch3_stall + ch2_neglect from chamber entries vs coop progress.

    Windows anchor on the densest stretch of on-topic talk so transcripts
    show agents actively (and unsuccessfully) negotiating, not the chamber
    entry. Scores are bounded so these episode-level failures compete fairly
    with the message-level ones in exemplar selection.
    """
    coop_evs = real_coop_events(ctx)
    msgs = _directed_messages(ctx)
    series = _series_by_agent(ctx)
    events = []
    for ep, coop in ctx["coop_by_ep"].items():
        entries = coop.get("chamber_entry_steps") or {}
        final_step = int(coop.get("final_step") or 0)
        # Ch3: latency from entry to first real press
        e3 = entries.get("ch3")
        if e3 is not None:
            press = sorted(t for (ep2, t, mid, *_r) in coop_evs
                           if ep2 == ep and mid.startswith("m17"))
            latency = press[0] - int(e3) if press else None
            if latency is None or latency >= CH3_STALL_MIN:
                t_end = press[0] if press else min(int(e3) + 200,
                                                  final_step or int(e3) + 200)
                talk = [m for m in msgs
                        if m[0] == ep and int(e3) <= m[1] <= t_end
                        and lexicons.SWITCH_TALK.search(m[4])]
                anchor, _n = _densest_window([m[1] for m in talk], 60)
                t0 = anchor if anchor is not None else int(e3)
                stall = latency if latency is not None else t_end - int(e3)
                events.append({
                    "type": "ch3_stall", "success": False,
                    **_prov(ctx, ep, t0, min(t0 + 60, t_end),
                            sorted({m[2] for m in talk})),
                    "score": (2.0 + min(stall, 300) / 150.0
                              + min(len(talk), 50) / 25.0),
                    "W": None,
                    "detail": {"entry": int(e3), "latency": latency,
                               "n_switch_talk": len(talk),
                               "pressed": bool(press)},
                })
        # Ch2: reached but first anvil never broken
        e2 = entries.get("ch2")
        if e2 is not None and not any(
                ep2 == ep and mid.startswith(("m8_", "m9_"))
                for (ep2, _t, mid, *_r) in coop_evs):
            # only anvil talk sent FROM ch2 counts — agents routinely
            # mention anvils while planning (or hallucinating) elsewhere
            anvil_msgs = [m for m in msgs if m[0] == ep
                          and str(m[6] or "").startswith("ch2")
                          and re.search(r"\banvils?\b", m[4], re.I)]
            digs = sum(1 for (k, rows) in series.items() if k[0] == ep
                       for r in rows
                       if str(r.get("chamber") or "").startswith("ch2")
                       and r.get("action") == "Dig")
            anchor, _n = _densest_window([m[1] for m in anvil_msgs], 50)
            t0 = anchor if anchor is not None else int(e2)
            events.append({
                "type": "ch2_neglect", "success": False,
                **_prov(ctx, ep, t0, t0 + 50,
                        sorted({m[2] for m in anvil_msgs})),
                "score": (2.0 + min(len(anvil_msgs), 50) / 25.0
                          + min(digs, 200) / 200.0),
                "W": None,
                "detail": {"entry": int(e2), "n_anvil_msgs": len(anvil_msgs),
                           "n_ch2_digs": digs},
            })
    return events


def detect_events(ctx):
    return (request_outcomes(ctx) + coop_milestone_events(ctx)
            + mutual_deadlocks(ctx) + chamber_failures(ctx))


# ── aggregation ──────────────────────────────────────────────────────────


def run_level(events):
    req = [e for e in events if e["type"] == "request_outcome"]
    n_req = len(req)
    by = lambda s: sum(1 for e in req if e["detail"]["status"] == s)  # noqa: E731
    judged = [e for e in req if e["detail"]["complied"] is not None]
    coop = [e for e in events if e["type"] == "coop_milestone"]
    cbc = [e for e in coop if e["detail"]["comm_before_coop"]]
    stalls = [e for e in events if e["type"] == "ch3_stall"]
    lat = [e["detail"]["latency"] for e in stalls
           if e["detail"]["latency"] is not None]
    rec = {
        "n_requests": n_req,
        "frac_fulfilled": round(by("fulfilled") / n_req, 4) if n_req else None,
        "frac_answered": round(by("answered") / n_req, 4) if n_req else None,
        "frac_ignored": round(by("ignored") / n_req, 4) if n_req else None,
        "n_directives_judged": len(judged),
        "comply_rate": (round(sum(1 for e in judged if e["detail"]["complied"])
                              / len(judged), 4) if judged else None),
        "n_coop_success": len(coop),
        "frac_comm_before_coop": (round(len(cbc) / len(coop), 4)
                                  if coop else None),
        "n_deadlocks": sum(1 for e in events
                           if e["type"] == "mutual_deadlock"),
        "n_ch3_stalls": len(stalls),
        "ch3_stall_latency_mean": (round(sum(lat) / len(lat), 1)
                                   if lat else None),
        "n_ch2_neglect": sum(1 for e in events if e["type"] == "ch2_neglect"),
    }
    # bond-conditioned request outcomes (Hebbian runs only)
    w_f = [e["W"] for e in req if e["success"] and e["W"] is not None]
    w_i = [e["W"] for e in req
           if e["detail"]["status"] == "ignored" and e["W"] is not None]
    rec["mean_W_fulfilled"] = (round(sum(w_f) / len(w_f), 4) if w_f else None)
    rec["mean_W_ignored"] = (round(sum(w_i) / len(w_i), 4) if w_i else None)
    rec["n_W_fulfilled"], rec["n_W_ignored"] = len(w_f), len(w_i)
    return rec


_POOL_NUM = ["n_requests", "n_directives_judged", "n_coop_success",
             "n_deadlocks", "n_ch3_stalls", "n_ch2_neglect", "n_W_fulfilled",
             "n_W_ignored"]
_POOL_MEAN = ["frac_fulfilled", "frac_answered", "frac_ignored",
              "comply_rate", "frac_comm_before_coop",
              "ch3_stall_latency_mean", "mean_W_fulfilled", "mean_W_ignored"]


def within_episode_W_contrast(events):
    """Paired within-episode test of 'bonds track responsiveness': per
    episode with both outcomes, is mean W higher on fulfilled request edges
    than on ignored ones? Removes any cross-episode W-scale confound.
    Static-topology controls (Allied-*, No-bonds) tie by construction."""
    per = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e["type"] != "request_outcome" or e.get("W") is None:
            continue
        st = e["detail"]["status"]
        if st not in ("fulfilled", "ignored"):
            continue
        per[(e["exp"], e["seed"], e["ep"])][st].append(e["W"])
    wins = losses = ties = 0
    for v in per.values():
        if v["fulfilled"] and v["ignored"]:
            mf = sum(v["fulfilled"]) / len(v["fulfilled"])
            mi = sum(v["ignored"]) / len(v["ignored"])
            if abs(mf - mi) <= 1e-9:
                ties += 1
            elif mf > mi:
                wins += 1
            else:
                losses += 1
    return {"eps_W_fulfilled_higher": wins, "eps_W_fulfilled_lower": losses,
            "eps_W_tie": ties}


def condition_level(run_rows):
    """Pool run-level records of one condition: sums for counts, request-count
    weighted means for rates."""
    out = {"n_runs": len(run_rows)}
    for k in _POOL_NUM:
        out[k] = sum(r.get(k) or 0 for r in run_rows)
    for k in _POOL_MEAN:
        wkey = ("n_directives_judged" if k == "comply_rate"
                else "n_coop_success" if k == "frac_comm_before_coop"
                else "n_ch3_stalls" if k == "ch3_stall_latency_mean"
                else "n_W_fulfilled" if k == "mean_W_fulfilled"
                else "n_W_ignored" if k == "mean_W_ignored"
                else "n_requests")
        num = den = 0.0
        for r in run_rows:
            if r.get(k) is None:
                continue
            w = r.get(wkey) or 0
            num += r[k] * w
            den += w
        out[k] = round(num / den, 4) if den else None
    return out


# ── exemplar selection + report ──────────────────────────────────────────

MAX_EXEMPLARS = 3


def pick_exemplars(events_by_label):
    """Per condition and polarity: the best exemplar of EACH event type first
    (so deadlocks/ignored streaks are not drowned out by episode-level
    failures), then remaining slots by score; de-duplicated on
    (run, ep, ~anchor) so a fulfilled request and its milestone don't both
    render."""
    picks = []
    for label, evs in sorted(events_by_label.items()):
        for want_success in (True, False):
            cands = sorted((e for e in evs if e["success"] is want_success),
                           key=lambda e: -e["score"])
            best_by_type = {}
            for e in cands:
                best_by_type.setdefault(e["type"], e)
            pool = sorted(best_by_type.values(), key=lambda e: -e["score"])
            pool += [e for e in cands if e not in pool]
            chosen, seen = [], set()
            for e in pool:
                key = (e["exp"], e["seed"], e["ep"], e["t0"] // 30)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(e)
                if len(chosen) == MAX_EXEMPLARS:
                    break
            for rank, e in enumerate(chosen):
                kind = "success" if want_success else "failure"
                slug = f"collab_{kind}_{label}_{rank}".replace("+", "p")
                picks.append((slug, e))
    return picks


def _reason(e):
    d = e["detail"]
    if e["type"] == "request_outcome":
        return (f"{d['status']} {'/'.join(d['categories'])} "
                f"a{e['agents'][0]}->a{e['agents'][1]}"
                + (f" streak x{d['ignored_streak']}"
                   if d.get("ignored_streak") else "")
                + (f" -> {','.join(d['milestones'])}" if d["milestones"]
                   else "")
                + f": \"{d['text'][:80]}\"")
    if e["type"] == "coop_milestone":
        return (f"{d['milestone']} (+{d['reward']:g}) at t={d['t_milestone']}"
                f", comm_before_coop={d['comm_before_coop']}")
    if e["type"] == "mutual_deadlock":
        return (f"a{e['agents'][0]}<->a{e['agents'][1]} both ask, nobody "
                f"presses: \"{d['ask_ab'][:60]}\" / \"{d['ask_ba'][:60]}\"")
    if e["type"] == "ch3_stall":
        return (f"locked {d['latency'] if d['latency'] is not None else '>?'}"
                f" steps before {'a' if d['pressed'] else 'NO'} real press"
                f" ({d['n_switch_talk']} switch-talk msgs)")
    return (f"reached ch2, first anvil never broken "
            f"({d['n_anvil_msgs']} anvil msgs, {d['n_ch2_digs']} ch2 digs)")


def _md_table(rows, cols, headers=None):
    hdr = headers or cols
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            "" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return lines


def write_report(out: Path, cond_rows, pair_rows, picks, n_events):
    lines = [
        "# Collaboration case mining — success/failure episodes, "
        "Hebbian vs baseline",
        "",
        f"Detected **{n_events} events** across all runs "
        "(`collab/events.jsonl.gz`). A *request* is a valid directed "
        "message classified request/directive; it is **fulfilled** if the "
        "target performs the asked action within "
        f"{COMPLY_WIN} steps or a real (non-timer) cooperative milestone "
        f"fires within {HORIZON} steps, **answered** if it only gets a "
        "directed reply, **ignored** otherwise. Milestones within "
        f"±{TIMER_SLACK} steps of chamber-timer boundaries and entry "
        "milestones are never counted as successes. RL rows use the "
        "rl_thoughts comm stream; treat their message text as generated "
        "commentary, not the policy input.",
        "",
        "## Condition-level outcomes",
        "",
    ]
    cols = ["label", "n_runs", "n_requests", "frac_fulfilled",
            "frac_ignored", "comply_rate", "n_coop_success",
            "frac_comm_before_coop", "n_deadlocks", "n_ch3_stalls",
            "ch3_stall_latency_mean", "n_ch2_neglect"]
    lines += _md_table(cond_rows, cols)
    lines += [
        "",
        "## Matched pairs — baseline vs Hebbian",
        "",
    ]
    pcols = ["pair", "condition", "n_requests", "frac_fulfilled",
             "frac_ignored", "comply_rate", "n_coop_success",
             "frac_comm_before_coop", "n_deadlocks",
             "mean_W_fulfilled", "mean_W_ignored",
             "eps_W_fulfilled_higher", "eps_W_fulfilled_lower", "eps_W_tie"]
    lines += _md_table(pair_rows, pcols)
    lines += [
        "",
        "`mean_W_fulfilled` vs `mean_W_ignored` (Hebbian rows only): "
        "mean end-of-episode bond strength on the sender->target edge of "
        "fulfilled vs ignored requests. A gap in favour of fulfilled means "
        "bonds track who actually responds. `eps_W_fulfilled_higher/lower` "
        "is the paired within-episode version (episodes containing both "
        "outcomes), immune to cross-episode W-scale differences; "
        "static-topology controls tie by construction, which doubles as a "
        "sanity check of the measure.",
        "",
        "## Exemplar transcripts (`cases/collab/`)",
        "",
    ]
    for slug, e in picks:
        lines.append(f"- **{slug}** — `{e['exp']}/seed_{e['seed']}` "
                     f"ep{e['ep']} t{e['t0']}-{e['t1']}: {_reason(e)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── CLI entry ────────────────────────────────────────────────────────────


def run(args) -> int:
    runs = registry.iter_runs(args.runs_root, args.only)
    all_events = []
    run_rows = []
    events_by_label = defaultdict(list)
    ctxs = {}
    for run_ in runs:
        ctx = metrics_mod.load_ctx(run_, args.out)
        if ctx is None:
            print(f"[skip] {run_.run_id}: not parsed")
            continue
        if ctx["quarantined"]:
            print(f"[quarantined] {run_.run_id}: skipped")
            continue
        ctxs[run_.run_id] = ctx
        evs = detect_events(ctx)
        all_events.extend(evs)
        events_by_label[run_.label].extend(evs)
        rec = run_level(evs)
        run_rows.append({"run": run_.run_id, "label": run_.label,
                         "seed": run_.seed, "is_rl": run_.is_rl, **rec})
        print(f"[ok] {run_.run_id}: {len(evs)} events, "
              f"req={rec['n_requests']} fulfilled={rec['frac_fulfilled']} "
              f"coop={rec['n_coop_success']}")

    provenance.write_jsonl(args.out / "collab" / "events.jsonl.gz",
                           all_events)
    metrics_mod._write_csv(args.out / "tables" / "collab_run_level.csv",
                           run_rows,
                           ["run", "label", "seed", "is_rl"]
                           + _POOL_NUM[:1] + _POOL_MEAN[:4] + _POOL_NUM[1:]
                           + _POOL_MEAN[4:])

    by_label = defaultdict(list)
    for r in run_rows:
        by_label[r["label"]].append(r)
    cond_rows = [{"label": lab, **condition_level(rs),
                  **within_episode_W_contrast(events_by_label[lab])}
                 for lab, rs in sorted(by_label.items())]
    metrics_mod._write_csv(args.out / "tables" / "collab_condition.csv",
                           cond_rows)

    pair_rows = []
    for base, heb in MATCHED_PAIRS:
        for lab in (base, heb):
            if lab in by_label:
                pair_rows.append({"pair": f"{base} vs {heb}",
                                  "condition": lab,
                                  **condition_level(by_label[lab]),
                                  **within_episode_W_contrast(
                                      events_by_label[lab])})
    metrics_mod._write_csv(args.out / "tables" / "collab_matched_pairs.csv",
                           pair_rows)

    picks = pick_exemplars(events_by_label)
    case_dir = args.out / "cases" / "collab"
    shortlist = []
    for slug, e in picks:
        rid = f"{e['exp']}/seed_{e['seed']}"
        ctx = ctxs.get(rid)
        if ctx is None:
            continue
        render_transcript(ctx["run"], ctx["timeline"], e["ep"], e["t0"],
                          e["t1"], slug, _reason(e), case_dir)
        shortlist.append({"slug": slug, "type": e["type"],
                          "success": e["success"], "run": rid,
                          "label": e["label"], "ep": e["ep"],
                          "t0": e["t0"], "t1": e["t1"],
                          "score": round(e["score"], 2), "W": e["W"],
                          "reason": _reason(e)})
        print(f"[case] {slug}: {rid} ep{e['ep']} t{e['t0']}-{e['t1']}")
    metrics_mod._write_csv(case_dir / "shortlist.csv", shortlist,
                           ["slug", "type", "success", "label", "run", "ep",
                            "t0", "t1", "score", "W", "reason"])
    (case_dir / "shortlist.json").write_text(
        json.dumps(shortlist, indent=1), encoding="utf-8")

    write_report(args.out / "report" / "collab_report.md", cond_rows,
                 pair_rows, picks, len(all_events))
    print(f"collab done: {len(all_events)} events, {len(picks)} exemplars "
          f"-> {case_dir}, report -> {args.out / 'report' / 'collab_report.md'}")
    return 0
