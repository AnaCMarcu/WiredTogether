"""Dimension 1 — communication content metrics (deterministic)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from qual_lib import lexicons

_WS = re.compile(r"\s+")


def _norm(s) -> str:
    return _WS.sub(" ", str(s or "")).strip().lower()


def message_records(ctx) -> list:
    """All routed messages of a run with chamber + validity + text."""
    msgs = []
    for ep_i, ep_msgs in ctx["messages"].items():
        for m in ep_msgs:
            msgs.append({**m, "ep": ep_i})
    return msgs


def per_run(ctx) -> dict:
    """Category mix, dup rates, referential density, routing, Ch3 stats."""
    msgs = message_records(ctx)
    valid = [m for m in msgs if m.get("valid")]
    n_steps = ctx["n_team_steps"] or 1

    cat_by_ch = defaultdict(Counter)   # chamber -> Counter(category)
    n_by_ch = Counter()
    routing_by_ch = defaultdict(Counter)
    naming = Counter()
    texts_norm = []
    for m in msgs:
        ch = m.get("chamber") or "?"
        routing_by_ch[ch][m.get("routing") or "?"] += 1
        if not m.get("valid"):
            continue
        text = m.get("text") or ""
        n_by_ch[ch] += 1
        for c in lexicons.categorize_message(text):
            cat_by_ch[ch][c] += 1
        if lexicons.TEAMMATE_NAME.search(text):
            naming["teammate"] += 1
        if lexicons.ENV_OBJECT.search(text):
            naming["object"] += 1
        if re.search(r"\d", text):
            naming["numeral"] += 1
        texts_norm.append(_norm(text))

    dup = 0
    seen = Counter(texts_norm)
    dup_rate = (sum(c - 1 for c in seen.values() if c > 1) / len(texts_norm)
                if texts_norm else None)

    # Ch3: switch talk + request->act chains vs silent presses
    ch3_msgs = [m for m in valid if str(m.get("chamber", "")).startswith("ch3")]
    switch_talk = [m for m in ch3_msgs
                   if lexicons.SWITCH_TALK.search(m.get("text") or "")]
    presses = []  # (ep, t) of m17_switch_pressed / m18_door_opened events
    for ep_i, evs in ctx["events"].items():
        for e in evs:
            mid = e.get("id") or e.get("milestone") or ""
            if mid.startswith(("m17_", "m18_")):
                presses.append((ep_i, int(e.get("step", -1)), mid))
    chains, latencies = 0, []
    chained_press_keys = set()
    for m in switch_talk:
        for (ep_i, t, mid) in presses:
            if ep_i == m["ep"] and 0 <= t - int(m.get("t", -1)) <= 20:
                chains += 1
                latencies.append(t - int(m["t"]))
                chained_press_keys.add((ep_i, t, mid))
                break
    silent_presses = len({p for p in presses}) - len(chained_press_keys)
    latencies.sort()
    lat_med = latencies[len(latencies) // 2] if latencies else None

    return {
        "n_messages": len(msgs), "n_valid": len(valid),
        "msgs_per_100_steps": round(100 * len(valid) / n_steps, 2),
        "dup_rate": round(dup_rate, 4) if dup_rate is not None else None,
        "teammate_naming_rate": round(naming["teammate"] / len(valid), 4) if valid else None,
        "object_naming_rate": round(naming["object"] / len(valid), 4) if valid else None,
        "numeral_rate": round(naming["numeral"] / len(valid), 4) if valid else None,
        "categories_by_chamber": {ch: dict(c) for ch, c in cat_by_ch.items()},
        "valid_by_chamber": dict(n_by_ch),
        "routing_by_chamber": {ch: dict(c) for ch, c in routing_by_ch.items()},
        "ch3_switch_talk": len(switch_talk),
        "ch3_msgs": len(ch3_msgs),
        "ch3_chains": chains,
        "ch3_chain_latency_median": lat_med,
        "ch3_silent_presses": silent_presses,
        "_valid_texts": [m.get("text") or "" for m in valid],  # for clustering
    }
