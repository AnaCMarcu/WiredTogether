"""Ordinal step alignment + module attribution for parsed llm_log units.

Core idea: agents act strictly sequentially, so the i-th retry-0 action unit
(action_selection on_messages, or rl_thoughts in RL runs) of agent A is A's
i-th environment step. step_log.csv is the master step index (one row per
(step, agent) while alive). Wall-clock timestamps (1 s resolution) are used
only to attribute the agent-anonymous modules (critic / beliefs / curriculum /
episodic memory) to the enclosing turn window, validated against the known
module cadences from config.json.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections import defaultdict

from qual_lib import episode_io

ACTION_METHODS = {"on_messages", "rl_thoughts"}
_WS = re.compile(r"\s+")


def _norm_text(s) -> str:
    if not isinstance(s, str):
        return ""
    return _WS.sub(" ", s).strip().lower()


def _agent_idx(name) -> int:
    try:
        return int(str(name).split("_")[-1])
    except (ValueError, IndexError):
        return -1


# ── step index ───────────────────────────────────────────────────────────


def build_step_index(run):
    """Per-agent ordered step rows + lookup maps.

    Returns (per_agent_rows, row_map) where per_agent_rows[a] is the ordered
    list of (ep_1based, t, row) across episodes and row_map[(ep, t, a)] = row.

    Rows are clamped to final_metrics.episode_lengths when available: appended
    foreign attempts (e.g. a 2500-step job writing into the same episode dirs)
    leave phantom rows beyond the real episode length.
    """
    fm = episode_io.read_final_metrics(run) or {}
    ep_lens = [int(x) for x in fm.get("episode_lengths", [])]
    per_agent = defaultdict(list)
    row_map = {}
    for ep_i, ep_dir in enumerate(episode_io.valid_episode_dirs(run), start=1):
        rows = episode_io.read_steps(ep_dir)
        if ep_i <= len(ep_lens):
            rows = [r for r in rows if r["step"] < ep_lens[ep_i - 1]]
        rows.sort(key=lambda r: (r["step"], r["agent_id"]))
        for r in rows:
            a = r["agent_id"]
            per_agent[a].append((ep_i, r["step"], r))
            row_map[(ep_i, r["step"], a)] = r
    return per_agent, row_map


# ── action-unit alignment ────────────────────────────────────────────────


def action_units_per_agent(units):
    """Ordered action units per agent idx (order = line order in their file)."""
    per_agent = defaultdict(list)
    for u in units:
        if u["method"] in ACTION_METHODS and u.get("agent"):
            per_agent[_agent_idx(u["agent"])].append(u)
    for a in per_agent:
        per_agent[a].sort(key=lambda u: u["line_start"])
    return per_agent


def _match_rate(agent_units, agent_rows, offset):
    """Fraction of comparable turns whose comm text matches step_log message."""
    hits = total = 0
    for i, u in enumerate(agent_units):
        j = i + offset
        if j < 0 or j >= len(agent_rows):
            continue
        resp = u.get("response") or {}
        comm = _norm_text(resp.get("communication"))
        row_msg = _norm_text(agent_rows[j][2].get("message"))
        if not comm and not row_msg:
            continue
        total += 1
        if comm and row_msg and (comm == row_msg or comm in row_msg or row_msg in comm):
            hits += 1
    return (hits / total) if total else None, total


def _candidate_offsets(units, rows):
    """Offset hypotheses: identity ±2, tail-anchored ±2, text anchors.

    Tail anchor handles RELAUNCHED runs: llm_call's FileHandler appends, so an
    aborted first attempt leaves its units at the head of the log while
    episodes/ holds only the final attempt — the last len(rows) units are the
    real ones (offset = -(n_units - n_rows)). Text anchors match the first few
    row messages against unit comms anywhere in the stream.
    """
    cands = {0, -1, 1, -2, 2}
    extra = len(units) - len(rows)
    if extra > 0:
        for d in (-2, -1, 0, 1, 2):
            cands.add(-extra + d)
    # text anchors from the first rows with a non-trivial message
    anchored = 0
    for j, (_, _, row) in enumerate(rows[:50]):
        msg = _norm_text(row.get("message"))
        if len(msg) < 8:
            continue
        for i, u in enumerate(units):
            comm = _norm_text((u.get("response") or {}).get("communication"))
            if comm and (comm == msg or comm in msg or msg in comm):
                cands.add(j - i)
        anchored += 1
        if anchored >= 3:
            break
    return sorted(cands)


def align_actions(per_agent_units, per_agent_rows):
    """Choose a per-agent global offset maximizing comm-text match.

    Returns {agent: {"offset", "match_rate", "n_compared", "n_units",
    "n_rows", "status"}}. status: exact | repaired | unaligned | no_text.
    """
    report = {}
    for a, units in sorted(per_agent_units.items()):
        rows = per_agent_rows.get(a, [])
        best = (None, -1.0, 0)
        for d in _candidate_offsets(units, rows):
            rate, n = _match_rate(units, rows, d)
            if rate is not None and rate > best[1]:
                best = (d, rate, n)
        offset, rate, n = best
        if offset is None:
            status, offset, rate = "no_text", 0, None
        elif rate >= 0.98 and offset == 0:
            status = "exact"
        elif rate >= 0.90:
            status = "repaired" if offset != 0 else "exact"
        else:
            status, offset = "unaligned", 0
        report[a] = {
            "offset": offset, "match_rate": rate, "n_compared": n,
            "n_units": len(units), "n_rows": len(rows), "status": status,
        }
    return report


# ── anonymous-module attribution ─────────────────────────────────────────


def attribute_anonymous(units, per_agent_units):
    """Assign agent-anonymous units to the enclosing action-turn window.

    Global turn order = all action units sorted by (ts_last, line_start) —
    valid because agents act sequentially and all action units live in one
    file. An anonymous unit with first-timestamp T belongs to the first
    action unit whose response time is >= T (modules run before their turn's
    action response). Confidence drops to "med" on same-second boundaries.

    Mutates each anonymous unit: adds agent_attr, turn_of_agent, attr_conf.
    Returns count stats.
    """
    actions = []
    for a, us in per_agent_units.items():
        for i, u in enumerate(us):
            actions.append((u["ts_last"], u["line_start"], a, i))
    actions.sort()
    keys = [x[0] for x in actions]

    stats = {"attributed": 0, "unattributed": 0, "low_conf": 0}
    for u in units:
        if u["method"] in ACTION_METHODS or u["module"] in (
            "action_selection", "rl_thoughts"
        ):
            continue
        if u.get("agent"):  # SocialModule[agent_N] carries its own agent
            u["agent_attr"] = _agent_idx(u["agent"])
            u["attr_conf"] = "high"
            stats["attributed"] += 1
            continue
        k = bisect_left(keys, u["ts"])
        if k >= len(actions):
            u["agent_attr"] = None
            u["attr_conf"] = "none"
            stats["unattributed"] += 1
            continue
        ts_k, _, agent_k, turn_k = actions[k]
        u["agent_attr"] = agent_k
        u["turn_of_agent"] = turn_k
        # boundary tie: unit ts equals PREVIOUS turn's response second
        if k > 0 and actions[k - 1][0] == u["ts"]:
            u["attr_conf"] = "med"
            stats["low_conf"] += 1
        else:
            u["attr_conf"] = "high"
        stats["attributed"] += 1
    return stats


def social_units_turns(units, per_agent_units):
    """Map SocialModule units to turn indices via their agent's action stream.

    Social units carry their agent in the prefix but not the turn; assign the
    turn whose action window contains them (same rule as attribute_anonymous,
    restricted to that agent's actions).
    """
    for a, us in per_agent_units.items():
        keys = [u["ts_last"] for u in us]
        for su in units:
            if su["module"] != "social_module" or _agent_idx(su.get("agent")) != a:
                continue
            k = bisect_left(keys, su["ts"])
            su["turn_of_agent"] = k if k < len(us) else None


def map_units_to_rows(per_agent_units, per_agent_rows, align):
    """Per-agent {row_idx: (unit_idx, unit)} under the chosen strategy.

    - status exact/repaired: ordinal mapping row = unit_idx + offset.
    - status clock: units carry u["clock"] = [ep, t] (from stepclock);
      the LAST unit clocked into a step wins (driver retries emit several
      units for one step; the final one produced the executed decision).
    """
    mapping = {}
    for a, units in per_agent_units.items():
        rows = per_agent_rows.get(a, [])
        rep = align.get(a, {})
        m: dict = {}
        if rep.get("status") in ("exact", "repaired"):
            offset = rep.get("offset", 0)
            for i, u in enumerate(units):
                j = i + offset
                if 0 <= j < len(rows):
                    m[j] = (i, u)
        elif rep.get("status") == "clock":
            row_by_ept = {(ep, t): j for j, (ep, t, _r) in enumerate(rows)}
            for i, u in enumerate(units):
                ck = u.get("clock")
                if not ck:
                    continue
                j = row_by_ept.get((ck[0], ck[1]))
                if j is not None:
                    m[j] = (i, u)  # later units overwrite -> last wins
        mapping[a] = m
        rep["coverage"] = round(len(m) / len(rows), 4) if rows else None
    return mapping


def promote_clock_alignment(per_agent_units, per_agent_rows, align,
                            min_coverage: float = 0.75):
    """Upgrade unaligned agents to clock alignment when coverage suffices.

    Call after stepclock.assign_clock has stamped u["clock"]. Mutates align.
    """
    for a, rep in align.items():
        if rep.get("status") != "unaligned":
            continue
        rows = per_agent_rows.get(a, [])
        if not rows:
            continue
        row_keys = {(ep, t) for ep, t, _r in rows}
        clocked = {tuple(u["clock"]) for u in per_agent_units.get(a, [])
                   if u.get("clock")}
        cov = len(clocked & row_keys) / len(row_keys)
        if cov >= min_coverage:
            rep["status"] = "clock"
            rep["clock_coverage"] = round(cov, 4)


def mark_stale(units, per_agent_units, mapping, align):
    """Flag units from aborted earlier attempts (appended llm_logs).

    Action units: stale unless they won a row in ``mapping``. Anonymous
    units: stale when their attributed turn's action unit is stale (text
    mode) or their clock is unset (clock mode). Metrics reading
    llm_calls.jsonl directly must filter on ``stale``.
    """
    mapped_ids = {a: {id(u) for (_i, u) in m.values()}
                  for a, m in mapping.items()}
    unit_pos = {}
    for a, us in per_agent_units.items():
        for i, u in enumerate(us):
            unit_pos[id(u)] = (a, i)

    n_stale = 0
    for u in units:
        pos = unit_pos.get(id(u))
        if pos is not None:  # action unit
            a, _i = pos
            u["stale"] = id(u) not in mapped_ids.get(a, set())
        else:
            a, i = u.get("agent_attr"), u.get("turn_of_agent")
            if a is None:
                u["stale"] = None
                continue
            mode = align.get(a, {}).get("status")
            if mode == "clock":
                u["stale"] = not u.get("clock")
            elif i is None:
                u["stale"] = None
                continue
            else:
                aus = per_agent_units.get(a, [])
                u["stale"] = (i >= len(aus)
                              or id(aus[i]) not in mapped_ids.get(a, set()))
        if u["stale"]:
            n_stale += 1
    return n_stale


# ── cadence validation ───────────────────────────────────────────────────


def cadence_report(units, intervals):
    """Fraction of attributed module units landing on expected turn residues.

    Expected (1-based turn = turn_of_agent+1): perception/interaction beliefs
    at turn % belief_interval == 0; critic at % critic_interval == 0; social
    at % social_interval == 0 or turn 1 (first-call exception).
    """
    def frac_on(pred, module, methods=None):
        ok = tot = 0
        for u in units:
            if u["module"] != module:
                continue
            if methods and u["method"] not in methods:
                continue
            t = u.get("turn_of_agent")
            if t is None:
                continue
            tot += 1
            if pred(t + 1):
                ok += 1
        return (ok / tot if tot else None), tot

    bi, ci, si = (intervals["belief_interval"], intervals["critic_interval"],
                  intervals["social_interval"])
    out = {}
    out["belief_perception"] = frac_on(
        lambda n: n % bi == 0, "belief_system", {"create_perception_beliefs"})
    out["critic"] = frac_on(lambda n: n % ci == 0, "critic")
    out["social"] = frac_on(lambda n: n == 1 or n % si == 0, "social_module")
    return {k: {"rate": v[0], "n": v[1]} for k, v in out.items()}


# ── timeline assembly ────────────────────────────────────────────────────


def build_timeline(run, units, per_agent_units, per_agent_rows, align, mapping):
    """One row per (ep, t, agent): step_log master + everything joined on.

    ``mapping`` comes from map_units_to_rows (text-ordinal or clock strategy);
    agents with align status unaligned keep env fields with llm fields None.
    """
    # messages: (ep, t, sender_idx) -> record
    msg_map = {}
    for ep_i, ep_dir in enumerate(episode_io.valid_episode_dirs(run), start=1):
        for m in episode_io.read_messages(ep_dir):
            msg_map[(ep_i, int(m.get("t", -1)), _agent_idx(m.get("sender")))] = m
    # events: (ep, step) -> [records]
    ev_map = defaultdict(list)
    for ep_i, ep_dir in enumerate(episode_io.valid_episode_dirs(run), start=1):
        for e in episode_io.read_events(ep_dir):
            try:
                ev_map[(ep_i, int(e.get("step", -1)))].append(e)
            except (TypeError, ValueError):
                continue

    # module units by (agent, turn)
    by_turn = defaultdict(dict)
    for u in units:
        a = u.get("agent_attr")
        t = u.get("turn_of_agent")
        if a is None or t is None:
            continue
        mod, meth = u["module"], u["method"]
        resp = u.get("response") or {}
        if mod == "critic":
            by_turn[(a, t)]["critic"] = {
                "success": resp.get("success"),
                "critique": resp.get("critique"),
                "reasoning": resp.get("reasoning"),
                "conf": u.get("attr_conf"),
            }
        elif mod == "auto_curriculum" and "task" in resp:
            by_turn[(a, t)]["task_new"] = resp.get("task")
        elif mod == "belief_system":
            b = by_turn[(a, t)].setdefault("beliefs", {})
            key = {
                "create_perception_beliefs": "perception",
                "update_partner_beliefs": "partner",
                "update_interaction_beliefs": "interaction",
            }.get(u["method"], u["method"])
            val = resp.get("beliefs")
            if key == "partner":
                b.setdefault("partner", []).append(val)
            else:
                b[key] = val
        elif mod == "social_module":
            by_turn[(a, t)]["social"] = {**resp, "fresh": True}
        elif mod == "episodic_memory":
            by_turn[(a, t)]["episode_summary"] = resp.get("summary")

    rows_out = []
    for a, arows in sorted(per_agent_rows.items()):
        rep = align.get(a, {})
        aligned = rep.get("status") in ("exact", "repaired", "clock")
        unit_by_rowidx = mapping.get(a, {}) if aligned else {}
        task = None
        for j, (ep_i, t, row) in enumerate(arows):
            iu = unit_by_rowidx.get(j)
            turn_mods = {}
            thoughts = llm_action = comm = comm_target = None
            parse_ok = None
            if iu is not None:
                i, u = iu
                resp = u.get("response") or {}
                thoughts = resp.get("thoughts")
                llm_action = resp.get("action")
                comm = resp.get("communication")
                comm_target = resp.get("communication_target")
                parse_ok = u.get("parse_ok")
                turn_mods = by_turn.get((a, i), {})
                if "task_new" in turn_mods:
                    task = turn_mods["task_new"]
            m = msg_map.get((ep_i, t, a))
            rows_out.append({
                "exp": run.exp, "seed": run.seed, "ep": ep_i, "t": t,
                "agent": a, "chamber": row.get("chamber"),
                "pos": [row.get("pos_x"), row.get("pos_y"), row.get("pos_z")],
                "action": row.get("action"),
                "reward_task": row.get("reward_task"),
                "reward_comm": row.get("reward_comm"),
                "hp": row.get("hp"), "wielded": row.get("wielded_item"),
                "msg": ({
                    "text": m.get("text"), "receiver": m.get("receiver"),
                    "model_target": m.get("model_target"),
                    "model_target_canonical": m.get("model_target_canonical"),
                    "routing": m.get("routing"), "valid": m.get("valid"),
                    "rewarded_base": m.get("rewarded_base"),
                    "rewarded_milestone": m.get("rewarded_milestone"),
                } if m else None),
                "thoughts": thoughts, "llm_action": llm_action,
                "llm_comm": comm, "llm_comm_target": comm_target,
                "parse_ok": parse_ok,
                "task": task,
                "task_new": turn_mods.get("task_new") is not None or None,
                "critic": turn_mods.get("critic"),
                "beliefs": turn_mods.get("beliefs"),
                "social": turn_mods.get("social"),
                "events": ev_map.get((ep_i, t)) or None,
                "aligned": aligned,
            })
    rows_out.sort(key=lambda r: (r["ep"], r["t"], r["agent"]))
    return rows_out
