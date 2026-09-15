"""Who messaged whom / who co-earned milestones — the "wired together" readout.

Phase B's question is whether transplanted pairs keep preferring each other.
The raw evidence already exists in every run:

  * ``episodes/*/messages.jsonl`` — one record per delivered message with
    ``t, sender, receiver, chamber`` (sender/receiver as ``agent_N``);
  * ``final_metrics.json -> milestone_events`` — one record per contributor
    with ``step, milestone_id, contributor`` (Lua spells it ``agentN``,
    Python ``agent_N`` — normalized here). Contributors sharing the same
    (milestone_id, step) co-earned that milestone.

This module turns those streams into per-dyad statistics:

  * directed message matrix M[i][j] = messages i -> j (per episode + total);
  * seatmate preference P(target = seatmate) per agent — chance level is
    1/(N-1), so 0.2 at N=6: the single cleanest "wiring" number;
  * co-milestone matrix C[i][j] = milestones i and j earned together,
    with the ``selective`` variant dropping all-hands milestones (a
    milestone every agent earned carries no pairing information);
  * a seat-pair table labelled GENUINE / CONTROL / strangers from the merged
    manifest's ``seat_pairs`` provenance, so the built-in control dyad is
    read out explicitly.

CLI:
  python src/mindforge/tools/analyze_wiring.py <run_dir> \
      [--manifest <merged_manifest.json>] [--per-episode]

Pure stdlib; importable and testable anywhere.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_AGENT_IDX_RE = re.compile(r"^agent[_ ]?(\d+)$", re.IGNORECASE)


def agent_index(name):
    """'agent_3' / 'agent3' / 'Agent 3' -> 3; None if not an agent name."""
    m = _AGENT_IDX_RE.match(str(name).strip())
    return int(m.group(1)) if m else None


def load_message_matrix(run_dir, num_agents):
    """Per-episode + total directed message counts from messages.jsonl.

    Returns (total, per_episode) where total is an NxN nested list and
    per_episode maps episode name -> NxN. Invalid/unparseable senders or
    receivers are skipped, not guessed.
    """
    run_dir = Path(run_dir)
    total = [[0] * num_agents for _ in range(num_agents)]
    per_episode = {}
    for mfile in sorted(run_dir.glob("episodes/*/messages.jsonl")):
        ep = mfile.parent.name
        mat = [[0] * num_agents for _ in range(num_agents)]
        with open(mfile, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = agent_index(rec.get("sender"))
                r = agent_index(rec.get("receiver"))
                if s is None or r is None or s == r:
                    continue
                if s >= num_agents or r >= num_agents:
                    continue
                mat[s][r] += 1
                total[s][r] += 1
        per_episode[ep] = mat
    return total, per_episode


def seatmate_preference(matrix, seat_of_partner=None):
    """P(target = seatmate) per agent from a directed message matrix.

    seat_of_partner: optional {agent: seatmate}; defaults to the standard
    seating (0<->1, 2<->3, 4<->5). Chance level under uniform targeting is
    1/(N-1). Returns {agent: (preference or None, n_messages)}.
    """
    n = len(matrix)
    if seat_of_partner is None:
        seat_of_partner = _standard_seating(n)
    out = {}
    for i in range(n):
        sent = sum(matrix[i])
        mate = seat_of_partner.get(i)
        pref = (matrix[i][mate] / sent) if (sent and mate is not None) else None
        out[i] = (pref, sent)
    return out


def _standard_seating(n):
    """Seat pairs (0,1), (2,3), ... as {agent: seatmate}."""
    return {i: i + 1 if i % 2 == 0 else i - 1 for i in range(n)}


def ring_neighbours(i, n):
    """Agent i's two task-adjacent seats under WIRE's Chamber-3 ring."""
    return {(i + 1) % n, (i - 1) % n}


def interior_seats(n):
    """Seats whose two ring neighbours are also spatially symmetric.

    The Ch4/Ch5 rescue spawns lay agents along one row in seat order, so the
    two END seats have one ring neighbour beside them and one at the far end
    of the row. Every other seat sits between its two ring neighbours.
    """
    return [i for i in range(n) if i not in (0, n - 1)]


def ring_conditioned_preference(matrix, seat_of_partner=None):
    """P(target = seatmate | target is a ring neighbour), per agent.

    Raw seatmate preference mixes relationship with task geometry. WIRE wires
    switch i to the door of agent (i+1) mod N (mods/wire/switches.lua), and
    the Ch4/Ch5 rescue spawns lay agents out along a row in seat order, so an
    agent's seatmate is always one of its two ring neighbours. An agent that
    holds no relationship at all, and simply talks to whoever can open its
    door or whoever is standing next to it, therefore already scores far above
    the 1/(N-1) chance line.

    Conditioning on the message having gone to a ring neighbour removes that
    advantage: every agent has exactly two ring neighbours and exactly one of
    them is its seatmate, so chance is 0.50 for every N. What is left is which
    of two equally task-relevant partners the agent chose.

    Caveat, and why interior_seats() exists: the spawn row is linear in seat
    index, so for the two END seats the seatmate is spatially adjacent while
    the other ring neighbour is at the far end of the row. The measure is
    geometry-free for the interior seats and mildly seatmate-favouring for the
    two end ones. Report the interior-only mean beside it as the robustness
    line.

    Returns {agent: (preference or None, n_ring_messages)}.
    """
    n = len(matrix)
    if seat_of_partner is None:
        seat_of_partner = _standard_seating(n)
    out = {}
    for i in range(n):
        ring = ring_neighbours(i, n)
        mate = seat_of_partner.get(i)
        # Undefined when the ring collapses (n < 4) or the seating is not
        # index-adjacent: there is then no two-way choice to measure.
        if len(ring) < 2 or mate is None or mate not in ring:
            out[i] = (None, 0)
            continue
        sent_ring = sum(matrix[i][j] for j in ring)
        pref = (matrix[i][mate] / sent_ring) if sent_ring else None
        out[i] = (pref, sent_ring)
    return out


def mean_preference(prefs, agents=None):
    """Mean over a preference dict, optionally restricted to `agents`."""
    vals = [p for i, (p, _) in prefs.items()
            if p is not None and (agents is None or i in agents)]
    return sum(vals) / len(vals) if vals else None


def load_co_milestone_matrix(run_dir, num_agents, selective=True):
    """NxN symmetric co-contribution counts from milestone_events.

    Contributors are grouped by (milestone_id, step); every pair inside a
    group co-earned that milestone. selective=True drops groups in which ALL
    agents contributed — an all-hands milestone (m_comm_*, all-in-communal)
    says nothing about who paired with whom.
    """
    try:
        with open(Path(run_dir) / "final_metrics.json") as f:
            events = json.load(f).get("milestone_events") or []
    except Exception:
        events = []

    groups = defaultdict(set)
    for e in events:
        idx = agent_index(e.get("contributor"))
        if idx is not None and idx < num_agents:
            groups[(e.get("milestone_id"), e.get("step"))].add(idx)

    mat = [[0] * num_agents for _ in range(num_agents)]
    for (_mid, _step), members in groups.items():
        if len(members) < 2:
            continue
        if selective and len(members) >= num_agents:
            continue
        for i in members:
            for j in members:
                if i != j:
                    mat[i][j] += 1
    return mat


def seat_pair_table(msg_matrix, co_matrix, seat_pairs=None):
    """Per-seat-pair wiring summary, labelled from manifest provenance.

    seat_pairs: the merged manifest's ``seat_pairs`` list (or None -> infer
    standard seating with no labels). Returns a list of dicts with the
    within-dyad message count, each member's seatmate preference, and the
    co-milestone count.
    """
    n = len(msg_matrix)
    if seat_pairs is None:
        seat_pairs = [{"seats": [k, k + 1], "same_source_run": None,
                       "cofired": None} for k in range(0, n, 2)]
    prefs = seatmate_preference(msg_matrix)
    rows = []
    for sp in seat_pairs:
        a, b = sp["seats"]
        if sp.get("same_source_run") is False:
            label = "strangers"
        elif sp.get("cofired") is True:
            label = "GENUINE"
        elif sp.get("cofired") is False:
            label = "CONTROL"
        else:
            label = "?"
        rows.append({
            "seats": [a, b],
            "label": label,
            "messages_within": msg_matrix[a][b] + msg_matrix[b][a],
            "pref_a": prefs[a][0],
            "pref_b": prefs[b][0],
            "co_milestones": co_matrix[a][b],
        })
    return rows


def _fmt_matrix(mat, title):
    n = len(mat)
    lines = [title, "      " + " ".join(f"ag{j:<4}" for j in range(n))]
    for i in range(n):
        lines.append(f"  ag{i} " + " ".join(f"{v:<5}" for v in mat[i]))
    return "\n".join(lines)


def analyze_run(run_dir, manifest_path=None, per_episode=False, out=sys.stdout):
    run_dir = Path(run_dir)
    seat_pairs = None
    if manifest_path:
        with open(manifest_path) as f:
            manifest = json.load(f)
        num_agents = manifest.get("num_agents", 6)
        seat_pairs = manifest.get("seat_pairs")
    else:
        try:
            with open(run_dir / "config.json") as f:
                cfg = json.load(f)
            num_agents = (cfg.get("num_agents")
                          or cfg.get("cli_args", {}).get("num_agents") or 6)
        except Exception:
            num_agents = 6

    total, episodes = load_message_matrix(run_dir, num_agents)
    co_sel = load_co_milestone_matrix(run_dir, num_agents, selective=True)
    co_all = load_co_milestone_matrix(run_dir, num_agents, selective=False)

    chance = 1.0 / (num_agents - 1) if num_agents > 1 else None
    print(f"run: {run_dir}", file=out)
    print(_fmt_matrix(total, "messages i->j (all episodes):"), file=out)
    if per_episode:
        for ep, mat in episodes.items():
            print(_fmt_matrix(mat, f"messages i->j ({ep}):"), file=out)
    print(_fmt_matrix(co_sel,
                      "co-milestones (selective, all-hands dropped):"),
          file=out)
    print(f"\nseatmate preference (chance = {chance:.2f}):", file=out)
    prefs = seatmate_preference(total)
    for i, (pref, sent) in prefs.items():
        p = f"{pref:.2f}" if pref is not None else " -  "
        print(f"  agent_{i}: {p}  ({sent} msgs sent)", file=out)
    m = mean_preference(prefs)
    print(f"  mean: {m:.2f}" if m is not None else "  mean: -", file=out)

    # Geometry-free companion. WIRE's Ch3 ring and its index-ordered spawn
    # rows make an agent's seatmate a task partner, which lifts the raw
    # number in every arm. See ring_conditioned_preference.
    ring = ring_conditioned_preference(total)
    print("", file=out)
    print("ring-conditioned preference "
          "(P(seatmate | ring neighbour), chance = 0.50):", file=out)
    for i, (pref, sent) in ring.items():
        p = f"{pref:.2f}" if pref is not None else " -  "
        print(f"  agent_{i}: {p}  ({sent} msgs to a ring neighbour)", file=out)
    rm = mean_preference(ring)
    ri = mean_preference(ring, agents=set(interior_seats(num_agents)))
    print(f"  mean: {rm:.2f}" if rm is not None else "  mean: -", file=out)
    print(f"  mean over interior seats {interior_seats(num_agents)}: "
          + (f"{ri:.2f}" if ri is not None else "-"), file=out)
    print("\nseat pairs:", file=out)
    for row in seat_pair_table(total, co_sel, seat_pairs):
        pa = f"{row['pref_a']:.2f}" if row["pref_a"] is not None else "-"
        pb = f"{row['pref_b']:.2f}" if row["pref_b"] is not None else "-"
        print(f"  seats {row['seats']} [{row['label']:<9}] "
              f"msgs_within={row['messages_within']:<5} "
              f"pref=({pa},{pb}) co_milestones={row['co_milestones']}",
              file=out)
    all_hands = sum(sum(r) for r in co_all) - sum(sum(r) for r in co_sel)
    if all_hands:
        print(f"\n({all_hands // 2} all-hands co-contribution pairs dropped "
              f"by the selective filter)", file=out)
    return {"messages": total, "co_milestones": co_sel,
            "seatmate_preference": prefs,
            "ring_conditioned_preference": ring,
            "seat_pairs": seat_pair_table(total, co_sel, seat_pairs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default=None,
                    help="merged_manifest.json for seat-pair labels "
                         "(GENUINE / CONTROL / strangers)")
    ap.add_argument("--per-episode", action="store_true")
    args = ap.parse_args(argv)
    analyze_run(args.run_dir, args.manifest, args.per_episode)


if __name__ == "__main__":
    main()
