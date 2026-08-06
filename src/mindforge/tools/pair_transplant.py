"""Pure logic for merging Phase A pair-bonding runs into a Phase B transplant.

Phase A produces 2-agent runs, each with a final 2x2 Hebbian W
(``hebbian_graph_final.json``) and per-agent cognitive state
(``agent_state/agent_{0,1}.json``, see agent_modules.agent_state_io). This
module builds the 6-agent merge:

  * ``merge_hebbian_W``       — block-diagonal 6x6 W with per-block
                                normalization so no pair starts advantaged;
  * ``build_slot_assignment`` — which source agent sits in which of the six
                                seats (identity, or seeded shuffle for the
                                shuffled-transplant control);
  * ``build_merged_manifest`` — per-seat cognitive state with every agent
                                name in free text remapped to the new seats;
  * ``rank_pair_runs``        — ranking table for picking the top pair runs.

Only stdlib + numpy: importable and testable anywhere.
"""

import json
import re
from pathlib import Path

import numpy as np

# Matches agent references in free text: "agent_0", "Agent_1", "agent 2",
# "AGENT_0", "agent0". Word-bounded so "management_0" or "agentx" never match.
_AGENT_REF_RE = re.compile(r"\bagent[_ ]?(\d+)\b", re.IGNORECASE)


def remap_agent_names(text, mapping):
    """Rewrite agent references in ``text`` per ``mapping`` ({old: new} ints).

    Single-pass, so simultaneous swaps are safe (0→1 and 1→0 never chain).
    Indices absent from ``mapping`` are left untouched. Matches are emitted
    in the canonical ``agent_{new}`` form regardless of the source spelling.
    """
    if not text:
        return text

    def _sub(m):
        idx = int(m.group(1))
        if idx in mapping:
            return f"agent_{mapping[idx]}"
        return m.group(0)

    return _AGENT_REF_RE.sub(_sub, str(text))


def merge_hebbian_W(pair_Ws, cross_weight=0.1, normalize="block_mean"):
    """Merge per-pair 2x2 W matrices into one (2K x 2K) start matrix.

    Pair k's block lands at seats (2k, 2k+1). Off-block entries are
    ``cross_weight`` — which must be > 0: W=0 is a fixed point of the gated
    Hebbian rule, so zero cross-pair bonds could never grow and would rig
    the persistence comparison.

    normalize="block_mean" rescales each block so its off-diagonal mean
    equals the mean over all blocks (source runs end at different overall W
    scales; without this the strongest pair starts Phase B with a built-in
    advantage). Within-block asymmetry (W01 vs W10 ratio) is preserved.
    normalize="none" keeps faithful magnitudes.

    Returns ``(W, provenance)`` where W is a float ndarray clipped to [0, 1]
    and provenance records the per-block scales and rescale factors.
    """
    if cross_weight <= 0.0:
        raise ValueError(
            f"cross_weight must be > 0 (got {cross_weight}): W=0 is a fixed "
            "point of the gated Hebbian rule — zero cross-pair bonds could "
            "never grow"
        )
    if normalize not in ("block_mean", "none"):
        raise ValueError(f"unknown normalize mode: {normalize!r}")

    blocks = [np.asarray(w, dtype=float) for w in pair_Ws]
    for k, b in enumerate(blocks):
        if b.shape != (2, 2):
            raise ValueError(f"pair {k}: expected a 2x2 W, got shape {b.shape}")

    block_means = [float((b[0, 1] + b[1, 0]) / 2.0) for b in blocks]
    if normalize == "block_mean":
        target = float(np.mean(block_means))
        factors = [
            (target / m) if m > 0 else 1.0 for m in block_means
        ]
    else:
        factors = [1.0] * len(blocks)

    n = 2 * len(blocks)
    W = np.full((n, n), float(cross_weight), dtype=float)
    for k, (b, f) in enumerate(zip(blocks, factors)):
        base = 2 * k
        W[base, base + 1] = b[0, 1] * f
        W[base + 1, base] = b[1, 0] * f
    np.fill_diagonal(W, 0.0)
    np.clip(W, 0.0, 1.0, out=W)

    provenance = {
        "normalize": normalize,
        "cross_weight": float(cross_weight),
        "block_means": block_means,
        "rescale_factors": [float(f) for f in factors],
    }
    return W, provenance


def build_slot_assignment(n_pairs=3, shuffled=False, seed=0):
    """Seat the 2*n_pairs source agents.

    Returns a list of ``(run_idx, agent_idx)`` in seat order — seat s gets
    that source agent; seats (2k, 2k+1) are seat-pairs (their mutual bond is
    the k-th block of the merged W).

    shuffled=False: identity — each source pair sits together.
    shuffled=True: seeded scramble in which NO seat-pair contains two agents
    from the same source run (each agent is seated next to a stranger while
    keeping the same seats/bond structure — the shuffled-transplant control).
    """
    import random

    if not shuffled:
        return [(k, a) for k in range(n_pairs) for a in (0, 1)]
    if n_pairs < 2:
        raise ValueError("shuffled assignment needs at least 2 source pairs")

    rng = random.Random(seed)
    run_order = list(range(n_pairs))
    rng.shuffle(run_order)
    flip = [rng.randint(0, 1) for _ in range(n_pairs)]
    # Rotation construction: seat-pair k = (run_order[k]'s first agent,
    # run_order[(k+1) % n]'s second agent). Adjacent runs differ, so no
    # seat-pair can hold two agents from the same run.
    assignment = []
    for k in range(n_pairs):
        run_a = run_order[k]
        run_b = run_order[(k + 1) % n_pairs]
        assignment.append((run_a, flip[k]))
        assignment.append((run_b, 1 - flip[(k + 1) % n_pairs]))

    for k in range(n_pairs):
        a, b = assignment[2 * k], assignment[2 * k + 1]
        if a[0] == b[0]:
            raise AssertionError(
                f"shuffle invariant violated: seat-pair {k} holds two agents "
                f"from source run {a[0]}"
            )
    if sorted(assignment) != sorted(
        (k, a) for k in range(n_pairs) for a in (0, 1)
    ):
        raise AssertionError("shuffle must seat every source agent exactly once")
    return assignment


def build_merged_manifest(pair_states, assignment, condition="transplant",
                          pair_meta=None):
    """Build the ``--agent-state-init`` manifest for the merged run.

    pair_states: per source run, ``{0: state, 1: state}`` — the parsed
    ``agent_state/agent_{i}.json`` dicts of that run's two agents.
    assignment: seat list from build_slot_assignment().

    Every free-text field (skill names/descriptions, episode texts,
    curriculum strings) is remapped so the agent's OLD self index points at
    its new seat and its OLD partner index points at its new seatmate. In the
    shuffled condition the seatmate is a stranger — the transplanted memories
    then claim a shared history with an agent who was never there, which is
    exactly the memory<->bond-mismatch control.
    """
    agents_out = {}
    for seat, (run_idx, agent_idx) in enumerate(assignment):
        state = pair_states[run_idx][agent_idx]
        seatmate = seat + 1 if seat % 2 == 0 else seat - 1
        mapping = {agent_idx: seat, 1 - agent_idx: seatmate}

        skills_out = {}
        for name, payload in (state.get("skills", {}) or {}).items():
            skills_out[remap_agent_names(name, mapping)] = {
                "code": remap_agent_names(payload.get("code", ""), mapping),
                "description": remap_agent_names(
                    payload.get("description", ""), mapping
                ),
            }

        episodes_out = []
        for ep in state.get("episodes", []) or []:
            episodes_out.append({
                **ep,
                "text": remap_agent_names(ep.get("text", ""), mapping),
            })

        cur = state.get("curriculum", {}) or {}
        curriculum_out = {
            "current_context": remap_agent_names(
                cur.get("current_context", "") or "", mapping
            ),
            "completed_tasks": [
                remap_agent_names(t, mapping)
                for t in (cur.get("completed_tasks", []) or [])
            ],
            "failed_tasks": [
                remap_agent_names(t, mapping)
                for t in (cur.get("failed_tasks", []) or [])
            ],
        }

        meta = (pair_meta or {}).get(run_idx, {}) if pair_meta else {}
        agents_out[str(seat)] = {
            "agent_name": f"agent_{seat}",
            "source": f"pair_run_{run_idx}/agent_{agent_idx}",
            "source_run": meta.get("run_dir"),
            # Did the SOURCE pair actually co-fire in Phase A? Seat pairs
            # built from a non-co-firing source are the built-in control:
            # same machinery, same bond magnitude, no shared achievement.
            "source_cofired": meta.get("cofired"),
            "source_cofiring_milestones": meta.get("cofiring_milestones"),
            "skills": skills_out,
            "episodes": episodes_out,
            "curriculum": curriculum_out,
        }

    # Seat-pair level summary: which of the three dyads are genuine. In the
    # shuffled condition a seat pair spans two different source runs, so
    # "genuine" is meaningless there and cofired is recorded as None.
    seat_pairs = []
    for k in range(len(assignment) // 2):
        a, b = assignment[2 * k], assignment[2 * k + 1]
        same = a[0] == b[0]
        m = (pair_meta or {}).get(a[0], {}) if pair_meta else {}
        seat_pairs.append({
            "seats": [2 * k, 2 * k + 1],
            "same_source_run": same,
            "cofired": m.get("cofired") if same else None,
            "source_run": m.get("run_dir") if same else None,
        })

    return {
        "condition": condition,
        "num_agents": len(assignment),
        "assignment": [list(a) for a in assignment],
        "seat_pairs": seat_pairs,
        "agents": agents_out,
    }


def read_behavioral_cofiring(run_dir):
    """Sum the behavioural co-firing counters across a run's episodes.

    Read from ``episodes/*/summary.json -> cooperation_metrics``, NOT from
    ``final_metrics.json``: the coop_eval nesting bug zeroes the
    ``coop_metrics.pair_interaction_total`` tensors there (they come back as
    all-zero matrices even when the episodes recorded real events).

    Returns {joint_dig, co_action, proximity, episodes} with None values if
    nothing could be read.
    """
    run_dir = Path(run_dir)
    totals = {"joint_dig": 0, "co_action": 0, "proximity": 0, "episodes": 0}
    found = False
    for summary in sorted(run_dir.glob("episodes/*/summary.json")):
        try:
            with open(summary) as f:
                cm = json.load(f).get("cooperation_metrics", {}) or {}
        except Exception:
            continue
        found = True
        totals["joint_dig"] += cm.get("joint_dig_events") or 0
        totals["co_action"] += cm.get("co_action_events") or 0
        totals["proximity"] += cm.get("proximity_events") or 0
        totals["episodes"] += 1
    if not found:
        return {k: None for k in totals}
    return totals


# Milestones that CANNOT be earned alone. Solo digging on an anvil is
# net-zero by construction (SOLO_DIG_RATE 1 - DECAY_RATE 1), so an anvil
# break is proof two agents dug the same anvil inside ACTIVE_WINDOW. The
# gear equips follow from a break, so they are corroborating evidence.
COFIRING_MILESTONES = frozenset({
    "m8_anvil_A1", "m9_anvil_B1",
    "m14_sword_equipped", "m15_chestplate_equipped",
})


def read_cofiring_milestones(run_dir):
    """Return the sorted co-firing milestones a pair run actually earned.

    This is the ground-truth co-firing signal, and it is NOT interchangeable
    with the bond: across the 2026-08-06 Phase A batch the two runs that
    broke an anvil ranked 5th and 8th of 8 by bond, and the top three by
    bond earned none. Nor with joint_dig_events, which counts same-step digs
    while the anvil credits any two digs inside ACTIVE_WINDOW (~30 ticks).
    """
    run_dir = Path(run_dir)
    try:
        with open(run_dir / "final_metrics.json") as f:
            per_agent = json.load(f).get("milestones_per_agent") or {}
    except Exception:
        return []
    earned = set()
    for v in per_agent.values():
        if isinstance(v, list):
            earned.update(m for m in v if m in COFIRING_MILESTONES)
    return sorted(earned)


def pair_cofired(run_dir):
    """True if this pair earned a milestone that is impossible to earn alone."""
    return bool(read_cofiring_milestones(run_dir))


def rank_pair_runs(run_dirs):
    """Rank candidate pair runs by final within-pair bond strength.

    Returns a list of dicts (strongest bond first):
      {run_dir, bond, w01, w10, joint_dig, co_action, proximity, error}

    NOTE on the ranking key: ``bond`` is the transplanted quantity, so it is
    the sort key — but it is a CONFOUNDED proxy for co-firing. Engagement
    g_i includes an always-on "did i communicate" term, and agents message
    on virtually every step, so W tracks proximity-plus-chatter. Observed in
    the 2026-08-05 smoke: seed_42 ranked first on bond (0.2646) with ZERO
    joint digs, while seed_123 (0.2484) had 4. The behavioural columns are
    reported alongside so that mismatch is visible before picking the top 3.

    Runs whose hebbian_graph_final.json is missing/unreadable sort last with
    an ``error`` note instead of being dropped silently.
    """
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        row = {
            "run_dir": str(run_dir),
            "bond": None,
            "w01": None,
            "w10": None,
            "error": None,
        }
        try:
            with open(run_dir / "hebbian_graph_final.json") as f:
                W = json.load(f)["W"]
            row["w01"] = float(W[0][1])
            row["w10"] = float(W[1][0])
            row["bond"] = (row["w01"] + row["w10"]) / 2.0
        except Exception as exc:
            row["error"] = f"hebbian_graph_final.json: {exc}"
        row.update(read_behavioral_cofiring(run_dir))
        row["cofiring_milestones"] = read_cofiring_milestones(run_dir)
        row["cofired"] = bool(row["cofiring_milestones"])
        rows.append(row)

    rows.sort(key=lambda r: (r["bond"] is None, -(r["bond"] or 0.0)))
    return rows
