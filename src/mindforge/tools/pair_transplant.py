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


def build_merged_manifest(pair_states, assignment, condition="transplant"):
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

        agents_out[str(seat)] = {
            "agent_name": f"agent_{seat}",
            "source": f"pair_run_{run_idx}/agent_{agent_idx}",
            "skills": skills_out,
            "episodes": episodes_out,
            "curriculum": curriculum_out,
        }

    return {
        "condition": condition,
        "num_agents": len(assignment),
        "assignment": [list(a) for a in assignment],
        "agents": agents_out,
    }


def rank_pair_runs(run_dirs):
    """Rank candidate pair runs by final within-pair bond strength.

    Returns a list of dicts (strongest bond first):
      {run_dir, bond, w01, w10, anvil_coop_attempts, error}
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
            "anvil_coop_attempts": None,
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
        try:
            with open(run_dir / "final_metrics.json") as f:
                metrics = json.load(f)
            row["anvil_coop_attempts"] = metrics.get("anvil_coop_attempts")
        except Exception:
            pass  # metrics are informational; the bond is the ranking key
        rows.append(row)

    rows.sort(key=lambda r: (r["bond"] is None, -(r["bond"] or 0.0)))
    return rows
