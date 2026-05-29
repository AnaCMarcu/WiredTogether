"""Per-teammate reward-propagation attribution + prompt formatting.

Today, ``hebbian_graph.diffuse_rewards()`` returns one diffused reward
per agent — a scalar that combines own + propagated-from-teammates
signal. To make this visible in the LLM's prompt we need to **decompose**
that scalar into per-teammate contributions, then format them as a
short prompt line so the agent sees, e.g.:

    Propagated rewards this step: +2.50 from agent_1 (m17_switch_pressed),
    +0.30 from agent_2

The decomposition is pure math against the Eq. 8 expansion:

    r'_i(t) = (1-γ)·r_i(t) + γ·Σ_{j≠i} w̄_ij · c_ij · r_j(t)
                              ─────────────────────────────
                                        contribution_ij

So ``contribution_ij = γ · w̄_ij · c_ij · r_j`` for each teammate j.
This matches what ``HebbianSocialGraph.diffuse_rewards()`` computes
internally at [src/hebbian/graph.py:496-543](../hebbian/graph.py).

Everything here is **pure-python + numpy** — no torch, no env, no LLM.
Testable in isolation. The trainer / legacy entry point are the
only callers.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def per_teammate_contributions(
    agent_id: int,
    raw_rewards: list[float],
    w_bar_row: "np.ndarray | list[float]",
    coactivity_row: "np.ndarray | list[float]",
    gamma: float,
) -> dict[int, float]:
    """Decompose a diffused reward into per-teammate additive contributions.

    Parameters
    ----------
    agent_id
        The agent whose decomposition we're computing.
    raw_rewards
        Length-N list of raw per-agent rewards at this step.
    w_bar_row
        Length-N vector from ``HebbianSocialGraph.get_normalized_weights(agent_id)``.
        Diagonal entry is masked to 0 by the graph; we don't re-check.
    coactivity_row
        Length-N vector — row ``agent_id`` of the co-activity matrix
        cached on the graph (``graph._last_coactivity[agent_id, :]``).
    gamma
        Reward diffusion strength (``HebbianConfig.reward_diffusion_gamma``).

    Returns
    -------
    dict[int, float]
        ``{teammate_id: delta}`` for every j ≠ agent_id. Includes
        zero-deltas — caller filters by threshold for display.

    Notes
    -----
    Sum of returned values equals ``gamma * Σ_{j≠i} w̄_ij · c_ij · r_j``,
    which matches the propagated half of Eq. 8. The own-half
    ``(1-γ)·r_i`` is the agent's individual reward signal and is *not*
    part of the "propagation" line in the prompt.
    """
    raw = np.asarray(raw_rewards, dtype=np.float64)
    w = np.asarray(w_bar_row, dtype=np.float64)
    c = np.asarray(coactivity_row, dtype=np.float64)
    n = raw.size
    if w.size != n or c.size != n:
        raise ValueError(
            f"shape mismatch: raw_rewards={n}, w_bar_row={w.size}, "
            f"coactivity_row={c.size}"
        )
    out: dict[int, float] = {}
    for j in range(n):
        if j == agent_id:
            continue
        out[j] = float(gamma * w[j] * c[j] * raw[j])
    return out


def format_propagation_prompt(
    contributions: dict[int, float],
    source_events: dict[int, str] | None = None,
    threshold: float = 1e-3,
    role_names: dict[int, str] | None = None,
) -> str:
    """Format per-teammate contributions into a prompt-line.

    Parameters
    ----------
    contributions
        Output of ``per_teammate_contributions``.
    source_events
        Optional ``{teammate_id: event_name}`` — when a teammate fired a
        milestone this step we annotate the contribution with the
        milestone id so the LLM can connect cause and effect.
    threshold
        Contributions with ``abs(delta) < threshold`` are omitted to
        keep the prompt short.
    role_names
        Optional ``{teammate_id: role_str}`` — when supplied, the
        teammate label includes the role, e.g.
        ``"+2.50 from agent_1 (gatherer)"``.

    Returns
    -------
    str
        Either ``""`` (no contributions above threshold — line is empty,
        prompt template collapses cleanly) or a single line:

        ``Propagated rewards this step: +2.50 from agent_1
        (m17_switch_pressed), +0.30 from agent_2``

    The line is intentionally short — the LLM's context budget is tight
    and the per-teammate level of detail is what the LLM needs to
    connect to its own decisions, not a histogram.
    """
    source_events = source_events or {}
    role_names = role_names or {}
    parts: list[str] = []
    # Stable ordering — teammate id ascending — so the prompt is
    # deterministic across steps with identical state.
    for j in sorted(contributions.keys()):
        delta = contributions[j]
        if abs(delta) < threshold:
            continue
        sign = "+" if delta >= 0 else ""
        label_bits = [f"agent_{j}"]
        if j in role_names:
            label_bits.append(f"({role_names[j]})")
        event = source_events.get(j)
        bits = [f"{sign}{delta:.2f} from {' '.join(label_bits)}"]
        if event and event != "reward":
            bits.append(f"({event})")
        parts.append(" ".join(bits))
    if not parts:
        return ""
    return "Propagated rewards this step: " + ", ".join(parts)


def attribute_source_events(
    events: Iterable[dict],
) -> dict[int, str]:
    """Walk this step's milestone events and build a ``{agent_id: milestone_id}``
    map for attribution annotations.

    Accepts two schemas — the GRPO stream and the legacy stream:

    * **GRPO** (``rollout_sampler._aggregate_env_events``):
      ``{"step", "agent_id": int, "milestone_id": str, ...}``

    * **Legacy** (``custom_environment_craftium.poll_milestone_events``):
      ``{"step", "milestone": str, "contributors": ["agent_N", ...], ...}``
      — the credited agent is the first entry of ``contributors``.

    When multiple milestones fire in the same step (rare but possible),
    the *last* one wins — these aren't semantically ordered so any pick
    is defensible; the alternative is to concatenate and bloat the
    prompt line.
    """
    out: dict[int, str] = {}
    for ev in events:
        mid = ev.get("milestone_id") or ev.get("milestone")
        if not isinstance(mid, str):
            continue

        # GRPO schema: explicit agent_id.
        aid = ev.get("agent_id")
        if isinstance(aid, int):
            out[aid] = mid
            continue

        # Legacy schema: first contributor as the credited agent.
        contribs = ev.get("contributors") or []
        if not contribs:
            continue
        first = contribs[0]
        # The Lua side emits names without an underscore ('agent0'); the
        # old parser additionally required the 'agent_' prefix, so 'agent0'
        # was filtered out at startswith() and the milestone never reached
        # the per-teammate attribution table that the LLM sees. Strip
        # either prefix shape and parse the trailing digits.
        if not isinstance(first, str):
            continue
        _s = first.removeprefix("agent_").removeprefix("agent")
        try:
            out[int(_s)] = mid
        except ValueError:
            continue
    return out


def build_interpretability_record(
    step: int,
    agent_id: int,
    chamber: str | None,
    bond_row: "list[float] | np.ndarray",
    parsed_action: dict | None,
    propagated_contribs: dict[int, float] | None = None,
    propagated_sources: dict[int, str] | None = None,
    thoughts_max_chars: int = 60,
) -> dict:
    """One ``interpretability.jsonl`` line — pure data, caller appends to disk.

    Captures everything the LLM saw + chose at one (env step, agent), so
    downstream notebooks can correlate bond strength with communication
    target choice, milestone attribution with action choice, etc.

    ``parsed_action`` is the dict the LLM emitted (with ``action``,
    ``communication_target``, ``thoughts`` keys). If ``None`` or
    malformed, the record still serializes — missing fields become ``None``.

    ``propagated_contribs`` and ``propagated_sources`` are typically the
    same dicts that fed the current step's prompt — i.e. the *previous*
    step's diffusion output. Passing them here ties the record's
    "propagation" view to what the agent actually saw at decision time.
    """
    propagated_contribs = propagated_contribs or {}
    propagated_sources = propagated_sources or {}

    parsed = parsed_action if isinstance(parsed_action, dict) else {}
    action = parsed.get("action")
    comm_target = parsed.get("communication_target")
    if isinstance(comm_target, bool):
        comm_target = None        # bools accidentally typed as ints — drop
    thoughts = parsed.get("thoughts")
    if isinstance(thoughts, str) and len(thoughts) > thoughts_max_chars:
        thoughts = thoughts[: thoughts_max_chars - 1].rstrip() + "…"

    # numpy → list for clean JSON.
    row = list(bond_row) if bond_row is not None else []
    row = [float(v) for v in row]

    return {
        "step": int(step),
        "agent_id": int(agent_id),
        "chamber": str(chamber) if chamber else None,
        "bond_row": row,
        "chosen_action": action if isinstance(action, str) else None,
        "communication_target": (
            int(comm_target) if isinstance(comm_target, int) else None
        ),
        "thoughts_excerpt": thoughts if isinstance(thoughts, str) else None,
        "propagated_delta_by_teammate": {
            str(k): float(v) for k, v in propagated_contribs.items()
        },
        "propagated_source_events": {
            str(k): str(v) for k, v in propagated_sources.items()
        },
    }


__all__ = [
    "per_teammate_contributions",
    "format_propagation_prompt",
    "attribute_source_events",
    "build_interpretability_record",
]
