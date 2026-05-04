"""Post-hoc cooperation metrics for the Five Chambers env.

Reads `runs/<run_id>/episodes/ep_*/{step_log.csv, event_log.jsonl,
summary.json}` and returns a dict suitable for inclusion in
`final_metrics.json["coop_metrics"]`.

The five-plane pair-interaction tensor (messages, joint_dig, proximity,
joint_kill, ch5_damage_overlap) is already computed online by
`CooperationMetric` and serialised into each `episodes/ep_*/summary.json`.
We just sum the per-episode planes here to get a run-level total, and
add post-hoc derived metrics that benefit from cross-episode aggregation.

Metrics computed here:

* **Pair interaction tensor** (run total): N×N×K, sum over episodes.
* **Milestone credit table**: per-agent per-milestone credit using each
  milestone's firing rule (damage share for combat, equal split for
  presence, identity for solo dig). No coalition sampling.
* **Per-chamber dwell** (run total): per-agent per-chamber tick count.
* **Per-chamber action histogram** (run total): per-agent per-chamber action counts.
* **Reward-decomposition fractions**: per-agent fraction of cumulative
  return attributable to each source (task / comm_base / comm_milestone /
  proximity / hebbian_diffuse). Reads `final_metrics.json` if present
  but also accepts the parallel array passed in directly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _zeros(n: int) -> list:
    return [[0 for _ in range(n)] for _ in range(n)]


def _add_matrices(a: list, b: list) -> list:
    if not a:
        return [row[:] for row in b]
    return [[a[i][j] + b[i][j] for j in range(len(a))] for i in range(len(a))]


# Milestone credit rules. Each entry returns a dict {agent_name: credit_share}
# given (contributors_list, damage_per_agent_for_target). Sum = 1.0.
def _credit_equal(contribs: list, *_a) -> dict:
    if not contribs:
        return {}
    share = 1.0 / len(contribs)
    return {c: share for c in contribs}


def _credit_damage_share(contribs: list, damage_for_target: dict) -> dict:
    """Split credit by damage dealt. Falls back to equal split when no damage
    info is available (which happens for episodes without infos.damage_events)."""
    if not damage_for_target or sum(damage_for_target.values()) == 0:
        return _credit_equal(contribs)
    total = sum(damage_for_target.values())
    out = {}
    for c in contribs:
        if c in damage_for_target:
            out[c] = damage_for_target[c] / total
    if not out:
        return _credit_equal(contribs)
    # Renormalise in case some contributors had zero damage.
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


# Map milestone id → credit rule.
_CREDIT_RULES: dict = {
    "m22_all_mobs_killed": "ch4_damage",
    "m23_all_alive_ch4":   "equal",
    "m25_first_boss_dmg":  "boss_damage",
    "m26_boss_half_hp":    "boss_damage",
    "m27_boss_defeated":   "boss_damage",
    "m28_all_alive_bonus": "equal",
    "m19_all_in_communal": "equal",
}


def compute_coop_metrics(run_root, num_agents: int) -> dict:
    """Aggregate per-episode summaries into a run-level cooperation report."""
    root = Path(run_root)
    ep_dirs = sorted((root / "episodes").glob("ep_*"))

    # ── Per-pair interaction tensor: sum across episodes ──
    pair_planes = ["messages", "joint_dig", "proximity",
                   "joint_kill", "ch5_damage_overlap"]
    pair_total = {p: _zeros(num_agents) for p in pair_planes}

    # Per-chamber dwell + action histogram (sum across episodes).
    dwell_total = defaultdict(lambda: defaultdict(int))
    action_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Damage-per-agent per target type, across the run, for credit splitting.
    ch4_damage_run = defaultdict(float)
    ch5_damage_run = defaultdict(float)

    # Milestone events to credit.
    all_milestones: list = []

    for ed in ep_dirs:
        sf = ed / "summary.json"
        if not sf.exists():
            sf = ed / "episode_summary.json"
        if not sf.exists():
            continue
        try:
            summary = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Pair tensor
        pi = summary.get("pair_interaction") or {}
        for plane in pair_planes:
            mat = pi.get(plane)
            if mat:
                pair_total[plane] = _add_matrices(pair_total[plane], mat)

        # Dwell + action hist
        for aid_str, by_chamber in (summary.get("dwell_steps") or {}).items():
            for ch, n in by_chamber.items():
                dwell_total[aid_str][ch] += int(n)
        for aid_str, by_chamber in (summary.get("action_hist") or {}).items():
            for ch, hh in by_chamber.items():
                for act, n in hh.items():
                    action_total[aid_str][ch][act] += int(n)

        # Damage per agent (for credit splitting)
        for k, v in (summary.get("ch4_damage_per_agent") or {}).items():
            ch4_damage_run[str(k)] += float(v)
        for k, v in (summary.get("ch5_damage_per_agent") or {}).items():
            ch5_damage_run[str(k)] += float(v)

        # Milestone events from event_log.jsonl
        for ev in _read_jsonl(ed / "event_log.jsonl"):
            if ev.get("type") in ("milestone", "comm_milestone"):
                all_milestones.append(ev)

    # ── Milestone credit table ──
    credit = defaultdict(lambda: defaultdict(float))
    for ev in all_milestones:
        mid = ev.get("milestone") or ev.get("id") or ""
        contribs = ev.get("contributors") or ev.get("agents") or []
        if "agent" in ev and not contribs:
            contribs = [ev["agent"]]
        if not contribs:
            continue

        rule = _CREDIT_RULES.get(mid, "equal")
        if rule == "equal":
            shares = _credit_equal(contribs)
        elif rule == "ch4_damage":
            shares = _credit_damage_share(contribs, dict(ch4_damage_run))
        elif rule == "boss_damage":
            shares = _credit_damage_share(contribs, dict(ch5_damage_run))
        else:
            shares = _credit_equal(contribs)

        reward = float(ev.get("reward", 0.0))
        for agent_name, frac in shares.items():
            credit[str(agent_name)][mid] += frac * reward

    # Per-agent total credit (used for fairness Gini below).
    per_agent_total_credit = {a: sum(d.values()) for a, d in credit.items()}

    # Credit Gini across agents.
    def _gini(vals: list) -> float:
        vs = sorted(vals)
        n = len(vs)
        if n == 0 or sum(vs) == 0:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(vs))
        return (2 * cum) / (n * sum(vs)) - (n + 1) / n

    credit_gini = _gini(list(per_agent_total_credit.values())) \
        if per_agent_total_credit else 0.0

    return {
        "pair_interaction_total": pair_total,
        "dwell_steps_total": {a: dict(by_ch) for a, by_ch in dwell_total.items()},
        "action_hist_total": {a: {ch: dict(hh) for ch, hh in by_ch.items()}
                              for a, by_ch in action_total.items()},
        "ch4_damage_per_agent": dict(ch4_damage_run),
        "ch5_damage_per_agent": dict(ch5_damage_run),
        "milestone_credit": {a: dict(d) for a, d in credit.items()},
        "milestone_credit_total": per_agent_total_credit,
        "credit_gini": credit_gini,
    }
