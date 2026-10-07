"""Protocol load of the hmas2 (HMAS-2 hard-orchestrator) arm, per run.

Every step the hmas2 orchestrator runs HMAS-2's plan -> per-agent check ->
revise protocol and writes one row to orchestrator/hmas2.jsonl. These are the
quantities to read off a smoke before queueing a sweep, and the hub-capacity
signals to compare across team sizes:

  rounds        check/revise rounds per step (1 = everybody agreed at once)
  objections    share of checks answered with an objection
  syntax        syntactic re-prompts per step; fallback = steps whose plan
                still failed the check after the last re-prompt
  empty         steps with no valid plan at all
  latency       seconds per step spent in the protocol (mean, p95) — serial,
                on the critical path of every step
  tok/step      orchestrator tokens per step (plan + revise + syntax + check)
  churn         reassignments per step
  msgs          coordinator messages per step, words per message, share
                delivered (budget runs block messages to exhausted agents)
  reports       agent reports read per step / dropped by the per-agent cap
  compliance    share of agent messages addressed to "orchestrator"

Usage:
    python analysis/make_hmas2_load.py --group agent_scaling_orch_smoke
    python analysis/make_hmas2_load.py --runs-root runs_from_daic/x --csv hmas2.csv

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import paths


def _rows(path: Path) -> list:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return statistics.fmean(xs) if xs else 0.0


def _p95(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def _num_agents(run_dir: Path) -> int | None:
    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        return int(cfg.get("num_agents"))
    except (OSError, ValueError, TypeError):
        return None


def analyze_run(run_dir: Path) -> dict | None:
    steps = _rows(run_dir / "orchestrator" / "hmas2.jsonl")
    if not steps:
        return None
    checks = sum(len(s.get("checked") or []) for s in steps)
    objections = sum(len(s.get("objections") or []) for s in steps)
    sent, delivered, words, blocked = 0, 0, [], 0
    for s in steps:
        for m in (s.get("messages") or {}).values():
            sent += 1
            delivered += bool(m.get("delivered"))
            words.append(int(m.get("words") or 0))
            blocked += m.get("budget_status") == "blocked"
    msgs = [m for p in sorted(run_dir.glob("episodes/ep_*/messages.jsonl"))
            for m in _rows(p)]
    hub_msgs = [m for m in msgs if m.get("routing") == "hub"]
    compliant = sum(m.get("model_target_canonical") == "orchestrator"
                    for m in hub_msgs)
    n = len(steps)
    return {
        "run": str(run_dir),
        "n_agents": _num_agents(run_dir),
        "steps": n,
        "rounds_mean": round(_mean(s.get("rounds", 0) for s in steps), 3),
        "objection_rate": round(objections / checks, 3) if checks else 0.0,
        "syntax_per_step": round(
            _mean(s.get("syntax_reprompts", 0) for s in steps), 3),
        "syntax_fallback_rate": round(
            _mean(not s.get("syntax_ok", True) for s in steps), 3),
        "empty_plan_rate": round(
            _mean(bool(s.get("plan_empty")) for s in steps), 3),
        "latency_mean": round(_mean(s.get("latency_s", 0.0) for s in steps), 2),
        "latency_p95": round(_p95([s.get("latency_s", 0.0) for s in steps]), 2),
        "tokens_per_step": round(_mean(sum((s.get("tokens") or {}).values())
                                       for s in steps)),
        "churn_per_step": round(
            _mean(len(s.get("reassigned") or []) for s in steps), 3),
        "msgs_per_step": round(sent / n, 3),
        "words_per_msg": round(_mean(words), 1),
        "delivered_rate": round(delivered / sent, 3) if sent else 0.0,
        "msgs_blocked": blocked,
        "reports_mean": round(_mean(s.get("reports_in", 0) for s in steps), 2),
        "reports_dropped": sum(int(s.get("reports_dropped") or 0)
                               for s in steps),
        "agent_msgs": len(hub_msgs),
        "compliance": (round(compliant / len(hub_msgs), 3)
                       if hub_msgs else 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", default="agent_scaling_orch",
                    help="run group name, resolved via paths.group() "
                         "[default: agent_scaling_orch]")
    ap.add_argument("--runs-root", type=Path, default=None,
                    help="explicit run root (overrides --group)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="write per-run rows to this CSV")
    args = ap.parse_args()

    root = args.runs_root or paths.group(args.group)
    runs = sorted({p.parent.parent
                   for p in root.glob("**/orchestrator/hmas2.jsonl")})
    rows = [r for r in (analyze_run(d) for d in runs) if r]
    if not rows:
        raise SystemExit(f"no hmas2 runs (orchestrator/hmas2.jsonl) under "
                         f"{root}")

    cols = ("n_agents", "steps", "rounds_mean", "objection_rate",
            "syntax_per_step", "syntax_fallback_rate", "empty_plan_rate",
            "latency_mean", "latency_p95", "tokens_per_step",
            "churn_per_step", "msgs_per_step", "delivered_rate",
            "reports_dropped", "compliance")
    print(f"{'run':<52} " + " ".join(f"{c[:9]:>9}" for c in cols))
    for r in rows:
        run = Path(r["run"])
        name = str(run.relative_to(root)) if run.is_relative_to(root) \
            else str(run)
        print(f"{name[-52:]:<52} " + " ".join(f"{r[c]!s:>9}" for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
