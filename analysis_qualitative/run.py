#!/usr/bin/env python3
"""Qualitative-analysis pipeline over medium-run logs.

Subcommands (see docs in each stage function):
    parse     llm_logs + episode files -> per-run llm_calls / timeline / alignment report
    metrics   four dimension tables + failure flags + bonds-vs-behavior
    sample    stratified batch files for Claude-in-session annotation
    validate  annotation schema/coverage check + agreement
    cases     archetype shortlists + transcript renderer
    collab    collaboration success/failure episodes + Hebbian-vs-baseline contrast
    report    qual_report.md + CSV tables (+ staged tex; the paper is never touched)

Usage:
    python analysis_qualitative/run.py parse --only exp08_llm_9b_social_prompt/seed_42
    python analysis_qualitative/run.py parse
    python analysis_qualitative/run.py metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qual_lib import (registry, episode_io, log_parser, turns, provenance,  # noqa: E402
                      stepclock)

DEFAULT_RUNS_ROOT = registry.REPO / "runs_from_daic" / "medium_runs"
DEFAULT_OUT = registry.REPO / "analysis_qualitative" / "out"


# ── parse ────────────────────────────────────────────────────────────────


def _cross_checks(run, units, per_agent_rows, align):
    """Run-level consistency checks against final_metrics (when present)."""
    checks = {}
    n_msgs = 0
    routing = Counter()
    for ep_dir in episode_io.valid_episode_dirs(run):
        for m in episode_io.read_messages(ep_dir):
            n_msgs += 1
            routing[m.get("routing") or "?"] += 1
    fm = episode_io.read_final_metrics(run)
    if fm:
        cm = fm.get("comm_metrics") or {}
        checks["messages_vs_comm_metrics"] = {
            "messages_jsonl": n_msgs,
            "comm_metrics_total": cm.get("total_messages"),
            "ok": cm.get("total_messages") in (None, n_msgs),
        }
        rb = cm.get("routing_breakdown") or {}
        checks["routing_vs_breakdown"] = {
            "messages_jsonl": dict(routing),
            "comm_metrics": rb,
        }
    # hebbian W consistency: per-episode snapshot vs episode summary
    snaps = episode_io.read_hebbian_snapshots(run)
    if snaps:
        agree = []
        for i, ep_dir in enumerate(episode_io.valid_episode_dirs(run)):
            if i >= len(snaps):
                break
            coop = episode_io.read_cooperation_metrics(ep_dir)
            w_sum = coop.get("hebbian_W")
            w_snap = snaps[i].get("W")
            agree.append(w_sum == w_snap)
        checks["hebbian_W_snapshot_vs_summary"] = {
            "episodes_compared": len(agree), "all_equal": all(agree) if agree else None,
        }
    # parse quality per module
    per_mod = Counter()
    bad = Counter()
    for u in units:
        per_mod[u["module"]] += 1
        if not u["parse_ok"]:
            bad[u["module"]] += 1
    checks["parse_fail_by_module"] = {
        m: {"n": per_mod[m], "fail": bad.get(m, 0)} for m in sorted(per_mod)
    }
    return checks


def stage_parse(args) -> int:
    runs = registry.iter_runs(args.runs_root, args.only)
    if not runs:
        print(f"no runs found under {args.runs_root} (only={args.only})")
        return 1
    manifest = provenance.load_manifest(args.out)
    n_done = n_skip = 0
    for run in runs:
        if not run.episode_dirs or not run.llm_logs:
            print(f"[skip] {run.run_id}: missing episodes/ or llm_logs/")
            continue
        if not args.force and not provenance.needs_update(
            manifest, run, registry.PARSER_VERSION
        ):
            n_skip += 1
            continue
        try:
            _parse_one(run, args, manifest)
            n_done += 1
        except Exception as e:  # one bad run must not kill the batch
            print(f"[fail] {run.run_id}: {type(e).__name__}: {e}")
    print(f"parse done: {n_done} parsed, {n_skip} up-to-date")
    return 0


def _parse_one(run, args, manifest):
        t0 = time.time()
        units = log_parser.parse_run_llm_logs(run)
        per_agent_units = turns.action_units_per_agent(units)
        per_agent_rows, _row_map = turns.build_step_index(run)
        align = turns.align_actions(per_agent_units, per_agent_rows)
        attr_stats = turns.attribute_anonymous(units, per_agent_units)
        turns.social_units_turns(units, per_agent_units)
        # Clock fallback: RL runs (and any appended-log relaunches) can't be
        # text-anchored -> stamp (ep, t) from log.txt step markers instead.
        clock_info = None
        if any(r.get("status") == "unaligned" for r in align.values()):
            clock_info = stepclock.stamp_best_clock(
                run, units, per_agent_units, per_agent_rows)
            turns.promote_clock_alignment(per_agent_units, per_agent_rows,
                                          align)
        mapping = turns.map_units_to_rows(per_agent_units, per_agent_rows,
                                          align)
        attr_stats["stale_units"] = turns.mark_stale(
            units, per_agent_units, mapping, align)
        intervals = episode_io.module_intervals(run)
        cadence = turns.cadence_report(units, intervals)
        timeline = turns.build_timeline(run, units, per_agent_units,
                                        per_agent_rows, align, mapping)
        out_dir = args.out / "parsed" / run.exp / f"seed_{run.seed}"
        provenance.write_jsonl(out_dir / "llm_calls.jsonl.gz", units)
        provenance.write_jsonl(out_dir / "timeline.jsonl.gz", timeline)
        checks = _cross_checks(run, units, per_agent_rows, align)
        # Quarantine step-keyed analyses when the run dir shows cross-job
        # contamination: text alignment failed AND the message cross-check is
        # off by >10% (two jobs appended into the same artifacts; observed on
        # exp11_llm_9b_allied_none/seed_456).
        mc = checks.get("messages_vs_comm_metrics") or {}
        contaminated = False
        if mc.get("comm_metrics_total") and not mc.get("ok"):
            a_, b_ = mc["messages_jsonl"], mc["comm_metrics_total"]
            contaminated = abs(a_ - b_) / max(a_, b_) > 0.10
        text_ok = any(r.get("status") in ("exact", "repaired")
                      for r in align.values())
        quarantine = contaminated and not text_ok
        report = {
            "run": run.run_id, "degraded": run.degraded, "is_rl": run.is_rl,
            "quarantine_step_keyed": quarantine,
            "align": align, "attribution": attr_stats, "cadence": cadence,
            "clock": clock_info, "intervals": intervals,
            "checks": checks,
            "n_units": len(units), "n_timeline_rows": len(timeline),
            "elapsed_s": round(time.time() - t0, 1),
        }
        (out_dir / "alignment_report.json").write_text(
            json.dumps(report, indent=1, default=str), encoding="utf-8"
        )
        manifest[run.run_id] = {
            "parser_version": registry.PARSER_VERSION,
            "fingerprint": provenance.fingerprint_run(run),
            "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        provenance.save_manifest(args.out, manifest)
        statuses = {a: r["status"] for a, r in align.items()}
        rates = {a: (None if r["match_rate"] is None else round(r["match_rate"], 3))
                 for a, r in align.items()}
        print(f"[ok] {run.run_id}: {len(units)} units, "
              f"{len(timeline)} rows, align={statuses} match={rates} "
              f"({report['elapsed_s']}s)")


# ── stubs wired in later stages ──────────────────────────────────────────


def stage_metrics(args) -> int:
    from qual_lib import metrics as metrics_mod
    return metrics_mod.run_all(args)


def stage_sample(args) -> int:
    from qual_lib import sampling
    return sampling.run(args)


def stage_validate(args) -> int:
    from qual_lib import annotate
    return annotate.validate(args)


def stage_cases(args) -> int:
    from qual_lib import cases
    return cases.run(args)


def stage_collab(args) -> int:
    from qual_lib import collab
    return collab.run(args)


def stage_report(args) -> int:
    from qual_lib import report as report_mod
    return report_mod.run(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    common.add_argument("--out", type=Path, default=DEFAULT_OUT)
    common.add_argument("--only", type=str, default=None,
                        help="restrict to one run: exp_dir/seed_N (or exp_dir)")
    common.add_argument("--force", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("parse", "metrics", "sample", "validate", "cases", "collab",
                 "report"):
        sub.add_parser(name, parents=[common])
    args = p.parse_args(argv)
    stage = {
        "parse": stage_parse, "metrics": stage_metrics, "sample": stage_sample,
        "validate": stage_validate, "cases": stage_cases,
        "collab": stage_collab, "report": stage_report,
    }[args.cmd]
    return stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
