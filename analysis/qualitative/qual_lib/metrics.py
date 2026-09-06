"""Metrics orchestrator: loads parsed runs, computes the four dimensions +
bonds-vs-behavior, writes run-level CSVs, flags, and metrics_summary.json."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from qual_lib import (registry, episode_io, provenance, comm_content,
                      detectors, social_metrics, beliefs_metrics,
                      bonds_behavior)


def load_ctx(run, out_root: Path):
    """Everything a dimension module needs for one run, or None if unparsed."""
    pdir = out_root / "parsed" / run.exp / f"seed_{run.seed}"
    rep_path = pdir / "alignment_report.json"
    if not rep_path.exists():
        return None
    report = json.loads(rep_path.read_text(encoding="utf-8"))
    timeline = provenance.read_jsonl(pdir / "timeline.jsonl.gz")
    units = provenance.read_jsonl(pdir / "llm_calls.jsonl.gz")

    messages, events, coop_by_ep = {}, {}, {}
    for ep_i, ep_dir in enumerate(episode_io.valid_episode_dirs(run), start=1):
        messages[ep_i] = episode_io.read_messages(ep_dir)
        events[ep_i] = episode_io.read_events(ep_dir)
        coop_by_ep[ep_i] = episode_io.read_cooperation_metrics(ep_dir)
    fm = episode_io.read_final_metrics(run)
    intervals = episode_io.module_intervals(run)

    # Give text-aligned social/anonymous units a clock from their turn index
    # so downstream modules can rely on u["clock"] uniformly.
    rows_by_agent = defaultdict(list)
    for r in timeline:
        rows_by_agent[r["agent"]].append(r)
    for a in rows_by_agent:
        rows_by_agent[a].sort(key=lambda r: (r["ep"], r["t"]))
    align = report.get("align", {})
    for u in units:
        if u.get("clock") or u.get("stale"):
            continue
        a = u.get("agent_attr")
        if a is None and u.get("agent"):
            a = social_metrics._aidx(u["agent"])
        t = u.get("turn_of_agent")
        rep_a = align.get(str(a)) or {}
        if a is None or t is None or rep_a.get("status") not in (
                "exact", "repaired"):
            continue
        j = t + (rep_a.get("offset") or 0)
        rows = rows_by_agent.get(a, [])
        if 0 <= j < len(rows):
            u["clock"] = [rows[j]["ep"], rows[j]["t"]]

    return {
        "run": run, "report": report, "timeline": timeline, "units": units,
        "messages": messages, "events": events, "coop_by_ep": coop_by_ep,
        "final_metrics": fm, "intervals": intervals,
        "hebbian_snapshots": episode_io.read_hebbian_snapshots(run),
        "n_agents": intervals.get("num_agents", 3),
        "n_team_steps": sum(int(x) for x in (fm or {}).get(
            "episode_lengths", [])) or len(timeline) // max(
                1, intervals.get("num_agents", 3)),
        "quarantined": report.get("quarantine_step_keyed", False),
    }


def _write_csv(path: Path, rows: list, field_order=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = field_order or sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cluster_condition_texts(texts: list, cap: int = 4000):
    """Top message templates for a condition via comm_eval clustering."""
    from mindforge.agent_modules.comm_eval import _cluster_messages
    rng = random.Random(1234)
    if len(texts) > cap:
        texts = rng.sample(texts, cap)
    if not texts:
        return []
    labels = _cluster_messages(texts)
    by_c = defaultdict(list)
    for t, c in zip(texts, labels):
        by_c[c].append(t)
    out = []
    for c, ts in sorted(by_c.items(), key=lambda kv: -len(kv[1]))[:20]:
        rep = Counter(t.strip().lower() for t in ts).most_common(1)[0][0]
        out.append({"cluster": c, "size": len(ts),
                    "share": round(len(ts) / len(texts), 4),
                    "representative": rep[:160]})
    return out


def run_all(args) -> int:
    runs = registry.iter_runs(args.runs_root, args.only)
    out_tables = args.out / "tables"
    flags_all = []
    per_run = {}
    cond_texts = defaultdict(list)
    cond_pairs = defaultdict(list)

    for run in runs:
        ctx = load_ctx(run, args.out)
        if ctx is None:
            print(f"[skip] {run.run_id}: not parsed")
            continue
        rec = {"run": run.run_id, "exp": run.exp, "seed": run.seed,
               "label": run.label, "is_rl": run.is_rl,
               "quarantined": ctx["quarantined"]}
        if ctx["quarantined"]:
            per_run[run.run_id] = rec
            print(f"[quarantined] {run.run_id}: final-metrics-only")
            continue

        cc = comm_content.per_run(ctx)
        cond_texts[run.label].extend(cc.pop("_valid_texts", []))
        rec["comm"] = cc

        flags = detectors.run_detectors(ctx["timeline"], run.is_rl)
        for f in flags:
            f["label"] = run.label
        flags_all.extend(flags)
        # hallucinated_object split by source: thoughts routinely mention
        # other chambers when planning, so only the belief/message sources
        # are honest hallucination signals; keep all three visible.
        cnt = Counter(
            (f["detector"] if f["detector"] != "hallucinated_object"
             else f"hallucinated_object_{f['detail'].get('source', '?')}")
            for f in flags)
        rec["failures"] = dict(cnt)
        rec["failures_per_1k"] = {
            k: round(1000 * v / max(ctx["n_team_steps"], 1), 2)
            for k, v in cnt.items()}

        soc = social_metrics.analyze_run(ctx)
        if soc:
            rec["social"] = soc
        rec["beliefs"] = beliefs_metrics.analyze_run(ctx)

        bb = bonds_behavior.analyze_run(ctx)
        if bb:
            cond_pairs[run.label].extend(bb.pop("_pair_records", []))
            rec["bonds"] = bb

        per_run[run.run_id] = rec
        print(f"[ok] {run.run_id}: msgs={cc['n_valid']}, "
              f"flags={sum(cnt.values())}, "
              f"social={'y' if soc else '-'}")

    # ── write outputs ──
    provenance.write_jsonl(args.out / "flags" / "flags.jsonl.gz", flags_all)

    comm_rows, soc_rows, bel_rows, fail_rows, bond_rows = [], [], [], [], []
    for rid, rec in per_run.items():
        base = {"run": rid, "label": rec["label"], "seed": rec["seed"],
                "quarantined": rec.get("quarantined", False)}
        if "comm" in rec:
            c = rec["comm"]
            comm_rows.append({**base, **{k: v for k, v in c.items()
                                         if not isinstance(v, dict)}})
        if "social" in rec:
            soc_rows.append({**base, **rec["social"]})
        if "beliefs" in rec:
            bel_rows.append({**base, **rec["beliefs"]})
        if "failures_per_1k" in rec:
            fail_rows.append({**base, **rec["failures_per_1k"]})
        if "bonds" in rec:
            bond_rows.append({**base, **{k: v for k, v in rec["bonds"].items()
                                         if not isinstance(v, (list, dict))}})
    _write_csv(out_tables / "comm_run_level.csv", comm_rows)
    _write_csv(out_tables / "social_interpretability.csv", soc_rows)
    _write_csv(out_tables / "beliefs.csv", bel_rows)
    _write_csv(out_tables / "failure_taxonomy_run_level.csv", fail_rows)
    _write_csv(out_tables / "bonds_behavior_runs.csv", bond_rows)

    # category mix long table
    cat_rows = []
    for rid, rec in per_run.items():
        for ch, cats in (rec.get("comm", {}).get("categories_by_chamber")
                         or {}).items():
            n_ch = (rec["comm"].get("valid_by_chamber") or {}).get(ch, 0)
            for cat, n in cats.items():
                cat_rows.append({
                    "run": rid, "label": rec["label"], "chamber": ch,
                    "category": cat, "n": n,
                    "frac": round(n / n_ch, 4) if n_ch else None})
    _write_csv(out_tables / "comm_categories.csv", cat_rows,
               ["run", "label", "chamber", "category", "n", "frac"])

    # condition-level: templates + pooled bond correlations
    tmpl_rows = []
    for label, texts in sorted(cond_texts.items()):
        for t in _cluster_condition_texts(texts):
            tmpl_rows.append({"label": label, "n_texts": len(texts), **t})
    _write_csv(out_tables / "comm_templates.csv", tmpl_rows,
               ["label", "n_texts", "cluster", "size", "share",
                "representative"])

    cond_bond_rows = []
    for label, recs in sorted(cond_pairs.items()):
        cond_bond_rows.append({"label": label,
                               **bonds_behavior.pooled_condition(recs)})
    _write_csv(out_tables / "bonds_behavior_condition.csv", cond_bond_rows)
    provenance.write_jsonl(args.out / "tables" / "bond_pair_records.jsonl.gz",
                           [{"label": lab, **r}
                            for lab, recs in cond_pairs.items()
                            for r in recs])

    (args.out / "metrics_summary.json").write_text(
        json.dumps(per_run, indent=1, default=str), encoding="utf-8")
    print(f"metrics done: {len(per_run)} runs, {len(flags_all)} flags, "
          f"tables -> {out_tables}")
    return 0
