"""Stratified sampling into batch files for Claude-in-session annotation.

Selection is deterministic and append-only: within each stratum cell the
candidates are ordered by item hash and the first K taken, so re-running
after new seeds land only adds/reshuffles marginally and never invalidates
existing annotation records (they are joined by item id).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qual_lib import registry, provenance, metrics as metrics_mod
from qual_lib.annotate import RUBRICS

PER_CELL = {"messages": 8, "beliefs": 6, "social": 10}
FAIL_FLAGGED, FAIL_CONTROL = 25, 10
BATCH_SIZE = 25


def _mk_item(dim, label, run, ep, t, agent, payload):
    body = {"dim": dim, "label": label, "exp": run.exp, "seed": run.seed,
            "ep": ep, "t": t, "agent": agent, "payload": payload}
    return {"id": provenance.item_hash(body), **body,
            "rubric": RUBRICS[dim]["name"]}


def _context_window(rows_by_ept, ep, t, agent, before=3, after=2):
    """Compact multi-agent context lines around (ep, t)."""
    lines = []
    for dt in range(-before, after + 1):
        for a in sorted({k[2] for k in rows_by_ept if k[0] == ep}):
            r = rows_by_ept.get((ep, t + dt, a))
            if not r:
                continue
            msg = (r.get("msg") or {}).get("text")
            lines.append(
                f"t={t + dt} agent_{a} [{r.get('chamber')}] {r.get('action')}"
                + (f' says->"{msg}"' if msg else ""))
    return lines


def collect_candidates(runs, out_root: Path):
    cands = defaultdict(list)  # (dim, cell_key) -> [item]
    flags = provenance.read_jsonl(out_root / "flags" / "flags.jsonl.gz")
    flags_by_run = defaultdict(list)
    for f in flags:
        flags_by_run[f"{f['exp']}/seed_{f['seed']}"].append(f)

    for run in runs:
        ctx = metrics_mod.load_ctx(run, out_root)
        if ctx is None or ctx["quarantined"]:
            continue
        rows_by_ept = {(r["ep"], r["t"], r["agent"]): r
                       for r in ctx["timeline"]}
        label = run.label

        # dim: messages — valid messages by chamber
        for ep_i, msgs in ctx["messages"].items():
            for m in msgs:
                if not m.get("valid") or not m.get("text"):
                    continue
                a = social_idx = int(str(m.get("sender", "-1")).split("_")[-1]) \
                    if str(m.get("sender", "")).split("_")[-1].isdigit() else None
                item = _mk_item(
                    "messages", label, run, ep_i, int(m.get("t", -1)), a,
                    {"text": m["text"], "receiver": m.get("receiver"),
                     "routing": m.get("routing"), "chamber": m.get("chamber"),
                     "context": _context_window(rows_by_ept, ep_i,
                                                int(m.get("t", -1)), a)})
                cands[("messages", (label, m.get("chamber") or "?"))].append(item)

        # dim: beliefs — perception / partner / interaction rows
        for r in ctx["timeline"]:
            b = r.get("beliefs") or {}
            for btype in ("perception", "partner", "interaction"):
                val = b.get(btype)
                if not val:
                    continue
                ch = str(r.get("chamber") or "?")
                chbin = ("ch1" if ch == "ch1"
                         else "ch2-3" if ch.startswith(("ch2", "ch3"))
                         else "ch4-5")
                truth = _context_window(rows_by_ept, r["ep"], r["t"],
                                        r["agent"], before=0, after=0)
                item = _mk_item(
                    "beliefs", label, run, r["ep"], r["t"], r["agent"],
                    {"belief_type": btype,
                     "belief": val if isinstance(val, (str, list)) else str(val),
                     "chamber": ch, "ground_truth_now": truth})
                cands[("beliefs", (label, btype, chbin))].append(item)

        # dim: social — fresh deliberations
        n_eps = max(ctx["messages"].keys(), default=0)
        for r in ctx["timeline"]:
            s = r.get("social")
            if not s:
                continue
            terc = min(2, (r["ep"] - 1) * 3 // max(n_eps, 1)) if n_eps else 0
            nxt = []
            for dt in range(0, 8):
                rr = rows_by_ept.get((r["ep"], r["t"] + dt, r["agent"]))
                mtext = rr and (rr.get("msg") or {}).get("text")
                if mtext:
                    tgt = (rr.get("msg") or {}).get("model_target_canonical")
                    nxt.append(f"t+{dt} ->{tgt}: {mtext}")
                if len(nxt) >= 3:
                    break
            item = _mk_item(
                "social", label, run, r["ep"], r["t"], r["agent"],
                {"thought": {k: s.get(k) for k in
                             ("reasoning", "referenced_bonds", "ask_target",
                              "ask_message", "respond_to",
                              "bond_change_explanation", "confidence")},
                 "own_next_messages": nxt})
            cands[("social", (label, bool(s.get("ask_target")), terc))].append(item)

        # dim: failures — flagged windows + controls
        rflags = flags_by_run.get(run.run_id, [])
        for f in rflags:
            item = _mk_item(
                "failures", label, run, f["ep"], f["t0"], f.get("agent"),
                {"detector": f["detector"], "detail": f.get("detail"),
                 "window": _context_window(rows_by_ept, f["ep"], f["t0"],
                                           f.get("agent"), 4, 8),
                 "flagged": True})
            cands[("failures", (f["detector"], True))].append(item)
        # controls: random steps not covered by any flag of that detector
        flagged_ts = defaultdict(set)
        for f in rflags:
            for t in range(f["t0"], f["t1"] + 1):
                flagged_ts[f["detector"]].add((f["ep"], t))
        for det in {f["detector"] for f in rflags}:
            for r in ctx["timeline"][:: max(1, len(ctx["timeline"]) // 40)]:
                if (r["ep"], r["t"]) in flagged_ts[det]:
                    continue
                item = _mk_item(
                    "failures", label, run, r["ep"], r["t"], r["agent"],
                    {"detector": det, "detail": None,
                     "window": _context_window(rows_by_ept, r["ep"], r["t"],
                                               r["agent"], 4, 8),
                     "flagged": False})
                cands[("failures", (det, False))].append(item)
    return cands


def run(args) -> int:
    runs = registry.iter_runs(args.runs_root, args.only)
    cands = collect_candidates(runs, args.out)
    chosen = defaultdict(list)  # dim -> [items]
    DEDUP_KEYS = {"messages": ("text",), "beliefs": ("belief",),
                  "social": ("thought",), "failures": ("detector", "window")}
    for (dim, cell), items in sorted(cands.items(), key=lambda kv: str(kv[0])):
        items = sorted(items, key=lambda it: it["id"])
        # collapse exact-duplicate payloads before capping (per-dim key)
        seen_payload = set()
        uniq = []
        for it in items:
            key = provenance.item_hash(
                {k: it["payload"].get(k) for k in DEDUP_KEYS[dim]})
            if key in seen_payload:
                continue
            seen_payload.add(key)
            uniq.append(it)
        if dim == "failures":
            k = FAIL_FLAGGED if cell[1] else FAIL_CONTROL
        else:
            k = PER_CELL[dim]
        chosen[dim].extend(uniq[:k])

    total = 0
    for dim, items in chosen.items():
        items.sort(key=lambda it: it["id"])
        ddir = args.out / "samples" / dim
        # wipe previous batch files (selection is deterministic anyway)
        if ddir.exists():
            for old in ddir.glob("batch_*.jsonl"):
                old.unlink()
        for bi in range(0, len(items), BATCH_SIZE):
            batch = items[bi:bi + BATCH_SIZE]
            provenance.write_jsonl(
                ddir / f"batch_{bi // BATCH_SIZE:03d}.jsonl", batch)
        # 10% repass subset for intra-annotator agreement (blind second pass).
        # Salted re-hash — the per-cell min-k-by-hash selection concentrates
        # raw ids near 0, so bucketing on the id itself would catch ~100%.
        import hashlib
        repass = [it for it in items
                  if int(hashlib.sha1((it["id"] + "repass").encode())
                         .hexdigest()[:2], 16) < 26]
        provenance.write_jsonl(ddir / "repass.jsonl", repass)
        total += len(items)
        print(f"[sample] {dim}: {len(items)} items "
              f"({(len(items) + BATCH_SIZE - 1) // BATCH_SIZE} batches, "
              f"{len(repass)} repass)")
    # rubric instructions for the annotator (Claude reads this file)
    (args.out / "samples" / "RUBRICS.md").write_text(
        _rubrics_md(), encoding="utf-8")
    print(f"sample done: {total} items -> {args.out / 'samples'}")
    return 0


def _rubrics_md() -> str:
    lines = ["# Annotation rubrics",
             "",
             "Annotate each batch file `out/samples/<dim>/batch_NNN.jsonl` into "
             "`out/annotations/<dim>/batch_NNN.jsonl`: one JSON per line, "
             "schema `{\"id\": <item id>, ...rubric fields}`. Repass files are "
             "annotated the same way into `annotations/<dim>/repass.jsonl` in a "
             "SEPARATE session without looking at the first-pass labels.", ""]
    for dim, r in RUBRICS.items():
        lines.append(f"## {dim} — {r['name']}")
        for f, spec in r["fields"].items():
            lines.append(f"- `{f}`: {spec}")
        lines.append("")
    return "\n".join(lines)
