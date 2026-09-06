"""Case-study archetype shortlists + transcript renderer."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qual_lib import registry, provenance, metrics as metrics_mod, lexicons

MAX_WINDOW = 60  # steps per transcript


def _fmt_row(r):
    msg = r.get("msg") or {}
    parts = [f"| {r['t']} | a{r['agent']} | {r.get('chamber') or '?'} "
             f"| {r.get('action') or ''} |"]
    m = ""
    if msg.get("text"):
        tgt = msg.get("model_target_canonical") or msg.get("receiver") or "?"
        route = msg.get("routing") or ""
        rt = "" if route == "model" else f" [{route}]"
        m = f"->{tgt}{rt}: {str(msg['text'])[:90]}"
    notes = []
    if r.get("task_new"):
        notes.append(f"NEW TASK: {str(r.get('task'))[:70]}")
    c = r.get("critic")
    if c and c.get("success") is not None:
        notes.append(f"critic:{'OK' if c['success'] else 'FAIL'}")
    s = r.get("social")
    if s and s.get("ask_target"):
        notes.append(f"social asks {s['ask_target']}")
    for e in (r.get("events") or []):
        mid = e.get("id") or e.get("milestone")
        notes.append(f"** {mid} (+{e.get('reward')}) **")
    return parts[0] + f" {m} | {'; '.join(notes)} |"


def render_transcript(run, timeline, ep, t0, t1, slug, reason, out_dir: Path):
    t0, t1 = max(0, t0), max(t0, t1)
    if t1 - t0 > MAX_WINDOW:
        t1 = t0 + MAX_WINDOW
    rows = [r for r in timeline if r["ep"] == ep and t0 <= r["t"] <= t1]
    rows.sort(key=lambda r: (r["t"], r["agent"]))
    lines = [
        f"# Case: {slug}",
        "",
        f"**Run:** `{run.run_id}` ({run.label})  **Episode:** {ep}  "
        f"**Steps:** {t0}-{t1}",
        f"**Why shortlisted:** {reason}",
        "",
        "| t | agent | chamber | action | message | events/notes |",
        "|---|---|---|---|---|---|",
    ]
    lines += [_fmt_row(r) for r in rows]
    d = out_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "transcript.md").write_text("\n".join(lines), encoding="utf-8")


def run(args) -> int:
    runs = registry.iter_runs(args.runs_root, args.only)
    flags = provenance.read_jsonl(args.out / "flags" / "flags.jsonl.gz")
    shortlist = []

    ctxs = {}
    for run_ in runs:
        ctx = metrics_mod.load_ctx(run_, args.out)
        if ctx is None or ctx["quarantined"]:
            continue
        ctxs[run_.run_id] = (run_, ctx)

    # 1/2: ch3 coordination + anvil coop, 3: directive chain, 7: hallucination
    best = defaultdict(list)  # archetype -> [(score, run, ep, t0, t1, reason)]
    for rid, (run_, ctx) in ctxs.items():
        for ep_i, evs in ctx["events"].items():
            by_id = {}
            for e in evs:
                mid = e.get("id") or e.get("milestone") or ""
                by_id.setdefault(mid.split("_")[0], e)
            if any(k.startswith("m18") for k in
                   [e.get("id") or e.get("milestone") or "" for e in evs]):
                # exclude presses at chamber-timer boundaries (forced
                # teleports at 20/40/60/80% of max_steps produce
                # timer-race milestones, not coordination)
                ms = int(((ctx["final_metrics"] or {}).get("config", {})
                          .get("cli_args", {}) or {}).get("max_steps", 1000))
                bounds_t = {k * ms // 5 for k in (1, 2, 3, 4)}
                press = [int(e.get("step", 9999)) for e in evs
                         if (e.get("id") or "").startswith("m17")
                         and not any(abs(int(e.get("step", 0)) - b) <= 5
                                     for b in bounds_t)]
                if press:
                    t = min(press)
                    best["best_ch3_coordination"].append(
                        (-t, run_, ep_i, max(0, t - 25), t + 25,
                         f"m18 door opened; first non-timer switch press "
                         f"at t={t}"))
            for e in evs:
                mid = e.get("id") or ""
                if mid.startswith(("m8_", "m9_")):
                    t = int(e.get("step", 0))
                    ml = (ctx["coop_by_ep"].get(ep_i) or {}).get(
                        "milestone_log") or []
                    cbc = any(x.get("milestone", "").startswith(("m8", "m9"))
                              and x.get("comm_before_coop") for x in ml)
                    best["anvil_coop"].append(
                        (int(cbc), run_, ep_i, max(0, t - 30), t + 5,
                         f"{mid} at t={t}, comm_before_coop={cbc}"))
        # directive chain
        for r in ctx["timeline"]:
            s = r.get("social")
            if not (s and s.get("ask_target")):
                continue
            msg = r.get("msg") or {}
            if (msg.get("model_target_canonical") or "") != s["ask_target"]:
                continue
            fut = [e for rr in ctx["timeline"]
                   if rr["ep"] == r["ep"] and r["t"] < rr["t"] <= r["t"] + 30
                   for e in (rr.get("events") or [])]
            coop_ev = [e for e in fut
                       if registry.MILESTONE_TRACK.get(
                           e.get("id") or e.get("milestone") or "")
                       in registry.COOP_TRACKS]
            if coop_ev:
                best["social_directive_chain"].append(
                    (max(e.get("reward", 0) for e in coop_ev), run_,
                     r["ep"], max(0, r["t"] - 5), r["t"] + 32,
                     f"ask {s['ask_target']} -> message sent -> "
                     f"{coop_ev[0].get('id') or coop_ev[0].get('milestone')}"))
        # hebbian fallback density
        fb = [(ep_i, int(m.get("t", -1))) for ep_i, ms in ctx["messages"].items()
              for m in ms if str(m.get("routing", "")).startswith(
                  "hebbian_fallback")]
        if fb:
            ep_i, t = max(((e, t) for e, t in fb),
                          key=lambda et: sum(1 for e2, t2 in fb
                                             if e2 == et[0]
                                             and abs(t2 - et[1]) <= 20))
            best["hebbian_fallback_heavy"].append(
                (len(fb), run_, ep_i, max(0, t - 10), t + 20,
                 f"{len(fb)} hebbian_fallback-routed messages in run"))

    # 5/6 from flags
    for f in flags:
        rid = f"{f['exp']}/seed_{f['seed']}"
        if rid not in ctxs:
            continue
        run_, ctx = ctxs[rid]
        if f["detector"] == "task_repeat_loop":
            best["stuck_ch1_failure"].append(
                ((f.get("detail") or {}).get("n_repeats", 0), run_, f["ep"],
                 f["t0"], min(f["t1"], f["t0"] + 50),
                 f"task repeated x{(f.get('detail') or {}).get('n_repeats')}: "
                 f"{(f.get('detail') or {}).get('task', '')[:60]}"))
        if f["detector"] == "rl_action_collapse":
            best["rl_action_collapse"].append(
                ((f.get("detail") or {}).get("share", 0), run_, f["ep"],
                 f["t0"], f["t0"] + 40,
                 f"RL emits {(f.get('detail') or {}).get('action')} "
                 f"{(f.get('detail') or {}).get('share')} of window"))
        if (f["detector"] == "hallucinated_object"
                and (f.get("detail") or {}).get("source") == "belief"):
            best["hallucination_cascade"].append(
                (1, run_, f["ep"], max(0, f["t0"] - 5), f["t0"] + 15,
                 f"belief mentions {(f.get('detail') or {}).get('objects')} "
                 f"in {(f.get('detail') or {}).get('chamber')}"))

    # 8: bonds track behaviour (+ counterexample) from pair records
    pr = provenance.read_jsonl(args.out / "tables"
                               / "bond_pair_records.jsonl.gz")
    scored = [(r.get("W", 0) * (r.get("joint_dig", 0) + r.get("messages", 0)),
               r) for r in pr]
    if scored:
        scored.sort(key=lambda x: -x[0])
        shortlist.append({"archetype": "bonds_track_behaviour",
                          "record": scored[0][1],
                          "reason": "highest W x interaction product"})
        counter = [r for _s, r in scored
                   if r.get("W", 0) > 0.25
                   and (r.get("joint_dig", 0) + r.get("messages", 0)) < 2]
        if counter:
            shortlist.append({"archetype": "bonds_track_behaviour_counter",
                              "record": counter[0],
                              "reason": "high W, near-zero interaction"})

    rows_csv = []
    for arch, cands in best.items():
        cands.sort(key=lambda x: -x[0])
        # one exemplar per (run, ep): drop same-episode near-duplicates
        seen_ep = set()
        picked = []
        for c in cands:
            key = (c[1].run_id, c[2])
            if key in seen_ep:
                continue
            seen_ep.add(key)
            picked.append(c)
            if len(picked) == 2:
                break
        for rank, (score, run_, ep, t0, t1, reason) in enumerate(picked):
            slug = f"{arch}_{rank}"
            _, ctx = ctxs[run_.run_id]
            render_transcript(run_, ctx["timeline"], ep, t0, t1, slug,
                              reason, args.out / "cases")
            rows_csv.append({"archetype": arch, "rank": rank,
                             "run": run_.run_id, "label": run_.label,
                             "ep": ep, "t0": t0, "t1": t1, "score": score,
                             "reason": reason})
            print(f"[case] {slug}: {run_.run_id} ep{ep} t{t0}-{t1} — {reason}")
    from qual_lib.metrics import _write_csv
    _write_csv(args.out / "cases" / "shortlist.csv", rows_csv,
               ["archetype", "rank", "run", "label", "ep", "t0", "t1",
                "score", "reason"])
    (args.out / "cases" / "bond_exemplars.json").write_text(
        json.dumps(shortlist, indent=1), encoding="utf-8")
    print(f"cases done: {len(rows_csv)} transcripts -> {args.out / 'cases'}")
    return 0
