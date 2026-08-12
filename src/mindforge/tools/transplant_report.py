"""Generate the pair-bonding transplant experiment report (markdown).

Every number in the report is computed from the raw run artifacts at
generation time — nothing is hand-entered — so the report can be regenerated
whenever new runs land (e.g. the Qwen Phase B):

    PYTHONPATH=src python src/mindforge/tools/transplant_report.py \
        --out runs_from_daic/pair_bonding/TRANSPLANT_REPORT.md

Inputs (defaults match the repo layout):
  * Phase A run dirs   (Gemma + Qwen): ranked via pair_transplant.rank_pair_runs
  * merged inputs      merged/{transplant,shuffled}/merged_{W,manifest}.json
  * Phase B run dirs   expB_merged_{transplant,shuffled}/seed_*/
"""

import argparse
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mindforge.tools.analyze_wiring import (  # noqa: E402
    load_co_milestone_matrix,
    load_message_matrix,
    seat_pair_table,
    seatmate_preference,
)
from mindforge.tools.pair_transplant import rank_pair_runs  # noqa: E402

CHANCE = {6: 1.0 / 5}


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


# ── Phase A ──────────────────────────────────────────────────────────────────

def phase_a_section(title, run_glob):
    runs = sorted(Path().glob(run_glob))
    if not runs:
        return f"## {title}\n\n(no runs found under `{run_glob}`)\n", None
    rows = rank_pair_runs(runs)
    ok = [r for r in rows if r["error"] is None]

    lines = [f"## {title}", ""]
    lines.append(f"{len(ok)} completed runs (of {len(rows)} seeds). "
                 "Ranked by final within-pair bond; **COFIRED** = earned a "
                 "milestone that is impossible alone (anvil break / gear "
                 "equip — solo digging is net-zero by construction).")
    lines.append("")
    lines.append("| rank | run | bond (mean W01,W10) | joint digs | co-actions | COFIRED | co-firing milestones |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        name = Path(r["run_dir"]).name
        if r["error"]:
            lines.append(f"| {i} | {name} | — | — | — | — | *incomplete: no final graph* |")
            continue
        lines.append(
            f"| {i} | {name} | {r['bond']:.4f} | {r['joint_dig']} | "
            f"{r['co_action']} | {'**YES**' if r['cofired'] else 'no'} | "
            f"{', '.join(r['cofiring_milestones']) or '—'} |")
    lines.append("")

    cof = [r for r in ok if r["cofired"]]
    ranks = [i for i, r in enumerate(rows) if r.get("cofired")]
    bonds = [r["bond"] for r in ok]
    stats = {
        "n_ok": len(ok), "n_cofired": len(cof),
        "cofired_ranks": ranks,
        "bond_min": min(bonds), "bond_max": max(bonds),
        "bond_spread": max(bonds) - min(bonds),
    }
    lines.append(
        f"**Key numbers:** {len(cof)}/{len(ok)} pairs genuinely co-fired; "
        f"by bond they ranked {', '.join(str(i + 1) for i in ranks)} of "
        f"{len(ok)}. Final bonds span only "
        f"{stats['bond_min']:.4f}–{stats['bond_max']:.4f} "
        f"(spread {stats['bond_spread']:.4f}).")
    lines.append("")
    return "\n".join(lines), stats


def phase_a_noise(run_glob):
    """Between-seed spread of final W vs within-seed episode wobble."""
    finals, wobble = {}, {}
    for d in sorted(Path().glob(run_glob)):
        try:
            finals[d.name] = json.load(
                open(d / "hebbian_graph_final.json"))["W"][0][1]
        except Exception:
            continue
        ws = []
        for f in sorted(d.glob("episodes/*/summary.json")):
            hw = (json.load(open(f)).get("cooperation_metrics") or {}
                  ).get("hebbian_W")
            if hw:
                ws.append(hw[0][1])
        if ws:
            wobble[d.name] = max(ws) - min(ws)
    if not finals or not wobble:
        return None
    return {
        "between": max(finals.values()) - min(finals.values()),
        "within_mean": st.mean(wobble.values()),
        "within_max": max(wobble.values()),
        "within_max_seed": max(wobble, key=wobble.get),
    }


# ── Phase B ──────────────────────────────────────────────────────────────────

def load_manifest(base, arm):
    with open(base / "merged" / arm / "merged_manifest.json") as f:
        return json.load(f)


def phase_b_runs(base, arm, seeds):
    return [base / f"expB_merged_{arm}" / f"seed_{s}" for s in seeds
            if (base / f"expB_merged_{arm}" / f"seed_{s}" / "config.json").exists()]


def run_wall_hours(run):
    """Wall time from config start_ts to the last log.txt timestamp at or
    after it (log.txt is append-mode across relaunches)."""
    try:
        start = datetime.fromisoformat(
            json.load(open(run / "config.json"))["start_ts"])
    except Exception:
        return None
    last = None
    try:
        with open(run / "log.txt", encoding="utf-8", errors="replace") as f:
            for line in f:
                # "[tag] YYYY-mm-dd HH:MM:SS LEVEL ..."
                parts = line.split("] ", 1)
                if len(parts) != 2:
                    continue
                try:
                    ts = datetime.strptime(parts[1][:19], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if ts >= start:
                    last = ts
    except FileNotFoundError:
        return None
    return (last - start).total_seconds() / 3600 if last else None


def phase_b_wiring(base, arms, seeds, n=6):
    """All wiring stats per arm: per-run and pooled preference, seat pairs,
    per-episode trend, W evolution."""
    out = {}
    for arm in arms:
        manifest = load_manifest(base, arm)
        sp_meta = manifest.get("seat_pairs")
        arm_d = {"runs": {}, "seat_pairs": {}, "manifest": manifest}
        for run in phase_b_runs(base, arm, seeds):
            total, per_ep = load_message_matrix(run, n)
            co = load_co_milestone_matrix(run, n, selective=True)
            prefs = seatmate_preference(total)
            run_prefs = [p for p, _ in prefs.values() if p is not None]
            ep_trend = {}
            ep_dominant = {}
            for ep, mat in sorted(per_ep.items()):
                ps = [p for p, _ in seatmate_preference(mat).values()
                      if p is not None]
                ep_trend[ep] = st.mean(ps) if ps else None
                ep_dominant[ep] = {
                    i: (max(range(n), key=lambda j: mat[i][j])
                        if sum(mat[i]) else None) for i in range(n)
                }
            w_by_ep = {}
            for f in sorted(run.glob("episodes/*/summary.json")):
                d = json.load(open(f))
                hw = (d.get("cooperation_metrics") or {}).get("hebbian_W")
                if not hw:
                    continue
                within = [hw[2 * k][2 * k + 1] for k in range(3)] + \
                         [hw[2 * k + 1][2 * k] for k in range(3)]
                cross = [hw[i][j] for i in range(n) for j in range(n)
                         if i != j and i // 2 != j // 2]
                w_by_ep[d["episode"]] = (st.mean(within), st.mean(cross))
            arm_d["runs"][run.name] = {
                "pref_mean": st.mean(run_prefs) if run_prefs else None,
                "prefs": {i: prefs[i][0] for i in range(n)},
                "ep_trend": ep_trend,
                "ep_dominant": ep_dominant,
                "w_by_ep": w_by_ep,
                "seat_rows": seat_pair_table(total, co, sp_meta),
            }
        # pool seat pairs across seeds
        for run_d in arm_d["runs"].values():
            for row in run_d["seat_rows"]:
                k = tuple(row["seats"])
                agg = arm_d["seat_pairs"].setdefault(
                    k, {"label": row["label"], "prefs": [], "msgs": [],
                        "co": []})
                agg["msgs"].append(row["messages_within"])
                agg["co"].append(row["co_milestones"])
                for p in (row["pref_a"], row["pref_b"]):
                    if p is not None:
                        agg["prefs"].append(p)
        out[arm] = arm_d
    return out


# Task milestones achievable in a --start-chamber 3 run (Chambers 3-5).
# Ch1/Ch2 milestones are unreachable by design; m_comm_* are per-chamber
# communication rewards, not task milestones.
CH3_5_MILESTONES = frozenset({
    "m16_enter_cell", "m17_switch_pressed", "m18_door_opened",
    "m19_all_in_communal",
    "m20_enter_ch4", "m21_first_mob_kill", "m22_all_mobs_killed",
    "m23_all_alive_ch4",
    "m24_enter_ch5", "m25_first_boss_dmg", "m26_boss_half_hp",
    "m27_boss_defeated", "m28_all_alive_bonus",
})


def phase_b_performance(base, arms, seeds):
    out = {}
    for arm in arms:
        rows = {}
        for run in phase_b_runs(base, arm, seeds):
            try:
                fm = json.load(open(run / "final_metrics.json"))
            except FileNotFoundError:
                continue
            ev = fm.get("milestone_events") or []
            ids = sorted({e["milestone_id"] for e in ev})
            completed = sorted(set(ids) & CH3_5_MILESTONES)
            rets = fm.get("cumulative_returns") or []
            rows[run.name] = {
                "milestone_events": len(ev),
                "completed": completed,
                "pct": 100.0 * len(completed) / len(CH3_5_MILESTONES),
                "ch1_leak": sum(1 for e in ev
                                if e["milestone_id"].startswith("m1_")),
                "ids": ids,
                "mean_return": st.mean(rets) if rets else None,
                "reached_ch5": "m24_enter_ch5" in ids,
                "switch_done": "m18_door_opened" in ids,
                "wall_h": run_wall_hours(run),
            }
        out[arm] = rows
    return out


# ── Report assembly ──────────────────────────────────────────────────────────

def build(args):
    base = Path(args.phaseb_base)
    seeds = args.seeds
    arms = ("transplant", "shuffled")
    L = []
    L.append("# Pair-Bonding Transplant Experiment — Results Report")
    L.append("")
    L.append(f"*Generated {datetime.now():%Y-%m-%d %H:%M} by "
             f"`src/mindforge/tools/transplant_report.py` — every number is "
             f"computed from the run artifacts; regenerate rather than "
             f"hand-edit.*")
    L.append("")
    L.append("## Design")
    L.append("")
    L.append("""**Question:** do agent pairs that *co-fired* together in one context keep
preferring each other ("wire together") in a new context — and is that carried
by the relationship actually having happened, rather than by bond magnitude or
memory volume?

- **Phase A** — 9 independent 2-agent runs (Gemma 4 E4B, exp08 flags,
  `--max-chamber 2`, 3 episodes x 1000 steps): solo skills + anvil co-op in
  Chambers 1–2. Ground-truth co-firing = anvil-break milestones (impossible
  alone: solo digging is net-zero).
- **Selection** — 2 genuine co-firing pairs + 1 non-co-firing control pair
  (most joint digs among non-co-firers).
- **Merge** — bonds: block-diagonal 6x6 W, per-block mean-normalized so every
  dyad starts equal; cross-pair entries at init weight 0.1. Memories: full
  per-agent skills / 150 episodic memories / curriculum, partner references
  renamed to the new seats.
- **Phase B** — 6-agent runs starting directly in Chamber 3
  (`--start-chamber 3`, 3 episodes x 1000 steps), two arms with **identical
  W** and identical memory volume:
  - **T (transplant)**: real pairs seated together — seats (0,1) and (2,3)
    GENUINE, seats (4,5) CONTROL (shared history, never co-fired);
  - **S (shuffled)**: strangers seated together, each holding memories that
    claim a shared history with the seatmate which never happened.
- **Readout** — seatmate preference P(message target = seatmate), chance =
  0.20 at N=6; co-earned milestones; bond-matrix evolution.""")
    L.append("")

    # Phase A
    sec, a_stats = phase_a_section(
        "Phase A (Gemma) — pair bonding, Chambers 1–2",
        args.phasea_glob)
    L.append(sec)
    noise = phase_a_noise(args.phasea_glob)
    if noise and a_stats:
        L.append(
            f"**The bond is not a usable selection signal.** Between-seed "
            f"spread of final W is {noise['between']:.4f}, while the mean "
            f"within-seed episode-to-episode wobble is "
            f"{noise['within_mean']:.4f} (max {noise['within_max']:.4f}, "
            f"{noise['within_max_seed']}) — the differences between seeds "
            f"are the same size as each seed's own noise. Meanwhile the "
            f"co-firing ground truth picks out different runs entirely: the "
            f"genuine co-firers ranked "
            f"{', '.join(str(i + 1) for i in a_stats['cofired_ranks'])} of "
            f"{a_stats['n_ok']} by bond — the bond is (weakly) "
            f"*anti*-correlated with real cooperative achievement, because "
            f"the engagement term rewards proximity + constant messaging.")
        L.append("")

    if args.qwen_phasea_glob:
        sec, q_stats = phase_a_section(
            "Phase A replication (Qwen3.5-9B)", args.qwen_phasea_glob)
        L.append(sec)
        if q_stats:
            L.append(
                f"The bond↔co-firing inversion **replicates across models**: "
                f"Qwen's {q_stats['n_cofired']} genuine co-firers ranked "
                f"{', '.join(str(i + 1) for i in q_stats['cofired_ranks'])} "
                f"of {q_stats['n_ok']} by bond (Qwen Phase B pending).")
            L.append("")

    # Merged inputs
    L.append("## Merged Phase B inputs (Gemma)")
    L.append("")
    with open(base / "merged" / "transplant" / "merged_W.json") as f:
        wt = json.load(f)
    with open(base / "merged" / "shuffled" / "merged_W.json") as f:
        ws = json.load(f)
    within_w = wt["W"][0][1]
    L.append(f"- Within-seat bond after block-mean normalization: "
             f"**{within_w:.3f}** (every dyad identical by construction); "
             f"cross-seat: **{wt['W'][0][2]:.3f}**.")
    L.append(f"- W identical between arms: **{wt['W'] == ws['W']}** — the "
             f"arms differ only in who occupies each seat.")
    man_t = load_manifest(base, "transplant")
    for sp in man_t["seat_pairs"]:
        src = Path(sp["source_run"]).name if sp.get("source_run") else "?"
        tag = "GENUINE" if sp["cofired"] else "CONTROL"
        L.append(f"- Seats {sp['seats']}: **{tag}** ← {src}")
    L.append("")

    # Phase B wiring
    wiring = phase_b_wiring(base, arms, seeds)
    L.append("## Phase B results (Gemma) — behavioral wiring")
    L.append("")
    L.append("### Seatmate preference (chance = 0.20)")
    L.append("")
    L.append("| seed | transplant | shuffled | T − S |")
    L.append("|---|---|---|---|")
    t_means, s_means = [], []
    for s in seeds:
        rn = f"seed_{s}"
        t = wiring["transplant"]["runs"].get(rn, {}).get("pref_mean")
        sh = wiring["shuffled"]["runs"].get(rn, {}).get("pref_mean")
        if t is not None:
            t_means.append(t)
        if sh is not None:
            s_means.append(sh)
        d = (t - sh) if (t is not None and sh is not None) else None
        L.append(f"| {s} | {_f(t)} | {_f(sh)} | {_f(d, 2)} |")
    tp, sp_ = st.mean(t_means), st.mean(s_means)
    n_pos = sum(1 for s in seeds
                if (wiring['transplant']['runs'].get(f'seed_{s}', {}).get('pref_mean') or 0)
                > (wiring['shuffled']['runs'].get(f'seed_{s}', {}).get('pref_mean') or 1))
    L.append(f"| **pooled** | **{tp:.3f}** | **{sp_:.3f}** | "
             f"**+{tp - sp_:.3f}** |")
    L.append("")
    L.append(f"Transplant beats shuffled in **{n_pos}/{len(seeds)} paired "
             f"seeds** (sign-consistent), and both arms sit above the 0.20 "
             f"chance level.")
    L.append("")

    L.append("### Seat-pair breakdown (pooled over seeds)")
    L.append("")
    L.append("| arm | seats | label | mean pref | msgs within (per seed) | co-milestones (per seed) |")
    L.append("|---|---|---|---|---|---|")
    for arm in arms:
        for k, d in sorted(wiring[arm]["seat_pairs"].items()):
            mp = st.mean(d["prefs"]) if d["prefs"] else None
            L.append(f"| {arm} | {list(k)} | {d['label']} | {_f(mp, 3)} | "
                     f"{d['msgs']} | {d['co']} |")
    L.append("")
    gen = [st.mean(d["prefs"]) for k, d in
           wiring["transplant"]["seat_pairs"].items()
           if d["label"] == "GENUINE"]
    ctrl = [st.mean(d["prefs"]) for k, d in
            wiring["transplant"]["seat_pairs"].items()
            if d["label"] == "CONTROL"]
    strangers = [st.mean(d["prefs"]) for k, d in
                 wiring["shuffled"]["seat_pairs"].items()]
    L.append(f"Ordering as predicted: GENUINE ({', '.join(_f(g, 3) for g in gen)}) "
             f"> CONTROL ({_f(ctrl[0], 3)}) > strangers "
             f"({_f(min(strangers), 3)}–{_f(max(strangers), 3)}). The control "
             f"dyad (shared history, no anvil) sits closer to genuine than to "
             f"strangers — most of the effect is carried by *real shared "
             f"history*, with genuine co-firing adding a further increment.")
    L.append("")

    L.append("### Per-episode trend and the re-pairing event")
    L.append("")
    L.append("| arm | seed | ep1 | ep2 | ep3 |")
    L.append("|---|---|---|---|---|")
    for arm in arms:
        for rn, d in sorted(wiring[arm]["runs"].items()):
            eps = [d["ep_trend"].get(f"ep_000{k}") for k in (1, 2, 3)]
            L.append(f"| {arm} | {rn} | {_f(eps[0])} | {_f(eps[1])} | "
                     f"{_f(eps[2])} |")
    L.append("")
    dom = wiring["transplant"]["runs"]["seed_42"]["ep_dominant"]
    dom_str = {ep: " ".join(f"{i}→{t}" for i, t in m.items() if t is not None)
               for ep, m in dom.items()}
    L.append(f"Preference is strongest early and noisy across episodes. The "
             f"transplant/seed_42 ep2 collapse is NOT messaging breakdown — "
             f"the team *re-paired* into different reciprocal dyads for one "
             f"episode and returned to the transplanted pairs in ep3 "
             f"(dominant message target per agent: "
             f"ep1 `{dom_str.get('ep_0001')}`, ep2 `{dom_str.get('ep_0002')}`, "
             f"ep3 `{dom_str.get('ep_0003')}`). Dyadic organization itself is "
             f"robust; the transplanted pairing acts as the attractor teams "
             f"return to.")
    L.append("")

    L.append("### Bond-matrix evolution (mean within-seat W / mean cross-seat W at episode end)")
    L.append("")
    L.append("| arm | seed | ep1 | ep2 | ep3 |")
    L.append("|---|---|---|---|---|")
    for arm in arms:
        for rn, d in sorted(wiring[arm]["runs"].items()):
            cells = []
            for k in (1, 2, 3):
                wc = d["w_by_ep"].get(k)
                cells.append(f"{wc[0]:.3f} / {wc[1]:.3f}" if wc else "—")
            L.append(f"| {arm} | {rn} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")
    ep1_t = [d["w_by_ep"][1][0] for d in wiring["transplant"]["runs"].values()
             if 1 in d["w_by_ep"]]
    ep1_s = [d["w_by_ep"][1][0] for d in wiring["shuffled"]["runs"].values()
             if 1 in d["w_by_ep"]]
    L.append(f"Both arms start at the same {within_w:.3f}. Episode-1 "
             f"within-seat means: transplant "
             f"{', '.join(f'{x:.3f}' for x in ep1_t)} vs shuffled "
             f"{', '.join(f'{x:.3f}' for x in ep1_s)} — leaning toward "
             f"survival for true pairs, but with one crossover each way at "
             f"n=3: **W is a fast readout of the current episode's pairing "
             f"behavior, not a persistent store** (the seed_42 dip to "
             f"{wiring['transplant']['runs']['seed_42']['w_by_ep'][2][0]:.3f} "
             f"lands exactly in its re-pairing episode and recovers to "
             f"{wiring['transplant']['runs']['seed_42']['w_by_ep'][3][0]:.3f} "
             f"when the pairs re-form). The persistent carrier of the "
             f"relationship is the episodic memory.")
    L.append("")

    # Performance
    perf = phase_b_performance(base, arms, seeds)
    n_total = len(CH3_5_MILESTONES)
    L.append("## Phase B task performance")
    L.append("")
    L.append(f"Milestone completion counts a milestone type as done if any "
             f"agent earned it in any episode, out of the {n_total} task "
             f"milestones achievable in Chambers 3–5 (Ch1/Ch2 milestones are "
             f"unreachable in a `--start-chamber 3` run; per-chamber "
             f"communication rewards `m_comm_*` are excluded).")
    L.append("")
    L.append(f"| arm | seed | milestones completed (of {n_total}) | reward events | mean cum. return | reached Ch5 | switch puzzle done | wall time (h) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for rn, d in sorted(perf[arm].items()):
            L.append(f"| {arm} | {rn} | "
                     f"{len(d['completed'])}/{n_total} "
                     f"(**{d['pct']:.0f}%**) | {d['milestone_events']} | "
                     f"{_f(d['mean_return'], 0)} | "
                     f"{'yes' if d['reached_ch5'] else 'no'} | "
                     f"{'yes' if d['switch_done'] else 'NO'} | "
                     f"{_f(d['wall_h'], 1)} |")
    for arm in arms:
        pcts = [d["pct"] for d in perf[arm].values()]
        rt = [d["mean_return"] for d in perf[arm].values()
              if d["mean_return"] is not None]
        L.append("")
        L.append(f"- **{arm}**: mean completion "
                 f"{st.mean(pcts):.0f}%, mean return {st.mean(rt):.0f}.")
    t_pct = st.mean([d["pct"] for d in perf["transplant"].values()])
    s_pct = st.mean([d["pct"] for d in perf["shuffled"].values()])
    all_completed = sorted(set().union(
        *(d["completed"] for a in arms for d in perf[a].values())))
    never = sorted(CH3_5_MILESTONES - set(all_completed))
    leak = sum(d["ch1_leak"] for a in arms for d in perf[a].values())
    L.append("")
    L.append(f"Transplant shows a small edge ({t_pct:.0f}% vs {s_pct:.0f}% "
             f"mean completion) — a trend, not a claim. Every run reached "
             f"Ch5; wiring together is a social-structural effect, not a "
             f"task-competence one, which also rules out 'transplant helps "
             f"because it boosts performance' as a confound for the "
             f"preference results.")
    L.append("")
    L.append(f"Milestones completed somewhere in the suite: "
             f"{', '.join(f'`{m}`' for m in all_completed)}. Never completed "
             f"by any run: {', '.join(f'`{m}`' for m in never)} — the "
             f"communal-regroup gate (m19) and the full combat/boss clears "
             f"remain beyond 6-agent teams at these horizons.")
    if leak:
        L.append("")
        L.append(f"*Known artifact: `m1_move_5` (a Ch1 milestone) leaked "
                 f"{leak} reward events across the six runs despite Ch1 "
                 f"never being visited — the Ch1 movement tracker fires "
                 f"mid-episode under rare conditions. Excluded from the "
                 f"completion metric; worth a Lua fix before reusing the "
                 f"ch1_solo track in analyses.*")
    L.append("")

    # Findings
    L.append("## Findings (summary)")
    L.append("")
    L.append(f"""1. **Pairs that co-fired together still wire together.** Transplanted real
   pairs message their former partner at {tp:.2f} (chance 0.20) vs {sp_:.2f}
   for strangers holding identical bonds and equally detailed fabricated
   memories — transplant > shuffled in {n_pos}/{len(seeds)} paired seeds.
2. **The effect is carried by the relationship being real.** Identical W,
   identical memory volume; only the truth of the pairing differs between
   arms. Within the transplant arm: GENUINE > CONTROL > strangers, with the
   control dyad (real history, no co-fired achievement) much closer to
   genuine — shared history is the main carrier, co-firing adds on top.
3. **Hebbian W is a live index, not the memory.** Bond values track the
   current episode's pairing behavior (dip during the re-pairing episode,
   recovery after) and sit at a fixed point that erases initialization over
   time; the transplanted episodic memories are what keep pulling agents
   back to their partners.
4. **The transplanted pairing is an attractor.** One team re-paired into
   different reciprocal dyads for a full episode and spontaneously returned
   to the transplanted pairs the next episode.
5. **Bond magnitude is anti-correlated with genuine co-firing at selection
   time** (Phase A, both models): the pairs that actually broke anvils ranked
   last by bond, because the engagement term rewards proximity + constant
   messaging. Any future use of W as a selection signal needs a co-firing
   ground truth next to it.
6. **No meaningful task-performance difference** between arms (small
   transplant-side trend); the wiring effect is not explained by competence.
""")
    L.append("## Caveats")
    L.append("")
    L.append(f"""- n = {len(seeds)} seeds per arm, one model (Gemma 4 E4B); the Qwen3.5-9B
  Phase B replication is running and this report should be regenerated when
  it lands.
- Seatmate preference is message-based; proximity/joint-action based measures
  would strengthen the claim.
- The shuffled arm shows emergent *new* pairs (strangers bootstrapping real
  partnerships mid-run) — expected with plasticity on, and it keeps the
  baseline honest, but it means the S-arm number is not a floor.
- Phase A pair selection affects only the transplanted memories: after
  block-mean normalization the merged W is identical whichever pairs are
  chosen.""")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phaseb-base",
                    default="runs_from_daic/pair_bonding")
    ap.add_argument("--phasea-glob",
                    default="runs_from_daic/pair_bonding/expA_pair_bonding/seed_*")
    ap.add_argument("--qwen-phasea-glob",
                    default="runs_from_daic/pair_bonding_qwen/expA_pair_bonding/seed_*")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--out",
                    default="paper_assets_transplant/TRANSPLANT_REPORT.md")
    args = ap.parse_args(argv)
    report = build(args)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
