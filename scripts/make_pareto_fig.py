"""Pareto figures for the model-size sweep: compute (x) vs performance (y).

Emits TWO figures from the same data, per the with/without-Qwen request:

    pareto_gemma.png       Gemma 4 only -- the within-family scaling claim
    pareto_gemma_qwen.png  Gemma 4 + Qwen3.5 -- adds a second family as a
                           replication check that the trend is not an artifact
                           of one architecture

Families are drawn as SEPARATE SERIES, never merged into one fitted frontier:
across families, size is confounded with architecture and training data. The
FLOPs x-axis is what makes putting them on one canvas legitimate at all --
per the scaling-law convention it is the only cross-model-valid compute axis
(token and call counts are not comparable once N differs).

Encoding (dataviz skill):
    colour  = arm      base #2a78d6 / hebbian #eb6834  (palette slots 1-2,
                       documented as validating all-pairs in both modes)
    shape   = family   circle Gemma / square Qwen  -- identity is never
                       carried by colour alone
    x       = inference FLOPs, log scale (2*N_eff*tokens, scripts/compute_flops.py)
    y       = performance, mean over agents then over seeds; error bars are
              the across-seed standard deviation

Usage:
    python scripts/make_pareto_fig.py runs_from_daic/pareto_gemma4 \
        runs_from_daic/new_exp_0_gemma --out-dir paper_assets_pareto
    python scripts/make_pareto_fig.py ... --metric return
    python scripts/make_pareto_fig.py ... --theme dark

Any size with no finished runs is skipped and reported, so this is safe to run
on a partial sweep -- the figure just grows as jobs land.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))            # make_results.py lives at the root
sys.path.insert(0, str(REPO / "scripts"))
import make_results as MR                  # noqa: E402
from compute_flops import analyze_run      # noqa: E402

# N_eff is what the 2*N*tokens rule counts: parameters in the ACTIVE compute
# path. Two deliberate choices, both stated in the paper:
#   E2B/E4B -> effective, not raw. Per-layer embeddings are looked up, not
#              multiplied, so raw 5B/8B would overstate FLOPs by ~1.8x.
#   26B-A4B -> ACTIVE (3.8B), not the 25.2B total. On a FLOPs axis an MoE has
#              one well-defined position, which is exactly the ambiguity the
#              parameter axis cannot resolve.
SIZES = {
    "e2b":    dict(family="gemma", n_eff=2.3e9,  label="Gemma E2B"),
    "e4b":    dict(family="gemma", n_eff=4.5e9,  label="Gemma E4B"),
    "12b":    dict(family="gemma", n_eff=12.0e9, label="Gemma 12B"),
    "26b":    dict(family="gemma", n_eff=3.8e9,  label="Gemma 26B-A4B"),
    "31b":    dict(family="gemma", n_eff=30.7e9, label="Gemma 31B"),
    "qwen2b": dict(family="qwen",  n_eff=2.0e9,  label="Qwen3.5-2B"),
    "qwen9b": dict(family="qwen",  n_eff=9.0e9,  label="Qwen3.5-9B"),
}

ARM_COLOR = {"base": "#2a78d6", "hebbian": "#eb6834"}
FAMILY_MARKER = {"gemma": "o", "qwen": "s"}

INK = {
    "light": dict(surface="#ffffff", primary="#1a1a19",
                  secondary="#55554e", grid="#e4e4e0"),
    "dark":  dict(surface="#1a1a19", primary="#ffffff",
                  secondary="#c3c2b7", grid="#33332f"),
}
# Dark steps for the same two hues, per the palette's dark column.
ARM_COLOR_DARK = {"base": "#3987e5", "hebbian": "#d95926"}


# Runs that predate this suite but ARE valid Pareto points, because their
# protocol and flags are identical to what new_exp_pareto.sbatch submits:
# 3 episodes x 1000 steps x 3 agents, base = "--simultaneous" and hebbian =
# "--hebbian --hebbian-eta-0 0.005 --hebbian-reward-norm 50 --hebbian-decay
# 0.005 --hebbian-gamma 0.2 --social-module prompt --simultaneous" (verified
# against the exp0*.sbatch files -- new_exp_pareto's arm was written to mirror
# exp07 exactly). Reusing them is why the Qwen family needs no new GPU time.
#
# CAVEAT to state in the paper: these ran on wiredtogether.sif, while the Gemma
# points ran on wiredtogether_gemma4.sif (newer transformers/torch). That is
# tolerable only because the families are plotted as separate series and never
# pooled into one frontier -- the stack difference lies BETWEEN series, never
# within one.
LEGACY_RUNS = {
    "exp01_llm_2b":               ("qwen2b", "base"),
    "exp02_llm_9b":               ("qwen9b", "base"),
    "exp07_llm_2b_social_prompt": ("qwen2b", "hebbian"),
    "exp08_llm_9b_social_prompt": ("qwen9b", "hebbian"),
    "new_exp_0_gemma_base":       ("e4b", "base"),
    "new_exp_0_gemma_hebbian":    ("e4b", "hebbian"),
}


def exp_to_size_arm(name: str):
    """Map a run directory name to (size, arm), or None if it is not ours."""
    if name in LEGACY_RUNS:
        return LEGACY_RUNS[name]
    if name.startswith("pareto_"):
        rest = name[len("pareto_"):]
        for arm in ("base", "hebbian"):
            if rest.endswith("_" + arm):
                size = rest[: -len(arm) - 1]
                return (size, arm) if size in SIZES else None
    return None


# The SINGLE source of truth for the performance metrics of every pareto
# figure and table (make_pareto_grid.py imports run_metrics from here). All
# five mirror make_results.py so the plots agree with the paper tables.
#
# History: the first version of this script used
# final_metrics["mean_milestone_count_per_agent"], which is
# (task completions + COMMUNICATION completions) / n_agents. m_comm_ch1-5
# fire whenever agents talk (~12-13 per-agent completions per episode in
# every arm), so that field is dominated by a talk reward: the 12B Hebbian
# "advantage" it showed was entirely the comm term (task completions were a
# 9.00 vs 9.00 tie). Never use it as a performance axis.
METRIC_LABEL = {
    "reward":           "Task return (team, per episode)",
    "milestone_pct":    "Milestone coverage (team-distinct) [%]",
    "coop_pct":         "Coop. milestone coverage (team-distinct) [%]",
    "completions":      "Milestone completions (all agents, per episode)",
    "coop_completions": "Coop. milestone completions (all agents, per episode)",
}


def run_metrics(run: dict) -> dict:
    """Per-run performance metrics from a make_results.load_runs() record.

    reward          team task return per episode (task + comm streams,
                    hebbian_diffuse excluded, chamber-entry honesty filter)
    milestone_pct   100 * |distinct team milestones outside the social-act
                    tracks| / NONCOMM_MAX, per episode
    coop_pct        100 * |distinct team milestones in Ch2-Ch5| / COOP_MAX
    completions     per-agent milestone completions summed over agents
                    (ATTAINMENT: the env credits a milestone only to the agents
                    who earned or took part in it -- fire_milestone's
                    contributor lists -- so this counts agents that got there,
                    not one event x3). Social-act tracks excluded.
    coop_completions  same, restricted to Ch2-Ch5.
    Every value is a per-episode list (3 entries for a 3-episode run).
    """
    sets = MR.episode_milestone_sets(run)
    if not sets:
        return {}
    returns, _ = MR.episode_task_returns(run)
    ms_pct = [100.0 * sum(1 for m in s
                          if MR.MILESTONE_TRACK.get(m) not in MR.SOCIAL_ACT_TRACKS)
              / MR.NONCOMM_MAX for s in sets]
    coop_pct = [100.0 * MR.coop_count(s) / MR.COOP_MAX for s in sets]
    per_agent = run.get("milestones_per_episode", [])
    compl, coop_compl = [], []
    for e in range(len(sets)):
        c = cc = 0
        for a in per_agent:
            if e < len(a):
                ms = set(a[e])
                c += sum(1 for m in ms
                         if MR.MILESTONE_TRACK.get(m) not in MR.SOCIAL_ACT_TRACKS)
                cc += MR.coop_count(ms)
        compl.append(c)
        coop_compl.append(cc)
    # PER-EPISODE lists, not per-run means: callers pool episodes across
    # seeds, which is the paper's convention (mean +- population SD over the
    # pooled episodes of all seeds, n = seeds x 3; tab:final_comparison).
    out = {"milestone_pct": ms_pct, "coop_pct": coop_pct,
           "completions": compl, "coop_completions": coop_compl}
    if returns:
        out["reward"] = list(returns)
    return out


def collect(roots, flops_args):
    """{(size, arm): {"flops": [...], "perf": {metric: [...]}, "proto": [...]}}."""
    out = defaultdict(lambda: {"flops": [], "proto": [], "n_runs": 0,
                               "perf": {m: [] for m in METRIC_LABEL}})
    for root in roots:
        if not root.is_dir():
            print("  (skipping missing root {})".format(root))
            continue
        for cond in sorted(p for p in root.iterdir() if p.is_dir()):
            sa = exp_to_size_arm(cond.name)
            if sa is None:
                continue
            size, arm = sa
            # make_results.load_runs applies the chamber-entry honesty filter
            # and the episode slicing that the paper's numbers rely on.
            for run in MR.load_runs(root, cond.name):
                run_dir = Path(run["_path"]).parent
                perf = run_metrics(run)
                if not perf:
                    continue
                args = SimpleNamespace(n_eff=SIZES[size]["n_eff"], **flops_args)
                row = analyze_run(run_dir, args)
                if row is None:
                    print("  !! {} has final_metrics but no log.txt/llm_logs — "
                          "cannot compute FLOPs, skipped".format(run_dir))
                    continue
                if row["decode_tokens_exact"] == 0:
                    print("  !! {} logged ZERO generated tokens — the model "
                          "never answered; excluded".format(run_dir))
                    continue
                out[(size, arm)]["flops"].append(row["flops"])
                out[(size, arm)]["n_runs"] += 1
                for metric, vals in perf.items():   # pool the episodes
                    out[(size, arm)]["perf"][metric].extend(vals)
                out[(size, arm)]["proto"].append(protocol(run_dir))
    return out


def protocol(run_dir: Path):
    """(num_episodes, max_steps_per_episode) — the run's protocol."""
    try:
        c = json.loads((run_dir / "final_metrics.json").read_text(
            encoding="utf-8")).get("config", {})
        return (c.get("num_episodes"), c.get("max_steps_per_episode"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return (None, None)


def mean_sd(xs):
    """Mean and POPULATION standard deviation (divide by n, not n-1).

    Matches make_results.mean_std / statistics.pstdev, i.e. the paper's
    "mean +- std over the pooled episodes of all seeds".
    """
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def fmt_flops_tex(x):
    """1.03e+18 -> 1.03\\times10^{18}, so the table reads like the axis."""
    import math
    e = int(math.floor(math.log10(x)))
    return "{:.2f}\\times10^{{{}}}".format(x / 10 ** e, e)


def write_tables(data, out_dir, stem="pareto_results"):
    """Results table as LaTeX (for the paper) and Markdown (to read here).

    One row per (model, arm) rather than a base/hebbian column pair: the two
    arms do NOT share an x position (the Hebbian arm makes more social calls,
    so it costs more compute), and a shared row would hide that.
    """
    rows = []
    for size, meta in SIZES.items():
        for arm in ("base", "hebbian"):
            d = data.get((size, arm))
            if not d or not d["flops"]:
                continue
            fx, _ = mean_sd(d["flops"])
            cells = {}
            for m in METRIC_LABEL:
                vals = d["perf"][m]
                cells[m] = mean_sd(vals) if vals else None
            rows.append((meta["family"], meta["label"], meta["n_eff"], arm,
                         d["n_runs"], fx, cells))
    rows.sort(key=lambda r: (r[0] != "gemma", r[2], r[3] != "base"))

    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / (stem + ".tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% Generated by scripts/make_pareto_fig.py — do not hand-edit.\n")
        f.write("% Compute = 2*N_eff*(prefill+decode tokens), summed over all\n")
        f.write("% agents/modules/retries. N_eff is ACTIVE parameters: effective\n")
        f.write("% (not raw) for the Gemma E-series, active (not total) for MoE.\n")
        f.write("% Coverage = distinct team milestones (excl. comm track) as a\n")
        f.write("% percentage of the 25 task milestones (Coop.: of the 17 in\n")
        f.write("% Ch2-Ch5). Completions = per-agent milestone completions summed\n")
        f.write("% over agents per episode (the env credits only contributors).\n")
        f.write("\\begin{tabular}{llrlrrrrrr}\n\\toprule\n")
        f.write("Family & Model & $N_{\\mathrm{eff}}$ & Arm & $n$ & "
                "Compute [FLOPs] & Coverage [\\%] & Coop.\\ cov.\\ [\\%] & "
                "Completions & Task return \\\\\n\\midrule\n")
        prev_family = None
        for family, label, n_eff, arm, n, fx, cells in rows:
            if prev_family is not None and family != prev_family:
                f.write("\\midrule\n")
            prev_family = family
            def cell(m):
                c = cells.get(m)
                return "--" if c is None else "${:.2f} \\pm {:.2f}$".format(*c)
            f.write("{} & {} & {:.1f}B & {} & {} & ${}$ & {} & {} & {} & {} \\\\\n".format(
                family.capitalize(), label, n_eff / 1e9, arm, n,
                fmt_flops_tex(fx), cell("milestone_pct"), cell("coop_pct"),
                cell("completions"), cell("reward")))
        f.write("\\bottomrule\n\\end{tabular}\n")

    md = out_dir / (stem + ".md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("| Family | Model | N_eff | Arm | n | Compute [FLOPs] | "
                "Coverage [%] | Coop. cov. [%] | Completions | Task return |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for family, label, n_eff, arm, n, fx, cells in rows:
            def cell(m):
                c = cells.get(m)
                return "—" if c is None else "{:.2f} ± {:.2f}".format(*c)
            f.write("| {} | {} | {:.1f}B | {} | {} | {:.2e} | {} | {} | {} | {} |\n".format(
                family.capitalize(), label, n_eff / 1e9, arm, n, fx,
                cell("milestone_pct"), cell("coop_pct"), cell("completions"),
                cell("reward")))
    print("  wrote {}\n  wrote {}".format(tex, md))
    return md


def draw(data, families, out_path, metric, theme):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink = INK[theme]
    colors = ARM_COLOR if theme == "light" else ARM_COLOR_DARK

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])

    plotted = 0
    labels = {}
    for family in families:
        for arm in ("base", "hebbian"):
            pts = []
            for size, meta in SIZES.items():
                if meta["family"] != family:
                    continue
                d = data.get((size, arm))
                if not d or not d["flops"]:
                    continue
                fx, _ = mean_sd(d["flops"])
                vals = d["perf"][metric]
                if not vals:
                    continue
                py, ps = mean_sd(vals)
                pts.append((fx, py, ps, meta["label"], d["n_runs"]))
            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            # One line per (family, arm): the within-family trend. Families are
            # never joined to each other.
            ax.errorbar(
                xs, ys, yerr=es,
                color=colors[arm], marker=FAMILY_MARKER[family],
                markersize=9, linewidth=2, capsize=3, elinewidth=1.5,
                markeredgecolor=ink["surface"], markeredgewidth=1.5,
                label="{} — {}".format(family.capitalize(), arm),
                zorder=3,
            )
            plotted += 1
            # Label each SIZE once, not once per arm: the two arms sit at
            # nearly the same x, so per-arm labels collide, and the size is the
            # same fact either way. Remember the topmost point (incl. its error
            # bar) so the label clears both arms; drawn after the loop.
            for x, y, e, lab, n in pts:
                key = lab.split()[-1]
                prev = labels.get(key)
                top = y + e
                if prev is None or top > prev[1]:
                    labels[key] = (x, top, n)

    for key, (x, top, n) in labels.items():
        ax.annotate("{} (n={})".format(key, n), (x, top),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7.5, color=ink["secondary"],
                    zorder=4)

    if plotted == 0:
        print("  nothing to plot for {} — no finished runs yet".format(families))
        plt.close(fig)
        return False

    ax.set_xscale("log")
    # Headroom so the direct labels (drawn above the topmost error bar, and at
    # the extreme x points) are not clipped by the axes box.
    ax.margins(x=0.10, y=0.16)
    ylabel = METRIC_LABEL[metric]
    ax.set_xlabel("Inference compute  [FLOPs, $2\\,N_{\\mathrm{eff}}\\,T$]",
                  fontsize=10, color=ink["primary"])
    ax.set_ylabel(ylabel, fontsize=10, color=ink["primary"])
    ax.tick_params(colors=ink["secondary"], labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink["grid"])
    ax.grid(True, which="major", color=ink["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Legend BELOW the axes in one row: loc="best" lands on the data whenever
    # the trend runs up-and-right (it collided with the top-right size label),
    # and "best" is data-dependent, so the collision would come and go as runs
    # land. Outside the axes it can never collide.
    leg = ax.legend(frameon=False, fontsize=9, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), ncol=min(plotted, 4),
                    handletextpad=0.6, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(ink["primary"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=ink["surface"])
    plt.close(fig)
    print("  wrote {}".format(out_path))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path,
                    help="run-group dirs holding <exp>/<seed>/ (e.g. "
                         "runs_from_daic/pareto_gemma4 runs_from_daic/new_exp_0_gemma)")
    ap.add_argument("--out-dir", type=Path, default=Path("paper_assets_pareto"))
    ap.add_argument("--metric", default="milestone_pct",
                    choices=list(METRIC_LABEL))
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--with-qwen", action="store_true",
                    help="also emit the Gemma+Qwen figure (off by default: "
                         "Qwen is excluded from the paper's plots)")
    ap.add_argument("--image-tokens", type=int, default=280)
    ap.add_argument("--overhead-tokens", type=int, default=60)
    ap.add_argument("--chars-per-token", type=float, default=None)
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the plotted points as CSV")
    args = ap.parse_args()

    flops_args = dict(image_tokens=args.image_tokens,
                      overhead_tokens=args.overhead_tokens,
                      chars_per_token=args.chars_per_token)
    data = collect(args.roots, flops_args)
    if not data:
        sys.exit("no finished runs found under: {}".format(
            ", ".join(str(r) for r in args.roots)))

    print("\npoints found (size, arm) -> seeds:")
    for (size, arm), d in sorted(data.items()):
        fx, _ = mean_sd(d["flops"])
        vals = d["perf"][args.metric]
        py, ps = mean_sd(vals)
        print("  {:<8} {:<8} seeds={} eps={}  FLOPs={:.2e}  {}={:.2f}±{:.2f}".format(
            size, arm, d["n_runs"], len(vals), fx, args.metric, py, ps))
    missing = [s for s in SIZES if not any(k[0] == s for k in data)]
    if missing:
        print("  (no finished runs yet for: {})".format(", ".join(missing)))

    # A 1x50 smoke run and a 3x1000 production run differ ~100x in compute, so
    # pooling the two trees produces a figure that looks like a scaling trend
    # and is not one. Refuse to do that silently.
    protos = {pr for d in data.values() for pr in d["proto"]}
    if len(protos) > 1:
        print("\n  !! MIXED PROTOCOLS across the runs collected: {}".format(
            sorted(protos)))
        print("     (num_episodes, max_steps_per_episode) must match for the "
              "compute axis to be comparable — you are probably pointing at a")
        print("     _smoke tree and a production tree at once. Per-size detail:")
        for (size, arm), d in sorted(data.items()):
            if len(set(d["proto"])) > 1 or set(d["proto"]) != protos:
                print("       {:<8} {:<8} {}".format(
                    size, arm, sorted(set(d["proto"]))))

    print("\ntables:")
    write_tables(data, args.out_dir)

    print("\nfigures:")
    suffix = "" if args.theme == "light" else "_dark"
    draw(data, ["gemma"], args.out_dir / "pareto_gemma{}.png".format(suffix),
         args.metric, args.theme)
    # Qwen is excluded from the paper's plots by decision (2026-08-28); the
    # two-family figure stays available behind a flag for comparison only.
    if args.with_qwen:
        draw(data, ["gemma", "qwen"],
             args.out_dir / "pareto_gemma_qwen{}.png".format(suffix),
             args.metric, args.theme)

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            cols = ["size", "family", "arm", "n_seeds", "n_eff", "mean_flops"]
            for m in METRIC_LABEL:
                cols += ["mean_" + m, "sd_" + m]
            w.writerow(cols)
            for (size, arm), d in sorted(data.items()):
                fx, _ = mean_sd(d["flops"])
                row = [size, SIZES[size]["family"], arm,
                       d["n_runs"], SIZES[size]["n_eff"],
                       "{:.4e}".format(fx)]
                for m in METRIC_LABEL:
                    vals = d["perf"][m]
                    if vals:
                        mm, ss = mean_sd(vals)
                        row += ["{:.4f}".format(mm), "{:.4f}".format(ss)]
                    else:
                        row += ["", ""]
                w.writerow(row)
        print("  wrote {}".format(args.csv))


if __name__ == "__main__":
    main()
