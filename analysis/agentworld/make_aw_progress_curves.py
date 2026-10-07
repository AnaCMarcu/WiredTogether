"""Team progress over rounds, baseline vs +Hebbian on the same axes (AgentWorld).

One panel per team size N; x = round; y = mean verifier progress over teams
(``--metric progress``) or the fraction of teams solved (``--metric solved``);
one line per arm with a ±1 SE band over seeds. Only the arms named in
``--arms`` are drawn (default: base and hebbian, as in the WIRE figures).

Runs are found by their ``config.json`` (arm, replicas) under the given roots;
each ``episode_k/replay/state.jsonl`` contributes one curve per team (the last
value is carried forward after a team stops early).

    python analysis/agentworld/make_aw_progress_curves.py runs_aw/<group> \\
        [--metric progress|solved] [--episode 1] [--out fig.png]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ARM_COLOR = {"base": "#2a78d6", "hebbian": "#eb6834", "shuffled": "#1baf7a",
             "prompt_only": "#eda100", "oracle": "#4a3aa7"}
ARM_LABEL = {"base": "Baseline", "hebbian": "+Hebbian", "shuffled": "Shuffled bonds",
             "prompt_only": "Bonds in prompt only", "oracle": "Oracle teams"}


def episode_curve(ep: Path, metric: str, rounds: int) -> np.ndarray:
    states = [json.loads(l) for l in (ep / "replay" / "state.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    teams = sorted(states[0]["teams"]) if states else []
    last = {t: 0.0 for t in teams}
    out = np.zeros(rounds)
    by_round = {s["round"]: s for s in states}
    for r in range(1, rounds + 1):
        s = by_round.get(r)
        if s:
            for t, v in s["teams"].items():
                val = float(v.get("solved", 0)) if metric == "solved" else float(v.get("progress") or 0.0)
                last[t] = max(last[t], val)
        out[r - 1] = np.mean(list(last.values())) if last else 0.0
    return out


def collect(roots: List[Path], metric: str, episode: int, rounds: int
            ) -> Dict[Tuple[int, str], List[np.ndarray]]:
    curves: Dict[Tuple[int, str], List[np.ndarray]] = defaultdict(list)
    for root in roots:
        for cfg_path in root.rglob("config.json"):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ep = cfg_path.parent / f"episode_{episode}"
            if not (ep / "replay" / "state.jsonl").is_file():
                continue
            first = json.loads((ep / "replay" / "state.jsonl").read_text(
                encoding="utf-8").splitlines()[0])
            n = len(first["agents"])
            curves[(n, cfg.get("arm", "?"))].append(episode_curve(ep, metric, rounds))
    return curves


def plot(curves, arms: List[str], metric: str, rounds: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({n for n, _ in curves})
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 2.8), sharey=True,
                             squeeze=False)
    x = np.arange(1, rounds + 1)
    for ax, n in zip(axes[0], sizes):
        for arm in arms:
            runs = curves.get((n, arm))
            if not runs:
                continue
            Y = np.vstack(runs)
            m = Y.mean(axis=0)
            se = Y.std(axis=0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros_like(m)
            ax.plot(x, m, color=ARM_COLOR.get(arm, "#52514e"), lw=2,
                    label=f"{ARM_LABEL.get(arm, arm)} (n={len(Y)})")
            ax.fill_between(x, m - se, m + se, color=ARM_COLOR.get(arm, "#52514e"),
                            alpha=0.18, lw=0)
        ax.set_title(f"N = {n}", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("round", color="#52514e")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0][0].set_ylabel("teams solved" if metric == "solved" else "team progress",
                          color="#52514e")
    axes[0][0].set_ylim(0, 1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--metric", choices=("progress", "solved"), default="progress")
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=55)
    ap.add_argument("--arms", default="base,hebbian")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    curves = collect(args.roots, args.metric, args.episode, args.rounds)
    if not curves:
        raise SystemExit("no runs with config.json + episode_k/replay/state.jsonl found")
    out = args.out or args.roots[0] / f"progress_{args.metric}_ep{args.episode}.png"
    plot(curves, [a.strip() for a in args.arms.split(",") if a.strip()], args.metric,
         args.rounds, out)
    print(out)


if __name__ == "__main__":
    main()
