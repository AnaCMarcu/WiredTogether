#!/usr/bin/env python
"""Group the collected runs by research question, and package them for release.

The cluster wrote one directory per submission wave (`medium_runs`, `cofiring_final`,
`pareto_social`, ...). This files those groups under the question they answer and
builds the published dataset from the result.

Usage:
  python analysis/runs_dataset.py plan                  # what would move, and what it weighs
  python analysis/runs_dataset.py regroup --apply       # file the groups under rq*/ dirs
  python analysis/runs_dataset.py bundle                # build dist/runs_dataset/
  python analysis/runs_dataset.py verify dist/runs_dataset

Layers, so nobody downloads more than they need — they stack, and none repeats
another's files:

  core   metrics, events, per-episode tables, graph snapshots, configs.
         Everything the paper's tables and quantitative figures read.
  logs   llm_logs/*.log and log.txt — the qualitative pipeline and, for the RL
         arms, its step-clock alignment.
  media  the .mp4 recordings, read only by the story-timeline figures.

RL checkpoints are never bundled: no script under analysis/ reads them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from paths import REPO, RUNS  # noqa: E402  (also puts siblings on sys.path)

# ─── Which question each run group answers ──────────────────────────────
# group dir -> (destination, what it holds). Group names are unique across the
# dataset, so analysis scripts keep asking for them by name (paths.group()) and
# never encode the grouping.
LAYOUT: dict[str, tuple[str, str]] = {
    # RQ1 — does reward-modulated Hebbian plasticity produce social intelligence?
    "medium_runs":       ("rq1_social_plasticity", "Qwen 2B/9B, MAPPO/IPPO, ±Hebbian, ±social module (exp01–exp11; exp09–11 are also the RQ3 topology arms)"),
    "new_exp_0_gemma":   ("rq1_social_plasticity", "Gemma-E4B base / +Hebbian / three-factor anchors"),
    "gemma4":            ("rq1_social_plasticity", "Gemma-E4B MAPPO and IPPO arms"),
    "orchestrator":      ("rq1_social_plasticity", "Centralised orchestration baseline (VillagerAgent, advisory)"),
    # RQ2 — which social acts should the wiring rule credit?
    "cofiring_final":    ("rq2_cofiring", "The seven co-firing channel arms reported in the paper"),
    "cofiring":          ("rq2_cofiring", "Earlier full channel sweep"),
    "cofiring_bidi":     ("rq2_cofiring", "Bidirectional-credit control"),
    "cofiring_noreward": ("rq2_cofiring", "Control with reward modulation switched off"),
    # RQ3 — does an imposed or transplanted topology transfer?
    "medium_2k":         ("rq3_topology_transfer", "Imposed topology (allied-all / pair / none) at 2k steps"),
    "pair_bonding":      ("rq3_topology_transfer", "Phase-A dyads and the Phase-B transplant"),
    # Compute — what the social layer costs.
    "pareto_social":     ("compute", "Deliberation-interval sweep"),
    "pareto_gemma4":     ("compute", "Model-size sweep"),
    "agent_scaling":     ("compute", "Team-size sweep, N ∈ {2..9}"),
    # Experience sharing (Eq. 7), opt-in arms.
    "social_replay_gemma4":       ("social_replay", "Weight-gated experience sharing, Gemma lane"),
    "social_replay_qwen":  ("social_replay", "Weight-gated experience sharing, Qwen lane. Ran with LLM_VISION_MODE=text while exp05/exp06, the Qwen arms it is tabled against, ran with vision"),
    # Cluster stdout/stderr, kept as provenance.
    "new_exp_0_gemma_slurm": ("cluster_logs", "SLURM stdout/stderr"),
    "social_replay_slurm":   ("cluster_logs", "SLURM stdout/stderr"),
    "pareto_probe":          ("cluster_logs", "NaN-logits probe and image-build logs"),
    "gemma4_slurm":          ("cluster_logs", "SLURM stdout/stderr (empty)"),
    # Smoke tests: kept locally, excluded from the release unless asked for.
    "agent_scaling_smoke":            ("smoke", "Smoke test"),
    "agent_scaling_smoke2":           ("smoke", "Smoke test"),
    "cofiring_smoke":                 ("smoke", "Smoke test"),
    "cofiring_smoke_v1":              ("smoke", "Smoke test"),
    "gemma4_smoke":                   ("smoke", "Smoke test"),
    "orchestrator_smoke":             ("smoke", "Smoke test"),
    "orchestrator_smoke_prefix":      ("smoke", "Smoke test"),
    "orchestrator_smoke_v3":          ("smoke", "Smoke test"),
    "orchestrator_smoke_v4":          ("smoke", "Smoke test"),
    "orchestrator_smoke_variants":    ("smoke", "Smoke test"),
    "orchestrator_smoke_villager_v2": ("smoke", "Smoke test"),
    "pareto_gemma4_smoke":            ("smoke", "Smoke test"),
    "social_replay_gemma4_smoke":     ("smoke", "Smoke test"),
    "social_replay_qwen_smoke":       ("smoke", "Smoke test"),
}

RQ_TITLE = {
    "rq1_social_plasticity": "RQ1 — does reward-modulated Hebbian plasticity produce social intelligence?",
    "rq2_cofiring":          "RQ2 — which social acts should the wiring rule credit?",
    "rq3_topology_transfer": "RQ3 — does an imposed or transplanted topology transfer?",
    "compute":               "Compute — what the social layer costs",
    "social_replay":         "Weight-gated experience sharing (Eq. 7)",
    "cluster_logs":          "Cluster stdout/stderr (provenance)",
    "smoke":                 "Smoke tests (not part of the release)",
}
DESTINATIONS = list(RQ_TITLE)
RELEASED = [d for d in DESTINATIONS if d != "smoke"]

# ─── Layers ─────────────────────────────────────────────────────────────
CHECKPOINT_DIRS = {"checkpoints"}
MEDIA_DIRS = {"gifs"}
LOG_DIRS = {"llm_logs"}
LOG_FILES = {"log.txt", "run.log"}
LOG_SUFFIXES = {".out", ".err"}          # SLURM stdout/stderr
LAYERS = ("core", "logs", "media")


def layer_of(rel: Path) -> str | None:
    """Which layer a run-relative file belongs to; None = never bundled."""
    parts = set(rel.parts[:-1])
    if parts & CHECKPOINT_DIRS:
        return None
    if parts & MEDIA_DIRS:
        return "media"
    if parts & LOG_DIRS or rel.name in LOG_FILES or rel.suffix in LOG_SUFFIXES:
        return "logs"
    return "core"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


# ─── Where a group currently lives ──────────────────────────────────────
def locate(runs: Path) -> dict[str, Path]:
    """group name -> its directory, whether or not the tree has been regrouped."""
    found: dict[str, Path] = {}
    for p in sorted(runs.glob("*")):
        if not p.is_dir():
            continue
        if p.name in DESTINATIONS:
            for q in sorted(p.glob("*")):
                if q.is_dir():
                    found[q.name] = q
        else:
            found[p.name] = p
    return found


def scan(d: Path) -> dict[str, tuple[int, int]]:
    """layer -> (bytes, files) under one group directory."""
    acc = {k: [0, 0] for k in LAYERS}
    for root, _, files in os.walk(d):
        rel_root = Path(root).relative_to(d)
        for f in files:
            lay = layer_of(rel_root / f)
            if lay is None:
                continue
            try:
                acc[lay][0] += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            acc[lay][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


# ─── plan / regroup ─────────────────────────────────────────────────────
def cmd_plan(args) -> int:
    found = locate(args.runs)
    unknown = sorted(set(found) - set(LAYOUT))
    for dest in DESTINATIONS:
        groups = [g for g in LAYOUT if LAYOUT[g][0] == dest and g in found]
        if not groups:
            continue
        print(f"\n{RQ_TITLE[dest]}")
        for g in groups:
            here = found[g]
            settled = here.parent.name == dest
            sizes = scan(here)
            tot = sum(s for s, _ in sizes.values())
            print(f"  {'ok ' if settled else '-> '}{g:32s} {human(tot):>9s}"
                  f"  (core {human(sizes['core'][0])}, logs {human(sizes['logs'][0])},"
                  f" media {human(sizes['media'][0])})")
    if unknown:
        print(f"\nNot in LAYOUT (left where they are): {', '.join(unknown)}")
    return 0


def cmd_regroup(args) -> int:
    found = locate(args.runs)
    moves = []
    for g, here in found.items():
        if g not in LAYOUT:
            continue
        dest = args.runs / LAYOUT[g][0]
        if here.parent == dest:
            continue
        moves.append((here, dest / g))
    if not moves:
        print("Already grouped; nothing to move.")
        return 0
    for src, dst in moves:
        print(f"{src.relative_to(args.runs)}  ->  {dst.relative_to(args.runs)}")
    if not args.apply:
        print(f"\n{len(moves)} groups would move. Re-run with --apply.")
        return 0
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"  [skip] {dst} exists", file=sys.stderr)
            continue
        os.rename(src, dst)          # same volume: metadata-only, no copy
    print(f"\nMoved {len(moves)} groups.")
    return 0


# ─── bundle ─────────────────────────────────────────────────────────────
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cmd_bundle(args) -> int:
    found = locate(args.runs)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dests = DESTINATIONS if args.include_smoke else RELEASED
    layers = args.layers or list(LAYERS)

    manifest: dict = {
        "dataset": "Wired Together — WIRE run artifacts",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layers": {
            "core":  "Metrics, events, per-episode tables, graph snapshots, configs. Reproduces every table and quantitative figure.",
            "logs":  "llm_logs/*.log and log.txt. Needed by analysis/qualitative/ and, for the RL arms, its step-clock alignment.",
            "media": "The .mp4 recordings, read only by the story-timeline figures.",
        },
        "excluded": "RL checkpoints (checkpoints/): no script under analysis/ reads them."
                    + ("" if args.include_smoke else " Smoke-test groups."),
        "questions": {},
        "archives": [],
    }

    for dest in dests:
        groups = [g for g in LAYOUT if LAYOUT[g][0] == dest and g in found]
        if not groups:
            continue
        manifest["questions"][dest] = {
            "title": RQ_TITLE[dest],
            "groups": {g: {"description": LAYOUT[g][1],
                           "arms": sorted(p.name for p in found[g].glob("*") if p.is_dir())}
                       for g in groups},
        }
        for layer in layers:
            members: list[tuple[Path, str]] = []
            for g in groups:
                base = found[g]
                for root, _, files in os.walk(base):
                    rel_root = Path(root).relative_to(base)
                    for f in files:
                        rel = rel_root / f
                        if layer_of(rel) != layer:
                            continue
                        members.append((Path(root) / f, str(Path(dest) / g / rel).replace("\\", "/")))
            if not members:
                continue
            tar_path = out / f"{dest}__{layer}.tar.gz"
            print(f"  {tar_path.name}: {len(members)} files ...", end="", flush=True)
            with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
                for src, arc in members:
                    try:
                        tf.add(src, arcname=arc, recursive=False)
                    except OSError as e:
                        print(f"\n    [warn] unreadable {src}: {e}", file=sys.stderr)
            size = tar_path.stat().st_size
            raw = sum(s.stat().st_size for s, _ in members if s.exists())
            print(f" {human(raw)} -> {human(size)}")
            manifest["archives"].append({
                "file": tar_path.name, "question": dest, "layer": layer,
                "files": len(members), "raw_bytes": raw, "bytes": size,
                "sha256": sha256(tar_path),
            })

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "SHA256SUMS").write_text(
        "".join(f"{a['sha256']}  {a['file']}\n" for a in manifest["archives"]), encoding="utf-8")
    total = sum(a["bytes"] for a in manifest["archives"])
    print(f"\n{len(manifest['archives'])} archives, {human(total)} total -> {out}")
    print("Extract into runs_from_daic/ to reproduce; MANIFEST.json lists what each holds.")
    return 0


def cmd_verify(args) -> int:
    d = args.bundle
    sums = (d / "SHA256SUMS")
    if not sums.exists():
        sys.exit(f"no SHA256SUMS in {d}")
    bad = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        want, name = line.split("  ", 1)
        p = d / name
        if not p.exists():
            print(f"MISSING  {name}"); bad += 1; continue
        got = sha256(p)
        print(f"{'ok      ' if got == want else 'MISMATCH'} {name}")
        bad += got != want
    print("\nall archives verified" if not bad else f"\n{bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS, help="run root (default: runs_from_daic/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("plan", help="show the grouping and what each layer weighs").set_defaults(fn=cmd_plan)
    r = sub.add_parser("regroup", help="file run groups under their research question")
    r.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    r.set_defaults(fn=cmd_regroup)
    b = sub.add_parser("bundle", help="build the release archives")
    b.add_argument("--out", type=Path, default=REPO / "dist" / "runs_dataset")
    b.add_argument("--layers", nargs="*", choices=LAYERS, help="default: all three")
    b.add_argument("--include-smoke", action="store_true")
    b.set_defaults(fn=cmd_bundle)
    v = sub.add_parser("verify", help="check archives against SHA256SUMS")
    v.add_argument("bundle", type=Path)
    v.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
