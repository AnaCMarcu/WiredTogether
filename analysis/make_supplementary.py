#!/usr/bin/env python
"""Assemble the supplementary bundle that ships with the paper.

One zip a reviewer can open without a cluster account: the paper, every figure
and results table, the video clips the qualitative analysis is built on, the
uncut episode recordings behind those clips, and the docs that say what the
environment and the run artifacts are.

What it is *not* is the dataset: the per-run artifacts are 46 GB and are
released as layered archives by `analysis/runs_dataset.py`. This bundle points
at those and stays small enough to attach to a submission.

Layout:

  paper/          the submitted paper, anonymised
  figures/        every figure in the paper (pdf + 300 dpi png)
  tables/         every generated table and summary CSV, by experiment family
  videos/clips/   the curated clips and the reviewed candidates (+ posters, manifests)
  videos/episodes/  the uncut recordings of the four runs those clips come from
  docs/           environment, agents, architecture, dataset, experiments
  README.md, MANIFEST.json, SHA256SUMS

Usage:
  python analysis/make_supplementary.py                 # build + zip
  python analysis/make_supplementary.py --no-episodes    # skip the 140 MB layer
  python analysis/make_supplementary.py --no-zip         # leave the tree only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from paths import ASSETS, REPO  # noqa: E402  (also puts siblings on sys.path)

OUT = REPO / "dist" / "supplementary"
ZIP = REPO / "dist" / "wired_together_supplementary.zip"
SITE_VIDEOS = REPO / "site" / "assets" / "videos"
PAPER_PDF = REPO / "site" / "assets" / "paper.pdf"

TEXT_SUFFIXES = (".tex", ".csv", ".md", ".txt", ".json")
FIGURE_SUFFIXES = (".pdf", ".png")

DOCS = ("README.md", "environment.md", "agents.md", "architecture.md",
        "configuration.md", "dataset.md", "experiments.md", "hebbian-graph.md",
        "rl-layer.md", "experiment_checklist.md")


def copy_into(src: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    return dest.stat().st_size


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


# ── sections ────────────────────────────────────────────────────────────
def add_paper(out: Path) -> int:
    if not PAPER_PDF.exists():
        print("  paper       — skipped, no compiled PDF at "
              f"{PAPER_PDF.relative_to(REPO)}", file=sys.stderr)
        return 0
    return copy_into(PAPER_PDF, out / "paper" / "wired_together.pdf")


def add_figures(out: Path) -> int:
    """The release figure set, plus the wide qualitative timelines."""
    total = 0
    for src in sorted((REPO / "figures").rglob("*")):
        if src.is_file() and (src.suffix in FIGURE_SUFFIXES or src.name == "README.md"):
            total += copy_into(src, out / "figures" / src.name)
    wides = ASSETS / "timelines" / "for_paper"
    for src in sorted(wides.glob("*")):
        if src.suffix in FIGURE_SUFFIXES:
            total += copy_into(src, out / "figures" / "qualitative" / src.name)
    return total


def add_tables(out: Path) -> int:
    """Every generated table and summary, keeping the experiment-family dirs."""
    total = 0
    for src in sorted(ASSETS.rglob("*")):
        if not src.is_file() or src.suffix not in TEXT_SUFFIXES:
            continue
        rel = src.relative_to(ASSETS)
        if rel.parts[0] == "timelines" and src.suffix != ".md":
            continue  # timeline dirs hold per-figure scratch, not results
        total += copy_into(src, out / "tables" / rel)
    return total


def add_clips(out: Path) -> int:
    if not SITE_VIDEOS.exists():
        print("  clips       — skipped, run analysis/make_media_clips.py first",
              file=sys.stderr)
        return 0
    total = 0
    for src in sorted(SITE_VIDEOS.glob("*")):
        if src.is_file() and src.stem != "hero":   # the page's montage, not a result
            total += copy_into(src, out / "videos" / "clips" / src.name)
    keep = SITE_VIDEOS / "candidates" / "keep.txt"
    kept = None
    if keep.exists():
        kept = {l.strip() for l in keep.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")}
    for src in sorted((SITE_VIDEOS / "candidates").glob("*")):
        if not src.is_file():
            continue
        if kept is not None and src.suffix in (".mp4", ".jpg") and src.stem not in kept:
            continue
        total += copy_into(src, out / "videos" / "clips" / "candidates" / src.name)
    return total


def add_episodes(out: Path) -> int:
    import make_media_clips as clips  # noqa: E402 — needs paths on sys.path

    # Only the scene runs ship their recordings; the other registered arms
    # (candidate sources) are not required to be present.
    missing = [k for k in clips.SCENE_RUNS if not clips.RUNS[k]["run"].is_dir()]
    if missing:
        print(f"  episodes    — skipped, run directories missing: {', '.join(missing)}",
              file=sys.stderr)
        return 0
    return clips.copy_episodes(out / "videos" / "episodes")


def add_docs(out: Path) -> int:
    total = 0
    for name in DOCS:
        src = REPO / "docs" / name
        if src.exists():
            total += copy_into(src, out / "docs" / name)
    for name in ("README.md",):
        src = REPO / "analysis" / name
        if src.exists():
            total += copy_into(src, out / "docs" / "analysis_README.md")
    return total


SECTIONS = {
    "paper": add_paper,
    "figures": add_figures,
    "tables": add_tables,
    "clips": add_clips,
    "episodes": add_episodes,
    "docs": add_docs,
}


# ── bundle metadata ─────────────────────────────────────────────────────
README = """\
# Wired Together — supplementary material

*Learning Persistent Relationships through Reward-Modulated Social Plasticity.*
Anonymous authors; paper under double-blind review.

Everything here is derived from the run artifacts by the scripts in `analysis/`
of the code repository, which is released on acceptance. Nothing was edited by
hand.

## Contents

| Path | What it is |
|---|---|
| `paper/` | The submitted paper, anonymised. |
| `figures/` | Every figure in the paper, as vector PDF and 300 dpi PNG. `figures/README.md` maps each one to where it appears and which script builds it. `figures/qualitative/` holds the full-width coordination timelines. |
| `tables/` | Every generated table and summary, in the experiment-family directories the analysis scripts write: LaTeX rows as included by the paper, plus the CSV and Markdown the rows are rendered from. `tables/final_ext/final_table_extended.md` is the full results table; the paper's main table is the subset in `tables/main_ext/`. |
| `videos/clips/` | The clips on the project page, cut to the exact step windows the qualitative figures annotate, plus `candidates/`: the wider set of successes and failures mined from every episode's milestone and message logs. Two agents side by side. Each directory's `manifest.json` records the run, seed, episode, agents and step window of every clip. |
| `videos/episodes/` | The uncut recordings those clips come from: the four runs the qualitative analysis uses, all agents, all episodes. One rendered frame per environment step at 2 fps — so a frame index *is* a step index. |
| `docs/` | The environment, the agent architecture, the Hebbian graph, the RL layer, the configuration surface, and what each run directory contains. |

## Reading the recordings

Every `.mp4` is a first-person agent view at 320×180, one frame per environment
step, written at 2 fps. Step *t* of an episode is therefore second *t*/2 of that
episode's recording, which is how the clips were cut and how any other moment in
the paper can be found. Agent views are unedited: the name tags, the hotbar and
the health bar are the environment's own HUD.

## The rest of the data

The per-run artifacts are 46 GB and are not in this bundle. They are released as
layered archives — `core` (metrics, events, per-episode tables, graph snapshots,
configs), `logs` (per-agent LLM logs and step clocks) and `media` (all 2,523
recordings) — built and verified by `analysis/runs_dataset.py`. Extract them into
`runs_from_daic/` in the code repository and every table and figure here
regenerates. `docs/dataset.md` documents the layout, the layers and the one
excluded run.

## Provenance

`MANIFEST.json` lists what each section holds and when it was built.
`SHA256SUMS` covers every file in the bundle:

    sha256sum -c SHA256SUMS
"""


def write_manifest(out: Path, sizes: dict[str, int], commit: str | None) -> None:
    files = sorted(p for p in out.rglob("*") if p.is_file()
                   and p.name not in ("MANIFEST.json", "SHA256SUMS"))
    manifest = {
        "bundle": "wired_together_supplementary",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": commit,
        "sections": {k: {"bytes": v, "size": human(v)} for k, v in sizes.items() if v},
        "files": len(files),
        "bytes": sum(sizes.values()),
        "dataset": {
            "note": "Per-run artifacts are released separately; see docs/dataset.md.",
            "builder": "analysis/runs_dataset.py",
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")

    lines = []
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is nice to have, not required
        return None


def make_zip(out: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            # Already-compressed media gains nothing and costs minutes.
            method = zipfile.ZIP_STORED if p.suffix in (".mp4", ".jpg", ".png") \
                else zipfile.ZIP_DEFLATED
            z.write(p, Path("wired_together_supplementary") / p.relative_to(out),
                    compress_type=method)
    return zip_path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--zip", dest="zip_path", type=Path, default=ZIP)
    ap.add_argument("--no-episodes", action="store_true",
                    help="leave out the uncut recordings (~140 MB)")
    ap.add_argument("--no-zip", action="store_true", help="build the tree only")
    ap.add_argument("--clean", action="store_true", help="remove the tree first")
    args = ap.parse_args()

    if args.clean and args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    sizes = {}
    for name, fn in SECTIONS.items():
        if name == "episodes" and args.no_episodes:
            continue
        size = fn(args.outdir)
        sizes[name] = size
        if size:
            print(f"  {name:<11} {human(size)}")

    (args.outdir / "README.md").write_text(README, encoding="utf-8")
    write_manifest(args.outdir, sizes, git_commit())
    print(f"\ntree   {args.outdir.relative_to(REPO)}  {human(sum(sizes.values()))}")

    if not args.no_zip:
        size = make_zip(args.outdir, args.zip_path)
        print(f"zip    {args.zip_path.relative_to(REPO)}  {human(size)}")


if __name__ == "__main__":
    main()
