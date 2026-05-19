#!/usr/bin/env python3
"""run_experiments.py — declarative, tiered, resumable experiment launcher.

Reads scripts/experiments.yaml and launches EPyMARL runs with a concurrency
cap. Resumable: a run that already has a successful entry in
results/runs.jsonl is skipped unless --force is passed. Tiered: pass
--tier 1 to run only the headline-claim variants.

Usage:
    python scripts/run_experiments.py                       # full plan
    python scripts/run_experiments.py --tier 1              # headline only
    python scripts/run_experiments.py --smoke               # 50k steps, 1 seed
    python scripts/run_experiments.py --dry-run             # show plan, don't run
    python scripts/run_experiments.py --parallel 2          # cap concurrent runs
    python scripts/run_experiments.py --force               # ignore prior completions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EPYMARL_ROOT = REPO_ROOT / "epymarl"
# Results dir is overridable via HEBBIAN_RESULTS_DIR for HPC (where the
# repo lives on /home but training output should land on /scratch). When
# unset, defaults to <repo-root>/results — the local-development layout.
RESULTS_DIR = Path(os.environ.get("HEBBIAN_RESULTS_DIR", str(REPO_ROOT / "results")))
RUNS_LOG = RESULTS_DIR / "runs.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "experiments.yaml"


def find_python() -> str:
    for c in (
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
    ):
        if c.is_file():
            return str(c)
    return sys.executable


@dataclass
class RunSpec:
    variant: str
    seed: int
    tier: int
    rationale: str
    env_config: str
    env_key: str
    time_limit: int
    t_max: int
    use_cuda: bool
    label_override: Optional[str] = None

    @property
    def label(self) -> str:
        if self.label_override:
            return self.label_override
        return f"{self.variant}_seed{self.seed}"

    @property
    def log_path(self) -> Path:
        return RESULTS_DIR / "logs" / f"{self.label}.log"


def load_manifest(path: Path, tier: Optional[int], smoke: bool) -> List[RunSpec]:
    data = yaml.safe_load(path.read_text())
    defaults = data.get("defaults", {}) or {}
    specs: List[RunSpec] = []
    for exp in data["experiments"]:
        if tier is not None and exp.get("tier") != tier:
            continue
        seeds = exp.get("seeds", defaults.get("seeds", [0]))
        if smoke:
            seeds = seeds[:1]
        for s in seeds:
            specs.append(RunSpec(
                variant=exp["variant"],
                seed=int(s),
                tier=int(exp.get("tier", 99)),
                rationale=str(exp.get("rationale", "")),
                env_config=exp.get("env_config", defaults.get("env_config", "lbf_comm")),
                env_key=exp.get("env_key", defaults.get("env_key")),
                time_limit=int(exp.get("time_limit", defaults.get("time_limit", 50))),
                t_max=50_000 if smoke else int(exp.get("t_max", defaults.get("t_max", 5_000_000))),
                use_cuda=bool(exp.get("use_cuda", defaults.get("use_cuda", False))),
                label_override=exp.get("label"),
            ))
    specs.sort(key=lambda r: (r.tier, r.variant, r.seed))
    return specs


def already_completed(spec: RunSpec) -> bool:
    if not RUNS_LOG.is_file():
        return False
    with RUNS_LOG.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("label") == spec.label
                and rec.get("exit_code") == 0
                and rec.get("t_max") == spec.t_max
                and rec.get("env_key") == spec.env_key
            ):
                return True
    return False


def append_record(rec: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_one(spec: RunSpec, py: str) -> dict:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        py, "src/main.py",
        f"--config={spec.variant}",
        f"--env-config={spec.env_config}",
        "with",
        f"env_args.key={spec.env_key}",
        f"env_args.time_limit={spec.time_limit}",
        f"seed={spec.seed}",
        f"t_max={spec.t_max}",
        f"use_cuda={'True' if spec.use_cuda else 'False'}",
        "use_tensorboard=True",
        f"name={spec.label}",
    ]
    # Redirect EPyMARL's results tree to repo-root results/ so sacred,
    # bonds, tb_logs, and launcher logs all live under one directory.
    env = os.environ.copy()
    env["HEBBIAN_RESULTS_DIR"] = str(RESULTS_DIR)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    with spec.log_path.open("w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n# started: {started}\n\n")
        f.flush()
        proc = subprocess.run(
            cmd, cwd=str(EPYMARL_ROOT), stdout=f, stderr=subprocess.STDOUT, env=env,
        )
    rec = {
        "label": spec.label,
        "variant": spec.variant,
        "seed": spec.seed,
        "tier": spec.tier,
        "env_key": spec.env_key,
        "t_max": spec.t_max,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 1),
        "exit_code": proc.returncode,
        "log_path": str(spec.log_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "rationale": spec.rationale,
    }
    append_record(rec)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--tier", type=int, default=None,
                   help="run only this tier (1=headline, 2=components, 3=context)")
    p.add_argument("--smoke", action="store_true",
                   help="50k steps and 1 seed per variant — for plumbing checks")
    p.add_argument("--parallel", type=int, default=2,
                   help="max concurrent runs (default 2; raise if you have cores+RAM)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="run even if already completed")
    p.add_argument("--list", action="store_true",
                   help="print indexed list of planned runs (after --tier filter, "
                        "BEFORE the already-completed filter) and exit. Use this "
                        "to determine SLURM --array bounds.")
    p.add_argument("--single-index", type=int, default=None,
                   help="run only the Nth spec from the planned list (same ordering "
                        "as --list). For SLURM array jobs: set --array=0-(N-1) and "
                        "pass --single-index $SLURM_ARRAY_TASK_ID. Resumability "
                        "still applies — already-completed specs are skipped unless "
                        "--force is also passed.")
    args = p.parse_args()

    py = find_python()
    specs = load_manifest(Path(args.manifest), args.tier, args.smoke)

    # --list prints the full plan ordering (resumability ignored) so the
    # SLURM array bounds are stable across re-submissions.
    if args.list:
        print(f"# {len(specs)} planned runs (manifest order, resumability NOT applied)")
        for idx, s in enumerate(specs):
            print(f"{idx:3d}  tier={s.tier}  {s.label:35s}  t_max={s.t_max:>9}  env={s.env_key}")
        return 0

    # --single-index picks one spec from the planned list (pre-resumability),
    # then re-applies resumability so a re-queued SLURM task with already-
    # completed data is a fast no-op.
    if args.single_index is not None:
        if not (0 <= args.single_index < len(specs)):
            print(f"--single-index {args.single_index} out of range "
                  f"(planned: {len(specs)})", file=sys.stderr)
            return 2
        specs = [specs[args.single_index]]

    queued: List[RunSpec] = []
    for s in specs:
        if not args.force and already_completed(s):
            print(f"[skip] {s.label} (already completed)")
            continue
        queued.append(s)

    print(f"\n[plan] {len(queued)} runs (parallel cap: {args.parallel}, python: {py})")
    for s in queued:
        print(f"  tier {s.tier}  {s.label:35s}  t_max={s.t_max:>9}  env={s.env_key}")

    if args.dry_run or not queued:
        return 0

    done = 0
    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, s, py): s for s in queued}
        for fut in as_completed(futures):
            rec = fut.result()
            done += 1
            tag = "OK" if rec["exit_code"] == 0 else f"FAIL({rec['exit_code']})"
            print(f"[{done}/{len(queued)}] {tag:9s} {rec['label']:35s} "
                  f"{rec['wall_seconds']}s  -> {rec['log_path']}")
            if rec["exit_code"] != 0:
                failed.append(rec["label"])

    print(f"\n[done] {done - len(failed)}/{done} succeeded; {len(failed)} failed")
    if failed:
        print("[failed]", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
