"""CLI for the pair-bonding transplant experiment (Phase A -> Phase B).

Rank the Phase A pair runs, then merge the top three into the Phase B
inputs:

  # 1. Which pair runs bonded best?
  python src/mindforge/tools/merge_pair_runs.py rank runs/pair_bonding/*/seed_*

  # 2. Build the transplant inputs from the top 3 (order = seat order):
  python src/mindforge/tools/merge_pair_runs.py merge \
      --pair-run runs/pair_bonding/expA/seed_42 \
      --pair-run runs/pair_bonding/expA/seed_1011 \
      --pair-run runs/pair_bonding/expA/seed_777 \
      --out-dir merged/transplant

  # 3. Same three runs, shuffled-transplant control (same W blocks, same
  #    memories, strangers seated together):
  python src/mindforge/tools/merge_pair_runs.py merge ... \
      --out-dir merged/shuffled --shuffled --seed 0

The out-dir then feeds multi_agent_craftium.py:
  --hebbian-init-file <out-dir>/merged_W.json
  --agent-state-init  <out-dir>/merged_manifest.json

The merged W is identical between the transplant and shuffled conditions
(same blocks, same seats) — only WHO sits in each seat changes, which is
exactly what makes the shuffled arm a clean control.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python src/mindforge/tools/merge_pair_runs.py` without PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mindforge.tools.pair_transplant import (
        build_merged_manifest,
        build_slot_assignment,
        merge_hebbian_W,
        pair_cofired,
        rank_pair_runs,
        read_cofiring_milestones,
    )
else:
    from .pair_transplant import (
        build_merged_manifest,
        build_slot_assignment,
        merge_hebbian_W,
        pair_cofired,
        rank_pair_runs,
        read_cofiring_milestones,
    )


def _load_pair_W(run_dir: Path):
    with open(run_dir / "hebbian_graph_final.json") as f:
        graph = json.load(f)
    n = graph.get("num_agents")
    if n != 2:
        raise SystemExit(
            f"{run_dir}: expected a 2-agent pair run, but "
            f"hebbian_graph_final.json has num_agents={n}"
        )
    return graph["W"]


def _load_agent_state(run_dir: Path, agent_idx: int) -> dict:
    """Run-end export preferred; latest checkpoint export as fallback."""
    primary = run_dir / "agent_state" / f"agent_{agent_idx}.json"
    if primary.exists():
        path = primary
    else:
        candidates = sorted(
            run_dir.glob(f"checkpoints/*/agent_state/agent_{agent_idx}.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit(
                f"{run_dir}: no agent_state/agent_{agent_idx}.json at run "
                f"level or in any checkpoint — did the run predate the "
                f"agent-state export?"
            )
        path = candidates[-1]
        print(f"  [fallback] {run_dir.name}/agent_{agent_idx}: using {path}")
    with open(path) as f:
        return json.load(f)


def cmd_rank(args):
    rows = rank_pair_runs(args.run_dirs)

    def _n(v):
        return "?" if v is None else str(v)

    print(f"{'rank':<5} {'bond':>7} {'W01':>7} {'W10':>7} "
          f"{'jointdig':>9} {'coaction':>9} {'COFIRED':>8}  run_dir")
    for i, row in enumerate(rows):
        if row["error"]:
            print(f"{i:<5} {'--':>7} {'--':>7} {'--':>7} "
                  f"{'--':>9} {'--':>9} {'--':>8}  "
                  f"{row['run_dir']}  [{row['error']}]")
        else:
            mark = "YES" if row.get("cofired") else "no"
            print(f"{i:<5} {row['bond']:>7.4f} {row['w01']:>7.4f} "
                  f"{row['w10']:>7.4f} "
                  f"{_n(row['joint_dig']):>9} {_n(row['co_action']):>9} "
                  f"{mark:>8}  {row['run_dir']}"
                  + (f"  {row['cofiring_milestones']}"
                     if row.get("cofiring_milestones") else ""))

    ok = [r for r in rows if r["error"] is None]
    # Ranking is on bond, but bond is a confounded proxy for co-firing (see
    # rank_pair_runs docstring). Say so loudly when the two disagree, rather
    # than letting the top-3 line imply the behavioural evidence agrees.
    if len(ok) >= 2 and all(r["joint_dig"] is not None for r in ok):
        by_dig = sorted(ok, key=lambda r: -r["joint_dig"])
        if by_dig[0]["run_dir"] != ok[0]["run_dir"]:
            print(f"\n  !! bond and behaviour DISAGREE: top bond is "
                  f"{Path(ok[0]['run_dir']).name} "
                  f"(joint_dig={ok[0]['joint_dig']}), but most joint digs is "
                  f"{Path(by_dig[0]['run_dir']).name} "
                  f"(joint_dig={by_dig[0]['joint_dig']}).")
            print("     W includes an always-on communication term, so it "
                  "tracks proximity+chatter as well as real co-firing.")
            print("     Decide the selection rule explicitly before merging.")

    if len(ok) >= 3:
        print("\nTop 3 by bond (pass to `merge` in this order):")
        for r in ok[:3]:
            print(f"  --pair-run {r['run_dir']}")


def cmd_merge(args):
    run_dirs = [Path(d) for d in args.pair_run]
    if len(run_dirs) < 2:
        raise SystemExit("merge needs at least 2 --pair-run dirs (3 for the "
                         "standard 6-agent experiment)")

    pair_Ws = [_load_pair_W(d) for d in run_dirs]
    pair_states = [
        {0: _load_agent_state(d, 0), 1: _load_agent_state(d, 1)}
        for d in run_dirs
    ]

    condition = "shuffled" if args.shuffled else "transplant"
    assignment = build_slot_assignment(
        n_pairs=len(run_dirs), shuffled=args.shuffled, seed=args.seed
    )
    W, provenance = merge_hebbian_W(
        pair_Ws, cross_weight=args.cross_weight, normalize=args.normalize
    )
    # Ground-truth co-firing per source pair, so Phase B analysis can tell a
    # genuine dyad from a control dyad. Derived from the run's own
    # milestones, never hand-entered.
    pair_meta = {
        k: {
            "run_dir": str(d),
            "cofired": pair_cofired(d),
            "cofiring_milestones": read_cofiring_milestones(d),
        }
        for k, d in enumerate(run_dirs)
    }
    manifest = build_merged_manifest(pair_states, assignment, condition,
                                     pair_meta=pair_meta)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_payload = {
        "num_agents": W.shape[0],
        "W": W.tolist(),
        "condition": condition,
        "seed": args.seed if args.shuffled else None,
        "assignment": [list(a) for a in assignment],
        "sources": [str(d) for d in run_dirs],
        "source_cofired": {str(k): v["cofired"] for k, v in pair_meta.items()},
        "seat_pairs": manifest["seat_pairs"],
        **provenance,
    }
    with open(out_dir / "merged_W.json", "w") as f:
        json.dump(w_payload, f, indent=2)
    manifest["sources"] = [str(d) for d in run_dirs]
    with open(out_dir / "merged_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{condition}] merged {len(run_dirs)} pair runs -> {out_dir}")
    print(f"  seats: " + ", ".join(
        f"seat{s}<-run{r}/agent_{a}" for s, (r, a) in enumerate(assignment)
    ))
    print("  seat pairs:")
    for sp in manifest["seat_pairs"]:
        if sp["same_source_run"]:
            tag = "GENUINE (co-fired)" if sp["cofired"] else "CONTROL (never co-fired)"
            src = Path(sp["source_run"]).name if sp["source_run"] else "?"
            print(f"    seats {sp['seats']}: {tag} <- {src}")
        else:
            print(f"    seats {sp['seats']}: strangers (shuffled)")
    print(f"  block means {provenance['block_means']} -> rescale "
          f"{provenance['rescale_factors']} (normalize={args.normalize})")
    for row in W:
        print("  " + " ".join(f"{w:.3f}" for w in row))
    print(f"\nPhase B flags:\n"
          f"  --hebbian-init-file {out_dir / 'merged_W.json'}\n"
          f"  --agent-state-init  {out_dir / 'merged_manifest.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # No required=True: the DAIC login node runs Python 3.6, where
    # add_subparsers() does not accept it. Enforced manually below.
    sub = parser.add_subparsers(dest="command")

    p_rank = sub.add_parser(
        "rank", help="rank pair runs by final within-pair bond strength"
    )
    p_rank.add_argument("run_dirs", nargs="+",
                        help="pair-run directories (shell glob friendly)")
    p_rank.set_defaults(func=cmd_rank)

    p_merge = sub.add_parser(
        "merge", help="build merged_W.json + merged_manifest.json"
    )
    p_merge.add_argument("--pair-run", action="append", required=True,
                         help="pair-run dir; repeat 3x, order = seat order")
    p_merge.add_argument("--out-dir", required=True)
    p_merge.add_argument("--shuffled", action="store_true",
                         help="shuffled-transplant control: same W blocks and "
                              "memories, strangers seated together")
    p_merge.add_argument("--seed", type=int, default=0,
                         help="shuffle seed (only with --shuffled)")
    p_merge.add_argument("--cross-weight", type=float, default=0.1,
                         help="cross-pair W entries (must be > 0; default = "
                              "hebbian init_weight 0.1)")
    p_merge.add_argument("--normalize", choices=["block_mean", "none"],
                         default="block_mean",
                         help="block_mean (default): rescale each pair block "
                              "to a common mean so no pair starts advantaged")
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
