"""Run a chain of AgentWorld tasks with MindForge agents (our harness, src/agentworld/).

    python chains/run_mf_chain.py chains/pendant_staff_4.yaml --arm hebbian --seed 42 \\
        --server-url http://127.0.0.1:7031 [--restart-file FILE] \\
        --hebbian-decay 0.005 --hold-complete

The same chain YAML, task generation (renamed usernames, direct-message wording,
verifiers copied byte for byte) and Hebbian rule as ``run_chain.py``, but the
agents are the MindForge cognitive stack of ``agentworld.agent`` and the round
loop is ``agentworld.scheduler``'s (see ``mf_chain.py`` for what carries over and
which protocol rules are matched). One process plays the whole chain, so the
agents' skill and episode memory carry over in memory.

Arms (``--arm``):
    base      MindForge; bonds are computed (observer only) but not shown, reading by recency
    hebbian   MindForge + social plasticity: bonds in the prompt and bond-ranked reading

``--plumbing`` replaces the model with a fixed script (no LLM, no MindForge
imports) to test the server, task generation, verifiers and outputs.

Output, ``<output>/<chain>/<run_id>/``::

    chain_summary.json             per step: success, verifier message, rounds, actions,
                                   held completes, caps, messages, bonds; totals
    tasks/<chain>/step_NN_task_MM/ the renamed task YAML + verifier (+ verifier_utils.py)
    step_NN_task_MM/episode_1/     the harness's own output: trajectories/, replay/state.jsonl,
                                   messages, events, chain_events.jsonl, summary.jsonl
    step_NN_task_MM/hebbian_W.npy  W after the step (agent-id order, see chain_summary)
    step_NN_task_MM/memory.json    each agent's skill and episode memory after the step
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"


def bootstrap() -> None:
    """src/, src/mindforge (MindForge's bare ``agent_modules`` imports) and chains/."""
    for p in (SRC, SRC / "mindforge", HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


HEBBIAN_KEYS = ("hebbian_eta_0", "hebbian_eta_plus", "hebbian_decay", "hebbian_eligibility_rho",
                "hebbian_coact_floor", "hebbian_radius", "hebbian_reward_norm", "hebbian_death_ltd",
                "hebbian_init_weight", "hebbian_salience", "cofiring_channels", "xfer_directed")


def build_parser() -> argparse.ArgumentParser:
    bootstrap()
    from agentworld.cli import build_parser as aw_parser
    d = aw_parser()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain", type=Path, help="chain YAML, e.g. chains/pendant_staff_4.yaml")
    ap.add_argument("--arm", choices=("base", "hebbian"), default="hebbian")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("runs_mf"),
                    help="parent output folder; the run goes in <output>/<chain>/<run_id>")
    ap.add_argument("--run-id", default=None, help="default: <time>_mf_<arm>_s<seed>")
    ap.add_argument("--agentworld-root", default=None, help="default: $AGENTWORLD_ROOT")
    ap.add_argument("--plumbing", action="store_true",
                    help="no model: a fixed script per agent (tests server, tasks, verifiers, outputs)")

    g = ap.add_argument_group("server")
    g.add_argument("--server-url", required=True, help="the game server's API, e.g. http://127.0.0.1:7031")
    g.add_argument("--restart-file", type=Path, default=None,
                   help="a host-side supervisor restarts the server when this file appears "
                        "(hpc/agentworld/aw_common.sh); then the server restarts before every step")
    g.add_argument("--setup-concurrency", type=int, default=16)
    g.add_argument("--setup-sleep-scale", type=float, default=1.0)

    g = ap.add_argument_group("protocol (defaults match the chains/ runs where the harness allows)")
    g.add_argument("--rounds", type=int, default=None, help="default: the task's (55)")
    g.add_argument("--max-actions", type=int, default=-1,
                   help="game actions per agent per step; -1 = the task's max_action_steps (25), 0 = no cap")
    g.add_argument("--hold-complete", action="store_true",
                   help="complete() before the step's verifier passes is refused and counts as an action")
    g.add_argument("--board", action="store_true",
                   help="keep the harness's shared board (off: direct messages only, as in chains/)")
    g.add_argument("--min-round-seconds", type=float, default=d.get_default("min_round_seconds"))
    g.add_argument("--deadline", type=float, default=d.get_default("deadline"))
    g.add_argument("--obs-radius", type=int, default=d.get_default("obs_radius"))
    g.add_argument("--board-budget", type=int, default=d.get_default("board_budget"))
    g.add_argument("--inbox-budget", type=int, default=d.get_default("inbox_budget"))
    g.add_argument("--contacts-budget", type=int, default=d.get_default("contacts_budget"))
    g.add_argument("--explore-slots", type=int, default=d.get_default("explore_slots"))

    g = ap.add_argument_group("cognition (agentworld/cli.py defaults)")
    g.add_argument("--belief-interval", type=int, default=d.get_default("belief_interval"))
    g.add_argument("--critic-interval", type=int, default=d.get_default("critic_interval"))
    g.add_argument("--partner-updates", type=int, default=d.get_default("partner_updates"))
    g.add_argument("--embedder", choices=("auto", "hash", "sentence"), default=d.get_default("embedder"))
    g.add_argument("--max-in-flight", type=int, default=64)

    g = ap.add_argument_group("Hebbian rule (defaults = agentworld/cli.py's calibrated values)")
    for dest in HEBBIAN_KEYS[:9]:
        g.add_argument("--" + dest.replace("_", "-"), dest=dest, type=float, default=d.get_default(dest))
    g.add_argument("--hebbian-salience", choices=("ego", "joint"), default=d.get_default("hebbian_salience"))
    g.add_argument("--cofiring-channels", default=d.get_default("cofiring_channels"))
    g.add_argument("--xfer-directed", action="store_true", default=d.get_default("xfer_directed"))
    return ap


def generate(chain_path: Path, root: Path, tasks_dir: Path, chain_name: str) -> List[Any]:
    """The renamed tasks, by chains/generate_tasks.py in its own process (AgentWorld's
    loader there must not share sys.modules with MindForge), plus verifier_utils.py
    next to every verifier, which agentworld.verify loads from the verifier's folder."""
    from mf_chain import ChainRunError, StepSpec
    cmd = [sys.executable, str(HERE / "generate_tasks.py"), str(chain_path),
           "--agentworld-root", str(root), "--out", str(tasks_dir)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise ChainRunError(f"generate_tasks.py failed:\n{out.stdout}\n{out.stderr}")
    manifest = json.loads((tasks_dir / chain_name / "manifest.json").read_text(encoding="utf-8"))
    steps = [StepSpec.from_manifest(e) for e in manifest["steps"]]
    for s in steps:
        utils = s.source_verifier.parent / "verifier_utils.py"
        if not utils.is_file():
            raise ChainRunError(f"no verifier_utils.py next to {s.source_verifier}")
        shutil.copyfile(utils, s.verifier.parent / "verifier_utils.py")
    return steps


def totals(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [s for s in steps if not s.get("error")]
    if not done:
        return {"steps": 0}

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    return {"steps": len(done), "SR_percent": round(100 * mean([float(s["success"]) for s in done]), 2),
            "rounds_mean": mean([s.get("rounds") for s in done]),
            "dms_mean": mean([s.get("dms") for s in done])}


async def run(args) -> int:
    from chain_spec import load_chain, resolve_root
    from mf_chain import (ChainCognition, ChainRunError, PlumbingAgent, bonds_by_id,
                          memory_snapshot, play_step, use_direct_only_prompts, write_json)
    import numpy as np

    root = resolve_root(args.agentworld_root)
    if root is None or not (root / "agents" / "verification.py").is_file():
        raise SystemExit("set --agentworld-root or $AGENTWORLD_ROOT to an AgentWorld checkout")
    chain = load_chain(args.chain, root)
    agent_ids = list(chain.agent_ids)
    run_id = args.run_id or (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_mf_{args.arm}_s{args.seed}")
    run_dir = (args.output / chain.name / run_id).resolve()
    if (run_dir / "chain_summary.json").exists():
        raise SystemExit(f"{run_dir} already holds a run; use a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = generate(Path(args.chain).resolve(), root, run_dir / "tasks", chain.name)

    from agentworld.vendor import load_game_tools
    vendor = load_game_tools(root)
    server = None
    if args.restart_file:
        from agentworld.server import SupervisedServer
        server = SupervisedServer(args.server_url, args.restart_file)
        logging.info("game server ready after %.1fs", server.start())

    if args.plumbing:
        agents = {a: PlumbingAgent(a) for a in agent_ids}
        cognition = ChainCognition(agents)
    else:
        if not args.board:
            use_direct_only_prompts()
        from agentworld.agent import AgentConfig, MindForgeCognition, ModelClients
        from agentworld.memory import SharedMemory, make_embedder
        mf = MindForgeCognition(agent_ids, ModelClients.from_env(),
                                memory=SharedMemory(make_embedder(args.embedder)),
                                config=AgentConfig(belief_interval=args.belief_interval,
                                                   critic_interval=args.critic_interval,
                                                   partner_updates=args.partner_updates),
                                max_in_flight=args.max_in_flight)
        cognition = ChainCognition(dict(zip(agent_ids, mf.agents)), ledger=mf.ledger)

    hebbian = {k: getattr(args, k) for k in HEBBIAN_KEYS}
    n = len(agent_ids)
    W = np.full((n, n), float(args.hebbian_init_weight), dtype=np.float32)
    np.fill_diagonal(W, 0.0)
    summary: Dict[str, Any] = {
        "chain": chain.name, "run_id": run_id, "chain_file": str(Path(args.chain).resolve()),
        "description": chain.description, "harness": "mindforge (src/agentworld, chains/mf_chain.py)",
        "arm": args.arm, "seed": args.seed, "plumbing": bool(args.plumbing),
        "hold_complete": bool(args.hold_complete), "max_actions": args.max_actions,
        "board": bool(args.board), "rounds": args.rounds, "min_round_s": args.min_round_seconds,
        "deadline_s": args.deadline, "obs_radius": args.obs_radius,
        "budgets": {"board": args.board_budget if args.board else 0, "inbox": args.inbox_budget,
                    "contacts": args.contacts_budget},
        "cognition": {"belief_interval": args.belief_interval, "critic_interval": args.critic_interval,
                      "partner_updates": args.partner_updates, "embedder": args.embedder},
        "hebbian": hebbian, "agent_ids": agent_ids, "agentworld_root": str(root),
        "server_restart_per_step": server is not None, "steps": [],
    }
    summary_path = run_dir / "chain_summary.json"

    def save():
        summary["totals"] = totals(summary["steps"])
        summary["updated"] = datetime.now().isoformat(timespec="seconds")
        write_json(summary_path, summary)

    save()
    status = 0
    try:
        for k, step in enumerate(steps):
            names = ", ".join(s["username"] for s in step.slots)
            print(f"\n=== {chain.name} step {step.index}/{len(steps)}: {step.task_key} [{names}] ===",
                  flush=True)
            if k > 0 and server is not None:
                logging.info("server restart %.1fs", server.restart())
            step_dir = run_dir / step.dir_name
            t0 = time.time()
            W_start = W.copy()
            try:
                res, W, ids = await play_step(
                    step, agent_ids=agent_ids, W=W, cognition=cognition,
                    make_tools=lambda: vendor.make_tools(args.server_url),
                    master_password=vendor.master_password, out_dir=step_dir, arm=args.arm,
                    seed=args.seed, hebbian=hebbian, rounds=args.rounds,
                    max_actions=args.max_actions, hold_complete=args.hold_complete,
                    board=args.board, board_budget=args.board_budget,
                    inbox_budget=args.inbox_budget, contacts_budget=args.contacts_budget,
                    explore_slots=args.explore_slots, obs_radius=args.obs_radius,
                    deadline_s=args.deadline, min_round_s=args.min_round_seconds,
                    setup_concurrency=args.setup_concurrency,
                    setup_sleep_scale=args.setup_sleep_scale)
            except Exception as exc:          # record, then stop: later state would be wrong
                summary["steps"].append({"step": step.index, "dir": step.dir_name,
                                         "task_key": step.task_key, "slots": step.slots,
                                         "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
                save()
                print(f"step {step.index} failed: {exc}", file=sys.stderr, flush=True)
                status = 1
                break
            team = next(iter(res["teams"].values()))
            ch = res["chain"]
            by_name = {s["username"]: s["agent_id"] for s in step.slots}
            np.save(step_dir / "hebbian_W.npy", W.astype(np.float32))
            write_json(step_dir / "memory.json",
                       {a: memory_snapshot(cognition.agents[a]) for a in agent_ids})
            rec = {
                "step": step.index, "dir": step.dir_name, "task_key": step.task_key,
                "task_yaml": str(step.task_yaml), "verifier": str(step.verifier), "slots": step.slots,
                "success": bool(team["success"]), "verifier_message": team["message"],
                "rounds": res["rounds"], "last_active_round": ch["last_active_round"],
                "duration_s": round(time.time() - t0, 1),
                "actions": {by_name[u]: v for u, v in ch["actions"].items()},
                "held_completes": sum(ch["held_completes"].values()),
                "held_by_agent": {by_name[u]: v for u, v in ch["held_completes"].items()},
                "refused_chats": sum(ch["refused_chats"].values()),
                "capped": {by_name[u]: r for u, r in ch["capped"].items()},
                "completed": {by_name[u]: r for u, r in ch["completed"].items()},
                "dms": ch["dms"], "posts": ch["posts"], "max_actions": ch["max_actions"],
                "bonds_start": bonds_by_id(W_start, agent_ids), "bonds_end": bonds_by_id(W, agent_ids),
                "tokens": cognition.ledger.snapshot() if cognition.ledger is not None else None,
            }
            summary["steps"].append(rec)
            save()
            print(f"step {step.index}: {'SUCCESS' if rec['success'] else 'FAILED'} - "
                  f"{rec['verifier_message']} | rounds {rec['rounds']} | dms {rec['dms']} | "
                  f"held {rec['held_completes']} | capped {sorted(rec['capped'])}", flush=True)
    finally:
        if server is not None:
            server.stop()
    print(f"\nrun: {run_dir}")
    return status


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
