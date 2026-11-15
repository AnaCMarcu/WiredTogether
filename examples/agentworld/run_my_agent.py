"""Run YOUR agent class in an AgentWorld world with the Wired Together harness.

    PYTHONPATH=src python examples/agentworld/run_my_agent.py \\
        --agent my_agent:ScriptedAgent --tasks task_01_magic_staff \\
        --rounds 10 --arm base --out runs_aw/mine/smoke

Same world, server, message routing, Hebbian graph, trajectories, verifiers and
videos as ``python -m mindforge.multi_agent_agentworld`` (src/agentworld/cli.py,
whose flags are all accepted here); only the cognition is yours instead of
MindForge's. ``--agent module:Class`` is imported from this folder, from
PYTHONPATH, or from the current directory. The class contract is in my_agent.py.
The "cognition" flags (--belief-interval, --critic-interval, --partner-updates,
--voyager, --embedder, --max-in-flight) configure MindForge and are ignored here.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "src"), str(HERE), str(Path.cwd())):
    if p not in sys.path:
        sys.path.insert(0, p)

from agentworld.cli import _render_videos, _resolve_tasks, build_parser  # noqa: E402


def load_class(target: str):
    module_name, sep, class_name = target.partition(":")
    if not sep or not module_name or not class_name:
        raise SystemExit(f"--agent must be module:Class, got {target!r}")
    return getattr(importlib.import_module(module_name), class_name)


def parser():
    ap = build_parser()
    ap.description = __doc__
    ap.add_argument("--agent", default="my_agent:ScriptedAgent",
                    help="your cognition class as module:Class (see my_agent.py)")
    return ap


async def run(args) -> List[dict]:
    import numpy as np

    from agentworld.comms import CommRouter, make_policy
    from agentworld.executor import AgentHandle, Executor, prepare_agent
    from agentworld.progress import ProgressTracker
    from agentworld.scheduler import RoundScheduler, SchedulerConfig
    from agentworld.server import (KaetramServer, ServerConfig, SupervisedServer, job_ports,
                                   local_command)
    from agentworld.tasks import anonymise_text, compose_world, load_task
    from agentworld.vendor import load_game_tools
    from agentworld.verify import Verifier, extract_targets
    from hebbian.multichannel import MultiChannelConfig, MultiChannelHebbianGraph

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=1))
    vendor = load_game_tools(args.agentworld_root)
    tasks = [load_task(p) for p in _resolve_tasks(args) for _ in range(args.replicas)]
    if args.anonymise:                      # as cli.py: role names instead of teammate usernames
        for t in tasks:
            roles = {a.key: "the " + a.username.split("_")[-2] if a.username.count("_") >= 2
                     else a.username for a in t.agents}
            names = {a.key: a.username for a in t.agents}
            t.objective = anonymise_text(t.objective, names, roles)
            t.game_context = anonymise_text(t.game_context, names, roles)
    world = compose_world(tasks, team_spacing=args.team_spacing)
    n = world.n_agents
    logging.info("world: %d agents in %d teams: %s", n, len(world.teams), world.global_names)

    server = None
    if args.server_url and args.restart_file:
        server = SupervisedServer(args.server_url, args.restart_file)
        logging.info("game server ready after %.1fs", server.start())
        api_url = server.api_url
    elif args.server_url:
        api_url = args.server_url
    else:
        game_port, api_port = job_ports()
        server = KaetramServer(ServerConfig(command=args.server_cmd or local_command(vendor.root),
                                            game_port=game_port, api_port=api_port,
                                            max_players=max(200, n + 8),
                                            log_path=args.out / "server.log",
                                            agentworld_root=vendor.root))
        logging.info("game server up in %.1fs", server.start())
        api_url = server.api_url

    same_team = world.true_team_matrix()
    # The bonds. Computed in every arm (base: logged only); see AGENTWORLD.md.
    graph = MultiChannelHebbianGraph(MultiChannelConfig(
        enabled=True, num_agents=n, eta_0=args.hebbian_eta_0, eta_plus=args.hebbian_eta_plus,
        decay=args.hebbian_decay, eligibility_rho=args.hebbian_eligibility_rho,
        coact_floor=args.hebbian_coact_floor, interaction_radius=args.hebbian_radius,
        reward_norm_R=args.hebbian_reward_norm, eta_minus_death=args.hebbian_death_ltd,
        init_weight=args.hebbian_init_weight, salience_mode=args.hebbian_salience,
        social_act_channels=tuple(c.strip() for c in args.cofiring_channels.split(",") if c.strip()),
        channel_symmetric={"xfer": not args.xfer_directed}))
    policy_name = {"base": "recency", "hebbian": "hebbian", "shuffled": "shuffled",
                   "prompt_only": "recency", "oracle": "oracle"}[args.arm]
    cfg = SchedulerConfig(max_rounds=args.rounds, obs_radius=args.obs_radius,
                          comm_mode=args.comm_mode, bonds_in_prompt=args.arm != "base",
                          bond_source="graph" if args.arm == "prompt_only" else "policy",
                          mirror_dms=args.mirror_dms, min_round_s=args.min_round_seconds)
    cognition = load_class(args.agent)(list(world.global_names))
    targets = {t.team: extract_targets(t.task.verifier_path()) for t in world.teams}
    verifiers = {t.team: Verifier(t.task.verifier_path()) for t in world.teams}

    summaries = []
    try:
        for ep in range(1, args.episodes + 1):
            if ep > 1 and server is not None and not args.no_restart:
                logging.info("server restart %.1fs", server.restart())
            gate = asyncio.Semaphore(max(1, args.setup_concurrency))
            scale = args.setup_sleep_scale

            async def setup(i: int, name: str) -> AgentHandle:
                tools = vendor.make_tools(api_url)
                spec = next(a for a in world.team_of(i).task.agents if a.key == world.global_key[i])
                async with gate:
                    ok = await asyncio.to_thread(prepare_agent, tools, spec, name,
                                                 vendor.master_password,
                                                 sleep=lambda sec: time.sleep(sec * scale),
                                                 location=world.global_spawn[i])
                if not ok:
                    raise RuntimeError(f"could not log in {name}")
                return AgentHandle(i, name, world.global_key[i], world.global_team[i], tools)

            handles = list(await asyncio.gather(*(setup(i, nm) for i, nm in
                                                  enumerate(world.global_names))))
            executor = Executor(handles, deadline_s=args.deadline)
            router = CommRouter(n, make_policy(policy_name, seed=args.seed + ep, same_team=same_team),
                                board_budget=args.board_budget, inbox_budget=args.inbox_budget,
                                contacts_budget=args.contacts_budget,
                                explore_slots=args.explore_slots, seed=args.seed + ep)
            if hasattr(cognition, "on_reset"):
                cognition.on_reset()
            sched = RoundScheduler(world, executor, router, cognition,
                                   ProgressTracker(world, targets), verifiers, args.out,
                                   graph=graph, config=cfg)
            summary = await sched.run_episode(ep)
            ledger = getattr(cognition, "ledger", None)
            summary["tokens"] = ledger.snapshot() if ledger is not None else None
            summary["arm"] = args.arm
            summary["agent"] = args.agent
            summaries.append(summary)
            ep_dir = args.out / f"episode_{ep}"
            np.save(ep_dir / "hebbian_W.npy", graph.W.astype(np.float16))
            np.save(ep_dir / "true_team.npy", same_team.astype(np.int8))
            (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1))
            executor.shutdown()
            if args.video != "none":
                _render_videos(args, ep, world)
    finally:
        if server is not None:
            server.stop()
    return summaries


def main(argv: Optional[List[str]] = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for s in asyncio.run(run(args)):
        solved = sum(t["success"] for t in s["teams"].values())
        logging.info("episode %d: %d/%d teams solved in %d rounds (%.0fs)",
                     s["episode"], solved, len(s["teams"]), s["rounds"], s["wall_s"])
    logging.info("output: %s", args.out)


if __name__ == "__main__":
    main()
