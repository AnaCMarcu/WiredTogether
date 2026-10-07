"""Run MindForge (+ Hebbian) agents in an AgentWorld world.

    python -m mindforge.multi_agent_agentworld \\
        --agentworld-root $AGENTWORLD_ROOT --tasks task_02_arrow_production \\
        --replicas 10 --team-spacing 20 --arm hebbian --episodes 3 --out runs/aw/...

Arms (one switch, all other settings shared):

    base         recency reading; W is still computed (observer only) for analysis
    hebbian      bond-gated reading + bonds in the prompt (three-factor, multi-channel)
    shuffled     as hebbian, but gating and prompt use a per-round permutation of W
    prompt_only  W in the prompt only; reading by recency
    oracle       reading gated by true team membership (multi-team worlds only)

Hebbian flag names and defaults mirror ``mindforge/cli.py`` where the same
quantity exists. The rate defaults (eta_0 0.01, eta_plus 0.1, decay 0.02,
rho 0.7) come from analysis/agentworld/calibrate_rule.py: on synthetic
55-round streams a pair that talks and trades reaches W ~0.6 by round 20, a
talk-only pair ~0.14, a silent co-located pair ~0.10, and a pair that stops
interacting relaxes to ~57 % within half an episode. Re-check on the N=10 pilot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

ARMS = ("base", "hebbian", "shuffled", "prompt_only", "oracle")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("world")
    g.add_argument("--agentworld-root", default=os.environ.get("AGENTWORLD_ROOT"))
    g.add_argument("--suite", default="data_v0.1_multi/v1.3_benchmark",
                   help="task directory, relative to --agentworld-root")
    g.add_argument("--tasks", nargs="+", required=True,
                   help="task file stems or paths; with --replicas each is repeated")
    g.add_argument("--replicas", type=int, default=1)
    g.add_argument("--team-spacing", type=int, default=20,
                   help="grid offset between teams in multi-team worlds (tiles)")
    g.add_argument("--anonymise", action="store_true",
                   help="replace teammate usernames in task text with role names")
    g.add_argument("--episodes", type=int, default=1)
    g.add_argument("--rounds", type=int, default=None, help="default: the tasks' own")
    g.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("server")
    g.add_argument("--server-url", default=None,
                   help="use an already-running server's API (e.g. http://127.0.0.1:7031)")
    g.add_argument("--restart-file", type=Path, default=None,
                   help="with --server-url: a host-side supervisor restarts the server when "
                        "this file appears (cluster mode, see hpc/agentworld/aw_common.sh)")
    g.add_argument("--server-cmd", nargs="+", default=None,
                   help="command that starts a server (else: local yarn); restarted per episode")
    g.add_argument("--no-restart", action="store_true",
                   help="do not restart the server between episodes (characters are reset)")
    g.add_argument("--setup-concurrency", type=int, default=16,
                   help="agents logged in / set up at once")
    g.add_argument("--setup-sleep-scale", type=float, default=1.0,
                   help="scale AgentWorld's setup waits (teleport 1.5 s, skills 1.0 s); 0 in tests")

    g = ap.add_argument_group("harness")
    g.add_argument("--arm", choices=ARMS, default="hebbian")
    g.add_argument("--comm-mode", choices=("side", "tool"), default="side")
    g.add_argument("--board-budget", type=int, default=5)
    g.add_argument("--inbox-budget", type=int, default=4)
    g.add_argument("--contacts-budget", type=int, default=6)
    g.add_argument("--explore-slots", type=int, default=2)
    g.add_argument("--obs-radius", type=int, default=24)
    g.add_argument("--deadline", type=float, default=30.0, help="per-round action deadline (s)")
    g.add_argument("--mirror-dms", action="store_true",
                   help="re-post DMs/board posts as local game chat (live video bubbles)")

    g = ap.add_argument_group("cognition")
    g.add_argument("--belief-interval", type=int, default=5)
    g.add_argument("--critic-interval", type=int, default=3)
    g.add_argument("--partner-updates", type=int, default=2)
    g.add_argument("--voyager", action="store_true", help="no beliefs / episodes (ablation)")
    g.add_argument("--embedder", choices=("auto", "hash", "sentence"), default="auto")
    g.add_argument("--max-in-flight", type=int, default=256,
                   help="concurrent LLM requests (vLLM: hundreds; API anchor: ~8)")

    g = ap.add_argument_group("hebbian")
    g.add_argument("--hebbian-eta-0", type=float, default=0.01)
    g.add_argument("--hebbian-eta-plus", type=float, default=0.1)
    g.add_argument("--hebbian-decay", type=float, default=0.02)
    g.add_argument("--hebbian-eligibility-rho", type=float, default=0.7)
    g.add_argument("--hebbian-coact-floor", type=float, default=0.25)
    g.add_argument("--hebbian-radius", type=float, default=4.0, help="tiles")
    g.add_argument("--hebbian-reward-norm", type=float, default=20.0)
    g.add_argument("--hebbian-death-ltd", type=float, default=0.05)
    g.add_argument("--hebbian-init-weight", type=float, default=0.1)
    g.add_argument("--hebbian-salience", choices=("ego", "joint"), default="ego")
    g.add_argument("--cofiring-channels", default="comm,xfer,combat",
                   help="credited channels (add 'read' to wire board reading)")
    g.add_argument("--xfer-directed", action="store_true")

    g = ap.add_argument_group("output")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--video", choices=("none", "team", "world", "both"), default="team",
                   help="render replay MP4s after each episode")
    return ap


def _resolve_tasks(args) -> List[Path]:
    from agentworld.vendor import find_root
    root = find_root(args.agentworld_root)
    suite = root / args.suite
    out = []
    for t in args.tasks:
        p = Path(t)
        if not p.suffix:
            p = suite / f"{t}.yaml"
        elif not p.is_absolute() and not p.exists():
            p = suite / p
        if not p.is_file():
            raise FileNotFoundError(p)
        out.append(p)
    return out


async def run(args) -> List[dict]:
    import numpy as np

    from agentworld.agent import AgentConfig, MindForgeCognition, ModelClients
    from agentworld.comms import CommRouter, make_policy
    from agentworld.executor import AgentHandle, Executor, prepare_agent
    from agentworld.memory import SharedMemory, make_embedder
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
    if args.anonymise:
        for t in tasks:
            roles = {a.key: "the " + a.username.split("_")[-2] if a.username.count("_") >= 2
                     else a.username for a in t.agents}
            names = {a.key: a.username for a in t.agents}
            t.objective = anonymise_text(t.objective, names, roles)
            t.game_context = anonymise_text(t.game_context, names, roles)
    world = compose_world(tasks, team_spacing=args.team_spacing)
    n = world.n_agents
    logging.info("world: %d agents in %d teams", n, len(world.teams))

    server = None
    if args.server_url and args.restart_file:
        server = SupervisedServer(args.server_url, args.restart_file)
        logging.info("game server ready after %.1fs", server.start())
        api_url = server.api_url
    elif args.server_url:
        api_url = args.server_url
    else:
        game_port, api_port = job_ports()
        cmd = args.server_cmd or local_command(vendor.root)
        server = KaetramServer(ServerConfig(command=cmd, game_port=game_port, api_port=api_port,
                                            max_players=max(200, n + 8),
                                            log_path=args.out / "server.log",
                                            agentworld_root=vendor.root))
        logging.info("game server up in %.1fs", server.start())
        api_url = server.api_url

    same_team = world.true_team_matrix()
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
                          comm_mode=args.comm_mode,
                          bonds_in_prompt=args.arm != "base",
                          bond_source="graph" if args.arm == "prompt_only" else "policy",
                          mirror_dms=args.mirror_dms)
    cognition = MindForgeCognition(
        world.global_names, ModelClients.from_env(),
        memory=SharedMemory(make_embedder(args.embedder)),
        config=AgentConfig(belief_interval=args.belief_interval,
                           critic_interval=args.critic_interval,
                           partner_updates=args.partner_updates, voyager=args.voyager),
        max_in_flight=args.max_in_flight)
    targets = {t.team: extract_targets(t.task.verifier_path()) for t in world.teams}
    verifiers = {t.team: Verifier(t.task.verifier_path()) for t in world.teams}

    summaries = []
    try:
        for ep in range(1, args.episodes + 1):
            if ep > 1 and server is not None and not args.no_restart:
                logging.info("server restart %.1fs", server.restart())
            # Same usernames every episode: names are how DMs and transfers are
            # resolved. prepare_agent resets inventory, skills and position; with
            # --no-restart only server-side leftovers (mobs, ground items, the game
            # chat our agents never read) carry over.
            setup_gate = asyncio.Semaphore(max(1, args.setup_concurrency))
            scale = args.setup_sleep_scale

            async def setup(i: int, name: str) -> AgentHandle:
                tools = vendor.make_tools(api_url)
                spec = next(a for a in world.team_of(i).task.agents if a.key == world.global_key[i])
                async with setup_gate:
                    ok = await asyncio.to_thread(
                        prepare_agent, tools, spec, name, vendor.master_password,
                        sleep=lambda sec: time.sleep(sec * scale),
                        location=world.global_spawn[i])
                if not ok:
                    raise RuntimeError(f"could not log in {name}")
                return AgentHandle(i, name, world.global_key[i], world.global_team[i], tools)

            t_setup = time.time()
            handles = list(await asyncio.gather(*(setup(i, nm) for i, nm in
                                                  enumerate(world.global_names))))
            logging.info("episode %d: %d agents set up in %.1fs", ep, n, time.time() - t_setup)
            executor = Executor(handles, deadline_s=args.deadline)

            def mirror(i, text, _h=handles):
                t = _h[i].tools
                executor._pool.submit(t._make_request, "POST", "/ai/chat",
                                      {"token": t.token, "message": text[:200], "global": False})

            router = CommRouter(n, make_policy(policy_name, seed=args.seed + ep, same_team=same_team),
                                board_budget=args.board_budget, inbox_budget=args.inbox_budget,
                                contacts_budget=args.contacts_budget,
                                explore_slots=args.explore_slots, seed=args.seed + ep)
            cognition.on_reset()
            sched = RoundScheduler(world, executor, router, cognition,
                                   ProgressTracker(world, targets), verifiers, args.out,
                                   graph=graph, config=cfg, mirror=mirror)
            summary = await sched.run_episode(ep)
            summary["tokens"] = cognition.ledger.snapshot()
            summary["arm"] = args.arm
            summaries.append(summary)
            np.save(args.out / f"episode_{ep}" / "hebbian_W.npy", graph.W.astype(np.float16))
            np.save(args.out / f"episode_{ep}" / "true_team.npy", same_team.astype(np.int8))
            (args.out / f"episode_{ep}" / "summary.json").write_text(json.dumps(summary, indent=1))
            executor.shutdown()
            if args.video != "none":
                _render_videos(args, ep, world)
    finally:
        if server is not None:
            server.stop()
    return summaries


def _render_videos(args, ep: int, world) -> None:
    from agentworld.replay.mapdata import load_map
    from agentworld.replay.render import RenderOptions, ReplayRenderer, load_states
    ep_dir = args.out / f"episode_{ep}"
    states = load_states(ep_dir)
    m = load_map(args.agentworld_root)
    cams = []
    if args.video in ("team", "both"):
        cams += [("team", t.team) for t in world.teams[:3]]
    if args.video in ("world", "both") and len(world.teams) > 1:
        cams.append(("world", None))
    for cam, team in cams:
        title = f"{args.arm} · {world.teams[0].task.task_id}" + (f" · {team}" if team else "")
        r = ReplayRenderer(states, m, RenderOptions(camera=cam, team=team, title=title))
        r.render_to(ep_dir / "videos" / (f"team_{team}.mp4" if team else "world.mp4"))


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    summaries = asyncio.run(run(args))
    for s in summaries:
        solved = sum(t["success"] for t in s["teams"].values())
        logging.info("episode %d: %d/%d teams solved in %d rounds (%.0fs)",
                     s["episode"], solved, len(s["teams"]), s["rounds"], s["wall_s"])
    logging.info("done in %.0fs → %s", time.time() - t0, args.out)


if __name__ == "__main__":
    main()
