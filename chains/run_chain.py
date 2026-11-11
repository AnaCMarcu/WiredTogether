"""Run a chain of AgentWorld tasks with one persistent team on one game server.

    python chains/run_chain.py chains/smithing_3.yaml \\
        --agent chains/configs/wired_openai.yaml --arm hebbian --seed 42

Steps run in chain order (``--task-folder`` would sort by task number), each
through AgentWorld's own ``TaskRunner.run_single_task`` from
``$AGENTWORLD_ROOT/agents/run.py``. A small subclass adds calls into
:class:`wired_state.StepContext`: after login (identity check, load state,
``on_task_start``), after every agent turn (``on_turn`` for both partners of
a message or transfer), at the start of every round (the previous round's
plasticity update) and before the task summary is saved (the last round's). After each step the step's verifier is run again here
and its outcome goes to every agent's ``on_task_end``, which saves the state.

Output, ``<output>/<chain>/<run_id>/``::

    chain_summary.json          per step: success, verifier message, PSR checkpoints,
                                rounds, chats, state loaded/saved; totals; checks
    agent_state/a1.json ...     persistent state, one file per agent id
    tasks/step_NN_task_MM/      the renamed task YAML + verifier this run played
    step_NN_task_MM/            AgentWorld's own output for the step (task_MM_trajectory.json,
                                task_runner_*.log, observation_logs/, ...) and
                                wired_rounds.jsonl (events, third factor, W per round)

Arms (``--arm``): ``reference`` shows the model nothing extra (AgentWorld's
reference agent; bonds and memory are still computed and saved),
``memory`` shows earlier outcomes and past interactions, ``hebbian`` also
shows the bond line. Metrics follow the AgentWorld paper: SR, reconstructed
PSR (``scripts/score_psr.py``, checkpoint_fraction_v1), rounds and chats; CCE
needs an LLM judge and is computed afterwards with their
``analysis/scripts/compute_cce_v2.py`` on each step's trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"


def bootstrap(root: Optional[Path] = None) -> None:
    """chains/, src/ and, first of all, AgentWorld's agents/ (bare-module imports).

    AgentWorld imports its own modules by bare name (agent_factory, config,
    run, ...). src/mindforge has an agent_factory.py too, and some of our entry
    points put src/mindforge on sys.path, so agents/ goes first and any
    same-named module already loaded from elsewhere is dropped.
    """
    for p in (SRC, HERE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if root is None:
        return
    agents = (root / "agents").resolve()
    if str(agents) in sys.path:
        sys.path.remove(str(agents))
    sys.path.insert(0, str(agents))
    for f in agents.glob("*.py"):
        mod = sys.modules.get(f.stem)
        origin = getattr(mod, "__file__", None)
        if mod is not None and origin and Path(origin).resolve().parent != agents:
            del sys.modules[f.stem]


# ── argument parsing ─────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain", type=Path, help="chain YAML, e.g. chains/smithing_3.yaml")
    ap.add_argument("--agent", required=True,
                    help="agent config YAML (provider: custom, llm.agent_class: wired_agent:...)")
    ap.add_argument("--output", type=Path, default=Path("runs"),
                    help="parent output folder; the run goes in <output>/<chain>/<run_id>")
    ap.add_argument("--run-id", default=None,
                    help="default: <time>_<arm>[_s<seed>]; reuse one to continue a run with --steps")
    ap.add_argument("--agentworld-root", default=None, help="default: $AGENTWORLD_ROOT")
    ap.add_argument("--arm", choices=("reference", "memory", "hebbian"), default="hebbian")
    ap.add_argument("--steps", default=None,
                    help="only these steps, e.g. '1' or '2-3'; earlier steps' state must be in the run")
    ap.add_argument("--split-screen", action="store_true",
                    help="AgentWorld's curses display (off by default for chains)")
    ap.add_argument("--dump-prompts", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the end-of-run checks")
    ap.add_argument("--hold-complete", action="store_true",
                    help="complete() ends an agent only once the step's verifier passes; until then "
                         "the agent is told the objective is not met and keeps acting (harness change)")

    exp = ap.add_argument_group("AgentWorld experiment flags (passed to every step)")
    exp.add_argument("--single-agent-upper-bound", action="store_true", help="not supported in chains")
    exp.add_argument("--discussion-rounds", type=int, default=0, metavar="N")
    exp.add_argument("--no-communication", action="store_true")
    exp.add_argument("--shared-plan-only", action="store_true")
    exp.add_argument("--random-agent", action="store_true",
                     help="no model calls: random legal actions (tests the plumbing only)")
    exp.add_argument("--max-rounds", type=int, default=None, metavar="N")
    exp.add_argument("--no-roles", action="store_true", help="not supported in chains")
    exp.add_argument("--random-spawn", action="store_true")
    exp.add_argument("--spawn-region", type=int, nargs=4, default=None,
                     metavar=("X1", "Y1", "X2", "Y2"))
    exp.add_argument("--no-task-docs", action="store_true")
    exp.add_argument("--seed", type=int, default=None, metavar="N")

    heb = ap.add_argument_group("Hebbian rule (defaults = agentworld/cli.py's calibrated values)")
    bootstrap()
    from agentworld.cli import build_parser as aw_parser
    defaults = aw_parser()
    for dest in ("hebbian_eta_0", "hebbian_eta_plus", "hebbian_decay", "hebbian_eligibility_rho",
                 "hebbian_coact_floor", "hebbian_radius", "hebbian_reward_norm",
                 "hebbian_death_ltd", "hebbian_init_weight"):
        heb.add_argument("--" + dest.replace("_", "-"), dest=dest, type=float,
                         default=defaults.get_default(dest))
    heb.add_argument("--hebbian-salience", choices=("ego", "joint"),
                     default=defaults.get_default("hebbian_salience"))
    heb.add_argument("--cofiring-channels", default=defaults.get_default("cofiring_channels"))
    heb.add_argument("--xfer-directed", action="store_true",
                     default=defaults.get_default("xfer_directed"))
    return ap


def parse_steps(spec: Optional[str], n: int) -> List[int]:
    if not spec:
        return list(range(1, n + 1))
    out: List[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        out.extend(range(lo, hi + 1))
    bad = [k for k in out if not 1 <= k <= n]
    if bad:
        raise SystemExit(f"--steps {spec!r}: the chain has steps 1..{n}")
    return sorted(set(out))


# ── AgentWorld pieces ────────────────────────────────────────────────────────
def _load_file_module(name: str, path: Path):
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def psr_for(score_psr, task_key: str, message: str, success: bool) -> Dict[str, Any]:
    """Reconstructed PSR (AgentWorld scripts/score_psr.py) plus the checkpoint breakdown."""
    if score_psr is None:
        return {"method": None, "PSR": None,
                "error": "scripts/score_psr.py not in the AgentWorld checkout (needs >= 23a07d52)"}
    try:
        detail = score_psr.checkpoint_score(task_key, message, False)["checkpoints"]
        official = score_psr.checkpoint_score(task_key, message, success)["PSR"]
        return {"method": score_psr.METHOD, "PSR": official, "checkpoints": detail,
                "n_checkpoints": len(detail),
                "n_met": sum(1 for c in detail if c["credit"] >= 1.0)}
    except (ValueError, KeyError, AttributeError) as exc:
        return {"method": getattr(score_psr, "METHOD", None),
                "PSR": 1.0 if success else None, "error": f"{type(exc).__name__}: {exc}"}


def chain_runner_class(TaskRunner):
    class ChainTaskRunner(TaskRunner):
        """AgentWorld's TaskRunner plus the StepContext calls; nothing else changes."""

        def __init__(self, *args, step_ctx=None, **kwargs):
            self.step_ctx = step_ctx
            super().__init__(*args, **kwargs)
            if step_ctx is not None:     # lets --hold-complete verify the trajectory so far
                step_ctx.current_traj = lambda: getattr(self, "trajectory_data", None)

        def _is_complete_action(self, response):
            if not getattr(self.step_ctx, "hold_complete", False):
                return super()._is_complete_action(response)
            # AgentWorld matches any "complete(" in the response (thoughts and messages
            # included). With --hold-complete only a completion the wrapper let through ends
            # the agent; a held complete() is an ordinary (failed) action.
            return "[TOOL_RESULT] TASK_COMPLETE:" in (response or "")

        def _initialize_agents(self, task_config):
            states = super()._initialize_agents(task_config)
            self.step_ctx.on_agents_ready(states)
            return states

        def _apply_comm_phase(self, agent_states, round_count):
            # Called at the top of every round, before any agent acts: apply the
            # previous round's plasticity update so every agent sees the same W.
            self.step_ctx.flush(self.trajectory_data)
            super()._apply_comm_phase(agent_states, round_count)

        def _collect_action_data(self, agent_name, agent_state, response):
            data = super()._collect_action_data(agent_name, agent_state, response)
            round_num = len(self.trajectory_data["rounds"]) + 1
            self.step_ctx.on_action(round_num, agent_name, response,
                                    data.get("observation"), self.trajectory_data)
            return data

        def _save_task_summary(self, task_config, results):
            self.step_ctx.flush(self.trajectory_data)
            super()._save_task_summary(task_config, results)

    return ChainTaskRunner


def reset_runner_logger() -> None:
    """Each TaskRunner adds file handlers to the shared 'TaskRunner' logger; drop the last step's."""
    lg = logging.getLogger("TaskRunner")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ── one step ─────────────────────────────────────────────────────────────────
def run_step(gen, run_dir: Path, args, experiment, store, verification, score_psr,
             runner_cls, hebbian: Dict[str, Any]) -> Dict[str, Any]:
    from wired_state import ChainRunError, StepContext
    step = gen.step
    step_dir = run_dir / step.dir_name
    if step_dir.exists() and any(step_dir.iterdir()):   # an attempt that crashed before saving
        step_dir.rename(step_dir.with_name(f"{step_dir.name}.failed_{int(time.time())}"))
    step_dir.mkdir(parents=True, exist_ok=True)
    task_yaml = gen.task_yaml

    def verify(traj):
        return verification.verify_trajectory(traj, task_path=str(task_yaml))

    ctx = StepContext(step=step.index, task_key=step.task_stem, task_yaml=task_yaml,
                      verifier=gen.verifier,
                      slots={a.slot: {"agent_id": a.agent_id, "role": a.role,
                                      "username": a.username} for a in step.assignments},
                      store=store, arm=args.arm, verify=verify, hebbian=hebbian,
                      log_path=step_dir / "wired_rounds.jsonl",
                      hold_complete=getattr(args, "hold_complete", False))
    reset_runner_logger()
    runner = runner_cls(args.agent, str(step_dir), args.split_screen, args.dump_prompts,
                        experiment, step_ctx=ctx)
    t0 = time.time()
    results = runner.run_single_task(str(task_yaml))
    if isinstance(results, dict) and results.get("error"):
        raise ChainRunError(f"step {step.index}: {results['error']}: {results.get('details')}")
    traj = runner.trajectory_data
    ok, msg = verify(traj)
    metrics = traj.get("metrics") or {}
    psr = psr_for(score_psr, step.task_stem, msg, ok)
    outcome = {"step": step.index, "task": step.task_stem, "success": bool(ok), "message": msg,
               "psr": psr.get("PSR"), "rounds": metrics.get("total_rounds"),
               "chats": metrics.get("total_chats")}
    saved = ctx.end(outcome)
    trajectories = sorted(step_dir.glob("*_trajectory.json"))
    return {
        "step": step.index, "dir": step.dir_name, "task_key": step.task_stem,
        "source": str(step.source), "task_yaml": str(task_yaml), "verifier": str(gen.verifier),
        "slots": [a.to_dict() for a in step.assignments],
        "success": bool(ok), "verifier_message": msg,
        "held_completes": ctx.held_completes,
        "runner_verified": metrics.get("task_verified_success"),
        "runner_message": metrics.get("task_verification_message"),
        "psr": psr, "rounds": metrics.get("total_rounds"), "max_rounds": metrics.get("max_rounds"),
        "chats": metrics.get("total_chats"), "actions": metrics.get("total_actions"),
        "early_stopped": metrics.get("early_stopped"), "duration_s": round(time.time() - t0, 1),
        "trajectory": str(trajectories[0]) if trajectories else None,
        "rounds_learned": ctx.rounds_learned,
        "bonds_end": {s.agent_id: {k: round(v, 4) for k, v in s.bonds.items()}
                      for s in ctx.sessions.values()},
        "state": {"loaded": ctx.loaded, "saved": saved},
    }


def totals(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [s for s in steps if not s.get("error")]
    if not done:
        return {"steps": 0}
    psr = [s["psr"]["PSR"] for s in done if s.get("psr", {}).get("PSR") is not None]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    return {"steps": len(done), "SR_percent": round(100 * mean([float(s["success"]) for s in done]), 2),
            "PSR_percent": round(100 * mean(psr), 2) if psr else None,
            "PSR_scored": len(psr), "rounds_mean": mean([s.get("rounds") for s in done]),
            "chats_mean": mean([s.get("chats") for s in done])}


# ── main ─────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from chain_spec import resolve_root
    root = resolve_root(args.agentworld_root)
    if root is None or not (root / "agents" / "run.py").is_file():
        raise SystemExit("set --agentworld-root or $AGENTWORLD_ROOT to an AgentWorld checkout")
    if args.no_roles:
        raise SystemExit("--no-roles replaces every name with Agent_N, which breaks persistent "
                         "identity; chains do not support it")
    if args.single_agent_upper_bound:
        raise SystemExit("--single-agent-upper-bound merges the team into one agent; "
                         "chains need the whole team")
    bootstrap(root)
    os.environ.setdefault("AGENTWORLD_ROOT", str(root))

    from chain_spec import load_chain
    from config_loader import load_agent_config          # AgentWorld
    from experiment import ExperimentConfig              # AgentWorld
    from generate_tasks import generate_chain, load_verification
    from wired_state import AgentStateStore
    import run as aw_run                                  # AgentWorld's agents/run.py

    cfg = load_agent_config(args.agent)["agent"]
    target = str((cfg.get("llm") or {}).get("agent_class", ""))
    if cfg.get("provider") != "custom" or not target.startswith("wired_agent:"):
        raise SystemExit(f"{args.agent}: use provider: custom with llm.agent_class: "
                         "wired_agent:<Class> (see chains/configs/)")
    experiment = ExperimentConfig.from_args(args)
    experiment.validate()

    chain = load_chain(args.chain, root)
    wanted = parse_steps(args.steps, len(chain.steps))
    run_id = args.run_id or (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{args.arm}"
                             + (f"_s{args.seed}" if args.seed is not None else ""))
    run_dir = (args.output / chain.name / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    store = AgentStateStore(run_dir / "agent_state")
    for step in chain.steps:                  # a step's state must not already contain it
        if step.index not in wanted:
            continue
        for a in step.assignments:
            done = store.load(a.agent_id).get("saved_after_step")
            if done is not None and done >= step.index:
                raise SystemExit(f"step {step.index} already ran in {run_dir} ({a.agent_id} was saved "
                                 f"after step {done}); use a new --run-id")
    generated = generate_chain(chain, root, run_dir / "tasks")
    verification = load_verification(root)
    score_psr = _load_file_module("agentworld_score_psr", root / "scripts" / "score_psr.py")
    runner_cls = chain_runner_class(aw_run.TaskRunner)
    hebbian = {k: getattr(args, k) for k in (
        "hebbian_eta_0", "hebbian_eta_plus", "hebbian_decay", "hebbian_eligibility_rho",
        "hebbian_coact_floor", "hebbian_radius", "hebbian_reward_norm", "hebbian_death_ltd",
        "hebbian_init_weight", "hebbian_salience", "cofiring_channels", "xfer_directed")}

    summary_path = run_dir / "chain_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update({
        "chain": chain.name, "run_id": run_id, "chain_file": str(chain.path),
        "description": chain.description, "arm": args.arm, "hold_complete": bool(args.hold_complete),
        "agent_config": str(Path(args.agent).resolve()),
        "agent_class": target, "agentworld_root": str(root), "agentworld_commit": git_commit(root),
        "flags": {k: getattr(args, k) for k in (
            "discussion_rounds", "no_communication", "shared_plan_only", "random_agent",
            "max_rounds", "random_spawn", "spawn_region", "no_task_docs", "seed")},
        "hebbian": hebbian, "agent_ids": chain.agent_ids,
        "metrics": "SR, PSR (AgentWorld checkpoint_fraction_v1), rounds, chats; CCE post hoc",
    })
    steps: Dict[int, Dict[str, Any]] = {s["step"]: s for s in summary.get("steps") or []}

    def save():
        summary["steps"] = [steps[k] for k in sorted(steps)]
        summary["totals"] = totals(summary["steps"])
        summary["updated"] = datetime.now().isoformat(timespec="seconds")
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    status = 0
    for gen in generated:
        if gen.step.index not in wanted:
            continue
        names = ", ".join(a.username for a in gen.step.assignments)
        print(f"\n=== {chain.name} step {gen.step.index}/{len(chain.steps)}: "
              f"{gen.step.task_stem} [{names}] ===", flush=True)
        try:
            steps[gen.step.index] = run_step(gen, run_dir, args, experiment, store, verification,
                                             score_psr, runner_cls, hebbian)
        except BaseException as exc:          # record, then stop: later state would be wrong
            steps[gen.step.index] = {"step": gen.step.index, "dir": gen.step.dir_name,
                                     "task_key": gen.step.task_stem,
                                     "slots": [a.to_dict() for a in gen.step.assignments],
                                     "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()}
            save()
            print(f"step {gen.step.index} failed: {exc}", file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                raise
            status = 1
            break
        s = steps[gen.step.index]
        print(f"step {s['step']}: {'SUCCESS' if s['success'] else 'FAILED'} - {s['verifier_message']} "
              f"| PSR {s['psr'].get('PSR')} | rounds {s['rounds']} | chats {s['chats']}", flush=True)
        save()

    if not args.no_check and status == 0:
        from check_chain_run import report, run_checks
        summary["checks"] = run_checks(run_dir)
        save()
        print("\n" + report(summary["checks"]))
        status = 0 if summary["checks"]["ok"] else 1
    print(f"\nrun: {run_dir}")
    return status


if __name__ == "__main__":
    sys.exit(main())
