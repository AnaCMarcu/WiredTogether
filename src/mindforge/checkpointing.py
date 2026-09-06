"""Checkpoint save/restore for chained cluster jobs.

A checkpoint is a directory holding the run state (episode/step/run id),
the metric recorder, the Hebbian graph, per-agent LoRA adapters and
cognitive state (curriculum, skills, episodic memory), so a job that hits
its wall-clock limit can be resumed by the next one with ``--resume``.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from mindforge.agent_modules.craftium_metric import CraftiumMetric
from rl_layer import HebbianSocialGraph


def save_checkpoint(
    checkpoint_dir: str,
    episode: int,
    step: int,
    run_id: str,
    args,
    metric: CraftiumMetric,
    agents,
    hebbian_graph: HebbianSocialGraph,
    frames_list=None,
    save_frames: bool = False,
    global_step: int = 0,
) -> None:
    """Serialize full run state to *checkpoint_dir* so a new SLURM job can resume.

    Files written:
      run_state.json         — scalar counters, CLI args, metric dicts
      hebbian_graph.json     — Hebbian weight matrix + config
      rl_agent_{i}/          — RL LoRA weights + optimizer (via rl_layer.save())
      agent_{i}_curriculum.json — curriculum task lists + current task/context
      frames_{i}.npy         — raw observation arrays (optional, --checkpoint-frames)

    The function is wrapped in try/except so a serialization error never kills the run.
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- run_state.json -------------------------------------------------
        metric_dict = {
            "num_agents": metric.num_agents,
            "communication": metric.communication,
            "run_id": metric.run_id,
            "timestep": metric.timestep,
            "cumulative_returns": [float(x) for x in metric.cumulative_returns],
            "episode_returns": [float(x) for x in metric.episode_returns],
            "per_episode_returns": [
                [float(x) for x in ep_list]
                for ep_list in metric.per_episode_returns
            ],
            "track_rewards_episode": {
                str(i): dict(metric.track_rewards_episode[i])
                for i in range(metric.num_agents)
            },
            "track_rewards_per_episode": [
                [dict(d) for d in agent_eps]
                for agent_eps in metric.track_rewards_per_episode
            ],
            "agent_milestones_episode": {
                str(i): sorted(metric._agent_milestones_episode[i])
                for i in range(metric.num_agents)
            },
            "milestones_per_episode": [
                [list(ms) for ms in agent_eps]
                for agent_eps in metric.milestones_per_episode
            ],
            "comm_count_episode": list(metric.comm_count_episode),
            "comm_count_per_episode": [
                list(c) for c in metric.comm_count_per_episode
            ],
            "episode_lengths": list(metric.episode_lengths),
            "comm_counts_per_step": list(metric.comm_counts_per_step),
            "communication_log": metric.communication_log,
            "rl_updates": metric.rl_updates,
            "rl_token_opts": metric.rl_token_opts,
            "milestones_per_agent": {name: sorted(ms) for name, ms in metric._agent_milestones.items()},
            "track_rewards": metric.track_rewards,
            "_graph_snapshots": metric._graph_snapshots,
            "ts_data": metric.ts_data,
            "phase_transitions": getattr(metric, "phase_transitions", []),
            "team_mode": getattr(metric, "team_mode", "heterogeneous"),
            "homogeneous_role": getattr(metric, "homogeneous_role", "agent"),
        }
        run_state = {
            "episode": episode,
            "step": step,
            "run_id": run_id,
            "metric": metric_dict,
            "cli_args": vars(args),
            "global_step": global_step,
            # Team composition
            "team_mode": getattr(metric, "team_mode", "heterogeneous"),
            "homogeneous_role": getattr(metric, "homogeneous_role", "agent"),
        }
        with open(os.path.join(checkpoint_dir, "run_state.json"), "w") as f:
            json.dump(run_state, f, indent=2, default=str)

        # --- hebbian_graph.json ---------------------------------------------
        with open(os.path.join(checkpoint_dir, "hebbian_graph.json"), "w") as f:
            json.dump(hebbian_graph.to_dict(), f, indent=2)

        # --- per-agent RL weights + optimizer --------------------------------
        for i, agent in enumerate(agents):
            if agent.rl_layer and agent.rl_layer.enabled:
                rl_save_dir = os.path.join(checkpoint_dir, f"rl_agent_{i}")
                os.makedirs(rl_save_dir, exist_ok=True)
                agent.rl_layer.save(path=rl_save_dir)

        # --- per-agent curriculum state -------------------------------------
        for i, agent in enumerate(agents):
            cur = agent.auto_curriculum
            curriculum_state = {
                "current_task": cur.current_task,
                "current_context": getattr(cur, "current_context", ""),
                "completed_tasks": list(cur.completed_tasks),
                "failed_tasks": list(cur.failed_tasks),
            }
            with open(os.path.join(checkpoint_dir, f"agent_{i}_curriculum.json"), "w") as f:
                json.dump(curriculum_state, f, indent=2)

        # --- per-agent cognitive state (skills / episodic memory) -----------
        # The vector DBs live on job-local /tmp under SLURM; this JSON copy in
        # the checkpoint dir is the durable form (used by merge_pair_runs.py).
        from mindforge.agent_modules.agent_state_io import export_agent_state
        export_agent_state(
            agents, os.path.join(checkpoint_dir, "agent_state"),
            episode=episode,
        )

        # --- optional frames ------------------------------------------------
        if save_frames and frames_list:
            for i in range(len(agents)):
                agent_frames = [f[i] for f in frames_list if f[i] is not None]
                if agent_frames:
                    frames_path = os.path.join(checkpoint_dir, f"frames_{i}.npy")
                    np.save(frames_path, np.stack(agent_frames, axis=0))

        print(f"[CKPT] Saved checkpoint ep={episode} step={step} → {checkpoint_dir}")

    except Exception as exc:
        logging.warning(f"[CKPT] save_checkpoint failed (ep={episode} step={step}): {exc}")


def load_checkpoint(
    checkpoint_dir: str,
    agents,
    hebbian_graph: HebbianSocialGraph,
    metric_path: str = "./run_metrics",
    run_paths=None,
) -> dict:
    """Restore run state from *checkpoint_dir*.

    Returns a dict with keys:
      episode  — last fully-checkpointed episode index
      step     — last checkpointed step within that episode
      run_id   — original run ID
      metric   — restored CraftiumMetric instance

    RL weights, optimizer, and Hebbian graph are restored in-place.
    Curriculum state is restored into each agent's auto_curriculum.

    When ``run_paths`` is supplied, the restored metric writes to that
    consolidated tree (``runs/<run_id>/``). Otherwise ``metric_path`` is
    used and the legacy ``./run_metrics/<run_id>/`` folder is created.
    """
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")
    if not os.path.exists(run_state_path):
        raise FileNotFoundError(f"[CKPT] No run_state.json in {checkpoint_dir}")

    with open(run_state_path, "r") as f:
        run_state = json.load(f)

    episode = run_state["episode"]
    step = run_state["step"]
    run_id = run_state.get("run_id", "resumed")

    # Restore metric — use run_paths when provided so resumed runs keep
    # writing into the same `runs/<run_id>/` tree as the live process.
    metric = CraftiumMetric.restore_from_dict(
        run_state["metric"], path=metric_path, run_paths=run_paths,
    )

    # Restore Hebbian graph
    hebbian_path = os.path.join(checkpoint_dir, "hebbian_graph.json")
    if os.path.exists(hebbian_path):
        with open(hebbian_path, "r") as f:
            hebbian_dict = json.load(f)
        hebbian_graph.from_dict(hebbian_dict)
        print(f"[CKPT] Restored Hebbian graph from {hebbian_path}")
    else:
        logging.warning(f"[CKPT] No hebbian_graph.json in {checkpoint_dir}, graph untouched")

    # Restore per-agent RL state
    for i, agent in enumerate(agents):
        rl_save_dir = os.path.join(checkpoint_dir, f"rl_agent_{i}")
        if agent.rl_layer and agent.rl_layer.enabled and os.path.isdir(rl_save_dir):
            agent.rl_layer.load(path=rl_save_dir)
            print(f"[CKPT] Restored RL state for agent_{i} from {rl_save_dir}")

    # Restore per-agent curriculum state
    for i, agent in enumerate(agents):
        cur_path = os.path.join(checkpoint_dir, f"agent_{i}_curriculum.json")
        if os.path.exists(cur_path):
            with open(cur_path, "r") as f:
                cur_state = json.load(f)
            cur = agent.auto_curriculum
            cur.current_task = cur_state.get("current_task")
            cur.current_context = cur_state.get("current_context", "")
            cur.completed_tasks = list(cur_state.get("completed_tasks", []))
            cur.failed_tasks = list(cur_state.get("failed_tasks", []))
            print(f"[CKPT] Restored curriculum for agent_{i}: task={cur.current_task!r}")

    metric._global_step_ckpt = run_state.get("global_step", 0)

    print(f"[CKPT] Loaded checkpoint: ep={episode} step={step} run_id={run_id}")
    return {"episode": episode, "step": step, "run_id": run_id, "metric": metric}
