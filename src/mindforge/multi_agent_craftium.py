"""Main loop for running Mindforge agents in Five Chambers (Craftium).

Homogeneous agents progress cooperatively through five enclosed chambers,
using LLM-based action selection, episodic memory, belief systems, and auto-curriculum.
Milestones M1–M28 are tracked via JSONL polling from the Lua mod.

Usage:
    cd src/mindforge
    python multi_agent_craftium.py
    python multi_agent_craftium.py --num-agents 6 --episodes 3 --max-steps 1000
"""

import argparse
import asyncio
import math
import os
import random
import sys
import time
import logging

sys.setrecursionlimit(10000)
from datetime import datetime

import numpy as np
import PIL
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import CancellationToken, Image
from autogen_core import EVENT_LOGGER_NAME

from custom_agent import CustomAgent
from custom_environment_craftium import CraftiumEnvironmentInterface, VALID_ACTIONS
from agent_modules.action_selection import ActionSelection
from agent_modules.auto_curriculum import AutoCurriculum
from agent_modules.belief_system import BeliefSystem
from agent_modules.critic import Critic
from agent_modules.skill_manager import SkillManager
from agent_modules.episodic_memory_manager import EpisodicMemoryManager
from agent_modules.craftium_metric import CraftiumMetric
from mindforge.env.communication_rewards import CommunicationTracker
from mindforge.env.cooperation_metric import CooperationMetric
from mindforge.env.episode_logger import EpisodeLogger
from mindforge.run_layout import RunPaths
import json as _json

from rl_layer import RLConfig, RLLayer, HebbianConfig, HebbianSocialGraph

ROLE_NAMES = ["agent"]

# Macro action names — used to detect when a macro was selected so the RL
# buffer defers store_reward() until the macro completes.
_MACRO_NAMES = frozenset({"TurnAround", "ScanArea", "ApproachTarget", "Escape"})


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mindforge agents in Craftium OpenWorld")
    parser.add_argument("--num-agents", type=int, default=3,
                        help="Number of agents in five-chambers (all share the agent role)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--obs-width", type=int, default=320,
                        help="Observation width in pixels")
    parser.add_argument("--obs-height", type=int, default=180,
                        help="Observation height in pixels")
    parser.add_argument("--no-communication", action="store_true",
                        help="Disable inter-agent communication entirely.")
    parser.add_argument("--sleep-time", type=float, default=0.0,
                        help="Seconds to sleep between LLM calls (rate-limit protection)")
    parser.add_argument("--belief-interval", type=int, default=5,
                        help="Refresh beliefs every N steps (default 5). Between refreshes "
                             "cached beliefs are reused, saving 4 LLM calls per skipped step.")
    parser.add_argument("--critic-interval", type=int, default=20,
                        help="Run critic every N steps (default 20). Between evaluations "
                             "cached success/critique are reused, saving 1 LLM call per skipped step.")
    parser.add_argument("--no-gif", action="store_true",
                        help="Disable GIF saving")
    parser.add_argument("--gif-dir", type=str, default="gifs",
                        help="Directory to save GIFs (default: gifs/ relative to cwd). "
                             "On HPC set to an absolute path under /scratch.")
    parser.add_argument("--gif-interval", type=int, default=100,
                        help="Save a checkpoint GIF every N steps (default 100). 0 = only save at episode end.")
    parser.add_argument("--warmup-time", type=int, default=60,
                        help="Minimum seconds before checking if media loaded (default 60). "
                             "Smart detection exits early once all clients show game world.")
    # ── Reproducibility ──
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. Seeds torch, numpy, random, "
                             "and the Minetest world. LLM sampling remains stochastic — "
                             "run multiple trials and report mean/std.")
    # ── RL layer ──
    parser.add_argument("--rl", action="store_true",
                        help="Enable the modular RL layer (action-level MAPPO)")
    parser.add_argument("--rl-model-path", type=str, default=None,
                        help="Path to base model for RL (e.g. /scratch/.../Qwen3.5-2B)")
    parser.add_argument("--rl-lora-rank", type=int, default=8,
                        help="LoRA rank for RL adapter")
    parser.add_argument("--rl-update-interval", type=int, default=256,
                        help="Steps between MAPPO updates")
    parser.add_argument("--rl-lr", type=float, default=1e-4,
                        help="Learning rate for RL optimiser")
    parser.add_argument("--rl-auto-token-opt", action="store_true",
                        help="Let agents self-trigger token-level optimisation")
    parser.add_argument("--rl-mode", type=str, default="action",
                        choices=["action", "token"],
                        help="RL mode: 'action' = MAPPO action head, "
                             "'token' = token-opt only (LLM picks actions)")
    parser.add_argument("--rl-critic-mode", type=str, default="centralized",
                        choices=["centralized", "independent"],
                        help="Critic architecture for action-mode RL. "
                             "'centralized' (default) = shared V(joint_state) critic across "
                             "all agents (true MAPPO). "
                             "'independent' = legacy per-agent value head on per-agent LLM "
                             "hidden state (IPPO).")
    parser.add_argument("--rl-prompt-max-tokens", type=int, default=512,
                        help="Max tokens for RL prompt encoding. Capping this is critical "
                             "for VRAM: at model_max_length=32768 a mini-batch of 8 prompts "
                             "needs ~21 GB just for hidden states. 512 is sufficient for "
                             "discrete action policy learning.")
    # ── Hebbian social plasticity ──
    parser.add_argument("--hebbian", action="store_true",
                        help="Enable Hebbian social plasticity graph")
    parser.add_argument("--hebbian-radius", type=float, default=5.0,
                        help="Interaction radius d (Minetest world units)")
    parser.add_argument("--hebbian-ltp", type=float, default=0.01,
                        help="η_+ LTP learning rate")
    parser.add_argument("--hebbian-ltd", type=float, default=0.005,
                        help="η_- LTD learning rate")
    parser.add_argument("--hebbian-decay", type=float, default=0.005,
                        help="λ passive decay rate")
    parser.add_argument("--hebbian-beta", type=float, default=1.0,
                        help="β modulation sensitivity")
    parser.add_argument("--hebbian-rho", type=float, default=0.3,
                        help="ρ social replay blend factor")
    parser.add_argument("--hebbian-gamma", type=float, default=0.2,
                        help="γ reward diffusion strength")
    parser.add_argument("--hebbian-init-weight", type=float, default=0.1,
                        help="Initial bond weight W_0 (default 0.1 = warm start)")
    parser.add_argument("--hebbian-no-comm-bond", action="store_true",
                        help="Set δ_comm=0 (spatial-only, for RQ4 ablation)")
    # ── Experiment tracking ──
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment identifier (e.g. E1a, E5) — saved in metrics for traceability")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print a reward/metric summary every N steps (default 10)")
    # ── Team composition (five-chambers: all agents are homogeneous) ──
    parser.add_argument(
        "--team-mode",
        type=str,
        default="homogeneous-agent",
        choices=["homogeneous-agent"],
        help="All agents share the same 'agent' role in five-chambers.",
    )
    parser.add_argument(
        "--homogeneous-role",
        type=str,
        default="agent",
        choices=["agent"],
        help="Role for all agents (fixed: agent).",
    )
    # ── Phased difficulty ──
    parser.add_argument("--survival-mode", action="store_true",
                        help="Enable the phased difficulty system. Without this flag "
                             "the run stays in exploration phase (current behavior).")
    parser.add_argument("--survival-episode", type=int, default=1,
                        help="Switch to survival at the start of this episode (1-indexed). "
                             "Only active when --survival-mode is set. (default: 1)")
    parser.add_argument("--survival-step", type=int, default=None,
                        help="Switch to survival at this cumulative global step count. "
                             "Whichever of --survival-episode / --survival-step triggers "
                             "first wins. Only active when --survival-mode is set.")
    parser.add_argument("--survival-gradual", action="store_true",
                        help="Ramp difficulty: enable mobs first, then hunger "
                             "--survival-gradual-delay steps later. "
                             "Only active when --survival-mode is set.")
    parser.add_argument("--survival-gradual-delay", type=int, default=500,
                        help="Steps between mobs-only and full survival in gradual mode "
                             "(default: 500). Only active when --survival-gradual is set.")
    # ── Checkpoint / resume ──
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to write checkpoints into. "
                             "Default: ./checkpoints/<run_id>")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Save a checkpoint every N steps within an episode (default 500)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory from a previous job. "
                             "Restores cognitive/RL/Hebbian state and continues from saved ep/step.")
    parser.add_argument("--resume-skip-warmup", action="store_true",
                        help="Skip the media-load warmup detection on resume "
                             "(use when VoxeLibre media cache is already populated)")
    parser.add_argument("--checkpoint-frames", action="store_true",
                        help="Include raw frames in the checkpoint for GIF continuity. "
                             "Off by default as frame arrays can be large.")
    return parser.parse_args()


def load_prompts():
    """Load all prompt files and return them as a dict."""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    belief_dir = os.path.join(prompt_dir, "belief_system")

    def _read(path):
        with open(path, "r") as f:
            return f.read()

    prompts = {
        "environment": _read(os.path.join(prompt_dir, "environment_prompt.txt")),
        "system_template": _read(os.path.join(prompt_dir, "system_prompt.txt")),
        "instruction": _read(os.path.join(prompt_dir, "instruction_prompt_p2.txt")),
        "critic": _read(os.path.join(prompt_dir, "critic_prompt.txt")),
        "curriculum_questions": _read(os.path.join(prompt_dir, "curriculum_questions.txt")),
        "skill_description": _read(os.path.join(prompt_dir, "skill_description_prompt.txt")),
        "skill_info": _read(os.path.join(prompt_dir, "skill_description_info.txt")),
        "perception": _read(os.path.join(belief_dir, "perception_beliefs.txt")),
        "partner": _read(os.path.join(belief_dir, "partner_beliefs.txt")),
        "interaction": _read(os.path.join(belief_dir, "interaction_belief.txt")),
        "context": _read(os.path.join(belief_dir, "update_context.txt")),
    }

    # Role prompts
    prompts["roles"] = {}
    for role in ROLE_NAMES:
        prompts["roles"][role] = _read(os.path.join(prompt_dir, f"role_{role}.txt"))

    return prompts


def build_role_configs(
    num_agents,
    role_prompts,
    team_mode="homogeneous-agent",
    homogeneous_role="agent",
):
    """Build ROLE_CONFIGS for num_agents. All agents share the 'agent' role."""
    role_name = "agent"
    return [
        {
            "name": role_name,
            "agent_name": f"agent_{i}",
            "curriculum_prompt": role_prompts[role_name].format(num_agents=num_agents),
        }
        for i in range(num_agents)
    ]


def build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                 rl_config=None, belief_interval=5, critic_interval=20,
                 centralized_critic=None):
    """Initialize all Mindforge agents.

    ``centralized_critic`` (when not None) is shared by all agents' RLLayers
    and turns the value-loss off in their PPO updates.
    """
    agents = []
    for i, role_cfg in enumerate(role_configs):
        # Build per-agent RL layer (no-op when rl_config.enabled is False)
        rl_layer = None
        if rl_config and rl_config.enabled:
            rl_layer = RLLayer(
                config=rl_config, role=role_cfg["name"], agent_id=i,
                centralized_critic=centralized_critic,
            )

        # Targeted communication policy lives in the static prompts. The LLM
        # uses its own agent name (passed in via the user message) to exclude
        # itself from the recipient list.
        agent_system_prompt = system_prompt

        agent = CustomAgent(
            name=role_cfg["agent_name"],
            description=f"{role_cfg['name']} agent in Craftium open world",
            action_selection=ActionSelection(system_prompt=agent_system_prompt),
            auto_curriculum=AutoCurriculum(
                override_curriculum_prompt=role_cfg["curriculum_prompt"],
                override_questions_prompt=prompts["curriculum_questions"],
                agent_name=role_cfg["agent_name"],
            ),
            critic=Critic(override_critic_prompt=prompts["critic"]),
            skill_manager=SkillManager(
                override_skill_prompt=prompts["skill_description"],
                override_skill_info_prompt=prompts["skill_info"],
                agent_name=role_cfg["agent_name"],
            ),
            episode_manager=EpisodicMemoryManager(
                agent_name=role_cfg["agent_name"],
            ),
            belief_system=BeliefSystem(
                number_of_agents=num_agents,
                override_perception_prompt=prompts["perception"],
                override_partner_prompt=prompts["partner"],
                override_interaction_prompt=prompts["interaction"],
                override_context_prompt=prompts["context"],
            ),
            number_of_agents=num_agents,
            metric=metric,
            voyager=False,
            rl_layer=rl_layer,
            belief_interval=belief_interval,
            critic_interval=critic_interval,
            num_agents=num_agents,
        )
        agents.append(agent)
        rl_status = " [RL enabled]" if rl_layer else ""
        print(f"Initialized agent {i}: {role_cfg['agent_name']} ({role_cfg['name']}){rl_status}")
    return agents


# ===========================
# Agent action loop
# ===========================
async def agent_do_action(
    agent,
    agent_id: int,
    frame_image,
    communications: list,
    reward_text: str,
    instruction_prompt,
    environment,
    error=None,
    error_count=0,
    social_bonds=None,
    position_text=None,
    player_status_text=None,
    current_chamber=None,
    completed_milestones=None,
):
    """Have one agent observe and choose an action.

    Returns:
        (content_dict, last_action_str, error_count)
    """
    formatted_communication = [
        f"{msg.source}: {msg.content}"
        for msg in communications
        if msg.source != agent.name
    ]

    # Don't pre-fill the instruction template here — action_selection.select_action()
    # will fill it once with real cognitive data (beliefs, skills, episodes) via llm_call.
    # Only pass communication context as the message content.
    comm_text = f"Communications from other agents: {formatted_communication}.\n"

    multi_modal_message = MultiModalMessage(
        content=[comm_text, Image.from_pil(frame_image)],
        source="user",
    )

    content, error_count = await agent.on_messages(
        [multi_modal_message],
        CancellationToken(),
        communication=formatted_communication,
        error=error,
        error_count=error_count,
        picked_object=environment.pickedup_object(agentId=agent_id),
        reward_text=reward_text,
        social_bonds=social_bonds,
        position_text=position_text,
        player_status_text=player_status_text,
        current_chamber=current_chamber,
        completed_milestones=completed_milestones,
    )

    last_action = "NoOp"
    try:
        action = content.get("action", "NoOp") if content else "NoOp"
        _, last_action = environment.step(action, agentId=agent_id)
    except Exception as e:
        logging.error(f"Error in environment step for agent {agent_id}: {e}")
        if error_count < 5:
            content, last_action, error_count = await agent_do_action(
                agent, agent_id, frame_image, communications, reward_text,
                instruction_prompt, environment,
                error=str(e), error_count=error_count + 1,
                social_bonds=social_bonds,
                position_text=position_text,
                player_status_text=player_status_text,
                current_chamber=current_chamber,
                completed_milestones=completed_milestones,
            )
        else:
            logging.error(f"Agent {agent_id} exceeded retry limit, using NoOp")
            environment.step("NoOp", agentId=agent_id)
            last_action = "NoOp"

    return content, last_action, error_count


def save_checkpoint(
    checkpoint_dir: str,
    episode: int,
    step: int,
    run_id: str,
    args,
    metric: "CraftiumMetric",
    agents,
    hebbian_graph: "HebbianSocialGraph",
    frames_list=None,
    save_frames: bool = False,
    current_phase: str = "exploration",
    global_step: int = 0,
    gradual_trigger_step=None,
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
            # Phase state — restored in run() so a resumed job re-signals the server
            "current_phase": current_phase,
            "global_step": global_step,
            "gradual_trigger_step": gradual_trigger_step,
            # Team composition
            "team_mode": getattr(metric, "team_mode", "heterogeneous"),
            "homogeneous_role": getattr(metric, "homogeneous_role", "agent"),
        }
        with open(os.path.join(checkpoint_dir, "run_state.json"), "w") as f:
            _json.dump(run_state, f, indent=2, default=str)

        # --- hebbian_graph.json ---------------------------------------------
        with open(os.path.join(checkpoint_dir, "hebbian_graph.json"), "w") as f:
            _json.dump(hebbian_graph.to_dict(), f, indent=2)

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
                _json.dump(curriculum_state, f, indent=2)

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
    hebbian_graph: "HebbianSocialGraph",
    metric_path: str = "./run_metrics",
) -> dict:
    """Restore run state from *checkpoint_dir*.

    Returns a dict with keys:
      episode  — last fully-checkpointed episode index
      step     — last checkpointed step within that episode
      run_id   — original run ID
      metric   — restored CraftiumMetric instance

    RL weights, optimizer, and Hebbian graph are restored in-place.
    Curriculum state is restored into each agent's auto_curriculum.
    """
    run_state_path = os.path.join(checkpoint_dir, "run_state.json")
    if not os.path.exists(run_state_path):
        raise FileNotFoundError(f"[CKPT] No run_state.json in {checkpoint_dir}")

    with open(run_state_path, "r") as f:
        run_state = _json.load(f)

    episode = run_state["episode"]
    step = run_state["step"]
    run_id = run_state.get("run_id", "resumed")

    # Restore metric
    metric = CraftiumMetric.restore_from_dict(run_state["metric"], path=metric_path)

    # Restore Hebbian graph
    hebbian_path = os.path.join(checkpoint_dir, "hebbian_graph.json")
    if os.path.exists(hebbian_path):
        with open(hebbian_path, "r") as f:
            hebbian_dict = _json.load(f)
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
                cur_state = _json.load(f)
            cur = agent.auto_curriculum
            cur.current_task = cur_state.get("current_task")
            cur.current_context = cur_state.get("current_context", "")
            cur.completed_tasks = list(cur_state.get("completed_tasks", []))
            cur.failed_tasks = list(cur_state.get("failed_tasks", []))
            print(f"[CKPT] Restored curriculum for agent_{i}: task={cur.current_task!r}")

    # Restore phase state — stashed on metric so run() can pick it up
    metric._current_phase_ckpt = run_state.get("current_phase", "exploration")
    metric._global_step_ckpt   = run_state.get("global_step", 0)
    metric._gradual_trigger_step_ckpt = run_state.get("gradual_trigger_step", None)

    print(f"[CKPT] Loaded checkpoint: ep={episode} step={step} run_id={run_id} "
          f"phase={metric._current_phase_ckpt}")
    return {"episode": episode, "step": step, "run_id": run_id, "metric": metric}


def _frames_to_mp4(pil_frames: list, mp4_path: str, fps: int = 2) -> None:
    """Write PIL frames directly to MP4 using imageio[ffmpeg] (bundled binary, no system ffmpeg)."""
    try:
        import imageio
        with imageio.get_writer(mp4_path, fps=fps, macro_block_size=1) as writer:
            for frame in pil_frames:
                writer.append_data(np.array(frame))
        print(f"  Saved MP4: {mp4_path}")
    except Exception as exc:
        logging.warning("MP4 save failed (%s): %s", mp4_path, exc)


def _should_transition_to_survival(episode: int, global_step: int, args) -> bool:
    """Return True when the run should leave exploration phase.

    episode     — 0-indexed current episode number
    global_step — cumulative step count across all episodes
    args        — parsed CLI namespace

    Returns False immediately when --survival-mode is not set, so existing
    runs with no new flags are completely unaffected.
    """
    if not args.survival_mode:
        return False
    # --survival-step fires on cumulative step count
    if args.survival_step is not None and global_step >= args.survival_step:
        return True
    # --survival-episode fires at start of that episode (1-indexed → 0-indexed)
    if episode + 1 >= args.survival_episode:
        return True
    return False


# ===========================
# Main episode loop
# ===========================
async def run(args):
    num_agents = args.num_agents
    num_episodes = args.episodes
    max_steps = args.max_steps
    obs_width = args.obs_width
    obs_height = args.obs_height
    communication = not args.no_communication
    sleep_time = args.sleep_time
    save_gif = not args.no_gif
    gif_dir = args.gif_dir
    gif_interval = args.gif_interval
    if save_gif:
        os.makedirs(gif_dir, exist_ok=True)

    # ── Reproducibility: seed all RNG sources ──
    # LLM sampling (temperature > 0) is inherently stochastic and not seeded —
    # run multiple trials with the same seed to get statistical reproducibility.
    seed = args.seed
    if seed is not None:
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seeded RNG: seed={seed}")

    # ── Run ID: short unique tag shared by all log/gif/metric filenames ──
    from uuid import uuid4
    _ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _exp = args.experiment_id or ""
    run_id = f"{_exp+'_' if _exp else ''}{_ts}_{uuid4().hex[:6]}"
    print(f"[RUN ID] {run_id}")

    # ── Single root for all run artifacts: runs/<run_id>/ ──
    # Replaces three previously-separate roots (run_metrics/, checkpoints/, logs/).
    run_paths = RunPaths.create(run_id=run_id, root="runs")
    os.makedirs("gifs", exist_ok=True)

    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_logger.disabled = True
    logging.basicConfig(
        level=logging.INFO,
        filename=str(run_paths.log_txt),
        filemode="a",
        format=f"[{run_id}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_interval = args.log_interval

    # Load prompts
    prompts = load_prompts()
    environment_prompt = prompts["environment"]
    instruction_prompt = prompts["instruction"]

    # Build system prompt with environment details baked in
    from agent_modules.util import safe_format
    system_prompt_template = prompts["system_template"]
    system_prompt = safe_format(system_prompt_template, environment_prompt=environment_prompt)

    # Roles & agents
    role_configs = build_role_configs(
        num_agents,
        prompts["roles"],
        team_mode=args.team_mode,
        homogeneous_role=args.homogeneous_role,
    )

    metric = CraftiumMetric(
        num_agents=num_agents,
        communication=communication,
        run_id=run_id,
        run_paths=run_paths,
    )
    # Attach team composition metadata for summary/checkpoint
    metric.team_mode = args.team_mode
    metric.homogeneous_role = args.homogeneous_role

    environment = CraftiumEnvironmentInterface(
        num_agents=num_agents,
        obs_width=obs_width,
        obs_height=obs_height,
        max_steps=max_steps,
        seed=seed,
    )

    # ── RL layer config ──
    # Give each run its own lora_save_dir so concurrent or sequential jobs
    # never share or accidentally load each other's adapters.
    # Layout: rl_checkpoints/<run_id>/<role>
    # On resume: peek at the checkpoint run_state.json NOW (before build_agents)
    # so the RLLayer loads the correct existing adapter instead of init-ing a
    # fresh one that would immediately be overwritten by load_checkpoint().
    _rl_run_id = run_id
    if args.resume:
        _ckpt_state_path = os.path.join(args.resume, "run_state.json")
        if os.path.exists(_ckpt_state_path):
            with open(_ckpt_state_path) as _f:
                _rl_run_id = _json.load(_f).get("run_id", run_id)
            print(f"[RL] Resume: using adapter dir from original run_id={_rl_run_id!r}")
    rl_save_dir = os.path.join("rl_checkpoints", _rl_run_id)
    rl_config = RLConfig(
        enabled=args.rl,
        mode=args.rl_mode,
        model_path=args.rl_model_path,
        lora_rank=args.rl_lora_rank,
        update_interval=args.rl_update_interval,
        lr=args.rl_lr,
        auto_token_opt=args.rl_auto_token_opt,
        rl_prompt_max_tokens=args.rl_prompt_max_tokens,
        lora_save_dir=rl_save_dir,
        critic_mode=args.rl_critic_mode,
    )
    if rl_config.enabled:
        print(f"RL layer ENABLED: model={rl_config.model_path}, "
              f"lora_rank={rl_config.lora_rank}, update_interval={rl_config.update_interval}, "
              f"critic_mode={rl_config.critic_mode}")

    # ── Centralised MAPPO critic (one shared V across all agents) ─────────
    # Built before agents so each RLLayer can hold a reference to it. In
    # 'independent' mode (or when RL is off) we skip construction.
    centralized_critic = None
    if rl_config.enabled and rl_config.mode == "action" \
            and rl_config.critic_mode == "centralized":
        from rl_layer.centralized_critic import CentralizedCritic
        from agent_modules.craftium_metric import MILESTONE_TRACK
        # Use the canonical milestone ID list as the bitmap order so the
        # critic input layout is stable across runs.
        milestone_ids = list(MILESTONE_TRACK.keys())
        centralized_critic = CentralizedCritic(
            num_agents=num_agents,
            config=rl_config,
            milestone_ids=milestone_ids,
        )
        print(f"[MAPPO] Centralised critic ENABLED: joint_dim={centralized_critic.joint_dim}, "
              f"milestones tracked={len(milestone_ids)}")

    agents = build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                         rl_config=rl_config,
                         belief_interval=args.belief_interval,
                         critic_interval=args.critic_interval,
                         centralized_critic=centralized_critic)

    # ── Hebbian social plasticity ──
    hebbian_config = HebbianConfig(
        enabled=args.hebbian,
        num_agents=num_agents,
        interaction_radius=args.hebbian_radius,
        ltp_lr=args.hebbian_ltp,
        ltd_lr=args.hebbian_ltd,
        decay=args.hebbian_decay,
        modulation_beta=args.hebbian_beta,
        social_replay_rho=args.hebbian_rho,
        reward_diffusion_gamma=args.hebbian_gamma,
        communication_coactivity_bonus=0.0 if args.hebbian_no_comm_bond else 0.5,
        init_weight=args.hebbian_init_weight,
    )
    agent_roles = [ROLE_NAMES.index(rc["name"]) for rc in role_configs]
    hebbian_graph = HebbianSocialGraph(hebbian_config, agent_roles=agent_roles)
    if hebbian_config.enabled:
        print(f"Hebbian social plasticity ENABLED: ltp={hebbian_config.ltp_lr}, "
              f"ltd={hebbian_config.ltd_lr}, radius={hebbian_config.interaction_radius}, "
              f"γ={hebbian_config.reward_diffusion_gamma}")

    comm_mode = "off" if not communication else "targeted"
    print(f"\nConfig: {num_agents} agents, {num_episodes} episodes, "
          f"{max_steps} max steps, comm={comm_mode}, "
          f"seed={seed}")

    # ── Feature activation summary ─────────────────────────────────────────
    _feat_sep = "─" * 60
    print(f"\n{_feat_sep}")
    print(f"[FEATURES] Team mode:        {args.team_mode}  ({num_agents} agents)")
    if args.survival_mode:
        if args.survival_step is not None:
            _surv_trigger = f"global_step >= {args.survival_step}"
        else:
            _surv_trigger = f"episode >= {args.survival_episode}"
        _surv_type = "gradual (mobs first, hunger later)" if args.survival_gradual else "immediate full survival"
        print(f"[FEATURES] Survival mode:    ENABLED — {_surv_type}  trigger: {_surv_trigger}")
    else:
        print(f"[FEATURES] Survival mode:    OFF  (exploration only — mobs passive, hunger frozen)")
    if hebbian_config.enabled:
        print(f"[FEATURES] Hebbian:          ENABLED  ltp={hebbian_config.ltp_lr}  "
              f"ltd={hebbian_config.ltd_lr}  gamma={hebbian_config.reward_diffusion_gamma}  "
              f"radius={hebbian_config.interaction_radius}  decay={hebbian_config.decay}")
        print(f"[FEATURES] Proximity bonus:  ENABLED  +0.3/pair/step within "
              f"{hebbian_config.interaction_radius} blocks")
    else:
        print(f"[FEATURES] Hebbian:          OFF")
        print(f"[FEATURES] Proximity bonus:  OFF  (requires --hebbian)")
    print(f"[FEATURES] Dig reward fixes: stage-gated dig_stage_res + diminishing returns active (Lua)")
    print(f"{_feat_sep}\n")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Checkpoint directory: lives under runs/<run_id>/checkpoints/ ──
    checkpoint_dir = args.checkpoint_dir or str(run_paths.checkpoints_dir)
    checkpoint_interval = args.checkpoint_interval
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[CKPT] Checkpoint directory: {checkpoint_dir}")

    # ── config.json snapshot: durable record of how this run was launched ──
    try:
        import subprocess as _sp
        _git = _sp.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        _git = None
    with open(run_paths.config_json, "w") as _f:
        _json.dump({
            "run_id": run_id,
            "start_ts": datetime.now().isoformat(),
            "git_commit": _git,
            "cli_args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in vars(args).items()},
            "num_agents": num_agents,
            "communication_mode": communication,
        }, _f, indent=2)

    # ── Signal handler: gracefully save on SIGTERM / SIGINT ──
    import signal as _signal
    _shutdown_requested = False
    _shutdown_episode = 0
    _shutdown_step = 0

    def _handle_shutdown(signum, frame):
        nonlocal _shutdown_requested
        sig_name = "SIGTERM" if signum == _signal.SIGTERM else "SIGINT"
        print(f"\n[CKPT] {sig_name} received — will checkpoint at end of current step.")
        _shutdown_requested = True

    _signal.signal(_signal.SIGTERM, _handle_shutdown)
    _signal.signal(_signal.SIGINT, _handle_shutdown)

    # ── Resume: restore state from a previous checkpoint ──
    resume_episode = 0   # first episode to run (0-indexed)
    resume_step = 0      # unused currently — always restart episode from step 0
    if args.resume:
        print(f"[CKPT] Resuming from {args.resume}")
        restored = load_checkpoint(
            checkpoint_dir=args.resume,
            agents=agents,
            hebbian_graph=hebbian_graph,
        )
        resume_episode = restored["episode"]
        resume_step = restored["step"]
        run_id = restored["run_id"]
        metric = restored["metric"]
        # Patch rl_config so adapters are loaded from the original run's directory
        rl_config.lora_save_dir = os.path.join("rl_checkpoints", run_id)
        print(f"[CKPT] Resuming from episode {resume_episode} step {resume_step}")

    # ── Phase state ──
    # current_phase and global_step persist across episodes for the survival trigger.
    current_phase = "exploration"
    global_step = 0
    _gradual_trigger_step = None   # set when survival_mobs_only is first written

    # If resuming, restore phase state from checkpoint metric
    if args.resume:
        current_phase = getattr(metric, "_current_phase_ckpt", "exploration")
        global_step = getattr(metric, "_global_step_ckpt", 0)
        _gradual_trigger_step = getattr(metric, "_gradual_trigger_step_ckpt", None)

    for episode in range(resume_episode, num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        environment.reset()
        environment.reset_milestone_offset()
        # Re-signal the Minetest server with the current phase (important on resume
        # or whenever the world is freshly reset).
        environment._write_phase_file(current_phase)

        # ── Warm-up: wait for media to load ──
        # VoxeLibre media download can take 5-15 minutes on HPC nodes
        # (first run only; subsequent runs use cached media).
        # Use warmup_noop() to keep TCP channels alive WITHOUT incrementing
        # the environment's step/timestep counters.
        # Detect completion by checking if ALL clients' screenshots have
        # moved past the loading bar (high color std-dev = game world).
        import time as _time
        # On resume the media cache is already populated — skip the warmup loop
        # unless explicitly requested.
        skip_warmup = args.resume is not None and args.resume_skip_warmup
        warmup_secs = 0 if skip_warmup else args.warmup_time
        max_warmup = 900  # hard cap: 15 min
        print(f"  * Waiting for media to load (min {warmup_secs}s, max {max_warmup}s)...")
        warmup_start = _time.time()
        all_loaded = False
        last_log_time = 0.0
        consecutive_loaded = 0  # require sustained signal across multiple checks
        while _time.time() - warmup_start < max_warmup:
            observations = environment.warmup_noop()
            elapsed = _time.time() - warmup_start

            # Compute per-client std-dev
            stds = []
            if observations:
                for obs in observations:
                    if obs is not None:
                        stds.append(np.std(obs.astype(np.float32)))
                    else:
                        stds.append(0.0)

            # Log progress every 30s
            if elapsed - last_log_time >= 30.0 and stds:
                std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds))
                print(f"    [{elapsed:.0f}s] std-dev: {std_str}  (>25 = loaded)")
                last_log_time = elapsed

            # After minimum warm-up time, check if loading screens are gone.
            # Use threshold 25 (not 30) and require 3 consecutive checks to
            # avoid false-positives from VoxeLibre's second loading phase.
            if elapsed >= warmup_secs and stds:
                if all(s > 25.0 for s in stds):
                    consecutive_loaded += 1
                    if consecutive_loaded >= 3:
                        all_loaded = True
                        break
                else:
                    consecutive_loaded = 0
            _time.sleep(2)
        elapsed = _time.time() - warmup_start
        if all_loaded:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds))
            print(f"  * All clients loaded ({elapsed:.0f}s). std-dev: {std_str}")
        else:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds)) if stds else "N/A"
            print(f"  * Warm-up timeout ({elapsed:.0f}s). std-dev: {std_str}. Starting anyway.")

        # All communication is targeted: each agent has its own inbox.
        agent_communications = {i: [] for i in range(num_agents)}
        agents_error_count = [0] * num_agents
        comm_tracker = CommunicationTracker(agent_ids=list(range(num_agents)))
        coop_metric = CooperationMetric(agent_ids=list(range(num_agents)))
        ep_logger = EpisodeLogger(run_dir=metric.target_folder, episode=episode + 1)

        # ── Macro credit assignment: accumulate rewards across macro ticks ──
        # When the RL policy selects a macro, store_reward() is deferred until
        # the macro completes so the buffer receives the full accumulated return.
        _macro_acc_reward = {i: 0.0 for i in range(num_agents)}   # accumulated reward
        _macro_acc_active = {i: False for i in range(num_agents)}  # deferred store_reward
        _was_macro_running = {i: False for i in range(num_agents)} # previous-step macro state
        frames_list = []

        def _save_gif_checkpoint(step_num):
            """Write a GIF for each agent from frames collected so far."""
            for i in range(num_agents):
                agent_frames = [
                    PIL.Image.fromarray(f[i]) for f in frames_list if f[i] is not None
                ]
                if agent_frames:
                    gif_path = (
                        f"{gif_dir}/{run_id}_{role_configs[i]['agent_name']}_ep{episode+1}"
                        f"_step{step_num}.gif"
                    )
                    agent_frames[0].save(
                        gif_path,
                        format="GIF",
                        append_images=agent_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )
                    print(f"  Saved GIF checkpoint: {gif_path}")
                    _frames_to_mp4(agent_frames, gif_path.replace(".gif", ".mp4"))

        _prox_window_count = 0  # proximity bonus events since last log print

        for step in range(max_steps):
            global_step += 1
            logging.info(f"ep={episode+1} step={step+1}/{max_steps} global_step={global_step}")

            if step % log_interval == 0:
                returns_str = "  ".join(
                    f"agent_{i}={metric.cumulative_returns[i]:.1f}"
                    for i in range(num_agents)
                )
                tasks_str = "  ".join(
                    f"agent_{i}={agents[i].auto_curriculum.current_task or 'None'!r}"
                    for i in range(num_agents)
                )
                phase_tag = f" | phase={current_phase}" if args.survival_mode else ""
                prox_tag  = f" | prox_events={_prox_window_count}" if hebbian_config.enabled else ""
                print(
                    f"[{run_id}] ep={episode+1} step={step+1}/{max_steps} | "
                    f"returns: {returns_str} | "
                    f"tasks: {tasks_str}{phase_tag}{prox_tag}"
                )
                _prox_window_count = 0

            # ── Phase transition check ────────────────────────────────────
            if current_phase == "exploration" and _should_transition_to_survival(
                episode, global_step, args
            ):
                new_phase = "survival_mobs_only" if args.survival_gradual else "survival"
                current_phase = new_phase
                _gradual_trigger_step = global_step
                environment._write_phase_file(current_phase)
                metric.record_phase_transition(global_step, episode + 1, current_phase)
                _border = "!" * 60
                print(f"\n{_border}")
                print(f"[PHASE TRANSITION] → {current_phase}  ep={episode+1}  global_step={global_step}")
                print(f"{_border}\n")
                logging.info("[PHASE TRANSITION] → %s ep=%d global_step=%d",
                             current_phase, episode + 1, global_step)

            elif (
                current_phase == "survival_mobs_only"
                and args.survival_gradual
                and _gradual_trigger_step is not None
                and global_step >= _gradual_trigger_step + args.survival_gradual_delay
            ):
                current_phase = "survival"
                environment._write_phase_file(current_phase)
                metric.record_phase_transition(global_step, episode + 1, current_phase)
                _border = "!" * 60
                print(f"\n{_border}")
                print(f"[PHASE TRANSITION] → {current_phase} (hunger enabled)  ep={episode+1}  global_step={global_step}")
                print(f"{_border}\n")
                logging.info("[PHASE TRANSITION] → %s (hunger) ep=%d global_step=%d",
                             current_phase, episode + 1, global_step)
            # ─────────────────────────────────────────────────────────────

            if environment.all_done():
                print(f"  All agents done at step {step+1}")
                break

            # Collect current frames for GIF
            current_frames = []
            for i in range(num_agents):
                frame = environment.get_agent_frame(i)
                current_frames.append(
                    frame if frame is not None
                    else np.zeros((obs_height, obs_width, 3), dtype=np.uint8)
                )
            frames_list.append(current_frames)

            # Periodic GIF checkpoint so partial episodes are visible on HPC time limits
            if save_gif and gif_interval > 0 and (step + 1) % gif_interval == 0:
                _save_gif_checkpoint(step + 1)

            # ── Phase 1: All agents act (collect data for Hebbian) ──
            step_comm_count = 0
            step_rewards_raw = [0.0] * num_agents
            step_contents = [None] * num_agents
            comm_events = []
            # Per-message metadata staged here in Phase 1a; rewards stamped
            # and the records flushed to messages.jsonl in Phase 1b.
            _messages_this_step = []

            # Build per-agent social bond summaries for the LLM prompt
            _bond_strings = {}
            if hebbian_config.enabled:
                for i in range(num_agents):
                    parts = []
                    for j in range(num_agents):
                        if j == i:
                            continue
                        raw_w = hebbian_graph.get_weight(i, j)
                        role_j = ROLE_NAMES[j % len(ROLE_NAMES)]
                        parts.append(f"agent_{j} ({role_j}): {raw_w:.2f}")
                    _bond_strings[i] = "Social bonds: " + ", ".join(parts)

            for agent_id, agent in enumerate(agents):
                agent_name = f"agent_{agent_id}"

                if environment._terminations.get(agent_name, False):
                    continue

                # ── Macro skip: advance macro queue without calling the LLM ──
                if environment.is_macro_running(agent_id):
                    environment.step("NoOp", agentId=agent_id)
                    step_rewards_raw[agent_id] = environment.get_step_reward(agent_id)
                    step_contents[agent_id] = None
                    _was_macro_running[agent_id] = True
                    continue
                # ─────────────────────────────────────────────────────────────

                # ── Macro just finished: flush accumulated reward to RL buffer ──
                # The pending transition was created when the macro was selected.
                # Rewards from all macro ticks are accumulated in _macro_acc_reward
                # and flushed here so the policy sees the full macro return signal.
                if _was_macro_running[agent_id]:
                    if agent.rl_layer and agent.rl_layer.enabled and _macro_acc_active[agent_id]:
                        agent_done = environment._terminations.get(agent_name, False)
                        agent.rl_layer.store_reward(_macro_acc_reward[agent_id], done=agent_done)
                    _macro_acc_reward[agent_id] = 0.0
                    _macro_acc_active[agent_id] = False
                    _was_macro_running[agent_id] = False
                # ─────────────────────────────────────────────────────────────

                error_count = agents_error_count[agent_id]
                frame_image = environment.get_pil_image(agent_id)
                reward_text = environment.get_reward_summary(agent_id)

                comms_for_agent = agent_communications[agent_id]
                # Prepend a one-line survival notice to the instruction prompt so
                # agents know the world has changed. Empty string in exploration
                # phase — no effect on existing behavior.
                _phase_prefix = (
                    "[SURVIVAL MODE ACTIVE: hostile mobs now spawn, hunger drains. "
                    "Prioritize safety alongside your role tasks.]\n\n"
                    if current_phase != "exploration" else ""
                )
                content, last_action, error_count = await agent_do_action(
                    agent, agent_id, frame_image, comms_for_agent, reward_text,
                    _phase_prefix + instruction_prompt, environment,
                    error_count=error_count,
                    social_bonds=_bond_strings.get(agent_id),
                    position_text=environment.get_position_text(agent_id),
                    player_status_text=environment.get_player_status_text(agent_id),
                    current_chamber=environment.get_chamber(agent_id),
                    completed_milestones=metric._agent_milestones.get(
                        f"agent_{agent_id}", set()
                    ),
                )
                agents_error_count[agent_id] = error_count
                step_rewards_raw[agent_id] = environment.get_step_reward(agent_id)
                step_contents[agent_id] = content

                # Handle communication (collect comm_events for Hebbian)
                if (
                    content
                    and content.get("communication")
                    and content["communication"] not in ("", "None")
                    and communication
                ):
                    msg_text = content["communication"]
                    comm_target = content.get("communication_target") or "all"
                    message = TextMessage(content=msg_text, source=agent.name)
                    metric.record_communication(agent.name, msg_text, target=comm_target)
                    step_comm_count += 1

                    try:
                        sender_idx = int(agent.name.split("_")[1])
                    except (IndexError, ValueError):
                        sender_idx = -1

                    if sender_idx < 0:
                        continue  # malformed agent name; drop the message

                    # Resolve recipient from communication_target. The LLM is
                    # required to pick a specific agent (no broadcasts).
                    is_valid_target = (
                        comm_target
                        and comm_target != "all"
                        and comm_target.startswith("agent_")
                    )
                    recv_idx = -1
                    routing_source = "model"
                    if is_valid_target:
                        try:
                            cand = int(comm_target.split("_")[1])
                            if 0 <= cand < num_agents and cand != sender_idx:
                                recv_idx = cand
                        except (IndexError, ValueError):
                            recv_idx = -1

                    if recv_idx < 0:
                        # Model returned "all" or a malformed/self target despite
                        # the targeted-communication policy. Fall back to the
                        # strongest Hebbian bond, or a random teammate, so the
                        # message still reaches exactly one agent.
                        if hebbian_graph.config.enabled:
                            candidates = [
                                (j, float(hebbian_graph.W[sender_idx, j]))
                                for j in range(num_agents) if j != sender_idx
                            ]
                            recv_idx = max(candidates, key=lambda x: x[1])[0]
                            routing_source = "hebbian_fallback"
                        else:
                            others = [j for j in range(num_agents) if j != sender_idx]
                            recv_idx = random.choice(others)
                            routing_source = "random_fallback"

                    agent_communications[recv_idx].append(message)
                    if len(agent_communications[recv_idx]) > num_agents - 1:
                        agent_communications[recv_idx].pop(0)
                    comm_events.append((sender_idx, recv_idx))
                    # Stash per-message metadata; rewards are stamped in Phase 1b
                    # once CommunicationTracker has processed the step.
                    _messages_this_step.append({
                        "t": step,
                        "sender": f"agent_{sender_idx}",
                        "receiver": f"agent_{recv_idx}",
                        "text": msg_text,
                        "tokens": len(msg_text.split()),
                        "routing": routing_source,
                        "model_target": comm_target,
                    })

                time.sleep(sleep_time)

            # ── Phase 1b: Communication rewards + cooperation metrics + step logging ──
            positions = []
            for i in range(num_agents):
                try:
                    pos = environment.env.env._positions[i]
                except (AttributeError, IndexError):
                    pos = None
                positions.append(pos)

            # Capture task rewards before comm bonus is added (for decomposition)
            _task_rewards_this_step = {i: step_rewards_raw[i] for i in range(num_agents)}
            _chat_this_step = {}
            _agent_pos_map = {}
            _actions_this_step = {}
            for _i in range(num_agents):
                _c = step_contents[_i]
                msg = _c.get("communication", "") if _c else ""
                if msg and msg != "None":
                    _chat_this_step[_i] = msg
                _actions_this_step[_i] = (_c.get("action", "NoOp") if _c else "NoOp")
                _env_pos = environment.get_agent_position(_i)
                _agent_pos_map[_i] = _env_pos if _env_pos is not None else positions[_i]

            _comm_rewards_this_step: dict = {}
            _comm_milestones: list = []
            _valid_speakers: set = set()
            if communication:
                _comm_rewards_this_step, _comm_milestones, _valid_speakers = comm_tracker.process_step(
                    step, _chat_this_step, _agent_pos_map
                )
                for _aid, _bonus in _comm_rewards_this_step.items():
                    step_rewards_raw[_aid] += _bonus
                for _aid, _mid, _rw in _comm_milestones:
                    _comm_ev = {
                        "step": step,
                        "milestone": _mid,
                        "contributors": [f"agent_{_aid}"],
                        "reward": _rw,
                    }
                    metric.record_milestone_event(_comm_ev)
                    coop_metric.observe_milestone(step, _mid, [f"agent_{_aid}"])
                    ep_logger.log_event({"step": step, "type": "comm_milestone",
                                         "milestone": _mid, "agent": f"agent_{_aid}",
                                         "reward": _rw})

            # ── Flush per-message records to messages.jsonl ──
            # Stamp reward fields now that CommunicationTracker has run.
            # Note: the tracker bundles base + milestone in one float per
            # speaker; we split using the milestone events list.
            _msg_milestone_per_agent = {}
            for _aid, _mid, _rw in _comm_milestones:
                _msg_milestone_per_agent[_aid] = _msg_milestone_per_agent.get(_aid, 0.0) + _rw
            for _msg in _messages_this_step:
                _sid = int(_msg["sender"].split("_")[1])
                _comm_total = float(_comm_rewards_this_step.get(_sid, 0.0))
                _ms = float(_msg_milestone_per_agent.get(_sid, 0.0))
                _msg["valid"] = _sid in _valid_speakers
                _msg["rewarded_base"] = max(0.0, _comm_total - _ms)
                _msg["rewarded_milestone"] = _ms
                _msg["chamber"] = environment.get_chamber(_sid)
                ep_logger.log_message(_msg)

            coop_metric.observe_step(
                step,
                positions=_agent_pos_map,
                actions=_actions_this_step,
                messages=_chat_this_step,
                task_rewards=_task_rewards_this_step,
            )
            ep_logger.log_step(
                step,
                positions=_agent_pos_map,
                actions=_actions_this_step,
                messages=_chat_this_step,
                task_rewards=_task_rewards_this_step,
                comm_rewards=_comm_rewards_this_step,
                infos={"chambers": {i: environment.get_chamber(i) for i in range(num_agents)}},
            )

            # ── Phase 1c: Centralised critic — encode joint state, evaluate V_global ──
            # Runs once per step, after all agents have acted but before Hebbian
            # diffusion. Each agent's pending transition is updated with the
            # SAME V_global and joint_state so the centralised critic sees the
            # same target across the team.
            joint_state_t = None
            v_global_t = 0.0
            if centralized_critic is not None:
                _hps = {}
                _inv = {}
                _chambers = {}
                for _i in range(num_agents):
                    _status = environment.get_player_status_text(_i) or ""
                    _hp = 20.0
                    if "Health:" in _status:
                        try:
                            _hp = float(_status.split("Health:")[1].split("/")[0].strip())
                        except (ValueError, IndexError):
                            _hp = 20.0
                    _hps[_i] = _hp
                    _inv[_i] = environment.pickedup_object(agentId=_i) or ""
                    _chambers[_i] = environment.get_chamber(_i)
                joint_state_t = centralized_critic.encode_joint(
                    positions=_agent_pos_map,
                    chambers=_chambers,
                    hps=_hps,
                    inventories=_inv,
                    milestones_by_agent=metric._agent_milestones,
                    raw_rewards=_task_rewards_this_step,
                    last_actions=_actions_this_step,
                    last_comms=_chat_this_step,
                )
                v_global_t = centralized_critic.evaluate(joint_state_t)
                for _agent in agents:
                    if _agent.rl_layer is not None and _agent.rl_layer.enabled:
                        _agent.rl_layer.set_pending_value_global(v_global_t, joint_state_t)

            # ── Phase 2: Hebbian update + reward diffusion ──

            # Per-agent one-step advantage δ_t = r_t - V(s_t).
            # V(s_t) was stored by select_action() in the pending transition.
            # We compute this before store_reward() so Hebbian sees the current
            # step's signal rather than a one-step-lagged value.
            # Agents without an active RL layer contribute None (falls back to
            # normalised reward for that agent inside _compute_modulator).
            step_advantages = []
            for _aid, _agent in enumerate(agents):
                v = _agent.rl_layer.get_pending_value() if _agent.rl_layer else None
                if v is not None:
                    step_advantages.append(step_rewards_raw[_aid] - v)
                else:
                    step_advantages.append(None)
            _any_advantage = any(a is not None for a in step_advantages)

            # ── Proximity collaboration bonus ─────────────────────────────────
            # Small per-step bonus for being within interaction_radius of a
            # teammate.  Gives a direct reward signal for co-location that
            # doesn't rely on Hebbian bonds already being built up first.
            # Only active when Hebbian is enabled (shares the radius config).
            if hebbian_config.enabled:
                _PROX_BONUS = 0.3
                for _pi in range(num_agents):
                    for _pj in range(_pi + 1, num_agents):
                        _pos_i, _pos_j = positions[_pi], positions[_pj]
                        if _pos_i is not None and _pos_j is not None:
                            _dist = math.sqrt(sum(
                                (_pos_i[k] - _pos_j[k]) ** 2
                                for k in range(min(len(_pos_i), len(_pos_j), 3))
                            ))
                            if _dist < hebbian_config.interaction_radius:
                                step_rewards_raw[_pi] += _PROX_BONUS
                                step_rewards_raw[_pj] += _PROX_BONUS
                                _prox_window_count += 1

            hebbian_graph.update(
                positions=positions,
                step_rewards=step_rewards_raw,
                advantages=step_advantages if _any_advantage else None,
                comm_events=comm_events if communication else None,
            )
            diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)

            # ── Reward decomposition: split each agent's diffused reward into
            #    its source streams. Recoverable from values already in scope:
            #      task              = _task_rewards_this_step[i]   (line ~1188 snapshot)
            #      comm_total        = _comm_rewards_this_step[i]   (CommTracker output)
            #      comm_milestone    = sum of milestone rewards in _comm_milestones for i
            #      comm_base         = comm_total - comm_milestone
            #      proximity         = step_rewards_raw[i] - task - comm_total
            #      hebbian_diffuse   = diffused_rewards[i] - step_rewards_raw[i] (signed)
            _comm_milestone_per_agent = {i: 0.0 for i in range(num_agents)}
            if communication:
                for _aid, _mid, _rw in (_comm_milestones or []):
                    _comm_milestone_per_agent[_aid] = (
                        _comm_milestone_per_agent.get(_aid, 0.0) + _rw
                    )
            _reward_decomp_this_step = {}
            for _aid in range(num_agents):
                _task = float(_task_rewards_this_step.get(_aid, 0.0))
                _comm_total = float(_comm_rewards_this_step.get(_aid, 0.0)) if communication else 0.0
                _comm_ms = float(_comm_milestone_per_agent.get(_aid, 0.0))
                _comm_base = _comm_total - _comm_ms
                _prox = float(step_rewards_raw[_aid]) - _task - _comm_total
                _hebb = float(diffused_rewards[_aid]) - float(step_rewards_raw[_aid])
                _reward_decomp_this_step[_aid] = {
                    "task":            _task,
                    "comm_base":       _comm_base,
                    "comm_milestone":  _comm_ms,
                    "proximity":       _prox,
                    "hebbian_diffuse": _hebb,
                }

            # ── Phase 3: Record (diffused) rewards for metrics + RL ──
            for agent_id, agent in enumerate(agents):
                agent_name = f"agent_{agent_id}"
                if environment._terminations.get(agent_name, False):
                    continue

                reward = diffused_rewards[agent_id]
                metric.record_reward(agent_id, reward)
                metric.record_reward_decomposed(agent_id, _reward_decomp_this_step[agent_id])

                # Feed reward to RL layer
                if agent.rl_layer and agent.rl_layer.enabled:
                    content = step_contents[agent_id]
                    action_chosen = content.get("action", "NoOp") if content else "NoOp"
                    agent_done = environment._terminations.get(agent_name, False)

                    if action_chosen in _MACRO_NAMES:
                        # Macro just selected this step — defer store_reward.
                        # The pending transition stays open; rewards accumulate
                        # across macro ticks and are flushed when macro finishes.
                        _macro_acc_active[agent_id] = True
                        _macro_acc_reward[agent_id] = reward
                    elif _macro_acc_active[agent_id]:
                        # Still accumulating across macro ticks.
                        # Hebbian still sees per-tick rewards via step_rewards_raw.
                        _macro_acc_reward[agent_id] += reward
                        # Flush now if agent terminated mid-macro
                        if agent_done:
                            agent.rl_layer.store_reward(_macro_acc_reward[agent_id], done=True)
                            _macro_acc_active[agent_id] = False
                            _macro_acc_reward[agent_id] = 0.0
                    else:
                        # Normal step — close the pending transition immediately.
                        agent.rl_layer.store_reward(
                            reward, done=agent_done,
                            reward_task=_task_rewards_this_step.get(agent_id, 0.0),
                            reward_comm=_comm_rewards_this_step.get(agent_id, 0.0),
                        )

                    agent.rl_layer.record_context(
                        action=content.get("action", "NoOp") if content else "NoOp",
                        reward=reward,
                        task=agent.auto_curriculum.current_task or "Explore",
                    )

                    # MAPPO update when enough steps collected.
                    # Pass all agents' buffers so social replay (Eq. 7) can
                    # mix in neighbour transitions weighted by Hebbian bonds.
                    if agent.rl_layer.should_update():
                        neighbour_buffers = {
                            aid: agents[aid].rl_layer.buffer
                            for aid in range(num_agents)
                            if aid != agent_id
                            and agents[aid].rl_layer
                            and agents[aid].rl_layer.enabled
                        }
                        update_info = agent.rl_layer.update(
                            neighbour_buffers=neighbour_buffers,
                            hebbian_graph=hebbian_graph,
                        )
                        metric.record_rl_update(agent_id, update_info)

                    # Agent-decided token-level optimisation
                    try:
                        token_info = await agent.rl_layer.maybe_token_optimize(
                            cancellation_token=CancellationToken(),
                            hebbian_graph=hebbian_graph,
                        )
                        if token_info:
                            metric.record_rl_token_opt(agent_id, token_info)

                        # Social propagation: when an agent trains, offer the
                        # same opportunity to strongly-bonded teammates.
                        if (token_info and token_info.get("decision") == "train"
                                and hebbian_config.enabled):
                            for j in range(num_agents):
                                if j == agent_id:
                                    continue
                                bond_w = float(hebbian_graph.W[agent_id, j])
                                if bond_w > 0.3 and agents[j].rl_layer.enabled:
                                    try:
                                        soc_info = await agents[j].rl_layer.maybe_token_optimize(
                                            cancellation_token=CancellationToken(),
                                            hebbian_graph=hebbian_graph,
                                        )
                                        if soc_info:
                                            logging.info(
                                                "[social token-opt] agent_%d triggered "
                                                "agent_%d (bond=%.3f) → decision=%s",
                                                agent_id, j, bond_w,
                                                soc_info.get("decision", "?"),
                                            )
                                            metric.record_rl_token_opt(j, soc_info)
                                    except Exception as _soc_exc:
                                        logging.warning(
                                            "Social token-opt agent_%d failed: %s",
                                            j, _soc_exc,
                                        )
                    except Exception as _tok_exc:
                        logging.warning(f"Agent {agent_id} token_optimize failed: {_tok_exc}")

            # ── Phase 3a: Centralised critic — store team step, maybe update ──
            if centralized_critic is not None and joint_state_t is not None:
                _alive_rewards = [
                    float(diffused_rewards[_i])
                    for _i in range(num_agents)
                    if not environment._terminations.get(f"agent_{_i}", False)
                ]
                _team_reward = (
                    sum(_alive_rewards) / len(_alive_rewards) if _alive_rewards else 0.0
                )
                _team_done = any(
                    environment._terminations.get(f"agent_{_i}", False)
                    for _i in range(num_agents)
                )
                centralized_critic.store_step(
                    joint_state_t, _team_reward, v_global_t, _team_done,
                )
                if centralized_critic.should_update():
                    _critic_info = centralized_critic.update()
                    if _critic_info:
                        metric.record_rl_update(-1, _critic_info)

            # ── Phase 3b: Five-chambers milestone events ──
            for _ev in environment.poll_milestone_events():
                metric.record_milestone_event(_ev)
                coop_metric.observe_milestone(
                    _ev.get("step", step),
                    _ev.get("milestone", ""),
                    _ev.get("contributors", []),
                )
                ep_logger.log_event({
                    "step": _ev.get("step", step),
                    "type": "milestone",
                    "id": _ev.get("milestone", ""),
                    "contributors": _ev.get("contributors", []),
                    "reward": _ev.get("reward", 0),
                })
                # Surface milestone fires in the SLURM .out file. The Lua side
                # already writes "[SRV] [MILESTONE] ..." into stderr (tailed
                # by craftium), but parsing those lines is brittle — this is
                # the authoritative Python-side line, one per polled event.
                _contribs = _ev.get("contributors", [])
                _contrib_str = ",".join(_contribs) if _contribs else "<none>"
                print(
                    f"[MILESTONE] ep={ep+1} step={step} "
                    f"id={_ev.get('milestone', '?')} "
                    f"agents=[{_contrib_str}] "
                    f"reward={_ev.get('reward', 0)}",
                    flush=True,
                )

            # ── Phase 4: Graph metrics snapshot + SLURM log ──
            if hebbian_config.enabled and step % hebbian_config.log_graph_every == 0:
                graph_metrics = hebbian_graph.get_graph_metrics()
                metric.record_graph_snapshot(global_step, graph_metrics)
                metric.log(f"[Hebbian step {step}] {graph_metrics}")

                # Print a compact weight table to stdout so it lands in
                # the SLURM .out file alongside the reward/task summaries.
                W = hebbian_graph.get_all_weights()
                N = num_agents
                mean_bond = graph_metrics.get("mean_bond_strength", 0.0)
                top = graph_metrics.get("top_3_pairs", [])

                # Header row
                col_hdrs = "      " + "  ".join(f"ag{j:>2}" for j in range(N))
                rows = [col_hdrs, "      " + "------" * N]
                for i in range(N):
                    role_i = role_configs[i]["name"][:3].upper()
                    cells = "  ".join(f"{W[i, j]:5.3f}" for j in range(N))
                    rows.append(f"ag{i} {role_i}  {cells}")

                top_str = "  ".join(
                    f"({p['i']}→{p['j']})={p['w']:.3f}" for p in top
                ) or "none"

                print(
                    f"[{run_id}] [HEBBIAN] ep={episode+1} step={step+1} "
                    f"mean={mean_bond:.4f}  top3: {top_str}\n"
                    + "\n".join(f"  {r}" for r in rows)
                )

            metric.store_timestep(step_comm_count=step_comm_count)

            # ── Periodic checkpoint (within episode) ──
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_step{step+1}")
                save_checkpoint(
                    checkpoint_dir=_ep_ckpt_dir,
                    episode=episode,
                    step=step + 1,
                    run_id=run_id,
                    args=args,
                    metric=metric,
                    agents=agents,
                    hebbian_graph=hebbian_graph,
                    frames_list=frames_list if args.checkpoint_frames else None,
                    save_frames=args.checkpoint_frames,
                    current_phase=current_phase,
                    global_step=global_step,
                    gradual_trigger_step=_gradual_trigger_step,
                )

            # ── Graceful shutdown on signal ──
            if _shutdown_requested:
                _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_step{step+1}_shutdown")
                save_checkpoint(
                    checkpoint_dir=_ep_ckpt_dir,
                    episode=episode,
                    step=step + 1,
                    run_id=run_id,
                    args=args,
                    metric=metric,
                    agents=agents,
                    hebbian_graph=hebbian_graph,
                    frames_list=frames_list if args.checkpoint_frames else None,
                    save_frames=args.checkpoint_frames,
                    current_phase=current_phase,
                    global_step=global_step,
                    gradual_trigger_step=_gradual_trigger_step,
                )
                print(f"[CKPT] Shutdown checkpoint saved → {_ep_ckpt_dir}")
                environment.close()
                return

        # ── End-of-episode: finalize cooperation metrics + structured logs ──
        _hebb_W = hebbian_graph.snapshot().get("W") if hebbian_config.enabled else None
        _ep_final_step = (step + 1) if max_steps > 0 else 0
        _coop_summary = coop_metric.episode_summary(
            final_step=_ep_final_step,
            hebbian_weights=_hebb_W,
        )
        _ep_summary = {
            "episode": episode + 1,
            "final_step": _ep_final_step,
            "total_reward_per_agent": {
                f"agent_{i}": metric.cumulative_returns[i]
                for i in range(num_agents)
            },
            "cooperation_metrics": _coop_summary,
        }
        ep_logger.finalize(_ep_summary)

        # Append Hebbian snapshot to run-level JSONL stream
        _hebb_snapshot_path = os.path.join(metric.target_folder, "hebbian_snapshots.jsonl")
        with open(_hebb_snapshot_path, "a", encoding="utf-8") as _hf:
            _hf.write(_json.dumps({
                "episode": episode + 1,
                "final_step": _coop_summary["final_step"],
                "W": _hebb_W,
                "cooperation_score": _coop_summary.get("cooperation_score", 0.0),
                "reward_total": sum(metric.cumulative_returns),
            }) + "\n")

        # ── End-of-episode checkpoint ──
        _ep_ckpt_dir = os.path.join(checkpoint_dir, f"ep{episode+1}_end")
        save_checkpoint(
            checkpoint_dir=_ep_ckpt_dir,
            episode=episode + 1,  # episode is complete
            step=0,
            run_id=run_id,
            args=args,
            metric=metric,
            agents=agents,
            hebbian_graph=hebbian_graph,
            current_phase=current_phase,
            global_step=global_step,
            gradual_trigger_step=_gradual_trigger_step,
        )

        # Save GIFs for this episode
        if save_gif and frames_list:
            for i in range(num_agents):
                agent_frames = [
                    PIL.Image.fromarray(f[i]) for f in frames_list if f[i] is not None
                ]
                if agent_frames:
                    gif_path = (
                        f"{gif_dir}/{run_id}_{role_configs[i]['agent_name']}_ep{episode+1}.gif"
                    )
                    agent_frames[0].save(
                        gif_path,
                        format="GIF",
                        append_images=agent_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )
                    print(f"[{run_id}] Saved GIF: {gif_path}")
                    _frames_to_mp4(agent_frames, gif_path.replace(".gif", ".mp4"))

    print(f"[{run_id}] Experiment complete! Timesteps logged: {metric.timestep}")
    # Attach run config for reproducibility before saving
    metric.seed = seed
    metric.max_steps = max_steps
    metric.num_episodes = num_episodes
    metric.experiment_id = args.experiment_id
    metric.cli_args = vars(args)
    metric.save_run_metrics()

    # Save RL checkpoints
    for agent in agents:
        if agent.rl_layer and agent.rl_layer.enabled:
            agent.rl_layer.save()
            print(f"  Saved RL checkpoint for {agent.name}")

    # Save Hebbian graph state
    if hebbian_config.enabled:
        graph_path = os.path.join(metric.target_folder, "hebbian_graph_final.json")
        with open(graph_path, "w") as f:
            _json.dump(hebbian_graph.to_dict(), f, indent=2)
        print(f"  Saved final Hebbian graph: {graph_path}")

        # Force one final snapshot and re-save metrics to include it
        metric.record_graph_snapshot(metric.timestep, hebbian_graph.get_graph_metrics())
        metric.save_run_metrics()

    environment.close()


if __name__ == "__main__":
    # Force unbuffered stdout so prints appear immediately in SLURM logs
    import functools
    print = functools.partial(print, flush=True)
    args = parse_args()
    asyncio.run(run(args))
