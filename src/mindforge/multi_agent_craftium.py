"""Main loop for running Mindforge agents in Craftium OpenWorld.

Role-specialized agents (gatherer, hunter, defender) interact with
the Craftium multi-agent environment, using LLM-based action selection,
episodic memory, belief systems, and auto-curriculum.

Supports any number of agents -- roles cycle (gatherer, hunter, defender,
gatherer, hunter, ...) when NUM_AGENTS > 3.

Usage:
    cd src/mindforge
    python multi_agent_craftium.py
    python multi_agent_craftium.py --num-agents 6 --episodes 3 --max-steps 1000
"""

import argparse
import asyncio
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
import json as _json

from rl_layer import RLConfig, RLLayer, HebbianConfig, HebbianSocialGraph

ROLE_NAMES = ["gatherer", "hunter", "defender"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mindforge agents in Craftium OpenWorld")
    parser.add_argument("--num-agents", type=int, default=3,
                        help="Number of agents (roles cycle: gatherer, hunter, defender)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--obs-width", type=int, default=320,
                        help="Observation width in pixels")
    parser.add_argument("--obs-height", type=int, default=180,
                        help="Observation height in pixels")
    parser.add_argument("--no-communication", action="store_true",
                        help="Disable inter-agent communication")
    parser.add_argument("--sleep-time", type=float, default=0.0,
                        help="Seconds to sleep between LLM calls (rate-limit protection)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Disable GIF saving")
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
    parser.add_argument("--rl-update-interval", type=int, default=64,
                        help="Steps between MAPPO updates")
    parser.add_argument("--rl-lr", type=float, default=3e-4,
                        help="Learning rate for RL optimiser")
    parser.add_argument("--rl-auto-token-opt", action="store_true",
                        help="Let agents self-trigger token-level optimisation")
    parser.add_argument("--rl-mode", type=str, default="action",
                        choices=["action", "token"],
                        help="RL mode: 'action' = MAPPO action head, "
                             "'token' = token-opt only (LLM picks actions)")
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
    parser.add_argument("--hebbian-no-comm-bond", action="store_true",
                        help="Set δ_comm=0 (spatial-only, for RQ4 ablation)")
    # ── Experiment tracking ──
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Experiment identifier (e.g. E1a, E5) — saved in metrics for traceability")
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


def build_role_configs(num_agents, role_prompts):
    """Build ROLE_CONFIGS for num_agents, cycling through roles."""
    configs = []
    for i in range(num_agents):
        role = ROLE_NAMES[i % len(ROLE_NAMES)]
        configs.append({
            "name": role,
            "agent_name": f"agent_{i}_{role}",
            "curriculum_prompt": role_prompts[role].format(num_agents=num_agents),
        })
    return configs


def build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                 rl_config=None):
    """Initialize all Mindforge agents."""
    agents = []
    for i, role_cfg in enumerate(role_configs):
        # Build per-agent RL layer (no-op when rl_config.enabled is False)
        rl_layer = None
        if rl_config and rl_config.enabled:
            rl_layer = RLLayer(config=rl_config, role=role_cfg["name"], agent_id=i)

        agent = CustomAgent(
            name=role_cfg["agent_name"],
            description=f"{role_cfg['name']} agent in Craftium open world",
            action_selection=ActionSelection(system_prompt=system_prompt),
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
            )
        else:
            logging.error(f"Agent {agent_id} exceeded retry limit, using NoOp")
            environment.step("NoOp", agentId=agent_id)
            last_action = "NoOp"

    return content, last_action, error_count


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

    # Logging
    os.makedirs("logs", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)

    event_logger = logging.getLogger(EVENT_LOGGER_NAME)
    event_logger.disabled = True
    logging.basicConfig(
        level=logging.INFO,
        filename=f"logs/craftium_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load prompts
    prompts = load_prompts()
    environment_prompt = prompts["environment"]
    instruction_prompt = prompts["instruction"]

    # Build system prompt with environment details baked in
    from agent_modules.util import safe_format
    system_prompt_template = prompts["system_template"]
    system_prompt = safe_format(system_prompt_template, environment_prompt=environment_prompt)

    # Roles & agents
    role_configs = build_role_configs(num_agents, prompts["roles"])

    metric = CraftiumMetric(
        num_agents=num_agents,
        communication=communication,
    )

    environment = CraftiumEnvironmentInterface(
        num_agents=num_agents,
        obs_width=obs_width,
        obs_height=obs_height,
        max_steps=max_steps,
        seed=seed,
    )

    # ── RL layer config ──
    rl_config = RLConfig(
        enabled=args.rl,
        mode=args.rl_mode,
        model_path=args.rl_model_path,
        lora_rank=args.rl_lora_rank,
        update_interval=args.rl_update_interval,
        lr=args.rl_lr,
        auto_token_opt=args.rl_auto_token_opt,
    )
    if rl_config.enabled:
        print(f"RL layer ENABLED: model={rl_config.model_path}, "
              f"lora_rank={rl_config.lora_rank}, update_interval={rl_config.update_interval}")

    agents = build_agents(role_configs, system_prompt, prompts, num_agents, communication, metric,
                         rl_config=rl_config)

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
    )
    agent_roles = [ROLE_NAMES.index(rc["name"]) for rc in role_configs]
    hebbian_graph = HebbianSocialGraph(hebbian_config, agent_roles=agent_roles)
    if hebbian_config.enabled:
        print(f"Hebbian social plasticity ENABLED: ltp={hebbian_config.ltp_lr}, "
              f"ltd={hebbian_config.ltd_lr}, radius={hebbian_config.interaction_radius}, "
              f"γ={hebbian_config.reward_diffusion_gamma}")

    print(f"\nConfig: {num_agents} agents, {num_episodes} episodes, "
          f"{max_steps} max steps, comm={'on' if communication else 'off'}, "
          f"seed={seed}")

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        environment.reset()

        # ── Warm-up: wait for media to load ──
        # VoxeLibre media download can take 5-15 minutes on HPC nodes
        # (first run only; subsequent runs use cached media).
        # Use warmup_noop() to keep TCP channels alive WITHOUT incrementing
        # the environment's step/timestep counters.
        # Detect completion by checking if ALL clients' screenshots have
        # moved past the loading bar (high color std-dev = game world).
        import time as _time
        warmup_secs = args.warmup_time
        max_warmup = 900  # hard cap: 15 min
        print(f"  * Waiting for media to load (min {warmup_secs}s, max {max_warmup}s)...")
        warmup_start = _time.time()
        all_loaded = False
        last_log_time = 0.0
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
                print(f"    [{elapsed:.0f}s] std-dev: {std_str}  (>30 = loaded)")
                last_log_time = elapsed

            # After minimum warm-up time, check if loading screens are gone
            if elapsed >= warmup_secs and stds:
                # Loading screen is mostly uniform dark gray/black.
                # Game world has varied colors -> higher std deviation.
                loaded_count = sum(1 for s in stds if s > 30.0)
                if loaded_count == num_agents:
                    all_loaded = True
                    break
            _time.sleep(2)
        elapsed = _time.time() - warmup_start
        if all_loaded:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds))
            print(f"  * All clients loaded ({elapsed:.0f}s). std-dev: {std_str}")
        else:
            std_str = ", ".join(f"agent_{i}={s:.1f}" for i, s in enumerate(stds)) if stds else "N/A"
            print(f"  * Warm-up timeout ({elapsed:.0f}s). std-dev: {std_str}. Starting anyway.")

        communications = []
        agents_error_count = [0] * num_agents
        frames_list = []

        for step in range(max_steps):
            logging.info(f"Episode {episode+1}, Step {step+1}/{max_steps}")

            if step % 50 == 0:
                print(f"  Step {step+1}/{max_steps}")

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

            # ── Phase 1: All agents act (collect data for Hebbian) ──
            step_comm_count = 0
            step_rewards_raw = [0.0] * num_agents
            step_contents = [None] * num_agents
            comm_events = []

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

                error_count = agents_error_count[agent_id]
                frame_image = environment.get_pil_image(agent_id)
                reward_text = environment.get_reward_summary(agent_id)

                content, last_action, error_count = await agent_do_action(
                    agent, agent_id, frame_image, communications, reward_text,
                    instruction_prompt, environment,
                    error_count=error_count,
                    social_bonds=_bond_strings.get(agent_id),
                    position_text=environment.get_position_text(agent_id),
                    player_status_text=environment.get_player_status_text(agent_id),
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
                    message = TextMessage(
                        content=content["communication"],
                        source=agent.name,
                    )
                    communications.append(message)
                    metric.record_communication(agent.name, content["communication"])
                    step_comm_count += 1

                    # Extract sender index for Hebbian comm_events
                    try:
                        sender_idx = int(agent.name.split("_")[1])
                        for recv_idx in range(num_agents):
                            if recv_idx != sender_idx:
                                comm_events.append((sender_idx, recv_idx))
                    except (IndexError, ValueError):
                        pass

                if len(communications) > num_agents - 1:
                    communications.pop(0)

                time.sleep(sleep_time)

            # ── Phase 2: Hebbian update + reward diffusion ──
            positions = []
            for i in range(num_agents):
                try:
                    pos = environment.env.env._positions[i]
                except (AttributeError, IndexError):
                    pos = None
                positions.append(pos)

            hebbian_graph.update(
                positions=positions,
                step_rewards=step_rewards_raw,
                advantages=None,
                comm_events=comm_events if communication else None,
            )
            diffused_rewards = hebbian_graph.diffuse_rewards(step_rewards_raw)

            # ── Phase 3: Record (diffused) rewards for metrics + RL ──
            for agent_id, agent in enumerate(agents):
                agent_name = f"agent_{agent_id}"
                if environment._terminations.get(agent_name, False):
                    continue

                reward = diffused_rewards[agent_id]
                metric.record_reward(agent_id, reward)

                # Feed reward to RL layer
                if agent.rl_layer and agent.rl_layer.enabled:
                    agent_done = environment._terminations.get(agent_name, False)
                    agent.rl_layer.store_reward(reward, done=agent_done)
                    content = step_contents[agent_id]
                    agent.rl_layer.record_context(
                        action=content.get("action", "NoOp") if content else "NoOp",
                        reward=reward,
                        task=agent.auto_curriculum.current_task or "Explore",
                    )

                    # MAPPO update when enough steps collected
                    if agent.rl_layer.should_update():
                        update_info = agent.rl_layer.update()
                        metric.record_rl_update(agent_id, update_info)

                    # Agent-decided token-level optimisation
                    token_info = await agent.rl_layer.maybe_token_optimize(
                        cancellation_token=CancellationToken(),
                    )
                    if token_info:
                        metric.record_rl_token_opt(agent_id, token_info)

            # ── Phase 4: Graph metrics snapshot ──
            if hebbian_config.enabled and step % hebbian_config.log_graph_every == 0:
                graph_metrics = hebbian_graph.get_graph_metrics()
                metric.record_graph_snapshot(step, graph_metrics)
                metric.log(f"[Hebbian step {step}] {graph_metrics}")

            metric.store_timestep(step_comm_count=step_comm_count)

        # Save GIFs for this episode
        if save_gif and frames_list:
            for i in range(num_agents):
                agent_frames = [
                    PIL.Image.fromarray(f[i]) for f in frames_list if f[i] is not None
                ]
                if agent_frames:
                    gif_path = (
                        f"gifs/{role_configs[i]['agent_name']}_ep{episode+1}"
                        f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.gif"
                    )
                    agent_frames[0].save(
                        gif_path,
                        format="GIF",
                        append_images=agent_frames[1:],
                        save_all=True,
                        duration=500,
                        loop=0,
                    )
                    print(f"  Saved GIF: {gif_path}")

    print(f"\nExperiment complete! Timesteps logged: {metric.timestep}")
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
