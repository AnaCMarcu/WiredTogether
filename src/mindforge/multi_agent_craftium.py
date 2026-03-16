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
from rl_layer import RLConfig, RLLayer

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
    parser.add_argument("--warmup-time", type=int, default=180,
                        help="Seconds to wait for media loading before starting (default 180)")
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
    )

    # ── RL layer config ──
    rl_config = RLConfig(
        enabled=args.rl,
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

    print(f"\nConfig: {num_agents} agents, {num_episodes} episodes, "
          f"{max_steps} max steps, comm={'on' if communication else 'off'}")

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        environment.reset()

        # ── Warm-up: wait for media to load ──
        # VoxeLibre media download takes 2-5 minutes on HPC nodes.
        # Use warmup_noop() to keep TCP channels alive WITHOUT incrementing
        # the environment's step/timestep counters (which would exhaust the
        # episode budget before real gameplay starts).
        import time as _time
        warmup_secs = args.warmup_time
        print(f"  * Waiting {warmup_secs}s for media to load...")
        warmup_start = _time.time()
        while _time.time() - warmup_start < warmup_secs:
            environment.warmup_noop()
            _time.sleep(2)
        elapsed = _time.time() - warmup_start
        print(f"  * Warm-up done ({elapsed:.0f}s). Starting episode.")

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

            # Each agent takes a turn
            step_comm_count = 0
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
                )
                agents_error_count[agent_id] = error_count

                # Record reward for metrics + RL
                step_reward = environment.get_step_reward(agent_id)
                metric.record_reward(agent_id, step_reward)

                # Feed reward to RL layer
                if agent.rl_layer and agent.rl_layer.enabled:
                    agent_done = environment._terminations.get(f"agent_{agent_id}", False)
                    agent.rl_layer.store_reward(step_reward, done=agent_done)
                    agent.rl_layer.record_context(
                        action=content.get("action", "NoOp") if content else "NoOp",
                        reward=step_reward,
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

                # Handle communication
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

                if len(communications) > num_agents - 1:
                    communications.pop(0)

                time.sleep(sleep_time)

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
    metric.save_run_metrics()

    # Save RL checkpoints
    for agent in agents:
        if agent.rl_layer and agent.rl_layer.enabled:
            agent.rl_layer.save()
            print(f"  Saved RL checkpoint for {agent.name}")

    environment.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
