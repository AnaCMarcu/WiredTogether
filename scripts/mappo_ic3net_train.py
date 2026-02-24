# MAPPO + IC3Net + LLM Communication training script for multi-agent OpenWorld.
#
# Architecture: IC3Net gated communication with LLM-grounded message passing.
#   - Shared CNN encoder + GRU (IC3Net recurrence)
#   - Learned gate: agents decide WHEN to communicate
#   - LLM message codebook: agents decide WHAT to say (32 predefined messages)
#   - Decentralized actor (parameter sharing) + Centralized critic (CTDE)
#
# Based on CleanRL's PPO and adapted for PettingZoo ParallelEnv with recurrence.
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import OpenWorldMultiAgentEnv
from src.models import IC3NetLLMAgent, CommState


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: Optional[int] = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "wired-together"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    save_agent: bool = False
    """save the agent's model"""
    save_num: int = 5
    """number of evenly-spaced checkpoints to save"""

    # Environment
    num_agents: int = 2
    """number of agents in the environment"""
    obs_width: int = 320
    """observation width from Craftium"""
    obs_height: int = 180
    """observation height from Craftium"""
    obs_size: int = 84
    """resized observation dimension (square) for the CNN"""
    max_env_steps: int = 10000
    """max steps per episode in the environment"""

    # Agent architecture
    gru_hidden_dim: int = 256
    """GRU hidden dimension for IC3Net"""
    msg_embed_dim: int = 384
    """message embedding dimension (matches sentence-transformers output)"""
    num_messages: int = 32
    """number of discrete messages in the codebook"""
    sentence_model: str = "all-MiniLM-L6-v2"
    """sentence-transformers model for message codebook initialization"""

    # Communication losses
    comm_gate_penalty: float = 0.01
    """penalty coefficient for communication gate (encourages sparsity)"""
    msg_entropy_coef: float = 0.001
    """entropy bonus for message selection diversity"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiment"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 128
    """the number of steps to run per policy rollout"""
    anneal_lr: bool = True
    """toggle learning rate annealing"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggles clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # Computed at runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def preprocess_obs(obs_dict, agents, obs_size, device):
    """Convert PettingZoo obs dict to batched tensor.

    Returns: (1, num_agents, 3, obs_size, obs_size) float32 tensor.
    """
    frames = []
    for agent in agents:
        frame = cv2.resize(obs_dict[agent], (obs_size, obs_size))
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        frames.append(frame)
    return torch.tensor(np.array(frames), dtype=torch.float32, device=device).unsqueeze(0)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // (args.num_steps * args.num_agents)

    t = int(time.time())
    run_name = f"{args.exp_name}__{args.seed}__{t}"
    if args.seed is None:
        args.seed = t

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.save_agent:
        agent_path = f"agents/{run_name}"
        os.makedirs(agent_path, exist_ok=True)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # Environment setup
    env = OpenWorldMultiAgentEnv(
        num_agents=args.num_agents,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        max_steps=args.max_env_steps,
    )

    num_agents = args.num_agents
    action_dim = env.action_space("agent_0").n
    agent_names = env.possible_agents

    print(f"Agents: {agent_names}")
    print(f"Action space: Discrete({action_dim})")
    print(f"Architecture: IC3Net + LLM Communication")
    print(f"GRU hidden: {args.gru_hidden_dim}, Messages: {args.num_messages}, Msg dim: {args.msg_embed_dim}")
    print(f"Iterations: {args.num_iterations}, Steps/rollout: {args.num_steps}")

    # Create agent and initialize message codebook
    agent = IC3NetLLMAgent(
        num_agents=num_agents,
        action_dim=action_dim,
        obs_size=args.obs_size,
        gru_hidden_dim=args.gru_hidden_dim,
        msg_embed_dim=args.msg_embed_dim,
        num_messages=args.num_messages,
    ).to(device)

    print("Loading message codebook from sentence-transformers...")
    agent.load_message_codebook(args.sentence_model)
    print("Message codebook loaded.")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {total_params:,}")

    # Rollout storage
    obs = torch.zeros(
        (args.num_steps, num_agents, 3, args.obs_size, args.obs_size), device=device
    )
    actions = torch.zeros((args.num_steps, num_agents), device=device)
    logprobs = torch.zeros((args.num_steps, num_agents), device=device)
    rewards = torch.zeros((args.num_steps, num_agents), device=device)
    dones = torch.zeros((args.num_steps, num_agents), device=device)
    values = torch.zeros((args.num_steps, 1), device=device)

    # Communication state storage (per-step, for PPO replay)
    comm_hiddens = torch.zeros(
        (args.num_steps, num_agents, args.gru_hidden_dim), device=device
    )
    comm_messages = torch.zeros(
        (args.num_steps, num_agents, args.msg_embed_dim), device=device
    )
    comm_gates = torch.zeros((args.num_steps, num_agents, 1), device=device)
    gate_values_log = torch.zeros((args.num_steps, num_agents), device=device)

    # Initialize
    global_step = 0
    start_time = time.time()

    obs_dict, _ = env.reset(seed=args.seed)
    next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)
    next_done = torch.zeros(num_agents, device=device)
    next_comm_state = agent.init_comm_state(batch_size=1, device=device)

    ep_rets = np.zeros(num_agents)
    ep_len = 0

    try:
        for iteration in range(1, args.num_iterations + 1):
            # LR annealing
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            for step in range(args.num_steps):
                global_step += num_agents
                obs[step] = next_obs.squeeze(0)
                dones[step] = next_done

                # Get actions, values, and communication state
                with torch.no_grad():
                    action, logprob, _, value, next_comm_state, comm_info = (
                        agent.get_action_and_value(
                            next_obs,
                            comm_state=next_comm_state,
                            dones=next_done.unsqueeze(0),
                        )
                    )
                    values[step] = value.squeeze(0)

                    # Store communication state for this step
                    comm_hiddens[step] = next_comm_state.hidden.squeeze(0)
                    comm_messages[step] = next_comm_state.prev_messages.squeeze(0)
                    comm_gates[step] = next_comm_state.prev_gate.squeeze(0)
                    gate_values_log[step] = comm_info["gate_values"].squeeze(0).squeeze(-1)

                actions[step] = action.squeeze(0)
                logprobs[step] = logprob.squeeze(0)

                # Convert to PettingZoo action dict
                action_np = action.squeeze(0).cpu().numpy()
                actions_dict = {
                    agent_names[i]: int(action_np[i]) for i in range(num_agents)
                }

                # Step environment
                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                    actions_dict
                )

                # Store rewards and dones
                for i, name in enumerate(agent_names):
                    rewards[step, i] = reward_dict.get(name, 0.0)
                    done = term_dict.get(name, False) or trunc_dict.get(name, False)
                    dones[step, i] = float(done)

                # Track episode stats
                for i, name in enumerate(agent_names):
                    ep_rets[i] += reward_dict.get(name, 0.0)
                ep_len += 1

                # Check for episode end
                any_done = any(term_dict.values()) or any(trunc_dict.values())
                if any_done:
                    mean_ret = ep_rets.mean()
                    gate_ratio = gate_values_log[: step + 1].mean().item()
                    print(
                        f"  global_step={global_step}, ep_len={ep_len}, "
                        f"mean_return={mean_ret:.3f}, gate_open={gate_ratio:.2f}, "
                        + ", ".join(
                            f"agent_{i}={ep_rets[i]:.3f}" for i in range(num_agents)
                        )
                    )
                    for i in range(num_agents):
                        writer.add_scalar(
                            f"charts/episodic_return_agent_{i}", ep_rets[i], global_step
                        )
                    writer.add_scalar("charts/episodic_return_mean", mean_ret, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len, global_step)

                    # Reset episode and communication state
                    ep_rets = np.zeros(num_agents)
                    ep_len = 0
                    obs_dict, _ = env.reset()
                    next_done = torch.zeros(num_agents, device=device)
                    next_comm_state = agent.init_comm_state(batch_size=1, device=device)
                else:
                    next_done = torch.tensor(
                        [
                            float(
                                term_dict.get(name, False)
                                or trunc_dict.get(name, False)
                            )
                            for name in agent_names
                        ],
                        device=device,
                    )

                next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)

            # -- GAE computation --
            with torch.no_grad():
                next_value, _ = agent.get_value(
                    next_obs,
                    comm_state=next_comm_state,
                    dones=next_done.unsqueeze(0),
                )
                next_value = next_value.squeeze(0)  # (1,)

                advantages = torch.zeros_like(rewards, device=device)
                for i in range(num_agents):
                    lastgaelam = 0.0
                    for t_step in reversed(range(args.num_steps)):
                        if t_step == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[i]
                            nextvalue = next_value[0]
                        else:
                            nextnonterminal = 1.0 - dones[t_step + 1, i]
                            nextvalue = values[t_step + 1, 0]
                        delta = (
                            rewards[t_step, i]
                            + args.gamma * nextvalue * nextnonterminal
                            - values[t_step, 0]
                        )
                        advantages[t_step, i] = lastgaelam = (
                            delta
                            + args.gamma
                            * args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                returns = advantages + values.expand_as(advantages)

            # -- PPO update --
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            last_gate_loss = 0.0
            last_msg_entropy = 0.0

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_size = len(mb_inds)

                    # Gather minibatch data (timestep-grouped)
                    mb_obs = obs[mb_inds]
                    mb_actions = actions[mb_inds]
                    mb_logprobs = logprobs[mb_inds]
                    mb_advantages = advantages[mb_inds]
                    mb_returns = returns[mb_inds]
                    mb_values = values[mb_inds]
                    mb_dones = dones[mb_inds]

                    # Reconstruct communication state from stored per-step buffers
                    mb_comm_state = CommState(
                        hidden=comm_hiddens[mb_inds],
                        prev_messages=comm_messages[mb_inds],
                        prev_gate=comm_gates[mb_inds],
                    )

                    # Forward pass with communication
                    _, newlogprob, entropy, newvalue, _, comm_info = (
                        agent.get_action_and_value(
                            mb_obs,
                            comm_state=mb_comm_state,
                            dones=mb_dones,
                            actions=mb_actions.long(),
                        )
                    )

                    # -- Policy loss --
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_adv_flat = mb_advantages.reshape(-1)
                    if args.norm_adv:
                        mb_adv_flat = (mb_adv_flat - mb_adv_flat.mean()) / (
                            mb_adv_flat.std() + 1e-8
                        )
                    mb_adv_norm = mb_adv_flat.reshape(mb_size, num_agents)

                    pg_loss1 = -mb_adv_norm * ratio
                    pg_loss2 = -mb_adv_norm * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # -- Value loss --
                    newvalue_flat = newvalue.squeeze(-1)
                    mb_returns_mean = mb_returns.mean(dim=1)
                    mb_values_flat = mb_values.squeeze(-1)

                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue_flat - mb_returns_mean) ** 2
                        v_clipped = mb_values_flat + torch.clamp(
                            newvalue_flat - mb_values_flat,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns_mean) ** 2
                        v_loss = (
                            0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        )
                    else:
                        v_loss = 0.5 * ((newvalue_flat - mb_returns_mean) ** 2).mean()

                    entropy_loss = entropy.mean()

                    # -- Communication auxiliary losses --
                    # Gate sparsity: penalize frequent communication
                    gate_loss = (
                        comm_info["gate_values"].mean() * args.comm_gate_penalty
                    )

                    # Message diversity: encourage using different messages
                    msg_probs = torch.softmax(comm_info["msg_logits"], dim=-1)
                    msg_ent = -(msg_probs * (msg_probs + 1e-8).log()).sum(dim=-1).mean()
                    msg_entropy_loss = -msg_ent * args.msg_entropy_coef

                    last_gate_loss = gate_loss.item()
                    last_msg_entropy = msg_ent.item()

                    # Combined loss
                    loss = (
                        pg_loss
                        - args.ent_coef * entropy_loss
                        + v_loss * args.vf_coef
                        + gate_loss
                        + msg_entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # -- Logging --
            y_pred = values.squeeze(-1).cpu().numpy()
            y_true = returns.mean(dim=1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            # Communication metrics
            writer.add_scalar(
                "comm/gate_open_ratio", gate_values_log.mean().item(), global_step
            )
            writer.add_scalar("comm/gate_loss", last_gate_loss, global_step)
            writer.add_scalar("comm/msg_entropy", last_msg_entropy, global_step)

            sps = int(global_step / (time.time() - start_time))
            print(
                f"Iteration {iteration}/{args.num_iterations}, SPS: {sps}, "
                f"gate_open: {gate_values_log.mean().item():.2f}"
            )
            writer.add_scalar("charts/SPS", sps, global_step)

            # Save checkpoint
            if args.save_agent and args.num_iterations > 0:
                save_interval = max(1, args.num_iterations // args.save_num)
                if iteration % save_interval == 0 or iteration == args.num_iterations:
                    ckpt_path = f"{agent_path}/agent_step_{global_step}.pt"
                    torch.save(agent.state_dict(), ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        print("Closing environment...")
        env.close()
        writer.close()
        print("Done.")
