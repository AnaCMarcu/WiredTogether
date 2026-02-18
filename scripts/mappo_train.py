# MAPPO training script for multi-agent OpenWorld (Craftium).
#
# Centralized Training with Decentralized Execution (CTDE):
#   - Shared CNN encoder + decentralized actor (parameter sharing)
#   - Centralized critic (concatenates features from ALL agents)
#
# Based on CleanRL's PPO implementation and adapted for PettingZoo ParallelEnv.
# Reference: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Add project root to path so we can import src.envs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import OpenWorldMultiAgentEnv


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def preprocess_obs(obs_dict, agents, obs_size, device):
    """Convert PettingZoo obs dict to batched tensor.

    Args:
        obs_dict: {agent_0: (H, W, 3), ...} uint8 numpy arrays
        agents: list of agent names in order
        obs_size: target square size for resize
        device: torch device

    Returns:
        Tensor of shape (1, num_agents, 3, obs_size, obs_size) float32
    """
    frames = []
    for agent in agents:
        frame = cv2.resize(obs_dict[agent], (obs_size, obs_size))
        frame = frame.transpose(2, 0, 1)  # HWC -> CHW
        frames.append(frame)
    return torch.tensor(np.array(frames), dtype=torch.float32, device=device).unsqueeze(0)


class MAPPOAgent(nn.Module):
    """MAPPO agent with shared encoder, decentralized actor, centralized critic."""

    def __init__(self, num_agents, action_dim, obs_size=84):
        super().__init__()
        self.num_agents = num_agents

        # Shared CNN encoder: (3, 84, 84) -> 512-dim features
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # Decentralized actor (shared weights across agents)
        self.actor = layer_init(nn.Linear(512, action_dim), std=0.01)

        # Centralized critic: concat all agent features -> value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(512 * num_agents, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_features(self, obs):
        """Extract features from observations.

        Args:
            obs: (batch, 3, H, W) float tensor

        Returns:
            (batch, 512) feature tensor
        """
        return self.encoder(obs / 255.0)

    def get_value(self, all_obs):
        """Centralized value function.

        Args:
            all_obs: (batch, num_agents, 3, H, W) float tensor

        Returns:
            (batch, 1) value tensor
        """
        batch_size = all_obs.shape[0]
        flat_obs = all_obs.reshape(-1, *all_obs.shape[2:])
        flat_features = self.get_features(flat_obs)
        features = flat_features.reshape(batch_size, self.num_agents, -1)
        global_features = features.reshape(batch_size, -1)
        return self.critic(global_features)

    def get_action_and_value(self, all_obs, actions=None):
        """Forward pass for both actor and critic.

        Args:
            all_obs: (batch, num_agents, 3, H, W) float tensor
            actions: (batch, num_agents) optional pre-specified actions

        Returns:
            actions: (batch, num_agents) sampled or given actions
            log_probs: (batch, num_agents) log probabilities
            entropy: (batch, num_agents) action entropy
            values: (batch, 1) centralized value
        """
        batch_size = all_obs.shape[0]

        # Shared encoder for all agents
        flat_obs = all_obs.reshape(-1, *all_obs.shape[2:])
        flat_features = self.get_features(flat_obs)
        features = flat_features.reshape(batch_size, self.num_agents, -1)

        # Decentralized actor
        flat_logits = self.actor(flat_features)
        logits = flat_logits.reshape(batch_size, self.num_agents, -1)
        probs = Categorical(logits=logits)

        if actions is None:
            actions = probs.sample()

        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()

        # Centralized critic
        global_features = features.reshape(batch_size, -1)
        values = self.critic(global_features)

        return actions, log_probs, entropy, values


if __name__ == "__main__":
    args = tyro.cli(Args)
    # batch_size = num_steps (timesteps per rollout)
    # Each timestep has num_agents samples for the actor
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
    print(f"Observation: ({args.obs_height}, {args.obs_width}, 3) -> ({args.obs_size}, {args.obs_size}, 3)")
    print(f"Iterations: {args.num_iterations}, Steps/rollout: {args.num_steps}")
    print(f"Batch size: {args.batch_size} timesteps, Minibatch: {args.minibatch_size} timesteps")

    agent = MAPPOAgent(num_agents, action_dim, args.obs_size).to(device)
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

    # Initialize
    global_step = 0
    start_time = time.time()

    obs_dict, _ = env.reset(seed=args.seed)
    next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)  # (1, N, 3, H, W)
    next_done = torch.zeros(num_agents, device=device)

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

                # Get actions and values
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.squeeze(0)
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

                # Check for episode end (any agent done)
                any_done = any(term_dict.values()) or any(trunc_dict.values())
                if any_done:
                    # Log episode stats
                    mean_ret = ep_rets.mean()
                    print(
                        f"  global_step={global_step}, ep_len={ep_len}, "
                        f"mean_return={mean_ret:.3f}, "
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

                    # Reset
                    ep_rets = np.zeros(num_agents)
                    ep_len = 0
                    obs_dict, _ = env.reset()
                    next_done = torch.zeros(num_agents, device=device)
                else:
                    next_done = torch.tensor(
                        [float(term_dict.get(name, False) or trunc_dict.get(name, False))
                         for name in agent_names],
                        device=device,
                    )

                next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)

            # -- GAE computation --
            with torch.no_grad():
                next_value = agent.get_value(next_obs).squeeze(0)  # (1,)
                # Compute per-agent advantages using shared centralized value
                advantages = torch.zeros_like(rewards, device=device)
                for i in range(num_agents):
                    lastgaelam = 0.0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[i]
                            nextvalue = next_value[0]
                        else:
                            nextnonterminal = 1.0 - dones[t + 1, i]
                            nextvalue = values[t + 1, 0]
                        delta = (
                            rewards[t, i]
                            + args.gamma * nextvalue * nextnonterminal
                            - values[t, 0]
                        )
                        advantages[t, i] = lastgaelam = (
                            delta
                            + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                returns = advantages + values.expand_as(advantages)

            # -- PPO update --
            # Minibatches are over TIMESTEP indices (not flattened agent samples)
            # because the centralized critic needs all agents at each timestep.
            b_inds = np.arange(args.batch_size)
            clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_size = len(mb_inds)

                    # Gather minibatch data (timestep-grouped)
                    mb_obs = obs[mb_inds]  # (mb, N, 3, H, W)
                    mb_actions = actions[mb_inds]  # (mb, N)
                    mb_logprobs = logprobs[mb_inds]  # (mb, N)
                    mb_advantages = advantages[mb_inds]  # (mb, N)
                    mb_returns = returns[mb_inds]  # (mb, N)
                    mb_values = values[mb_inds]  # (mb, 1)

                    # Forward pass
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_obs, mb_actions.long()
                    )
                    # newlogprob: (mb, N), entropy: (mb, N), newvalue: (mb, 1)

                    # -- Policy loss (per-agent, then averaged) --
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    # Flatten advantages across agents for normalization
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

                    # -- Value loss (centralized) --
                    newvalue_flat = newvalue.squeeze(-1)  # (mb,)
                    mb_returns_mean = mb_returns.mean(dim=1)  # (mb,) avg across agents
                    mb_values_flat = mb_values.squeeze(-1)  # (mb,)

                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue_flat - mb_returns_mean) ** 2
                        v_clipped = mb_values_flat + torch.clamp(
                            newvalue_flat - mb_values_flat,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns_mean) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue_flat - mb_returns_mean) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # -- Logging --
            y_pred = values.squeeze(-1).cpu().numpy()  # (num_steps,)
            y_true = returns.mean(dim=1).cpu().numpy()  # (num_steps,)
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

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

            sps = int(global_step / (time.time() - start_time))
            print(f"Iteration {iteration}/{args.num_iterations}, SPS: {sps}")
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
