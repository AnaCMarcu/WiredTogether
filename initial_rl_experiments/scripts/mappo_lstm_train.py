# MAPPO + LSTM training script for multi-agent OpenWorld (Craftium).
#
# Centralized Training with Decentralized Execution (CTDE):
#   - Shared CNN encoder + LSTM temporal memory + decentralized actor
#   - Centralized critic (concatenates LSTM features from ALL agents)
#
# Combines:
#   - MAPPO multi-agent structure from scripts/mappo_train.py
#   - LSTM state management from craftium/cleanrl_ppo_lstm_train.py
#
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
from torch.utils.tensorboard import SummaryWriter

# Add project root to path so we can import src.envs and src.models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.envs import OpenWorldMultiAgentEnv
from src.models.agent import MAPPOLSTMAgent


# ---------------------------------------------------------------------------
# Args (from mappo_train.py + lstm_hidden_size from cleanrl PPO-LSTM)
# ---------------------------------------------------------------------------
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

    # LSTM-specific (from cleanrl PPO-LSTM)
    lstm_hidden_size: int = 128
    """LSTM hidden state dimension"""

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
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# ---------------------------------------------------------------------------
# Observation preprocessing (from mappo_train.py, unchanged)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    # batch_size = num_steps (each timestep has num_agents actor samples)
    # With LSTM we process the full sequence — no minibatch splitting.
    args.batch_size = args.num_steps
    args.num_iterations = args.total_timesteps // (args.num_steps * args.num_agents)

    t = int(time.time())
    run_name = f"{args.exp_name}__{args.seed}__{t}"
    if args.seed is None:
        args.seed = t

    # -- Wandb / TensorBoard setup (from mappo_train.py) --
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

    # -- Seeding (from mappo_train.py) --
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # -- Environment setup (from mappo_train.py) --
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
    print(f"LSTM hidden size: {args.lstm_hidden_size}")
    print(f"Iterations: {args.num_iterations}, Steps/rollout: {args.num_steps}")
    print(f"Batch size: {args.batch_size} timesteps (full sequence, no minibatch split)")

    # -- Agent (new: MAPPOLSTMAgent instead of MAPPOAgent) --
    agent = MAPPOLSTMAgent(
        num_agents, action_dim, args.obs_size, args.lstm_hidden_size
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {total_params:,}")

    # -- Rollout storage (from mappo_train.py) --
    obs = torch.zeros(
        (args.num_steps, num_agents, 3, args.obs_size, args.obs_size), device=device
    )
    actions = torch.zeros((args.num_steps, num_agents), device=device)
    logprobs = torch.zeros((args.num_steps, num_agents), device=device)
    rewards = torch.zeros((args.num_steps, num_agents), device=device)
    dones = torch.zeros((args.num_steps, num_agents), device=device)
    values = torch.zeros((args.num_steps, 1), device=device)

    # -- Initialize state (from cleanrl PPO-LSTM: LSTM state + initial obs) --
    global_step = 0
    start_time = time.time()

    obs_dict, _ = env.reset(seed=args.seed)
    next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)  # (1, N, 3, H, W)
    next_done = torch.zeros(1, num_agents, device=device)  # (1, N) for get_states
    next_lstm_state = agent.init_lstm_state(num_agents, device)  # ((1,N,128), (1,N,128))

    ep_rets = np.zeros(num_agents)
    ep_len = 0

    # ===================================================================
    # Step 3: Rollout collection + Step 4: GAE & PPO + Step 5: Logging
    # ===================================================================
    try:
        for iteration in range(1, args.num_iterations + 1):
            # LR annealing (from mappo_train.py)
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            # --- From cleanrl PPO-LSTM: save initial LSTM state for PPO replay ---
            # We need this to re-run the forward pass during the PPO update.
            initial_lstm_state = (
                next_lstm_state[0].clone(),
                next_lstm_state[1].clone(),
            )

            # ---------------------------------------------------------------
            # Step 3: Rollout collection with LSTM state threading
            # ---------------------------------------------------------------
            for step in range(args.num_steps):
                global_step += num_agents
                obs[step] = next_obs.squeeze(0)   # (N, 3, H, W)
                dones[step] = next_done.squeeze(0) # (N,)

                # --- From cleanrl PPO-LSTM: pass lstm_state to agent ---
                with torch.no_grad():
                    action, logprob, _, value, next_lstm_state = (
                        agent.get_action_and_value(
                            next_obs,         # (1, N, 3, H, W)
                            next_lstm_state,  # ((1,N,128), (1,N,128))
                            next_done,        # (1, N) — LSTM resets done agents
                        )
                    )
                    values[step] = value.squeeze(0)  # (1,) -> scalar
                actions[step] = action.squeeze(0)    # (N,)
                logprobs[step] = logprob.squeeze(0)  # (N,)

                # --- From mappo_train.py: PettingZoo step ---
                action_np = action.squeeze(0).cpu().numpy()
                actions_dict = {
                    agent_names[i]: int(action_np[i]) for i in range(num_agents)
                }

                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(
                    actions_dict
                )

                # Store rewards and dones (from mappo_train.py)
                for i, name in enumerate(agent_names):
                    rewards[step, i] = reward_dict.get(name, 0.0)

                # Episode tracking (from mappo_train.py)
                for i, name in enumerate(agent_names):
                    ep_rets[i] += reward_dict.get(name, 0.0)
                ep_len += 1

                # Check for episode end (from mappo_train.py)
                any_done = any(term_dict.values()) or any(trunc_dict.values())
                if any_done:
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

                    ep_rets = np.zeros(num_agents)
                    ep_len = 0
                    obs_dict, _ = env.reset()
                    # LSTM state is NOT reset here — get_states() handles it
                    # via the done flags at the next step.
                    next_done = torch.ones(1, num_agents, device=device)
                else:
                    next_done = torch.tensor(
                        [[float(term_dict.get(name, False) or trunc_dict.get(name, False))
                          for name in agent_names]],
                        device=device,
                    )  # (1, N)

                next_obs = preprocess_obs(obs_dict, agent_names, args.obs_size, device)

            # ---------------------------------------------------------------
            # Step 4: GAE computation (from mappo_train.py)
            #   Per-agent advantages using the centralized value function.
            # ---------------------------------------------------------------
            with torch.no_grad():
                # Bootstrap value from the last observation
                next_value = agent.get_value(
                    next_obs, next_lstm_state, next_done
                ).squeeze(0)  # (1,)

                advantages = torch.zeros_like(rewards, device=device)  # (T, N)
                for i in range(num_agents):
                    lastgaelam = 0.0
                    for t_step in reversed(range(args.num_steps)):
                        if t_step == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[0, i]
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
                            + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                returns = advantages + values.expand_as(advantages)  # (T, N)

            # ---------------------------------------------------------------
            # Step 4 (cont): PPO update — sequence-aware, no minibatch split
            #
            # From cleanrl PPO-LSTM: we must preserve temporal order so the
            # LSTM can reconstruct its states from initial_lstm_state.
            # Each epoch replays the FULL sequence (all num_steps timesteps).
            # ---------------------------------------------------------------
            clipfracs = []
            for epoch in range(args.update_epochs):
                # Forward pass over the full sequence (from cleanrl PPO-LSTM)
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs,                 # (T, N, 3, H, W) — full rollout buffer
                    initial_lstm_state,  # LSTM state from start of this iteration
                    dones,               # (T, N) — done flags for LSTM reset
                    actions.long(),      # (T, N) — replay stored actions
                )
                # newlogprob: (T, N), entropy: (T, N), newvalue: (T, 1)

                # -- Policy loss (from mappo_train.py, per-agent then averaged) --
                logratio = newlogprob - logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                # Normalize advantages across ALL agents (from mappo_train.py)
                adv_flat = advantages.reshape(-1)
                if args.norm_adv:
                    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
                adv_norm = adv_flat.reshape(args.num_steps, num_agents)

                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # -- Value loss (centralized, from mappo_train.py) --
                newvalue_flat = newvalue.squeeze(-1)            # (T,)
                returns_mean = returns.mean(dim=1)              # (T,) avg across agents
                values_flat = values.squeeze(-1)                # (T,)

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue_flat - returns_mean) ** 2
                    v_clipped = values_flat + torch.clamp(
                        newvalue_flat - values_flat,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns_mean) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue_flat - returns_mean) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # ---------------------------------------------------------------
            # Step 5: Logging (from mappo_train.py)
            # ---------------------------------------------------------------
            y_pred = values.squeeze(-1).cpu().numpy()
            y_true = returns.mean(dim=1).cpu().numpy()
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

            # Checkpoint saving (from mappo_train.py)
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
