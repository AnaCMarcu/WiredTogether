"""IC3Net + LLM Communication Agent for Multi-Agent RL.

Architecture:
    CNN Encoder (shared) -> GRU (IC3Net) -> Gate (when to communicate)
                                         -> Message Selector (what to say, LLM-grounded)
    Communication aggregation -> Actor (decentralized) + Critic (centralized)

The LLM communication channel uses a frozen codebook of 32 natural language messages
encoded by a sentence-transformers model. The agent learns a soft attention over this
codebook, giving communication semantic grounding without LLM inference during training.

Reference: IC3Net (ICLR 2019) - "Learning when to Communicate at Scale in
Multiagent Cooperative and Competitive Tasks"
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# ---------------------------------------------------------------------------
# Predefined message vocabulary for LLM-grounded communication
# ---------------------------------------------------------------------------
MESSAGES = [
    # Navigation / Location
    "I am here",
    "Come to me",
    "Go north",
    "Go south",
    "Go east",
    "Go west",
    "Follow me",
    "Wait here",
    # Resource gathering
    "I found wood",
    "I found stone",
    "I found iron ore",
    "I found diamonds",
    "I need food",
    "I have food for you",
    "Gather resources",
    "Mine this block",
    # Combat / Danger
    "Danger ahead",
    "Monster nearby",
    "I need help",
    "Area is safe",
    "Attacking enemy",
    "Retreat now",
    "Defend this position",
    "I am under attack",
    # Crafting / Tools
    "Craft a pickaxe",
    "I made a tool for you",
    "Use this item",
    "Place a block",
    # General coordination
    "Yes",
    "No",
    "Ready",
    "Task complete",
]


# ---------------------------------------------------------------------------
# Communication state that persists across timesteps
# ---------------------------------------------------------------------------
class CommState(NamedTuple):
    hidden: torch.Tensor        # (batch, num_agents, gru_hidden_dim)
    prev_messages: torch.Tensor  # (batch, num_agents, msg_embed_dim)
    prev_gate: torch.Tensor     # (batch, num_agents, 1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------------
# IC3Net + LLM Communication Agent
# ---------------------------------------------------------------------------
class IC3NetLLMAgent(nn.Module):
    """Multi-agent architecture with IC3Net gated communication and
    LLM-grounded message passing.

    Args:
        num_agents: Number of cooperating agents.
        action_dim: Size of the discrete action space (17 for OpenWorld).
        obs_size: Square observation size after resize (default 84).
        gru_hidden_dim: GRU hidden state dimension (default 256).
        msg_embed_dim: Message embedding dimension, must match the sentence
            transformer output (default 384 for all-MiniLM-L6-v2).
        num_messages: Number of discrete messages in the codebook (default 32).
    """

    def __init__(
        self,
        num_agents: int,
        action_dim: int,
        obs_size: int = 84,
        gru_hidden_dim: int = 256,
        msg_embed_dim: int = 384,
        num_messages: int = 32,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.gru_hidden_dim = gru_hidden_dim
        self.msg_embed_dim = msg_embed_dim
        self.num_messages = num_messages

        # === 1. Shared CNN Encoder ===
        # (3, 84, 84) -> 512-dim feature vector
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

        # === 2. IC3Net GRU ===
        self.gru = nn.GRUCell(512, gru_hidden_dim)

        # === 3. IC3Net Gate Network ===
        # Produces a scalar gate per agent: should I communicate?
        self.gate_net = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # === 4. LLM Message Selector ===
        # Maps hidden state -> distribution over K predefined messages
        self.message_selector = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_messages),
        )

        # Frozen message codebook: (num_messages, msg_embed_dim)
        # Populated by load_message_codebook() from sentence-transformers
        self.register_buffer(
            "message_codebook", torch.zeros(num_messages, msg_embed_dim)
        )

        # === 5. Communication Aggregation ===
        # Projects incoming IC3Net hidden + message embeddings
        self.comm_proj = nn.Linear(gru_hidden_dim + msg_embed_dim, gru_hidden_dim)

        # === 6. Decentralized Actor (shared weights across agents) ===
        self.actor = layer_init(nn.Linear(gru_hidden_dim, action_dim), std=0.01)

        # === 7. Centralized Critic ===
        self.critic = nn.Sequential(
            layer_init(nn.Linear(gru_hidden_dim * num_agents, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def init_comm_state(self, batch_size: int, device: torch.device) -> CommState:
        """Create zero-initialized communication state."""
        return CommState(
            hidden=torch.zeros(batch_size, self.num_agents, self.gru_hidden_dim, device=device),
            prev_messages=torch.zeros(batch_size, self.num_agents, self.msg_embed_dim, device=device),
            prev_gate=torch.zeros(batch_size, self.num_agents, 1, device=device),
        )

    def load_message_codebook(self, model_name: str = "all-MiniLM-L6-v2"):
        """Encode predefined messages using a sentence-transformer.

        The sentence-transformer model is loaded, used to encode MESSAGES,
        then discarded. Only the frozen embeddings are kept as a buffer.
        """
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(MESSAGES[: self.num_messages], convert_to_tensor=True)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.message_codebook.copy_(embeddings.to(self.message_codebook.device))
        del model

    # ------------------------------------------------------------------
    # Core forward pass with communication
    # ------------------------------------------------------------------
    def _forward_comm(self, all_obs, comm_state, dones=None):
        """Full forward pass with IC3Net + LLM communication.

        Args:
            all_obs:    (batch, N, 3, H, W) float tensor
            comm_state: CommState
            dones:      (batch, N) or None - resets hidden for done agents

        Returns:
            agent_features: (batch, N, gru_hidden_dim) post-communication
            new_comm_state: CommState
            gate_values:    (batch, N, 1) for logging
            msg_logits:     (batch, N, num_messages) for auxiliary loss
        """
        batch_size = all_obs.shape[0]
        N = self.num_agents

        # Step 1: CNN encoding
        flat_obs = all_obs.reshape(-1, *all_obs.shape[2:])  # (B*N, 3, H, W)
        visual_feats = self.encoder(flat_obs / 255.0)        # (B*N, 512)

        # Step 2: Reset hidden states for done agents
        h_prev = comm_state.hidden.reshape(batch_size * N, self.gru_hidden_dim)
        if dones is not None:
            done_mask = 1.0 - dones.reshape(batch_size * N, 1).float()
            h_prev = h_prev * done_mask

        # Step 3: GRU update
        h_new = self.gru(visual_feats, h_prev)               # (B*N, 256)
        h_new_shaped = h_new.reshape(batch_size, N, self.gru_hidden_dim)

        # Step 4: Gate computation (IC3Net)
        gate_logits = self.gate_net(h_new)                    # (B*N, 1)
        gate_probs = torch.sigmoid(gate_logits)
        if self.training:
            # Straight-through estimator: hard forward, soft backward
            gate_hard = (gate_probs > 0.5).float()
            gate = gate_probs + (gate_hard - gate_probs).detach()
        else:
            gate = (gate_probs > 0.5).float()
        gate_shaped = gate.reshape(batch_size, N, 1)          # (B, N, 1)

        # Step 5: LLM message generation (soft attention over codebook)
        msg_logits = self.message_selector(h_new)             # (B*N, K)
        msg_probs = torch.softmax(msg_logits, dim=-1)
        # Soft attention: (B*N, K) @ (K, 384) -> (B*N, 384)
        msg_embed = torch.matmul(msg_probs, self.message_codebook)
        msg_embed_shaped = msg_embed.reshape(batch_size, N, self.msg_embed_dim)
        msg_logits_shaped = msg_logits.reshape(batch_size, N, self.num_messages)

        # Step 6: Gated broadcast and aggregation
        gated_hidden = h_new_shaped * gate_shaped             # (B, N, 256)
        gated_msg = msg_embed_shaped * gate_shaped            # (B, N, 384)

        # Mean-pool messages from OTHER agents (exclude self)
        comm_hidden = torch.zeros_like(h_new_shaped)
        comm_msg = torch.zeros_like(msg_embed_shaped)
        for i in range(N):
            others = [j for j in range(N) if j != i]
            comm_hidden[:, i] = gated_hidden[:, others].mean(dim=1)
            comm_msg[:, i] = gated_msg[:, others].mean(dim=1)

        # Step 7: Fuse communication into agent features (residual)
        comm_concat = torch.cat([comm_hidden, comm_msg], dim=-1)  # (B, N, 640)
        comm_proj = self.comm_proj(
            comm_concat.reshape(-1, self.gru_hidden_dim + self.msg_embed_dim)
        ).reshape(batch_size, N, self.gru_hidden_dim)
        agent_features = h_new_shaped + comm_proj             # (B, N, 256)

        new_comm_state = CommState(
            hidden=h_new_shaped,
            prev_messages=msg_embed_shaped,
            prev_gate=gate_shaped,
        )

        return agent_features, new_comm_state, gate_shaped, msg_logits_shaped

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_value(self, all_obs, comm_state=None, dones=None):
        """Centralized value function with communication.

        Args:
            all_obs:    (batch, N, 3, H, W)
            comm_state: CommState or None
            dones:      (batch, N) or None

        Returns:
            value:          (batch, 1)
            new_comm_state: CommState
        """
        batch_size = all_obs.shape[0]
        if comm_state is None:
            comm_state = self.init_comm_state(batch_size, all_obs.device)

        agent_features, new_comm_state, _, _ = self._forward_comm(
            all_obs, comm_state, dones
        )
        global_features = agent_features.reshape(batch_size, -1)
        value = self.critic(global_features)
        return value, new_comm_state

    def get_action_and_value(self, all_obs, comm_state=None, dones=None, actions=None):
        """Full forward pass for rollout collection and PPO training.

        Args:
            all_obs:    (batch, N, 3, H, W)
            comm_state: CommState or None
            dones:      (batch, N) or None
            actions:    (batch, N) optional pre-specified actions

        Returns:
            actions:        (batch, N)
            log_probs:      (batch, N)
            entropy:        (batch, N)
            values:         (batch, 1)
            new_comm_state: CommState
            comm_info:      dict with gate_values and msg_logits
        """
        batch_size = all_obs.shape[0]
        N = self.num_agents
        if comm_state is None:
            comm_state = self.init_comm_state(batch_size, all_obs.device)

        agent_features, new_comm_state, gate_values, msg_logits = self._forward_comm(
            all_obs, comm_state, dones
        )

        # Decentralized actor
        flat_features = agent_features.reshape(batch_size * N, -1)
        flat_logits = self.actor(flat_features)
        logits = flat_logits.reshape(batch_size, N, -1)
        probs = Categorical(logits=logits)

        if actions is None:
            actions = probs.sample()

        log_probs = probs.log_prob(actions)
        entropy = probs.entropy()

        # Centralized critic
        global_features = agent_features.reshape(batch_size, -1)
        values = self.critic(global_features)

        comm_info = {
            "gate_values": gate_values,
            "msg_logits": msg_logits,
        }

        return actions, log_probs, entropy, values, new_comm_state, comm_info

    def decode_messages(self, msg_logits):
        """Decode message logits to human-readable strings (for evaluation/logging).

        Args:
            msg_logits: (batch, N, num_messages)

        Returns:
            List of lists of message strings.
        """
        indices = msg_logits.argmax(dim=-1)  # (batch, N)
        result = []
        for b in range(indices.shape[0]):
            msgs = [MESSAGES[idx] for idx in indices[b].tolist()]
            result.append(msgs)
        return result


# ---------------------------------------------------------------------------
# MAPPO + LSTM Agent (no communication, temporal memory via LSTM)
# ---------------------------------------------------------------------------
class MAPPOLSTMAgent(nn.Module):
    """MAPPO agent with LSTM temporal memory.

    Architecture:
        CNN Encoder (shared) -> LSTM -> Actor (decentralized) + Critic (centralized)

    The LSTM processes each agent independently (parameter sharing) and maintains
    per-agent hidden states across timesteps.  Done flags reset individual agent
    states so the LSTM doesn't carry stale information across episode boundaries.

    Combines:
      - CNN encoder + CTDE structure from MAPPOAgent (mappo_train.py)
      - LSTM state management from cleanrl's PPO-LSTM (cleanrl_ppo_lstm_train.py)

    Args:
        num_agents:       Number of cooperating agents.
        action_dim:       Size of the discrete action space (17 for OpenWorld).
        obs_size:         Square observation size after resize (default 84).
        lstm_hidden_size: LSTM hidden state dimension (default 128).
    """

    def __init__(
        self,
        num_agents: int,
        action_dim: int,
        obs_size: int = 84,
        lstm_hidden_size: int = 128,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.lstm_hidden_size = lstm_hidden_size

        # === 1. Shared CNN Encoder (from MAPPOAgent) ===
        # (3, 84, 84) -> 512-dim feature vector
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

        # === 2. LSTM (from cleanrl PPO-LSTM) ===
        # Processes 512-dim CNN features sequentially per agent
        self.lstm = nn.LSTM(512, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # === 3. Decentralized Actor (from MAPPOAgent) ===
        self.actor = layer_init(nn.Linear(lstm_hidden_size, action_dim), std=0.01)

        # === 4. Centralized Critic (from MAPPOAgent) ===
        # Concatenates all agents' LSTM features -> single value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size * num_agents, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init_lstm_state(self, num_agents: int, device: torch.device):
        """Create zero-initialized LSTM state.

        Returns:
            (h0, c0) each of shape ``(1, num_agents, lstm_hidden_size)``
        """
        return (
            torch.zeros(1, num_agents, self.lstm_hidden_size, device=device),
            torch.zeros(1, num_agents, self.lstm_hidden_size, device=device),
        )

    # ------------------------------------------------------------------
    # Core: CNN + step-by-step LSTM with done masking
    # ------------------------------------------------------------------
    def get_states(self, all_obs, lstm_state, done):
        """Process observations through CNN + LSTM with done-flag state reset.

        Adapted from cleanrl's PPO-LSTM ``Agent.get_states()``:
        the CNN extracts features for all agents at all timesteps, then the
        LSTM processes one timestep at a time, zeroing the hidden/cell state
        for any agent whose ``done`` flag is 1.

        Args:
            all_obs:    ``(T, N, 3, H, W)`` float tensor.
                        T = num_steps during PPO update, T = 1 during rollout.
            lstm_state: ``((1, N, H_lstm), (1, N, H_lstm))``
            done:       ``(T, N)`` done flags (1.0 = episode ended)

        Returns:
            features:       ``(T*N, lstm_hidden_size)`` post-LSTM features
            new_lstm_state: ``((1, N, H_lstm), (1, N, H_lstm))``
        """
        T, N = all_obs.shape[0], all_obs.shape[1]

        # CNN feature extraction for all timesteps × agents at once
        flat_obs = all_obs.reshape(T * N, *all_obs.shape[2:])   # (T*N, 3, H, W)
        features = self.encoder(flat_obs / 255.0)                # (T*N, 512)

        # Reshape for step-by-step LSTM: (T, N, 512)
        features = features.reshape(T, N, self.lstm.input_size)
        done = done.reshape(T, N)

        # Process one timestep at a time (from cleanrl PPO-LSTM get_states)
        new_hidden = []
        for h, d in zip(features, done):
            # h: (N, 512)  — one timestep, all agents
            # d: (N,)      — done flags per agent
            #
            # Reset LSTM state for done agents: (1 - done) * state
            h_masked = (1.0 - d).view(1, -1, 1) * lstm_state[0]
            c_masked = (1.0 - d).view(1, -1, 1) * lstm_state[1]

            h_out, lstm_state = self.lstm(
                h.unsqueeze(0),          # (1, N, 512)
                (h_masked, c_masked),
            )
            new_hidden.append(h_out)     # each is (1, N, hidden)

        # Stack and flatten: (T, N, hidden) -> (T*N, hidden)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_value(self, all_obs, lstm_state, done):
        """Centralized value function.

        Args:
            all_obs:    ``(T, N, 3, H, W)``
            lstm_state: ``((1, N, H_lstm), (1, N, H_lstm))``
            done:       ``(T, N)``

        Returns:
            ``(T, 1)`` value tensor
        """
        T, N = all_obs.shape[0], all_obs.shape[1]
        features, _ = self.get_states(all_obs, lstm_state, done)  # (T*N, hidden)
        features = features.reshape(T, N, self.lstm_hidden_size)
        global_features = features.reshape(T, -1)                 # (T, N*hidden)
        return self.critic(global_features)                       # (T, 1)

    def get_action_and_value(self, all_obs, lstm_state, done, actions=None):
        """Forward pass for both actor and critic.

        Args:
            all_obs:    ``(T, N, 3, H, W)``  (T=1 during rollout, T=num_steps in PPO)
            lstm_state: ``((1, N, H_lstm), (1, N, H_lstm))``
            done:       ``(T, N)``
            actions:    ``(T, N)`` optional pre-specified actions

        Returns:
            actions:        ``(T, N)``
            log_probs:      ``(T, N)``
            entropy:        ``(T, N)``
            values:         ``(T, 1)``
            new_lstm_state: ``((1, N, H_lstm), (1, N, H_lstm))``
        """
        T, N = all_obs.shape[0], all_obs.shape[1]

        features, new_lstm_state = self.get_states(all_obs, lstm_state, done)
        # features: (T*N, hidden)

        # Decentralized actor (shared weights across agents)
        logits = self.actor(features)                   # (T*N, action_dim)
        logits = logits.reshape(T, N, -1)               # (T, N, action_dim)
        probs = Categorical(logits=logits)

        if actions is None:
            actions = probs.sample()                     # (T, N)

        log_probs = probs.log_prob(actions)              # (T, N)
        entropy = probs.entropy()                        # (T, N)

        # Centralized critic (sees all agents' features)
        features = features.reshape(T, N, self.lstm_hidden_size)
        global_features = features.reshape(T, -1)       # (T, N*hidden)
        values = self.critic(global_features)            # (T, 1)

        return actions, log_probs, entropy, values, new_lstm_state
