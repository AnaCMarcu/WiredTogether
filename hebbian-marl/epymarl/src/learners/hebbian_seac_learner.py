"""HebbianSEACLearner — PPO learner with optional cross-agent experience sharing.

Implements HEBBIAN_MARL_PLAN.md integration (b): each agent i's PPO loss
is augmented with an off-policy term that learns from teammate j's
transitions, importance-sampled with ``π_i_new(a_j|o_j) / π_j_old(a_j|o_j)``.

Two ablation knobs:

  - ``uniform_sharing: True``  → SEAC baseline: uniform weight over teammates.
  - ``weighted_sharing: True`` → Hebbian-weighted: teammate j contributes
    in proportion to ``W̄[i,j]`` from the active HebbianSocialGraph.

Both flags False reduces this learner to standard PPO (own data only).
Exactly one of the two should be set when sharing is desired.

**IS correction is present from the first commit** (HEBBIAN_MARL_PLAN.md
hard constraint #12). The ratio uses ``self.old_mac`` (already a snapshot
maintained by PPOLearner) for ``π_j_old`` — the policy that actually
produced the action ``a_j``.

Cross-agent forward (obs-swap)
------------------------------
To compute ``π_i_new(. | o_j)``, the learner constructs a "swapped" input
where every agent slot receives agent j's observation. The standard MAC
forward then evaluates each agent k's policy at o_j; slot i gives
exactly ``π_i_new(. | o_j)``. This works uniformly for:

  - ``BasicMAC`` (shared parameters + obs_agent_id): slot i has
    ``(o_j, id_i)``, so the shared net produces ``π_i_new(. | o_j)``.
  - ``NonSharedMAC``: slot i routes through ``agents[i]`` regardless of
    which obs is fed in.

Hidden state caveat: the cross-agent forward initialises hidden state to
zeros every call. This is correct for feed-forward agents
(``use_rnn: False``) and an approximation for RNN agents. Configs for
the `seac` / `hebb_s` variants therefore default to feed-forward.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch as th

from hebbian_module import get_graph
from learners.ppo_learner import PPOLearner


def _flag(args, name: str, default: bool) -> bool:
    """Read a learner toggle off the top-level args (e.g. ``args.uniform_sharing``).

    Falls back to ``args.hebbian.<name>`` and finally to the default.
    """
    if hasattr(args, name):
        return bool(getattr(args, name))
    hebbian = getattr(args, "hebbian", None)
    if hebbian is None:
        return default
    if isinstance(hebbian, dict):
        return bool(hebbian.get(name, default))
    return bool(getattr(hebbian, name, default))


def _scalar(args, name: str, default: float) -> float:
    if hasattr(args, name):
        return float(getattr(args, name))
    hebbian = getattr(args, "hebbian", None)
    if hebbian is None:
        return default
    if isinstance(hebbian, dict):
        return float(hebbian.get(name, default))
    return float(getattr(hebbian, name, default))


class HebbianSEACLearner(PPOLearner):
    """PPO + Hebbian-weighted or uniform off-policy SEAC sharing."""

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)

        self.uniform_sharing = _flag(args, "uniform_sharing", False)
        self.weighted_sharing = _flag(args, "weighted_sharing", False)
        # SEAC's λ in `L_i = L_i_own + λ * Σ_j w_ij * L_ij_cross`
        self.lambda_share = _scalar(args, "lambda_share", 0.5)

        if self.uniform_sharing and self.weighted_sharing:
            raise ValueError(
                "`uniform_sharing` and `weighted_sharing` are mutually exclusive — "
                "set exactly one."
            )

        if self.weighted_sharing and get_graph() is None:
            # Not fatal at construct time — the runner may register the graph
            # later. But warn loudly.
            self.logger.console_logger.warning(
                "HebbianSEACLearner: weighted_sharing=True but no Hebbian graph "
                "is registered. Configure `runner: hebbian` and `hebbian.enabled: True`."
            )

    # ──────────────────────────────────────────────────────────────────
    # Cross-agent forward (obs-swap)
    # ──────────────────────────────────────────────────────────────────

    def _build_swapped_inputs(self, batch, source_agent: int, t: int) -> th.Tensor:
        """Construct MAC input tensor where every agent slot has obs[source_agent].

        Mirrors BasicMAC._build_inputs but substitutes the obs.
        """
        bs = batch.batch_size
        obs_source = batch["obs"][:, t, source_agent, :]  # (bs, obs_dim)
        swapped_obs = obs_source.unsqueeze(1).expand(bs, self.n_agents, -1)

        inputs = [swapped_obs]
        if getattr(self.args, "obs_last_action", False):
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if getattr(self.args, "obs_agent_id", False):
            id_onehot = (
                th.eye(self.n_agents, device=swapped_obs.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )
            inputs.append(id_onehot)

        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _cross_agent_forward(self, mac, batch, source_agent: int, t: int) -> th.Tensor:
        """Forward `mac` with every agent slot receiving obs[source_agent].

        Returns ``(bs, n_agents, n_actions)`` where slot k is
        ``π_k(. | o_source_agent)``.
        """
        bs = batch.batch_size
        inputs = self._build_swapped_inputs(batch, source_agent, t)

        # Zero hidden state — correct for feed-forward, approximation for RNN.
        # See class docstring for the caveat.
        hidden = mac.agent.init_hidden()
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0).expand(bs, self.n_agents, -1).contiguous()
        elif hidden.dim() == 2:
            hidden = hidden.unsqueeze(0).expand(bs, -1, -1).contiguous()

        agent_outs, _ = mac.agent(inputs, hidden)

        # Mask + softmax — mirror BasicMAC.forward
        if mac.agent_output_type == "pi_logits":
            avail = batch["avail_actions"][:, t].reshape(bs * self.n_agents, -1)
            if getattr(self.args, "mask_before_softmax", True):
                agent_outs[avail == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(bs, self.n_agents, -1)

    # ──────────────────────────────────────────────────────────────────
    # Sharing weights
    # ──────────────────────────────────────────────────────────────────

    def _sharing_weights(self) -> np.ndarray:
        """Return per-pair sharing weights ``W̃[i,j]`` (no diagonal, sums per row).

        - ``weighted_sharing``: rows of ``W̄`` from the Hebbian graph.
        - ``uniform_sharing``: ``1 / (N - 1)`` off-diagonal, 0 on diagonal.
        - otherwise: zeros (no cross-agent term).
        """
        N = self.n_agents
        W = np.zeros((N, N), dtype=np.float32)
        if self.weighted_sharing:
            graph = get_graph()
            if graph is not None and graph.config.enabled:
                for i in range(N):
                    W[i] = graph.get_normalized_weights(i)
                return W
            # Fallback: if graph missing, behave like uniform so training continues
            # (the constructor already warned).
        if self.uniform_sharing or self.weighted_sharing:
            for i in range(N):
                for j in range(N):
                    if i != j:
                        W[i, j] = 1.0 / (N - 1)
        return W

    # ──────────────────────────────────────────────────────────────────
    # train() override — copy of PPOLearner.train with the SEAC term added.
    # ──────────────────────────────────────────────────────────────────

    def train(self, batch, t_env: int, episode_num: int):
        # Identical preamble to PPOLearner.train ────────────────────────
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # π_old for IS denominator
        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            old_mac_out.append(self.old_mac.forward(batch, t=t))
        old_mac_out = th.stack(old_mac_out, dim=1)
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)  # (bs, T, n_agents)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # Are we doing cross-agent sharing at all?
        do_share = self.uniform_sharing or self.weighted_sharing
        share_w = self._sharing_weights() if do_share else None  # (N, N)

        eps = self.args.eps_clip
        T = batch.max_seq_length - 1
        bs = batch.batch_size

        last_pg_loss = None
        last_cross_loss = None

        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(T):
                mac_out.append(self.mac.forward(batch, t=t))
            mac_out = th.stack(mac_out, dim=1)

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()

            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            # ── Own-data PPO loss (standard) ──
            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - eps, 1 + eps) * advantages
            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            own_loss_terms = (
                th.min(surr1, surr2) + self.args.entropy_coef * entropy
            ) * mask
            pg_loss = -own_loss_terms.sum() / mask.sum()

            # ── SEAC cross-agent term ──
            cross_loss = th.tensor(0.0, device=pi.device)
            if do_share:
                # For each source agent j, do one obs-swap forward producing
                # π_i(. | o_j) for every i in slot i.
                action_idx = actions.squeeze(-1)  # (bs, T, n_agents)
                lambda_share = self.lambda_share

                cross_loss_accum = th.zeros_like(mask)  # (bs, T, n_agents)

                for j in range(self.n_agents):
                    # If no target i has nonzero weight for source j, skip.
                    col = share_w[:, j]
                    if not np.any(col > 0):
                        continue

                    # Build π_i(. | o_j) for all i across all timesteps.
                    pi_at_oj_per_t = []
                    for t in range(T):
                        pi_at_oj_per_t.append(
                            self._cross_agent_forward(self.mac, batch, j, t)
                        )
                    pi_at_oj = th.stack(pi_at_oj_per_t, dim=1)  # (bs, T, n_agents, n_actions)

                    # Gather π_i(a_j | o_j) for each i
                    a_j = action_idx[:, :, j].unsqueeze(-1).unsqueeze(-1)  # (bs, T, 1, 1)
                    a_j = a_j.expand(-1, -1, self.n_agents, -1)            # (bs, T, n_agents, 1)
                    pi_i_at_oj_taken = th.gather(pi_at_oj, dim=3, index=a_j).squeeze(3)
                    log_pi_i_at_oj_taken = th.log(pi_i_at_oj_taken + 1e-10)

                    # π_j_old(a_j | o_j) — slot j of old_pi_taken
                    log_pi_j_old = old_log_pi_taken[:, :, j].detach()           # (bs, T)
                    A_j = advantages[:, :, j]                                   # (bs, T)
                    mask_j = mask[:, :, j]                                      # (bs, T)

                    for i in range(self.n_agents):
                        w_ij = float(share_w[i, j])
                        if i == j or w_ij <= 0.0:
                            continue
                        log_pi_i = log_pi_i_at_oj_taken[:, :, i]                # (bs, T)
                        is_ratio = th.exp(log_pi_i - log_pi_j_old)
                        s1 = is_ratio * A_j
                        s2 = th.clamp(is_ratio, 1 - eps, 1 + eps) * A_j
                        cross_term = th.min(s1, s2) * mask_j
                        cross_loss_accum[:, :, i] = (
                            cross_loss_accum[:, :, i] + w_ij * cross_term
                        )

                cross_loss = -(lambda_share * cross_loss_accum).sum() / mask.sum()

            total_loss = pg_loss + cross_loss

            # ── Optimise agents ──
            self.agent_optimiser.zero_grad()
            total_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

            last_pg_loss = pg_loss
            last_cross_loss = cross_loss

        self.old_mac.load_state(self.mac)

        # Target critic updates — same as parent
        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # Logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )
            self.logger.log_stat(
                "advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", last_pg_loss.item(), t_env)
            if do_share:
                self.logger.log_stat(
                    "seac_cross_loss", float(last_cross_loss.item()), t_env
                )
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.log_stats_t = t_env
