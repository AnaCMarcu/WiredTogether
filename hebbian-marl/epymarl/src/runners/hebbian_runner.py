"""HebbianRunner — EpisodeRunner subclass that runs the Hebbian update each step.

Implements HEBBIAN_MARL_PLAN.md integration point (a): reward diffusion.
After every env step:

  1. Read ``info['positions']`` and ``info['comm_events']`` produced by
     the comm-augmented LBF wrapper (or supplied as no-ops by stock envs).
  2. Call ``self.hebbian.update(...)`` to advance the social graph.
  3. If ``args.hebbian.reward_diffusion`` is True, replace the per-agent
     reward stored in the batch with ``self.hebbian.diffuse_rewards(...)``.

The runner is selected via ``runner: hebbian`` in the EPyMARL config.
It is a strict no-op when ``args.hebbian.enabled = False`` — behaviour is
bitwise-identical to ``EpisodeRunner`` for the same seed. The
``test_ablation_flags_passthrough.py`` test enforces this.

Requires per-agent rewards (``common_reward: False``) so that reward
diffusion has anything to do. The runner asserts this at construction.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import List, Optional, Tuple

import numpy as np

from hebbian_module import HebbianConfig, HebbianSocialGraph, set_graph
from runners.episode_runner import EpisodeRunner


def _build_hebbian_config(args) -> HebbianConfig:
    """Pull the ``hebbian:`` sub-config off ``args`` and instantiate."""
    hebbian_args = getattr(args, "hebbian", None)
    if hebbian_args is None:
        return HebbianConfig(enabled=False)

    # Sacred presents nested dicts as SimpleNamespace OR plain dict.
    def get(k, default):
        if isinstance(hebbian_args, dict):
            return hebbian_args.get(k, default)
        return getattr(hebbian_args, k, default)

    return HebbianConfig(
        enabled=bool(get("enabled", False)),
        num_agents=int(get("num_agents", getattr(args, "n_agents", 3))),
        interaction_radius=float(get("interaction_radius", 5.0)),
        engagement_reward_weight=float(get("engagement_reward_weight", 0.5)),
        communication_coactivity_bonus=float(get("communication_coactivity_bonus", 0.5)),
        ltp_lr=float(get("ltp_lr", 0.01)),
        ltd_lr=float(get("ltd_lr", 0.005)),
        ltd_threshold=float(get("ltd_threshold", 0.1)),
        base_ltp=float(get("base_ltp", 0.005)),
        decay=float(get("decay", 0.0003)),
        modulation_beta=float(get("modulation_beta", 1.0)),
        ltd_sustained_lr=float(get("ltd_sustained_lr", 0.002)),
        failure_memory_window=int(get("failure_memory_window", 50)),
        social_replay_rho=float(get("social_replay_rho", 0.0)),
        reward_diffusion_gamma=float(get("reward_diffusion_gamma", 0.2)),
        init_weight=float(get("init_weight", 0.1)),
        log_graph_every=int(get("log_graph_every", 50)),
        uniform_weights=bool(get("uniform_weights", False)),
        disable_coactivity_gate=bool(get("disable_coactivity_gate", False)),
    )


def _flag(args, name: str, default: bool) -> bool:
    """Read a Hebbian toggle flag (`reward_diffusion`, ...) off args.hebbian."""
    hebbian_args = getattr(args, "hebbian", None)
    if hebbian_args is None:
        return default
    if isinstance(hebbian_args, dict):
        return bool(hebbian_args.get(name, default))
    return bool(getattr(hebbian_args, name, default))


class HebbianRunner(EpisodeRunner):
    """EpisodeRunner with a Hebbian update + optional reward diffusion."""

    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.hebbian_config = _build_hebbian_config(args)
        self.hebbian = HebbianSocialGraph(self.hebbian_config)
        self.reward_diffusion = _flag(args, "reward_diffusion", False)

        if self.hebbian_config.enabled:
            assert not getattr(args, "common_reward", True), (
                "HebbianRunner requires `common_reward: False` so it has "
                "per-agent rewards to update and diffuse."
            )

        # Register the graph for the learner to pick up via singleton.
        set_graph(self.hebbian)

        # ── bonds.jsonl + episode-metrics logging ──
        # Snapshot intervals (env steps) and counters.
        self._bond_snapshot_every = int(getattr(self.hebbian_config, "log_graph_every", 50)) * 100
        self._last_bond_snapshot_t = -1
        self._bonds_file = None              # opened lazily once we know the run dir
        # args.n_agents isn't populated yet at __init__ time (set later from
        # env_info). Use env.n_agents which is available.
        n = self.env.n_agents
        self._signal_counts = np.zeros((n, n), dtype=np.int64)
        self._joint_load_count = 0
        self._time_to_first_coop_load = -1

    def _open_bonds_file(self):
        """Open bonds.jsonl in the local results path. Lazy because sacred
        sets up its directory after the runner is constructed."""
        if self._bonds_file is not None or not self.hebbian_config.enabled:
            return
        try:
            base = getattr(self.args, "local_results_path", "results")
            run_id = str(getattr(self.args, "name", "unknown"))
            seed = str(getattr(self.args, "seed", "0"))
            out_dir = os.path.join(base, "bonds", run_id)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"seed_{seed}.jsonl")
            self._bonds_file = open(path, "w", buffering=1)
        except Exception as e:
            self.logger.console_logger.warning(
                f"HebbianRunner: failed to open bonds.jsonl ({e}); bond snapshots disabled."
            )
            self._bonds_file = False  # sentinel: don't retry

    def _maybe_snapshot_bonds(self, force: bool = False):
        """Write a bond snapshot every `_bond_snapshot_every` env steps.

        ``force=True`` writes immediately (used once at startup so plots
        have a baseline data point).
        """
        if not self.hebbian_config.enabled:
            return
        if (not force) and (self.t_env - self._last_bond_snapshot_t < self._bond_snapshot_every):
            return
        self._open_bonds_file()
        if not self._bonds_file:
            return
        metrics = self.hebbian.get_graph_metrics()
        if not metrics:
            return
        W = np.array(metrics["W"], dtype=np.float32)
        record = {
            "t_env": int(self.t_env),
            "step": int(self.hebbian._step_count),
            "mean_bond_strength": float(metrics["mean_bond_strength"]),
            "sparsity": float(metrics["sparsity"]),
            "asymmetry_frob": float(np.linalg.norm(W - W.T)),
            "out_strength": list(metrics["per_agent_out_strength"]),
            "W": metrics["W"],
        }
        self._bonds_file.write(json.dumps(record) + "\n")
        self._last_bond_snapshot_t = self.t_env

    # ── helpers ──

    def _per_agent_rewards(self, reward) -> List[float]:
        if isinstance(reward, Iterable):
            return [float(r) for r in reward]
        # common_reward path — we asserted this off when enabled,
        # but be defensive for the disabled case.
        return [float(reward)] * self.args.n_agents

    @staticmethod
    def _maybe_positions(env_info) -> List[Optional[Tuple[float, float, float]]]:
        positions = env_info.get("positions") if isinstance(env_info, dict) else None
        if positions is None:
            return [None] * 0  # caller decides default
        return [tuple(p) for p in positions]

    @staticmethod
    def _maybe_comm_events(env_info) -> Optional[List[Tuple[int, int]]]:
        if not isinstance(env_info, dict):
            return None
        return env_info.get("comm_events")

    # ── run override ──
    #
    # Copied from EpisodeRunner.run with two added blocks marked
    # ``HEBBIAN-A:``. Kept the rest verbatim so future EPyMARL upstream
    # diffs are easy to spot.

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()

            # ── HEBBIAN-A: advance the social graph after the step ──
            if self.hebbian_config.enabled:
                raw_per_agent = self._per_agent_rewards(reward)
                positions = self._maybe_positions(env_info)
                if not positions:
                    positions = [None] * self.args.n_agents
                comm_events = self._maybe_comm_events(env_info)
                self.hebbian.update(
                    positions=positions,
                    step_rewards=raw_per_agent,
                    advantages=None,
                    comm_events=comm_events,
                )
                # Track per-episode signal-action counts (sender x receiver matrix).
                if comm_events:
                    for sender, receiver in comm_events:
                        self._signal_counts[sender, receiver] += 1
                # Optional reward diffusion BEFORE the batch sees the reward
                if self.reward_diffusion and isinstance(reward, Iterable):
                    reward = self.hebbian.diffuse_rewards(raw_per_agent)

            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
                # skip non-scalar info keys we added
                if not isinstance(env_info.get(k, 0), (list, tuple, dict))
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # ── HEBBIAN-A: log the bond + signal metrics at runner-log intervals ──
        if (
            self.hebbian_config.enabled
            and not test_mode
            and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval)
        ):
            metrics = self.hebbian.get_graph_metrics()
            if metrics:
                self.logger.log_stat(
                    "hebbian/mean_bond_strength",
                    metrics["mean_bond_strength"],
                    self.t_env,
                )
                self.logger.log_stat(
                    "hebbian/sparsity", metrics["sparsity"], self.t_env
                )
                W = np.array(metrics["W"], dtype=np.float32)
                self.logger.log_stat(
                    "hebbian/asymmetry_frob",
                    float(np.linalg.norm(W - W.T)),
                    self.t_env,
                )
            # Total signal actions this episode + per-agent count
            self.logger.log_stat(
                "hebbian/signal_total",
                int(self._signal_counts.sum()),
                self.t_env,
            )
            for i in range(self.env.n_agents):
                self.logger.log_stat(
                    f"hebbian/signal_from_agent_{i}",
                    int(self._signal_counts[i].sum()),
                    self.t_env,
                )

        # Snapshot the W matrix to bonds.jsonl on the configured cadence.
        # Force a snapshot at the start (t_env still 0 or near it) so plots
        # have a baseline data point.
        force = self._last_bond_snapshot_t < 0
        self._maybe_snapshot_bonds(force=force)

        # Reset per-episode signal counters for the next episode
        if not test_mode:
            self._signal_counts.fill(0)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch
