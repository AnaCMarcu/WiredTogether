"""HebbianParallelRunner — ParallelRunner subclass with Hebbian social graph.

Mirrors ``HebbianRunner`` (the EpisodeRunner subclass) but for MAPPO-style
parallel rollouts. After each parallel env-step:

  1. For every still-running env, read ``info['positions']`` and
     ``info['comm_events']`` from the comm-augmented LBF wrapper (or
     no-ops if the env doesn't add them).
  2. Call ``self.hebbian.update(...)`` against a *single shared graph*
     spanning all parallel envs. Agent ids are abstract (the policy is
     shared across envs), so cross-env aggregation of bonds is valid and
     speeds up bond formation by `batch_size_run`x.
  3. If ``args.hebbian.reward_diffusion`` is True, replace each env's
     per-agent reward with the diffused reward before it lands in the
     batch.

Selected via ``runner: hebbian_parallel`` in the alg config. Strict
no-op when ``args.hebbian.enabled = False`` — same data path as
``ParallelRunner`` for the same seed.

Requires per-agent rewards (``common_reward: False``). Asserted when
the Hebbian module is enabled.

Also patches the upstream stat-aggregation bug
([[project-lbf-comm-wrapper-bug]]): the parent class does
``sum(d.get(k, 0) for d in infos)`` over keys in ``final_env_infos``,
which crashes when the comm wrapper sets list-valued keys like
``info['comm_events']`` or ``info['positions']``. We strip non-numeric
keys before the aggregation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import List, Optional, Tuple

import numpy as np

from hebbian_module import HebbianSocialGraph, set_graph
from runners.parallel_runner import ParallelRunner
from runners.hebbian_runner import _build_hebbian_config, _flag


class HebbianParallelRunner(ParallelRunner):
    """ParallelRunner with a Hebbian social graph + optional reward diffusion."""

    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.hebbian_config = _build_hebbian_config(args)
        self.hebbian = HebbianSocialGraph(self.hebbian_config)
        self.reward_diffusion = _flag(args, "reward_diffusion", False)

        if self.hebbian_config.enabled:
            assert not getattr(args, "common_reward", True), (
                "HebbianParallelRunner requires `common_reward: False` so it "
                "has per-agent rewards to update and diffuse."
            )

        # Register the (single, shared) graph for the learner to read.
        set_graph(self.hebbian)

        # ── bonds.jsonl + episode-metrics logging ──
        self._bond_snapshot_every = (
            int(getattr(self.hebbian_config, "log_graph_every", 50)) * 100
        )
        self._last_bond_snapshot_t = -1
        self._bonds_file = None  # opened lazily

        # n_agents not on args yet at __init__; pull from env_info.
        n = self.env_info["n_agents"]
        self._n_agents_local = n
        self._signal_counts = np.zeros((n, n), dtype=np.int64)

    # ── helpers (duplicated from HebbianRunner to avoid a refactor of
    # the existing single-env runner) ───────────────────────────────────

    def _per_agent_rewards(self, reward) -> List[float]:
        if isinstance(reward, Iterable):
            return [float(r) for r in reward]
        return [float(reward)] * self._n_agents_local

    def _open_bonds_file(self):
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
                f"HebbianParallelRunner: failed to open bonds.jsonl ({e}); "
                "bond snapshots disabled."
            )
            self._bonds_file = False  # sentinel: don't retry

    def _maybe_snapshot_bonds(self, force: bool = False):
        if not self.hebbian_config.enabled:
            return
        if (not force) and (
            self.t_env - self._last_bond_snapshot_t < self._bond_snapshot_every
        ):
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

    def _hebbian_step(self, data, n_agents: int):
        """Per-env-step: update the shared graph; optionally diffuse rewards.

        Returns the (possibly diffused) reward to be written into the batch.
        """
        if not self.hebbian_config.enabled:
            return data["reward"]

        info = data["info"] if isinstance(data["info"], dict) else {}
        raw_per_agent = self._per_agent_rewards(data["reward"])

        positions = info.get("positions")
        if positions:
            positions = [tuple(p) for p in positions]
        else:
            positions = [None] * n_agents

        comm_events = info.get("comm_events")

        self.hebbian.update(
            positions=positions,
            step_rewards=raw_per_agent,
            advantages=None,
            comm_events=comm_events,
        )

        if comm_events:
            for sender, receiver in comm_events:
                self._signal_counts[sender, receiver] += 1

        if self.reward_diffusion and isinstance(data["reward"], Iterable):
            return self.hebbian.diffuse_rewards(raw_per_agent)
        return data["reward"]

    # ── run override ────────────────────────────────────────────────────
    #
    # Copied from ParallelRunner.run with three added blocks marked
    # ``HEBBIAN-A:``:
    #   1. Per-env-step graph update + reward diffusion (inside the recv
    #      loop, before the reward is stored).
    #   2. Strip non-numeric keys from final_env_infos before the upstream
    #      sum-aggregation (else ``info['comm_events']`` / ``positions``
    #      crash it with ``int + list``).
    #   3. Log hebbian/* metrics and snapshot bonds at the runner log
    #      cadence (analogue of HebbianRunner's logging block).

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []

        n_agents = self.args.n_agents

        while True:
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            cpu_actions = actions.to("cpu").numpy()

            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {"reward": [], "terminated": []}
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    # ── HEBBIAN-A: graph update + optional reward diffusion ──
                    reward_for_batch = self._hebbian_step(data, n_agents)

                    post_transition_data["reward"].append((reward_for_batch,))

                    episode_returns[idx] += reward_for_batch
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )
            self.t += 1
            self.batch.update(
                pre_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=True,
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))
        env_stats = []
        for parent_conn in self.parent_conns:
            env_stats.append(parent_conn.recv())

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # ── HEBBIAN-A: strip non-numeric info keys (comm_events, positions)
        # before the upstream sum-aggregation. Without this the cross-worker
        # `sum(d.get(k, 0) for d in infos)` crashes on `0 + [...]`. ──
        cleaned_final_env_infos = [
            {k: v for k, v in d.items() if isinstance(v, (int, float, np.integer, np.floating))}
            for d in final_env_infos
        ]
        infos = [cur_stats] + cleaned_final_env_infos
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        # ── HEBBIAN-A: log bond + signal metrics, snapshot bonds.jsonl ──
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
            self.logger.log_stat(
                "hebbian/signal_total",
                int(self._signal_counts.sum()),
                self.t_env,
            )
            for i in range(self._n_agents_local):
                self.logger.log_stat(
                    f"hebbian/signal_from_agent_{i}",
                    int(self._signal_counts[i].sum()),
                    self.t_env,
                )

        force = self._last_bond_snapshot_t < 0
        self._maybe_snapshot_bonds(force=force)

        if not test_mode:
            self._signal_counts.fill(0)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch
