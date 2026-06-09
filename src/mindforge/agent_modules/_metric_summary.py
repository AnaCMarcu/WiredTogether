"""Text-summary methods for :class:`CraftiumMetric` (mixin).

Split out of craftium_metric.py for readability; the bodies are unchanged
and run as bound methods of CraftiumMetric.
"""

import os
import statistics

from agent_modules.craftium_metric import TRACKS, TRACK_ORDER


class _SummaryMixin:
    # ─── Text summary ──────────────────────────────────────────────────

    def _save_text_summary(self):
        lines = []
        lines.extend(self._summary_header())
        lines.extend(self._summary_returns())
        lines.extend(self._summary_per_episode_aggregates())
        lines.extend(self._summary_milestones())
        lines.extend(self._summary_steps_to_milestone())
        lines.extend(self._summary_specialization())
        lines.extend(self._summary_communication())
        lines.extend(self._summary_action_health())
        lines.extend(self._summary_rl())
        lines.extend(self._summary_hebbian())
        lines.append("=" * 55)

        summary_path = os.path.join(self.target_folder, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _summary_header(self):
        exp_id = getattr(self, "experiment_id", None)
        comm_str = "on" if self.communication else "off"
        return [
            "=" * 55,
            f"  Five Chambers — {exp_id or 'Experiment Summary'}",
            "=" * 55,
            f"Agents: {self.num_agents}  |  Steps: {self.timestep}  |  Comm: {comm_str}",
            "",
        ]

    def _summary_returns(self):
        lines = ["--- Cumulative Returns (run total) ---"]
        for i in range(self.num_agents):
            lines.append(f"  Agent {i} (agent): {self.cumulative_returns[i]:.2f}")
        lines.append("")
        n_eps = max((len(r) for r in self.per_episode_returns), default=0)
        if n_eps > 0:
            lines.append(f"--- Per-Episode Returns (n={n_eps} episodes) ---")
            for i in range(self.num_agents):
                ep_returns = self.per_episode_returns[i]
                mean = statistics.fmean(ep_returns) if ep_returns else 0.0
                std = statistics.pstdev(ep_returns) if len(ep_returns) >= 2 else 0.0
                ep_str = ", ".join(f"{r:.2f}" for r in ep_returns)
                lines.append(
                    f"  Agent {i}: mean={mean:.2f}  std={std:.2f}  "
                    f"per-ep=[{ep_str}]"
                )
            lines.append("")
        return lines

    def _summary_milestones(self):
        lines = ["--- Milestones per Agent (run total) ---"]
        for i in range(self.num_agents):
            earned = sorted(self._agent_milestones.get(f"agent_{i}", set()))
            lines.append(f"  Agent {i} (agent): {', '.join(earned) if earned else 'none'}")
        lines.append("")
        return lines

    def _summary_per_episode_aggregates(self):
        """Per-episode mean ± std block for the headline aggregable metrics:
        episode length, milestone count, comm count, and per-track reward."""
        n_eps = len(self.episode_lengths)
        if n_eps == 0:
            return []
        lines = [f"--- Per-Episode Aggregates (n={n_eps} episodes) ---"]
        mean_len = statistics.fmean(self.episode_lengths)
        std_len = (
            statistics.pstdev(self.episode_lengths)
            if n_eps >= 2 else 0.0
        )
        lines.append(f"  Episode length:  mean={mean_len:.1f}  std={std_len:.1f}")
        lines.append("  Milestones per episode (count):")
        for i in range(self.num_agents):
            counts = [len(ms) for ms in self.milestones_per_episode[i]]
            mean_c = statistics.fmean(counts) if counts else 0.0
            std_c = statistics.pstdev(counts) if len(counts) >= 2 else 0.0
            lines.append(
                f"    Agent {i}: mean={mean_c:.2f}  std={std_c:.2f}  "
                f"per-ep={counts}"
            )
        lines.append("  Comm count per episode:")
        for i in range(self.num_agents):
            counts = self.comm_count_per_episode[i]
            mean_c = statistics.fmean(counts) if counts else 0.0
            std_c = statistics.pstdev(counts) if len(counts) >= 2 else 0.0
            lines.append(
                f"    Agent {i}: mean={mean_c:.2f}  std={std_c:.2f}  "
                f"per-ep={counts}"
            )
        lines.append("  Track rewards per episode (mean):")
        for i in range(self.num_agents):
            agent_eps = self.track_rewards_per_episode[i]
            if not agent_eps:
                continue
            per_track_means = {
                track: statistics.fmean(
                    [d.get(track, 0.0) for d in agent_eps]
                )
                for track in TRACKS
            }
            parts = "  ".join(
                f"{track}={v:.2f}" for track, v in per_track_means.items()
            )
            lines.append(f"    Agent {i}: {parts}")
        lines.append("")
        return lines

    def _summary_steps_to_milestone(self):
        lines = ["--- Steps to Milestone (first agent) ---"]
        for track, mids in self.steps_to_milestone_table().items():
            for mid, step in mids.items():
                step_str = str(step) if step is not None else "---"
                lines.append(f"  {track:<12} {mid:<28} {step_str:>8}")
        lines.append("")
        return lines

    def _summary_specialization(self):
        lines = ["--- Track Reward Breakdown ---"]
        for i in range(self.num_agents):
            si = self.specialization_index(i)
            parts = [f"{t}={si[t]:.2f}" for t in TRACK_ORDER]
            lines.append(f"  Agent {i} (agent): {', '.join(parts)}")
        lines.append("")
        return lines

    def _summary_communication(self):
        total = len(self.communication_log)
        return [
            "--- Communication ---",
            f"  Total messages: {total}",
            f"  Avg per step:   {total / max(self.timestep, 1):.2f}",
            "",
        ]

    def _summary_action_health(self):
        """Idle-force breakdown: how often the env had to rescue an agent that
        emitted NoOp, split by cause. A high ``invalid`` count means the policy
        is inventing action names (clamped to NoOp then force-moved); a high
        ``explicit`` count means it is deliberately stalling. Both are masked
        coordination failures — surfaced here so they are not silent.
        """
        total_inv = sum(c.get("invalid_action", 0) for c in self.idle_force_counts.values())
        total_exp = sum(c.get("explicit_noop", 0) for c in self.idle_force_counts.values())
        total_rec = sum(self.action_recovered_counts.values())
        if total_inv == total_exp == total_rec == 0:
            return []
        lines = ["--- Action Health ---"]
        lines.append(
            f"  Forced MoveForward (idle): {total_inv + total_exp}  "
            f"(invalid_action={total_inv}, explicit_noop={total_exp})"
        )
        lines.append(
            f"  Recovered (synonym/format -> valid): {total_rec}"
        )
        agents = sorted(set(self.idle_force_counts) | set(self.action_recovered_counts))
        for name in agents:
            c = self.idle_force_counts.get(name, {})
            inv, exp = c.get("invalid_action", 0), c.get("explicit_noop", 0)
            rec = self.action_recovered_counts.get(name, 0)
            lines.append(f"  {name}: invalid={inv}, explicit={exp}, recovered={rec}")
        lines.append("")
        return lines

    def _summary_rl(self):
        if not (self.rl_updates or self.rl_token_opts):
            return []

        action_updates = [u for u in self.rl_updates if "critic_loss" not in u[2]]
        critic_updates = [u for u in self.rl_updates if "critic_loss" in u[2]]

        lines = [
            "--- RL Layer ---",
            f"  Total updates: {len(self.rl_updates)} "
            f"(action={len(action_updates)}, critic={len(critic_updates)})",
        ]
        if action_updates:
            last = action_updates[-1][2]
            lines.append(
                f"    Last action: policy_loss={last.get('policy_loss', 0):.4f}, "
                f"value_loss={last.get('value_loss', 0):.4f}, "
                f"entropy={last.get('entropy', 0):.4f}"
            )
        if critic_updates:
            last = critic_updates[-1][2]
            lines.append(
                f"    Last critic: critic_loss={last.get('critic_loss', 0):.4f}, "
                f"returns_mean={last.get('critic_returns_mean', 0):.4f}, "
                f"returns_std={last.get('critic_returns_std', 0):.4f}"
            )
        train_count = sum(1 for _, _, d, _, _ in self.rl_token_opts if d == "train")
        skip_count  = sum(1 for _, _, d, _, _ in self.rl_token_opts if d != "train")
        lines.append(f"  Token-opt: {train_count} train, {skip_count} skip")
        lines.append("")
        return lines

    def _summary_hebbian(self):
        if not self._graph_snapshots:
            return []
        last = self._graph_snapshots[-1]
        lines = [
            "--- Hebbian Social Plasticity ---",
            f"  Final mean bond:  {last.get('mean_bond_strength', 0):.4f}",
            f"  Final sparsity:   {last.get('sparsity', 0):.2f}",
        ]
        top = last.get("top_3_pairs", [])
        if top:
            lines.append("  Top bonds:")
            for p in top[:3]:
                lines.append(f"    Agent {p['i']} -> Agent {p['j']}: {p['w']:.4f}")
        lines.append("")
        return lines

