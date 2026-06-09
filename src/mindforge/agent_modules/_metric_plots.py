"""Plotting methods for :class:`CraftiumMetric` (mixin).

Split out of craftium_metric.py for readability; the bodies are unchanged
and run as bound methods of CraftiumMetric, so they reference ``self`` state
set up by the main class.
"""

import os
import statistics

import matplotlib.pyplot as plt
import numpy as np

from agent_modules.craftium_metric import (
    TRACKS,
    TRACK_ORDER,
    _agent_id_from_name,
)


class _PlotsMixin:
    # ─── Plots ─────────────────────────────────────────────────────────

    def _plots_dir(self) -> str:
        """Return (and lazily create) the per-run plots subdirectory.

        Centralises the choice so JSON / txt / log artifacts stay in the
        run root while every .png lives under ``<run_root>/plots/``. The
        directory is created on first call; subsequent calls just return
        the path.
        """
        path = os.path.join(self.target_folder, "plots")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_plots(self):
        if not self.ts_data["timesteps"]:
            return
        self._plot_cumulative_returns()
        self._plot_returns_curve()
        self._plot_chamber_returns()
        self._plot_milestones()
        self._plot_track_rewards()
        self._plot_reward_decomposition()
        self._write_steps_to_milestone_txt()
        self._plot_communication_frequency()
        self._plot_comm_curve()
        self._plot_hebbian_bonds()
        self._plot_rl_losses()

    def _plot_chamber_returns(self):
        """Per-chamber mean ± std return across episodes.

        Left panel:  grouped bars per chamber, one bar per agent (showing
                     that agent's mean across episodes for that chamber's
                     reward track), with error bars = std across episodes.
        Right panel: single bar per chamber for the TEAM-TOTAL (sum across
                     agents, mean ± std across episodes). Headline view.

        Uses self.track_rewards_per_episode (which is per-agent, per-
        episode, per-track), which is reset cleanly at end_episode().
        """
        if not self.track_rewards_per_episode or not any(self.track_rewards_per_episode):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                "No completed episodes yet — per-chamber plot unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666666",
            )
            ax.set_axis_off()
            fig.savefig(
                os.path.join(self._plots_dir(), "chamber_returns.png"),
                dpi=150,
            )
            plt.close(fig)
            return

        tracks = list(TRACK_ORDER)
        track_labels = {
            "ch1_solo":      "Ch1",
            "ch2_anvils":    "Ch2",
            "ch3_switches":  "Ch3",
            "ch4_combat":    "Ch4",
            "ch5_boss":      "Ch5",
            "communication": "Comm",
        }
        labels = [track_labels.get(t, t) for t in tracks]
        n_eps = max(
            (len(self.track_rewards_per_episode[i])
             for i in range(self.num_agents)),
            default=0,
        )

        # Per-(agent, chamber) mean/std across episodes.
        per_agent_mean = [[] for _ in range(self.num_agents)]
        per_agent_std  = [[] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            agent_eps = self.track_rewards_per_episode[i]
            for t in tracks:
                vals = [d.get(t, 0.0) for d in agent_eps]
                per_agent_mean[i].append(
                    statistics.fmean(vals) if vals else 0.0
                )
                per_agent_std[i].append(
                    statistics.pstdev(vals) if len(vals) >= 2 else 0.0
                )

        # Team-total per chamber per episode → mean / std across episodes.
        team_means = []
        team_stds  = []
        for t in tracks:
            ep_sums = []
            for ep_idx in range(n_eps):
                total = 0.0
                for i in range(self.num_agents):
                    if ep_idx < len(self.track_rewards_per_episode[i]):
                        total += self.track_rewards_per_episode[i][ep_idx].get(t, 0.0)
                ep_sums.append(total)
            team_means.append(statistics.fmean(ep_sums) if ep_sums else 0.0)
            team_stds.append(
                statistics.pstdev(ep_sums) if len(ep_sums) >= 2 else 0.0
            )

        cmap = plt.get_cmap("tab10")
        agent_colors = [cmap(i % 10) for i in range(self.num_agents)]
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [3, 2]})

        # ── Left: grouped bars per chamber per agent ──
        bar_w = 0.8 / max(1, self.num_agents)
        x_chambers = np.arange(len(tracks))
        for i in range(self.num_agents):
            xs = x_chambers + (i - (self.num_agents - 1) / 2) * bar_w
            ax_l.bar(
                xs,
                per_agent_mean[i],
                bar_w,
                yerr=per_agent_std[i],
                capsize=3,
                color=agent_colors[i],
                edgecolor="black",
                linewidth=0.4,
                label=f"agent_{i}",
                error_kw=dict(ecolor="#222222", lw=1.0),
            )
        ax_l.set_xticks(x_chambers)
        ax_l.set_xticklabels(labels)
        ax_l.set_xlabel("Chamber")
        ax_l.set_ylabel("Reward (mean ± std across episodes)")
        ax_l.set_title(
            f"Per-agent return per chamber (n={n_eps} episode{'s' if n_eps != 1 else ''})"
        )
        ax_l.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_l.set_axisbelow(True)
        ax_l.axhline(0, color="#888888", linewidth=0.5)
        ax_l.legend(fontsize=8, loc="best", framealpha=0.9)

        # ── Right: team total per chamber ──
        ax_r.bar(
            x_chambers,
            team_means,
            yerr=team_stds,
            capsize=5,
            color="#4c72b0",
            edgecolor="black",
            linewidth=0.4,
            error_kw=dict(ecolor="#222222", lw=1.2),
        )
        # Annotate each bar with the team-total mean ± std.
        max_h = max([m + s for m, s in zip(team_means, team_stds)] + [1.0])
        for x, m, s in zip(x_chambers, team_means, team_stds):
            ax_r.text(
                x, m + max(s, 0) + max_h * 0.02,
                f"{m:.0f}\n±{s:.0f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax_r.set_xticks(x_chambers)
        ax_r.set_xticklabels(labels)
        ax_r.set_xlabel("Chamber")
        ax_r.set_ylabel("Team total reward (mean ± std across episodes)")
        ax_r.set_title("Team total per chamber")
        ax_r.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_r.set_axisbelow(True)
        ax_r.axhline(0, color="#888888", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "chamber_returns.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_returns_curve(self):
        """RL-paper-style learning curve: mean ± std across EPISODES.

        X-axis: within-episode step (0 .. max_ep_len).
        Y-axis: within-episode cumulative team return (summed across the
                N agents at each step, re-zeroed at the start of each ep).
        Solid line: mean across the N episodes at each within-episode step.
        Shaded band: ± 1 std across episodes.
        Thin lines: per-episode trajectories (for context).

        This is the canonical learning-curve framing for single-seed
        multi-episode RL runs — "what does a typical episode look like,
        and how much do episodes differ from each other?". For multi-seed
        plots, aggregate across seeds in a separate post-hoc script.

        Falls back to a "no episodes yet" placeholder if end_episode()
        hasn't been called (e.g. mid-ep1 crash dump).
        """
        ts = self.ts_data["timesteps"]
        series = self.ts_data["cumulative_returns"]  # per-agent cumulative across whole run
        ep_lens = getattr(self, "episode_lengths", []) or []
        if not ts or not series or not ep_lens:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                "No completed episodes yet — learning curve unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666666",
            )
            ax.set_axis_off()
            fig.savefig(
                os.path.join(self._plots_dir(), "returns_curve.png"),
                dpi=150,
            )
            plt.close(fig)
            return

        T = min(len(s) for s in series)
        arr = np.array([s[:T] for s in series], dtype=float)  # (n_agents, T)
        # Sum across agents to get a team cumulative-return-over-the-run series.
        team_cum = arr.sum(axis=0)  # (T,)
        # Convert to per-step deltas so we can re-cumulate within each episode.
        deltas = np.diff(team_cum, prepend=0.0)  # (T,)

        # Slice into episode-aligned trajectories, re-cumulating each.
        ep_curves = []  # list of (ep_len,) arrays
        idx = 0
        for ep_len in ep_lens:
            seg = deltas[idx: idx + int(ep_len)]
            if seg.size == 0:
                break
            ep_curves.append(np.cumsum(seg))
            idx += int(ep_len)
        if not ep_curves:
            return

        max_len = max(c.size for c in ep_curves)
        padded = np.full((len(ep_curves), max_len), np.nan)
        for i, c in enumerate(ep_curves):
            padded[i, : c.size] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0, ddof=0)
        x = np.arange(1, max_len + 1)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        # Thin per-episode curves for context.
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(ep_curves):
            ax.plot(np.arange(1, c.size + 1), c,
                    color=cmap(i % 10), alpha=0.30,
                    linewidth=0.9, label=f"ep{i+1}")
        # Mean across episodes + std band.
        ax.fill_between(
            x, mean - std, mean + std,
            color="#1f77b4", alpha=0.20,
            label=f"± 1 std (across {len(ep_curves)} episodes)",
        )
        ax.plot(x, mean, color="#1f77b4", linewidth=2.2,
                label=f"mean (across {len(ep_curves)} episodes)")

        ax.set_xlabel("Within-episode step")
        ax.set_ylabel("Cumulative team return")
        ax.set_title(
            f"Learning curve — mean ± std across {len(ep_curves)} episodes"
        )
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#888888", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "returns_curve.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_comm_curve(self):
        """RL-paper-style communication-rate curve: mean ± std across EPISODES.

        X-axis: within-episode step.
        Y-axis: smoothed messages-per-step rate.
        Solid line: mean across episodes at each within-episode step.
        Shaded band: ± 1 std across episodes.
        Thin lines: per-episode smoothed trajectories.

        Same framing as _plot_returns_curve — answers "what does a typical
        episode's comm rate look like, and how much do episodes differ?".
        """
        ccps = self.comm_counts_per_step
        ep_lens = getattr(self, "episode_lengths", []) or []
        if not ccps or not ep_lens:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                "No completed episodes yet — comm curve unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666666",
            )
            ax.set_axis_off()
            fig.savefig(
                os.path.join(self._plots_dir(), "comm_curve.png"),
                dpi=150,
            )
            plt.close(fig)
            return

        arr = np.array(ccps, dtype=float)

        ep_series = []
        idx = 0
        for ep_len in ep_lens:
            seg = arr[idx: idx + int(ep_len)]
            if seg.size == 0:
                break
            ep_series.append(seg)
            idx += int(ep_len)
        if not ep_series:
            return

        window = max(1, min(50, min(s.size for s in ep_series)))
        smoothed = []
        for s in ep_series:
            if window > 1:
                k = np.ones(window) / window
                smoothed.append(np.convolve(s, k, mode="valid"))
            else:
                smoothed.append(s.copy())

        max_len = max(c.size for c in smoothed)
        padded = np.full((len(smoothed), max_len), np.nan)
        for i, c in enumerate(smoothed):
            padded[i, : c.size] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0, ddof=0)
        x = np.arange(1, max_len + 1) + (window // 2)  # window-center offset

        fig, ax = plt.subplots(figsize=(10, 4.5))
        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(smoothed):
            ax.plot(
                np.arange(1, c.size + 1) + (window // 2), c,
                color=cmap(i % 10), alpha=0.30, linewidth=0.9,
                label=f"ep{i+1}",
            )
        ax.fill_between(
            x, mean - std, mean + std,
            color="#2ca02c", alpha=0.20,
            label=f"± 1 std (across {len(smoothed)} episodes)",
        )
        ax.plot(
            x, mean, color="#2ca02c", linewidth=2.2,
            label=f"mean (across {len(smoothed)} episodes)",
        )

        ax.set_xlabel("Within-episode step")
        ax.set_ylabel(f"Messages per step (smoothed, window={window})")
        ax.set_title(
            f"Communication rate — mean ± std across {len(smoothed)} episodes"
        )
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "comm_curve.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_cumulative_returns(self):
        """Two-panel per-episode return plot.

        Left:  grouped bars showing each agent's return for each episode
               (one cluster per episode, one bar per agent). Lets you see
               which episode each agent did well/poorly in at a glance.
        Right: per-agent mean ± std across episodes (error bar). Single-
               number summary for each agent, std captures cross-episode
               variability within this seed.

        Falls back to a single-panel "no per-episode data yet" plot if
        end_episode() hasn't been called yet (e.g., a crashed run before
        ep1 finished).
        """
        n_eps = len(self.per_episode_returns[0]) if self.per_episode_returns else 0
        if n_eps == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                "No completed episodes yet — per-episode plot unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666666",
            )
            ax.set_axis_off()
            fig.savefig(
                os.path.join(self._plots_dir(), "cumulative_returns.png"),
                dpi=150,
            )
            plt.close(fig)
            return

        cmap = plt.get_cmap("tab10")
        agent_colors = [cmap(i % 10) for i in range(self.num_agents)]
        ep_indices = list(range(1, n_eps + 1))

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5),
                                          gridspec_kw={"width_ratios": [2.5, 1]})

        # ── Left: grouped bars per episode ──
        bar_w = 0.8 / max(1, self.num_agents)
        for i in range(self.num_agents):
            xs = [ep + (i - (self.num_agents - 1) / 2) * bar_w
                  for ep in ep_indices]
            ys = self.per_episode_returns[i]
            ax_l.bar(xs, ys, bar_w, color=agent_colors[i],
                     edgecolor="black", linewidth=0.4,
                     label=f"agent_{i}")
        ax_l.set_xticks(ep_indices)
        ax_l.set_xticklabels([f"ep{e}" for e in ep_indices])
        ax_l.set_xlabel("Episode")
        ax_l.set_ylabel("Return")
        ax_l.set_title(f"Return per Episode (n={n_eps})")
        ax_l.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_l.set_axisbelow(True)
        ax_l.legend(fontsize=8, loc="best", framealpha=0.9)
        ax_l.axhline(0, color="#888888", linewidth=0.5)

        # ── Right: per-agent mean ± std across episodes ──
        means = [statistics.fmean(self.per_episode_returns[i]) if self.per_episode_returns[i] else 0.0
                 for i in range(self.num_agents)]
        stds = [statistics.pstdev(self.per_episode_returns[i]) if len(self.per_episode_returns[i]) >= 2 else 0.0
                for i in range(self.num_agents)]
        x_pos = list(range(self.num_agents))
        ax_r.bar(x_pos, means, yerr=stds, capsize=5,
                 color=[agent_colors[i] for i in range(self.num_agents)],
                 edgecolor="black", linewidth=0.4,
                 error_kw=dict(ecolor="#222222", lw=1.2))
        # Annotate each bar with the mean ± std value.
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_r.text(i, m + max(s, 0) + 2,
                      f"{m:.1f}\n±{s:.1f}",
                      ha="center", va="bottom", fontsize=8)
        ax_r.set_xticks(x_pos)
        ax_r.set_xticklabels([f"a{i}" for i in range(self.num_agents)])
        ax_r.set_xlabel("Agent")
        ax_r.set_ylabel("Mean Return ± std")
        ax_r.set_title("Per-agent mean ± std")
        ax_r.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_r.set_axisbelow(True)
        ax_r.axhline(0, color="#888888", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "cumulative_returns.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_milestones(self):
        """Gantt-style milestone timeline.

        X-axis: env step.
        Y-axis: every milestone in canonical order, grouped into chamber
                bands (Ch1 at bottom → Ch5 → comm at top).
        Markers: one per (agent, milestone, step) — colored by agent.
        Episode boundaries: vertical dashed lines.

        Reads from self.milestone_events (rich, includes contributor + step)
        rather than the cumulative-count series, so you can see WHICH
        milestone fired WHEN for WHICH agent — the actual story of the
        run, not just a count.
        """
        y_labels = []
        y_chamber = []  # parallel: chamber name per row, for banding
        for track in TRACK_ORDER:
            for mid, _ in TRACKS[track]:
                y_labels.append(mid)
                y_chamber.append(track)
        y_index = {mid: i for i, mid in enumerate(y_labels)}

        chamber_band_colors = {
            "ch1_solo":     "#f4f0ff",
            "ch2_anvils":   "#fff4e6",
            "ch3_switches": "#e8f8ff",
            "ch4_combat":   "#ffeeee",
            "ch5_boss":     "#fff0c2",
            "communication":"#eaeaea",
        }

        cmap = plt.get_cmap("tab10")
        agent_colors = {i: cmap(i % 10) for i in range(self.num_agents)}

        fig_h = max(6.5, 0.32 * len(y_labels))
        fig, ax = plt.subplots(figsize=(13, fig_h))

        x_max = max(
            (ev.get("step", 0) for ev in self.milestone_events),
            default=self.timestep,
        )
        x_max = max(x_max, 1)
        chamber_first_last = {}
        for row_i, ch in enumerate(y_chamber):
            if ch not in chamber_first_last:
                chamber_first_last[ch] = [row_i, row_i]
            chamber_first_last[ch][1] = row_i
        for ch, (lo, hi) in chamber_first_last.items():
            ax.axhspan(
                lo - 0.5, hi + 0.5,
                color=chamber_band_colors.get(ch, "#ffffff"),
                zorder=0,
            )
            # Chamber label at the right edge of the band.
            ax.text(
                x_max * 1.005, (lo + hi) / 2, ch,
                fontsize=8, va="center", ha="left",
                color="#555555", style="italic",
            )

        # Episode boundary verticals (only if we tracked them).
        if getattr(self, "episode_lengths", None):
            cum = 0
            for ep_i, ep_len in enumerate(self.episode_lengths):
                cum += int(ep_len)
                ax.axvline(
                    cum, color="#888888", linestyle="--",
                    linewidth=0.8, zorder=1,
                )
                ax.text(
                    cum, len(y_labels) - 0.2, f" ep{ep_i+1} end",
                    fontsize=7, color="#666666", va="top",
                )

        agents_seen = set()
        for ev in self.milestone_events:
            mid = ev.get("milestone_id") or ev.get("milestone")
            if mid not in y_index:
                continue
            row = y_index[mid]
            step = ev.get("step")
            contrib = ev.get("contributor", "")
            agent_id = _agent_id_from_name(contrib)
            if not (0 <= agent_id < self.num_agents):
                continue
            y_jitter = (agent_id - (self.num_agents - 1) / 2) * 0.12
            ax.scatter(
                step, row + y_jitter,
                s=42,
                color=agent_colors[agent_id],
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
                label=f"agent_{agent_id}" if agent_id not in agents_seen else None,
            )
            agents_seen.add(agent_id)

        m8_row = y_index.get("m8_anvil_A1")
        m9_row = y_index.get("m9_anvil_B1")
        coop_x, coop_y = [], []
        for ev in getattr(self, "anvil_coop_events", []):
            row = m8_row if ev.get("row") == "A" else m9_row
            if row is None:
                continue
            coop_x.append(ev.get("step", 0))
            coop_y.append(row)
        if coop_x:
            ax.scatter(
                coop_x, coop_y,
                marker="x", s=28, linewidths=1.0,
                color="#666666", alpha=0.55, zorder=2,
                label="coop attempts (≥2 agents punching, diagnostic)",
            )

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Env step")
        ax.set_xlim(0, x_max * 1.08)
        ax.set_ylim(-0.7, len(y_labels) - 0.3)
        ax.set_title(
            "Five-Chambers Milestone Timeline (markers = fire events; "
            "gray X = anvil coop attempts; rows grouped by chamber)"
        )
        ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=1)
        ax.set_axisbelow(True)
        if agents_seen:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "milestones.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_track_rewards(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(self.num_agents)
        bottom = np.zeros(self.num_agents)
        colors = plt.cm.tab10(np.linspace(0, 1, len(TRACKS)))
        for idx, track in enumerate(TRACK_ORDER):
            vals = np.array([
                self.track_rewards[i].get(track, 0.0) for i in range(self.num_agents)
            ])
            ax.bar(x, vals, bottom=bottom, label=track, color=colors[idx])
            bottom += vals
        ax.set_xlabel("Agent")
        ax.set_ylabel("Milestone-fire reward only (no dig_stage_res / shaping)")
        ax.set_title("Reward by Chamber Track per Agent (milestones only)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Agent {i}" for i in range(self.num_agents)])
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(os.path.join(self._plots_dir(), "track_rewards.png"), dpi=150)
        plt.close(fig)

    def _plot_reward_decomposition(self):
        """Stacked bar of per-agent cumulative return BY REWARD SOURCE.

        Reads `reward_history_decomposed` (populated each step by
        record_reward_decomposed in the main loop). Captures everything
        the env actually delivered, including sub-threshold dig progress
        that doesn't trigger a milestone — which is what the
        track_rewards chart misses.
        """
        if not any(self.reward_history_decomposed):
            return
        sources = ["task", "comm_base", "comm_milestone", "proximity", "hebbian_diffuse"]
        # Sum each source per agent.
        sums = np.zeros((self.num_agents, len(sources)))
        for i in range(self.num_agents):
            for rec in self.reward_history_decomposed[i]:
                for k, src in enumerate(sources):
                    sums[i, k] += float(rec.get(src, 0.0))

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(self.num_agents)
        bottom = np.zeros(self.num_agents)
        colors = ["#2ca02c", "#1f77b4", "#17becf", "#ff7f0e", "#9467bd"]
        for k, src in enumerate(sources):
            ax.bar(x, sums[:, k], bottom=bottom, label=src, color=colors[k])
            bottom += sums[:, k]
        ax.set_xlabel("Agent")
        ax.set_ylabel("Cumulative reward (per source)")
        ax.set_title("Reward decomposition by source (full env-delivered total)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Agent {i}" for i in range(self.num_agents)])
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(os.path.join(self._plots_dir(), "reward_decomposition.png"), dpi=150)
        plt.close(fig)

    def _write_steps_to_milestone_txt(self):
        table = self.steps_to_milestone_table()
        lines = [
            "Steps to Milestone (first agent to reach)\n",
            f"{'Track':<12} {'Milestone':<28} {'Step':>8}",
            "-" * 50,
        ]
        for track, mids in table.items():
            for mid, step in mids.items():
                step_str = str(step) if step is not None else "---"
                lines.append(f"{track:<12} {mid:<28} {step_str:>8}")
        with open(os.path.join(self.target_folder, "steps_to_milestone.txt"), "w") as f:
            f.write("\n".join(lines))

    def _plot_rl_losses(self):
        """Plot RL losses over training: per-agent policy loss + entropy
        and (when present) the centralized critic loss + return statistics.
        """
        if not self.rl_updates:
            return

        action_updates = [u for u in self.rl_updates if "critic_loss" not in u[2]]
        critic_updates = [u for u in self.rl_updates if "critic_loss" in u[2]]
        if not action_updates and not critic_updates:
            return

        n_subplots = 2 if critic_updates else 1
        fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots),
                                 squeeze=False)

        # ── Subplot 1: per-agent policy loss + entropy ──
        ax = axes[0, 0]
        if action_updates:
            per_agent = {}
            for ts, aid, info in action_updates:
                per_agent.setdefault(aid, {"ts": [], "policy": [], "entropy": []})
                per_agent[aid]["ts"].append(ts)
                per_agent[aid]["policy"].append(info.get("policy_loss", 0.0))
                per_agent[aid]["entropy"].append(info.get("entropy", 0.0))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(per_agent), 1)))
            for (aid, d), c in zip(sorted(per_agent.items()), colors):
                ax.plot(d["ts"], d["policy"], color=c,
                        label=f"agent {aid} policy_loss")
                ax.plot(d["ts"], d["entropy"], color=c, linestyle="--", alpha=0.5,
                        label=f"agent {aid} entropy")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Loss")
            ax.set_title("Per-agent action update — policy loss & entropy")
            ax.legend(fontsize=7, loc="best")
        else:
            ax.set_visible(False)

        # ── Subplot 2: centralized critic loss + returns mean ± std ──
        if critic_updates:
            ax = axes[1, 0]
            ts_c   = [u[0] for u in critic_updates]
            loss_c = [u[2].get("critic_loss", 0.0)         for u in critic_updates]
            ret_m  = [u[2].get("critic_returns_mean", 0.0) for u in critic_updates]
            ret_s  = [u[2].get("critic_returns_std", 0.0)  for u in critic_updates]

            ax.plot(ts_c, loss_c, color="#d62728", label="critic_loss")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Critic loss", color="#d62728")
            ax.tick_params(axis="y", labelcolor="#d62728")

            ax2 = ax.twinx()
            ax2.plot(ts_c, ret_m, color="#1f77b4", label="returns mean")
            ret_m_arr = np.asarray(ret_m, dtype=float)
            ret_s_arr = np.asarray(ret_s, dtype=float)
            ax2.fill_between(
                ts_c, ret_m_arr - ret_s_arr, ret_m_arr + ret_s_arr,
                color="#1f77b4", alpha=0.2, label="returns ±1σ"
            )
            ax2.set_ylabel("Team return", color="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#1f77b4")
            ax.set_title("Centralized critic — loss & GAE return target")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

        fig.tight_layout()
        fig.savefig(os.path.join(self._plots_dir(), "rl_losses.png"), dpi=150)
        plt.close(fig)

    def _plot_communication_frequency(self):
        """Two-panel per-episode communication plot.

        Left:  grouped bars of messages-sent per episode per agent.
        Right: per-agent mean ± std across episodes.

        Falls back to a "no data" placeholder if no episode finished yet.
        """
        n_eps = len(self.comm_count_per_episode[0]) if self.comm_count_per_episode else 0
        if n_eps == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(
                0.5, 0.5,
                "No completed episodes yet — per-episode plot unavailable.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#666666",
            )
            ax.set_axis_off()
            fig.savefig(
                os.path.join(self._plots_dir(), "communication_frequency.png"),
                dpi=150,
            )
            plt.close(fig)
            return

        cmap = plt.get_cmap("tab10")
        agent_colors = [cmap(i % 10) for i in range(self.num_agents)]
        ep_indices = list(range(1, n_eps + 1))

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.5),
                                          gridspec_kw={"width_ratios": [2.5, 1]})

        # ── Left: grouped bars per episode ──
        bar_w = 0.8 / max(1, self.num_agents)
        for i in range(self.num_agents):
            xs = [ep + (i - (self.num_agents - 1) / 2) * bar_w
                  for ep in ep_indices]
            ys = self.comm_count_per_episode[i]
            ax_l.bar(xs, ys, bar_w, color=agent_colors[i],
                     edgecolor="black", linewidth=0.4,
                     label=f"agent_{i}")
        ax_l.set_xticks(ep_indices)
        ax_l.set_xticklabels([f"ep{e}" for e in ep_indices])
        ax_l.set_xlabel("Episode")
        ax_l.set_ylabel("Messages sent")
        ax_l.set_title(f"Communication Volume per Episode (n={n_eps})")
        ax_l.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_l.set_axisbelow(True)
        ax_l.legend(fontsize=8, loc="best", framealpha=0.9)

        # ── Right: per-agent mean ± std across episodes ──
        means = [statistics.fmean(self.comm_count_per_episode[i]) if self.comm_count_per_episode[i] else 0.0
                 for i in range(self.num_agents)]
        stds = [statistics.pstdev(self.comm_count_per_episode[i]) if len(self.comm_count_per_episode[i]) >= 2 else 0.0
                for i in range(self.num_agents)]
        x_pos = list(range(self.num_agents))
        ax_r.bar(x_pos, means, yerr=stds, capsize=5,
                 color=[agent_colors[i] for i in range(self.num_agents)],
                 edgecolor="black", linewidth=0.4,
                 error_kw=dict(ecolor="#222222", lw=1.2))
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_r.text(i, m + max(s, 0) + max(means + [1]) * 0.02,
                      f"{m:.0f}\n±{s:.0f}",
                      ha="center", va="bottom", fontsize=8)
        ax_r.set_xticks(x_pos)
        ax_r.set_xticklabels([f"a{i}" for i in range(self.num_agents)])
        ax_r.set_xlabel("Agent")
        ax_r.set_ylabel("Mean messages ± std")
        ax_r.set_title("Per-agent mean ± std")
        ax_r.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax_r.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(
            os.path.join(self._plots_dir(), "communication_frequency.png"),
            dpi=150,
        )
        plt.close(fig)

    def _plot_hebbian_bonds(self):
        """Render bond evolution plots. Calls both the legacy mean-bond
        line plot and the per-pair asymmetry plot. The asymmetry plot is
        the one designed for paper figures; the line plot stays for
        backward-compat with tooling that expects graph_bond_evolution.png.
        """
        if not self._graph_snapshots:
            return
        # Legacy view (mean + top-3): kept for continuity.
        self._plot_hebbian_mean()
        # Paper-quality view: per-pair asymmetric bonds.
        self._plot_hebbian_asymmetry()

    def _plot_hebbian_mean(self):
        """Mean-bond + top-3 lines. Note: top-3 pair lines can ARTIFICIALLY
        drop to 0 when a pair falls out of the current top-3 set — only
        meaningful for snapshots where the FULL W matrix isn't stored.
        For runs with W stored per snapshot, prefer _plot_hebbian_asymmetry.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        snap_steps = [s["step"] for s in self._graph_snapshots]
        mean_bond  = [s.get("mean_bond_strength", 0) for s in self._graph_snapshots]
        ax.plot(snap_steps, mean_bond, label="Mean bond strength", linewidth=2)

        last_top = self._graph_snapshots[-1].get("top_3_pairs", [])
        for pair in last_top:
            i_idx, j_idx = pair["i"], pair["j"]
            vals = [self._bond_weight_at(s, i_idx, j_idx) for s in self._graph_snapshots]
            ax.plot(snap_steps, vals, label=f"Agent {i_idx} -> {j_idx}", alpha=0.6)

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Bond Strength")
        ax.set_title("Hebbian Social Graph — Bond Evolution (mean view)")
        ax.legend()
        fig.savefig(os.path.join(self._plots_dir(), "graph_bond_evolution.png"), dpi=150)
        plt.close(fig)

    def _plot_hebbian_asymmetry(self):
        """Per-pair asymmetric bond figure for the paper.

        For each undirected pair (i, j) with i < j, draws W[i,j] and
        W[j,i] as two lines on the same subplot, with the area between
        them shaded to highlight the directed-trust asymmetry. Requires
        the full W matrix per snapshot (added to get_graph_metrics in
        hebbian/graph.py); silently falls back to "no plot" otherwise.
        """
        snaps = self._graph_snapshots
        if not snaps:
            return
        first_W = snaps[0].get("W")
        if not first_W:
            # Run pre-dates the W-in-snapshot fix; nothing to plot.
            return

        N = len(first_W)
        steps = [s["step"] for s in snaps]
        # Undirected pairs (i, j) with i < j — one subplot each.
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        if not pairs:
            return

        fig, axes = plt.subplots(
            len(pairs), 1, figsize=(8, 2.4 * len(pairs)),
            sharex=True, constrained_layout=True,
        )
        if len(pairs) == 1:
            axes = [axes]

        # Match colour pair to direction: forward (i→j) blue, reverse (j→i) orange.
        c_forward = "#2c7fb8"
        c_reverse = "#d95f0e"
        c_asym    = "#888888"

        for ax, (i, j) in zip(axes, pairs):
            w_ij = [s["W"][i][j] for s in snaps]
            w_ji = [s["W"][j][i] for s in snaps]

            ax.plot(steps, w_ij, color=c_forward, linewidth=2,
                    label=f"agent_{i} → agent_{j}")
            ax.plot(steps, w_ji, color=c_reverse, linewidth=2,
                    label=f"agent_{j} → agent_{i}")

            # Shaded asymmetry band: between min and max of the two directions.
            lo = [min(a, b) for a, b in zip(w_ij, w_ji)]
            hi = [max(a, b) for a, b in zip(w_ij, w_ji)]
            ax.fill_between(steps, lo, hi, color=c_asym, alpha=0.18,
                            label="|W_ij − W_ji|")

            ax.set_ylabel("Bond W")
            ax.set_ylim(0, 1)
            ax.set_title(f"Pair (agent_{i}, agent_{j})", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

        axes[-1].set_xlabel("Timestep")
        fig.suptitle("Asymmetric Hebbian Bonds Over One Episode",
                     fontsize=13, y=1.02)
        fig.savefig(
            os.path.join(self._plots_dir(), "graph_bond_asymmetry.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    @staticmethod
    def _bond_weight_at(snapshot, i_idx, j_idx) -> float:
        W = snapshot.get("W")
        if W and 0 <= i_idx < len(W) and 0 <= j_idx < len(W[i_idx]):
            return float(W[i_idx][j_idx])
        for tp in snapshot.get("top_3_pairs", []):
            if tp["i"] == i_idx and tp["j"] == j_idx:
                return tp["w"]
        return 0.0

