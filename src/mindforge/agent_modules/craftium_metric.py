"""Evaluation metrics for Craftium multi-agent experiments.

Tracks:
- Cumulative return per agent
- Steps-to-milestone for each progression track (Tools, Hunt, Defend)
- Per-track reward breakdown and specialization index
- Communication events
- Generates plots and saves JSON data
"""

import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# Stage completion reward thresholds (from Lua mods).
# Stages emit rewards 128, 256, 1024, 2048 — we detect them via reward spikes.
STAGE_REWARDS = {128.0, 256.0, 1024.0, 2048.0}

# Ordered milestones per track (tier_index -> (name, stage_reward_value))
TRACKS = {
    "tools": [
        ("wood", 128.0),
        ("stone", 256.0),
        ("iron", 1024.0),
        ("diamond", 2048.0),
    ],
    "hunt": [
        ("chicken", 128.0),
        ("sheep", 256.0),
        ("pig", 1024.0),
        ("cow", 2048.0),
    ],
    "defend": [
        ("zombie", 128.0),
        ("skeleton", 256.0),
        ("spider", 1024.0),
        ("cave_spider", 2048.0),
    ],
}

# Reward multipliers per role for stage events (from Lua client mod).
# Role 0=gatherer (tools 1.0x, hunt 0.3x, defend 0.3x)
# Role 1=hunter   (tools 0.3x, hunt 1.0x, defend 0.3x)
# Role 2=defender (tools 0.3x, hunt 0.3x, defend 1.0x)
ROLE_STAGE_MULTIPLIERS = [
    {"tools": 1.0, "hunt": 0.3, "defend": 0.3},  # gatherer
    {"tools": 0.3, "hunt": 1.0, "defend": 0.3},  # hunter
    {"tools": 0.3, "hunt": 0.3, "defend": 1.0},  # defender
]


class CraftiumMetric:
    """Tracks evaluation metrics for Craftium multi-agent experiments."""

    def __init__(
        self,
        num_agents=3,
        communication=True,
        path="./run_metrics",
    ):
        self.num_agents = num_agents
        self.communication = communication
        self.timestep = 0

        # Per-agent cumulative return
        self.cumulative_returns = [0.0] * num_agents

        # Per-agent, per-step reward history
        self.reward_history = [[] for _ in range(num_agents)]

        # Milestone tracking: agent_id -> track -> list of (timestep, tier_name)
        self.milestones = {
            i: {"tools": [], "hunt": [], "defend": []}
            for i in range(num_agents)
        }

        # Steps-to-milestone: track -> tier_index -> first timestep achieved (any agent)
        self.first_milestone_step = {
            track: {} for track in TRACKS
        }

        # Per-track reward accumulation (for specialization index)
        # Estimated from reward spikes matching known stage values
        self.track_rewards = {
            i: {"tools": 0.0, "hunt": 0.0, "defend": 0.0, "other": 0.0}
            for i in range(num_agents)
        }

        # Communication log: list of (timestep, source_agent, message_preview)
        self.communication_log = []
        self.comm_counts_per_step = []

        # RL layer tracking
        self.rl_updates = []       # (timestep, agent_id, info_dict)
        self.rl_token_opts = []    # (timestep, agent_id, decision, reason, info_dict)

        # Timestep-level data for plotting
        self.ts_data = {
            "timesteps": [],
            "cumulative_returns": [[] for _ in range(num_agents)],
            "milestone_count": [[] for _ in range(num_agents)],
            "total_milestones": [],  # joint across all agents
        }

        self.target_folder = self._mkdir_metrics(path)

    # ------------------------------------------------------------------
    # Recording methods (called from the main loop)
    # ------------------------------------------------------------------

    def record_reward(self, agent_id: int, reward: float):
        """Record a single-step reward for an agent.

        Detects milestone events from reward spikes and classifies them
        by track using the agent's role multiplier.
        """
        self.cumulative_returns[agent_id] += reward
        self.reward_history[agent_id].append((self.timestep, reward))

        # Detect stage completion from reward value
        self._detect_milestone(agent_id, reward)

    def record_communication(self, source_agent: str, message: str):
        """Record a communication event."""
        preview = message[:100] if message else ""
        self.communication_log.append((self.timestep, source_agent, preview))

    def record_rl_update(self, agent_id: int, info: dict):
        """Record an action-level MAPPO update event."""
        self.rl_updates.append((self.timestep, agent_id, info))
        self.log(f"[RL] Agent {agent_id} MAPPO update at step {self.timestep}: "
                 f"policy_loss={info.get('policy_loss', '?'):.4f}, "
                 f"value_loss={info.get('value_loss', '?'):.4f}, "
                 f"entropy={info.get('entropy', '?'):.4f}")

    def record_rl_token_opt(self, agent_id: int, info: dict):
        """Record a token-level optimisation decision (train or skip)."""
        decision = info.get("decision", "unknown")
        reason = info.get("reason", "")
        self.rl_token_opts.append((self.timestep, agent_id, decision, reason, info))
        if decision == "train":
            self.log(f"[RL] Agent {agent_id} TOKEN-OPT at step {self.timestep}: "
                     f"reason='{reason}', skill='{info.get('skill_focus', '?')}', "
                     f"token_policy_loss={info.get('token_policy_loss', '?')}")
        else:
            self.log(f"[RL] Agent {agent_id} token-opt SKIPPED at step {self.timestep}: "
                     f"reason='{reason}'")

    def store_timestep(self, step_comm_count: int = 0):
        """Snapshot metrics at end of a timestep."""
        self.ts_data["timesteps"].append(self.timestep)

        for i in range(self.num_agents):
            self.ts_data["cumulative_returns"][i].append(
                self.cumulative_returns[i]
            )
            total_ms = sum(
                len(self.milestones[i][t]) for t in TRACKS
            )
            self.ts_data["milestone_count"][i].append(total_ms)

        # Joint milestones (union across agents per track)
        joint = 0
        for track in TRACKS:
            achieved_tiers = set()
            for i in range(self.num_agents):
                for ts, tier_name in self.milestones[i][track]:
                    achieved_tiers.add(tier_name)
            joint += len(achieved_tiers)
        self.ts_data["total_milestones"].append(joint)

        self.comm_counts_per_step.append(step_comm_count)
        self.timestep += 1

    # ------------------------------------------------------------------
    # Milestone detection
    # ------------------------------------------------------------------

    def _detect_milestone(self, agent_id: int, reward: float):
        """Infer which track/tier a large reward belongs to.

        Stage rewards are 128, 256, 1024, 2048. The agent receives:
          own_track:   stage_value * 1.0
          other_track: stage_value * 0.3

        We check if the reward matches a known stage value (within tolerance)
        at either the 1.0x or 0.3x multiplier.
        """
        role_idx = agent_id % len(ROLE_STAGE_MULTIPLIERS)
        role_mults = ROLE_STAGE_MULTIPLIERS[role_idx]

        for track, tiers in TRACKS.items():
            mult = role_mults[track]
            for tier_idx, (tier_name, base_value) in enumerate(tiers):
                expected = base_value * mult
                if abs(reward - expected) < 1.0:
                    # Check if this tier was already achieved by this agent
                    achieved = [name for _, name in self.milestones[agent_id][track]]
                    if tier_name not in achieved:
                        self.milestones[agent_id][track].append(
                            (self.timestep, tier_name)
                        )
                        self.track_rewards[agent_id][track] += reward

                        # Record first global achievement
                        if tier_idx not in self.first_milestone_step[track]:
                            self.first_milestone_step[track][tier_idx] = self.timestep

                        logging.info(
                            f"Milestone: agent {agent_id} reached "
                            f"{track}/{tier_name} at step {self.timestep} "
                            f"(reward={reward:.1f})"
                        )
                        return

        # Not a milestone — classify as "other"
        if abs(reward) > 0.01:
            self.track_rewards[agent_id]["other"] += reward

    # ------------------------------------------------------------------
    # Computed metrics
    # ------------------------------------------------------------------

    def specialization_index(self, agent_id: int) -> dict:
        """Fraction of track rewards from each track for an agent.

        A fully specialized gatherer would have tools~1.0, hunt~0.0, defend~0.0.
        """
        tr = self.track_rewards[agent_id]
        total = sum(tr[t] for t in TRACKS)
        if total == 0:
            return {t: 0.0 for t in TRACKS}
        return {t: tr[t] / total for t in TRACKS}

    def steps_to_milestone_table(self) -> dict:
        """Returns {track: {tier_name: first_step}} for the team."""
        result = {}
        for track, tiers in TRACKS.items():
            result[track] = {}
            for tier_idx, (tier_name, _) in enumerate(tiers):
                step = self.first_milestone_step[track].get(tier_idx)
                result[track][tier_name] = step  # None if not reached
        return result

    def social_lift_data(self) -> dict:
        """Returns data needed for social lift comparison.

        To compute social lift, run once with communication=True and once
        with communication=False, then compare steps_to_milestone and
        cumulative_returns.
        """
        return {
            "communication": self.communication,
            "steps_to_milestone": self.steps_to_milestone_table(),
            "final_returns": list(self.cumulative_returns),
            "total_comm_events": len(self.communication_log),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_run_metrics(self, file_name="data.json"):
        """Save all metrics to JSON and generate plots."""
        data = {
            "config": {
                "num_agents": self.num_agents,
                "communication": self.communication,
                "total_steps": self.timestep,
            },
            "cumulative_returns": list(self.cumulative_returns),
            "steps_to_milestone": self.steps_to_milestone_table(),
            "milestones_per_agent": {
                str(i): self.milestones[i] for i in range(self.num_agents)
            },
            "specialization_index": {
                str(i): self.specialization_index(i)
                for i in range(self.num_agents)
            },
            "track_rewards": {
                str(i): self.track_rewards[i] for i in range(self.num_agents)
            },
            "social_lift_data": self.social_lift_data(),
            "timestep_data": self.ts_data,
            "comm_counts_per_step": self.comm_counts_per_step,
            "rl_updates": [
                {"timestep": ts, "agent_id": aid, "info": info}
                for ts, aid, info in self.rl_updates
            ],
            "rl_token_opts": [
                {"timestep": ts, "agent_id": aid, "decision": d, "reason": r, "info": info}
                for ts, aid, d, r, info in self.rl_token_opts
            ],
        }

        file_path = os.path.join(self.target_folder, file_name)

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as _np
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        self._save_plots()
        self._save_text_summary()

        # Save communication log separately
        comm_path = os.path.join(self.target_folder, "communication_log.json")
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump(self.communication_log, f, indent=2, ensure_ascii=False)

        print(f"Metrics saved to {self.target_folder}")
        return file_path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _save_plots(self):
        path = self.target_folder
        ts = self.ts_data["timesteps"]
        if not ts:
            return

        # 1. Cumulative return per agent
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(self.num_agents):
            ax.plot(ts, self.ts_data["cumulative_returns"][i],
                    label=f"Agent {i}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Cumulative Return per Agent")
        ax.legend()
        fig.savefig(os.path.join(path, "cumulative_returns.png"), dpi=150)
        plt.close(fig)

        # 2. Milestones over time (per agent + joint)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(self.num_agents):
            ax.plot(ts, self.ts_data["milestone_count"][i],
                    label=f"Agent {i}", alpha=0.7)
        ax.plot(ts, self.ts_data["total_milestones"],
                label="Joint (team)", linewidth=2, color="black")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Milestones Achieved")
        ax.set_title("Track Milestones Over Time")
        ax.legend()
        fig.savefig(os.path.join(path, "milestones.png"), dpi=150)
        plt.close(fig)

        # 3. Specialization index bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(self.num_agents)
        width = 0.25
        for idx, track in enumerate(TRACKS):
            vals = [self.specialization_index(i).get(track, 0) for i in range(self.num_agents)]
            ax.bar(x + idx * width, vals, width, label=track.capitalize())
        ax.set_xlabel("Agent")
        ax.set_ylabel("Fraction of Track Rewards")
        ax.set_title("Specialization Index")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"Agent {i}" for i in range(self.num_agents)])
        ax.legend()
        fig.savefig(os.path.join(path, "specialization_index.png"), dpi=150)
        plt.close(fig)

        # 4. Steps-to-milestone table as text file
        table = self.steps_to_milestone_table()
        lines = ["Steps to Milestone (first agent to reach)\n"]
        lines.append(f"{'Track':<10} {'Tier':<15} {'Step':>8}")
        lines.append("-" * 35)
        for track, tiers in table.items():
            for tier_name, step in tiers.items():
                step_str = str(step) if step is not None else "---"
                lines.append(f"{track:<10} {tier_name:<15} {step_str:>8}")
        with open(os.path.join(path, "steps_to_milestone.txt"), "w") as f:
            f.write("\n".join(lines))

        # 5. Communication frequency over time
        if self.comm_counts_per_step:
            fig, ax = plt.subplots(figsize=(10, 4))
            # Smooth with rolling window
            window = min(50, len(self.comm_counts_per_step))
            if window > 1:
                smoothed = np.convolve(
                    self.comm_counts_per_step,
                    np.ones(window) / window,
                    mode="valid",
                )
                ax.plot(range(len(smoothed)), smoothed)
            else:
                ax.plot(self.comm_counts_per_step)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Messages per Step (smoothed)")
            ax.set_title("Communication Frequency")
            fig.savefig(os.path.join(path, "communication_frequency.png"), dpi=150)
            plt.close(fig)

    def _save_text_summary(self):
        """Write a human-readable summary.txt with all key metrics."""
        role_names = ["gatherer", "hunter", "defender"]
        lines = []
        lines.append("=" * 50)
        lines.append("  Experiment Summary")
        lines.append("=" * 50)
        comm_str = "on" if self.communication else "off"
        lines.append(f"Agents: {self.num_agents}  |  Steps: {self.timestep}  |  Communication: {comm_str}")
        lines.append("")

        # Final cumulative returns
        lines.append("--- Cumulative Returns ---")
        for i in range(self.num_agents):
            role = role_names[i % len(role_names)]
            lines.append(f"  Agent {i} ({role}): {self.cumulative_returns[i]:.2f}")
        lines.append("")

        # Milestones per agent
        lines.append("--- Milestones Reached ---")
        for i in range(self.num_agents):
            role = role_names[i % len(role_names)]
            agent_ms = []
            for track in TRACKS:
                for ts, tier_name in self.milestones[i][track]:
                    agent_ms.append(f"{track}/{tier_name} (step {ts})")
            if agent_ms:
                lines.append(f"  Agent {i} ({role}): {', '.join(agent_ms)}")
            else:
                lines.append(f"  Agent {i} ({role}): none")
        lines.append("")

        # Steps to milestone (team-level)
        lines.append("--- Steps to Milestone (first agent to reach) ---")
        table = self.steps_to_milestone_table()
        for track, tiers in table.items():
            for tier_name, step in tiers.items():
                step_str = str(step) if step is not None else "---"
                lines.append(f"  {track:<10} {tier_name:<15} {step_str:>8}")
        lines.append("")

        # Specialization index
        lines.append("--- Specialization Index ---")
        for i in range(self.num_agents):
            role = role_names[i % len(role_names)]
            si = self.specialization_index(i)
            parts = [f"{t}={si[t]:.2f}" for t in TRACKS]
            lines.append(f"  Agent {i} ({role}): {', '.join(parts)}")
        lines.append("")

        # Communication stats
        total_msgs = len(self.communication_log)
        avg_per_step = total_msgs / max(self.timestep, 1)
        lines.append("--- Communication ---")
        lines.append(f"  Total messages: {total_msgs}")
        lines.append(f"  Avg per step:   {avg_per_step:.2f}")
        lines.append("")

        # RL stats
        if self.rl_updates or self.rl_token_opts:
            lines.append("--- RL Layer ---")
            lines.append(f"  MAPPO updates: {len(self.rl_updates)}")
            if self.rl_updates:
                last = self.rl_updates[-1][2]
                lines.append(f"    Last update — policy_loss={last.get('policy_loss', '?'):.4f}, "
                             f"value_loss={last.get('value_loss', '?'):.4f}, "
                             f"entropy={last.get('entropy', '?'):.4f}")
            train_count = sum(1 for _, _, d, _, _ in self.rl_token_opts if d == "train")
            skip_count = sum(1 for _, _, d, _, _ in self.rl_token_opts if d != "train")
            lines.append(f"  Token-opt decisions: {train_count} train, {skip_count} skip")
            lines.append("")

        lines.append("=" * 50)

        summary_path = os.path.join(self.target_folder, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mkdir_metrics(self, path="./run_metrics"):
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comm_str = "comm" if self.communication else "noComm"
        folder_name = f"craftium_{self.num_agents}agents_{comm_str}_{timestamp}"
        target = os.path.join(path, folder_name)
        os.makedirs(target, exist_ok=True)
        return target

    def log(self, text, filepath="log.txt"):
        full_path = os.path.join(self.target_folder, filepath)
        with open(full_path, "a") as f:
            f.write(text + "\n")

    # ------------------------------------------------------------------
    # Compatibility with CustomAgent (which calls these on self.metric)
    # ------------------------------------------------------------------

    def found_skill(self, description: str, main=True):
        """Record a learned skill (maps to milestone tracking)."""
        logging.info(f"Skill learned: {description}")

    def save_predictions(self, *args, **kwargs):
        """No-op: causal predictions not used in Craftium."""
        pass

    def check_surgical(self, action, held_item, valid_interventions=None):
        """No-op: surgical interventions not used in Craftium."""
        return False, ""
