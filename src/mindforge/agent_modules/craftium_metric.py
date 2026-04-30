"""Evaluation metrics for Craftium multi-agent experiments.

Tracks:
- Cumulative return per agent
- Milestone events from five-chambers JSONL (M1-M28)
- Steps-to-milestone per track
- Communication events
- Generates plots and saves JSON data
"""

import json
import logging
import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# ─── Milestone / track definitions ─────────────────────────────────────

# milestone_id -> track name
MILESTONE_TRACK = {
    "m1_move_5":               "ch1_solo",
    "m2_dig_3_any":            "ch1_solo",
    "m3_pickup_3":             "ch1_solo",
    "m4_dig_5_wood":           "ch1_solo",
    "m5_kill_1_animal":        "ch1_solo",
    "m6_kill_2_animals":       "ch1_solo",
    "m7_dig_3_stone":          "ch1_solo",
    "m8_anvil_A1":             "ch2_anvils",
    "m9_anvil_A2":             "ch2_anvils",
    "m10_anvil_A3":            "ch2_anvils",
    "m11_anvil_B1":            "ch2_anvils",
    "m12_anvil_B2":            "ch2_anvils",
    "m13_anvil_B3":            "ch2_anvils",
    "m14_sword_equipped":      "ch2_anvils",
    "m15_chestplate_equipped": "ch2_anvils",
    "m16_enter_cell":          "ch3_switches",
    "m17_switch_pressed":      "ch3_switches",
    "m18_door_opened":         "ch3_switches",
    "m19_all_in_communal":     "ch3_switches",
    "m20_enter_ch4":           "ch4_combat",
    "m21_first_mob_kill":      "ch4_combat",
    "m22_all_mobs_killed":     "ch4_combat",
    "m23_all_alive_ch4":       "ch4_combat",
    "m24_enter_ch5":           "ch5_boss",
    "m25_first_boss_dmg":      "ch5_boss",
    "m26_boss_half_hp":        "ch5_boss",
    "m27_boss_defeated":       "ch5_boss",
    "m28_all_alive_bonus":     "ch5_boss",
    "m_comm_ch1":              "communication",
    "m_comm_ch2":              "communication",
    "m_comm_ch3":              "communication",
    "m_comm_ch4":              "communication",
    "m_comm_ch5":              "communication",
}

# Ordered (milestone_id, reward) per track — drives steps-to-milestone table + reward total
TRACKS = {
    "ch1_solo": [
        ("m1_move_5", 10.0), ("m2_dig_3_any", 30.0), ("m3_pickup_3", 30.0),
        ("m4_dig_5_wood", 50.0), ("m5_kill_1_animal", 50.0),
        ("m6_kill_2_animals", 80.0), ("m7_dig_3_stone", 60.0),
    ],
    "ch2_anvils": [
        ("m8_anvil_A1",  40.0), ("m9_anvil_A2",  40.0), ("m10_anvil_A3", 40.0),
        ("m11_anvil_B1", 40.0), ("m12_anvil_B2", 40.0), ("m13_anvil_B3", 40.0),
        ("m14_sword_equipped", 50.0), ("m15_chestplate_equipped", 30.0),
    ],
    "ch3_switches": [
        ("m16_enter_cell", 20.0), ("m17_switch_pressed", 40.0),
        ("m18_door_opened", 60.0), ("m19_all_in_communal", 100.0),
    ],
    "ch4_combat": [
        ("m20_enter_ch4", 30.0), ("m21_first_mob_kill", 60.0),
        ("m22_all_mobs_killed", 150.0), ("m23_all_alive_ch4", 100.0),
    ],
    "ch5_boss": [
        ("m24_enter_ch5", 50.0), ("m25_first_boss_dmg", 80.0),
        ("m26_boss_half_hp", 120.0), ("m27_boss_defeated", 300.0),
        ("m28_all_alive_bonus", 250.0),
    ],
    "communication": [
        ("m_comm_ch1", 40.0),
        ("m_comm_ch2", 20.0), ("m_comm_ch3", 30.0),
        ("m_comm_ch4", 15.0), ("m_comm_ch5", 20.0),
    ],
}

STAGE_REWARDS = {
    10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0,
    100.0, 120.0, 150.0, 250.0, 300.0,
}

TRACK_ORDER = list(TRACKS.keys())

# Two milestones fired by different agents within this many steps count as co-completion.
_CO_COMPLETION_WINDOW = 5


# ─── Module helpers ────────────────────────────────────────────────────

def _agent_id_from_name(name: str) -> int:
    """Return integer id from "agent_N" or -1 on malformed input."""
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _get_git_info() -> dict:
    """Return current commit + branch, or {key: None} on failure."""
    info = {}
    for key, cmd in [
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ]:
        try:
            info[key] = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
        except Exception:
            info[key] = None
    return info


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── Main class ────────────────────────────────────────────────────────

class CraftiumMetric:
    """Tracks evaluation metrics for Craftium five-chambers multi-agent experiments."""

    def __init__(
        self,
        num_agents=3,
        communication=True,
        path="./run_metrics",
        run_id=None,
    ):
        self.num_agents = num_agents
        self.communication = communication
        self.run_id = run_id
        self.timestep = 0

        # Rewards
        self.cumulative_returns = [0.0] * num_agents
        self.reward_history = [[] for _ in range(num_agents)]

        # Milestones
        self.milestone_events = []                  # flat log: (milestone, contributor) pairs
        self._agent_milestones = {}                 # agent_name -> set of milestone ids
        self.first_milestone_step = {}              # mid -> first global step (any agent)
        self.track_rewards = {                      # per-agent, per-track reward sum
            i: {track: 0.0 for track in TRACKS}
            for i in range(num_agents)
        }

        # Communication
        self.communication_log = []
        self.comm_counts_per_step = []

        # RL
        self.rl_updates = []
        self.rl_token_opts = []

        # Social graph / phases
        self._graph_snapshots = []
        self._co_completion_events = []
        self._last_milestone_step = {}              # agent_id -> last milestone timestep
        self.phase_transitions = []

        # Team composition metadata
        self.team_mode = "heterogeneous"
        self.homogeneous_role = "agent"

        # Per-timestep rollups for plotting
        self.ts_data = {
            "timesteps": [],
            "cumulative_returns": [[] for _ in range(num_agents)],
            "milestone_count":    [[] for _ in range(num_agents)],
            "total_milestones":   [],
        }

        self.target_folder = self._mkdir_metrics(path)

    # ─── Recording ─────────────────────────────────────────────────────

    def record_reward(self, agent_id: int, reward: float):
        self.cumulative_returns[agent_id] += reward
        self.reward_history[agent_id].append((self.timestep, reward))

    def record_milestone_event(self, ev: dict):
        """Record a milestone event from poll_milestone_events().

        ev = {"step": int, "milestone": str, "contributors": [str, ...], "reward": int}
        """
        mid = ev.get("milestone", "")
        reward = ev.get("reward", 0)
        lua_step = ev.get("step", self.timestep)
        contributors = ev.get("contributors", [])

        self.first_milestone_step.setdefault(mid, self.timestep)

        for name in contributors:
            self._append_milestone_event(mid, lua_step, name, reward)
            self._register_co_completion(_agent_id_from_name(name), mid)

        logging.info(
            "Milestone %s fired for %s at step %d (reward=%d)",
            mid, contributors, self.timestep, reward,
        )

    def _append_milestone_event(self, mid, lua_step, agent_name, reward):
        self.milestone_events.append({
            "step":         self.timestep,
            "lua_step":     lua_step,
            "milestone_id": mid,
            "contributor":  agent_name,
            "reward":       reward,
        })
        self._agent_milestones.setdefault(agent_name, set()).add(mid)

        agent_id = _agent_id_from_name(agent_name)
        track = MILESTONE_TRACK.get(mid)
        if track and 0 <= agent_id < self.num_agents:
            self.track_rewards[agent_id][track] += reward

    def _register_co_completion(self, agent_id: int, mid: str):
        if not (0 <= agent_id < self.num_agents):
            return
        for other_id, other_step in self._last_milestone_step.items():
            if other_id != agent_id and (self.timestep - other_step) <= _CO_COMPLETION_WINDOW:
                self._co_completion_events.append({
                    "step":      self.timestep,
                    "agent_i":   agent_id,
                    "agent_j":   other_id,
                    "milestone": mid,
                })
        self._last_milestone_step[agent_id] = self.timestep

    def record_communication(self, source_agent: str, message: str, target: str = None):
        preview = message[:100] if message else ""
        self.communication_log.append(
            (self.timestep, source_agent, preview, target or "all")
        )

    def record_rl_update(self, agent_id: int, info: dict):
        self.rl_updates.append((self.timestep, agent_id, info))
        logging.info(
            "[RL] Agent %d MAPPO update at step %d: "
            "policy_loss=%.4f, value_loss=%.4f, entropy=%.4f",
            agent_id, self.timestep,
            info.get("policy_loss", 0),
            info.get("value_loss", 0),
            info.get("entropy", 0),
        )

    def record_rl_token_opt(self, agent_id: int, info: dict):
        decision = info.get("decision", "unknown")
        reason = info.get("reason", "")
        self.rl_token_opts.append((self.timestep, agent_id, decision, reason, info))

    def record_graph_snapshot(self, step: int, graph_dict: dict):
        self._graph_snapshots.append({"step": step, **graph_dict})

    def record_phase_transition(self, step: int, episode: int, phase: str):
        self.phase_transitions.append({"step": step, "episode": episode, "phase": phase})
        logging.info("[PHASE] ep=%d step=%d → %s", episode, step, phase)

    def store_timestep(self, step_comm_count: int = 0):
        """Snapshot per-timestep metrics and advance the timestep counter."""
        self.ts_data["timesteps"].append(self.timestep)

        for i in range(self.num_agents):
            self.ts_data["cumulative_returns"][i].append(self.cumulative_returns[i])
            count = len(self._agent_milestones.get(f"agent_{i}", set()))
            self.ts_data["milestone_count"][i].append(count)

        joint = set().union(*self._agent_milestones.values()) if self._agent_milestones else set()
        self.ts_data["total_milestones"].append(len(joint))

        self.comm_counts_per_step.append(step_comm_count)
        self.timestep += 1

    # ─── Computed metrics ──────────────────────────────────────────────

    def specialization_index(self, agent_id: int) -> dict:
        tr = self.track_rewards[agent_id]
        total = sum(tr.values())
        if total == 0:
            return {t: 0.0 for t in TRACKS}
        return {t: tr[t] / total for t in TRACKS}

    def steps_to_milestone_table(self) -> dict:
        """{track: {milestone_id: first_step_or_None}} for the team."""
        return {
            track: {mid: self.first_milestone_step.get(mid) for mid, _ in entries}
            for track, entries in TRACKS.items()
        }

    def social_lift_data(self) -> dict:
        return {
            "communication": self.communication,
            "steps_to_milestone": self.steps_to_milestone_table(),
            "final_returns": list(self.cumulative_returns),
            "total_comm_events": len(self.communication_log),
        }

    def milestones_per_agent(self) -> dict:
        return {
            name: sorted(ms_set)
            for name, ms_set in self._agent_milestones.items()
        }

    # ─── Saving ────────────────────────────────────────────────────────

    def save_run_metrics(self, file_name="data.json"):
        data = self._build_metrics_dict()
        file_path = os.path.join(self.target_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

        self._save_plots()
        self._save_text_summary()

        comm_path = os.path.join(self.target_folder, "communication_log.json")
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump(self.communication_log, f, indent=2, ensure_ascii=False)

        print(f"Metrics saved to {self.target_folder}")
        return file_path

    def _build_metrics_dict(self) -> dict:
        git = _get_git_info()
        return {
            "config": {
                "num_agents":           self.num_agents,
                "communication":        self.communication,
                "total_steps":          self.timestep,
                "seed":                 getattr(self, "seed", None),
                "max_steps_per_episode": getattr(self, "max_steps", None),
                "num_episodes":         getattr(self, "num_episodes", None),
                "experiment_id":        getattr(self, "experiment_id", None),
                "timestamp":            datetime.now().isoformat(),
                "git_commit":           git.get("git_commit"),
                "git_branch":           git.get("git_branch"),
                "cli_args":             getattr(self, "cli_args", None),
            },
            "cumulative_returns":   list(self.cumulative_returns),
            "steps_to_milestone":   self.steps_to_milestone_table(),
            "milestones_per_agent": self.milestones_per_agent(),
            "milestone_events":     self.milestone_events,
            "specialization_index": {
                str(i): self.specialization_index(i) for i in range(self.num_agents)
            },
            "track_rewards": {
                str(i): self.track_rewards[i] for i in range(self.num_agents)
            },
            "social_lift_data":     self.social_lift_data(),
            "timestep_data":        self.ts_data,
            "comm_counts_per_step": self.comm_counts_per_step,
            "rl_updates": [
                {"timestep": ts, "agent_id": aid, "info": info}
                for ts, aid, info in self.rl_updates
            ],
            "rl_token_opts": [
                {"timestep": ts, "agent_id": aid, "decision": d, "reason": r, "info": info}
                for ts, aid, d, r, info in self.rl_token_opts
            ],
            "graph_snapshots":      self._graph_snapshots,
            "co_completion_events": self._co_completion_events,
            "phase_transitions":    self.phase_transitions,
            "team_mode":            self.team_mode,
            "homogeneous_role":     self.homogeneous_role,
        }

    # ─── Plots ─────────────────────────────────────────────────────────

    def _save_plots(self):
        if not self.ts_data["timesteps"]:
            return
        self._plot_cumulative_returns()
        self._plot_milestones()
        self._plot_track_rewards()
        self._write_steps_to_milestone_txt()
        self._plot_communication_frequency()
        self._plot_hebbian_bonds()

    def _plot_cumulative_returns(self):
        ts = self.ts_data["timesteps"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(self.num_agents):
            ax.plot(ts, self.ts_data["cumulative_returns"][i], label=f"Agent {i}")
        for pt in self.phase_transitions:
            ax.axvline(x=pt["step"], color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Cumulative Return per Agent")
        ax.legend()
        fig.savefig(os.path.join(self.target_folder, "cumulative_returns.png"), dpi=150)
        plt.close(fig)

    def _plot_milestones(self):
        ts = self.ts_data["timesteps"]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(self.num_agents):
            ax.plot(ts, self.ts_data["milestone_count"][i], label=f"Agent {i}", alpha=0.7)
        ax.plot(ts, self.ts_data["total_milestones"],
                label="Joint (unique)", linewidth=2, color="black")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Milestones Achieved")
        ax.set_title("Five-Chambers Milestone Progress")
        ax.legend()
        fig.savefig(os.path.join(self.target_folder, "milestones.png"), dpi=150)
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
        ax.set_ylabel("Total Milestone Reward")
        ax.set_title("Reward by Chamber Track per Agent")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Agent {i}" for i in range(self.num_agents)])
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(os.path.join(self.target_folder, "track_rewards.png"), dpi=150)
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

    def _plot_communication_frequency(self):
        if not self.comm_counts_per_step:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        window = min(50, len(self.comm_counts_per_step))
        if window > 1:
            smoothed = np.convolve(
                self.comm_counts_per_step, np.ones(window) / window, mode="valid"
            )
            ax.plot(range(len(smoothed)), smoothed)
        else:
            ax.plot(self.comm_counts_per_step)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Messages per Step (smoothed)")
        ax.set_title("Communication Frequency")
        fig.savefig(os.path.join(self.target_folder, "communication_frequency.png"), dpi=150)
        plt.close(fig)

    def _plot_hebbian_bonds(self):
        if not self._graph_snapshots:
            return
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
        ax.set_title("Hebbian Social Graph — Bond Evolution")
        ax.legend()
        fig.savefig(os.path.join(self.target_folder, "graph_bond_evolution.png"), dpi=150)
        plt.close(fig)

    @staticmethod
    def _bond_weight_at(snapshot, i_idx, j_idx) -> float:
        for tp in snapshot.get("top_3_pairs", []):
            if tp["i"] == i_idx and tp["j"] == j_idx:
                return tp["w"]
        return 0.0

    # ─── Text summary ──────────────────────────────────────────────────

    def _save_text_summary(self):
        lines = []
        lines.extend(self._summary_header())
        lines.extend(self._summary_returns())
        lines.extend(self._summary_milestones())
        lines.extend(self._summary_steps_to_milestone())
        lines.extend(self._summary_specialization())
        lines.extend(self._summary_communication())
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
        lines = ["--- Cumulative Returns ---"]
        for i in range(self.num_agents):
            lines.append(f"  Agent {i} (agent): {self.cumulative_returns[i]:.2f}")
        lines.append("")
        return lines

    def _summary_milestones(self):
        lines = ["--- Milestones per Agent ---"]
        for i in range(self.num_agents):
            earned = sorted(self._agent_milestones.get(f"agent_{i}", set()))
            lines.append(f"  Agent {i} (agent): {', '.join(earned) if earned else 'none'}")
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

    def _summary_rl(self):
        if not (self.rl_updates or self.rl_token_opts):
            return []
        lines = ["--- RL Layer ---", f"  MAPPO updates: {len(self.rl_updates)}"]
        if self.rl_updates:
            last = self.rl_updates[-1][2]
            lines.append(
                f"    Last: policy_loss={last.get('policy_loss', 0):.4f}, "
                f"value_loss={last.get('value_loss', 0):.4f}"
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

    # ─── Checkpoint restore ────────────────────────────────────────────

    @classmethod
    def restore_from_dict(cls, d: dict, path: str = "./run_metrics") -> "CraftiumMetric":
        num_agents = d["num_agents"]
        metric = cls(
            num_agents=num_agents,
            communication=d.get("communication", True),
            path=path,
            run_id=d.get("run_id"),
        )

        metric.timestep = d.get("timestep", 0)
        metric.cumulative_returns = [
            float(x) for x in d.get("cumulative_returns", [0.0] * num_agents)
        ]
        metric.reward_history = [
            [tuple(x) for x in agent_h]
            for agent_h in d.get("reward_history", [[] for _ in range(num_agents)])
        ]

        metric.milestone_events = d.get("milestone_events", [])
        metric.first_milestone_step = dict(d.get("first_milestone_step", {}))
        metric._agent_milestones = {
            name: set(ids) for name, ids in d.get("milestones_per_agent", {}).items()
        }

        tr = d.get("track_rewards", {})
        metric.track_rewards = {
            i: tr.get(str(i), {t: 0.0 for t in TRACKS})
            for i in range(num_agents)
        }

        metric.communication_log    = [tuple(x) for x in d.get("communication_log", [])]
        metric.comm_counts_per_step = d.get("comm_counts_per_step", [])
        metric.rl_updates           = [tuple(x) for x in d.get("rl_updates", [])]
        metric.rl_token_opts        = [tuple(x) for x in d.get("rl_token_opts", [])]
        metric._graph_snapshots     = d.get("_graph_snapshots", d.get("graph_snapshots", []))
        metric._co_completion_events = d.get("co_completion_events", [])
        metric.phase_transitions    = d.get("phase_transitions", [])
        metric.team_mode            = d.get("team_mode", "heterogeneous")
        metric.homogeneous_role     = d.get("homogeneous_role", "agent")
        metric._last_milestone_step = {
            int(k): v for k, v in d.get("_last_milestone_step", {}).items()
        }

        metric.ts_data = d.get("ts_data", {
            "timesteps": [],
            "cumulative_returns": [[] for _ in range(num_agents)],
            "milestone_count":    [[] for _ in range(num_agents)],
            "total_milestones":   [],
        })
        return metric

    # ─── Misc ──────────────────────────────────────────────────────────

    def _mkdir_metrics(self, path="./run_metrics"):
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comm_str  = "comm" if self.communication else "noComm"
        base = self.run_id or f"five_chambers_{self.num_agents}agents_{comm_str}_{timestamp}"
        target = os.path.join(path, base)
        os.makedirs(target, exist_ok=True)
        return target

    def log(self, text, filepath="log.txt"):
        full_path = os.path.join(self.target_folder, filepath)
        with open(full_path, "a") as f:
            f.write(text + "\n")

    # ─── Compatibility stubs ───────────────────────────────────────────

    def found_skill(self, description: str, main=True):
        logging.info("Skill learned: %s", description)

    def save_predictions(self, *args, **kwargs):
        pass

    def check_surgical(self, action, held_item, valid_interventions=None):
        return False, ""
