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
import statistics
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
    "m11_anvil_B1":            "ch2_anvils",
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
        # Ch2 has exactly 2 anvils. The 6-anvil m9/m10/m12/m13 entries
        # were a LEGACY design and have been removed from Lua's
        # MILESTONE_DEFS — keep them out here too so plots and the
        # milestone-progress prompt block don't show ghost OPEN entries.
        ("m8_anvil_A1",  40.0), ("m11_anvil_B1", 40.0),
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


# Chambers as the prompt sees them → track name in TRACKS.
_PROMPT_CHAMBER_TO_TRACK = {
    "ch1":           "ch1_solo",
    "ch2":           "ch2_anvils",
    "ch3":           "ch3_switches",
    "ch3_communal":  "ch3_switches",
    "ch4":           "ch4_combat",
    "ch5":           "ch5_boss",
}


def format_milestone_progress(current_chamber, agent_completed, team_completed):
    """Format a milestone-progress block for the agent prompts.

    Used by BOTH the curriculum LLM (so it can pick a task targeting an
    open milestone) and the action LLM (so it can pick an action that
    advances an open milestone). Tells the model:

      - exactly which milestones THIS agent has already fired,
      - which ones a teammate fired (so we don't redundantly chase
        team-shared milestones like M_door1_open),
      - which ones are still open per chamber,
      - which chamber the agent is CURRENTLY in (so it focuses there).

    Returns a multi-line plain-string block, ready to drop into a
    ``{milestone_progress}`` placeholder. Communication milestones
    (m_comm_*) are stage-agnostic chatter rewards and are intentionally
    excluded — they fire automatically whenever agents talk.
    """
    agent_completed = set(agent_completed or [])
    team_completed  = set(team_completed  or [])
    current_track   = _PROMPT_CHAMBER_TO_TRACK.get(current_chamber)

    chamber_label = {
        "ch1_solo":     "Ch1",
        "ch2_anvils":   "Ch2",
        "ch3_switches": "Ch3",
        "ch4_combat":   "Ch4",
        "ch5_boss":     "Ch5",
    }

    lines = []
    for track in TRACK_ORDER:
        if track == "communication":
            continue
        label = chamber_label.get(track, track)
        all_mids   = [mid for mid, _ in TRACKS[track]]
        you_done   = [m for m in all_mids if m in agent_completed]
        team_only  = [
            m for m in all_mids
            if m in team_completed and m not in agent_completed
        ]
        remaining  = [m for m in all_mids if m not in team_completed]

        marker = "  ← YOU ARE HERE" if track == current_track else ""
        parts = [f"  {label}{marker}:"]
        if you_done:
            parts.append(f"[you done] {', '.join(you_done)}")
        if team_only:
            parts.append(f"[team done, you didn't fire] {', '.join(team_only)}")
        if remaining:
            parts.append(f"[OPEN] {', '.join(remaining)}")
        else:
            parts.append("[chamber complete]")
        lines.append(" ".join(parts))
    return "\n".join(lines)

# Two milestones fired by different agents within this many steps count as co-completion.
_CO_COMPLETION_WINDOW = 5


# ─── Module helpers ────────────────────────────────────────────────────

def _agent_id_from_name(name: str) -> int:
    """Return integer id from "agent_N" / "agentN" or -1 on malformed input.

    The Lua side emits player names without an underscore ('agent0') —
    Craftium's player-name convention. Python often uses 'agent_0'. The
    old `int(name.split('_')[1])` parser only handled the underscored form
    and returned -1 for everything else. Every Lua-sourced contributor
    name was silently bucketed as -1 and rejected by downstream
    ``0 <= agent_id < num_agents`` guards — track_rewards and
    co-completion bookkeeping never got credited from Lua milestones.
    """
    s = str(name).removeprefix("agent_").removeprefix("agent")
    try:
        return int(s)
    except ValueError:
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
        run_paths=None,
    ):
        """Build a CraftiumMetric.

        Parameters
        ----------
        run_paths : RunPaths or None
            Preferred. When supplied, every artifact lands under
            ``run_paths.root`` (i.e. ``runs/<run_id>/``).
        path : str
            Legacy fallback. Used only when ``run_paths`` is None — produces
            the old ``./run_metrics/<run_id>/`` layout. Kept for tooling that
            constructs CraftiumMetric outside the main loop.
        """
        self.num_agents = num_agents
        self.communication = communication
        self.run_id = run_id
        self.run_paths = run_paths
        self.timestep = 0

        # Rewards
        self.cumulative_returns = [0.0] * num_agents
        # Per-episode return for the CURRENTLY-RUNNING episode. Reset to
        # zero by end_episode() after the previous episode's deltas are
        # snapshotted into per_episode_returns. This is what gets reported
        # in the per-step log and per-episode summary so episodes aren't
        # contaminated by carry-over from prior episodes.
        self.episode_returns = [0.0] * num_agents
        # History: per_episode_returns[i] = list of returns for agent i,
        # one entry per completed episode. Final-summary mean/std are
        # computed from this list.
        self.per_episode_returns = [[] for _ in range(num_agents)]
        self.reward_history = [[] for _ in range(num_agents)]
        # Reward decomposition: parallel to reward_history, splits each step
        # reward into its source streams. Lets us answer "how much of the
        # policy improvement came from comm rewards vs. task reward".
        self.reward_history_decomposed = [[] for _ in range(num_agents)]

        # Milestones
        self.milestone_events = []                  # flat log: (milestone, contributor) pairs
        self._agent_milestones = {}                 # agent_name -> set of milestone ids
        self.first_milestone_step = {}              # mid -> first global step (any agent)
        self.track_rewards = {                      # per-agent, per-track reward sum
            i: {track: 0.0 for track in TRACKS}
            for i in range(num_agents)
        }
        # Per-episode mirrors for aggregable headline numbers. Reset by
        # end_episode(), snapshotted into *_per_episode for cross-episode
        # mean/std reporting in final_metrics.json / summary.txt.
        self.track_rewards_episode = {
            i: {track: 0.0 for track in TRACKS}
            for i in range(num_agents)
        }
        self.track_rewards_per_episode = [[] for _ in range(num_agents)]
        # agent_id -> set of milestone ids reached THIS episode (clears each ep)
        self._agent_milestones_episode = {i: set() for i in range(num_agents)}
        # per-agent list of per-episode milestone id sets (kept as sorted lists
        # for JSON serialisation)
        self.milestones_per_episode = [[] for _ in range(num_agents)]
        # per-agent message-sent count for the current episode
        self.comm_count_episode = [0] * num_agents
        # per-agent list of per-episode message counts
        self.comm_count_per_episode = [[] for _ in range(num_agents)]
        # episode lengths (final_step at end_episode time), shared across agents
        self.episode_lengths = []

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

        # If a RunPaths was provided, the run root IS the metrics folder.
        # Falls back to the legacy ./run_metrics/<run_id>/ layout otherwise.
        if run_paths is not None:
            self.target_folder = str(run_paths.root)
        else:
            self.target_folder = self._mkdir_metrics(path)

    # ─── Recording ─────────────────────────────────────────────────────

    def record_reward(self, agent_id: int, reward: float):
        self.cumulative_returns[agent_id] += reward
        self.episode_returns[agent_id] += reward
        self.reward_history[agent_id].append((self.timestep, reward))

    def end_episode(self, final_step: int = 0):
        """Snapshot the just-finished episode's per-agent aggregables into
        the *_per_episode histories and reset the per-episode mirrors.
        Must be called once per episode, after the final record_* call and
        before the next episode's first step.

        Covers: reward, track_rewards, milestones reached, message count,
        episode length. final_step is what gets stored in episode_lengths.
        """
        for i in range(self.num_agents):
            self.per_episode_returns[i].append(self.episode_returns[i])
            self.track_rewards_per_episode[i].append(
                dict(self.track_rewards_episode[i])
            )
            self.milestones_per_episode[i].append(
                sorted(self._agent_milestones_episode[i])
            )
            self.comm_count_per_episode[i].append(self.comm_count_episode[i])
        self.episode_lengths.append(int(final_step))

        self.episode_returns = [0.0] * self.num_agents
        self.track_rewards_episode = {
            i: {track: 0.0 for track in TRACKS}
            for i in range(self.num_agents)
        }
        self._agent_milestones_episode = {
            i: set() for i in range(self.num_agents)
        }
        self.comm_count_episode = [0] * self.num_agents

    def record_reward_decomposed(self, agent_id: int, components: dict):
        """Record a per-step reward broken down by source.

        components keys (all floats, default 0):
          task              env-step reward + pitch-cap penalty + drained
                            five-chambers milestone rewards (m1..m28)
          comm_base         BASE_MSG_REWARD per valid message
          comm_milestone    Tier-2 per-chamber communication milestones
          proximity         vestigial — the +0.3/pair proximity bonus was
                            removed; field stays at 0 for schema back-compat
          hebbian_diffuse   reward bled from peers via Hebbian W (signed)

        The five streams sum to the value passed to record_reward().
        Persisted to reward_history_decomposed so we can answer "what fraction
        of cumulative return came from each source".
        """
        rec = {
            "t": self.timestep,
            "task":           float(components.get("task", 0.0)),
            "comm_base":      float(components.get("comm_base", 0.0)),
            "comm_milestone": float(components.get("comm_milestone", 0.0)),
            "proximity":      float(components.get("proximity", 0.0)),
            "hebbian_diffuse": float(components.get("hebbian_diffuse", 0.0)),
        }
        self.reward_history_decomposed[agent_id].append(rec)

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
            self.track_rewards_episode[agent_id][track] += reward
        if 0 <= agent_id < self.num_agents:
            self._agent_milestones_episode[agent_id].add(mid)

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
        agent_id = _agent_id_from_name(source_agent)
        if 0 <= agent_id < self.num_agents:
            self.comm_count_episode[agent_id] += 1

    def record_rl_update(self, agent_id: int, info: dict):
        self.rl_updates.append((self.timestep, agent_id, info))
        if "critic_loss" in info:
            # Centralized-critic update — distinct from per-agent action updates.
            # explained_variance is the headline metric: <0.1 means the critic
            # is predicting the mean and advantages will be near-noise.
            logging.info(
                "[RL] Centralized critic update #%d at step %d: "
                "loss=%.4f  ev=%.3f  ret(μ=%.2f σ=%.2f range=[%.1f,%.1f])  "
                "v_old(μ=%.2f σ=%.2f)  buf=%d",
                info.get("critic_update_count", 0), self.timestep,
                info.get("critic_loss", 0),
                info.get("critic_explained_variance", 0),
                info.get("critic_returns_mean", 0),
                info.get("critic_returns_std", 0),
                info.get("critic_returns_min", 0),
                info.get("critic_returns_max", 0),
                info.get("critic_values_mean", 0),
                info.get("critic_values_std", 0),
                info.get("critic_buffer_size", 0),
            )
        else:
            # adv_std and frac_pos_advantage diagnose advantage degeneration.
            # ratio_max > 2 means data is off-policy.
            logging.info(
                "[RL] Agent %d MAPPO update at step %d: "
                "policy=%.4f  entropy=%.4f  kl=%.4f  clip_frac=%.3f  "
                "ratio_max=%.2f  adv(μ=%.3f σ=%.3f range=[%.2f,%.2f] frac+=%.2f)",
                agent_id, self.timestep,
                info.get("policy_loss", 0),
                info.get("entropy", 0),
                info.get("approx_kl", 0),
                info.get("clip_frac", 0),
                info.get("ratio_max", 1.0),
                info.get("adv_mean", 0),
                info.get("adv_std", 0),
                info.get("adv_min", 0),
                info.get("adv_max", 0),
                info.get("frac_pos_advantage", 0.5),
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

    def save_run_metrics(self, file_name="final_metrics.json"):
        # Run post-hoc evaluators that read the on-disk JSONL artifacts.
        # Done before _build_metrics_dict so the new fields are in the dump.
        self._run_posthoc_evaluators()

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

    def _run_posthoc_evaluators(self):
        """Compute communication & cooperation metrics from the on-disk JSONL.

        These don't need any extra recording during the hot loop — they're
        derived post-hoc from messages.jsonl, step_log.jsonl, event_log.jsonl
        which the EpisodeLogger already wrote.
        """
        try:
            from agent_modules.comm_eval import compute_comm_metrics
            self.comm_metrics = compute_comm_metrics(
                run_root=self.target_folder, num_agents=self.num_agents
            )
        except Exception as e:
            logging.warning("comm_eval failed: %s", e)
            self.comm_metrics = {}

        try:
            from agent_modules.coop_eval import compute_coop_metrics
            self.coop_metrics = compute_coop_metrics(
                run_root=self.target_folder, num_agents=self.num_agents
            )
        except Exception as e:
            logging.warning("coop_eval failed: %s", e)
            self.coop_metrics = {}

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
            "per_episode_returns":  [list(r) for r in self.per_episode_returns],
            "mean_return_per_agent": [
                (statistics.fmean(r) if r else 0.0)
                for r in self.per_episode_returns
            ],
            "std_return_per_agent": [
                (statistics.pstdev(r) if len(r) >= 2 else 0.0)
                for r in self.per_episode_returns
            ],
            "episode_lengths":      list(self.episode_lengths),
            "mean_episode_length":  (
                statistics.fmean(self.episode_lengths)
                if self.episode_lengths else 0.0
            ),
            "std_episode_length":   (
                statistics.pstdev(self.episode_lengths)
                if len(self.episode_lengths) >= 2 else 0.0
            ),
            "track_rewards_per_episode": [
                [dict(d) for d in agent_eps]
                for agent_eps in self.track_rewards_per_episode
            ],
            "mean_track_reward_per_agent": [
                {
                    track: (
                        statistics.fmean([d.get(track, 0.0) for d in agent_eps])
                        if agent_eps else 0.0
                    )
                    for track in TRACKS
                }
                for agent_eps in self.track_rewards_per_episode
            ],
            "std_track_reward_per_agent": [
                {
                    track: (
                        statistics.pstdev([d.get(track, 0.0) for d in agent_eps])
                        if len(agent_eps) >= 2 else 0.0
                    )
                    for track in TRACKS
                }
                for agent_eps in self.track_rewards_per_episode
            ],
            "milestones_per_episode": [
                [list(ms) for ms in agent_eps]
                for agent_eps in self.milestones_per_episode
            ],
            "milestone_count_per_episode": [
                [len(ms) for ms in agent_eps]
                for agent_eps in self.milestones_per_episode
            ],
            "mean_milestone_count_per_agent": [
                (
                    statistics.fmean([len(ms) for ms in agent_eps])
                    if agent_eps else 0.0
                )
                for agent_eps in self.milestones_per_episode
            ],
            "std_milestone_count_per_agent": [
                (
                    statistics.pstdev([len(ms) for ms in agent_eps])
                    if len(agent_eps) >= 2 else 0.0
                )
                for agent_eps in self.milestones_per_episode
            ],
            "comm_count_per_episode": [
                list(c) for c in self.comm_count_per_episode
            ],
            "mean_comm_count_per_agent": [
                (statistics.fmean(c) if c else 0.0)
                for c in self.comm_count_per_episode
            ],
            "std_comm_count_per_agent": [
                (statistics.pstdev(c) if len(c) >= 2 else 0.0)
                for c in self.comm_count_per_episode
            ],
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
            # Reward decomposition: parallel to reward_history, lets analysis
            # subtract comm rewards to isolate task-only return curves.
            "reward_history_decomposed": self.reward_history_decomposed,
            # Post-hoc-computed metrics (filled in by _run_posthoc_evaluators).
            "comm_metrics":         getattr(self, "comm_metrics", {}),
            "coop_metrics":         getattr(self, "coop_metrics", {}),
        }

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

        # Track order is the canonical chamber progression + comm at the
        # end so the bars read left-to-right in playthrough order.
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

        # Pad episodes to a common length with NaN so nanmean/nanstd
        # handle ragged eps correctly (some may be shorter if truncated).
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

        # Slice into per-episode series; smooth within each episode so the
        # smoothing kernel never crosses an episode boundary (which would
        # bleed late-ep N traffic into early ep N+1).
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

        # Smoothing window: clip to the shortest episode so convolution
        # produces at least one valid value per episode.
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
        # Y-axis ordering: walk TRACKS in canonical order so chambers
        # stack bottom-up. Each entry gets a row index; we also remember
        # which chamber each row belongs to for the background bands.
        y_labels = []
        y_chamber = []  # parallel: chamber name per row, for banding
        for track in TRACK_ORDER:
            for mid, _ in TRACKS[track]:
                y_labels.append(mid)
                y_chamber.append(track)
        y_index = {mid: i for i, mid in enumerate(y_labels)}

        # Chamber band colors. Lightly shaded horizontal stripes so the
        # reader can see at a glance which chamber a milestone lives in
        # without reading the y-tick label.
        chamber_band_colors = {
            "ch1_solo":     "#f4f0ff",
            "ch2_anvils":   "#fff4e6",
            "ch3_switches": "#e8f8ff",
            "ch4_combat":   "#ffeeee",
            "ch5_boss":     "#fff0c2",
            "communication":"#eaeaea",
        }

        # Agent marker colors. Use a small categorical palette that
        # stays readable when 3-6 agents overlap.
        cmap = plt.get_cmap("tab10")
        agent_colors = {i: cmap(i % 10) for i in range(self.num_agents)}

        # Figure size scales with the milestone count so labels stay
        # readable for the full 33-row layout (5 chambers + comm).
        fig_h = max(6.5, 0.32 * len(y_labels))
        fig, ax = plt.subplots(figsize=(13, fig_h))

        # Background chamber bands. Draw rectangles from x=0 to the
        # rightmost event x so they span the whole plot.
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

        # Scatter every milestone event. Larger dot + slight jitter on y
        # if multiple agents hit the same milestone at the same step, so
        # they don't fully overlap.
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
            # Tiny vertical jitter per agent so 3 agents on same (step, mid)
            # are visually distinct rather than stacked.
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

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Env step")
        ax.set_xlim(0, x_max * 1.08)
        ax.set_ylim(-0.7, len(y_labels) - 0.3)
        ax.set_title(
            "Five-Chambers Milestone Timeline (markers = fire events; "
            "rows grouped by chamber)"
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
        # Spell out the scope so this chart isn't mistaken for total return:
        # it sums only milestone-fire rewards. Per-block dig progress
        # (dig_stage_res) and proximity/Hebbian shaping go to a separate
        # chart, _plot_reward_decomposition().
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
        # Prefer full W if present (post-fix runs); fall back to top_3_pairs
        # lookup for legacy snapshots.
        W = snapshot.get("W")
        if W and 0 <= i_idx < len(W) and 0 <= j_idx < len(W[i_idx]):
            return float(W[i_idx][j_idx])
        for tp in snapshot.get("top_3_pairs", []):
            if tp["i"] == i_idx and tp["j"] == j_idx:
                return tp["w"]
        return 0.0

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

    def _summary_rl(self):
        if not (self.rl_updates or self.rl_token_opts):
            return []

        # Split per-agent action updates from centralized-critic updates so the
        # two losses (which are not comparable) are reported separately.
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

    # ─── Checkpoint restore ────────────────────────────────────────────

    @classmethod
    def restore_from_dict(
        cls,
        d: dict,
        path: str = "./run_metrics",
        run_paths=None,
    ) -> "CraftiumMetric":
        """Rebuild a CraftiumMetric from a dict (typically loaded from
        run_state.json on resume).

        Pass ``run_paths`` to land output under the consolidated
        ``runs/<run_id>/`` tree. The legacy ``path=`` argument is kept for
        back-compat tooling that constructs metrics outside the main loop;
        when ``run_paths`` is supplied, ``path`` is ignored.
        """
        num_agents = d["num_agents"]
        metric = cls(
            num_agents=num_agents,
            communication=d.get("communication", True),
            path=path,
            run_id=d.get("run_id"),
            run_paths=run_paths,
        )

        metric.timestep = d.get("timestep", 0)
        metric.cumulative_returns = [
            float(x) for x in d.get("cumulative_returns", [0.0] * num_agents)
        ]
        metric.episode_returns = [
            float(x) for x in d.get("episode_returns", [0.0] * num_agents)
        ]
        metric.per_episode_returns = [
            [float(x) for x in ep_list]
            for ep_list in d.get(
                "per_episode_returns", [[] for _ in range(num_agents)]
            )
        ]
        _tre = d.get("track_rewards_episode", {})
        metric.track_rewards_episode = {
            i: dict(_tre.get(str(i), {t: 0.0 for t in TRACKS}))
            for i in range(num_agents)
        }
        for i in range(num_agents):
            for t in TRACKS:
                metric.track_rewards_episode[i].setdefault(t, 0.0)
        metric.track_rewards_per_episode = [
            [dict(d_ep) for d_ep in agent_eps]
            for agent_eps in d.get(
                "track_rewards_per_episode", [[] for _ in range(num_agents)]
            )
        ]
        _ame = d.get("agent_milestones_episode", {})
        metric._agent_milestones_episode = {
            i: set(_ame.get(str(i), [])) for i in range(num_agents)
        }
        metric.milestones_per_episode = [
            [list(ms) for ms in agent_eps]
            for agent_eps in d.get(
                "milestones_per_episode", [[] for _ in range(num_agents)]
            )
        ]
        metric.comm_count_episode = [
            int(x) for x in d.get("comm_count_episode", [0] * num_agents)
        ]
        metric.comm_count_per_episode = [
            [int(x) for x in c] for c in d.get(
                "comm_count_per_episode", [[] for _ in range(num_agents)]
            )
        ]
        metric.episode_lengths = [
            int(x) for x in d.get("episode_lengths", [])
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
