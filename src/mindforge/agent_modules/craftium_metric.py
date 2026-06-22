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
    "m_door1_open":            "ch1_solo",
    "m8_anvil_A1":             "ch2_anvils",
    "m9_anvil_B1":             "ch2_anvils",
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
        ("m_door1_open", 50.0),
    ],
    "ch2_anvils": [
        ("m8_anvil_A1",  40.0), ("m9_anvil_B1", 40.0),
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

    current_line = None
    others = []
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

        is_current = (track == current_track) or (
            current_track is None and current_line is None and remaining
        )
        if is_current:
            parts = [f"  {label} (YOU ARE HERE):"]
            if you_done:
                parts.append(f"[you done] {', '.join(you_done)}")
            if team_only:
                parts.append(f"[team done, you didn't fire] {', '.join(team_only)}")
            if remaining:
                parts.append(f"[OPEN] {', '.join(remaining)}")
            else:
                parts.append("[chamber complete]")
            current_line = " ".join(parts)
        else:
            done_ct = len(all_mids) - len(remaining)
            if not remaining:
                status = "complete"
            elif done_ct == 0:
                status = "not started"
            else:
                status = f"{done_ct}/{len(all_mids)} done"
            others.append(f"{label}: {status}")

    block = current_line or ""
    if others:
        sep = "\n  " if block else "  "
        block += f"{sep}Other chambers: " + "; ".join(others)
    return block

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

from agent_modules._metric_plots import _PlotsMixin
from agent_modules._metric_summary import _SummaryMixin


class CraftiumMetric(_PlotsMixin, _SummaryMixin):
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
        self.episode_returns = [0.0] * num_agents
        self.per_episode_returns = [[] for _ in range(num_agents)]
        self.reward_history = [[] for _ in range(num_agents)]
        self.reward_history_decomposed = [[] for _ in range(num_agents)]

        # Milestones
        self.milestone_events = []                  # flat log: (milestone, contributor) pairs
        self._agent_milestones = {}                 # agent_name -> set of milestone ids
        self.first_milestone_step = {}              # mid -> first global step (any agent)
        self.anvil_coop_events = []
        self.track_rewards = {                      # per-agent, per-track reward sum
            i: {track: 0.0 for track in TRACKS}
            for i in range(num_agents)
        }
        self.track_rewards_episode = {
            i: {track: 0.0 for track in TRACKS}
            for i in range(num_agents)
        }
        self.track_rewards_per_episode = [[] for _ in range(num_agents)]
        # agent_id -> set of milestone ids reached THIS episode (clears each ep)
        self._agent_milestones_episode = {i: set() for i in range(num_agents)}
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

        self.idle_force_counts = {}
        self.action_recovered_counts = {}

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
                            five-chambers milestone rewards (m1..m28) +
                            drained death / would-die penalties (−50 / −10)
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

    def record_anvil_coop_event(self, ev: dict):
        """Record an anvil-coop diagnostic event from poll_anvil_coop_events().

        ev = {"step": int (lua tick), "anvil": str, "row": str,
              "n_active": int, "active": [str, ...]}

        NO reward is attached — this is purely diagnostic. The event
        is stored with the Python env step (self.timestep) for X-axis
        consistency with other metrics, plus the raw Lua step from the
        source dict so post-hoc analysis can correlate with anvil HP
        evolution at Lua-tick granularity.
        """
        record = {
            "step":     self.timestep,
            "lua_step": ev.get("step", 0),
            "anvil":    ev.get("anvil", ""),
            "row":      ev.get("row", ""),
            "n_active": int(ev.get("n_active", 0)),
            "active":   list(ev.get("active", [])),
        }
        self.anvil_coop_events.append(record)

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
            "idle_force_counts":    self.idle_force_counts,
            "action_recovered_counts": self.action_recovered_counts,
            "steps_to_milestone":   self.steps_to_milestone_table(),
            "milestones_per_agent": self.milestones_per_agent(),
            "milestone_events":     self.milestone_events,
            "anvil_coop_events":    list(self.anvil_coop_events),
            "anvil_coop_attempts":  len(self.anvil_coop_events),
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
            "reward_history_decomposed": self.reward_history_decomposed,
            # Post-hoc-computed metrics (filled in by _run_posthoc_evaluators).
            "comm_metrics":         getattr(self, "comm_metrics", {}),
            "coop_metrics":         getattr(self, "coop_metrics", {}),
        }

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
