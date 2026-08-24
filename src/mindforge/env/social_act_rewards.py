"""Reward tracker for the non-verbal social acts (Experiment 2, act-reward
symmetry suite).

Pays observation and imitation acts EXACTLY like communication is paid by
``CommunicationTracker``: the same base reward per valid act (0.5), the same
per-agent cap (50 rewarded acts/episode), the same rate limit (>= 2 steps
between rewarded acts of the same type), and the same per-chamber milestones
(4 acts in a chamber fire ``m_obs_chN`` / ``m_imit_chN`` once per agent, at
the communication track's reward values 10/10/20/10/10).

Mirrored validity semantics: a rate-limited act is invisible (no pay, no
chamber count); a RESCUED act (the model's target was self/"all"/garbage and
routing had to repair it) earns no base pay but still counts toward the
chamber milestone — identical to how bad-target speakers are treated for
messages.

Off unless ``--social-act-rewards`` is set — the default keeps every
existing suite's reward stream byte-identical.
"""

from collections import defaultdict

from env.communication_rewards import (
    BASE_MSG_CAP,
    BASE_MSG_REWARD,
    CHAMBER_BOUNDS,
    CHAMBER_COMM_THRESHOLDS,
    RATE_LIMIT_STEPS,
)

ACTS = ("obs", "imit")

# Same thresholds and reward values as the communication milestones, with
# per-act milestone ids: m_obs_ch1..5, m_imit_ch1..5.
CHAMBER_ACT_THRESHOLDS = {
    act: {ch: (thr, rw, f"m_{act}_{ch}")
          for ch, (thr, rw, _mid) in CHAMBER_COMM_THRESHOLDS.items()}
    for act in ACTS
}


class SocialActRewardTracker:
    def __init__(self, agent_ids, acts=ACTS):
        self.agent_ids = agent_ids
        self.acts = tuple(acts)
        # per (act, agent): rewarded-act count, last rewarded-act step
        self.total_valid = defaultdict(int)
        self.last_act_step = defaultdict(lambda: -999)
        # per (act, agent, chamber): act count toward the chamber milestone
        self.chamber_counts = defaultdict(int)
        self.fired_milestones = defaultdict(set)   # (act, agent) -> {mid}

    @staticmethod
    def _chamber_for(pos):
        if pos is None:
            return None
        for name, fn in CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def process_step(self, step, act_events, agent_positions):
        """Pay this step's observation/imitation acts.

        ``act_events``: iterable of ``(agent_id, act, rescued)`` where act is
        "obs" | "imit" and rescued marks a routing-repaired target (no base
        pay, chamber count still advances — mirroring bad-target speakers).

        Returns (rewards, milestones_fired) shaped exactly like
        ``CommunicationTracker.process_step``'s first two outputs.
        """
        rewards = defaultdict(float)
        milestones_fired = []
        for agent_id, act, rescued in act_events:
            if act not in self.acts:
                continue
            key = (act, agent_id)
            if step - self.last_act_step[key] < RATE_LIMIT_STEPS:
                continue                      # rate-limited: invisible
            self.last_act_step[key] = step

            if not rescued and self.total_valid[key] < BASE_MSG_CAP:
                rewards[agent_id] += BASE_MSG_REWARD
                self.total_valid[key] += 1

            chamber = self._chamber_for(agent_positions.get(agent_id))
            if chamber in CHAMBER_ACT_THRESHOLDS[act]:
                ckey = (act, agent_id, chamber)
                self.chamber_counts[ckey] += 1
                threshold, reward, mid = CHAMBER_ACT_THRESHOLDS[act][chamber]
                if (self.chamber_counts[ckey] >= threshold
                        and mid not in self.fired_milestones[key]):
                    rewards[agent_id] += reward
                    self.fired_milestones[key].add(mid)
                    milestones_fired.append((agent_id, mid, reward))
        return dict(rewards), milestones_fired
