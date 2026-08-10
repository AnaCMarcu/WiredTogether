"""Communication reward tracker for the Five Chambers environment.

Tracks chat messages per agent, applies validity rules, and emits rewards for
both the per-message base (Tier 1) and per-chamber communication milestones (Tier 2).
"""

from collections import defaultdict

CHAMBER_BOUNDS = {
    "ch1": lambda p: 0 <= p[2] <= 15,
    "ch2": lambda p: 17 <= p[2] <= 30,
    "ch3": lambda p: 32 <= p[2] <= 50,
    "ch4": lambda p: 52 <= p[2] <= 62,
    "ch5": lambda p: 64 <= p[2] <= 72,
}

BASE_MSG_REWARD = 0.5
BASE_MSG_CAP = 50  # Max rewarded messages per agent per episode (so 25 reward total).
MIN_MSG_LEN = 5
RATE_LIMIT_STEPS = 2  # Min steps between valid messages per agent

CHAMBER_COMM_THRESHOLDS = {
    "ch1": (4, 10.0, "m_comm_ch1"),
    "ch2": (4, 10.0, "m_comm_ch2"),
    "ch3": (4, 20.0, "m_comm_ch3"),
    "ch4": (4, 10.0, "m_comm_ch4"),
    "ch5": (4, 10.0, "m_comm_ch5"),
}


class CommunicationTracker:
    def __init__(self, agent_ids, reward_scale: float = 1.0):
        """``reward_scale`` multiplies every comm PAYOUT (base + chamber
        milestones). At 0.0 (Experiment 2 noreward suite) messages still
        route, validity/spam rules still apply, and comm milestones still
        FIRE as recorded events — they just pay nothing, so talking cannot
        manufacture bondable reward or trip the milestone-success banner."""
        self.agent_ids = agent_ids
        self.reward_scale = float(reward_scale)
        self.total_valid_msgs = defaultdict(int)
        self.chamber_msg_counts = defaultdict(lambda: defaultdict(int))
        self.last_msg = defaultdict(str)
        self.last_msg_step = defaultdict(lambda: -999)
        self.fired_milestones = defaultdict(set)

    def _chamber_for(self, pos):
        if pos is None:
            return None
        for name, fn in CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def _is_valid(self, agent_id, message, step):
        if not message or len(message.strip()) < MIN_MSG_LEN:
            return False
        if message == self.last_msg[agent_id]:
            return False
        if step - self.last_msg_step[agent_id] < RATE_LIMIT_STEPS:
            return False
        return True

    def process_step(self, step, agent_messages, agent_positions,
                     bad_target_speakers=None):
        """Returns (rewards, milestones_fired, valid_speakers) for this step.

        - rewards: {agent_id: extra_reward}
        - milestones_fired: list of (agent_id, milestone_id, reward)
        - valid_speakers: set of agent_ids whose message passed validity checks
        - bad_target_speakers: set of agent_ids whose model output was a self-
          target / "all" / unparseable. Routing rescues these so the receiver
          still gets the message, but the sender does NOT earn the base
          message reward — otherwise self-talking is positively reinforced.
          They still count toward chamber-comm milestones, since the message
          itself reached a real teammate via fallback routing.
        """
        bad_target_speakers = bad_target_speakers or set()
        rewards = defaultdict(float)
        milestones_fired = []
        valid_speakers = set()

        for agent_id, message in agent_messages.items():
            if not self._is_valid(agent_id, message, step):
                continue

            valid_speakers.add(agent_id)

            if (agent_id not in bad_target_speakers
                    and self.total_valid_msgs[agent_id] < BASE_MSG_CAP):
                rewards[agent_id] += BASE_MSG_REWARD * self.reward_scale
                self.total_valid_msgs[agent_id] += 1

            chamber = self._chamber_for(agent_positions.get(agent_id))
            if chamber in CHAMBER_COMM_THRESHOLDS:
                self.chamber_msg_counts[agent_id][chamber] += 1
                threshold, reward, mid = CHAMBER_COMM_THRESHOLDS[chamber]
                if (self.chamber_msg_counts[agent_id][chamber] >= threshold
                        and mid not in self.fired_milestones[agent_id]):
                    # Milestone EVENT always fires (metrics stay comparable);
                    # only the payout is scaled.
                    scaled = reward * self.reward_scale
                    rewards[agent_id] += scaled
                    self.fired_milestones[agent_id].add(mid)
                    milestones_fired.append((agent_id, mid, scaled))

            self.last_msg[agent_id] = message
            self.last_msg_step[agent_id] = step

        return dict(rewards), milestones_fired, valid_speakers
