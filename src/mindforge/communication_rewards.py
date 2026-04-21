"""Communication reward tracker for the Five Chambers environment.

Tracks chat messages per agent, applies validity rules, and emits rewards for
both the per-message base (Tier 1) and per-chamber communication milestones (Tier 2).
"""

from collections import defaultdict

CHAMBER_BOUNDS = {
    "ch1": lambda p: 0 <= p[2] <= 11,
    "ch2": lambda p: 13 <= p[2] <= 22,
    "ch3": lambda p: 24 <= p[2] <= 38,
    "ch4": lambda p: 40 <= p[2] <= 46,
    "ch5": lambda p: 48 <= p[2] <= 52,
}

BASE_MSG_REWARD = 2.0
BASE_MSG_CAP = 10  # Max rewarded messages per agent per episode
MIN_MSG_LEN = 5
RATE_LIMIT_STEPS = 2  # Min steps between valid messages per agent

CHAMBER_COMM_THRESHOLDS = {
    "ch2": (3, 20.0, "m_comm_ch2"),
    "ch3": (2, 30.0, "m_comm_ch3"),
    "ch4": (2, 15.0, "m_comm_ch4"),
    "ch5": (2, 20.0, "m_comm_ch5"),
}


class CommunicationTracker:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
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

    def process_step(self, step, agent_messages, agent_positions):
        """Returns (rewards, milestones_fired, valid_speakers) for this step.

        - rewards: {agent_id: extra_reward}
        - milestones_fired: list of (agent_id, milestone_id, reward)
        - valid_speakers: set of agent_ids whose message passed validity checks
        """
        rewards = defaultdict(float)
        milestones_fired = []
        valid_speakers = set()

        for agent_id, message in agent_messages.items():
            if not self._is_valid(agent_id, message, step):
                continue

            valid_speakers.add(agent_id)

            if self.total_valid_msgs[agent_id] < BASE_MSG_CAP:
                rewards[agent_id] += BASE_MSG_REWARD
                self.total_valid_msgs[agent_id] += 1

            chamber = self._chamber_for(agent_positions.get(agent_id))
            if chamber in CHAMBER_COMM_THRESHOLDS:
                self.chamber_msg_counts[agent_id][chamber] += 1
                threshold, reward, mid = CHAMBER_COMM_THRESHOLDS[chamber]
                if (self.chamber_msg_counts[agent_id][chamber] >= threshold
                        and mid not in self.fired_milestones[agent_id]):
                    rewards[agent_id] += reward
                    self.fired_milestones[agent_id].add(mid)
                    milestones_fired.append((agent_id, mid, reward))

            self.last_msg[agent_id] = message
            self.last_msg_step[agent_id] = step

        return dict(rewards), milestones_fired, valid_speakers
