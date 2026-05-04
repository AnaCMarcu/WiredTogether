"""Cooperation metric observer for the Five Chambers environment.

Observes the step loop (no reward effect) and emits per-episode and
per-chamber cooperation statistics for post-hoc analysis.
"""

from collections import defaultdict
import math

import numpy as np

CHAMBER_BOUNDS = {
    "ch1": lambda p: 0 <= p[2] <= 15,
    "ch2": lambda p: 17 <= p[2] <= 30,
    "ch3": lambda p: 32 <= p[2] <= 50,
    "ch4": lambda p: 52 <= p[2] <= 62,
    "ch5": lambda p: 64 <= p[2] <= 72,
}


class CooperationMetric:
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.reset()

    def reset(self):
        self.proximity_events = 0
        self.co_action_events = 0
        self.joint_dig_events = 0
        self.messages_per_agent = defaultdict(int)
        self.milestone_log = []
        self.chamber_entry_step = {}
        self.ch4_damage = defaultdict(float)
        self.ch5_damage = defaultdict(float)
        self.recent_messages = []  # (step, agent_id, message)

        # Per-pair interaction tensor: I[i][j][k] for k ∈
        # {messages, joint_dig, proximity, joint_kill, ch5_damage_overlap}.
        # `messages` is asymmetric (sender→receiver); the others are symmetric.
        # Stored as nested defaultdicts and aggregated at episode end into
        # plain N×N lists for serialisation.
        self.pair_messages = defaultdict(lambda: defaultdict(int))
        self.pair_joint_dig = defaultdict(lambda: defaultdict(int))
        self.pair_proximity = defaultdict(lambda: defaultdict(int))
        self.pair_joint_kill = defaultdict(lambda: defaultdict(int))
        self.pair_boss_overlap = defaultdict(lambda: defaultdict(int))

        # Per-chamber dwell (tick count) + action histograms per agent.
        # dwell[i][chamber] = int, action_hist[i][chamber][action] = int.
        self.dwell_steps = defaultdict(lambda: defaultdict(int))
        self.action_hist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Sequence of (step, killer, target) for joint-kill detection.
        # A pair is credited a joint_kill when they damaged the same target
        # within the last 5 steps before its death.
        self._damage_log = []     # (step, attacker, target, amount)
        self._kill_log = []       # (step, killer, target)

    def _chamber_for(self, pos):
        if pos is None:
            return None
        for name, fn in CHAMBER_BOUNDS.items():
            if fn(pos):
                return name
        return None

    def observe_step(self, step, positions, actions, messages, task_rewards, infos=None):
        """Observe one environment step. positions/actions/messages are {agent_id: value} dicts."""
        if infos is None:
            infos = {}

        # Proximity events (pairs within 4 blocks)
        valid_pos = {i: p for i, p in positions.items() if p is not None}
        ids = list(valid_pos.keys())
        for k in range(len(ids)):
            for l in range(k + 1, len(ids)):
                i, j = ids[k], ids[l]
                dist = np.linalg.norm(np.array(valid_pos[i]) - np.array(valid_pos[j]))
                if dist < 4.0:
                    self.proximity_events += 1
                    # Symmetric pair count
                    self.pair_proximity[i][j] += 1
                    self.pair_proximity[j][i] += 1

        # Co-action events (same action by 2+ agents)
        action_counts = defaultdict(int)
        for a in actions.values():
            if a:
                action_counts[a] += 1
        if any(c >= 2 for c in action_counts.values()):
            self.co_action_events += 1

        # Joint dig events (2+ agents both digging within 3 blocks)
        digging = [i for i, a in actions.items() if a == "Dig" and i in valid_pos]
        if len(digging) >= 2:
            for k in range(len(digging)):
                for l in range(k + 1, len(digging)):
                    i, j = digging[k], digging[l]
                    dist = np.linalg.norm(np.array(valid_pos[i]) - np.array(valid_pos[j]))
                    if dist < 3.0:
                        self.joint_dig_events += 1
                        self.pair_joint_dig[i][j] += 1
                        self.pair_joint_dig[j][i] += 1

        # Per-chamber dwell + action histogram per agent.
        for i in self.agent_ids:
            pos = positions.get(i)
            chamber = self._chamber_for(pos)
            if chamber:
                self.dwell_steps[i][chamber] += 1
                act = actions.get(i)
                if act:
                    self.action_hist[i][chamber][act] += 1

        # Message tracking + rolling buffer for comm_before_coop
        for agent_id, msg in messages.items():
            if msg and len(msg.strip()) >= 5:
                self.messages_per_agent[agent_id] += 1
                self.recent_messages.append((step, agent_id, msg))
        self.recent_messages = [
            (s, a, m) for (s, a, m) in self.recent_messages if step - s <= 10
        ]

        # Chamber entry tracking
        for i in self.agent_ids:
            pos = positions.get(i)
            chamber = self._chamber_for(pos)
            if chamber and chamber not in self.chamber_entry_step:
                self.chamber_entry_step[chamber] = step

        # Damage tracking from infos
        for dmg_event in infos.get("damage_events", []):
            target = dmg_event.get("target", "")
            attacker = dmg_event.get("attacker")
            amount = dmg_event.get("amount", 0.0)
            if target == "ch4_zombie":
                self.ch4_damage[attacker] += amount
            elif target == "boss":
                self.ch5_damage[attacker] += amount
            # Per-target damage log: feeds joint_kill detection at episode end.
            self._damage_log.append({
                "step": step, "attacker": attacker,
                "target": target, "amount": float(amount),
            })

        # Per-pair message matrix: extra per-message metadata can be passed
        # via infos["routed_messages"] = [{sender, receiver}, ...] from the
        # main loop (already routed by Hebbian/random fallback). Falls back
        # to "all" broadcast when not provided (legacy behaviour).
        for rm in infos.get("routed_messages", []):
            si = self._to_int_id(rm.get("sender", -1))
            ri = self._to_int_id(rm.get("receiver", -1))
            if si in self.agent_ids and ri in self.agent_ids and si != ri:
                self.pair_messages[si][ri] += 1

    def observe_kill(self, step: int, killer, target: str):
        """Called when a target dies; credits joint kills to recent attackers."""
        ki = self._to_int_id(killer) if killer is not None else None
        # Find all attackers who damaged this target in the last 5 steps.
        recent_attackers = set()
        for d in self._damage_log:
            if step - d["step"] <= 5 and d["target"] == target and d["attacker"] is not None:
                aid = self._to_int_id(d["attacker"])
                if aid in self.agent_ids:
                    recent_attackers.add(aid)
        # Pair every recent attacker with every other for the joint-kill matrix
        # (and with the killer specifically). Symmetric.
        atk_list = sorted(recent_attackers)
        for k_i in range(len(atk_list)):
            for k_j in range(k_i + 1, len(atk_list)):
                a, b = atk_list[k_i], atk_list[k_j]
                self.pair_joint_kill[a][b] += 1
                self.pair_joint_kill[b][a] += 1
        # Boss-specific pair overlap (single boss per episode but co-damage
        # often spans many ticks).
        if target == "boss":
            for k_i in range(len(atk_list)):
                for k_j in range(k_i + 1, len(atk_list)):
                    a, b = atk_list[k_i], atk_list[k_j]
                    self.pair_boss_overlap[a][b] += 1
                    self.pair_boss_overlap[b][a] += 1
        self._kill_log.append({"step": step, "killer": ki, "target": target})

    @staticmethod
    def _to_int_id(agent_id):
        if isinstance(agent_id, int):
            return agent_id
        try:
            return int(str(agent_id).split("_")[-1])
        except (ValueError, IndexError):
            return agent_id

    def observe_milestone(self, step, milestone_id, contributors):
        """Record a milestone firing event with cooperation context."""
        int_contributors = {self._to_int_id(c) for c in contributors}
        recent = [m for (s, a, m) in self.recent_messages if a in int_contributors]
        self.milestone_log.append({
            "step": step,
            "milestone": milestone_id,
            "contributors": list(contributors),
            "contributor_count": len(contributors),
            "contribution_entropy": self._entropy(contributors),
            "comm_before_coop": len(recent) > 0,
        })

    @staticmethod
    def _entropy(contributors):
        if not contributors:
            return 0.0
        counts = defaultdict(int)
        for c in contributors:
            counts[c] += 1
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)

    @staticmethod
    def _gini(value_dict):
        values = sorted(value_dict.values())
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(values))
        return (2 * cum) / (n * sum(values)) - (n + 1) / n

    def _comm_efficacy(self):
        multi = [m for m in self.milestone_log if m["contributor_count"] >= 2]
        if not multi:
            return 0.0
        return sum(1 for m in multi if m["comm_before_coop"]) / len(multi)

    def _carry_imbalance(self):
        per_agent = defaultdict(int)
        for m in self.milestone_log:
            for c in m["contributors"]:
                per_agent[c] += 1
        if not per_agent:
            return 0.0
        return max(per_agent.values()) - min(per_agent.values())

    def _cooperation_score(self):
        joint_dig_norm = min(self.joint_dig_events / 50.0, 1.0)
        proximity_norm = min(self.proximity_events / 300.0, 1.0)
        comm_eff = self._comm_efficacy()
        ch5_fairness = 1.0 - self._gini(self.ch5_damage)
        balance = 1.0 - min(self._carry_imbalance() / 10.0, 1.0)
        return (0.2 * joint_dig_norm + 0.2 * proximity_norm +
                0.2 * comm_eff + 0.2 * ch5_fairness + 0.2 * balance)

    def _pair_to_matrix(self, nested) -> list:
        """Convert defaultdict[i][j] → N×N list (zero-filled, symmetric handled
        upstream). Uses the agent_ids list as the canonical ordering."""
        n = len(self.agent_ids)
        out = [[0 for _ in range(n)] for _ in range(n)]
        for i, ai in enumerate(self.agent_ids):
            row = nested.get(ai) if hasattr(nested, "get") else nested[ai]
            if not row:
                continue
            for j, aj in enumerate(self.agent_ids):
                out[i][j] = int(row.get(aj, 0)) if hasattr(row, "get") else int(row[aj])
        return out

    def episode_summary(self, final_step, hebbian_weights=None) -> dict:
        return {
            "final_step": final_step,
            "proximity_events": self.proximity_events,
            "co_action_events": self.co_action_events,
            "joint_dig_events": self.joint_dig_events,
            "messages_per_agent": dict(self.messages_per_agent),
            "chamber_entry_steps": dict(self.chamber_entry_step),
            "ch4_damage_gini": self._gini(self.ch4_damage),
            "ch5_damage_gini": self._gini(self.ch5_damage),
            "ch4_damage_per_agent": dict(self.ch4_damage),
            "ch5_damage_per_agent": dict(self.ch5_damage),
            "milestone_log": self.milestone_log,
            "communication_efficacy": self._comm_efficacy(),
            "carry_imbalance": self._carry_imbalance(),
            "cooperation_score": self._cooperation_score(),
            # Per-pair interaction tensor — five N×N planes covering the
            # cooperative mechanics in each chamber.
            "pair_interaction": {
                "messages":           self._pair_to_matrix(self.pair_messages),
                "joint_dig":          self._pair_to_matrix(self.pair_joint_dig),
                "proximity":          self._pair_to_matrix(self.pair_proximity),
                "joint_kill":         self._pair_to_matrix(self.pair_joint_kill),
                "ch5_damage_overlap": self._pair_to_matrix(self.pair_boss_overlap),
            },
            # Per-chamber dwell and action histograms (per-agent).
            "dwell_steps":  {str(i): dict(self.dwell_steps[i]) for i in self.agent_ids},
            "action_hist":  {str(i): {ch: dict(hh) for ch, hh in self.action_hist[i].items()}
                             for i in self.agent_ids},
            "hebbian_W": hebbian_weights,
        }
