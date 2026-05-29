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

        # Damage tracking from infos. Currently no producer in the live
        # code populates infos["damage_events"], but if/when one is added,
        # ``attacker`` may arrive as either an int (already an agent id),
        # the Python-side 'agent_N' string, or the Lua-side 'agentN'
        # string. Normalise to int so:
        #   (a) ch4_damage / ch5_damage dicts have a consistent key shape,
        #       which matters because _chamber_fairness reads them with
        #       `self.ch4_damage.get(a, 0.0) for a in self.agent_ids`
        #       where self.agent_ids is a list of ints — mismatched keys
        #       silently return 0 and inflate fairness to 1.0.
        #   (b) downstream pair-matrix conversion in _pair_to_matrix can
        #       use the same int keys it already uses for proximity /
        #       messages / joint_dig.
        # Events whose attacker can't be parsed are dropped — they would
        # have been silently miscredited under the old code path anyway.
        for dmg_event in infos.get("damage_events", []):
            target = dmg_event.get("target", "")
            attacker = self._to_int_id(dmg_event.get("attacker"))
            amount = dmg_event.get("amount", 0.0)
            if not isinstance(attacker, int) or attacker not in self.agent_ids:
                continue
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
        # Handle both 'agent_0' (Python side) and 'agent0' (Lua side player
        # names, no underscore). The previous `split('_')[-1]` parser only
        # worked for the underscored form — for 'agent0', split returned
        # ['agent0'], int('agent0') raised ValueError, and the function
        # silently returned the original string. Downstream code that
        # treated the result as an int key (milestone_log set membership,
        # ch4_damage / ch5_damage dict access, etc.) missed every Lua-
        # sourced contributor.
        s = str(agent_id).removeprefix("agent_").removeprefix("agent")
        try:
            return int(s)
        except ValueError:
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
        # milestone_log preserves contributor names in whatever shape the
        # source emitted — 'agent_0' for Python-fired (comm milestones),
        # 'agent0' for Lua-fired (m1..m_door1_open, m8..m13, etc.). Without
        # normalisation, the same agent's contributions split across two
        # dict keys and max-min computes wrong. Route through _to_int_id
        # so both shapes collapse to the same int id; also enforce that
        # every agent has an entry (otherwise an agent with zero firings
        # is invisible and min() ignores them, inflating fairness).
        per_agent = {a: 0 for a in self.agent_ids}
        for m in self.milestone_log:
            for c in m["contributors"]:
                aid = self._to_int_id(c)
                if isinstance(aid, int) and aid in per_agent:
                    per_agent[aid] += 1
        if not per_agent:
            return 0.0
        return max(per_agent.values()) - min(per_agent.values())

    # ── Per-chamber cooperation scoring ───────────────────────────────
    # Cooperative chambers contribute perf × fair to the overall score
    # ONLY if the team reached them. Ch1 is solo, so it's excluded.
    _COOPERATIVE_CHAMBERS = ("ch2", "ch3", "ch4", "ch5")

    # Milestone-prefix groups used to compute per-chamber performance /
    # fairness from the per-fire contributor lists in milestone_log.
    # NOTE on Ch2: the env exposes only 2 anvils (sword + chestplate; m8 + m11)
    # per the deliberate "RL tractability" simplification in anvil.lua. The
    # other m9/m10/m12/m13 IDs are defined in milestones.lua but never fire.
    _CH2_ANVIL_PREFIXES = ("m8_", "m11_")
    _CH3_PRESS_PREFIXES = ("m17_",)
    _CH3_DOOR_PREFIXES  = ("m18_",)
    _CH3_REGROUP_PREFIX = ("m19_",)

    def _milestone_count(self, prefixes):
        return sum(
            1 for m in self.milestone_log
            if any(m["milestone"].startswith(p) for p in prefixes)
        )

    def _milestone_contributor_counts(self, prefixes):
        """Per-agent count of contributions to milestones matching ``prefixes``.
        Returns a dict keyed by every agent id (0 for non-contributors) so the
        Gini reflects participation gaps, not just non-zero-only inequality."""
        counts = {a: 0 for a in self.agent_ids}
        for m in self.milestone_log:
            if not any(m["milestone"].startswith(p) for p in prefixes):
                continue
            for c in m["contributors"]:
                aid = self._to_int_id(c)
                if aid in counts:
                    counts[aid] += 1
        return counts

    def _chamber_performance(self, chamber):
        """Performance in [0, 1] — how much of the chamber's cooperative
        content the team actually completed."""
        if chamber == "ch2":
            # 2 anvils to break (m8 sword + m11 chestplate), once-each.
            return min(self._milestone_count(self._CH2_ANVIL_PREFIXES) / 2.0, 1.0)
        if chamber == "ch3":
            # Switch puzzle: 3 switches pressed (m17) + 3 cell doors opened
            # (m18, consequence of teammate presses) + 1 team regroup (m19).
            # Max contributions = 7.
            total = (self._milestone_count(self._CH3_PRESS_PREFIXES)
                     + self._milestone_count(self._CH3_DOOR_PREFIXES)
                     + self._milestone_count(self._CH3_REGROUP_PREFIX))
            return min(total / 7.0, 1.0)
        if chamber == "ch4":
            # 3 zombies × ~20 HP each = 60 total combat damage budget.
            return min(sum(self.ch4_damage.values()) / 60.0, 1.0)
        if chamber == "ch5":
            # Boss = 60 HP.
            return min(sum(self.ch5_damage.values()) / 60.0, 1.0)
        return 0.0

    def _chamber_fairness(self, chamber):
        """Fairness in [0, 1] = 1 − Gini of per-agent contributions.
        1.0 = perfectly equal across agents, 0.0 = one agent did everything.

        For Ch2 anvils and Ch3 switches: count milestone contributor entries.
        For Ch4 / Ch5 combat: per-agent damage values."""
        if chamber == "ch2":
            counts = self._milestone_contributor_counts(self._CH2_ANVIL_PREFIXES)
        elif chamber == "ch3":
            # Stick to switch presses — door-openings (m18) double-count the
            # same cooperation since they are CAUSED by another agent's press
            # in the rotational wiring (A→B→C→A).
            counts = self._milestone_contributor_counts(self._CH3_PRESS_PREFIXES)
        elif chamber == "ch4":
            counts = {a: self.ch4_damage.get(a, 0.0) for a in self.agent_ids}
        elif chamber == "ch5":
            counts = {a: self.ch5_damage.get(a, 0.0) for a in self.agent_ids}
        else:
            return 1.0
        return 1.0 - self._gini(counts)

    def _cooperation_breakdown(self):
        """Component breakdown of the cooperation_score, for interpretability.
        ``sum(component values) / 5`` equals cooperation_score."""
        out = {"comm_eff": self._comm_efficacy()}
        for chamber in self._COOPERATIVE_CHAMBERS:
            reached = chamber in self.chamber_entry_step
            if reached:
                perf = self._chamber_performance(chamber)
                fair = self._chamber_fairness(chamber)
                score = perf * fair
            else:
                perf = 0.0
                fair = 0.0
                score = 0.0
            out[chamber] = {
                "reached": reached,
                "performance": perf,
                "fairness": fair,
                "score": score,
            }
        return out

    def _cooperation_score(self):
        """Mean of (comm_eff, perf×fair per cooperative chamber).

        Five components, each in [0, 1]:
          1. comm_efficacy — fraction of multi-contributor milestones preceded
             by communication.
          2-5. Per cooperative chamber (Ch2..Ch5): performance × fairness.
               Unreached chambers contribute 0 — so a team that never engages
               cooperative content cannot inflate the score by virtue of "no
               data = perfect fairness", which was the old metric's bug.

        Ch1 is excluded: it is by design a solo-learning chamber with no
        cooperative mechanic to score.
        """
        components = [self._comm_efficacy()]
        for chamber in self._COOPERATIVE_CHAMBERS:
            if chamber in self.chamber_entry_step:
                perf = self._chamber_performance(chamber)
                fair = self._chamber_fairness(chamber)
                components.append(perf * fair)
            else:
                components.append(0.0)
        return sum(components) / len(components)

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
            # Per-component breakdown of cooperation_score (interpretability).
            # Each cooperative chamber reports reached/perf/fair/score so the
            # headline number can be traced back to "Ch3 was the weak link"
            # vs. "everyone skipped Ch4" etc.
            "cooperation_breakdown": self._cooperation_breakdown(),
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
