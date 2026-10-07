"""Direct messages, the board, and bond-gated reading.

Every agent may send one direct message (DM) and one board post per round.
What each agent *reads* is budgeted, so the prompt stays O(K + B) however many
agents there are:

    inbox     ≤ K DMs addressed to the agent (overflow trimmed by the policy)
    board     ≤ B posts from the last ``board_window`` rounds
    contacts  ≤ C names the agent is told about (never the whole roster)

The :class:`ReadPolicy` decides which messages survive the budgets. This is
where the Hebbian bonds act on behaviour (observation gating):

    recency   newest first — the base arm
    hebbian   W[i, author] · exp(−age / τ), with ``explore`` board slots kept
              for authors outside the reader's top bonds
    shuffled  as hebbian, but each round row i of W is randomly permuted
              (same magnitudes, wrong identities) — the random-gate control
    oracle    same-team authors first — the upper bound in multi-team worlds

While N − 1 ≤ B every post fits and gating cannot matter; the policies only
diverge once the board overflows, which is the scaling prediction under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

POST_KINDS = ("need", "offer", "status", "done", "none")


@dataclass
class Message:
    round: int
    sender: int
    text: str
    kind: str = "dm"            # "dm" | "post"
    receiver: Optional[int] = None
    post_kind: str = "status"
    msg_id: int = -1


@dataclass
class Inbox:
    agent: int
    dms: List[Message] = field(default_factory=list)
    board: List[Message] = field(default_factory=list)
    contacts: List[int] = field(default_factory=list)
    dropped_dms: int = 0
    unread_posts: int = 0


class ReadPolicy:
    name = "recency"

    def begin_round(self, round_num: int, W: Optional[np.ndarray]) -> None:
        self.round = round_num
        self.W = W

    def bond(self, reader: int, author: int) -> float:
        return 0.0

    def score(self, reader: int, msg: Message) -> float:
        return -float(self.round - msg.round)

    def rank(self, reader: int, msgs: Sequence[Message]) -> List[Message]:
        return sorted(msgs, key=lambda m: (-self.score(reader, m), -m.msg_id))

    def rank_contacts(self, reader: int, candidates: Sequence[int]) -> List[int]:
        return list(candidates)


class RecencyPolicy(ReadPolicy):
    name = "recency"


class HebbianPolicy(ReadPolicy):
    name = "hebbian"

    def __init__(self, tau: float = 3.0):
        self.tau = tau

    def bond(self, reader: int, author: int) -> float:
        if self.W is None:
            return 0.0
        return float(self.W[reader, author])

    def score(self, reader: int, msg: Message) -> float:
        age = max(0, self.round - msg.round)
        return self.bond(reader, msg.sender) * math.exp(-age / self.tau)

    def rank_contacts(self, reader: int, candidates: Sequence[int]) -> List[int]:
        return sorted(candidates, key=lambda j: (-self.bond(reader, j), j))


class ShuffledPolicy(HebbianPolicy):
    """Hebbian scoring over a per-round random permutation of each W row."""

    name = "shuffled"

    def __init__(self, tau: float = 3.0, seed: int = 0):
        super().__init__(tau)
        self.rng = np.random.default_rng(seed)
        self._perm: Dict[int, np.ndarray] = {}

    def begin_round(self, round_num: int, W: Optional[np.ndarray]) -> None:
        super().begin_round(round_num, W)
        self._perm = {}

    def _row(self, reader: int) -> np.ndarray:
        if reader not in self._perm:
            n = self.W.shape[0]
            others = np.array([j for j in range(n) if j != reader])
            perm = np.arange(n)
            perm[others] = self.rng.permutation(others)
            self._perm[reader] = perm
        return self._perm[reader]

    def bond(self, reader: int, author: int) -> float:
        if self.W is None:
            return 0.0
        return float(self.W[reader, self._row(reader)[author]])


class OraclePolicy(ReadPolicy):
    name = "oracle"

    def __init__(self, same_team: np.ndarray, tau: float = 3.0):
        self.same = same_team
        self.tau = tau

    def bond(self, reader: int, author: int) -> float:
        return float(self.same[reader, author])

    def score(self, reader: int, msg: Message) -> float:
        age = max(0, self.round - msg.round)
        return (1.0 + self.bond(reader, msg.sender)) * math.exp(-age / self.tau)

    def rank_contacts(self, reader: int, candidates: Sequence[int]) -> List[int]:
        return sorted(candidates, key=lambda j: (-self.bond(reader, j), j))


def make_policy(name: str, *, seed: int = 0, tau: float = 3.0,
                same_team: Optional[np.ndarray] = None) -> ReadPolicy:
    if name == "recency":
        return RecencyPolicy()
    if name == "hebbian":
        return HebbianPolicy(tau)
    if name == "shuffled":
        return ShuffledPolicy(tau, seed)
    if name == "oracle":
        if same_team is None:
            raise ValueError("oracle policy needs the true team matrix")
        return OraclePolicy(same_team, tau)
    raise ValueError(f"unknown read policy {name!r}")


class CommRouter:
    def __init__(self, n_agents: int, policy: ReadPolicy, *, board_budget: int = 5,
                 inbox_budget: int = 4, contacts_budget: int = 6, explore_slots: int = 2,
                 board_window: int = 3, nearest: int = 2, seed: int = 0):
        self.n = n_agents
        self.policy = policy
        self.B = board_budget
        self.K = inbox_budget
        self.C = contacts_budget
        self.explore = min(explore_slots, board_budget)
        self.window = board_window
        self.nearest = nearest
        self.rng = np.random.default_rng(seed + 1)
        self.board: List[Message] = []
        self.pending_dms: List[Message] = []
        self._next_id = 0
        self.history: List[Message] = []

    # ── sending ────────────────────────────────────────────────────────────
    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def post(self, round_num: int, author: int, text: str, post_kind: str = "status") -> Message:
        m = Message(round_num, author, text, "post", None,
                    post_kind if post_kind in POST_KINDS else "status", self._id())
        self.board.append(m)
        self.history.append(m)
        return m

    def send_dm(self, round_num: int, sender: int, receiver: int, text: str) -> Optional[Message]:
        if not (0 <= receiver < self.n) or receiver == sender:
            return None
        m = Message(round_num, sender, text, "dm", receiver, "none", self._id())
        self.pending_dms.append(m)
        self.history.append(m)
        return m

    # ── reading ────────────────────────────────────────────────────────────
    def route(self, round_num: int, W: Optional[np.ndarray] = None,
              positions: Optional[Sequence[Optional[Sequence[float]]]] = None,
              readers: Optional[Sequence[int]] = None) -> Dict[int, Inbox]:
        """Deliver this round's reading to every reader (default: everyone)."""
        self.policy.begin_round(round_num, W)
        live = [m for m in self.board if round_num - m.round <= self.window]
        self.board = live
        dms_for: Dict[int, List[Message]] = {}
        for m in self.pending_dms:
            dms_for.setdefault(m.receiver, []).append(m)
        self.pending_dms = []

        out: Dict[int, Inbox] = {}
        for i in (readers if readers is not None else range(self.n)):
            inbox = Inbox(agent=i)
            mine = dms_for.get(i, [])
            ranked = self.policy.rank(i, mine)
            inbox.dms = ranked[: self.K]
            inbox.dropped_dms = max(0, len(mine) - self.K)
            inbox.board = self._board_for(i, live)
            inbox.unread_posts = sum(1 for m in live if m.sender != i) - len(inbox.board)
            inbox.contacts = self._contacts(i, inbox, positions)
            out[i] = inbox
        return out

    def _board_for(self, reader: int, live: List[Message]) -> List[Message]:
        cands = [m for m in live if m.sender != reader]
        if len(cands) <= self.B:
            return self.policy.rank(reader, cands)
        ranked = self.policy.rank(reader, cands)
        if self.explore <= 0 or isinstance(self.policy, RecencyPolicy):
            return ranked[: self.B]
        keep = ranked[: self.B - self.explore]
        top_authors = set(self.policy.rank_contacts(
            reader, [j for j in range(self.n) if j != reader])[: self.C])
        rest = [m for m in ranked[self.B - self.explore:] if m.sender not in top_authors] \
            or ranked[self.B - self.explore:]
        picks = self.rng.choice(len(rest), size=min(self.explore, len(rest)), replace=False)
        return keep + [rest[k] for k in sorted(picks)]

    def _contacts(self, i: int, inbox: Inbox,
                  positions: Optional[Sequence[Optional[Sequence[float]]]]) -> List[int]:
        heard = []
        for m in inbox.dms + inbox.board:
            if m.sender not in heard:
                heard.append(m.sender)
        near: List[int] = []
        if positions is not None and i < len(positions) and positions[i] is not None:
            pi = np.asarray(positions[i], dtype=float)[:2]
            dists = []
            for j, p in enumerate(positions):
                if j != i and p is not None:
                    dists.append((float(np.abs(np.asarray(p, dtype=float)[:2] - pi).sum()), j))
            near = [j for _, j in sorted(dists)]
        others = [j for j in range(self.n) if j != i]
        bonded = self.policy.rank_contacts(i, others) if not isinstance(
            self.policy, RecencyPolicy) else []
        if bonded:
            order = bonded[: max(0, self.C - self.nearest)] + near[: self.nearest] + heard
        else:
            order = near[: self.C] + heard
        seen, out = set(), []
        for j in order:
            if j not in seen and j != i:
                seen.add(j)
                out.append(j)
            if len(out) >= self.C:
                break
        return out
