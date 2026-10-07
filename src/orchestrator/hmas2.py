"""HMAS-2 adapted to WIRE (variant "hmas2"): the hard-orchestrator baseline.

Source: Chen, Arkin, Zhang, Roy, Fan, "Scalable Multi-Robot Collaboration
with Large Language Models: Centralized or Decentralized Systems?", ICRA
2024 (arXiv 2309.15943); reference code github.com/yongchao98/
multi-agent-framework. The protocol, the loop bounds and the prompt
sentences are ported from that code (BoxNet2: env2-box-arrange.py,
env2_create.py, prompt_env2.py), not rewritten. Every step:

1. The central planner gets the task description, the state-action history
   (``_w_only_state_action_history``, the paper's chosen setting), the current
   state and each agent's possible actions, and outputs a JSON plan
   ("Include an agent only if it has a task next").
2. A rule-based syntactic check re-prompts with the error
   (``with_action_syntactic_check_func``, <= 6 re-prompts).
3. <= 3 rounds: every agent in the plan checks its own assignment and answers
   'I Agree' or an objection; objections go back to the planner as a
   multi-turn follow-up and the revised plan is checked again.
4. The plan is executed.

Documented adaptations (stated with the results):
- Granularity: an assignment is a subtask ("task" text) that the MindForge
  worker executes with its own action loop; the worker's curriculum task is
  PINNED to it (CustomAgent ``orchestrator_pinned_task``). Assignments persist
  until the planner changes them, so "has a task next" = its task changes or
  it needs a message.
- Selective information: the planner sees the whole team, as in HMAS-2. A
  worker sees only its own state, its own assignment and the planner's
  message to it — the original local prompt's "current states and possible
  actions of all other agents", global state and joint plan are removed, so
  teammate information reaches a worker only through an explicit message.
- Hub-and-spoke reports: agents' per-step messages go to the planner
  (orchestrator/hub.py); there is no agent-to-agent channel.
- Syntactic failure after the last re-prompt keeps the valid entries and the
  previous assignments (HMAS-2 aborts the trial instead).
- The history token budget applies to the history alone (WIRE's base prompt
  is larger than BoxNet's whole budget).
- Budget runs: planner messages are charged to the recipient, objections to
  the objector; an agent at 0 is not checked and gets no messages. At budget
  0 the protocol reduces to CMAS with history.

Heavy imports (environment, craftium metric, the curriculum validator) are
lazy or injected, so this module stays light-importable for tests.
"""

from __future__ import annotations

import json
import logging as _stdlog
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from orchestrator.config import OrchestratorConfig
from orchestrator.core import (
    _normalize_agent,
    collect_env_state,
    parse_orchestrator_json,
)
from orchestrator.hub import HubReports, clip, sanitize_hub_text
from orchestrator.villager import (
    build_chamber_facts,
    known_milestone_ids,
)

logger = _stdlog.getLogger(__name__)

# ── Verbatim strings from the HMAS-2 reference code ──────────────────────

#: message_construct_func (prompt_env*.py): the system turn of every call.
SYSTEM_PROMPT = "You are a helpful assistant."
#: env2-box-arrange.py, HMAS-2 branch: appended after the objections. Only
#: the JSON example differs (ours is the WIRE plan format).
FEEDBACK_REPROMPT = (
    "\nThis is the feedback from local agents. If you find some errors in "
    "your previous plan, try to modify it. Otherwise, output the same plan "
    "as before. The output should have the same json format {example}, as "
    "above. Do not explain, just directly output json directory. Your "
    "response:"
)
#: env2_create.py, with_action_syntactic_check_func: per bad entry, then the
#: replan suffix (the original's "ouput" typo kept).
SYNTAX_BAD_ENTRY = "Your assigned task for {agent} is not in the doable action list; "
SYNTAX_REPLAN = "Please replan for all the agents again with the same ouput format:"
SYNTAX_BAD_JSON = "Your assigned plan is not in the correct json format as before. "
#: Adaptation (persistent assignments): an agent without a task must be
#: given one; phrased in the same pattern as SYNTAX_BAD_ENTRY.
SYNTAX_MISSING_ENTRY = "Your assigned task for {agent} is missing; "
#: env2-box-arrange.py: `'I Agree' in response or 'I agree' in response`.
AGREE_MARKERS = ("I Agree", "I agree")

#: The plan format shown to the planner (BoxNet2's position in the prompt).
PLAN_FORMAT_EXAMPLE = ('{"agent_0":{"task":"...","message":"..."}, '
                       '"agent_1":{"task":"...","message":"..."}}')

MAX_TASK_WORDS = 25
_COORD_RE = re.compile(r"-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?")
_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


CENTRAL_TEMPLATE = _load("hmas2_central.txt")
LOCAL_TEMPLATE = _load("hmas2_local.txt")


def is_agreement(response: str) -> bool:
    text = str(response or "")
    return any(m in text for m in AGREE_MARKERS)


def extract_plan(raw: str):
    """HMAS-2's extraction (``re.search(r'{.*}', response, re.DOTALL)``), then
    the tolerant multi-chunk parser. Returns a dict or None."""
    match = re.search(r"{.*}", str(raw or ""), re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except (ValueError, TypeError):
        parsed = parse_orchestrator_json(match.group())
    return parsed if isinstance(parsed, dict) and parsed else None


def format_history(entries: list, count_tokens: Callable[[str], int],
                   budget: int) -> str:
    """HMAS-2's ``_w_only_state_action_history`` rendering: newest pairs are
    kept, oldest dropped once the token budget is reached. ``entries`` is
    [(state, action)], oldest first; numbering is the step index."""
    out = ""
    used = 0
    for i in range(len(entries) - 1, -1, -1):
        state, action = entries[i]
        nxt = f"State{i + 1}: {state}\nAction{i + 1}: {action}\n\n"
        cost = count_tokens(nxt)
        if used + cost >= budget:
            break
        out = nxt + out
        used += cost
    return out


@dataclass
class HMAS2TickResult:
    reassigned: list = field(default_factory=list)   # agents whose task changed
    hub: Optional[dict] = None                       # {"messages", "record"}


class HMAS2Controller:
    """The per-step HMAS-2 protocol, duck-typed to the villager controller
    interface the training loop uses (tick / directive_text /
    assigned_objective / pinned_task / reset) plus the hub hooks
    (receive_report / record_delivery / episode_stats)."""

    def __init__(self, cfg: OrchestratorConfig, num_agents: int,
                 planner_client, check_client, orch_logger=None, *,
                 milestone_track: Optional[dict] = None,
                 chamber_describe=None,
                 task_validator: Optional[Callable[[str, str], bool]] = None,
                 token_counter: Optional[Callable[[str], int]] = None):
        self.cfg = cfg
        self.num_agents = num_agents
        self.planner_client = planner_client
        self.check_client = check_client
        self.orch_logger = orch_logger
        self.all_agents = [f"agent_{i}" for i in range(num_agents)]
        self._milestone_track = milestone_track
        self._chamber_describe = chamber_describe
        self._task_validator = task_validator
        if token_counter is None:
            from mindforge.env.comm_budget import make_token_counter
            token_counter = make_token_counter()
        self._count = token_counter
        self.reports = HubReports(self.all_agents, cfg.hmas2_report_cap)
        self._chamber_facts: Optional[str] = None
        self.reset()

    # ── Episode state ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Episode start: no assignments, empty histories, fresh reports."""
        self.reports.reset()
        self.assignments: dict = {}
        self.history: list = []                       # [(team state, plan)]
        self.own_history: dict = {a: [] for a in self.all_agents}
        self.last_messages: dict = {}
        self._events: list = []
        self._step_tokens: dict = {}
        self._step_record: dict = {"syntax_reprompts": 0}
        self.failed_calls = 0
        self.stats = {"steps": 0, "plan_calls": 0, "check_calls": 0,
                      "rounds": 0, "objections": 0, "checks": 0,
                      "syntax_reprompts": 0, "syntax_fallbacks": 0,
                      "reassignments": 0, "messages": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "latency_s": 0.0}

    # ── Coupling surfaces ────────────────────────────────────────────────

    def directive_text(self, agent_name: str) -> str:
        """{social_directive} slot: this agent's own task only (no crew
        list — teammate information travels only in messages)."""
        name = _normalize_agent(agent_name) or agent_name
        task = self.assignments.get(name)
        if not task:
            return ("Orchestrator assignment: (none yet — the central planner "
                    "will assign you a task shortly.)")
        return (f"Orchestrator assignment (from the central planner, which "
                f"coordinates the whole team): your task is \"{task}\". It "
                f"stays your task until the planner changes it.")

    def assigned_objective(self, agent_name: str) -> str:
        return ""     # hmas2 pins the task instead of constraining the curriculum

    def pinned_task(self, agent_name: str) -> Optional[str]:
        return self.assignments.get(_normalize_agent(agent_name) or agent_name)

    def receive_report(self, *, t: int, sender: str, text: str,
                       chamber: Optional[str] = None) -> None:
        self.reports.receive(t=t, sender=sender, text=text, chamber=chamber)

    # ── Validation (the analogue of with_action_syntactic_check_func) ───

    def _validate_task(self, task: str, chamber: str) -> bool:
        if not task or len(task.split()) > MAX_TASK_WORDS:
            return False
        if _COORD_RE.search(task):
            return False
        validator = self._task_validator
        if validator is None:
            from mindforge.agent_modules.auto_curriculum import _is_achievable_task
            validator = _is_achievable_task
        return bool(validator(task, chamber or ""))

    def check_syntax(self, parsed, *, living: list, chambers: dict,
                     prior_tasks: Optional[dict] = None):
        """Returns (plan, feedback). ``plan`` maps agent -> {"task",
        "message"} for every valid entry; ``feedback`` is "" when the
        whole plan is valid, else the HMAS-2 re-prompt text.

        ``prior_tasks`` (agent -> task) are tasks that already stand this
        step — the committed assignments plus, during a revision, the plan
        being revised — so an agent left out (or sent only a message) keeps
        its task instead of counting as missing."""
        if not isinstance(parsed, dict):
            return {}, SYNTAX_BAD_JSON + SYNTAX_REPLAN
        standing = dict(self.assignments)
        standing.update(prior_tasks or {})
        plan, feedback, flagged = {}, "", set()
        for key, entry in parsed.items():
            agent = _normalize_agent(key)
            if agent is None or agent not in living:
                feedback += SYNTAX_BAD_ENTRY.format(agent=key)
                continue
            if isinstance(entry, str):
                task, message = entry.strip(), ""
            elif isinstance(entry, dict):
                task = str(entry.get("task") or "").strip()
                message = str(entry.get("message") or "")
            else:
                feedback += SYNTAX_BAD_ENTRY.format(agent=agent)
                flagged.add(agent)
                continue
            if not task:
                if standing.get(agent):
                    task = standing[agent]              # message-only update
                else:
                    feedback += SYNTAX_MISSING_ENTRY.format(agent=agent)
                    flagged.add(agent)
                    continue
            elif not self._validate_task(task, chambers.get(agent, "")):
                feedback += SYNTAX_BAD_ENTRY.format(agent=agent)
                flagged.add(agent)
                continue
            plan[agent] = {
                "task": task,
                "message": sanitize_hub_text(message,
                                             self.cfg.hmas2_message_words),
            }
        for agent in living:
            if agent not in plan and agent not in flagged \
                    and not standing.get(agent):
                feedback += SYNTAX_MISSING_ENTRY.format(agent=agent)
        if feedback:
            feedback += SYNTAX_REPLAN
        return plan, feedback

    # ── LLM calls ────────────────────────────────────────────────────────

    async def _call(self, client, user_turns: list, assistant_turns: list,
                    call_type: str, episode: int, t: int) -> str:
        """One call with HMAS-2's message construction
        (``message_construct_func(..., '_w_all_dialogue_history')``):
        System, then user/assistant turns alternating, ending on a user
        turn. Returns the raw text ("" on failure)."""
        from autogen_core import CancellationToken
        from autogen_core.models import (
            AssistantMessage, SystemMessage, UserMessage,
        )

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for i, user in enumerate(user_turns):
            messages.append(UserMessage(content=user, source="user"))
            if i < len(user_turns) - 1:
                messages.append(AssistantMessage(content=assistant_turns[i],
                                                 source="assistant"))
        started = time.perf_counter()
        raw, prompt_tokens, completion_tokens, failed = "", 0, 0, False
        try:
            response = await client.create(
                messages, cancellation_token=CancellationToken())
            raw = response.content if isinstance(response.content, str) \
                else str(response.content)
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(
                    getattr(usage, "completion_tokens", 0) or 0)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failed = True
            logger.error("HMAS-2 %s call failed: %s", call_type,
                         str(exc)[:300])
        latency = time.perf_counter() - started
        key = "plan_calls" if call_type != "check" else "check_calls"
        self.stats[key] += 1
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        self._step_tokens[call_type] = self._step_tokens.get(
            call_type, 0) + prompt_tokens + completion_tokens
        if self.orch_logger is not None:
            self.orch_logger.log_call({
                "episode": episode, "t": t, "call_type": call_type,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_s": round(latency, 3), "failed": failed,
            })
        return raw

    async def _plan_checked(self, user_turns: list, assistant_turns: list,
                            *, living: list, chambers: dict, call_type: str,
                            episode: int, t: int,
                            prior_tasks: Optional[dict] = None):
        """A planner call followed by HMAS-2's syntactic check loop: the
        conversation grows with (response, feedback) pairs on each failure,
        up to ``hmas2_syntax_retries`` re-prompts. Returns (plan, ok,
        canonical_json)."""
        users = list(user_turns)
        assistants = list(assistant_turns)
        raw = await self._call(self.planner_client, users, assistants,
                               call_type, episode, t)
        plan, feedback = self.check_syntax(extract_plan(raw), living=living,
                                           chambers=chambers,
                                           prior_tasks=prior_tasks)
        attempts = 0
        while feedback and attempts < self.cfg.hmas2_syntax_retries:
            attempts += 1
            self.stats["syntax_reprompts"] += 1
            self._step_record["syntax_reprompts"] += 1
            assistants.append(raw)
            users.append(feedback)
            raw = await self._call(self.planner_client, users, assistants,
                                   "syntax", episode, t)
            plan, feedback = self.check_syntax(extract_plan(raw),
                                               living=living,
                                               chambers=chambers,
                                               prior_tasks=prior_tasks)
        ok = not feedback
        if not ok:
            self.stats["syntax_fallbacks"] += 1
        return plan, ok, json.dumps(plan, separators=(",", ":"))

    # ── Prompt blocks ────────────────────────────────────────────────────

    def _chamber_text(self) -> str:
        if self._chamber_facts is None:
            self._chamber_facts = build_chamber_facts(
                self.num_agents, describe=self._chamber_describe)
        return self._chamber_facts

    @staticmethod
    def _local_line(agent: str, info: dict) -> str:
        """HMAS-2's local-state line ("Agent[...]: I am in square[...], I can
        observe [...], I can do [...]") in WIRE terms."""
        return (
            f"{agent}: I am in {info.get('chamber') or '?'} at "
            f"{info.get('position') or '?'}, I can observe "
            f"{clip(info.get('observe') or 'nothing notable', 300)}, I hold "
            f"{info.get('holding') or 'nothing'}, my health is "
            f"{info.get('health') or '?'}, my current task is "
            f"\"{info.get('task') or 'none'}\" (last check: "
            f"{info.get('last_check') or 'not checked yet'}), I can do tasks "
            f"that dig, attack, move to, face, place, collect or explore the "
            f"things in my chamber")

    def _team_state(self, env_state: dict, completed: set) -> dict:
        known = known_milestone_ids(self._milestone_track)
        state = {
            "doors_open": [d for d, v in (env_state.get("doors") or {}).items()
                           if v],
            "unbroken_anvils": env_state.get("anvils") or [],
            "cell_doors_open": env_state.get("cell_doors_open") or [],
            "open_milestones": sorted(known - completed),
            "completed_milestones": sorted(completed & known),
        }
        if self._events:
            state["events"] = list(self._events)
        return state

    @staticmethod
    def _own_state(info: dict) -> str:
        return json.dumps({k: info.get(k) for k in
                           ("chamber", "position", "health", "holding",
                            "last_check")}, separators=(",", ":"))

    def _reports_block(self, living: list) -> str:
        rows = self.reports.pending()
        if not rows:
            return ""
        living_set = set(living)
        lines = [f"t={t}, {a} [in {ch}]{'' if a in living_set else ' (dead)'}:"
                 f" \"{text}\"" for t, a, ch, text in rows]
        return ("The latest updates the agents sent you are:\n"
                + "\n".join(lines) + "\n")

    @staticmethod
    def _budget_block(budget_left: Optional[dict]) -> str:
        if budget_left is None:
            return ""
        left = ", ".join(f"{a}={int(v)}" for a, v in sorted(budget_left.items()))
        return (f"Communication budget left per agent (tokens): {left}. A "
                f"message to an agent is charged to that agent; an agent at 0 "
                f"cannot receive messages.\n")

    def build_central_prompt(self, *, living: list, local: dict,
                             env_state: dict, completed: set,
                             budget_left: Optional[dict]) -> str:
        return CENTRAL_TEMPLATE.format(
            chamber_facts=self._chamber_text(),
            state_action_prompt=format_history(
                self.history, self._count, self.cfg.hmas2_history_tokens),
            current_state=json.dumps(self._team_state(env_state, completed),
                                     separators=(",", ":")),
            possible_actions="\n".join(
                self._local_line(a, local.get(a) or {}) for a in living),
            reports_block=self._reports_block(living),
            budget_block=self._budget_block(budget_left),
            format_example=PLAN_FORMAT_EXAMPLE,
        )

    def build_local_prompt(self, agent: str, info: dict, task: str) -> str:
        """The worker's check prompt: its own state, its own history and its
        own assignment only (no teammates, no joint plan, no message)."""
        return LOCAL_TEMPLATE.format(
            agent_name=agent,
            local_state=self._local_line(agent, info),
            state_action_prompt=format_history(
                self.own_history.get(agent, []), self._count,
                self.cfg.hmas2_history_tokens),
            assigned_task=task,
        )

    # ── The per-step tick ────────────────────────────────────────────────

    async def tick(self, *, state, living_agents: list, episode: int, t: int,
                   environment=None, agents=None, metric=None,
                   local_state_fn: Optional[Callable[[str], dict]] = None,
                   comm_budget=None, env_state: Optional[dict] = None,
                   agent_milestones: Optional[dict] = None,
                   **_unused) -> HMAS2TickResult:
        # Drain the event buffer (nothing else bounds it); milestones and
        # deaths since the last step go into the planner's state.
        self._events = []
        for ev in state.event_buffer:
            if ev.get("type") == "milestone":
                who = ", ".join(ev.get("contributors") or []) or "?"
                self._events.append(f"t={ev.get('t')} {ev.get('id')} by {who}")
            elif ev.get("type") == "death":
                self._events.append(f"t={ev.get('t')} {ev.get('agent')} died")
        state.event_buffer = []

        result = HMAS2TickResult()
        living = [_normalize_agent(a) or a for a in living_agents]
        if not living:
            return result
        started = time.perf_counter()
        self._step_tokens = {}
        self._step_record = {"syntax_reprompts": 0}

        if env_state is None:
            env_state = collect_env_state(environment, self.num_agents, t)
        if agent_milestones is None:
            agent_milestones = (dict(getattr(metric, "_agent_milestones", {}))
                                if metric is not None else {})
        completed = set()
        for ids in (agent_milestones or {}).values():
            completed.update(ids)
        local = {a: (local_state_fn(a) if local_state_fn else {}) or {}
                 for a in living}
        for a in living:
            local[a].setdefault("task", self.assignments.get(a))
        chambers = {a: local[a].get("chamber") or "" for a in living}

        def _left(agent):
            if comm_budget is None:
                return None
            return comm_budget.left(int(agent.rsplit("_", 1)[-1]))

        budget_left = ({a: _left(a) for a in living}
                       if comm_budget is not None else None)

        # 1-2. Plan + syntactic check.
        central = self.build_central_prompt(
            living=living, local=local, env_state=env_state,
            completed=completed, budget_left=budget_left)
        plan, ok, plan_json = await self._plan_checked(
            [central], [], living=living, chambers=chambers,
            call_type="plan", episode=episode, t=t)

        # 3. Local checks (<= max_rounds), objections -> revision.
        rounds, objections_log, checked_log = 0, [], []
        while plan and rounds < self.cfg.hmas2_max_rounds:
            rounds += 1
            feedback = ""
            for agent, entry in plan.items():
                if budget_left is not None and _left(agent) <= 0:
                    continue          # exhausted: cannot object (implicit agree)
                checked_log.append(agent)
                self.stats["checks"] += 1
                reply = await self._call(
                    self.check_client,
                    [self.build_local_prompt(agent, local[agent],
                                             entry["task"])],
                    [], "check", episode, t)
                if is_agreement(reply):
                    continue
                text = " ".join(str(reply or "").split())
                if comm_budget is not None and text:
                    charge = comm_budget.charge(
                        int(agent.rsplit("_", 1)[-1]), text, t)
                    if charge.status == "blocked":
                        continue
                    text = charge.text
                if not text:
                    continue
                self.stats["objections"] += 1
                objections_log.append({"round": rounds, "agent": agent,
                                       "objection": text})
                feedback += f"{agent}: {text}\n"
            if not feedback:
                break
            revised, ok, _ = await self._plan_checked(
                [central, feedback + FEEDBACK_REPROMPT.format(
                    example=PLAN_FORMAT_EXAMPLE)],
                [plan_json], living=living, chambers=chambers,
                call_type="revise", episode=episode, t=t,
                prior_tasks={a: e["task"] for a, e in plan.items()})
            if revised:
                # HMAS-2's revision is the whole plan; an agent the revision
                # leaves out keeps its entry from the plan being revised. The
                # next round re-checks every agent in the merged plan, as the
                # original re-checks the whole revised plan.
                merged = dict(plan)
                merged.update(revised)
                plan = merged
                plan_json = json.dumps(plan, separators=(",", ":"))

        # 4. Commit: pinned tasks + messages to deliver.
        messages = {}
        for agent, entry in plan.items():
            if entry["task"] != self.assignments.get(agent):
                self.assignments[agent] = entry["task"]
                result.reassigned.append(agent)
            if entry["message"]:
                messages[agent] = entry["message"]
        if not plan:
            self.failed_calls += 1
        self.stats["steps"] += 1
        self.stats["rounds"] += rounds
        self.stats["reassignments"] += len(result.reassigned)
        latency = time.perf_counter() - started
        self.stats["latency_s"] += latency

        team_state = json.dumps(self._team_state(env_state, completed),
                                separators=(",", ":"))
        self.history.append((team_state, plan_json))
        for agent in living:
            self.own_history.setdefault(agent, []).append(
                (self._own_state(local[agent]),
                 self.assignments.get(agent) or "none"))
        record = {
            "episode": episode, "t": t, "rounds": rounds,
            "syntax_reprompts": self._step_record["syntax_reprompts"],
            "syntax_ok": ok, "plan_empty": not plan,
            "checked": checked_log, "objections": objections_log,
            "plan": plan, "reassigned": list(result.reassigned),
            "assignments": dict(self.assignments),
            "reports_in": self.reports.count(),
            "reports_dropped": self.reports.dropped_since_consume,
            "tokens": dict(self._step_tokens),
            "latency_s": round(latency, 3),
        }
        self.reports.consume()
        result.hub = {"messages": messages, "record": record}
        return result

    # ── After delivery ───────────────────────────────────────────────────

    def record_delivery(self, hub: dict, delivery: dict) -> dict:
        """Merge the main loop's delivery outcome into this step's record
        and write the hmas2.jsonl row."""
        record = dict((hub or {}).get("record") or {})
        rows = {}
        for agent, text in ((hub or {}).get("messages") or {}).items():
            d = (delivery or {}).get(agent) or {}
            if d.get("delivered"):
                self.last_messages[agent] = d.get("text") or text
                self.stats["messages"] += 1
            rows[agent] = {
                "text": text, "words": len(str(text).split()),
                "delivered": bool(d.get("delivered")),
                "budget_status": d.get("status"),
                "charged": d.get("charged", 0),
                "budget_left": d.get("budget_left"),
            }
        record["messages"] = rows
        if self.orch_logger is not None and hasattr(self.orch_logger,
                                                    "log_hmas2"):
            self.orch_logger.log_hmas2(record)
        return record

    def episode_stats(self) -> dict:
        s = self.stats
        n = max(1, s["steps"])
        return {
            "steps": s["steps"], "plan_calls": s["plan_calls"],
            "check_calls": s["check_calls"],
            "mean_rounds": round(s["rounds"] / n, 3),
            "objection_rate": round(s["objections"] / max(1, s["checks"]), 3),
            "syntax_reprompts": s["syntax_reprompts"],
            "syntax_fallbacks": s["syntax_fallbacks"],
            "reassignments": s["reassignments"],
            "messages_delivered": s["messages"],
            "mean_latency_s": round(s["latency_s"] / n, 3),
            "prompt_tokens": s["prompt_tokens"],
            "completion_tokens": s["completion_tokens"],
            "reports_in": self.reports.reports_in,
            "reports_dropped": self.reports.reports_dropped,
        }
