# The agent

Each agent is a MindForge cognitive stack over one LLM. Per step it receives a first-person frame,
structured context and its inbox, and returns a reasoning trace, one discrete action, and one
targeted message. `mindforge/custom_agent.py` holds the agent; the modules live in
`mindforge/agent_modules/`, the templates in `mindforge/prompts/`.

## Modules

| Module | What it contributes to the action prompt |
|---|---|
| `belief_system` | Four belief blocks: what the agent sees (in action-relative terms), what its teammates are doing, actionable content extracted from recent messages, and the current task framed as an action → expected-observation scheme |
| `auto_curriculum` | The next concrete task, conditioned on chamber, fired milestones and inventory. Validity filters reject proposals referencing fixtures outside the current chamber |
| `critic` | Every 20 steps, judges whether the current task succeeded. Successes promote their action into the skill store, repeated failures force a task change, and every verdict is appended to episodic memory |
| `skill_manager` | Vector store of action–thought pairs that worked, retrieved for similar tasks |
| `episodic_memory_manager` | ChromaDB store keyed on task embeddings, retrieved at a 70/30 success-to-failure ratio |
| `social_module` | The Hebbian coupling: bond row + deltas → request help / offer help / stay on task |
| `chamber_facts` | The room-facts text for the chamber the agent is standing in, N-templated |

The prompt also carries a hard-grounded list of the chambers the agent has actually visited this
episode, which rules out a hallucination mode where the model claims to be somewhere it has never
been.

## Action selection

`action_selection.py` returns JSON: `thoughts`, `action`, `communication`, `communication_target`.
Targeted messaging is mandatory — there is no broadcast channel — and messages that are too short,
duplicated, self-targeted or aimed at a nonexistent agent are filtered before delivery, which is
also what makes them ineligible for communication reward.

Under `--rl` the same prompt is used minus the `action` field: the model emits thoughts, and the
policy then scores each candidate action string as a continuation of prompt + thoughts
(constrained decoding, see [rl-layer.md](rl-layer.md)).

## The social module

Runs before action selection, every `--social-interval` steps; between deliberations the previous
directive is cached and re-rendered. Its input is the agent's outgoing bond row and each bond's
change over the last 50 steps, tagged STRENGTHENING / DECAYING / STABLE. Its output is a validated
JSON object: `ask_target` + `ask_message`, or a list of senders to `respond_to`, or neither.

Two fields exist purely to make the coupling auditable: `referenced_bonds` requires the model to
copy back the exact (teammate, weight) pairs it used, and `bond_change_explanation` requires a
stated cause for every moving bond. Whether a decision was actually conditioned on the graph is
therefore checkable per step rather than assumed — `analysis/qualitative/` mines those fields.

`--social-module prompt` renders the directive into the action prompt. `bias` additionally
overrides the action model's `communication_target` with `ask_target` at the routing layer.

## Orchestration baselines

`src/orchestrator/` replaces the graph with a central coordinator (mutually exclusive with
`--hebbian`). Five variants, all `--orchestrator-variant`:

| Variant | What the coordinator sees and does |
|---|---|
| `task` | Keeps a within-episode task ledger; issues per-agent comm-target/help directives. Relational content filtered out |
| `social` | Information-matched to the Hebbian rule: pair co-presence, message counts and co-reward in, a directive in the social module's exact format out; ledger persists across episodes like `W` |
| `plan` | `social` plus each agent's current curriculum task, and a plan note delivered into that agent's next task generation |
| `villager` | VillagerAgent-style: an LLM decomposes the remaining objective into milestone-verified subtasks in a dependency graph and assigns them to agents. Assignments bind the objective, not the primitives |
| `hmas2` | The hard orchestrator: HMAS-2 (Chen et al., ICRA 2024) adapted to WIRE. Every step a central planner assigns subtasks and each assigned agent checks its own assignment; hub-and-spoke communication |

`villager` is the baseline reported in the paper. Subtasks complete only when a real WIRE milestone
fires — the coordinator never asks an agent to self-report success — and failures are retired
rather than retried. Decomposition is event-driven and rate-limited to the same cadence as the
social module, which is what makes the compute comparison fair.

### The hard orchestrator (`hmas2`)

`villager` is a *soft* orchestrator: it decides who does what, but agents still message each
other and the coordinator never reads those messages. `hmas2` is the *hard* one — a centralised
hub-and-spoke system. It is a port of a published embodied baseline, HMAS-2 (Chen, Arkin, Zhang,
Roy, Fan, "Scalable Multi-Robot Collaboration with Large Language Models: Centralized or
Decentralized Systems?", ICRA 2024; code github.com/yongchao98/multi-agent-framework), not a new
design. Its protocol, loop bounds and prompt sentences come from that code (BoxNet2):

1. **Plan.** The central planner gets the task, the state–action history (the paper's
   `_w_only_state_action_history` setting, newest pairs within 3,000 tokens), the current state and
   each agent's possible actions, and returns a JSON plan. "Include an agent only if it has a task
   next."
2. **Syntactic check.** Invalid entries are sent back with HMAS-2's wording ("Your assigned task
   for agent_k is not in the doable action list; … Please replan for all the agents again with the
   same ouput format:"), up to 6 re-prompts.
3. **Local check, ≤ 3 rounds.** Each agent in the plan answers `I Agree` or an objection.
   Objections return to the planner as a multi-turn follow-up ("This is the feedback from local
   agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the
   same plan as before…"), and the revised plan is checked again.
4. **Execute.**

Documented adaptations (state them with the results):

| | HMAS-2 | `hmas2` |
|---|---|---|
| Assignment | a primitive action per robot | a subtask (`"task"`) the MindForge worker executes; its curriculum task is **pinned** to it (no task-choice call) |
| Worker information | own state + every robot's state + the joint plan | own observations and memory, own assignment, and the planner's message to it — nothing else |
| Worker → coordinator | check replies | check replies + per-step reports (an agent's `communication` goes to the orchestrator) |
| Persistence | a new action every step | an assignment stays until the planner changes it; "has a task next" = a new task or a message |
| Syntactic failure | ends the trial | keeps the valid entries and the previous assignments |
| History budget | 3,000 tokens for the whole prompt | 3,000 tokens for the history alone (WIRE's base prompt is larger) |

The planner sees the whole team, as in HMAS-2. A worker gets teammate information only through
the planner's message (delivered by replacing its inbox, header "Message from the orchestrator"),
so selective, message-borne information — the property the Hebbian comparison is about — holds in
both conditions. Agents cannot message each other: the routing site sends every message to the
orchestrator (`messages.jsonl`: `receiver: "orchestrator"`, `routing: "hub"`), and the prompts say
so (`agent_modules/hub_prompts.py`, the `WT_COMM_TOPOLOGY=hub` wording in `env/comm_budget.py`).

The planner runs inside the per-step tick, before the agents read their inboxes, so a report sent
at step t is answered at step t+1 — the same delay as a peer message under `--simultaneous`.

Under `--comm-budget-tokens`, reports are charged to the sender, the planner's messages to the
recipient and objections to the objector; an exhausted agent is not checked and receives nothing.
At budget 0 the protocol reduces to CMAS with history (central planning, no feedback).

Prompt port, original → `hmas2` (templates in `src/orchestrator/prompts/hmas2_*.txt`, verbatim
re-prompt strings in `src/orchestrator/hmas2.py`; `tests/test_orchestrator_hmas2.py` pins them):

| HMAS-2 (BoxNet2) | `hmas2` |
|---|---|
| "You are a central planner directing agents in a grid-like field to move colored boxes." + grid mechanics | "…directing agents in a five-chamber cooperative dungeon to complete milestones." + the chamber facts |
| "Actions are like: move(box_red, target_red)…" | "Actions are like: "Dig anvil A1 together with agent_1"…" |
| "Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates… optimally." | "…to complete the open milestones and progress through all five chambers." — rest verbatim |
| "The previous state and action pairs at each step are:" + "Please learn from previous steps…" | verbatim |
| "Hence, the current state is {state}, with the possible actions: {actions}" | verbatim; state = doors, anvils, cells, milestones; one "agent_k: I am in …, I can observe …, I can do …" line per agent |
| — | **added:** the agents' latest reports; the budget left (budget runs) |
| "Specify your action plan in this format: {…}. Include an agent only if it has a task next. Now, plan the next step:" | verbatim, format `{"agent_0":{"task":…,"message":…}}`; **added:** persistence rule and "tell that agent only what it needs to know about the other agents" |
| Local: "You're a box-moving agent in a multi-agent system… A central planner coordinates all agents to achieve the goal: …" | "You're agent_k, an agent in a five-chamber cooperative dungeon, in a multi-agent system…" — rest verbatim |
| Local: "The current states and possible actions of all other agents are: …", "The current state is {global state}" | **removed** (selective information) |
| Local: "The central planner's current action plan is: {joint plan}." | "…action plan for you is: {own task}." |
| Local: "If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:" | verbatim |

Logs: `orchestrator/hmas2.jsonl` (one row per step: rounds, objections, syntax re-prompts, final
plan, reassignments, reports, message deliveries, tokens, latency) and `calls.jsonl` rows with
`call_type` plan / revise / syntax / check. `analysis/make_hmas2_load.py` summarises them. Known
asymmetries: the protocol adds serial LLM calls to every step; a hub inbox holds one entry (fewer
partner-belief calls than in mesh); Hebbian/co-firing `comm_events` stay empty. Information the
environment itself provides — `{milestone_progress}` (team completions), `{chamber_state}` (anvil
punchers, switch wiring), teammate names — is the same in every arm.
