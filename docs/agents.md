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
`--hebbian`). Four variants, all `--orchestrator-variant`:

| Variant | What the coordinator sees and does |
|---|---|
| `task` | Keeps a within-episode task ledger; issues per-agent comm-target/help directives. Relational content filtered out |
| `social` | Information-matched to the Hebbian rule: pair co-presence, message counts and co-reward in, a directive in the social module's exact format out; ledger persists across episodes like `W` |
| `plan` | `social` plus each agent's current curriculum task, and a plan note delivered into that agent's next task generation |
| `villager` | VillagerAgent-style: an LLM decomposes the remaining objective into milestone-verified subtasks in a dependency graph and assigns them to agents. Assignments bind the objective, not the primitives |

`villager` is the baseline reported in the paper. Subtasks complete only when a real WIRE milestone
fires — the coordinator never asks an agent to self-report success — and failures are retired
rather than retried. Decomposition is event-driven and rate-limited to the same cadence as the
social module, which is what makes the compute comparison fair.
