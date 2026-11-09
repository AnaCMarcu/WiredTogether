# Chained AgentWorld tasks

One team plays several AgentWorld tasks in a row. The game resets for every
task (`new_character: true`). Each agent keeps its identity, its bonds to
teammates and its memory of past interactions. Steps run through AgentWorld's
own runner (`agents/run.py`, `TaskRunner.run_single_task`). The agents are
AgentWorld's provider agents plus the hooks in `wired_agent.py`.

| File | What |
|---|---|
| `metallurgy_4.yaml` | main 3-agent chain: tasks 08 → 12 → 04 → 07 |
| `metallurgy_4_reversed.yaml` | order control: the same tasks hardest first |
| `metallurgy_5.yaml` | 5-agent chain: tasks 49 → 46 → 52 |
| `smithing_3.yaml` | first chain: tasks 03 → 06 → 07, a1/a2/a3 in fixed slots |
| `smithing_3_rotate.yaml` | the same tasks with roles rotating one slot per step |
| `generate_tasks.py` | writes the renamed task YAML + verifier for each step |
| `run_chain.py` | runs a chain; writes the summary, state files and per-round learning log |
| `check_chain_run.py` | checks a finished run (names, reset, state files, verifiers) |
| `wired_agent.py` | `agent.provider: custom` adapters with the hooks |
| `wired_state.py` | state store, memory, the link to the Hebbian rule |
| `configs/` | agent configs: OpenAI-compatible, Claude, random plumbing |

You need an AgentWorld checkout at `hpc/agentworld/AGENTWORLD_COMMIT`
(23a07d52 or later, for `scripts/score_psr.py`), set as `$AGENTWORLD_ROOT`.
You also need a running game server, set as `$AGENTWORLD_BASE_URL`, for
example `http://localhost:7031`.

## Which tasks, in which order

Carried state can only help when every step makes the same teammates
interact in the same way. The metallurgy chains therefore keep one division
of labour: suppliers hand raw materials to a smith (a1/a2 → a3 in
`metallurgy_4`; miners → smelter a4 → crafter a5 in `metallurgy_5`). They also
grow one pipeline, raw material → bars → item, by one stage per step, and go
from easy to hard so that early successes come before the hard coordination
tests.

`metallurgy_4`:

1. **Task 08, bronze bar.** a1 already holds the ore and hands it to a3. This
   is the smallest real handoff.
2. **Task 12, gold ring.** a1 must mine gold ore, because the nuggets it holds
   do not smelt; a2 supplies coal.
3. **Task 04, axe.** Everything needed is already in a1's and a2's
   inventories. Earlier runs failed 0/8 because the miner tried to smelt
   instead of handing over.
4. **Task 07, pickaxe.** The same pipeline at its largest: 5 bars and logs.

Tasks 03 (no interaction) and 06 (0/6 even in 150 rounds) are left out.
`metallurgy_4_reversed` separates the effect of carried state from the effect
of the order itself.

`metallurgy_5` uses AgentWorld's own 5-agent tasks. They are much harder:
inventories start empty and up to 11 ore and 11 coal must be mined. Their
verifiers report counts, so PSR still gives partial credit.

## Defining a chain

```yaml
chain: smithing_3                  # lowercase; names the output folder
description: free text
steps:                             # run in this order
  - task: data_v0.1_multi/v1.3_benchmark/task_03_silver_ring.yaml   # relative to $AGENTWORLD_ROOT
    agents: {agent_1: a1, agent_2: a2, agent_3: a3}                 # task slot -> agent id
  - task: data_v0.1_multi/v1.3_benchmark/task_06_heavy_sword.yaml
    agents: {agent_1: a1, agent_2: a2, agent_3: a3}
```

In the game an agent is called `<agent_id>_<role>`. The role is the task's
own username without `tNN_` and `_agent`, so `t06_weapon_smith_agent` played
by a3 becomes `a3_weapon_smith`. Every step must map each task slot exactly
once.

Agent ids are lowercase letters and digits with no underscore and are not of
the form `tNN`. Usernames must be valid on the server, lowercase and at most
32 characters. Anything else fails before a step starts.

To rotate roles, change the map per step. Ids, bonds and memory stay with the
agent. See `smithing_3_rotate.yaml`:

```yaml
  - task: .../task_06_heavy_sword.yaml
    agents: {agent_1: a2, agent_2: a3, agent_3: a1}    # a1 is now a1_weapon_smith
```

## Generating the tasks

```bash
python chains/generate_tasks.py chains/smithing_3.yaml
# -> chains/generated/smithing_3/step_01_task_03/task_03_silver_ring.yaml
#                                               task_03_silver_ring_success_criteria.py
```

- **Renaming.** Every old username is replaced, both the `username:` fields
  and the mentions in the text (`t03_smith_agent` and the bare `smith_agent`).
  The script checks that nothing else changed: same keys, same values, same
  line endings. It fails if a task does not set `new_character: true`.
- **Verifiers.** Each task's verifier is copied byte for byte. AgentWorld's
  `agents/verification.py` picks it up next to the YAML. Its `verifier_utils`
  import resolves because that loader preloads `verifier_utils`, and
  `benchmarks/score.py` loads verifiers the same way.
- **Tasks 17–20.** These verifiers find an agent's inventory by a substring
  of its username (`"palm_logger" in "a1_palm_logger"`), so they still work
  after renaming. `tests/test_aw_chains.py` checks this.
- **Source files.** The original task files are never written to.
  `generated/` is git-ignored because its files are MPL-2.0 copies.

`run_chain.py` runs the same generator into each run folder (`tasks/`), so
every run records what it played and two runs of one chain never share files.

## Running

```bash
export AGENTWORLD_ROOT=/path/to/agentworld AGENTWORLD_BASE_URL=http://localhost:7031

# 1. Plumbing only: no model calls, random legal actions
python chains/run_chain.py chains/smithing_3.yaml \
    --agent chains/configs/random_plumbing.yaml --random-agent --seed 1

# 2. One step with your model
export MODEL_NAME=... MODEL_API_KEY=... MODEL_BASE_URL=http://localhost:8000/v1
python chains/run_chain.py chains/smithing_3.yaml \
    --agent chains/configs/wired_openai.yaml --steps 1 --run-id model_smoke

# 3. The whole chain, one run per arm and seed
for arm in reference memory hebbian; do
  python chains/run_chain.py chains/smithing_3.yaml \
      --agent chains/configs/wired_openai.yaml --arm $arm --seed 42
done
```

**Agent config.** Start from your current config and change two lines:
`provider: custom`, plus `llm.agent_class: wired_agent:WiredOpenAIAgent`
(or `WiredClaudeAgent`, `WiredGeminiAgent`, `WiredDeepSeekAgent`, matching
the old provider). Leave `system_prompt`, `user_prompt_template` and the
model settings unchanged.

**Experiment flags.** AgentWorld's flags pass through to every step:
`--max-rounds`, `--random-spawn`, `--spawn-region`, `--no-task-docs`,
`--discussion-rounds`, `--no-communication`, `--shared-plan-only`,
`--random-agent` and `--seed`. Two flags are refused. `--no-roles` would
rename every agent to `Agent_N`, and `--single-agent-upper-bound` would merge
the team into one agent.

**Other options.** `--steps 2-3 --run-id <id>` continues a run using the
state files already in it. `--split-screen` turns AgentWorld's curses display
back on.

### Arms

| `--arm` | Added to the system prompt |
|---|---|
| `reference` | nothing (AgentWorld's reference agent) |
| `memory` | earlier task outcomes, the last 4 interactions with each teammate from earlier tasks, and up to 6 skill-log lessons (this task's items first) |
| `hebbian` | as `memory`, plus `Your strongest bonds (higher = you worked well together recently): ...` |

Bonds and memory are computed and saved in every arm, as in the `base` arm of
our own harness, so the arms differ only in what the model sees. The block
goes into the system prompt, which AgentWorld rebuilds on every call. Putting
it in the user message would copy the whole task prompt into the
conversation each time a bond moved.

## What persists, and how it learns

`runs/<chain>/<run_id>/agent_state/a1.json` is loaded when a task starts and
saved when it ends. It holds:

- `bonds`, keyed by the partner's agent id;
- `memory`: one entry per message sent or received and per item transfer,
  with the partner's agent id, the partner's role at the time, the task, the
  step and the round;
- `history`: one entry per step played, with role, outcome, PSR, rounds,
  chats, and bonds at the start and end;
- `skills`: crafts and harvests that produced items, read from the agent's own
  inventory and not counting items handed to it, plus crafts that failed, with
  the error. A lesson such as "smelting ironbar failed for me as miner" thus
  reaches the next task.

The hooks are methods on the adapter. The runner calls them, because only the
runner knows the teammates, who received a chat or transfer, and the verified
outcome. `--random-agent` also skips the agent's own code.

- `on_task_start(agent_id, role, teammates)` runs after login. `teammates`
  holds `(agent_id, role, username)` for the others.
- `on_turn(round, interactions)` runs for the actor after each turn, and for
  the partner of each message or transfer.
- `on_skill(round, outcome)` runs for each craft or harvest result.
- `on_task_end(agent_id, outcome)` runs after the step's verifier.

The update rule is your `MultiChannelHebbianGraph` (`src/hebbian/multichannel.py`),
unchanged. Its rates default to the calibrated values in `src/agentworld/cli.py`
and can be overridden with the same `--hebbian-*` flags. Each task builds one
graph from the saved bonds and updates it once per round. The inputs are those
`src/agentworld/scheduler.py` uses:

- `EventExtractor` events: transfers, joint combat and deaths;
- communication pairs;
- positions;
- `ProgressTracker`'s dense third factor, from verifier progress.

Only W carries over between tasks; the eligibility and salience traces start
fresh. AgentWorld's chat is global, so a chat counts as communication with the
teammates its text names (username, id or role), or with everyone if it names
nobody. Each round's events, third factor and W are logged to
`step_NN_task_MM/wired_rounds.jsonl`.

## Outputs and metrics

`chain_summary.json` records the following for each step:

- `success` and `verifier_message`, from the step's verifier run again after
  the step, and the runner's own verdict;
- `psr`: reconstructed PSR, AgentWorld's `checkpoint_fraction_v1` from
  `scripts/score_psr.py`, with each checkpoint's credit, `n_checkpoints` and
  `n_met`;
- `rounds`, `chats`, `actions` and `early_stopped`;
- which state file each agent loaded and saved;
- bonds at the end of the step.

`totals` gives SR, mean PSR, mean rounds and mean chats over the steps, the
same quantities as the AgentWorld paper. The `checks` section holds the
results of `check_chain_run.py`.

CCE needs an LLM judge, so it is computed afterwards from each step's
trajectory with AgentWorld's script (see their `docs/benchmark/scoring.md`):

```bash
python $AGENTWORLD_ROOT/analysis/scripts/compute_cce_v2.py \
    --trajectory runs/smithing_3/<run_id>/step_02_task_06/task_06_trajectory.json \
    --output runs/smithing_3/<run_id>/step_02_task_06/cce.json
```

For tasks 03, 06 and 07 the verifier reports a single yes/no checkpoint, so
PSR equals SR on these three.

## On Snellius

`hpc/agentworld/aw_chain.sbatch` runs one chain in one GPU job: vLLM (Gemma,
32k context), one game server, and `run_chain.py` in the Gemma image. It uses
`hpc/agentworld/chain_common.sh` on top of the unchanged `aw_common.sh`. Submit
from the repository root after `source hpc/snellius/env.sh`:

```bash
bash hpc/agentworld/setup_aw.sh fetch                  # AgentWorld at the pinned 23a07d52
bash hpc/agentworld/submit_aw_chain.sh plumbing        # random agent, no model, 4 rounds/step (~80 s each)
bash hpc/agentworld/submit_aw_chain.sh check           # step 1 with Gemma (hebbian arm)
bash hpc/agentworld/submit_aw_chain.sh pilot           # 3 arms x seed 42, metallurgy_4
CHAIN=chains/metallurgy_5.yaml bash hpc/agentworld/submit_aw_chain.sh pilot
```

Output goes to `runs_aw/<group>/<chain>/<arm>_s<seed>_j<job>/`, with
`vllm_server.log`, `server.log` and `run.log` next to the run. `--time` is
always set: 1 h for `check` and `plumbing`, 2 h for a `metallurgy_4` run and
5 h for `metallurgy_5`, or `TIME=HH:MM:SS`. These come from AgentWorld's runner
pace: about 36 s per harvest or attack, 0.1–1 s per other action, plus the
model call. Use `ARRAY=0-2` for seeds 42, 123 and 456.

## Checks

`python chains/check_chain_run.py runs/smithing_3/<run_id>` runs four checks,
and `run_chain.py` runs them at the end of every run:

- **usernames:** each step's trajectory uses `<agent_id>_<role>` names that
  match the slot map, with no old names left;
- **reset:** no agent starts a step holding items from its previous step, and
  every agent starts at the task's spawn;
- **state:** files are written after step 1, each later step loads the file
  from the agent's previous step, and the histories are complete;
- **verifiers:** every step has a verifier message without errors.
