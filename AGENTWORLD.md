# Running AgentWorld with this repository, and plugging in your own agent

[AgentWorld](https://github.com/openagents-org/agentworld) is a multi-agent
benchmark built on Kaetram, a 2D online role-playing game. Teams of LLM agents
log in as players and must cooperate on crafting, gathering and combat tasks.
For example, one agent cuts logs, a second turns them into sticks and a third
crafts a staff. Each task ships with a verifier that scores the recorded
trajectory.

This branch (`agentworld-starter`) contains everything needed to run it, plus
a template for bringing your own agent. AgentWorld's own code stays outside
this repository (MPL-2.0); you clone it next to this one.

## Contents

1. [Three ways to run it](#1-three-ways-to-run-it)
2. [Setup](#2-setup-once)
3. [Check the setup without a model](#3-check-the-setup-without-a-model)
4. [Your agent in our harness (route B)](#4-your-agent-in-our-harness-route-b)
5. [Your agent in AgentWorld's runner (routes A and A+)](#5-your-agent-in-agentworlds-runner-routes-a-and-a)
6. [Serving a model](#6-serving-a-model)
7. [Things that will bite you](#7-things-that-will-bite-you)
8. [Where things live](#8-where-things-live)

## 1. Three ways to run it

| Route | Runner | Your agent is | Pick it when |
|---|---|---|---|
| **A** | AgentWorld's own `agents/run.py` | a subclass of their `BaseAgent` | you want numbers comparable to the AgentWorld paper and leaderboard (their protocol: agents take turns, one action per turn) |
| **A+** | `chains/run_chain.py` (AgentWorld's runner, several tasks in a row) | the same subclass, wrapped with our hooks | the same team should play several tasks in a row and keep its identity, memory and bonds between them |
| **B** | `examples/agentworld/run_my_agent.py` (our harness, `src/agentworld/`) | any object with `async decide(turn) -> Decision` | you want rounds where all agents act at once, private messages, bonds between agents, many teams in one world, or the MindForge agent as a baseline |

Route B is the quickest way to plug in an agent of your own: one method,
plain Python objects in and out, and nothing to subclass. Route A is the one
to use for a leaderboard number. Both play the same tasks, and the same
AgentWorld verifiers score them.

## 2. Setup (once)

**You need:** Linux or WSL2 on Windows (see section 7 for native Windows),
Python 3.11 or later, Node 20 with corepack (for Yarn 4), and git. You need a
GPU only if you serve a model yourself.

### 2.1 This repository

```bash
git clone https://github.com/AnaCMarcu/WiredTogether.git
cd WiredTogether
git checkout agentworld-starter
python -m venv .venv && source .venv/bin/activate
```

Python packages:

```bash
# Route B with your own agent: harness, template agent, replay videos
pip install numpy httpx pyyaml requests jinja2 openai pillow matplotlib imageio-ffmpeg
# the offline tests also need pytest and torch (the CPU build is enough)
pip install pytest torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Only if you also want the MindForge agent (python -m mindforge.multi_agent_agentworld,
# chains/run_mf_chain.py): the full project environment, which includes torch, autogen and chromadb
pip install poetry && poetry install
```

### 2.2 AgentWorld, at the commit this code was tested against

```bash
git clone https://github.com/openagents-org/agentworld ../agentworld
git -C ../agentworld checkout $(cat hpc/agentworld/AGENTWORLD_COMMIT)
(cd ../agentworld && corepack enable && HUSKY=0 yarn install && cp .env.defaults .env)
pip install -r ../agentworld/agents/requirements.txt      # their runner (routes A / A+)
export AGENTWORLD_ROOT=$(realpath ../agentworld)
```

### 2.3 A game server

Route B can start and restart the server itself: leave out `--server-url`
and it runs `yarn workspace @kaetram/server start` on free ports. For routes
A and A+, and for repeated route-B runs, start one yourself and leave it
running:

```bash
AGENTWORLD_HOME=$AGENTWORLD_ROOT PORT=7030 API_PORT=7031 \
    sh hpc/agentworld/run_world.sh > server.log 2>&1 &
# ready once this answers (about a minute the first time):
curl -s http://localhost:7031/ai/world-status
export AGENTWORLD_BASE_URL=http://localhost:7031
```

Use `run_world.sh` rather than exporting `PORT` yourself: the game reads its
ports only from `.env` files (see section 7).

## 3. Check the setup without a model

```bash
# 1. Unit tests against an in-memory fake game server (no server, no model):
#    expect about 176 passed, 12 skipped. Name the files: the rest of tests/ needs the
#    full environment.
python -m pytest tests/test_aw_*.py -q

# 2. Route B with a scripted agent against the real server: logins, moves, messages,
#    trajectory, verifier, video
python examples/agentworld/run_my_agent.py --agent my_agent:ScriptedAgent \
    --tasks task_01_magic_staff --rounds 5 --server-url $AGENTWORLD_BASE_URL \
    --out runs_aw/mine/scripted

# 3. Route A+ with AgentWorld's random agent (no model calls), one step of a chain
python chains/run_chain.py chains/pendant_staff_4.yaml \
    --agent chains/configs/random_plumbing.yaml --random-agent --seed 1 --steps 1
```

Check 2 passes if `runs_aw/mine/scripted/episode_1/` holds `summary.json`,
`trajectories/`, `messages.jsonl` (with the scripted agents' hello messages)
and `videos/team_t00.mp4`. The scripted agent does not try to solve the task, so
`"success": 0` is expected.

## 4. Your agent in our harness (route B)

### 4.1 The contract

The harness (`src/agentworld/scheduler.py`) runs rounds. In each round, every
agent that is neither done nor busy with a long action receives a `Turn`
and returns a `Decision`. Your class needs:

```python
class HerAgent:
    def __init__(self, names: list[str]): ...      # one object for the whole team;
                                                   # names[i] = agent i's in-game username
    async def decide(self, turn) -> Decision: ...  # once per agent per round
    def on_reset(self): ...                        # optional: before every episode
    ledger = ...                                   # optional: .snapshot() -> dict, saved in summary.json
```

All agents' `decide` calls run at the same time (`asyncio.gather`), so make
it `async`. Wrap anything blocking (a local model, a synchronous library) in
`await asyncio.to_thread(...)`. If `decide` raises, that agent waits this
round, and the error is logged in its thoughts as `cognition error: ...`.

What a `Turn` gives you (`from agentworld.scheduler import Turn`):

| Field | What |
|---|---|
| `round`, `agent`, `name`, `team` | round number; this agent's index, username and team id. **Key your per-agent state on `turn.agent`.** |
| `objective`, `game_context` | the task text from the task YAML |
| `observation` | the raw observation dict from the game API (position, HP, inventory, nearby players, mobs, resources, with instance ids) |
| `obs_text` | the same observation already written out as text (the nearest 6 of each kind) |
| `last_result` | what the game said about this agent's previous action, including errors |
| `inbox.dms` | private messages delivered to this agent this round (`Message.sender` is an index into `turn.names`, `.text`, `.round`) |
| `contacts` | usernames of teammates this agent may message this round |
| `bonds` | `[(username, weight), ...]` from the Hebbian graph; **empty in the `base` arm** |
| `progress_text` | the team's verifier progress, e.g. `Arrows: 3/10` |
| `names` | every username in the world |

What you return (`from agentworld.scheduler import Decision`):

| Field | What |
|---|---|
| `action` | one game call as text, e.g. `harvest_resource(targetInstance="645004284")`, `transfer_items(targetPlayer="a2_woodworker", itemKey="stick", count=5)`, `wait()`, `complete()`. Every call is listed in `agentworld.actions.TOOLS`, and `tool_reference()` writes the list out for a prompt. A malformed call fails and its error comes back in the next `last_result`. |
| `thoughts` | free text, saved in the trajectory |
| `dm_target`, `dm_text` | optional: one private message to one teammate (a username). It is delivered at the start of the next round, together with this round's action. |
| `board_post` | optional: a post on the shared board, which agents read through `turn.inbox.board` (at most `--board-budget` posts per round, default 5) |

### 4.2 Start from the template

`examples/agentworld/my_agent.py` has two classes:

- `ScriptedAgent`: no model. Use it for the plumbing check in section 3.
- `MyLLMAgent`: one call to any OpenAI-compatible endpoint per turn, with the
  reply as JSON (`thoughts`, `action`, `dm_target`, `dm_text`). Its parts are
  marked `YOUR CODE`: `remember()` is a stand-in for a memory module,
  `build_messages()` builds the prompt, and `on_reset()` decides what is kept
  between episodes.

```bash
cp examples/agentworld/my_agent.py examples/agentworld/her_agent.py   # or anywhere on PYTHONPATH
# edit HerAgent / MyLLMAgent in her_agent.py, then:
export MY_LLM_BASE_URL=http://localhost:8000/v1 MY_LLM_MODEL=<model id> # MY_LLM_API_KEY if needed
python examples/agentworld/run_my_agent.py --agent her_agent:MyLLMAgent \
    --tasks task_01_magic_staff --server-url $AGENTWORLD_BASE_URL \
    --min-round-seconds 20 --arm base --out runs_aw/mine/staff_base
```

`--agent module:Class` is imported from `examples/agentworld/`, from
`PYTHONPATH`, or from the current directory. Your module can import anything,
so your own package can live in its own repository: put it on `PYTHONPATH` and
keep only a thin `decide()` adapter here. `tests/test_aw_custom_agent.py`
shows how to test an agent offline against the fake server, with a fake model
client.

### 4.3 What the harness does around your agent

- **World.** `--tasks` takes task file stems from `$AGENTWORLD_ROOT/data_v0.1_multi/v1.3_benchmark`
  (or paths). `--replicas K` puts K copies of the team in one world, spaced
  `--team-spacing` tiles apart, so several teams share one server.
- **Rounds.** `--rounds` defaults to the task's own limit, usually 55.
  `--episodes N` replays the task, restarting the server in between
  (`on_reset` is called each time). `--deadline` caps how long one round's
  actions may run (30 s).
- **Pace.** The game runs in real time: trees regrow and mobs respawn on
  timers. With all agents acting at once, a round is far shorter than in
  AgentWorld's turn-by-turn runner, so set `--min-round-seconds 20` to give
  the world comparable time per round.
- **Messages.** Private messages only reach their target. Each round an agent
  reads at most `--inbox-budget` (4) messages and sees at most
  `--contacts-budget` (6) contacts.
- **Ending.** The verifier runs from round 5, then every 3 rounds; a team
  stops early once its task is verified. `complete()` ends that agent only.
- **Arms** (`--arm`). These change only what your agent is shown, never its
  code:

  | Arm | `turn.bonds` | Which messages and contacts reach it |
  |---|---|---|
  | `base` | empty | newest first |
  | `hebbian` | the learned bond weights | ranked by bond (and recency) |
  | `shuffled` | bonds permuted at random each round | ranked by the permuted bonds |
  | `prompt_only` | the learned bond weights | newest first |
  | `oracle` | true team membership | ranked by team (worlds with several teams only) |

  The bond graph is updated from co-location, messages, item transfers, joint
  combat, deaths and verifier progress (`src/hebbian/multichannel.py`, rates
  set by the `--hebbian-*` flags). It is computed and saved in every arm,
  `base` included, so you can analyse it either way.

### 4.4 Outputs

`<out>/episode_k/`:

| File | What |
|---|---|
| `summary.json` | per team: `success` and the verifier message; rounds, wall time, `tokens` (from your `ledger`) |
| `trajectories/*_trajectory.json` | AgentWorld's trajectory format, scored by the task's verifier |
| `messages.jsonl` | every DM and board post: round, sender, receiver, delivered |
| `events.jsonl` | transfers, crafts, joint combat, kills and deaths found by the harness (created at the first such event) |
| `replay/state.jsonl` | per round: positions, inventories, decisions, what each agent read |
| `hebbian_W.npy`, `true_team.npy` | the bond matrix at the end of the episode, and true teams |
| `videos/team_t00.mp4` | a replay video (`--video none|team|world|both`) |

Re-render a video afterwards:
`PYTHONPATH=src python -m agentworld.replay <out>/episode_1 --team t00`.

### 4.5 Several tasks in a row with your agent

`chains/run_mf_chain.py` plays a chain YAML (section 5.2) through this
harness. Each agent keeps its object and its bonds between steps, while the
game resets. It is wired to the MindForge agent. To chain your own agent,
copy it and replace the line that builds `cognition`:

```python
cognition = ChainCognition({aid: YourPerAgentObject(aid) for aid in agent_ids})
```

Each per-agent object needs `async decide(turn)`, and may have `on_reset()`
(called at every step) and a writable `name` (set to the agent's in-game
username for that step).

## 5. Your agent in AgentWorld's runner (routes A and A+)

### 5.1 Route A: one task, their protocol

AgentWorld documents this route itself:
`$AGENTWORLD_ROOT/docs/benchmark/custom-agents.md`, with
`examples/agents/custom_agent.py` as a starting point. In short:

- Your class inherits `BaseAgent` (usually through `OpenAIAgent`). Its
  constructor accepts the keywords `api_key`, `model`, `username`,
  `password`, `base_url` (the **game** URL), `dump_prompts` and `llm_params`.
- You override `_make_api_call(messages)`, which returns an OpenAI-shaped
  response, and `_extract_tool_calls(message)`. Keep
  `execute_single_tool_call` (one action per turn, plus the
  `[TOOL_CALL_INFO]` / `[TOOL_RESULT]` records the runner reads).
- In the agent YAML, set `provider: custom` and
  `llm.agent_class: your_module:YourClass`.

```bash
cd $AGENTWORLD_ROOT
# her.yaml = a copy of agents/configs/custom.example.yaml with agent_class: her_module:HerAgent
PYTHONPATH=/path/to/her/module python agents/run.py \
    --task data_v0.1_multi/v1.3_benchmark/task_01_magic_staff.yaml \
    --agent her.yaml --output runs/mine/smoke --no-split-screen
python benchmarks/score.py --suite main --trajectories runs/mine/smoke --output runs/mine/smoke/scores.json
```

### 5.2 Route A+: chains of tasks on their runner (`chains/`)

`chains/README.md` has the full description. A chain YAML lists tasks and
maps each task's slots to persistent agent ids (`a1`, `a2`, `a3`). Each agent
keeps a state file between tasks: bonds, its log of interactions and its
skill log. Three arms (`--arm reference|memory|hebbian`) change only what is
added to the system prompt.

To run your own `BaseAgent` subclass in a chain, give it the chain hooks
(direct messages, state file, prompt block) with the same wrapper our
adapters use:

```python
# chains/her_wired_agent.py
from wired_agent import _wired
from her_aw_agent import HerAgent          # your BaseAgent subclass (route A)

HerWiredAgent = _wired("HerWiredAgent", HerAgent, "HerAgent + chain hooks")
```

The wrapper passes `HerAgent.__init__` only the keywords it names, so give
`__init__` the explicit keyword parameters listed in 5.1, as AgentWorld's
`custom_agent.py` does. A bare `**kwargs` would receive nothing. The wrapper
also overrides `_execute_tool_call`, `_rebuild_tools` and
`_update_system_prompt`, and calls the base class's version of each, so
overrides of these in `HerAgent` keep working.

```yaml
# chains/configs/her_agent.yaml: copy wired_openai.yaml and change one line
    agent_class: her_wired_agent:HerWiredAgent
```

```bash
python chains/run_chain.py chains/pendant_staff_4.yaml \
    --agent chains/configs/her_agent.yaml --arm memory --seed 42 --hold-complete
```

Chains change AgentWorld's protocol in two ways, and you should report both
next to any comparison. Global chat is replaced by
`direct_chat(target, message)`. With `--hold-complete`, `complete()` only
takes effect once the step's verifier passes. Both are explained in
`chains/README.md`.

## 6. Serving a model

Every route accepts any OpenAI-compatible endpoint: vLLM, OpenRouter, OpenAI
and others. With vLLM:

```bash
vllm serve <hf model id> --port 8000 --max-model-len 16384 --enable-prefix-caching
```

Route A's reference agent sends native tool calls with
`tool_choice="required"`. vLLM then needs
`--enable-auto-tool-choice --tool-call-parser <the parser for your model>`,
or every call returns HTTP 400. With small models the native tool calls often
come back empty: `chains/configs/wired_json.yaml` (`WiredJSONAgent`) asks for
one JSON reply instead. Route B's template asks for JSON from the start.

## 7. Things that will bite you

- **Game ports.** Kaetram builds its config from `.env` files only and
  ignores the process environment. `hpc/agentworld/run_world.sh` (and the
  harness, when it starts the server) writes a `.env.aw<API_PORT>` file and
  selects it with `NODE_ENV`, so several servers can share one checkout. Two
  runs must never share one server: give each its own ports.
- **Windows.** Run the server and the Python side inside WSL. Native Windows
  Python crashes writing AgentWorld trajectories (cp1252 vs emoji) unless
  `PYTHONUTF8=1` is set. From Windows, reach a server running in WSL at
  `http://localhost:7031`; `127.0.0.1` is refused.
- **Instance ids.** Trees, rocks and mobs disappear when used up and come back
  with a new id. Only ids in the current observation are valid; an agent that
  saved an id in its plan retries a dead one over and over.
- **Early `complete()`.** Models call `complete()` long before the objective
  is met, although every task says not to. In route B this ends that agent;
  in chains, `--hold-complete` keeps it in the task until the step is
  verified.
- **Model size.** A 4B model (Gemma 4 E4B) solved the short hand-off tasks
  (01 magic staff, 05 beryl pendant) but none of the longer metallurgy
  chains. `chains/pendant_staff_4.yaml` is the chain that small models can
  play.
- **Videos** need `imageio-ffmpeg`. Use `--video none` to skip them.

## 8. Where things live

| Path | What |
|---|---|
| `examples/agentworld/` | **start here**: `my_agent.py` (template agents and the contract), `run_my_agent.py` (runner) |
| `src/agentworld/` | our harness: `scheduler.py` (round loop, `Turn`, `Decision`), `actions.py` (tool list and parsing), `comms.py` (messages, read policies), `executor.py` (AgentWorld's game tools, unmodified), `verify.py`, `progress.py`, `replay/`, `cli.py` (MindForge entry point) |
| `src/agentworld/agent.py` | the MindForge agent (beliefs, critic, curriculum, memory), a full `decide()` example |
| `src/hebbian/multichannel.py` | the bond learning rule |
| `chains/` | chains of tasks: on AgentWorld's runner (`run_chain.py`, `wired_agent.py`) and on our harness (`run_mf_chain.py`); `chains/README.md` |
| `hpc/agentworld/` | SLURM job scripts and container setup (written for Snellius; adapt the paths) and the pinned `AGENTWORLD_COMMIT` |
| `tests/test_aw_*.py` | offline tests; `tests/fake_kaetram.py` is the in-memory game server |
