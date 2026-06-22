# 08 — Cognitive Agent (CustomAgent + agent_modules)

**Source files:** `src/mindforge/custom_agent.py`, `src/mindforge/agent_modules/{action_selection, belief_system, critic, auto_curriculum, skill_manager, episodic_memory_manager, social_module, llm_call, local_model_client, util}.py`, `src/mindforge/prompts/`
**Paper sections:** LLM cognitive-layer architecture (agent modules, social-reasoning step), Eq. 6 inputs (W_ij rows + windowed deltas fed to the social module), App. D run protocol (temperature 0.7)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

`CustomAgent` (an autogen `BaseChatAgent`) is the per-agent cognitive loop. One instance per agent; the orchestrator (`multi_agent_craftium.py`, see 07-orchestrator.md) calls `on_messages` once per environment step and `on_reset` between episodes.

## 1. CustomAgent orchestration

### on_messages pipeline (`custom_agent.py:232-591`)

Ordered stages; cadence counters set at `custom_agent.py:254-256` (`run_beliefs = call_count % belief_interval == 0`, `run_critic = call_count % critic_interval == 0`; CLI defaults `--belief-interval 5`, `--critic-interval 20`, `multi_agent_craftium.py:75-78` — the constructor defaults of 1 are overridden at `multi_agent_craftium.py:1023-1024`).

| # | Stage | When | Anchor (custom_agent.py) |
|---|---|---|---|
| 1 | One-time vector-DB wipe (skills, episodes, curriculum) | first call only (`_initialized` guard) | 260-264 |
| 2 | Parse step reward; build `{milestone_event}` banner if reward >= 10 and not dead | every step | 26-41, 272-289 |
| 3 | Critic task-success check; result cached for skipped steps | every `critic_interval` steps | 293-318 |
| 4 | Reward override: milestone reward forces `success = True` regardless of critic | every step | 320-321 |
| 5 | On success: persist skill (templated, no LLM) | on success | 323-332 |
| 6 | `rl_layer.record_success` (token-opt self-trigger input) | if RL enabled | 337-339 |
| 7 | Episodic-memory write (`add_episode`) + mark summary dirty | on critic steps with an active task | 341-350 |
| 8 | Curriculum refresh: new task when success, no task, `error_count > 10`, or chamber changed | conditional | 351-390 |
| 9 | Parallel fetch: skill retrieval (task string as query) + episode summary (regenerated only when dirty) + belief updates (cached off-cadence) | beliefs every `belief_interval` steps | 393-447 |
| 10 | Social-module deliberation -> `social_directive` (falls back to raw `"Social bonds: ..."` string when module off) | every step entered; internally every `social_interval` | 449-475 |
| 11 | Beliefs-dict assembly: beliefs + reward/position/status + chamber whitelist (`_describe_chamber`) + chamber_state + milestones + social directive | every step | 147-164, 477-495 |
| 12 | RL path: build text `rl_prompt`, run thoughts-first LLM call, hand to `rl_layer.select_action(thoughts_prefix=...)` | if RL enabled | 497-569 |
| 13 | Non-RL path: single full-JSON action LLM call | otherwise | 571-585 |
| 14 | Store `last_response`, return `(content, error_count)` | every step | 590-591 |

Anti-hallucination grounding: `_CHAMBER_OBJECT_WHITELIST` + `_CHAMBER_FACTS` (`custom_agent.py:62-144`) render a per-chamber "VISIBLE HERE / ROOM FACTS" block into the `{current_chamber}` placeholder — the agent only carries facts for the room it stands in (see 05-five-chambers-world.md).

### on_reset semantics (`custom_agent.py:593-624`)

| Reset (working memory) | Preserved (long-term) |
|---|---|
| `auto_curriculum.current_task = None` (fresh Ch1 task generated) | Skill vector DB |
| `_last_chamber`, cached critic verdict, 20-step action/reward log | Episodic-memory vector DB |
| All beliefs (`BeliefSystem.reset`, `belief_system.py:50-61`) + cached beliefs | Hebbian bond matrix W (lives in the orchestrator's graph, untouched here) |

Without this, episode N+1 inherits episode N's end-of-run Ch5 context (observed: 6 vs 16 milestones). The `_initialized` guard prevents `current_task=None` from re-triggering the one-time DB wipe.

## 2. Module table

| Module | Role | Cadence | Key params / behavior | Anchor |
|---|---|---|---|---|
| `ActionSelection` | Final policy LLM call; also RL-mode thoughts+comm pre-call | every step | system prompt = `system_prompt.txt` with `environment_prompt.txt` embedded | action_selection.py:24-153 |
| `BeliefSystem` | 4 belief stores: perception (frame+comms+error), partner (per-teammate, from convo), interaction (task+convo), task (curriculum context) | every `belief_interval` (5) steps; cached otherwise | forgiving `load_belief` parser never retries; blank update never clobbers previous belief (`coerce_belief_text`) | belief_system.py:63-153; util.py:233-272 |
| `Critic` | LLM verdict `{reasoning, success, critique}` on current task; sees rendered 20-step action/reward history with `*` milestone flags | every `critic_interval` (20) steps; verdict cached between | reward >= 10 overrides to success; failure on a critic step increments `error_count` | critic.py:29-71; custom_agent.py:44-59, 320-334 |
| `AutoCurriculum` | Proposes next task; tracks completed/failed lists; ChromaDB context store | on success / no task / 10 errors / chamber change | validity filters: achievable-keyword allowlist + blocklist + chamber-fixture check (anvil->ch2, switch/cell->ch3, zombie->ch4, boss->ch5); invalid tasks replaced by role default and logged to `failed_tasks` as `[INVALID: ...]`; question/answer sub-pipeline exists but is disabled (`do_question_answers=False`, custom_agent.py:385) | auto_curriculum.py:29-76, 228-241 |
| `SkillManager` | Persists successful discrete actions as skills; per-agent ChromaDB (top-k 5, score >= 0.4) | on critic success | `add_skill` is templated — no LLM call (`skill_manager.py:129-147`); LLM path `generate_skill_description` exists but is unused by CustomAgent; retrieval queries with the raw task string (custom_agent.py:396) | skill_manager.py:67-200 |
| `EpisodicMemoryManager` | (task, beliefs, action, critique, success) episodes in ChromaDB; LLM summary into the action prompt | write on critic steps; summary regenerated only when dirty | `retrieve_episodes` fetches 2k candidates then re-ranks: ceil(0.7k) successes + rest failures (70/30), padded if a pool is short | episodic_memory_manager.py:57-159 |
| `SocialModule` | Hebbian-graph -> LLM social directive (Section 5) | every `social_interval` (8) steps | see Section 5 | social_module.py:72-214 |
| `llm_call` | Shared call wrapper: per-module log files, format-with-kwargs, JSON parse, retry | per call | retries on transport error (1 s sleep) and on parse failure; gives up after 5 retries -> `{}`; prompt body intentionally NOT logged (metadata only), response IS logged. Caveat: the transport-error retry (llm_call.py:135-144) drops `parse_check`/`parser` from the recursion — after one network error, subsequent attempts skip schema assertions | llm_call.py:73-172 |
| comm routing helpers | `normalize_agent_target` canonicalizes `agent0`/`Agent_1` -> `agent_N` (prevents double-counted contributors) | per response | unparseable targets passed through for orchestrator fallback routing | util.py:31-54 |

ChromaDB persistence lives under `/tmp/mindforge_$SLURM_JOB_ID` on HPC (SQLite locks fail on Lustre) — `skill_manager.py:23-39`; embeddings via sentence-transformers `all-MiniLM-L6-v2` (`util.py:26`).

## 3. Model factory (`util.py:316-340`)

`create_model_client(response_format)` — every module gets its own client pinned to its pydantic schema:

| Path | Trigger | Client | Notes |
|---|---|---|---|
| Local in-process | `LLM_MODEL_PATH` set | `LocalModelClient` (HF transformers, singleton model per process; auto-detects vision models) | `temperature=0.7` default (`local_model_client.py:312`); HPC/thesis path |
| Remote | otherwise | `OpenAIChatCompletionClient` | `LLM_BASE_URL` (default OpenRouter) + `LLM_MODEL` (default `google/gemini-2.5-flash`, `util.py:22-24`); no temperature pinned |

> PAPER MISMATCH — see PAPER_INCONSISTENCIES.md #5 (0.7 consistent on local path only; remote fallback unpinned) and #8 (base model is config-driven; Qwen via `LLM_MODEL_PATH` on HPC).

## 4. Action-selection contract

Non-RL path — one call returning `AgentResponse` (`util.py:59-64`):

```json
{"thoughts": str, "action": str, "communication": str, "communication_target": str}
```

User prompt = `instruction_prompt_p2.txt` + per-step observation text; `communication_target` is normalized before return (`action_selection.py:73-76`).

RL path — two stages (`custom_agent.py:527-569`):
1. `generate_thoughts_and_comm` (`action_selection.py:79-153`) uses `instruction_prompt_p2_thoughts.txt` — same context, JSON **without** the `action` field (`TargetedCommunicationResponse`, `util.py:66-80`; `communication_target` is a required string, so a recipient is always chosen).
2. `rl_layer.select_action(rl_prompt, thoughts_prefix=thoughts)` appends `"Thoughts: ...\nAction:"` and scores the 14 candidate actions by constrained-generation sequence log-prob (`src/rl_layer/rl_layer.py:176-246`) — chain-of-thought ordering: action sampled from p(a | prompt, thoughts). See 03-rl-layer.md.

The pre-generated thoughts/communication/target are merged back into the RL result dict (`custom_agent.py:565-569`), so both paths return the same 4-field shape to the orchestrator.

## 5. Social module — the Hebbian->LLM coupling

This is the thesis-relevant bridge: it converts the agent's row of the bond matrix W into an explicit social directive that the action LLM conditions on. Requires `--hebbian`; enabled via `--social-module {none,prompt,bias}` (default `none`; enforcement at `multi_agent_craftium.py:2713-2721`).

**Cadence.** `--social-interval` default **8** (`multi_agent_craftium.py:248`; the constructor default of 1 in `social_module.py:86` is overridden at instantiation, `multi_agent_craftium.py:447-451`). `deliberate` runs on the first call unconditionally, then every Nth call; off-cadence calls return the cached `last_thought` with no LLM call (`social_module.py:118-124`).

**Inputs** (assembled in the orchestrator per step, `multi_agent_craftium.py:1443-1478`, cached against the graph's step counter):

| Input | Content | Source |
|---|---|---|
| `bond_weights` | own row W_i: `{agent_j: w_ij}` for all teammates | `hebbian_graph.get_weight(i, j)` |
| `bond_deltas` | windowed Δw over the last 50 steps | `hebbian_graph.bond_delta_row(i)` (`src/hebbian/graph.py:111-112, 753`) |
| `incoming` | messages routed to this agent this step | comm inbox |
| self-state | last action, raw reward text, inventory, position | `custom_agent.py:461-471` |

The (w, Δ) pair is rendered as a table with verbal tags — `STRENGTHENING / DECAYING / STABLE` at |Δ| > 0.02 — so the LLM doesn't do sign arithmetic (`social_module.py:38-63`).

**Output schema** — `SocialThought` (`util.py:115-134`):

| Field | Meaning |
|---|---|
| `bond_change_explanation` | dict teammate -> one-line WHY the bond is moving (interpretability artifact) |
| `reasoning` | 1-3 sentences tying bonds + deltas + messages to the decision |
| `referenced_bonds` | dict teammate -> weight the LLM claims it actually used — verifies the graph drove the decision |
| `ask_target` / `ask_message` | asker side: highest-bond plausible helper + suggested request text (or null) |
| `respond_to` | responder side: incoming senders the agent will help (prompt advises ignoring w < 0.2 senders) |
| `confidence` | 0-1, default 0.5 |

Failed calls degrade to an empty thought; non-dict/list fields are coerced (`social_module.py:145-161`). The parsed thought is cached in `self.last_thought` so the directive renderer and the routing layer read the **same** deliberation without a re-call.

**Coupling 1 — prompt** (`--social-module prompt` and also active under `bias`): `render_directive` (`social_module.py:166-214`) turns the thought into a text block — Reasoning / Outgoing ("Ask agent_X for help... put agent_X in your communication_target field") / Incoming / Bond changes — injected as `{social_directive}` into `instruction_prompt_p2.txt` (placeholder at line 24) via the beliefs dict (`custom_agent.py:472-481`). The action LLM may still ignore it.

**Coupling 2 — bias** (`--social-module bias`): at the message-routing site the sender's cached `last_thought["ask_target"]`, if it parses to a valid other-agent index, **overwrites** the action LLM's `communication_target`; routing source logged as `"social_bias"` (`multi_agent_craftium.py:1788-1815`). This makes the bond-weighted pick a hard routing guarantee rather than a suggestion. A third "override" variant (task rewriting) was deliberately not implemented (`social_module.py:12-18`).

When the module is off, the action prompt falls back to the legacy raw string `"Social bonds: agent_1 (hunter): 0.42, ..."` (`custom_agent.py:450`; built at `multi_agent_craftium.py:1449-1456`). Bond mechanics (eta_0, eta_plus, eta_minus, lambda, R) are in 02-hebbian-graph.md; comm rewards in 06-rewards.md.

## 6. Prompt map (`src/mindforge/prompts/`)

| Template | Consumer | Purpose |
|---|---|---|
| `system_prompt.txt` | ActionSelection (system msg) | agent persona + rules; embeds environment_prompt |
| `environment_prompt.txt` | embedded into system_prompt (action_selection.py:31) | world/action-set description |
| `instruction_prompt_p2.txt` | ActionSelection.select_action | per-step policy prompt (full JSON incl. action) |
| `instruction_prompt_p2_thoughts.txt` | ActionSelection.generate_thoughts_and_comm | RL-mode variant: thoughts+comm only, no action field |
| `belief_system/perception_beliefs.txt` | BeliefSystem.create_perception_beliefs | beliefs from current frame |
| `belief_system/partner_beliefs.txt` | BeliefSystem.update_partner_beliefs | per-teammate model from conversation |
| `belief_system/interaction_belief.txt` | BeliefSystem.update_interaction_beliefs | task-conversation interaction state |
| `belief_system/update_context.txt` | BeliefSystem.update_task_beliefs | task-context refresh |
| `critic_prompt.txt` / `critic_info.txt` | Critic.check_task_success (system/user) | task-success verdict |
| `curriculum_prompt.txt` / `curriculum_info.txt` | AutoCurriculum.get_new_task (system/user) | next-task proposal |
| `curriculum_questions.txt` / `curriculum_answer.txt` | AutoCurriculum Q/A sub-pipeline | exploration Q/A (disabled in main loop) |
| `skill_description_prompt.txt` / `skill_description_info.txt` | SkillManager.generate_skill_description | LLM skill naming (unused by CustomAgent) |
| `skill_construct_query.txt` | SkillManager.construct_query | retrieval-query builder (unused by CustomAgent) |
| `episode_summary_prompt.txt` | EpisodicMemoryManager.generate_episode_summary | compress retrieved episodes |
| `social_module.txt` | SocialModule.deliberate | bond-table -> SocialThought deliberation |
| `learning_belief.txt` | `src/rl_layer/token_opt.py:33` | token-opt "should I train now" self-trigger (see 03-rl-layer.md) |
| `role_agent.txt`, `role_harvester.txt`, `role_hunter.txt`, `role_scouter.txt` | **none — orphaned** | role personas; only the ROLE_NAMES label strings survive (multi_agent_craftium.py:37) |
