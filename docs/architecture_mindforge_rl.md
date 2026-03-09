# MindForge + RL Layer Architecture

## Overview

The system combines **LLM-based cognitive agents** (MindForge) with a **modular RL fine-tuning layer** (MAPPO + LoRA). The RL layer is fully decoupled — toggled by a single `--rl` flag. When disabled, the system runs as a pure LLM prompting pipeline with zero overhead.

```
┌─────────────────────────────────────────────────────────────┐
│                    Craftium Environment                       │
│        PettingZoo ParallelEnv (OpenWorld Multi-Agent)         │
│   17 discrete actions · 3 progression tracks · N agents      │
└──────────────────────────┬──────────────────────────────────┘
                           │ obs, reward, done
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Main Loop                      │
│              (multi_agent_craftium.py)                        │
│                                                              │
│  For each agent, each step:                                  │
│    1. Get frame + reward from environment                    │
│    2. Agent selects action (MindForge pipeline or RL)        │
│    3. Execute action in environment                          │
│    4. Record reward → metric + RL buffer                     │
│    5. Periodically: MAPPO update / token-opt check           │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│   MindForge Agent    │  │   RL Layer (optional)        │
│  (custom_agent.py)   │  │   (rl_layer/rl_layer.py)     │
│                      │  │                              │
│  6+ LLM calls/step   │  │  LoRA-adapted local model    │
│  Zero gradients      │  │  Action head + Value head    │
│  RAG-based learning  │  │  Gradient-based learning     │
└──────────────────────┘  └──────────────────────────────┘
```

---

## MindForge Agent Pipeline

Each step, the agent runs this sequence of LLM calls:

```
                    ┌─────────────┐
                    │   Critic    │ ← Was last action successful?
                    │  (LLM call) │   Returns: success (bool), critique (str)
                    └──────┬──────┘
                           │
              success?─────┼─────failure?
              │            │            │
        Save skill    error_count++    │
        to ChromaDB        │            │
              │            │            │
              ▼            ▼            ▼
                    ┌─────────────────┐
                    │ Auto-Curriculum │ ← What task should I do next?
                    │   (LLM call)    │   Uses: role prompt, completed/failed
                    │                 │   tasks, Q&A about environment
                    └──────┬──────────┘
                           │ task
                           ▼
                    ┌─────────────────┐
                    │ Skill Retrieval │ ← Have I done something like this?
                    │  (LLM + ChromaDB│   Semantic search over past skills
                    │   vector store) │
                    └──────┬──────────┘
                           │ relevant skills
                           ▼
                    ┌─────────────────┐
                    │ Episodic Memory │ ← What happened in similar episodes?
                    │  (LLM + ChromaDB│   Retrieves + summarises past episodes
                    │   vector store) │
                    └──────┬──────────┘
                           │ episode summary
                           ▼
                    ┌─────────────────┐
                    │  Belief System  │ ← What do I believe about the world?
                    │  (4 LLM calls)  │   Perception, Partner, Interaction, Task
                    └──────┬──────────┘
                           │ beliefs
                           ▼
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────┐     ┌────────────────────┐
   │ Action Selection  │     │    RL Layer         │
   │   (LLM call)      │     │  (local inference)  │
   │                    │     │                     │
   │ Input: task,       │     │ Input: text context  │
   │ beliefs, skills,   │     │ Output: 1 of 17     │
   │ episodes, frame    │     │ discrete actions     │
   │                    │     │ + log_prob + value   │
   │ Output: JSON with  │     │                     │
   │ thoughts, action,  │     │ Used when --rl is   │
   │ communication      │     │ enabled             │
   └────────────────────┘     └─────────────────────┘
              │                         │
              └────────────┬────────────┘
                           │ action
                           ▼
                    ┌──────────────┐
                    │  Environment │
                    │    .step()   │
                    └──────────────┘
```

### Key insight: Roles only affect task generation

Roles (gatherer, hunter, defender) are injected **only** through the Auto-Curriculum prompt. The role prompt tells the curriculum LLM to prioritise certain tasks:

- **Gatherer** → "Focus on the Tools track: dig trees, mine stone, find iron..."
- **Hunter** → "Focus on the Hunt track: kill chickens, sheep, pigs..."
- **Defender** → "Focus on the Defend track: fight zombies, skeletons..."

All downstream decisions (beliefs, skills, action selection) follow from the generated task. The agent doesn't "know" its role directly — it receives role-appropriate tasks and acts on them.

---

## RL Layer Architecture

### When `--rl` is disabled (default)

The RL layer does not exist. No model is loaded. No buffers are allocated. The system is identical to vanilla MindForge.

### When `--rl` is enabled

```
┌─────────────────────────────────────────────────────────┐
│                      RL Layer                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Base Model (e.g. Qwen3.5-2B)                    │   │
│  │  + LoRA Adapter (per role: gatherer/hunter/def.)  │   │
│  │                                                    │   │
│  │  Input: tokenised text prompt                      │   │
│  │  Output: last hidden state → pooled (1, H)         │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│           ┌─────────┴─────────┐                         │
│           ▼                   ▼                          │
│  ┌─────────────────┐ ┌─────────────────┐                │
│  │   Action Head   │ │   Value Head    │                │
│  │ Linear(H → 17)  │ │ MLP(H → 256 →1)│                │
│  │                 │ │                 │                │
│  │ → action dist   │ │ → V(s) estimate │                │
│  │ → sample action │ │                 │                │
│  │ → log π(a|s)    │ │                 │                │
│  └─────────────────┘ └─────────────────┘                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Rollout Buffer                                   │   │
│  │  Stores per step: prompt, action_idx, log_prob,   │   │
│  │                   value, reward, done              │   │
│  │  Computes: GAE advantages, returns                 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Optimizer                                        │   │
│  │  Adam over: LoRA params + action head + value head│   │
│  │  (base model weights are frozen)                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Per-role LoRA adapters (inspired by MARFT)

Each role gets its own LoRA adapter on the shared base model. The base weights are frozen; only the low-rank matrices are trained.

```
Base Model (frozen)
    │
    ├── LoRA "gatherer" ← Agent 0, Agent 3, Agent 6...
    ├── LoRA "hunter"   ← Agent 1, Agent 4, Agent 7...
    └── LoRA "defender" ← Agent 2, Agent 5, Agent 8...
```

This means:
- Agents with the same role share learning (their LoRA updates are independent but start from the same init)
- Different roles can specialise their policy independently
- The base model's general knowledge is preserved

---

## Two Levels of Optimisation

### 1. Action-Level (MAPPO)

Runs every `--rl-update-interval` steps (default: 64).

```
Collect 64 transitions
    │
    ▼
Compute GAE advantages
    │  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    │  Â_t = Σ (γλ)^k · δ_{t+k}
    │
    ▼
PPO Clipped Update (4 epochs × mini-batches of 8)
    │
    │  Policy loss:  -min(r(θ)·Â, clip(r(θ), 1±ε)·Â)
    │  Value loss:   max(MSE(V_new, R), MSE(V_clip, R))
    │  Entropy bonus: H(π)
    │
    │  where r(θ) = π_new(a|s) / π_old(a|s)
    │
    ▼
Update LoRA weights + action head + value head
```

**What it learns**: Which of the 17 actions to take given the current context. It does NOT change how the agent reasons, communicates, or generates tasks — those still go through the LLM API (SGLang).

### 2. Token-Level (Agent-Decided)

Only runs when the **agent itself decides** it needs deeper training.

```
Every step (after hard guards pass):
    │
    ▼
Build learning belief prompt with:
  - Recent success rate (e.g. "20% (2/10)")
  - Reward trend ("declining")
  - Failure pattern ("7 consecutive failures on 'Mine stone'")
  - Recent actions taken
    │
    ▼
LLM call → Agent reasons about its own competence
    │
    ▼
Agent outputs: {"needs_training": true/false, "reason": "...", "skill_focus": "..."}
    │
    ├── false → Skip, log reason
    │
    └── true → Run token-level PPO
                │
                ▼
         Filter transitions by skill_focus
                │
                ▼
         Token-level PPO over full generated sequences
           - Per-token log-probs across entire response
           - Sequence-level advantage from reward
           - Updates LoRA + reasoning + communication patterns
```

**What it learns**: Not just which action, but how to reason (thoughts), what to communicate, and how to process context. This is deeper and more expensive than action-level.

**Hard guards prevent abuse**:
- Cooldown: minimum N steps between training events
- Minimum data: need enough transitions to learn from
- Minimum history: need a full window of outcomes before being asked

---

## Communication Between Agents

Communication is **free-form natural language**, generated by the LLM as part of the action response.

```
Agent 0 generates: {"action": "Dig", "communication": "Mining stone, you go hunt"}
    │
    ▼
Message added to shared communications list (capped at N-1 messages)
    │
    ▼
Agent 1 sees: "agent_0_gatherer: Mining stone, you go hunt"
    │
    ▼
This text is injected into:
  - Critic prompt (to judge cooperation-dependent tasks)
  - Auto-Curriculum (to inform task selection)
  - Belief System (perception, partner, interaction beliefs)
  - Action Selection (as context for the next action)
```

When RL is enabled with token-level optimisation, the communication tokens also receive gradient signal — meaning the agent can learn to produce more useful messages.

---

## Metrics & Evaluation

`CraftiumMetric` tracks everything:

| Category | Metrics |
|----------|---------|
| **Performance** | Cumulative return per agent, per-step reward history |
| **Progression** | Steps-to-milestone per track (Tools/Hunt/Defend), milestone events |
| **Specialisation** | Per-agent track reward fractions (does the gatherer actually gather?) |
| **Social** | Communication frequency, message log, social lift (comm vs no-comm) |
| **RL** | MAPPO update losses (policy, value, entropy), token-opt decisions (train/skip + agent's reason) |

Outputs saved to `run_metrics/`:
- `data.json` — all structured data
- `communication_log.json` — full message history
- `cumulative_returns.png` — return curves
- `milestones.png` — progression over time
- `specialization_index.png` — role adherence
- `communication_frequency.png` — message density
- `steps_to_milestone.txt` — text table
- `log.txt` — free-form event log

---

## File Structure

```
src/mindforge/
├── multi_agent_craftium.py          # Main loop, CLI args, orchestration
├── custom_agent.py                  # MindForge agent (pipeline orchestrator)
├── custom_environment_craftium.py   # Craftium PettingZoo adapter
│
├── agent_modules/
│   ├── action_selection.py          # LLM-based action selection
│   ├── auto_curriculum.py           # Task generation (role-specific)
│   ├── belief_system.py             # 4 belief types (perception/partner/interaction/task)
│   ├── critic.py                    # LLM judges task success
│   ├── episodic_memory_manager.py   # ChromaDB episode store + LLM summariser
│   ├── skill_manager.py             # ChromaDB skill store + retrieval
│   ├── llm_call.py                  # Unified LLM call with retry logic
│   ├── util.py                      # Model client, response formats, env var config
│   └── craftium_metric.py           # Evaluation metrics + RL event tracking
│
├── rl_layer/
│   ├── __init__.py                  # Exports RLConfig, RLLayer
│   ├── config.py                    # RLConfig dataclass (all hyperparameters)
│   ├── trajectory_buffer.py         # RolloutBuffer with GAE computation
│   ├── mappo.py                     # action_level_ppo_step + token_level_ppo_step
│   └── rl_layer.py                  # RLLayer orchestrator (model, LoRA, heads, training)
│
└── prompts/
    ├── system_prompt.txt            # Agent identity + action space
    ├── environment_prompt.txt       # 17 actions + 3 progression tracks
    ├── instruction_prompt_p2.txt    # Per-step instruction template
    ├── critic_prompt.txt            # Task success evaluation
    ├── curriculum_prompt.txt        # Default curriculum (fallback)
    ├── role_gatherer.txt            # Tools track curriculum
    ├── role_hunter.txt              # Hunt track curriculum
    ├── role_defender.txt            # Defend track curriculum
    ├── learning_belief.txt          # Agent self-assessment for token-opt
    ├── belief_system/               # Belief generation prompts
    │   ├── perception_beliefs.txt
    │   ├── partner_beliefs.txt
    │   ├── interaction_belief.txt
    │   └── update_context.txt
    └── ...                          # Other prompt files
```

---

## Running the System

### Without RL (pure LLM agents)
```bash
python multi_agent_craftium.py --num-agents 3 --episodes 10 --max-steps 500
```

### With action-level RL
```bash
python multi_agent_craftium.py --num-agents 3 \
  --rl --rl-model-path /path/to/Qwen3.5-2B \
  --rl-lora-rank 8 --rl-update-interval 64
```

### With agent-decided token-level optimisation
```bash
python multi_agent_craftium.py --num-agents 3 \
  --rl --rl-model-path /path/to/Qwen3.5-2B \
  --rl-auto-token-opt
```

### LLM configuration (via environment variables)
```bash
export LLM_BASE_URL=http://localhost:8000/v1   # SGLang server
export LLM_MODEL=Qwen3.5-2B
export LLM_API_KEY=no-key-needed
```

---

## Design Principles

1. **Decoupled**: `--rl` off = zero overhead. The RL layer is not imported, no model loaded, no buffers allocated.
2. **Single insertion point**: RL replaces only the action selection LLM call. Everything else (critic, curriculum, beliefs, skills, communication) is untouched.
3. **Agent autonomy**: Token-level optimisation is the agent's choice, not an external schedule. The agent assesses its own competence and requests training when needed.
4. **Shared base, specialised adapters**: Per-role LoRA adapters (from MARFT) allow role-specific policy learning while preserving the base model's general knowledge.
5. **Compatible with MindForge**: The full cognitive pipeline (beliefs, skills, episodic memory, curriculum) runs regardless of RL. RL augments action selection; it doesn't replace the reasoning infrastructure.
