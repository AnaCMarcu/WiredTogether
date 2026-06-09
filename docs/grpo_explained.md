# How GRPO with RLVR works in this codebase

A walkthrough of the algorithm as it's actually wired up here — what data
flows where, what gets optimized, what stays frozen. For the implementation
plan and design rationale see [`rlvr_grpo_plan.md`](rlvr_grpo_plan.md).

---

## The big idea in one paragraph

**RLVR (Reinforcement Learning with Verifiable Rewards)** = the reward comes
from deterministic, programmatic checks — milestone fires, JSON-validity,
survival — not from a human-feedback model or a learned critic.
**GRPO (Group Relative Policy Optimization)** = instead of comparing each
rollout to a learned baseline (the PPO critic), you compare it to the mean
reward of a *group* of G sibling rollouts that started in comparable states.
The policy update is the PPO clipped surrogate, but the advantage signal is
`(r_i − mean(r)) / std(r)` instead of `r_i − V(s)`. RLVR provides the
reward; GRPO provides the gradient.

---

## Where the pieces live

```
one GRPOTrainer.step()  =  one gradient update

  1. MultiAgentRolloutSampler.sample_joint_group()
       → collect G joint rollouts
       → src/rlvr/rollout_sampler.py

  2. FiveChambersVerifier.score_joint_group()
       → assign a scalar reward to each per-agent trajectory
       → src/rlvr/verifier.py

  3. assemble_(composed_)multi_agent_batch()
       → group-relative advantages
       → src/rlvr/grpo_trainer.py

  4. _update(batch)
       → clipped surrogate + KL penalty + Adam step
       → src/rlvr/grpo_trainer.py
```

---

## Step-by-step with concrete numbers

Defaults from `configs/rlvr/grpo_hebbian_full.yaml`:
**G = 4** trajectories per group, **H = 50** env steps per trajectory,
**N = 3** agents trained, **ε = 0.2** clip, **β = 0.05** KL coefficient,
**lr = 5e-6**.

### Step 1 — Sample G joint rollouts

`MultiAgentRolloutSampler.sample_joint_group()` keeps rolling joint episodes
until G of them share the same equivalence class (same chamber + same
2-block position bucket for every trained agent).

A **joint rollout**:

```
for t in 1..50:
    for aid in [0, 1, 2]:
        action_text, tokens, logprobs = policy.generate(prompt(obs[aid], info[aid]))
        # LLM emits JSON: {"action": "dig", "communication_target": null, "thoughts": "..."}
        action_dict = parse_action_json(action_text)
        # tokens/logprobs are captured ONLY for trained agents (here: all 3)

    obs, reward, done, info = env.step({aid: action_dict[aid] for aid in [0,1,2]})

    # Stage 4a: Hebbian graph observes every env step
    hebbian_bridge.observe_step(positions, step_rewards, comm_events)

    # Early-terminate if any trained agent fires a milestone, dies, or
    # changes chamber. Keeps the trajectory focused on the "interesting
    # moment" so a single milestone fire isn't diluted by 47 no-op steps.
    if any_milestone_fire or any_death or any_chamber_transition:
        break
```

Result: one `JointRollout` =
`{0: (traj_0, tokens_0), 1: (traj_1, tokens_1), 2: (traj_2, tokens_2)}`.
Repeat until 4 joint rollouts share the same `joint_prompt_id`.

**Why "comparable" matters:** the position-bucket key is
`"ch3:x0_z0"` for a 2-block grid. Two joint rollouts only go into the same
group if all 3 agents started in the same (chamber, 2-block bucket).
Without comparability you can't normalize.

### Step 2 — Score each trajectory (RLVR)

`FiveChambersVerifier.score_joint_group(joints, team_reward=False)` walks
every per-agent trajectory and assigns a scalar reward. **This is where
RLVR lives.** No model. No learning. Pure functions of what the env
emitted.

Per trajectory:

```
reward = milestone_reward + format_reward + alive_bonus
```

Concrete example for agent 0 in joint 2:

- Milestone events:
  `[{"step": 12, "agent_id": 0, "milestone_id": "m17_switch_pressed"}]`
  → lookup in `TRACKS["ch3_switches"]` → **+40.0**
- Format reward: 10/12 LLM outputs were fully valid JSON, 2 were partial →
  sum `[1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0] = 11.0`
  × `weight=0.1` → **+1.1**
- Alive bonus: agent 0 didn't die in event_log → **+5.0**
- **Total: 46.1**

Repeat for all G·N = 4·3 = 12 trajectories. Result:
`list[dict[agent_id, reward]]` of length 4.

**(Stage 4a) Hebbian reward diffusion** runs after the per-agent scores
are computed: within each joint, spread reward across bonded teammates via
`(1-γ)·r_i + γ·Σ w̄_ij·c_ij·r_j` (γ=0.2). If agent 0 fires a milestone and
is strongly bonded to agent 1, agent 1 receives a fraction of that reward.

### Step 3 — Group-relative advantage

This is **the only thing that makes GRPO different from PPO.** No critic,
no value function, no GAE.

```python
# For each trained agent, collect its 4 rewards across the 4 joints:
agent_0_rewards = [46.1, 5.1, 5.1, 5.1]   # only joint 0 fired a milestone
agent_1_rewards = [5.1, 5.1, 5.1, 5.1]    # never fired
agent_2_rewards = [5.1, 25.1, 5.1, 5.1]   # joint 1 fired m_comm_ch3 (+20)

# Group-relative advantages — z-score within the group:
agent_0_advs = (rewards - mean) / (std + ε)
            = ([46.1, 5.1, 5.1, 5.1] - 15.35) / 17.75
            = [+1.73, -0.58, -0.58, -0.58]

agent_1_advs = [0.0, 0.0, 0.0, 0.0]      # std=0 → no signal
agent_2_advs = [-0.5, +1.5, -0.5, -0.5]
```

The advantage is **constant within a trajectory** — every token of agent
0's joint-0 response gets the same advantage `+1.73`. This is
"whole-trajectory credit assignment."

Why this works: agent 0 in joint 0 did better than its three sibling
rollouts in the same starting state. The advantage `+1.73` tells the
optimizer "make this kind of response more likely in this kind of state."
No critic needed — the comparison is to other actual rollouts, not to a
learned prediction.

**(Stage 4b)** When `hebbian_group_composition=True`, agent 0's group of 4
is **3 own + 1 borrowed from a teammate**, with selection probability
proportional to W̄[0, :]. Borrowed trajectories get `origin_agent` tagged.
The advantage normalization runs over the mixed group. This is the
"social replay" mechanism — if agent 1 is strongly bonded to agent 0,
agent 0's update sees one of agent 1's recent rollouts as a comparable
peer.

### Step 4 — Update the LoRA

For each of the 12 `ScoredTrajectory` items, the trainer computes a
per-token loss and averages. Following the
[DeepSeek-R1 GRPO formulation](https://arxiv.org/abs/2501.12948):

```python
new_logprobs = model.logprobs(prompt, response_tokens)    # under grpo_policy adapter, WITH grad
old_logprobs = trajectory.response_logprobs               # captured at rollout time, frozen
ratio        = exp(new_logprobs - old_logprobs)           # per token

# PPO clipped surrogate — bounds the policy update's step size
unclipped = ratio * advantage
clipped   = clamp(ratio, 1-ε, 1+ε) * advantage
surrogate = min(unclipped, clipped)                       # per token

# KL penalty — keeps the policy near the frozen reference
ref_logprobs = reference.compute_kl(...)                  # under grpo_reference adapter, no grad
kl_term      = new_logprobs - ref_logprobs                # per token

# Per-trajectory loss
loss_traj = -surrogate.mean() + β * kl_term.mean()

# Average across all 12 trajectories
total_loss = mean(loss_traj for s in batch)
total_loss.backward()
optimizer.step()    # updates grpo_policy LoRA only
```

The **reference adapter** is a frozen snapshot of the policy adapter at
training start (`ReferencePolicy.__init__` copies the initial weights and
sets `requires_grad=False`). The KL penalty pulls the trained policy back
toward the reference, preventing degenerate "exploit the reward function"
solutions where the model produces unusual outputs that happen to score
well but break elsewhere.

### Step 5 — Persist metrics + checkpoint

Every step appends one JSON record to `grpo_metrics.jsonl`:

```json
{"step": 42, "group_size": 12, "group_mean_reward": 11.83,
 "milestone_fire_rate": 0.083, "borrowed_fraction": 0.25,
 "per_agent_reward": {"0": 15.35, "1": 5.1, "2": 10.1}}
```

Every 100 steps: save the policy LoRA adapter to `grpo_lora/step_NNNNNN/`.

---

## What is actually optimized

**One thing only: the `grpo_policy` LoRA adapter weights.**

Concretely, two small matrices per attention layer (rank 16, on `q_proj`
and `v_proj` — see `GRPOModelConfig`). For a 2B model with that LoRA
config, this is roughly **2–5 million trainable parameters** out of ~2
billion total — about 0.1–0.3% of the model.

### What's frozen (NOT optimized)

| Component | State |
|---|---|
| Base LLM weights (Qwen3.5-2B) | frozen by PEFT — `requires_grad=False` |
| `grpo_reference` LoRA adapter | frozen at init by `ReferencePolicy._freeze_adapter` — snapshot of `grpo_policy`'s starting weights, used only to compute KL |
| Hebbian graph weight matrix `W` | has its **own** plasticity update rule (`graph.update()`), not part of the GRPO gradient — separate learning channel |
| Verifier (`FiveChambersVerifier`) | pure function, no parameters |
| Tokenizer | static |
| Env | static |

### The optimizer setup

From `GRPOTrainer.__init__`:

```python
trainable = [p for p in self.model.model.parameters() if p.requires_grad]
self.optimizer = torch.optim.Adam(trainable, lr=5e-6)
```

Adam runs **only** on parameters where `requires_grad=True` — which after
`ReferencePolicy._freeze_adapter` is precisely the `grpo_policy` LoRA
matrices.

### What this means in practice

You're **not** training the LLM. You're training a tiny adapter that
nudges the LLM's existing policy toward Craftium-useful behavior. That's
why:

- It's fast (small parameter count) — each step is ~600 forward passes of
  a 2B model, ~5 min on A100, and only the adapter weights update
- It's reversible — just don't load the adapter and you have the base
  model back
- The reference adapter exists — it's the baseline you're nudging away from
- The KL penalty matters — without it, the small adapter would overfit
  hard to milestone-firing-shaped outputs and lose the base model's
  coherence

---

## Why GRPO beats PPO+critic on Craftium (the bet you're making)

| | PPO + critic | GRPO |
|---|---|---|
| Reward signal | Same milestone events | Same milestone events (RLVR) |
| Advantage estimation | `A = r − V_φ(s)` where V_φ is learned | `A = (r − mean) / std` over G siblings |
| Critic quality | Audit showed `explained_variance < 0.1` on five-chambers — V_φ is basically predicting the mean, so A ≈ r − mean(r) anyway, but with extra learning noise | The group-relative form is **exactly** "r − mean(r)" but computed over a small comparable sample at no learning cost |
| Failure mode | Critic learns to predict mean, advantages collapse, policy stops moving | Group-mean can degenerate when std≈0 (every trajectory got the same reward), but the ε in `std + ε` keeps the math well-defined and the policy just doesn't move on that group |
| Hebbian fit | Social-replay needs an IS correction (`π_i / π_j`) for cross-agent borrowing; legacy zeroes `social_replay_rho` because of this | Group composition is a natural plug-in: borrowed trajectories enter the group's reward distribution, advantage is normalized in the same way, the math doesn't break (in shared-LoRA mode it's exactly on-policy) |

---

## The full picture in one diagram

```
┌─ env (Craftium / Five Chambers) ─────────────────────────────────────────┐
│                                                                          │
│   ┌─ MultiAgentRolloutSampler ────────────────────────────────────────┐ │
│   │   collect G=4 joint rollouts (3 agents × ≤50 steps each)           │ │
│   │   per env step → Hebbian bridge updates W (Stage 4a)               │ │
│   └──────────────┬─────────────────────────────────────────────────────┘ │
│                  │  list[JointRollout]                                    │
│                  ▼                                                        │
│   ┌─ FiveChambersVerifier ────────────────────────────────────────────┐ │
│   │   per (joint, agent):                                              │ │
│   │     r = sum(milestone_rewards) + format·0.1 + alive·5.0            │ │
│   │   optionally diffuse across bonded teammates (Stage 4a)            │ │
│   └──────────────┬─────────────────────────────────────────────────────┘ │
│                  │  list[dict[aid, reward]]                               │
│                  ▼                                                        │
│   ┌─ assemble_composed_multi_agent_batch (or assemble_…) ─────────────┐ │
│   │   per agent i:                                                    │ │
│   │     own  = K samples from per_agent_buffer[i]                      │ │
│   │     borr = (G-K) samples from teammates, P ∝ W̄[i,:] (Stage 4b)    │ │
│   │     advantages = z-score(rewards) within agent_i's group of G     │ │
│   │   flat batch = [G·N ScoredTrajectory]                             │ │
│   └──────────────┬─────────────────────────────────────────────────────┘ │
│                  │  list[ScoredTrajectory]                                │
│                  ▼                                                        │
│   ┌─ GRPOTrainer._update ──────────────────────────────────────────────┐ │
│   │   per trajectory:                                                 │ │
│   │     new_lp = grpo_policy(prompt, tokens)        # WITH grad        │ │
│   │     old_lp = trajectory.response_logprobs       # FROZEN           │ │
│   │     ref_lp = grpo_reference(prompt, tokens)     # FROZEN, no grad  │ │
│   │     loss = -clipped_surrogate(new_lp, old_lp, A) + β·KL(new,ref)  │ │
│   │   loss.mean().backward(); optimizer.step()                         │ │
│   │   ⇒ updates grpo_policy LoRA only                                  │ │
│   └──────────────┬─────────────────────────────────────────────────────┘ │
│                  │  GRPOStepMetrics                                       │
│                  ▼                                                        │
│        write step record to grpo_metrics.jsonl                            │
│        every 100 steps: save grpo_lora/step_NNNNNN/                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The math in three lines

For trajectory `i` in agent_j's group of G:

1. **Reward** (RLVR, deterministic):
   `r_i = Σ milestone_rewards + 0.1·Σ format_score + 5·alive`

2. **Advantage** (group-relative, no critic):
   `A_i = (r_i − μ_group) / (σ_group + ε)`

3. **Loss** (clipped PPO surrogate + KL to frozen reference):
   `L_i = −E_t[min(ρ_t·A_i, clip(ρ_t, 1±0.2)·A_i)] + 0.05 · E_t[KL(π_θ ∥ π_ref)]`
   where `ρ_t = π_θ(o_t|s_t) / π_θ_old(o_t|s_t)` is the per-token policy ratio.

That's the whole algorithm. Everything else — the env adapter, the prompt
template, the Hebbian additions, the milestone tables, the metrics, the
comparison plots — is plumbing around those three lines.
