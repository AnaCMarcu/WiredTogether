# RL Approaches for LLM-Based Multi-Agent Policy

All approaches below are compatible with the Hebbian social graph, which operates at the reward level (diffuses rewards before the RL update) and is independent of the update algorithm.

---

## 1. REINFORCE with Baseline

Update LoRA once per episode using `advantage = reward - mean(rewards)`.

```
loss = -mean(advantage_i * log_prob(action_i | prompt_i))
```

**Pros**
- Trivial to implement, no value head needed
- LLM retains full reasoning ability — actions still come from LLM
- Works with any reward signal including Hebbian-diffused rewards
- Single backward pass per episode

**Cons**
- High variance — needs many episodes to converge
- No credit assignment within an episode (every action gets the episode return)

---

## 2. Reward-Weighted Regression (RWR)

Same as REINFORCE but only train on positive-reward episodes — ignore episodes where reward ≤ 0.

```
loss = -mean(reward_i * log_prob(action_i | prompt_i))   # only where reward_i > 0
```

**Pros**
- Even simpler than REINFORCE, no baseline needed
- Never pushes the LLM toward bad behaviors
- Safe to run continuously

**Cons**
- Very slow early when most episodes give zero reward
- Ignores useful signal from negative outcomes (e.g. death)

---

## 3. Token-Level PPO (current `mode="token"`)

LLM generates action text normally. Optimize the log probability of the chosen action token using PPO clipping and a value head for variance reduction.

**Pros**
- Gradient flows into the LLM's actual decision token
- PPO clipping prevents large destabilizing updates
- Established algorithm with known convergence properties

**Cons**
- Log prob of a single token is noisy signal
- Still requires a value head for advantage estimation (GAE)
- Full LLM forward + backward every step — expensive
- Value head adds complexity and training instability

---

## 4. GRPO (Group Relative Policy Optimization)

For each situation, sample K actions from the LLM, rank by reward within the group, use rank-normalized reward as advantage. No value head — baseline comes from the group itself.

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

Used in DeepSeek-R1 for reasoning tasks.

**Pros**
- No value head — variance reduction from group comparison
- Natural fit for discrete action spaces
- Gradient signal is relative (which action was better), not absolute

**Cons**
- Requires K LLM calls per step (K=4–8 typically) — K× slower
- Needs environment to be resettable to the same state to compare K actions — hard in Minetest
- Not straightforward with a slow stochastic environment

---

## 5. Advantage-Weighted Regression (AWR)

Collect a replay buffer of (prompt, action, reward) tuples across multiple episodes. Fine-tune LoRA in batch with exponentially weighted advantages:

```
loss = -mean(exp(advantage_i / β) * log_prob(action_i | prompt_i))
```

The temperature `β` controls aggressiveness: low β → conservative updates, high β → heavily upweights best transitions.

**Pros**
- Off-policy: can reuse experience across many episodes — very sample efficient
- No value head needed
- Exponential weighting naturally amplifies rare high-reward transitions
- Works well with sparse rewards — the occasional +128/+256 stage reward gets heavily amplified
- Batch updates are stable and easy to tune

**Cons**
- Requires careful tuning of β (too low = ignores reward, too high = overfits rare transitions)
- Off-policy: stale transitions from old policy may mislead updates if policy changes fast
- Replay buffer needs memory management

---

## 6. In-Context RL (no gradient)

Add successful past episodes to the LLM's context as few-shot examples: "last time I did LookDown → Dig → MoveForward → Dig I got +128". The LLM generalizes from its own history without any weight update.

The `episode_summary` system already exists in this project and partially implements this.

**Pros**
- Zero training cost, no LoRA updates, no backward passes
- Immediate effect — LLM adapts within the same run
- Naturally compatible with everything (Hebbian, curriculum, LoRA)
- Can be combined with any other approach

**Cons**
- Context window limits how much history fits
- No persistent learning — resets between runs unless summaries are saved to disk
- Relies on LLM being capable enough to generalize from examples

---

## 7. Hindsight Experience Replay (HER) + REINFORCE

When an episode fails its assigned goal, relabel it as if the outcome it *did* achieve was the intended goal. Feed relabeled episodes into REINFORCE.

Example: agent was supposed to mine stone but instead dug 2 trees → relabel as "goal: dig 2 trees" and treat as success.

**Pros**
- Dramatically improves sample efficiency for sparse rewards
- Directly addresses the core problem: agents rarely reach stage rewards
- Every episode produces useful signal, even failed ones
- Compatible with LoRA fine-tuning and Hebbian graph

**Cons**
- Requires a goal-conditioned policy (prompt must be reformatted per relabeling)
- More complex infrastructure: need to track what actually happened vs what was intended
- Relabeled goals must be valid and coherent to avoid confusing the LLM

---

## Recommendation

Given the project's constraints — sparse rewards, slow Minetest environment, Hebbian graph already handling social credit assignment, LLM doing rich reasoning — the best path is:

| Priority | Approach | Reason |
|----------|----------|--------|
| Short term | **AWR (option 5)** | Collect replay buffer across episodes, heavily upweight rare stage rewards, batch LoRA update. Most sample-efficient for sparse rewards. |
| Free wins | **In-context RL (option 6)** | The `episode_summary` system already exists. Extend it to include successful action sequences. Zero cost. |
| Future | **HER (option 7)** | Once AWR is stable, add hindsight relabeling to extract signal from every episode regardless of outcome. |

REINFORCE (option 1) is a good baseline to validate the LoRA update loop before moving to AWR.
