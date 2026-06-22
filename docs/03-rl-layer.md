# 03 — RL Layer (LoRA-MAPPO over an LLM Actor)

**Source files:** `src/rl_layer/{config.py, rl_layer.py, trajectory_buffer.py, ppo_update.py, ippo.py, centralized_critic.py, heads.py, token_opt.py, persistence.py, __init__.py}`
**Paper sections:** §B (RL formulation, Table 4 action space), Table 6 (PPO/LoRA hyperparameters), App C.1 (social replay), Eq. 7 (replay weighting)
**Verified at commit:** 52bb302 (wired_final) + post-commit fixes from this verification (6 metrics/analysis-layer bug fixes - see PAPER_INCONSISTENCIES.md #14).

The whole layer is gated by `RLConfig.enabled` (default `False`, config.py:12); when off, every public method is a no-op and agents fall back to plain LLM action selection. Wiring into the per-step loop is described in 07-orchestrator.md; the Hebbian graph it interacts with is in 02-hebbian-graph.md.

## 1. Architecture: one frozen base, per-agent LoRA

| Aspect | Value | Anchor |
|---|---|---|
| Base LLM | ONE `AutoModelForCausalLM` shared process-wide via class attrs `_shared_model/_shared_tokenizer` | rl_layer.py:69-72, 374-423 |
| Base weights | Frozen; loaded from `config.model_path` (same weights SGLang serves), fp16 (`dtype="float16"`) | config.py:16,22; rl_layer.py:403-409 |
| LoRA | rank 8, alpha 16, dropout 0.05, targets `q_proj`+`v_proj`, bias none, CAUSAL_LM | config.py:17-19; rl_layer.py:411-418, 457-464 |
| Adapter per agent | One named adapter per role registered on the shared PeftModel (`add_adapter` / `load_adapter` if checkpoint exists); idempotent | rl_layer.py:425-465 |
| Adapter switching | `model.set_adapter(self._adapter_name)` before every forward (`_score_actions`, `_encode_prompt`, save) | rl_layer.py:525, 611; persistence.py:46 |
| Optimizer isolation | Adam over params whose name contains `.<adapter>.` + this agent's value head only — agent i's step never touches agent j's adapter | rl_layer.py:120-131 |
| Grad checkpointing | Enabled once on the shared base (guard flag `_grad_ckpt_enabled`) | config.py:35; rl_layer.py:109-114 |
| Memory effect | 3-agent Qwen-9B: 3×18 GB → 1×18 GB + 3 ~1 MB adapters + 3 value heads | rl_layer.py:62-66 |

Prompts are truncated to `rl_prompt_max_tokens = 256` for all RL forwards (config.py:34).

> PAPER MISMATCH — base model is config-driven, not hard-pinned to Qwen — see PAPER_INCONSISTENCIES.md #8

## 2. Constrained-decoding policy (`_score_actions`)

The actor has **no classifier head**: the policy distribution is the LLM's own sequence log-probability of each candidate action string as a prompt continuation.

- **Candidate set:** the 22-action tuple (config.py:70-76) minus the 8 `Slot*` actions when `mask_slot_actions=True` (default) → **14 candidates**. Membership in `_candidate_actions` *is* the masking; built once in `_build_candidate_actions` (rl_layer.py:467-480), pre-tokenized with a leading space (rl_layer.py:153-159).
  > PAPER MISMATCH check #7 (slot masking → 14 candidates) — CONSISTENT, see PAPER_INCONSISTENCIES.md #7
- **Scoring** (rl_layer.py:482-603): ONE batched forward over C right-padded rows `[prompt + cand_i]`; per-token log-prob at the candidate positions via gather − logsumexp (fp32), summed per row → `cand_log_probs (C,)`. A KV-cache variant was abandoned because grad-checkpointing silently disables `use_cache`, making tail forwards score candidates without prompt context (rl_layer.py:493-507).
- **Sampling** (`select_action`, rl_layer.py:176-248): `Categorical(logits=cand_logp).sample()` under `no_grad`; stores prompt, canonical action idx, log-prob, and value into the buffer.
- **Thoughts-prefix conditioning** (rl_layer.py:192-197, 208-213): the one-sentence rationale from an *earlier* LLM call is appended as `"\n\nThoughts: {…}\nAction:"`, so the policy is p(action | prompt, thoughts) — thoughts drive the action. The **augmented** prompt is what gets buffered, and the PPO update re-scores the same augmented prompt, keeping the importance ratio consistent.
- **Message has no gradient:** `select_action` returns `communication: ""` (rl_layer.py:244-248); messages come from a separate LLM call outside the RL graph (see 08-cognitive-agent.md).
- Pooled hidden state for the value head = last-prompt-token hidden of row 0, upcast to fp32 (rl_layer.py:593-594).
- `length_normalize_action_logp` is read via `getattr` (rl_layer.py:596-601) but is **not** an `RLConfig` field — effectively always False.

Mode `"token"` makes `select_action` return `None` (LLM picks actions; RL only does token-level fine-tuning, §4).

## 3. Trajectory buffer (`trajectory_buffer.py`)

`Transition` fields (trajectory_buffer.py:18-35):

| Field | Set by | Meaning |
|---|---|---|
| `prompt_text` | store_action | augmented scoring prompt (re-scored at update) |
| `action_idx` | store_action | index into the canonical 22-action tuple |
| `old_log_prob` | store_action | log π_old(a\|s) over the 14 candidates |
| `old_value` | store_action | per-agent ValueHead V(s) (0.0 in centralized mode) |
| `reward`, `done` | store_reward | sanitised env reward; terminal flag |
| `advantage`, `returns` | compute_gae | filled in-place at update time |
| `reward_task`, `reward_comm` | store_reward | decomposition for post-hoc analysis only |
| `old_value_global`, `joint_state` | set_pending_value_global | MAPPO critic V_global + the joint state it was computed from |

Mechanics:
- Two-phase write: `store_action` parks a pending transition; `store_reward` completes it. A second `store_action` flushes the pending one with reward 0; `store_reward` with nothing pending warns and discards (trajectory_buffer.py:48-109).
- **Reward sanitisation** (trajectory_buffer.py:94-103): non-float → 0, NaN/inf → 0, clamp to ±1e6. Pinned by `tests/test_trajectory_buffer.py`.
- **GAE** (`compute_gae`, trajectory_buffer.py:122-162), backward over the rollout with bootstrap `V(s_T+1) = last_value`:
  - δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
  - A_t = δ_t + γ·λ_GAE·(1−done_t)·A_{t+1}
  - returns_t = A_t + V(s_t)
  with γ=0.995, λ_GAE=0.95 (config.py:38-39; λ_GAE here is the GAE trace, distinct from the Hebbian decay λ).
- `use_global_value=True` (MAPPO) reads `old_value_global` per transition, **falling back to `old_value`** for any transition missing it (trajectory_buffer.py:135-138).
- **Per-rollout advantage standardisation** to mean 0 / std 1 over the full rollout, not per mini-batch (trajectory_buffer.py:155-162). **Single-transition guard** (post-commit fix): `std()` of one element is NaN, so standardisation is skipped below 2 transitions (trajectory_buffer.py:157-159; pinned by `tests/test_trajectory_buffer.py::test_gae_single_transition_unstandardised_and_finite`).
- `sample_batches` shuffles the pool and accepts `extra_transitions` (social replay) so replayed transitions spread across mini-batches (trajectory_buffer.py:166-183). `max_size=2048` is stored but never enforced; the update at `update_interval=128` keeps the buffer bounded in practice.

Reward transforms applied **before** storage (`RLLayer.store_reward`, rl_layer.py:282-317): optional `death_penalty` (default 0); running-std reward normalisation (`RunningMeanStd`, Welford, heads.py:15-35 — note: divides by std but does **not** subtract the mean) **only in IPPO mode** — disabled when centralized, see §5.

## 4. PPO paths

### Action-level (default; `ippo.action_level_ppo_step`, ippo.py:40-208)

Orchestrated by `ppo_update.run_ppo_update` (ppo_update.py:30-120):

1. Bootstrap `V(s_T)` — centralized critic on the last stored `joint_state`, else per-agent value head; 0 if last transition is `done` (ppo_update.py:125-138).
2. `compute_gae(..., use_global_value=rl._use_centralized)`.
3. **Social replay hook** (ppo_update.py:141-169): samples neighbour-buffer transitions via `hebbian_graph.get_social_replay_indices(..., rho=...)` weighted by bonds W (Eq. 7). With ρ (`social_replay_rho`) = 0 this returns nothing — a no-op. The *dataclass* default is 0, but the CLI default `--hebbian-rho 0.3` overrides it, so RL+Hebbian runs launched via `multi_agent_craftium.py` without an explicit `--hebbian-rho 0` had replay ACTIVE.
   > PAPER MISMATCH — social replay was active (ρ=0.3) in the evaluated RL+Hebbian runs despite App C.1's "not evaluated" claim; see PAPER_INCONSISTENCIES.md #9
4. **Entropy anneal** (ppo_update.py:172-178): linear 0.05 → 0.001 over 500 *PPO updates* (config.py:54-56). Quirk: if `entropy_anneal_steps <= 0` it returns `entropy_coef` (0.01) — neither start nor end. Pinned by `tests/test_heads_and_anneal.py::test_anneal_entropy_zero_steps_falls_back_to_entropy_coef`.
5. Mini-batch loop: 2 epochs × shuffled batches of 4; first-epoch sanity check warns if mean ratio deviates from 1.0 by >0.05 (tokenization drift detector, ppo_update.py:78-94); grad-norm clip 0.5 with non-finite-norm step skip (ppo_update.py:96-109); buffer cleared after the update.

Inside `action_level_ppo_step`:
- **Per-transition backward** to bound memory: each transition does forward → `loss_term/N` → `backward()`, releasing the graph immediately; mathematically identical to the batched mean loss but peak memory is one forward regardless of mini-batch size (rationale ippo.py:52-74).
- Standard clipped surrogate with ε_clip=0.2 (ippo.py:137-141); entropy bonus; transitions whose `action_idx` is no longer in the candidate set are dropped pre-forward (mask toggled across a resume, ippo.py:85-90); non-finite logits skip the transition with a warning (ippo.py:122-124).
- Value loss (IPPO only, `value_loss_enabled=not _use_centralized`): PPO2 clip — `max(MSE(v_new, R), MSE(v_clipped, R))` with clip ξ = `value_clip_eps` = 1.0 (ippo.py:143-157).

### Token-level (`token_opt.py` + `ippo.token_level_ppo_step`)

Agent-decided fine-tuning on full sequences, only when `auto_token_opt=True` (default False):
- **Guards** (token_opt.py:106-130): success window full (10 outcomes), cooldown ≥ `token_opt_window` steps since last opt, buffer ≥ `token_opt_min_samples` = 32.
- The agent itself decides via an LLM call against `learning_belief.txt` (success rate, reward trend, fail streak, optional Hebbian bond context, token_opt.py:41-101, 177-192); on "train", transitions are filtered by skill keyword and — when Hebbian is enabled — top-k by |advantage| (token_opt.py:223-244).
- `token_level_ppo_step` (ippo.py:213-265): length-normalised sequence log-prob, sequence-level ratio + clipped surrogate, batch-standardised rewards as advantages; 2 epochs (token_opt.py:249-298).

## 5. Centralized critic (MAPPO; `centralized_critic.py`)

Single shared V(joint_state) replacing the per-agent baselines. Joint state = Stream A (compact) ⊕ Stream B (semantic), built by `encode_joint` (centralized_critic.py:218-252).

Per-agent Stream A block (`_CompactEncoder.encode`, centralized_critic.py:60-112; dim = 18 + M per agent):

| Slice | Dim | Encoding |
|---|---|---|
| position | 3 | raw xyz (unnormalised; world spans ~50 blocks) |
| chamber | 5 | one-hot over ch1..ch5 |
| hp | 1 | hp/20 (1.0 if unknown) |
| inventory | 8 | bag-of-words **substring match** over diamond_sword, diamond_chestplate, tree, stone, wood, log, cobble, dirt |
| milestones | M | bitmap over the configured milestone ids |
| raw reward | 1 | pre-diffusion step reward clipped to ±100 |

Stream B: per-agent sentence-transformer embedding (`ST_MODEL_NAME`, default all-MiniLM-L6-v2, 384-d) of `"{last_action}. {last_comm}"`, L2-normalised (centralized_critic.py:187-192, 239-252).

- **Net:** `_CriticNet` = 3 Linear layers with GELU, hidden 256 (centralized_critic.py:155-167); Adam lr 3e-4 (config.py:29-30).
- **Training** (centralized_critic.py:279-343): own `_CriticBuffer` of (joint_state, team_reward, V_t, done); GAE returns on the **team-mean reward** stream (centralized_critic.py:134-146); MSE with PPO2 value clip ξ = `critic_value_clip_eps` = **10.0** (config.py:31; clip at centralized_critic.py:314-322); logs explained variance pre-update.
- **Integration:** orchestrator calls `encode_joint` + `evaluate` once per step after all agents acted, then `RLLayer.set_pending_value_global` attaches the identical V_global to each agent's pending transition (rl_layer.py:271-280; trajectory_buffer.py:70-83). `get_pending_value` (rl_layer.py:250-269) exposes V_global to the Hebbian advantage computation (see 02-hebbian-graph.md).
- **Why reward normalisation is IPPO-only** (rl_layer.py:293-307): the critic trains on raw team rewards, so V_global is raw-scale; normalising per-agent rewards would make GAE mix scales (δ = r_norm + γ·V_raw − V_raw). Both streams stay raw in MAPPO; per-rollout advantage standardisation still gives the policy unit-variance advantages.

## 6. IPPO ablation (`critic_mode="independent"`)

Each agent keeps its own `ValueHead` — MLP(hidden → 256 → Tanh → 1) on the pooled last-prompt-token hidden state, fp32 (heads.py:49-61; rl_layer.py:116-118). Trained inside `action_level_ppo_step` with value clip ξ = 1.0 and `value_coef` 0.5; reward normalisation ON (§5). `RLLayer._use_centralized` requires both a critic instance *and* `critic_mode=="centralized"` (rl_layer.py:87-91). `ActionHead` (heads.py:38-46) is legacy and unused by the actor.

## 7. Persistence (`persistence.py`)

Checkpoint layout under `<lora_save_dir>/<adapter_name>/` (persistence.py:1-16):

| File | Contents |
|---|---|
| `adapter_config.json` + `adapter_model.safetensors` | this agent's LoRA only (`selected_adapters=[name]`, persistence.py:44-53) |
| `value_head.pt` | ValueHead state_dict |
| `rl_state.pt` | optimizer state, `step_count`, `_update_count`, `_last_token_opt_step`, `_recent_successes/_recent_actions/_recent_rewards`, `_current_task`, RMS `mean/var/count` (persistence.py:57-70) |

No `action_head.pt` is written anymore; legacy ones are ignored with a warning on load (persistence.py:81-91). LoRA weights load at init via `_ensure_adapter`; `load()` restores heads/optimizer/RMS/counters (persistence.py:75-122). The centralized critic persists separately as `critic_net.pt` + `critic_state.pt` (centralized_critic.py:347-370).

## 8. Key hyperparameters (defaults, config.py)

Pinned against paper Table 6 by `tests/test_paper_defaults.py::test_rlconfig_table6_defaults`. Full table in 10-configuration.md.

| Group | Values |
|---|---|
| LoRA / model | rank 8, alpha 16, dropout 0.05, fp16, prompt cap 256 tok, grad ckpt on |
| Discounting | γ = 0.995, GAE λ = 0.95 |
| PPO | ε_clip 0.2, 2 epochs, mini-batch 4, actor lr 1e-4, grad-norm clip 0.5 |
| Value | value_coef 0.5, IPPO clip ξ = 1.0, critic clip ξ = 10.0, critic hidden 256, critic lr 3e-4 |
| Entropy | 0.05 → 0.001 over 500 updates (anneal) |
| Schedule | update_interval 128 steps, buffer cap 2048 |
| Rewards | normalize_rewards True (IPPO path only), death_penalty 0 |
| Actions | 22 total, mask_slot_actions True → 14 candidates |
| Token-opt | off by default; window 10, min samples 32, 2 epochs |
