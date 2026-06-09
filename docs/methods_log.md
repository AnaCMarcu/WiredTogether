# Methods Log — WiredTogether

A chronological log of the broad algorithmic directions tried during the thesis,
why each was chosen, what the experiments showed, and the decision to keep,
iterate, or move on. The intent is that every direction has a one-page entry
the supervisor (or future-me) can read to reconstruct the methodology timeline
without spelunking through commits.

Bug fixes, infrastructure work, and one-off tweaks are NOT recorded here —
those live in commit messages and PR descriptions. This file is for
thesis-section-worthy decisions only.

Entries are append-only and dated. New approaches go at the bottom.

---

## 1. Legacy stack — MAPPO/IPPO + IC3Net comm + Hebbian observer

**Dates:** ≈ Mar – early May 2026

**What:** Multi-agent RL on the Five Chambers Craftium env using on-policy
actor-critic (MAPPO with a shared centralized critic, IPPO with per-agent
critics for the ablation) over LoRA-adapted Qwen3.5-2B as policy. Agents
exchange targeted text messages through an IC3Net-style continuous
communication channel; the legacy Hebbian module observes co-activity
events and maintains a bond-strength graph used for in-prompt social
context and post-hoc cooperation metrics.

Experiment grid:
- M1–M5 — model-axis baselines (plain LLM, MAPPO, IPPO, with/without
  Hebbian) at N=3 agents.
- E1, E2, E4, E5 — comm on/off and Hebbian on/off baselines, RQ-level
  comparison runs.
- H1, H10c/d/e/f, H12 — scaling sweep (3 / 6 / 9 agents) and Hebbian-only
  comm-disabled run (RQ4).

**Why tried:** This is the standard MARL recipe — a known-good way to
get a LoRA-on-LLM policy that responds to environment reward. IC3Net
provides a precedent for learning differentiable communication, and
the Hebbian observer was added as the thesis's core contribution
hypothesis (RQ1 / RQ2): local-rule social plasticity captures
co-coordination that a global critic cannot.

**Setup:** Qwen3.5-2B LoRA r=8, `rl_critic_mode=centralized` (or
`independent` for IPPO), `hebbian_radius=20`, `hebbian_ltp=0.05`,
`hebbian_ltd=0.005`, `hebbian_decay=0.001`. Per-step rewards
(Craftium task + comm milestone tiers + Hebbian reward diffusion).
N=3 default, `max_steps=2000–5000` per episode, 1–3 episodes per run.

**Findings (5 runs in `runs_from_hpc/`):**
- Agents plateau at chamber 1 across all conditions (RL or not, Hebbian
  or not). The furthest milestone reached was `m3_pickup_3` — chambers
  2–5 are never entered through skill, only through the Ch1 timeout
  teleport.
- Severe per-agent reward inequality in RL+Hebbian runs: returns of
  [-49, -104, +47] and [-8, -69, +80]. One agent dominates the
  task reward while the others trend net-negative.
- Hebbian bonds collapse rather than form: mean bond strength
  drops 0.128 → 0.003 across episodes in the long Hebbian run.
  The hypothesis "co-activity reinforces bonds" cannot be tested
  because there is no co-activity to reinforce — agents are stuck
  in Ch1 doing solo dig/move.
- Cooperation score (5-component aggregate) sits at the ~0.6 floor
  in nearly every run regardless of behaviour; it cannot serve as
  an absolute "cooperation level" claim, only as a delta.

**Decision:** **Superseded** by the GRPO pivot (entry 4). Reasons:
(a) per-step actor-critic on a sparse-reward 5-chamber env makes
gradient signal too noisy to learn beyond Ch1; (b) the Hebbian
hypothesis cannot even be tested while agents never co-engage;
(c) a separate env-side bug — agents being teleported to Ch2 during
warmup — was discovered analysing these runs (logged in commit msg,
not here) and invalidates the milestone-coverage numbers further.
The legacy scripts remain in `scripts/experiments/E*.sh, M*.sh`
for reproducibility but no new runs are planned in this stack.

---

## 2. LLM-as-policy without RL gradient — Hebbian in prompt and propagated rewards

**Dates:** May 2026

**What:** Two inference-time-only experiments where the LLM policy is
NOT trained — RL gradient is disabled entirely (`--rl false`) — but the
agent's prompt is augmented with information from the Hebbian module.
- **L1** surfaces current bond weights as a `{social_bonds}` block in
  the system prompt; the agent sees the live N×N graph each step.
- **L2** adds per-step propagated-reward attribution to the prompt,
  e.g. `Propagated rewards this step: +2.50 from agent_1 (m17_switch_pressed)`,
  so the agent can in-context-learn from teammate-driven rewards.

**Why tried:** Separates the contribution of Hebbian social structure
from the contribution of RL gradient. If the LLM alone, given the bond
graph as context, coordinates better than a baseline LLM, that is
evidence that the **representation** of social plasticity has signal —
independent of whether RL successfully exploits it. Conversely, if L1
matches the plain-LLM baseline (M1), the Hebbian representation
isn't legible to the LLM and the contribution has to come through
the gradient path.

**Setup:** Qwen3.5-2B, no RL training, `--hebbian` enabled (L1, L2),
L2 additionally requires reward-propagation logging. Phase B+
`interpretability.jsonl` per-step emission. N=3, 1000 max steps.

**Findings:** Pending — runs are queued / partially complete. Comparison
targets are M1 (no Hebbian, no RL) and M3 (MAPPO + Hebbian).

**Decision:** **Active.** These are cheap to run (inference-only, no
gradient updates) and answer a methodologically clean question, so
keep them as part of the final thesis matrix regardless of GRPO
results.

---

## 4. GRPO pivot — per-trajectory advantage replacing per-step actor-critic

**Dates:** May 2026 (in progress)

**What:** Replace the legacy MAPPO/IPPO trainer with Group-Relative
Policy Optimization. Each GRPO step samples G joint rollouts of
horizon H from the current policy, computes group-relative advantages
(per-trajectory return minus the group mean), and updates the LoRA
adapter via a clipped surrogate loss with KL regularization to a
frozen reference adapter. The new entry point is
`src/mindforge/multi_agent_craftium_grpo.py`; metrics land in
`grpo_metrics.jsonl` + sidecars (`episode_summary.jsonl`,
`time_to_first.json`, `hebbian_snapshots.jsonl`).

Ablation grid:
- G2 — multi-agent GRPO, 3B per-agent reward, no Hebbian (the
  headline baseline).
- G2b — same but with team-shared reward (cooperation-axis ablation).
- G3a — GRPO + Hebbian reward diffusion ONLY (Stage 4a; composition off).
- G3b — GRPO + Hebbian-weighted group composition ONLY (Stage 4b; diffusion off).
- G4 — both axes together (the headline Hebbian claim).
- G5 — read-only aggregator that produces the T1–T5 thesis tables
  from the JSONLs.

**Why tried:** The legacy stack (entry 1) plateaued at Ch1 and
produced reward signal too noisy to test the Hebbian hypothesis.
GRPO addresses two specific limitations:
1. Per-trajectory advantage means no value-function bias; agents
   are rewarded for their *whole rollout* relative to peers, not for
   each step against a possibly-broken critic.
2. The group-relative baseline removes the need for a learned critic
   entirely — directly relevant given the centralized critic was
   the suspected culprit for agent-2 dominance in entry 1.

It also opens a clean place to plug the Hebbian module into the
training loop: Stage 4a (reward diffusion across the joint trajectory)
and Stage 4b (Hebbian-weighted group composition for off-policy
trajectory borrowing).

**Setup:** Qwen3.5-2B, LoRA r=16, GRPO `n_per_group=4`, `horizon=50`,
`total_steps=1000` per seed, `kl_coefficient=0.05`, `clip_epsilon=0.2`.
Three seeds per ablation via `sbatch --array=0-2`.

**Findings (partial):** Pipeline stands up. T1–T5 table renderers,
bootstrap CIs, and Wilcoxon-signed-rank cross-checks are implemented
and tested (419 rlvr tests pass). First G2 run hit a CUDA OOM at the
first update step — patched separately (per-sample backward in
`_update`, gradient checkpointing in `GRPOLanguageModel`). Headline
G2/G3a/G3b/G4 runs not yet completed; waiting on cluster time.

**Decision:** **Active and primary.** This is the thesis's main
experimental track. Legacy stack (entry 1) remains only for
reproducibility of historical baselines (M1, M3, etc.) that the
thesis cites in the "prior approaches" section.

---

## 5. Hebbian-axis decomposition under GRPO — diffusion vs composition

**Dates:** May 2026

**What:** Split the Hebbian augmentation into two orthogonal mechanisms
and test each in isolation before combining, so the thesis can attribute
any GRPO+Hebbian gain to the correct axis rather than just claiming
"Hebbian helps."
- **Reward diffusion (Stage 4a, G3a):** within a single joint rollout,
  each agent's per-step reward is partially redistributed across
  teammates via the current Hebbian bond matrix W. The trajectory
  return that GRPO scores is the *diffused* return, so a strongly-
  bonded teammate's success shows up in the gradient as your own.
- **Group composition (Stage 4b, G3b):** when building the G
  trajectories that GRPO compares for relative advantage, replace
  some samples with trajectories *borrowed* from teammates, weighted
  by bond strength. Treats the policy update as if the group included
  imagined samples from co-bonded peers. Uses Option 4b-i (clipped
  off-policy — the surrogate clip handles the off-policy bias
  on its own, per `docs/rlvr_grpo_plan.md` §5.4).

**Why tried:** The two mechanisms operate on different parts of the
GRPO loss. Diffusion changes the *return*; composition changes the
*sample distribution*. Combining them without isolating them would
make a positive G4 result unattributable. The decomposition is a
direct methodological contribution — a future Hebbian-augmented RL
paper can cite this as "diffusion provides X, composition provides Y,
combined provides X+Y or X·Y" (or, equally informatively, "the gain
is dominated by one axis, the other contributes nothing").

**Setup:** Same as entry 4 for G2 (no-Hebbian baseline). G3a sets
`verifier.hebbian_reward_diffusion=true` and
`grpo.hebbian_group_composition=false`. G3b inverts those. G4 sets
both true with `hebbian_borrow_fraction=0.25`.

**Findings:** Pending — runs queued or partially complete on HPC
following the OOM patch. T3 ("Hebbian-axis decomposition") in the
thesis tables is the dedicated result row.

**Decision:** **Active.** This is the substantively novel
experimental design of the thesis — even if G4 just matches G2,
showing *which axis carries weight (if any)* is a publishable
negative result.
