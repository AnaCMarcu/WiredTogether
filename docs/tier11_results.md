# Tier 11 results — forced cooperation on Foraging-Comm-10x10-3p-3f-coop-v3

**Date:** 2026-05-29
**Status:** Final. 15 / 15 runs completed but **no run reached non-zero return.**
**Headline:** The forced-cooperation env at this map size is unsolvable in 3M steps with random exploration. No reward signal was ever generated, so policy gradient had nothing to optimize. The bond-differentiation hypothesis is **not testable** here — there's no learning to compare.

---

## 1. Setup

Identical to tier 9 except for one environment flag:

| | tier 9 | tier 11 (this) |
|---|---|---|
| Env | `Foraging-Comm-10x10-3p-3f-v3` | `Foraging-Comm-10x10-3p-3f-coop-v3` |
| `force_coop` | False (mixed levels 1..3) | **True** (every food level 3) |
| Algorithm | MAPPO + `hebbian_parallel` runner | same |
| `common_reward` | False | False |
| Variants | mappo_hebbian, mappo_hebbian_r, mappo_hebbian_uniform_r | same |
| Seeds | 5 (paired) | same |
| `t_max` | 3,000,000 env steps | same |
| `time_limit` | 50 | same |
| Cluster | DelftBlue | DAIC |

`force_coop=True` is plumbed in via the lbf_comm wrapper extension (commit landing alongside this run): the wrapper now registers
`Foraging-Comm-WxH-Np-Mf-coop-v3` ids and constructs the underlying
lbforaging env with `force_coop=True` passed via `gym.make` kwargs.

---

## 2. Results

### 2.1 Policy performance

| variant | tier-11 final | tier-9 ref final | Δ |
|---|---|---|---|
| `mappo_hebbian`           | **0.000** ± 0.000 | 0.855 ± 0.071 | −0.855 |
| `mappo_hebbian_r`         | **0.000** ± 0.000 | 0.723 ± 0.029 | −0.723 |
| `mappo_hebbian_uniform_r` | **0.000** ± 0.000 | 0.902 ± 0.040 | −0.902 |

Every seed of every variant ends training with `test_total_return_mean = 0.000` (and `total_return_mean = 0.000`). Sampled at training-time:

```
t_env=50,000     test = 0.000, train = 0.000
t_env=300,000    test = 0.000, train = 0.000
t_env=550,000    test = 0.000, train = 0.000
t_env=1,050,000  test = 0.000, train = 0.000
t_env=2,050,000  test = 0.000, train = 0.000
t_env=2,950,000  test = 0.000, train = 0.000
```

This is not "policy collapse" (which would show positive returns early then decay). It is **exploration failure** — the policy never produces a trajectory that triggers a reward, so the gradient signal stays at zero.

### 2.2 Why this happens mechanically

With `force_coop=True` and 3 agents, every food has level 3. To collect any reward in an episode the three agents must:

1. All three be simultaneously adjacent (Chebyshev-1) to the same food cell.
2. All three execute the `LOAD` action on the same timestep.
3. With an init-random policy that hasn't learned coordination.

The rough probability for a random rollout in a 10×10 grid with 50-step episodes is well below `1e-4` per episode. Across 3M steps × 10 parallel envs × ~50-step episodes that's ~60k episodes total — and the empirical hit rate is **literally 0** across 15 runs.

By contrast, the non-coop tier-9 env has random food levels in {1, 2, 3}. Roughly a third of foods are level-1, loadable by any single agent that happens to be adjacent — far more common from random play. That trickle of reward is what bootstrapped tier 9's learning.

### 2.3 Signal-channel diagnostic (surprising)

`hebbian/signal_total` per episode-batch, sampled along training (seed 0 of `mappo_hebbian_r` is representative):

| snapshot | tier 9 | tier 11 |
|---|---|---|
| #0 (init)   | 111 | 307 |
| ~10%        | 10  | 373 |
| ~25%        | 7   | 358 |
| ~50%        | 3   | 390 |
| ~75%        | 1   | 386 |
| #59 (final) | **0**   | **367** |

**Tier 9:** signaling decayed to zero — the opportunity-cost penalty (signal = no movement that step) drove the action's advantage negative once partial rewards from solo loading existed.

**Tier 11:** signaling stayed high throughout. With zero rewards anywhere in training, there's no gradient to push signal probability either up *or* down. The signal action's empirical advantage is exactly zero, so the policy's signal-action probability stays near init.

This is informative as a control: **what looks like "agents are using the comm channel" can equally well be "agents have no gradient signal at all and the policy is at random init."** Without a paired comparison to an env where learning *does* happen, signal counts alone don't tell us if the channel is functional.

### 2.4 Files

| | |
|---|---|
| Analysis script | [`scripts/analysis/tier11_analysis.py`](../scripts/analysis/tier11_analysis.py) |
| Per-seed CSV | [`scripts/analysis/out/tier11_finals.csv`](../scripts/analysis/out/tier11_finals.csv) |
| Learning curves | [`scripts/analysis/out/tier11_curves.png`](../scripts/analysis/out/tier11_curves.png) |
| Tier-9 vs tier-11 bars | [`scripts/analysis/out/tier11_bars.png`](../scripts/analysis/out/tier11_bars.png) |
| Raw logs | [`runs_from_daic/hebbian-marl/logs/mappo_hebbian*_seed*.log`](../runs_from_daic/hebbian-marl/logs/) (15 files) |

---

## 3. What this means for the thesis question

The bond-differentiation hypothesis was the *reason* to run a forced-cooperation variant — we wanted a setting where bonds had to do work. But we cannot test bond differentiation in a setting where the policy never learns anything. The current data is a clean **null** on every comparison; it does not falsify or support the headline claim.

---

## 4. Recovery options

In order of cost and likelihood-of-being-decisive:

### 4.1 Smaller forced-coop env (cheapest)

Use `Foraging-Comm-8x8-2p-2f-coop-v3` or `Foraging-Comm-8x8-3p-2f-coop-v3`. The canonical EPyMARL benchmark for `Foraging-8x8-2p-2f-coop-v3` reaches ~0.9 by 2M steps (verified in tier 6). The smaller grid and fewer foods make random cooperative loading occasionally happen, bootstrapping learning. 3 agents on 8×8 is harder than 2 on 8×8 but still within reach.

Trade-off: 2-agent versions don't have enough pairs to make bond-structure analysis interesting (only one bond pair). 3-agent on 8×8 is the right size — 3 bond pairs, but small enough for random coop to occasionally happen.

### 4.2 Longer training budget

10M or 20M steps. The EPyMARL paper benchmarks LBF at ~20M. The exploration barrier might be crossed eventually. But: in our 3M runs there is **literally zero** reward across 15 runs combined, so the barrier looks taller than just "needs more steps."

### 4.3 Reward shaping or curriculum

Add partial credit for "all 3 agents near the same food" or "some subset attempted to load together." This is conceptually a fix to the env, not the algorithm. Defensible but it changes the task being studied.

### 4.4 Initialize from a learned tier-9 policy

Pretrain on non-coop (where rewards exist), then fine-tune on coop. This guarantees learning starts from a non-trivial cooperative policy. Adds engineering work to the launcher.

### 4.5 Pivot the asymmetric-task test elsewhere

Drop forced-coop in LBF, go to a different env where asymmetric coordination is the natural setting (RWARE, Hanabi, custom). Higher engineering cost; most decisive for the thesis question.

### Recommendation

Run **4.1 first** as a tier 12 (`Foraging-Comm-8x8-3p-2f-coop-v3`, same 3 variants × 5 seeds, ~30 min each). It's the cheapest way to find out if the bond hypothesis is testable in *any* LBF coop setting. If that also fails to learn, pivot to 4.5.
