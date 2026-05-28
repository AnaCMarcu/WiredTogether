# Tier 9 results — Hebbian reward diffusion on Foraging-Comm-10x10-3p-3f-v3

**Date:** 2026-05-28
**Status:** Final. 15 / 15 runs completed successfully.
**Headline:** The pre-registered thesis claim — *learned Hebbian bonds drive cooperation beyond uniform reward sharing* — is **not supported** by these results. Uniform diffusion outperforms Hebbian-weighted diffusion on every one of 5 seeds.

---

## 1. Setup

| | |
|---|---|
| Environment | `Foraging-Comm-10x10-3p-3f-v3` (3 agents, 3 food items, level-3 cooperative loading) |
| Episode horizon | 50 steps (env time_limit) |
| Training budget | 3,000,000 env steps per run |
| Eval | 100 episodes every 50,000 env steps |
| Seeds | 5 (paired across variants: `seed ∈ {0, 1, 2, 3, 4}`) |
| Algorithm stack | MAPPO (parallel runner, `cv_critic`, `basic_mac`, RNN actor, `obs_agent_id=True`, `entropy_coef=0.001`, `epochs=4`, `eps_clip=0.2`, `batch_size_run=10`, `common_reward=False`) — identical across all variants |

### Variants compared

All three variants share the same MAPPO learner code and the same env. The
*only* algorithmic difference is the reward routing applied by
`HebbianParallelRunner` before rewards enter the batch:

```
r'_i(t) = (1-γ) · r_i(t) + γ · Σ_{j≠i} W̄[i,j] · c[i,j] · r_j(t)
```

| variant | γ | W̄[i, j] | c[i, j] |
|---|---|---|---|
| `mappo_hebbian`           (baseline, no diffusion) | 0   | —                                  | —                |
| `mappo_hebbian_r`         (Hebbian-weighted) | 0.2 | learned bond matrix, row-normalized | spatial co-activity gate (learned) |
| `mappo_hebbian_uniform_r` (uniform control) | 0.2 | 1 / (N − 1) (uniform)              | 1 (gate disabled) |

The uniform control is implemented via two flags on `HebbianConfig`
(`uniform_weights=True`, `disable_coactivity_gate=True`) so the
runner code path is bit-identical to the Hebbian variant — only the
arithmetic inside `diffuse_rewards` differs.

### Pre-registered comparisons

Paired Wilcoxon sign-rank, one-sided, on the final-window mean of
`test_total_return_mean` (averaged over the last 5 eval points per run):

- **A:** `mappo_hebbian_r > mappo_hebbian`            — does Hebbian diffusion help?
- **B:** `mappo_hebbian_r > mappo_hebbian_uniform_r`  — **does the learned bond structure outperform trivial uniform sharing?** *(thesis-pivotal)*

---

## 2. Results

### 2.1 Final-window performance (last 5 evals)

| variant | mean | std | per-seed |
|---|---|---|---|
| `mappo_hebbian_uniform_r` | **0.902** | 0.040 | 0.946, 0.954, 0.869, 0.880, 0.862 |
| `mappo_hebbian`           | 0.855     | 0.071 | 0.822, 0.914, 0.729, 0.914, 0.893 |
| `mappo_hebbian_r`         | 0.723     | 0.029 | 0.751, 0.674, 0.746, 0.738, 0.705 |

Ordering: **uniform > baseline > Hebbian**. The Hebbian variant is the
worst performer.

### 2.2 Paired Wilcoxon sign-rank (one-sided, n=5)

| comparison | median Δ | W | p | conclusion |
|---|---|---|---|---|
| **A** (`hebbian_r > baseline`) | −0.176 | 1.0  | 0.969 | **rejected** — diffusion hurts |
| **B** (`hebbian_r > uniform_r`) | −0.158 | 0.0  | 1.000 | **rejected** — Hebbian is worse than uniform |
| reverse A (`baseline > hebbian_r`)        | +0.176 | 14.0 | 0.063 | borderline, n-limited |
| **reverse B (`uniform_r > hebbian_r`)**   | **+0.158** | **15.0** | **0.031** | **significant** — all 5 seeds favor uniform |
| extra (`uniform_r > baseline`)            | +0.040 | 12.0 | 0.156 | trends positive, not significant |

With n=5, the minimum achievable one-sided p is 0.0312 (when all 5 paired
differences share the same sign). We hit exactly that floor for
`uniform_r > hebbian_r` — i.e., on every one of the 5 seeds, the uniform
variant beat the Hebbian variant. That is as clean as the test allows.

### 2.3 Learning trajectories

See [`scripts/analysis/out/tier9_curves.png`](../scripts/analysis/out/tier9_curves.png)
for the mean ± std learning curves.

Qualitative read of the curves:

- **Uniform diffusion (green):** fastest early learner; passes 0.7 by ~1M
  steps; tightest seed spread; never overtaken.
- **Baseline no-diffusion (blue):** slower to start; catches up in the
  late phase; trails uniform by ~5% throughout.
- **Hebbian-weighted diffusion (red):** matches the others early
  (~0.3 at 500k) then plateaus around 0.7 by 1.5M and never closes
  the gap. The plateau is tight (std 0.03), so the ceiling is
  structural, not noise.

### 2.4 Files

| | |
|---|---|
| Per-seed tidy CSV | [`scripts/analysis/out/tier9_finals.csv`](../scripts/analysis/out/tier9_finals.csv) |
| Learning curves | [`scripts/analysis/out/tier9_curves.png`](../scripts/analysis/out/tier9_curves.png) |
| Final-window bars + per-seed dots | [`scripts/analysis/out/tier9_bars.png`](../scripts/analysis/out/tier9_bars.png) |
| Per-seed paired differences | [`scripts/analysis/out/tier9_paired.png`](../scripts/analysis/out/tier9_paired.png) |
| Raw logs | [`logs/mappo_hebbian*_seed*.log`](../logs/) (15 files) |
| Bonds trajectories (cluster only, not yet pulled) | `runs/hebbian-marl/bonds/mappo_hebbian_*/seed_*.jsonl` |

---

## 3. Interpretation

### 3.1 What the data falsifies

The thesis-headline claim — *learned Hebbian bonds drive cooperation
beyond what trivial uniform sharing delivers* — is falsified by these
results. On this environment, with this algorithm stack, **the learned
bond structure is worse than no structure at all**.

The mechanism (reward diffusion) is not the problem: uniform diffusion
trends positive vs. the no-diffusion baseline (Δ = +0.040, n.s. but
suggestive). The problem is *which weights* are used to do the
diffusion. Replacing the learned W̄ with uniform 1/(N − 1) and dropping
the spatial gate strictly improves performance.

### 3.2 Why this likely happened

Two non-mutually-exclusive explanations consistent with the data:

1. **The task's optimal credit structure is symmetric.** LBF-3p-3f with
   level-3 cooperative loading requires all three agents to contribute
   equally to load a food. The Pareto-optimal coordination is symmetric
   in agent role. Any non-uniform W̄ moves the diffused reward signal
   *away from* the task's symmetry. Uniform sharing matches the task's
   structure; learned bonds add structure where the task wants none.

2. **The co-activity gate over-restricts diffusion.** With
   `c[i,j] ∈ [0, 1]` driven by spatial proximity, the Hebbian variant
   only diffuses reward when agents are currently near each other.
   When agent j scores and bonded teammate i is at the other end of
   the map, no diffusion occurs — even though agent i is on the
   policy team and *will* be there next episode. Uniform-r ignores
   both the bond and the gate (c=1), so the diffused signal reaches
   every teammate every step. The denser, structure-free signal is
   evidently more useful here.

### 3.3 What is *still* defensible from this data

The empirically supported claim is:

> **Per-agent reward diffusion improves cooperative MARL performance on
> LBF, but the gain comes from the diffusion mechanism itself — not from
> the learned Hebbian structure. In this homogeneous-agent setting, the
> learned structure underperforms a trivial uniform baseline.**

This is a clean negative result on a pre-registered hypothesis with a
plausible mechanistic explanation. It is publishable as-is and
informative for follow-up work.

### 3.4 What would *not* be defensible

The original framing — that Hebbian-as-social-intelligence improves
cooperative learning — cannot be claimed on the basis of this evidence.
A writeup that suggests otherwise would misrepresent the result.

---

## 4. Where this points next

### 4.1 Tasks with asymmetric coordination structure

The strongest follow-up is the *prediction*: if (3.2.1) is the right
explanation, Hebbian bonds should outperform uniform sharing on tasks
where the optimal coordination structure is asymmetric. Candidate envs:

- LBF with mixed food levels (some level-2, some level-3, requiring
  different sub-teams).
- LBF with asymmetric food rewards or asymmetric agent levels.
- RWARE (heterogeneous robot–shelf assignments).
- Multi-room LBF where bonds should track *which* room teammates
  habitually use.

If Hebbian outperforms uniform on these tasks but underperforms on the
symmetric LBF-3p-3f case, the thesis has a refined and defensible
form: *Hebbian bonds capture useful structure when the task has
structure to capture; on symmetric tasks they add noise.*

### 4.2 Descriptive analysis of the bonds themselves

Independent of policy performance, the `bonds.jsonl` traces let us ask:
do bonds *describe* anything meaningful about the agents' interaction
patterns, even if they don't help training? Specifically:

- Bond asymmetry: does W differ from W^T more than chance? Is there a
  consistent "high-out-degree" agent across seeds?
- Bond–success correlation: do per-pair bond strengths correlate with
  per-pair cooperative-load counts?
- Bond stability vs. policy convergence: does bond structure stabilize
  before, with, or after policy convergence?

These descriptive claims survive even when the intervention claim
fails. They are the natural fallback for the thesis chapter.

### 4.3 Ablation on the gate / weights independently

The uniform variant drops *both* the learned W̄ and the spatial gate
c[i, j]. A finer-grained ablation would tell us which knob is doing
the work:

| variant | W̄ | c |
|---|---|---|
| `mappo_hebbian_r`            | learned          | gated  |
| `mappo_hebbian_uniform_r`    | uniform 1/(N − 1) | 1 (off) |
| **new: `mappo_uniform_gated`**  | uniform 1/(N − 1) | gated (learned)  |
| **new: `mappo_hebbian_ungated`** | learned          | 1 (off) |

If `uniform_gated` ≈ `uniform_r`, the gate isn't load-bearing — the
issue is purely the learned W̄. If `hebbian_ungated` recovers most of
the uniform performance, the gate was the active hurdle. This is a
2-config addition and a 10-run cost (2 × 5 seeds) — cheap and
informative.

---

## 5. Descriptive bond analysis (new)

After running the per-policy comparison, we read the `bonds.jsonl` trajectories
for the 10 runs where the Hebbian module was enabled
(`mappo_hebbian_r` and `mappo_hebbian_uniform_r`). Both variants update the
same Hebbian graph machinery; the difference is that `mappo_hebbian_r`
*uses* the learned `W̄` in diffusion, while `mappo_hebbian_uniform_r`
overrides it with uniform weights (so the bonds are tracked but ignored).

### 5.1 Bonds collapse rapidly in both variants

| variant | seed | W mean (first snap) | W mean (final) | sparsity (final) | asymmetry (final) |
|---|---|---|---|---|---|
| `mappo_hebbian_r`         | 0 | 0.286 | 0.037 | 1.00 | 0.003 |
| `mappo_hebbian_r`         | 1 | 0.280 | 0.024 | 1.00 | 0.002 |
| `mappo_hebbian_r`         | 2 | 0.250 | 0.036 | 1.00 | 0.001 |
| `mappo_hebbian_r`         | 3 | 0.333 | 0.037 | 1.00 | 0.001 |
| `mappo_hebbian_r`         | 4 | 0.274 | 0.031 | 1.00 | 0.001 |
| `mappo_hebbian_uniform_r` | 0 | 0.286 | 0.034 | 1.00 | 0.001 |
| `mappo_hebbian_uniform_r` | 1 | 0.280 | 0.036 | 1.00 | 0.001 |
| `mappo_hebbian_uniform_r` | 2 | 0.250 | 0.028 | 1.00 | 0.001 |
| `mappo_hebbian_uniform_r` | 3 | 0.333 | 0.030 | 1.00 | 0.001 |
| `mappo_hebbian_uniform_r` | 4 | 0.274 | 0.032 | 1.00 | 0.001 |

In every one of the 10 runs:

- W mean **peaks at ~0.3–0.6 within the first 100k steps** then **decays to ~0.03 by 200k–300k** and remains there for the rest of training.
- **Final sparsity is 1.0** — every bond falls below the sparsity threshold.
- **Asymmetry is essentially zero throughout** (||W − W^T||_F < 0.005). No role differentiation between agents.
- The trajectories for `mappo_hebbian_r` and `mappo_hebbian_uniform_r` are **statistically indistinguishable** — same peak, same collapse rate, same final values. This is despite the fact that the *policies* differ substantially between the two variants. Bond formation is dominated by raw co-activity patterns, not by what the policy does with the bonds.

### 5.2 Final W matrices are essentially uniform

Average final W across 5 seeds, off-diagonal entries:

| variant | mean | std across seeds |
|---|---|---|
| `mappo_hebbian_r`         | 0.0327 | 0.0052 |
| `mappo_hebbian_uniform_r` | 0.0322 | 0.0032 |

All 10 final W matrices look the same: small uniform off-diagonal values
near 0.03, near-zero asymmetry, no role separation. See
[`tier9_bonds_heatmaps.png`](../scripts/analysis/out/tier9_bonds_heatmaps.png).

### 5.3 What this means for the result interpretation

The story from §3 needs refining. The pre-registered claim — "the
*learned* bond structure outperforms uniform sharing" — fails not
because the learned structure is *bad*, but because **the Hebbian update
rule fails to learn any informative structure** in this environment.

Concretely: after row-normalization, a uniform-tiny W matrix yields
approximately-uniform `W̄[i,j] ≈ 1/(N − 1)`. So the diffusion equation
in `mappo_hebbian_r` reduces, *in effect*, to:

```
r'_i(t) ≈ (1 − γ) r_i(t)  +  γ · (1 / (N − 1)) · Σ_{j ≠ i} c[i, j] · r_j(t)
```

The *only* meaningful difference from `mappo_hebbian_uniform_r` is then
the **spatial co-activity gate `c[i, j]`**. In `uniform_r`, the gate is
disabled (`c = 1`), so diffusion always reaches every teammate. In
`hebbian_r`, the gate is active, so reward only diffuses to teammates
who happen to be spatially close at that step.

The ~18-point performance gap between the two variants is therefore
attributable to the **spatial gate restricting diffusion**, not to the
(approximately absent) Hebbian structure. The hypothesis "*learned bond
structure should help*" is not strictly tested by these runs, because
the structure never differentiated.

### 5.4 Bond magnitude vs. policy return (per seed, n=5)

Even though the structure is approximately uniform, the *magnitude* of
the final bond matrix varies slightly across seeds. We checked whether
this magnitude correlates with the seed's final return:

| variant | Pearson r(mean_bond, final_return) |
|---|---|
| `mappo_hebbian_r`         | +0.98 |
| `mappo_hebbian_uniform_r` | +0.82 |

Both positive and notable, with the caveat that n=5 makes precise
inference impossible. Mechanistic interpretation: more-successful
policies generate more cooperative-load episodes → more co-activity
events → bonds reinforced more often → larger residual W after decay.
That is, **bond magnitude is a downstream consequence of policy
performance, not an upstream driver of it**, at least in this
environment.

### 5.5 Why didn't the bonds differentiate

Several mechanisms in the Hebbian update rule plausibly produce the
observed collapse to uniform-tiny:

1. **Symmetric exploration co-activity.** During exploration (and even
   during convergent cooperation in this env), all 3 agents are
   approximately equally likely to be spatially co-active with each
   other. No pair gets a sustained co-activity advantage.
2. **Decay outpaces differential LTP.** Default `decay = 0.0003` per
   step exceeds the differential-LTP signal once policies stabilize —
   bonds drift downward toward `base_ltp / (decay + base_ltp)` regardless of which pair we look at.
3. **Failure-grace symmetrizes.** The failure-grace LTP bonus fires for
   *every* pair on co-failure steps. Because failure is more common
   than success early in training, this term dominates and applies
   symmetric LTP across all pairs, washing out asymmetric reward-driven
   LTP.
4. **Homogeneous agents.** With shared MAPPO parameters and identical
   reward structures, there's nothing in the task that *should* drive
   bonds toward differentiation. LBF-3p-3f doesn't have an asymmetric
   structure for the bonds to capture.

(1) and (4) reinforce the §3.2 reading: this environment is symmetric;
the Hebbian module isn't broken in any obvious way, it simply has no
asymmetric signal to fit. (2) and (3) suggest the update rule itself
may also be biased toward symmetry, but disentangling that from (1)/(4)
requires running on an asymmetric environment.

### 5.6 Files

| | |
|---|---|
| Bonds trajectories per run | [`bonds/<run_id>/seed_<n>.jsonl`](../bonds/) |
| Bonds analysis script | [`scripts/analysis/tier9_bonds.py`](../scripts/analysis/tier9_bonds.py) |
| Bond magnitude trajectory plot | [`scripts/analysis/out/tier9_bonds_trajectory.png`](../scripts/analysis/out/tier9_bonds_trajectory.png) |
| Sparsity + asymmetry plots | [`scripts/analysis/out/tier9_bonds_meta.png`](../scripts/analysis/out/tier9_bonds_meta.png) |
| Per-seed final-W heatmaps | [`scripts/analysis/out/tier9_bonds_heatmaps.png`](../scripts/analysis/out/tier9_bonds_heatmaps.png) |

---

## 6. Reproducing this report

```
# On the cluster:
cd /scratch/$USER/WiredTogether/hebbian-marl
sbatch --array=0-14 scripts/slurm/hebb_tier9.sh

# After all 15 array tasks complete, on the local machine:
scp -r acmarcu@login.delftblue.tudelft.nl:/scratch/acmarcu/WiredTogether/runs/hebbian-marl/logs/* \
       ./logs/

python scripts/analysis/tier9_analysis.py
```

Manifest entry: [`hebbian-marl/scripts/experiments.yaml`](../hebbian-marl/scripts/experiments.yaml) tier 9 block.
SLURM script: [`hebbian-marl/scripts/slurm/hebb_tier9.sh`](../hebbian-marl/scripts/slurm/hebb_tier9.sh).
