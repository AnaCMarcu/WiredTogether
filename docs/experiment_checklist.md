# Remaining experiments — Hebbian 2.0 re-run checklist

Status as of **2026-09-16**, branch `submission`, verified against **`$REPO/runs/` on
DAIC** — not against `runs_from_daic/`. The local dataset lags the cluster badly right
now (the whole `pareto_social_3f` sweep is on the cluster and absent locally), so every
count below is cluster-side. Seeds are counted only from directories matching
`seed_<digits>` exactly.

**Two data-integrity traps, both live:**

1. OOM-killed runs are parked as `seed_<N>.oom` and **still contain a
   `final_metrics.json`**. `make_results.load_runs` globs `seed_*/final_metrics.json`,
   so copying an arm directory wholesale pools truncated runs into the tables silently.
   **Copy per-seed, never `scp -r` an arm dir.**
2. A finished run on the cluster is invisible to any local check. Confirm on the login
   node before concluding an arm is unrun.

## What "Hebbian 2.0" means here

```
--hebbian-mode three_factor  --hebbian-eta-0 0.001  --hebbian-decay 0.001
--hebbian-eligibility-rho 0.9  --hebbian-coact-floor 0.25
--hebbian-death-ltd 0.05  --hebbian-death-cap 10
--hebbian-reward-norm 50  --hebbian-gamma 0.2
```

**Trap:** `new_exp_0_gemma_hebbian3f` (6 seeds) is `three_factor` **without** the signed
death LTD. It is *not* Hebbian 2.0. The Gemma-E4B Hebbian 2.0 row comes from
`new_exp_0_gemma_si3f8`, which doubles as the interval-8 point of the social Pareto.

> **Superseded 2026-09-23.** The Gemma-E4B +plast. row of `tab:final_comparison` is
> `new_exp_0_gemma_hebbian3f` after all, chosen for its 6 seeds over `si3f8`'s 3. The two
> differ in exactly one hyperparameter (`hebbian_death_ltd`: 0.0 vs 0.05), verified from
> `config.json`. `si3f8` becomes the death-LTD ablation and keeps its queued parity seeds.
> The paper's "Rule versions" paragraph must stop claiming the signed failure-decay term
> for *every* +trace row — row (i) is now the exception.

---

## Table 1 — RQ1 main comparison (`tab:final_comparison`)

RL "+Heb2.0" always means the social-replay arm (exp36/exp37); diffusion-only rows dropped.

| Row | Arm dir | Group | Cluster | Note |
|---|---|---|---|---|
| Qwen-2B | `exp01_llm_2b` | `legacy` | ✅ 6/6 | |
| Qwen-2B + Heb2.0 | `exp34_llm_2b_three_factor_fdecay` | `medium_runs` | ⚠️ 3/6 | 3 queued |
| Qwen-9B | `exp02_llm_9b` | `legacy` | ✅ 6/6 | |
| Qwen-9B + Heb2.0 | `exp35_llm_9b_three_factor_fdecay` | `medium_runs` | ⚠️ 3/6 | 3 queued |
| Gemma-E4B | `new_exp_0_gemma_base` | `new_exp_0_gemma` | ✅ 6/6 | |
| **Gemma-E4B + Heb2.0** | `new_exp_0_gemma_si3f8` | `pareto_social_3f` | **✅ 3/3** | **done — was marked 0/3** |
| Gemma-E4B + Central Orch. | `new_exp_0_gemma_orch_villager_advisory` | `orchestrator` | ✅ 6/6 | |
| Qwen-2B MAPPO | `exp03_mappo` | `legacy` | ✅ 6/6 | |
| Qwen-2B MAPPO + Heb2.0 + SR | `exp36_mappo_hebbian_replay_3f` | `social_replay_3f_qwen` | ❌ 0/6 | group absent; **6 queued** |
| **Qwen-2B IPPO** | `exp04_ippo` | `legacy` | **⚠️ 4/6** | **1011/1213 OOM-killed** |
| Qwen-2B IPPO + Heb2.0 + SR | `exp37_ippo_hebbian_replay_3f` | `social_replay_3f_qwen` | ❌ 0/6 | **6 queued** |
| Gemma-E4B MAPPO | `exp03_mappo` | `gemma4` | ✅ 6/6 | |
| Gemma-E4B MAPPO + Heb2.0 + SR | `exp36_…_3f` | `social_replay_3f_gemma4` | ❌ 0/6 | **6 queued** |
| Gemma-E4B IPPO | `exp04_ippo` | `gemma4` | ✅ 6/6 | |
| Gemma-E4B IPPO + Heb2.0 + SR | `exp37_…_3f` | `social_replay_3f_gemma4` | ❌ 0/6 | **6 queued** |

### To run

- [x] ~~T1.1 — Gemma-E4B + Heb2.0~~ **Complete.** `si3f8` has 42/123/456 on the cluster.
      Not yet copied to `runs_from_daic/`.
- [x] ~~T1.2 — social-replay 3f, both lanes~~ **Submitted at SIX seeds: 24 jobs**
      (2 arms × 2 lanes × {42,123,456,789,1011,1213}), all pending. These four rows
      will therefore land at n=6, matching their baselines — no parity gap.
- [ ] **T1.3 — Qwen seed extension (6 jobs).** Submitted as `medium_runs-exp3{4,5}_…_s{789,1011,1213}`, pending.
- [ ] **T1.4 — exp04_ippo OOM re-runs (2 jobs).** Seeds 1011 and 1213 died at 32 GB.
      Row (h) is at n=4 against n=6 baselines until these land.
      ```
      for s in 1011 1213; do SEED=$s RUN_GROUP=legacy EPISODES=3 MAX_STEPS=1000 \
        WANDB_PROJECT=medium_wired_together \
        MODEL_2B=$W/models/Qwen3.5-2B MODEL_9B=$W/models/Qwen3.5-9B \
        sbatch --mem=64GB --job-name=legacy-exp04_ippo_s$s exp04_ippo.sbatch; done
      ```
- [ ] **T1.5 — seed parity for Heb2.0.** `si3f8` is 3/3 while Gemma-E4B base is 6/6.
      Either extend si3f8 to 6 (`INTERVALS="8" SEEDS="789 1011 1213" bash submit_pareto_social_3f.sh`)
      or report the Gemma row at n=3 and say so. **The n=3→n=6 lesson from
      `new_exp_0_gemma_hebbian3f` (a +2.3 coop gain that vanished at 6 seeds) argues for
      extending.**

### Analysis change (no compute)

- [ ] `analysis/make_final_table.py`: add a `"replay_3f"` entry to `RL_HEB_ARMS`
      pointing at exp36/exp37; add the Gemma RL rows (ROWS is Qwen-only for RL);
      repoint Gemma-E4B+Heb from `new_exp_0_gemma_hebbian` to `new_exp_0_gemma_si3f8`.
      `analysis/make_final_table_extended.py` already covers the extra arms.

---

## Table 2 — RQ2 co-firing channels (`tab:cofire_main`)

Hebbian 2.0 + delivery-symmetric wiring, into `runs/cofiring_bidi_3f/`.
**Submitted and in flight: 1 finished, 12 running, 8 pending.**

| Table row | Arm | Cluster |
|---|---|---|
| (a) None | `exp28_cofire_null` | ❌ 0/3 (pending) |
| (b) Obs | `exp21_cofire_pro` | ❌ 0/3 (running) |
| (c) Imit | `exp22_cofire_pri` | ❌ 0/3 (running) |
| (d) Comm | `exp20_cofire_prc` | ⚠️ **1/3** (seed 42 done) |
| (e) Comm+Obs | `exp29_cofire_prco` | ❌ 0/3 (running) |
| (f) Comm+Obs+Imit | `exp23_cofire_prcoi` | ❌ 0/3 (running) |
| — anchor (not a paper row) | `exp27_cofire_anchor` | ❌ 0/3 (pending) |

- [x] ~~T2.1/T2.2/T2.3 — submit~~ **Done**; the full 21 are queued or running.
- [ ] **T2.4 — verify the rule landed on the first finished run.** Do this now, not in a
      week: if it says `reward_modulated` the whole sweep is a repeat of `cofiring_bidi`.
      ```
      grep -o '"hebbian_mode": "[a-z_]*"' \
        $REPO/runs/cofiring_bidi_3f/exp20_cofire_prc/seed_42/config.json
      ```

---

## Table 3 — RQ3 transplant + memory × bond ablation (`tab:transplant_main`)

Into `runs/pair_bonding_3f/` — **group does not exist yet, 0/15.**
**All five cells are queued** (15 jobs, 3 seeds each): `expB-merged-transplant-3f`,
`expB-merged-shuffled-3f`, `expB-memory-only-3f`, `expB-bond-only-3f`,
`expB-neither-3f`.

| Cell | Memory | Bond | Arm | Cluster |
|---|---|---|---|---|
| A — Co-fired Partner | retained | retained | `expB_merged_transplant_3f` | ❌ 0/3 (queued) |
| A′ — Re-paired Partner | fabricated | retained | `expB_merged_shuffled_3f` | ❌ 0/3 (queued) |
| B — memory only | retained | reset | `expB_memory_only_3f` | ❌ 0/3 (queued) |
| C — bond only | reset | retained | `expB_bond_only_3f` | ❌ 0/3 (queued) |
| D — neither | reset | reset | `expB_neither_3f` | ❌ 0/3 (queued) |

- [x] ~~T3.1 — submit all five cells~~ **Done**, 15 jobs pending. All five move
      together, as required: a `three_factor` cell B read against a `reward_modulated`
      cell A would measure the rule, not the factor.
- [ ] **T3.2 — confirm the uniform-W prerequisite ran.** Cells B and D need the
      magnitude-matched flat W (`submit_transplant_2x2_3f.sh uniform`). Check the
      artifact exists before the queued B/D jobs start, or they will use the wrong W.

Old-rule reference on disk: `pair_bonding/expB_merged_transplant` 4 seeds,
`expB_merged_shuffled` 3, `expA_pair_bonding` 9. Phase A is **not** re-run — the
transplanted W and memories still come from the `reward_modulated` pair runs. Say so in
the paper: the bonds handed to Phase B grew under the old rule.

---

## Pareto curves

### P1 — Agent-count scaling (N ∈ {2,3,4,5,6,9}) — **nearly done**

`runs/agent_scaling_3f/`, 500 steps. Base arm complete and rule-independent.

| N | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|
| seeds | ✅ 3 | ✅ 3 | ⚠️ **2** | ✅ 3 | ✅ 3 | ⏳ 0 (3 running) |

- [x] ~~P1.1/P1.2 — submit~~ **Done.** 14/18 finished, N=9 running (2d12h elapsed, 72 h wall).
- [ ] **P1.3 — N=4 seed 456 is missing.** Resubmit: `NS="4" SEEDS="456" bash submit_agent_scaling_3f.sh`
- The 1000-step variant (`agent_scaling_3f_1k`) was submitted and cancelled. Out of scope;
  the sweep stays at 500 steps to match the base arm.

### P2 — Social-module interval — **nearly done**

| Point | Arm | Cluster |
|---|---|---|
| x = 0 | `new_exp_0_gemma_base` | ✅ 6/6 (rule-free) |
| 2 | `new_exp_0_gemma_si3f2` | ⚠️ **2/3** (456 missing) |
| 8 | `new_exp_0_gemma_si3f8` | ✅ 3/3 — shared with T1.1 |
| 20 | `new_exp_0_gemma_si3f20` | ✅ 3/3 |
| 50 | `new_exp_0_gemma_si3f50` | ✅ 3/3 |
| 100 | `new_exp_0_gemma_si3f100` | ✅ 3/3 |
| 200 (added) | `new_exp_0_gemma_si3f200` | ⚠️ 1/3 (42, 123 missing) |
| 500 (added) | `new_exp_0_gemma_si3f500` | ⚠️ 2/3 (123 missing) |

- [x] ~~P2.1/P2.2 — submit~~ **Done.** 17/24 across all seven arms.
- [ ] **P2.3 — backfill the 4 stragglers.** All five earlier deaths were `TIMEOUT` at the
      36 h `medium` wall, so resubmit with headroom:
      `INTERVALS="2 200 500" QOS=long TIME=72:00:00 bash submit_pareto_social_3f.sh`
      (idempotent — it queues exactly 2/456, 200/42, 200/123, 500/123.)

### P3 — Backbone capability (Gemma E2B ↔ 12B) — **almost done, and over-submitted**

`runs/pareto_gemma4_3f/`. Base arms complete at 6 seeds and rule-independent.

| Arm | Cluster |
|---|---|
| `pareto_12b_hebbian` | ✅ 3/3 |
| `pareto_e2b_hebbian` | ⚠️ 2/3 (**42 missing**) |

- [ ] **P3.1 — ⚠️ CANCEL 5 OF THE 6 QUEUED PARETO JOBS.** Jobs 12892883–12892888 were
      submitted as a bare loop over both sizes × 3 seeds. `new_exp_pareto_3f.sbatch` has
      **no idempotent skip**, so five of them will re-run and **overwrite finished runs**.
      Only E2B seed 42 is needed. Job-ID→arm mapping can't be read from `scontrol`
      (env vars aren't recorded), so cancel all six and resubmit the one:
      ```
      scancel 12892883 12892884 12892885 12892886 12892887 12892888
      MODEL_SIZE=e2b SEED=42 sbatch --mem=32G --qos=long --time=48:00:00 \
        new_exp_pareto_3f.sbatch
      ```
- [ ] **Decide the E4B mid-point.** Reuse `new_exp_0_gemma_si3f8` (different group, same
      config) or run `MODEL_SIZE=e4b` into `pareto_gemma4_3f`. Reusing is cheaper;
      `make_pareto_grid.py` must be told where to look.
- Old-rule comparison on disk: `pareto_gemma4/pareto_{e2b,12b}_hebbian`, 6 seeds each.

---

## Orchestrator

- [x] **Table 1 row** — `new_exp_0_gemma_orch_villager_advisory`, 6 seeds, complete. The
      orchestrator replaces the Hebbian coupling (mutually exclusive flags), carries no
      Hebbian rule, and is untouched by the 2.0 transposition. Stands as published.
- [ ] **N=6 scaling point — now running** (jobs 12892879–81, `scale_gemma_orch.sbatch`,
      villager/advisory cadence 8, `--team-scaling --ch4-mob-count 3`, 3 × 1000 steps →
      `runs/agent_scaling_orch/`, group not yet created). Previously recorded here as
      "not being run" — that decision was reversed.
      - Untested combination: `--team-scaling` N-templates the prompts while the
        orchestrator builds its own. If all three fail early, that interaction is why.
      - Horizon is 1000 steps, so it does **not** compare directly to the 500-step P1
        curve. It compares to `agent_scaling_3f_1k/scale_gemma_hebbian_n6`, which was
        cancelled — so as submitted this point has **no matched Hebbian counterpart**.
- [ ] No orchestrator control for Table 2 or Table 3. Not being run.

---

## Totals — remaining work

| Block | Finished | In flight | Still to submit |
|---|---|---|---|
| Table 1 | 8 rows complete | 30 (T1.2 ×24 + T1.3 ×6) | 2 (exp04 OOM) + 3 optional (si3f8 → 6 seeds) |
| Table 2 | 1 run | 20 | 0 |
| Table 3 | 0 | 15 (all five cells) | 0 |
| P1 agent count | 14/18 | 3 (N=9) | 1 (N=4/456) |
| P2 social interval | 17/21 | 0 | 4 stragglers |
| P3 backbone | 5/6 | 6 — **5 destructive, cancel** | 1 (E2B/42) |
| Orchestrator | 6 (Table 1 row) | 3 (N=6) | 0 |

The original estimate of 84 jobs is obsolete: most of it is submitted, running or
done — **77 jobs are in the queue**. Only **8 jobs of genuinely new work remain**
(2 exp04 OOM re-runs, 1 N=4 seed, 4 P2 stragglers, 1 E2B seed), plus one cancellation
that prevents data loss and 4 optional parity jobs.

## Standing checks

- [ ] `auks -a` **weekly**. The credential dies 7 days after *submit*, not after start.
      With ~57 jobs pending behind `QOSMaxGRESPerUser`, a large share will wait longer
      than that and die with `EKEYEXPIRED`, writing no `run.log` at all — invisible in
      `runs_from_daic`. Last renewed 2026-09-16.
- [ ] Verify the rule per suite as each one lands:
      `grep -o '"hebbian_mode": "[a-z_]*"' runs/<group>/*/seed_*/config.json | sort -u`
      → `three_factor` everywhere, and `hebbian_death_ltd` present.
- [ ] Use the full-width queue format; the default truncates names to 8 characters and
      `scale-gemma-orch` is indistinguishable from `scale-gemma`:
      `squeue -u $USER -o "%.10i %.44j %.8T %.12M %.12l %R"`
- [ ] Sync per-seed, never per-arm (see the `.oom` trap above).

---

## Communication-budget sweep (added 2026-09-22)

Per-agent, per-episode token budget the agent is told about; exhausted → muted in code.
Design + empirical anchors: `src/mindforge/env/comm_budget.py` docstring and
`analysis/calibrate_comm_budget.py`. Launchers: `hpc/daic/experiments/budget_gemma.sbatch`,
`submit_comm_budget.sh`. Figure: `analysis/make_budget_fig.py`.

```
N ∈ {3,5,7} × budget ∈ {0, 600, 2600, 10400} tokens × arm ∈ {base, hebbian}   (3 ep × 1000 steps)
--comm-budget-tokens B --comm-budget-msg-cap 32 --comm-reward-scale 0   (both arms)
hebbian = Hebbian 2.0 block of scale_gemma_3f.sbatch; base = no Hebbian, no social module
```

| Level | Tokens | ≈ msgs / agent / ep | Share of 1000 steps |
|---|---|---|---|
| zero | 0 | 0 | 0 % |
| low | 600 | 50 | 5 % (≈ the measured request rate) |
| medium | 2600 | 200 | 20 % |
| high | 10400 | 800 | 80 % (binds only for every-step chatter) |

Token values PINNED 2026-09-22 with the real tokenizer (p50 13 tok/msg); re-pin only for another backbone:
`python analysis/calibrate_comm_budget.py --model $WORKSPACE/models/gemma-4-E4B-it`
(message counts are the invariant). N=3 free references already exist:
`new_exp_0_gemma/{new_exp_0_gemma_base, new_exp_0_gemma_hebbian3f}` (Gemma-E4B, 3 × 1000).

- [ ] **B0 calibrate** the ladder on DAIC (login node, no GPU) and, if p50 ≠ 16, submit with
      `BUDGETS="0 <low> <med> <high>"`.
- [ ] **B1 smoke** — `SMOKE=1 bash submit_comm_budget.sh` (N=3, b=800, both arms, 1 ep × 150).
      Check: `messages.jsonl` has `tokens_model/charged/budget_left/truncated`; `event_log.jsonl`
      has `comm_budget_exhausted` (+ `comm_budget_blocked` afterwards); `summary.json` has the
      `comm_budget` block; `llm_logs/action_selection.log` shows the budget line in `kwargs`.
- [ ] **B2 pilot** — `SEEDS="42" bash submit_comm_budget.sh` → 24 jobs
      (N=3 → long 72 h / 32 GB, N=5 → long 96 h / 48 GB, N=7 → long 120 h / 96 GB; ≈ 1,300 GPU-h).
- [ ] **B3 pull + figure** — `bash pull_new.sh` → `runs_from_daic/comm_budget/`, then
      `python analysis/make_budget_fig.py` (coop % + milestone % vs budget per N, both arms;
      utilisation curves; per-cell CSV).
- [ ] **B4 extend** — `SEEDS="123 456" bash submit_comm_budget.sh` → +48 jobs (finished cells skipped).

Hypotheses to read the pilot against: Hebbian advantage largest at low/medium, absent at zero
and high; base exhausts earlier than Hebbian; request share of sent messages rises as budget
falls; the bond graph still forms at low via proximity/co-action co-firing.
