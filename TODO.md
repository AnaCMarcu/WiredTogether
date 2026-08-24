# Experiment status

Shared tracker — update when you launch or finish a run group.
Last updated: **2026-08-21**. A run counts as done iff `final_metrics.json` exists.

| Experiment | Run group | Conditions | Seeds | State |
|---|---|---|---|---|
| Main results | `medium_runs` | 11 arms: LLM-2B, LLM-9B, MAPPO, IPPO, MAPPO+Heb, IPPO+Heb, 2B/9B social-prompt, allied all/pair/none | 6 (exp04: 4, exp05: 5) | **Done** — in `results.tex` |
| Main results, 2k steps | `medium_2k` | 3 arms: allied all / pair / none | 2 | **Done** |
| Exp. 0 — Gemma-4 anchors | `new_exp_0_gemma` | 2 arms: base, +Hebbian | 6 | **Done** |
| Pareto — social interval | `pareto_social` | 4 arms: Δ=2, 20, 50, 100 (anchors Δ=8 and no-module = Exp. 0) | 3 | 8/12 done; **4 re-running** at 72h (si2/42, si2/123, si20/123, si100/123 timed out at 36h) |
| Pareto — model size | `pareto_gemma4` | 8 arms: E2B, 12B, 26B-A4B, 31B × base/+Hebbian | 6* | **Running** (E2B/12B) — 16 running, 5 queued. **26B-A4B + 31B blocked**: every `generate()` fails with NaN/Inf logits (`torch.multinomial`), found in the 2026-08-20 smoke; correlates with `device_map="auto"` 2-GPU sharding |
| NaN-logits probe | `pareto_probe` | 31B (sdpa_auto, eager_auto), 12B (sdpa_auto) | — | **Re-running** on the rebuilt image (2026-08-21, torch ≥ 2.6) — results pending. First attempt was inconclusive: both jobs aborted before reaching the NaN (31B on `torch<2.6`, which Gemma-4 vision masks require; 12B on an image-token mismatch, 266 vs 280) |
| Agent-count scaling | `agent_scaling` | 12 arms: N ∈ {2,3,4,5,6,9} × base/+Hebbian, 500 steps | 3 | **Queued** — 38 jobs, none started |
| Exp. 2 — co-firing (v1) | `cofiring` | 6 arms: prc, pro, pri, prcoi, anchor, null | 3 | **Done** |
| Exp. 2 — co-firing (final) | `cofiring_final` | 7 arms: + prco | 3 | **Done** |
| Exp. 2 — co-firing (no reward) | `cofiring_noreward` | 5 arms: prc, pro, pri, prcoi, prco | 3 | **Done** |
| Exp. 2 — co-firing (act reward) | `cofiring_actrew` | 4 arms: pro, pri, prco, prcoi | 3 | **Queued** — 12 jobs |
| Social replay (Eq. 7) | — | 2 arms: exp30 MAPPO+Heb, exp31 IPPO+Heb | 3 | **Queued** — 12 jobs |
| RL arms re-run (Gemma-4 PEFT fix) | — | 4 arms: exp03 MAPPO, exp04 IPPO, exp05 MAPPO+Heb, exp06 IPPO+Heb | 3 | **Queued** — 12 jobs |
| Orchestrator baseline (O2) | `orchestrator` | advisory + bias couplings | 3 | **Queued** — 1 job |
| Pair transplant | `pair_bonding` | expA pair-bonding; expB merged shuffled / transplant | 9 / 3 / 3 | 14/15 done (Gemma), 12/14 (Qwen lane) |

Analysis outputs: `paper_assets_pareto_social/` (`make_pareto_social_fig.py`),
`paper_assets_medium*/`, `paper_assets_cofiring*/`, `paper_assets_transplant/`.

**Notes**

- Queue counts are the `squeue` snapshot of 2026-08-21 ~22:57; seed counts
  marked `*` are launcher defaults, not confirmed against the queue.
- Probe: `scripts/probe_nan_logits.py` + `hpc/daic/probe_nan.sbatch`, 5 configs
  (sdpa/eager × auto/single, fp32_auto) to separate sharding from
  checkpoint-specific causes. Reading the result: if `eager_*` is clean the
  cause is the sliding-window/sdpa mask; if `*_single` is clean it is the
  2-GPU sharding; if only `fp32_auto` is clean it is bf16 overflow.
- Use `QOS=long TIME=72:00:00` — 36h is not enough (successful `pareto_social`
  runs took 19–35h, and seed_123 needs ~20% more LLM calls than any other seed).
- Before re-running a failed seed, move its run dir aside: `log.txt` and
  `llm_logs/*.log` are append-mode, so a rerun in place silently doubles the
  FLOPs that `scripts/compute_flops.py` reports.
