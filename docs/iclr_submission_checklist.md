# ICLR 2027 anonymised release — checklist

Branch: `iclr-anon` (cut from `submission` on 2026-09-24; carries the uncommitted
`RL_HEB_ARMS = "replay"` edit). Reference paper: **WIRED_v6.pdf** — the tex under
`Downloads\WIRED_TOGETHER_revised` is one revision behind it (still has the +inst. appendix and
the three-agent Figure 4), so "in the paper" below always means the v6 PDF.

Legend: `[x]` done · `[ ]` to do · `[?]` decision needed.

**Decisions (2026-09-25):**

- Keep the communication-budget module, its launchers, its analysis scripts and its tests.
- Orchestrator: villager only, and the default. **Done** in `9ce12ac`.
- CLI flags: investigated flag by flag in [cli_flag_audit.md](cli_flag_audit.md).
- `figures/` and `paper_assets/` do not ship in the code release (§3.7).

---

## 0. Where the anonymised artefacts go

| Artefact | Host | Why |
|---|---|---|
| Code | **Anonymous GitHub** — <https://anonymous.4open.science> — mirroring a **new private** GitHub repo that holds only the orphan release branch (§1) | Hides the origin URL, author, and history; supports private sources; term-replacement list catches stragglers; expiry date is set per mirror. Never mirror `AnaCMarcu/WiredTogether` or `tapri-lab/wired-together` directly — both names identify the authors. |
| Project page (`site/`) | Same Anonymous GitHub mirror (it can serve the repo's GitHub Pages site), **or** GitHub Pages from a throwaway account with a neutral name | `.github/workflows/pages.yml` on the personal repo would publish under `anacmarcu.github.io` — do not use it during review. |
| Run artefacts (§4) | **OSF** project with an *anonymised view-only link* (hides contributors), holding the `core` archive (~300 MB packed) and the `logs` archive (~2 GB packed) | OpenReview supplementary is capped at 100 MB; Zenodo has no anonymous mode. At camera-ready: Zenodo DOI + the lab release repo. |
| OpenReview supplementary zip | Code snapshot (`git archive` of the release branch, ~90 MB because VoxeLibre is vendored) | Reviewers who never click a link still get the code. If it must stay under 100 MB, drop `src/marl_craftium/craftium-envs/wire/games/VoxeLibre/` from the zip and point to the mirror. |
| Craftium engine fork | **Do not link** `AnaCMarcu/craftium_wired_together`. Ship `craftium.patch` (the fork is upstream `mikelma/craftium` @ `e8290cb` + a 14-line diff in `craftium/craftium_env.py` and `craftium/minetest.py`) | A personal fork URL de-anonymises; a patch does not. |

Also: set `AnaCMarcu/WiredTogether` and `AnaCMarcu/craftium_wired_together` to **private for the
review period** — GitHub code search on a phrase like "Wired Inter-agent Reasoning Evaluation"
would otherwise find the source repo in one query.

---

## 1. Branch and history

- [x] `git checkout -b iclr-anon submission`
- [x] Commit the pending edits on `iclr-anon` (`f5e4e88`).
- [x] Track the fourteen scripts behind Tables 1–3, 9–12 and Figures 4, 6–8, 10–13, which had
      never been committed (`5506746`). Their imports close over tracked files.
- [ ] Before the orphan snapshot, the working tree must hold **only** release files: the
      snapshot recipe below uses `git add -A`, which would sweep in every untracked file in the
      tree (pull scripts, `wt_pull_pair_bonding_3f.tgz`, loose figures). Either delete them first
      or build the snapshot with `git archive iclr-anon | tar -x` into a clean directory.
- [ ] Do the pruning (§3) and scrubbing (§2) as ordinary commits on `iclr-anon` — this branch is for
      *us*, history is fine here.
- [ ] Build the **release snapshot** as an orphan branch with one anonymous commit (the 556 commits
      by `A.C.Marcu-1@student.tudelft.nl` must not travel with it):

  ```bash
  git checkout --orphan iclr-anon-release
  git rm -r --cached . -q && git add -A
  git -c user.name="Anonymous" -c user.email="anonymous@example.com" \
      commit -q -m "WIRE: anonymised code release for ICLR 2027 review"
  git remote add anon git@github.com:<throwaway-or-private>/wire-anon.git   # NEW private repo
  git push anon iclr-anon-release:main
  ```

- [ ] Point Anonymous GitHub at that repo; add the term list from §2; set expiry ≥ decision date + rebuttal.
- [ ] Open the mirror in a private browser window and read `README.md`, one launcher, one analysis
      script, `LICENSE`, and the site — term replacement can also mangle identifiers (e.g. a
      replacement for `delft` would hit `hpc/delft_blue/` paths if any survived).

---

## 2. Scrub identifying information

Verification command (must print nothing on the release branch):

```bash
git grep -niE "marcu|acmarcu|tudelft|tu delft|delft|daic|snellius|tapri|anamarcu|student|thesis|C:\\\\Users|Users/marcu|46790150" -- . ':!src/marl_craftium/craftium-envs'
```

| File(s) | What | Fix |
|---|---|---|
| `LICENSE` | `Copyright (c) 2026 Ana Marcu` | `Copyright (c) 2026 Anonymous authors` (restore at camera-ready) |
| `pyproject.toml` | `authors = ["Ana Cristiana Marcu"]` | `authors = ["Anonymous"]` |
| `README.md` | `git clone https://github.com/AnaCMarcu/craftium_wired_together.git` | upstream clone + `git apply craftium.patch` (§3.8) |
| `hpc/daic/experiments/_common.sh`, every `*.sbatch`, `build_image*.sbatch`, `download_gemma4.*`, `*.def` | `WORKSPACE=/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu`, `--bind /tudelft.net` | `WORKSPACE="${WT_WORKSPACE:?set to your cluster workspace}"`; drop the bind or make it `${WT_BIND:-}` |
| `hpc/**`, `docs/**`, `environment.yml`, `analysis/**` | "DAIC", "DelftBlue", "Snellius", "TU Delft" in prose and comments | "the cluster" / "SLURM cluster"; delete the DelftBlue and Snellius trees (§3.4) |
| `hpc/daic/experiments/bad_gpu_nodes.txt` + its use in `_common.sh` | cluster node names | delete file, remove the `MIN_GPU_MEM_MIB`/bad-node bookkeeping or leave it reading an empty list |
| `analysis/wandb_compute_budget.py` | W&B entity `anamarcu/...` | delete (no table in v6 uses it) |
| `hpc/**` W&B blocks | project names are fine; ensure no entity/API key | keep `WANDB=0` default in the release |
| `.github/workflows/pages.yml` | deploys to the personal Pages site | not part of the code release (§3.7) |
| `src/mindforge/cli.py:43`, `agent_modules/skill_manager.py:27`, `tools/merge_pair_runs.py:259` | "DAIC", "DelftBlue" in comments/help text | "the cluster" |
| `src/mindforge/cli.py:103` | W&B project default `wired-together` | leave (project names are not identifying) or rename |
| `tests/test_paper_defaults.py` docstring | "thesis paper Tables 6 & 7" | "the paper's hyper-parameter tables" |
| `docs/experiment_checklist.md` | cluster job ids, queue names, `auks` | delete (internal) |
| `site/assets/paper.pdf` | must be the v6 anonymous build | replace |
| git remotes | `origin` = AnaCMarcu, `release` = tapri-lab | never push `iclr-anon*` to either |

Not identifying, keep: SLURM/Apptainer mechanics, model names, W&B project names.

---

## 3. Prune to what the paper reports

Rule: a file stays if a v6 table/figure, a v6 condition's launcher, or a test of surviving code
reaches it. Everything else goes (it stays on `submission`, nothing is lost).

### 3.1 v6 provenance map (what must keep working)

| Paper item | Script | Runs it reads |
|---|---|---|
| Fig 1, 2, 3 | hand-made (`fig1_graph_formation.png`, `fig2_couplings.png`, `WIRE_FINAL.png`) | — |
| Fig 4 | `make_counterfactual_compact_n6.py` | `agent_scaling_orch/scale_gemma_orch_villager_n6/seed_42`, `agent_scaling_3f/scale_gemma_hebbian_n6/seed_456` |
| Fig 5 | `make_chamber_gallery.py` (frames extracted by `make_final_figures.py`) | `new_exp_0_gemma_hebbian3f/seed_123`, `medium_runs/exp08…/seed_42` frames, `exp02…/seed_789` |
| Fig 6 (`social_frontier_all`) | `make_agent_completion_figs.py` | `new_exp_0_gemma_base`, `pareto_social_3f/*` via the `compute/_social_3f_{anchors,sweep}` symlink views |
| Fig 7 (`rq2_pareto_{coop,solo}`) | `make_agent_completion_figs.py` → `make_rq2_pareto_fig.py`, `cofire_table.py` | `cofiring_bidi_3f/*` |
| Fig 8 (`pareto_perception_all`, `pareto_partner_all`) | `make_agent_completion_figs.py` (+ `make_pareto_grid.load_beliefs`, `make_pareto_fig.SIZES`) | `compute/_pareto_3f_view` symlinks → `medium_runs/exp01,02,34,35`, `new_exp_0_gemma/base,hebbian3f`, `pareto_gemma4/*_base`, `pareto_gemma4_3f/*_hebbian`; **and** `paper_assets/perception_3f/beliefs_3f.csv` (built by `make_beliefs_3f_view.py` from `analysis/qualitative/out_*/tables/beliefs.csv`) |
| Fig 9 (`wide_gemma_hebbian3f`) | `make_final_figures.py` arm `gemma3f_seed123` | `new_exp_0_gemma_hebbian3f/seed_123` (+ its `.mp4` recordings for the frames) |
| Fig 10, 11 | `make_counterfactual_n6.py` | as Fig 4 |
| Fig 12, 13 (= `counterfactual_{a,b}`) | `make_counterfactual_story.py` | `orchestrator/…villager_advisory/seed_42`, `pareto_social_3f/new_exp_0_gemma_si3f8/seed_42` |
| Fig 14 (`team_tenure_{a,b}`) | `make_team_tenure.py` (hard-codes `RUNS/"orchestrator/…"` — switch to `paths.group()`) | orchestrator + `new_exp_0_gemma_hebbian3f`, `seed_456` |
| Tables 1, 2, 3, 12 | `make_agent_completion_tables.py` (imports `make_final_table`, `_extended`, `_latex`, `cofire_table`, `make_transplant_tables`, `make_results`) | Table-1 arms below; `cofiring_bidi_3f`; `pair_bonding_3f`; `medium_runs/exp09-11` + `medium_2k/exp09-11` |
| Table 9 | `make_steps_table_pct.py` (imports `make_steps_table_latex.py`) | Table-1 arms |
| Table 10 | `make_bond_behaviour_rho.py` (imports `qual_lib.bonds_behavior`) | `exp34`, `exp35`, `new_exp_0_gemma_hebbian3f` |
| Table 11 | `make_transplant_tables.py` (imports `mindforge.tools.analyze_wiring`) | `pair_bonding_3f` |
| Tables 4–8, 13, Algorithms 1–4, prompts | static / pinned by `tests/test_paper_defaults.py`, `test_lua_spec.py`, `test_chamber_facts.py` | — |

Shared dependencies that must stay: `paths.py`, `make_results.py`, `compute_flops.py`,
`make_directive_timelines.py`, `make_final_figures.py`, `make_counterfactual_table.py`,
`make_pareto_fig.py`, `make_pareto_grid.py`, `make_pareto_social_fig.py`,
`replay_hebbian_terms.py`, `prototype_three_factor_rule.py`, `runs_dataset.py`, `analysis/qualitative/`
(`run.py`, `run_3f.py`, `qual_lib/`, README; the `out/annotations/*.jsonl` labels validate the
perception metric — keep, they are small).

### 3.2 `analysis/` — delete

(`calibrate_comm_budget.py` and `make_budget_fig.py` stay with the budget module.)

```
analysis/assist_inference.py
analysis/make_baseline_deltas.py
analysis/make_bond_asymmetry.py
analysis/make_bond_asymmetry_fig.py
analysis/make_bond_asymmetry_report.py
analysis/make_bond_behavior_fig.py
analysis/make_ch2_conditioned_success.py
analysis/make_cofire_excerpts.py
analysis/make_cofire_latex.py
analysis/make_compose_assets.py
analysis/make_coordination_timelines.py
analysis/make_counterfactual_compact.py
analysis/make_counterfactual_fig.py
analysis/make_counterfactual_scan.py
analysis/make_final_figures_callouts.py
analysis/make_final_table_credits.py
analysis/make_iclr_figure.py
analysis/make_main_table_extended.py
analysis/make_mechanistic_figure.py
analysis/make_orchestrator_n6_timeline.py
analysis/make_pareto_delta.py
analysis/make_pareto_perception_fig.py
analysis/make_pareto_perception_pdf.py
analysis/make_plan_vs_completion.py
analysis/make_qualitative_figure.py
analysis/make_qwen_hebbian_analysis.py
analysis/make_scaling_fig.py
analysis/make_social_dynamics.py
analysis/make_story_timelines_multi.py
analysis/make_team_comparison.py
analysis/make_transplant_bonds.py
analysis/make_transplant_excerpts.py
analysis/wandb_compute_budget.py
analysis/site_layout_probe.html
```

`[?]` Site & supplementary tooling — `make_media_clips.py`, `make_hero.py`, `make_site_assets.py`,
`make_site_gallery.py`, `scan_clip_candidates.py`, `mob_visibility.py`, `serve_site.py`,
`make_supplementary.py`. Keep **only** if the project page and its videos ship with the
submission; otherwise move them out of the code release (they are not paper results).
Recommendation: keep `make_supplementary.py` (adjusted) and drop the rest from the code mirror;
the site is deployed from its own folder anyway.

`[?]` `make_counterfactual_scan.py` chose the Fig 12/13 cells (documented in
`make_counterfactual_story.py`). Provenance only — delete unless you want the selection reproducible.

Keep and **adjust**: `make_agent_completion_figs.py` (read the real groups through `paths.group()`
instead of the `compute/_*_3f_view` symlink dirs — symlinks do not survive a tar on every
platform and Windows needs privileges to create them), `make_team_tenure.py` (same),
`make_final_table.py` (drop the `"diffusion"` branch and the exp05–08/exp30–31 rows),
`make_results.py` (drop the registry entries for exp05–08, exp27-less cofiring groups, the
`scale_gemma_*` N≠6 arms, `pareto_*` old-rule arms — keep only arms in §4), `runs_dataset.py`
(§4), `analysis/README.md` (rewrite to the table above).

### 3.3 `hpc/` — keep

```
hpc/README.md                                     (rewrite: one cluster, one container, arm table)
hpc/daic/wiredtogether.def                         (Qwen image)      scrub WORKSPACE
hpc/daic/wiredtogether_gemma4.def                  (Gemma image)     scrub WORKSPACE
hpc/daic/build_image.sbatch, build_image_gemma4.sbatch                scrub
hpc/daic/experiments/_common.sh                                        scrub; drop bad-node list
hpc/daic/experiments/gpu_filter.sh
hpc/daic/experiments/exp01_llm_2b.sbatch … exp04_ippo.sbatch          Table 1 (a)(c)(h)(j)(l)(n)
hpc/daic/experiments/exp09_… exp10_… exp11_…                           Table 12 (with submit_medium2k.sh)
hpc/daic/experiments/exp_qwen_three_factor_fdecay.sbatch               Table 1 (b)(d)
hpc/daic/experiments/new_exp_0_gemma.sbatch (HEBBIAN=0 path)           Table 1 (e)
hpc/daic/experiments/new_exp_0_gemma_three_factor.sbatch               Table 1 (g)
hpc/daic/experiments/new_exp_orchestrator.sbatch + submit_orchestrator.sh   Table 1 (f)
hpc/daic/experiments/exp36_… exp37_… + submit_social_replay_3f.sh      Table 1 (i)(k)(m)(o)
hpc/daic/experiments/submit_gemma4.sh                                  Gemma RL baselines (l)(n)
hpc/daic/experiments/submit_seed_extension.sh                          seeds 789/1011/1213 of exp01–04
hpc/daic/experiments/exp20,21,22,23,27,28,29_cofire_*.sbatch + submit_cofiring_3f.sh   Table 2 / Fig 7
hpc/daic/experiments/expA_pair_bonding.sbatch + submit_transplant.sh (phase A)         Table 3 Phase A
hpc/daic/experiments/expB_*_3f.sbatch (5) + submit_transplant_2x2_3f.sh                Table 3 / 11
hpc/daic/experiments/submit_medium2k.sh                                Table 12 (2,000-step arms)
hpc/daic/experiments/new_exp_0_gemma_si3f.sbatch + submit_pareto_social_3f.sh          Fig 6, Fig 12/13
hpc/daic/experiments/new_exp_pareto.sbatch, new_exp_pareto_3f.sbatch + submit_pareto.sh   Fig 8
hpc/daic/experiments/scale_gemma_3f.sbatch + submit_agent_scaling_3f.sh (N=6 only)     Fig 4, 10, 11
hpc/daic/experiments/scale_gemma_orch.sbatch                           Fig 4, 10
hpc/daic/experiments/budget_gemma.sbatch, submit_comm_budget.sh       communication budget (kept)
```

### 3.4 `hpc/` — delete

```
hpc/delft_blue/                      (entire tree — earlier cluster generation, not in v6)
hpc/snellius/                        (smoke only)
hpc/diagnostics/
hpc/daic/probe_nan.sbatch
hpc/daic/sim_smoke.sbatch
hpc/daic/tmp_cleanup_test.sbatch
hpc/daic/download_gemma4.sbatch, download_gemma4.sh
hpc/daic/build_image_gemma4_torch26.sbatch, wiredtogether_gemma4_torch26.def   (image appears only in its own build log; every v6 run used wiredtogether.sif or wiredtogether_gemma4.sif)
hpc/daic/experiments/bad_gpu_nodes.txt
hpc/daic/experiments/exp05_… exp06_… exp07_… exp08_…                  (old-rule Hebbian arms)
hpc/daic/experiments/exp30_… exp31_… submit_social_replay.sh           (old-rule replay)
hpc/daic/experiments/exp_qwen_three_factor.sbatch                       (exp32/33, no death term)
hpc/daic/experiments/expB_bond_only.sbatch, expB_memory_only.sbatch, expB_merged_shuffled.sbatch,
    expB_merged_transplant.sbatch, expB_neither.sbatch, submit_transplant_2x2.sh   (old-rule Phase B)
hpc/daic/experiments/new_exp_0_gemma_si.sbatch, submit_pareto_social.sh (old-rule interval sweep)
hpc/daic/experiments/scale_gemma.sbatch, submit_agent_scaling.sh        (old-rule team-size sweep)
hpc/daic/experiments/submit_all.sh, submit_seed_extension2.sh           (seeds 1415+ unused in v6)
```

### 3.5 `src/` — from the dead-code sweep (import-graph traced from `multi_agent_craftium.py`)

Four things that looked deletable and are **not**:

- `reward_modulated` is the default Hebbian mode (`src/hebbian/config.py:21`, `cli.py:164`) and
  exp09–11 (Table 12), `expA_pair_bonding` (Phase A) and the base arms pass no `--hebbian-mode`,
  so they ran on it. Keep it. Only `legacy` and `coactivity` are unreachable.
- `src/marl_craftium/openworld_multi_agents.py` **is** the live WIRE env (`marl_craftium/__init__.py:18`
  → `custom_environment_craftium.py:8`). Keep all four `marl_craftium` modules; rename at most.
- `checkpointing.py`'s save half runs every 500 steps and writes the `agent_state/` that the
  transplant consumes (`expA_pair_bonding.sbatch:33`). Only `load_checkpoint` + `--resume*` are
  DelftBlue-chain-only.
- `_metric_plots.py` is a base class of `CraftiumMetric`; `wandb_logger.py` is on by default in
  every launcher. Keep both.

Delete, with the edits each one drags along:

| Feature | Delete | Also edit | Tests affected |
|---|---|---|---|
| Communication budget | **Kept** (decision 2026-09-25) | — | `tests/test_comm_budget.py` stays |
| Hebbian `legacy` + `coactivity` modes | `graph.py:176-368` (`_compute_coactivity`, `_compute_modulator`, `_update_failure_window`, `_compute_sustained_ltd`), legacy branch of `update()` `:795-875`, `get_ltd_heatmap` `:1147-1161`, `:535-536`; config fields `config.py:57-81` (`ltp_lr`, `ltd_lr`, `ltd_threshold`, `base_ltp`, `modulation_beta`, `ltd_sustained_lr`, `failure_*`) | `multi_agent_craftium.py:501-504,654-659,906,2296-2356`; `cli.py:165` choices. **Keep** `eta_minus`, `coop_eps`, `coop_window`, `neg_theta`, `eta_plus`, `reward_norm_R`, `decay`, `eligibility_rho`, `coact_floor`, `eta_minus_death`, `death_cap`, `_windowed_stats`, `_coactivity_gated`, `_engagement`, `_chamber_gate` — three_factor uses them | `test_hebbian_graph_api.py:345-366,385`; `test_hebbian_update.py:47-72,231-233` |
| Orchestrator variants `task`/`social`/`plan`, mode `bias` | **Done** (`9ce12ac`): 2,857 lines removed; three prompt templates, seven CLI flags and the `bias` routing deleted; `--orchestrator-variant` accepts only `villager` and defaults to it | Launchers and `submit_orchestrator.sh` keep the run names `new_exp_0_gemma_orch_villager_advisory` / `scale_gemma_orch_villager_n6` | 50 tests of deleted code removed; 585 pass |
| OpenWorld roles | `prompts/role_hunter.txt`, `role_harvester.txt`, `role_scouter.txt` | `agent_factory.py:26` (`ROLE_NAMES`), `:52-54` (eager load), `:77-96`; `cli.py:507-531` (`--team-mode`, `--homogeneous-role`, `--roles`); `multi_agent_craftium.py:353-365,650-651`; `team_scaling.py:44,91-102,119` (`regroup_teammates` exists only for `role_scouter.txt`). **Keep `role_agent.txt`** (the default role) | `test_team_scaling.py:27,80-83,164-169` |
| Resume chains | `checkpointing.py:153-225` (`load_checkpoint`); `cli.py:542-550,614-621`; `multi_agent_craftium.py:47,410-415,459,709-731,792,2789-2790,2806-2807` | keep `save_checkpoint` and `--checkpoint-interval` (or hard-code 500) | none |
| Token-mode RL | `src/rl_layer/token_opt.py`, `prompts/learning_belief.txt`; `--rl-auto-token-opt`, `--rl-mode` (`cli.py:143-148`); `multi_agent_craftium.py:423,2611-2649`; `rl_layer.py:414-420` | — | none |
| Voxel obs | `--voxel-obs` (`cli.py:551`), `custom_environment_craftium.py:908+`, `openworld_multi_agents.py:49-52,65-68,76-81`, `multi_agent_craftium.py:404,1406,1518` | — | none |
| Old table tools | `src/mindforge/tools/{make_rq3_table,make_topology_horizon_table,make_transplant_table,memory_bond_report,transplant_report}.py` (hard-code `runs_from_daic/...` paths; superseded by `analysis/make_agent_completion_tables.py` and `make_transplant_tables.py`) | keep `pair_transplant.py`, `merge_pair_runs.py`, `analyze_wiring.py`; drop the "regenerate with …" header lines in `paper_assets/transplant/*.tex|md` | `test_analyze_wiring.py`, `test_pair_transplant.py` cover only the keepers |
| Dead odds and ends | `--interpretability` (`cli.py:288-292`, never read); `util.py:310-345` (`visualize_frames`, `autogenImg_to_Pil`); `ippo.py:34 _normalize`; `wandb_logger.is_active`; `graph.py:160-169` `star`/`ring` presets + `--hebbian-hub`; `craftium_metric.py:543,899,902`; `critic.py:70`; `run_layout.py` unused properties; `dag.busy_agents`; `custom_environment_craftium.py:732,867-899`; `_patched_env.step_agent` | — | none |

CLI flags: see [cli_flag_audit.md](cli_flag_audit.md). Of 99 flags, 15 go with the code they
gate (table G there); the rest stay, including 12 that carry a Table 7/8 hyperparameter at its
default. Argparse prefix matching is now off, so a stale flag fails loudly.

`[?]` **The Hebbian CLI defaults are the old single-timescale rule, not Table 8** (mode,
η₀, R, λ, η₋ᵈ, ρ). No reported number is affected — every launcher passes them — but a reviewer
running `--hebbian` alone gets the wrong rule, and `tests/test_paper_defaults.py` pins the old
values. Recommendation and the launchers that must first pin their values explicitly are in the
audit's last section.

Do **not** delete despite looking unused: the PettingZoo `ParallelEnv` API methods in
`openworld_multi_agents.py:143-156`, `custom_agent.produced_message_types` (autogen),
`local_model_client.py:566-595` (`ChatCompletionClient` ABC), `comm_eval.py` / `coop_eval.py`
(invoked lazily from `craftium_metric.py:620,629` on every run).

Identifying strings inside `src/` (the sweep found no names, e-mails or usernames):
`cli.py:43` "DAIC 36h SLURM budget", `skill_manager.py:27` "DelftBlue /scratch",
`merge_pair_runs.py:259` "the DAIC login node", `cli.py:103-104` W&B project default
`wired-together` (rename or leave as opt-in), and the `runs_from_daic/...` defaults in the old
tools above. `tests/` is clean.

### 3.6 `tests/`

- [x] orchestrator tests slimmed to the surviving code (`9ce12ac`); `test_comm_budget.py` stays.
- drop the legacy/coactivity cases in
  `test_hebbian_graph_api.py` and `test_hebbian_update.py`; drop the scouter cases in
  `test_team_scaling.py`; edit the `test_paper_defaults.py` docstring (§2).
- keep everything else; run `python -m pytest tests -q` after every prune commit.

### 3.7 root, `docs/`, `figures/`

- delete: `fetch_and_file.sh`, `fetch_pair_bonding_3f.sh`, `pack_for_pull.sh`,
  `pack_pair_bonding_3f.sh`, `pull_cluster_list.txt`, `pull_local_list.txt`, `pull_media.sh`,
  `pull_missing.ps1`, `pull_missing.sh`, `pull_new.sh`, `wt_pull_pair_bonding_3f.tgz` (54 MB,
  move outside the repo), `.github/`, `docs/experiment_checklist.md`, `tests/README.md` if it
  names the cluster.
- `figures/` and `paper_assets/`: **not in the release.** `paper_assets/` (498 MB) is already
  git-ignored, so it never enters the snapshot. `figures/` holds ten tracked and ten untracked
  images; no code reads any of them, and every paper figure is regenerated by its script (§3.1).
  Delete `figures/` except the two images `README.md` embeds
  (`overview_social_plasticity_loop.png`, `wire_five_chambers.png`), which are hand-made and
  cannot be regenerated; move those two to `docs/img/`. The 6.7 MB `wire_first_person_views.png`
  goes too.
- The one derived file a script needs but cannot rebuild from the `core` data layer is
  `paper_assets/perception_3f/beliefs_3f.csv` (Figure 8, built from the qualitative pipeline's
  outputs over the 21 GB `logs` layer). Ship it inside the OSF data bundle (§4), not in the repo.
- `docs/`: rewrite `experiments.md` (only the §3.3 arms, one cluster), `dataset.md` (§4 layout and
  sizes), `configuration.md` (drop removed flags), `README.md`; scrub the rest.
- `README.md`: reviewer-facing — what WIRE is, install (upstream Craftium + `craftium.patch`),
  run one arm locally, reproduce every table/figure (one command each), where the dataset is,
  license. No author, lab, cluster, or acknowledgement text.

### 3.8 `craftium.patch`

```bash
git -C craftium diff upstream/main HEAD > craftium.patch        # 2 files, 14 lines
```

Commit it at the repo root; README installs with
`git clone https://github.com/mikelma/craftium && git -C craftium checkout e8290cb && git -C craftium apply ../craftium.patch`.

---

## 4. Paper runs: folder + zip

The 49 arms below are everything v6 reads (§3.1). Sizes are raw; `core` packs ~5×.

| Question / group | Arms | seeds | core | logs | media |
|---|---|---|---|---|---|
| rq1 / `medium_runs` | exp01, exp02, exp03, exp04, exp09, exp10, exp11, exp34, exp35 | 6 each | 372 MB | 6.4 GB | 1.9 GB |
| rq1 / `new_exp_0_gemma` | `new_exp_0_gemma_base`, `new_exp_0_gemma_hebbian3f` | 6 | 114 MB | 0.7 GB | 0.3 GB |
| rq1 / `gemma4` | exp03_mappo, exp04_ippo | 6 (+3 junk dirs) | 97 MB | 1.7 GB | 0.4 GB |
| rq1 / `orchestrator` | `new_exp_0_gemma_orch_villager_advisory` | 6 | 49 MB | 0.5 GB | 0.2 GB |
| rq1 / `social_replay_3f_gemma4` | exp36, exp37 | 3 | 51 MB | 0.8 GB | 0.1 GB |
| rq1 / `social_replay_3f_qwen` | exp36, exp37 | 4 | 69 MB | 1.1 GB | 0.2 GB |
| rq2 / `cofiring_bidi_3f` | exp20, 21, 22, 23, 27, 28, 29 | 3 | 170 MB | 2.0 GB | 0 |
| rq3 / `medium_2k` | exp09, exp10, exp11 | 2 | 81 MB | 2.6 GB | 0.5 GB |
| rq3 / `pair_bonding` | `expA_pair_bonding`, `merged/` | 9 | 54 MB | 0.6 GB | 0.2 GB |
| rq3 / `pair_bonding_3f` | five `expB_*_3f` cells | 1 | 63 MB | 0.8 GB | 0 |
| compute / `pareto_social_3f` | si3f2, 8, 20, 50, 100, 200, 500 | 1–3 | 126 MB | 1.9 GB | 0.3 GB |
| compute / `pareto_gemma4` | `pareto_e2b_base`, `pareto_12b_base` | 6 | 86 MB | 0.9 GB | 0.4 GB |
| compute / `pareto_gemma4_3f` | `pareto_e2b_hebbian`, `pareto_12b_hebbian` | 2, 3 | 39 MB | 0.4 GB | 0 |
| compute / `agent_scaling_3f` | `scale_gemma_hebbian_n6` | 3 | 25 MB | 0.3 GB | 34 MB |
| compute / `agent_scaling_orch` | `scale_gemma_orch_villager_n6` | 1 | 16 MB | 0.2 GB | 65 MB |
| **total** | 49 arms | | **1.4 GB** | **21 GB** | **4.7 GB** |

Traps found while sizing:

- `gemma4/exp03_mappo` contains `seed_42.failed_copy`, `seed_123.failed_copy`, `seed_42.hang_0827`
  — `runs_dataset.py` and `make_results.load_runs` glob `seed_*`, so the bundle must accept
  `seed_<digits>` only.
- `checkpoints/` and `rl_live/` (the "other" column, ~700 MB in the RL arms) are never bundled.
- The three `compute/_*_3f_view` directories are symlink views; do not bundle them, fix the two
  scripts that read them (§3.2).

Steps:

- [ ] Add a `PAPER_ARMS` allowlist (the table above) and a `--paper` flag to
      `analysis/runs_dataset.py bundle`, plus the `seed_<digits>`-only filter; write the arm list and
      seed counts into `MANIFEST.json`.
- [ ] `python analysis/runs_dataset.py bundle --paper --layers core --out dist/paper_runs`
      → one `<question>__core.tar.gz` per question (~300 MB total) + `MANIFEST.json` + `SHA256SUMS`;
      then `--layers logs` for the ~2 GB logs tier (needed only for the qualitative pipeline and
      the RL step-clock alignment); `media` (4.7 GB) only if the Fig 9 frames must be reproducible
      from scratch — otherwise ship the extracted frames under `paper_assets/timelines/*/frames/`.
- [ ] Add the derived inputs a script needs but cannot recompute from `core`:
      `paper_assets/perception_3f/beliefs_3f.csv` and the `analysis/qualitative/out_*/tables/beliefs.csv`
      it was built from (Fig 8), and the Fig 5/9 frame PNGs.
- [ ] `zip -r wire_paper_runs_core.zip dist/paper_runs/*core*` (+ a separate `_logs.zip`); upload
      both to the OSF project; paste the anonymised view-only link into the reproducibility statement.
- [ ] Put `beliefs_3f.csv` (Figure 8) into the core archive under `derived/`.
- [ ] `docs/dataset.md`: replace the 46 GB layout with this table and the extract-and-run recipe.

---

## 5. Verification gate (before the mirror goes live)

- [ ] `git grep` scrub command (§2) prints nothing.
- [ ] `python -m pytest tests -q` green on the release branch.
- [ ] Fresh clone of the release branch + extracted `core` archive → run, in order:
      `make_beliefs_3f_view.py` (or ship its CSV), `make_agent_completion_tables.py`,
      `make_steps_table_pct.py`, `make_bond_behaviour_rho.py`, `make_transplant_tables.py`,
      `make_agent_completion_figs.py`, `make_final_figures.py gemma3f_seed123`,
      `make_chamber_gallery.py`, `make_counterfactual_story.py`, `make_counterfactual_n6.py`,
      `make_counterfactual_compact_n6.py`, `make_team_tenure.py` — every number must match v6.
- [ ] `python -m compileall -q src analysis` (catches imports of deleted modules).
- [ ] Open the anonymous mirror in a private window; check README renders, the site loads, and
      term replacement did not rewrite code.
- [ ] Reproducibility statement: replace `[TODO: anonymised code and website link.]` with the
      mirror URL, the site URL and the OSF link.
- [ ] Zip size of the OpenReview supplementary < 100 MB.
- [ ] Rebuild `dist/supplementary/` (`analysis/make_supplementary.py`) after the docs rewrite —
      it mirrors `docs/*.md` byte-for-byte, and the current bundle records commit `ebb9737`.

---

## 6. Order of work

1. §1 commit pending edits → 2. §3.2–3.7 prune (one commit per directory, tests after each) →
3. §2 scrub + `craftium.patch` → 4. §4 dataset bundle + `results/` → 5. README/docs rewrite →
6. §5 gate → 7. orphan snapshot, private repo, Anonymous GitHub, OSF → 8. links into the paper.
