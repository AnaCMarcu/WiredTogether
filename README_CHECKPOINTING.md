# Checkpoint / Resume System

Long HPC runs on DelftBlue are split across multiple 8-hour SLURM jobs.
This document explains how the checkpoint system works and how to use it.

---

## Overview

At the end of every episode and every `--checkpoint-interval` steps within an episode,
the run saves its full state to a directory. A new SLURM job can restore that state
and continue without re-running completed episodes.

What is saved:

| File | Contents |
|------|----------|
| `run_state.json` | Episode/step counter, metric history, CLI args |
| `hebbian_graph.json` | Hebbian weight matrix, config |
| `rl_agent_{i}/` | LoRA adapter, action/value head weights, optimizer, RMS, counters |
| `agent_{i}_curriculum.json` | Current task, task lists (completed/failed) |
| `frames_{i}.npy` | Raw observation arrays *(optional, `--checkpoint-frames`)* |

The environment (Craftium/Luanti) is **not** checkpointed — it always does a fresh
`reset()` on resume, which is correct because world state is procedurally generated.

---

## Quick Start

### Single long run (manual chaining)

```bash
# Job 1 — fresh start
sbatch scripts/run_first.sh

# Job 2 — continue after job 1 finishes
sbatch scripts/run_continue.sh
```

### Automatic chaining

```bash
# Submit 1 first job + 3 continuations (4 × 8h = up to 32h)
bash scripts/chain_jobs.sh 3
```

This uses `--dependency=afterany` so each continuation job starts automatically
as soon as the previous one finishes (whether successfully or due to a timeout).

---

## Checkpoint Directory Layout

```
checkpoints/
└── hebbian_rl_v1/                  ← CKPT_ROOT (shared across all jobs)
    ├── latest_checkpoint.txt       ← path to most recent ep*_end dir
    ├── ep1_end/                    ← end-of-episode checkpoint
    │   ├── run_state.json
    │   ├── hebbian_graph.json
    │   ├── rl_agent_0/
    │   │   ├── adapter_model.safetensors
    │   │   ├── action_head.pt
    │   │   ├── value_head.pt
    │   │   └── rl_state.pt
    │   ├── rl_agent_1/ …
    │   ├── agent_0_curriculum.json
    │   └── agent_1_curriculum.json …
    ├── ep1_step200/                ← mid-episode checkpoint (every 200 steps)
    ├── ep1_step400/ …
    └── ep2_end/ …
```

`latest_checkpoint.txt` always contains the path of the most recently completed
episode checkpoint. `run_continue.sh` reads this file to know where to resume.

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-dir PATH` | `./checkpoints/<run_id>` | Where to write checkpoints |
| `--checkpoint-interval N` | `500` | Save every N steps within an episode |
| `--resume PATH` | — | Path to a checkpoint directory to resume from |
| `--resume-skip-warmup` | off | Skip the media-load warmup on resume (safe when cache is warm) |
| `--checkpoint-frames` | off | Also save raw observation frames (large; only needed for GIF continuity) |

---

## Graceful Shutdown

The run hooks `SIGTERM` and `SIGINT`. When SLURM sends `SIGTERM` at the end of the
time limit, the current step completes and a `*_shutdown` checkpoint is written before
the process exits. The next job picks that checkpoint up via `run_continue.sh`.

To trigger a manual graceful shutdown:
```bash
scancel --signal=SIGTERM <job_id>
```

---

## Resuming Manually

```bash
# Inspect the latest checkpoint
cat /scratch/acmarcu/WiredTogether/checkpoints/hebbian_rl_v1/latest_checkpoint.txt

# Resume from a specific checkpoint
cd src/mindforge
python multi_agent_craftium.py \
    --num-agents 3 --episodes 5 --rl --rl-model-path ... \
    --hebbian --targeted-communication \
    --experiment-id hebbian_rl_v1 \
    --checkpoint-dir /scratch/acmarcu/WiredTogether/checkpoints/hebbian_rl_v1 \
    --resume /scratch/acmarcu/WiredTogether/checkpoints/hebbian_rl_v1/ep2_end \
    --resume-skip-warmup
```

---

## Notes

- The `--episodes` flag in `run_continue.sh` must be **≥** the total episodes you
  want to run, not just the remaining ones. The resume logic skips already-completed
  episodes internally.
- Mid-episode step checkpoints (`ep{N}_step{M}`) resume from the **start** of
  episode N, not from step M. Step-level resume would require replaying the
  environment, which is not supported.
- Checkpoint directories are never deleted automatically. Clean up old runs from
  `/scratch/acmarcu/WiredTogether/checkpoints/` periodically to save disk space.
