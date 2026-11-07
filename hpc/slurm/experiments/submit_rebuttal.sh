#!/bin/bash
# ───────────────────────────────────────────────────────────────────────────
# submit_rebuttal.sh — the ICLR rebuttal ablations (Gemma 4 E4B, 3 ep × 1000).
#
# Three families, told apart by the arm's prefix:
#
#   rl_<algo>_<variant>   RL fine-tuning, N=3, IN-PROCESS: vLLM cannot train
#                         LoRA. algo = ippo | mappo. Flags match exp03/exp04
#                         (base) and exp36/exp37 (full). RL_VLLM=1 serves the
#                         agents' cognition from vLLM and keeps the RL actor
#                         and PPO in-process (RL_WITH_VLLM in _common.sh);
#                         those runs are named reb_<arm>_vllm.
#     base     no graph
#     full     diffusion + sharing + bond text in the prompt
#     diff     diffusion + bond text      (--hebbian-rho 0)
#     share    sharing + bond text        (--hebbian-gamma 0)
#     text     bond text only             (both 0)
#     uniform  full, W frozen flat at the measured mean bond
#     freq     full, no reward growth     (--hebbian-eta-plus 0)
#
#   t1_<variant>          zero-shot, N=3, Table 1 regime on vLLM: mandatory
#                         messaging, message pay on, no budget.
#     base     no graph
#     full     the Methods rule with the social module
#     nofloor  no association floor       (--hebbian-eta-0 0)
#     nopay    no message pay             (--comm-reward-scale 0)
#     contrib  no floor and no message pay: bonds grow from milestones only
#     freq     no reward growth           (--hebbian-eta-plus 0)
#
#   zs_<variant>          zero-shot, scaling regime on vLLM: NS × BUDGET, flags
#                         of budget_gemma.sbatch (message pay already off, so
#                         its reward growth is milestones only). The base and
#                         full cells come from submit_comm_budget.sh.
#     uniform   W frozen flat at the measured mean bond
#     freq      no reward growth          (--hebbian-eta-plus 0)
#     nofloor   no association floor      (--hebbian-eta-0 0)
#     msgpay    message pay back on       (--comm-reward-scale 1)
#     nospatial no proximity term         (--hebbian-radius 0)
#
# Usage (login node, env.sh sourced):
#   cd $WT_WORKSPACE/WiredTogether/hpc/slurm/experiments
#   ARMS="rl_ippo_full" SMOKE=1 bash submit_rebuttal.sh
#   ARMS="rl_ippo_full" SMOKE=1 RL_VLLM=1 bash submit_rebuttal.sh
#   ARMS="rl_ippo_base rl_ippo_full" SEEDS="456 1011 1213" bash submit_rebuttal.sh
#   ARMS="zs_uniform zs_freq" NS="3 7 15" BUDGET=600 bash submit_rebuttal.sh
#   DRY_RUN=1 ...   prints the jobs without submitting
#
# Wall time per job is the measured worst case with margin; TIME overrides it.
# Snellius refuses a job at start once the running jobs' LIMITS × 128 SBU/h
# exceed the balance, so keep limits tight.
#
# Idempotent (final_metrics.json skip) + in-queue dedup: rerunning the SAME
# command only fills in what is neither finished nor queued.
# ───────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"
mkdir -p slurm_logs

WORKSPACE="${WT_WORKSPACE:?set WT_WORKSPACE to your cluster workspace (it holds WiredTogether/, images/ and models/)}"
REPO="$WORKSPACE/WiredTogether"

: "${MODEL_LLM:=$WORKSPACE/models/gemma-4-E4B-it}"
: "${WT_IMAGE:=$WORKSPACE/images/wiredtogether_gemma4.sif}"
: "${LLM_VISION_MODE:=vision}"
: "${RUN_GROUP:=rebuttal}"
: "${EPISODES:=3}"
: "${MAX_STEPS:=1000}"
: "${WANDB_PROJECT:=rebuttal}"
: "${COMM_MSG_CAP:=32}"
: "${CH4_MOBS:=3}"

ARM_LIST=(${ARMS:?set ARMS, e.g. ARMS=\"rl_ippo_base rl_ippo_full\"})
SEED_LIST=(${SEEDS:-42 123 456 789 1011 1213})
NS_LIST=(${NS:-3})
BUDGET="${BUDGET:-600}"
RL_VLLM="${RL_VLLM:-0}"

if [ "${SMOKE:-0}" = "1" ]; then
    SEED_LIST=(42)
    RUN_GROUP="${RUN_GROUP}_smoke"
    EPISODES=1
    export WANDB=0
fi
export MODEL_LLM WT_IMAGE LLM_VISION_MODE RUN_GROUP EPISODES WANDB_PROJECT

HEB2="--hebbian --hebbian-mode three_factor --hebbian-eta-0 0.001 --hebbian-reward-norm 50 --hebbian-decay 0.001 --hebbian-eligibility-rho 0.9 --hebbian-coact-floor 0.25 --hebbian-death-ltd 0.05 --hebbian-death-cap 10"

# Mean off-diagonal W measured on the finished runs, so a frozen flat graph
# reads like the learned one in the prompt: exp37 0.38, exp36 0.33, budget
# b600 cells 0.13 at N=3 and 0.10 at N=7 and N=15.
uniform_rl() {
    case "$1" in
        ippo) echo "--hebbian-freeze --hebbian-preset uniform --hebbian-bond-strong 0.38" ;;
        *)    echo "--hebbian-freeze --hebbian-preset uniform --hebbian-bond-strong 0.33" ;;
    esac
}
uniform_zs() {
    if [ "$1" -le 3 ]; then
        echo "--hebbian-freeze --hebbian-preset uniform --hebbian-bond-strong 0.13"
    else
        echo "--hebbian-freeze --hebbian-preset uniform --hebbian-bond-strong 0.10"
    fi
}

# Sets R_EXP R_ARGS R_SERVER R_N R_TIME R_MEM R_STAGGER R_STEPS R_RL_VLLM
# R_VLLM_UTIL for one cell.
resolve_arm() {
    local arm="$1" n="$2" variant algo critic rule pay
    R_N="$n"
    R_STAGGER=0
    R_RL_VLLM=""
    R_VLLM_UTIL="${VLLM_GPU_UTIL:-}"
    case "$arm" in
        rl_*)
            algo="${arm#rl_}"; variant="${algo#*_}"; algo="${algo%%_*}"
            case "$algo" in
                ippo)  critic=independent ;;
                mappo) critic=centralized ;;
                *) echo "ERROR: unknown RL algorithm in '$arm'" >&2; return 1 ;;
            esac
            local rl="--rl --rl-critic-mode $critic --rl-model-path $MODEL_LLM --rl-update-interval 64 --rl-lr 3e-4 --simultaneous"
            case "$variant" in
                base)    R_ARGS="$rl" ;;
                full)    R_ARGS="$rl $HEB2 --hebbian-gamma 0.2 --hebbian-rho 0.3" ;;
                diff)    R_ARGS="$rl $HEB2 --hebbian-gamma 0.2 --hebbian-rho 0" ;;
                share)   R_ARGS="$rl $HEB2 --hebbian-gamma 0 --hebbian-rho 0.3" ;;
                text)    R_ARGS="$rl $HEB2 --hebbian-gamma 0 --hebbian-rho 0" ;;
                uniform) R_ARGS="$rl $HEB2 --hebbian-gamma 0.2 --hebbian-rho 0.3 $(uniform_rl "$algo")" ;;
                freq)    R_ARGS="$rl $HEB2 --hebbian-gamma 0.2 --hebbian-rho 0.3 --hebbian-eta-plus 0" ;;
                *) echo "ERROR: unknown RL variant in '$arm'" >&2; return 1 ;;
            esac
            R_N=3
            R_EXP="reb_$arm"
            R_SERVER=""
            R_STAGGER=1          # env-idle hang fix, every Gemma RL arm
            R_TIME=96:00:00      # worst seed on the other cluster: 74 h
            R_MEM=64GB
            R_STEPS=150
            if [ "$RL_VLLM" = "1" ]; then
                # The RL copy of Gemma (~15 GiB + scoring and PPO memory)
                # shares the 40 GB A100 with the server: 0.46 leaves the
                # server ~1.9 GiB of KV cache (~50k tokens, prompts <= 5.5k).
                R_EXP="reb_${arm}_vllm"
                R_SERVER=vllm
                R_RL_VLLM=1
                R_VLLM_UTIL="${VLLM_GPU_UTIL:-0.46}"
            fi
            ;;
        t1_*)
            variant="${arm#t1_}"
            rule="$HEB2 --hebbian-gamma 0.2 --hebbian-rho 0.0 --social-module prompt"
            case "$variant" in
                base)    R_ARGS="--simultaneous" ;;
                full)    R_ARGS="$rule --simultaneous" ;;
                nofloor) R_ARGS="$rule --simultaneous --hebbian-eta-0 0" ;;
                nopay)   R_ARGS="$rule --simultaneous --comm-reward-scale 0" ;;
                contrib) R_ARGS="$rule --simultaneous --hebbian-eta-0 0 --comm-reward-scale 0" ;;
                freq)    R_ARGS="$rule --simultaneous --hebbian-eta-plus 0" ;;
                *) echo "ERROR: unknown t1 variant in '$arm'" >&2; return 1 ;;
            esac
            R_N=3
            R_EXP="reb_$arm"
            R_SERVER=vllm
            R_STEPS=20
            ;;
        zs_*)
            variant="${arm#zs_}"
            rule="$HEB2 --hebbian-gamma 0.2 --hebbian-rho 0.0 --social-module prompt"
            local env_args="--team-scaling --ch4-mob-count $CH4_MOBS --simultaneous --comm-budget-tokens $BUDGET --comm-budget-msg-cap $COMM_MSG_CAP"
            case "$variant" in
                uniform)   R_ARGS="$rule $env_args --comm-reward-scale 0 $(uniform_zs "$n")" ;;
                freq)      R_ARGS="$rule $env_args --comm-reward-scale 0 --hebbian-eta-plus 0" ;;
                nofloor)   R_ARGS="$rule $env_args --comm-reward-scale 0 --hebbian-eta-0 0" ;;
                msgpay)    R_ARGS="$rule $env_args --comm-reward-scale 1" ;;
                nospatial) R_ARGS="$rule $env_args --comm-reward-scale 0 --hebbian-radius 0" ;;
                *) echo "ERROR: unknown zs variant in '$arm'" >&2; return 1 ;;
            esac
            R_EXP="reb_${arm}_n${n}_b${BUDGET}"
            R_SERVER=vllm
            R_STEPS=20
            ;;
        *)
            echo "ERROR: arm '$arm' must start with rl_, t1_ or zs_ (see the header)" >&2
            return 1
            ;;
    esac
    if [ -n "$R_SERVER" ] && [ -z "$R_RL_VLLM" ]; then
        # Measured worst case on vLLM + margin: N=7 7.0 h, N=10 9.3 h, N=15 14.1 h.
        if   [ "$R_N" -le 3 ]; then R_TIME=6:00:00;  R_MEM=48GB
        elif [ "$R_N" -le 6 ]; then R_TIME=10:00:00; R_MEM=64GB
        elif [ "$R_N" -le 7 ]; then R_TIME=10:00:00; R_MEM=96GB
        elif [ "$R_N" -le 10 ]; then R_TIME=14:00:00; R_MEM=96GB
        else                        R_TIME=20:00:00; R_MEM=96GB
        fi
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        # An RL smoke runs two PPO update rounds (~15 min per agent update).
        case "$arm" in
            rl_*) R_TIME=6:00:00 ;;
            *)    R_TIME=1:30:00 ;;
        esac
    fi
    R_TIME="${TIME:-$R_TIME}"
}

echo "== submit_rebuttal.sh =="
echo "  run_group : $RUN_GROUP"
echo "  arms      : ${ARM_LIST[*]}"
echo "  seeds     : ${SEED_LIST[*]}"
echo "  N (zs_*)  : ${NS_LIST[*]}   budget: $BUDGET"
[ "${SMOKE:-0}" = "1" ]   && echo "  mode      : SMOKE"
[ "${DRY_RUN:-0}" = "1" ] && echo "  mode      : DRY_RUN (no submission)"
echo "========================"

missing=0
[ -f "$WT_IMAGE" ] || { echo "ERROR: image not found: $WT_IMAGE" >&2; missing=1; }
[ -f "$MODEL_LLM/config.json" ] || { echo "ERROR: weights not staged: $MODEL_LLM/config.json" >&2; missing=1; }
[ "$missing" = "1" ] && exit 1

queued_names=$(squeue -u "${USER:-$(whoami)}" -h -o "%j" 2>/dev/null || true)

n_queued=0
n_skipped=0
n_failed=0
for arm in "${ARM_LIST[@]}"; do
    case "$arm" in
        zs_*) cell_ns=("${NS_LIST[@]}") ;;
        *)    cell_ns=(3) ;;
    esac
    for n in "${cell_ns[@]}"; do
        resolve_arm "$arm" "$n" || exit 1
        if [ -n "$R_SERVER" ] && [ ! -e "${VLLM_IMAGE:-$WORKSPACE/images/vllm}" ] && [ ! -e "$WORKSPACE/images/vllm.sif" ]; then
            echo "ERROR: $arm needs vLLM but there is no image; run hpc/snellius/setup.sh vllm" >&2
            exit 1
        fi
        steps="$MAX_STEPS"
        [ "${SMOKE:-0}" = "1" ] && steps="${SMOKE_STEPS:-$R_STEPS}"
        for seed in "${SEED_LIST[@]}"; do
            jobname="${RUN_GROUP}-${R_EXP}_s${seed}"
            if [ -f "$REPO/runs/$RUN_GROUP/$R_EXP/seed_$seed/final_metrics.json" ]; then
                n_skipped=$((n_skipped + 1))
                continue
            fi
            if printf '%s\n' "$queued_names" | grep -Fqx "$jobname"; then
                echo "in queue     $R_EXP  seed_$seed"
                n_skipped=$((n_skipped + 1))
                continue
            fi
            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "would queue  $R_EXP  seed_$seed  (server=${R_SERVER:-in-process}${R_RL_VLLM:+ rl+vllm util=$R_VLLM_UTIL} N=$R_N mem=$R_MEM time=$R_TIME steps=$steps)"
                echo "             $R_ARGS"
                n_queued=$((n_queued + 1))
                continue
            fi
            jobid=$(REB_ARM="$R_EXP" REB_ARGS="$R_ARGS" SEED="$seed" NUM_AGENTS="$R_N" \
                MAX_STEPS="$steps" LLM_SERVER="$R_SERVER" RL_UPDATE_STAGGER="$R_STAGGER" \
                RL_WITH_VLLM="$R_RL_VLLM" VLLM_GPU_UTIL="$R_VLLM_UTIL" \
                sbatch --parsable --job-name="$jobname" --mem="$R_MEM" --time="$R_TIME" \
                rebuttal_arm.sbatch)
            if [ -z "$jobid" ]; then
                echo "FAILED  $R_EXP  seed_$seed — not submitted; re-run this command later" >&2
                n_failed=$((n_failed + 1))
                continue
            fi
            echo "queued  $R_EXP  seed_$seed  →  job $jobid  (mem=$R_MEM time=$R_TIME)"
            n_queued=$((n_queued + 1))
        done
    done
done
echo "── done: $n_queued submitted, $n_skipped skipped, $n_failed FAILED ──"
[ "$n_failed" -gt 0 ] && exit 1
exit 0
