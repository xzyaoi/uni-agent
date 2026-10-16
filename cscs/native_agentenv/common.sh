#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-${BUNDLE_ROOT}/config.env}
[[ -f ${CONFIG_FILE} ]] || { echo "Copy ${BUNDLE_ROOT}/config.env.example to ${CONFIG_FILE} and edit it." >&2; exit 1; }
set -a
source "${CONFIG_FILE}"
: "${SCRATCH:?Set SCRATCH to your shared scratch directory}"
WORK_ROOT=${WORK_ROOT:-${SCRATCH}/swe-pro-native}
PARTITION=${PARTITION:-preemptable}
TIME_LIMIT=${TIME_LIMIT:-11:30:00}
CONCURRENCY=${CONCURRENCY:-64}
TRAINING_STEPS=${TRAINING_STEPS:-5}
ROLLOUT_PROMPT_BSZ=${ROLLOUT_PROMPT_BSZ:-64}
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
CONTINUOUS_ROLLOUT=${CONTINUOUS_ROLLOUT:-False}
ROLLOUT_TARGET_PROMPTS=${ROLLOUT_TARGET_PROMPTS:-128}
ROLLOUT_READY_PROMPTS=${ROLLOUT_READY_PROMPTS:-64}
ROLLOUT_REFILL_PROMPTS=${ROLLOUT_REFILL_PROMPTS:-16}
ROLLOUT_REFILL_INTERVAL=${ROLLOUT_REFILL_INTERVAL:-1.0}
SAVE_FREQ=${SAVE_FREQ:-5}
AGENTENV_ENDPOINT=${AGENTENV_ENDPOINT:-http://clnode286.clemson.cloudlab.us:8000}
AGENTENV_CREDENTIALS=${AGENTENV_CREDENTIALS:-${WORK_ROOT}/secrets/agentenv.env}
MODEL_SIZE=${MODEL_SIZE:-27B}
BENCHMARK=${BENCHMARK:-swebench-pro}
case ${MODEL_SIZE} in
    4B) MODEL_PATH=${MODEL_PATH_4B:-${SCRATCH}/data/models/Qwen3.5-4B} ;;
    9B) MODEL_PATH=${MODEL_PATH_9B:-${SCRATCH}/data/models/Qwen3.5-9B} ;;
    27B) MODEL_PATH=${MODEL_PATH_27B:-${MODEL_PATH:-${SCRATCH}/data/models/Qwen3.5-27B}} ;;
    *) echo 'MODEL_SIZE must be 4B, 9B, or 27B.' >&2; exit 1 ;;
esac
case ${BENCHMARK} in
    swebench-pro) TRAIN_FILE=${TRAIN_FILE_PRO:-${WORK_ROOT}/data/train.parquet} ;;
    swebench-verified) TRAIN_FILE=${TRAIN_FILE_VERIFIED:-${WORK_ROOT}/data/swebench-verified.parquet} ;;
    webarena) TRAIN_FILE=${TRAIN_FILE_WEBARENA:-${WORK_ROOT}/data/webarena.parquet} ;;
    *) echo 'BENCHMARK must be swebench-pro, swebench-verified, or webarena.' >&2; exit 1 ;;
esac
WEBARENA_ENV_FILE=${WEBARENA_ENV_FILE:-${WORK_ROOT}/secrets/webarena.env}
WEBARENA_IMAGE=${WEBARENA_IMAGE:-}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-2}
CLUSTER_NNODES=${CLUSTER_NNODES:-10}
GEN_TP=${GEN_TP:-4}
TP=${TP:-4}
PP=${PP:-2}
CP=1
if [[ ${BENCHMARK} = webarena ]]; then
    # One shared mutable website deployment: serialize episodes by default.
    CONCURRENCY=${WEBARENA_CONCURRENCY:-1}
fi
set +a
export BUNDLE_ROOT WORK_ROOT
[[ ${WORK_ROOT} = /* ]] || { echo 'WORK_ROOT must be absolute.' >&2; exit 1; }
