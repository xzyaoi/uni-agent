#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in config.env}"
: "${TRAINING_IMAGE:?Set TRAINING_IMAGE in config.env}"
: "${MODEL_PATH:?Set MODEL_PATH in config.env}"
for path in "${TRAINING_IMAGE}" "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${AGENTENV_CREDENTIALS}"; do
    [[ -r ${path} && -s ${path} ]] || { echo "Missing or unreadable input: ${path}" >&2; exit 1; }
done
[[ -x ${WORK_ROOT}/venv/bin/harbor ]] || { echo 'Run setup.sh first.' >&2; exit 1; }
if [[ ${BENCHMARK} = webarena ]]; then
    : "${WEBARENA_IMAGE:?Set WEBARENA_IMAGE to your WebArena-capable browser image}"
    [[ -r ${WEBARENA_ENV_FILE} ]] || { echo "Missing WebArena site environment: ${WEBARENA_ENV_FILE}" >&2; exit 1; }
fi
export EXP_NAME=${EXP_NAME:-qwen35_${MODEL_SIZE,,}_${BENCHMARK}_native_$(date +%Y%m%d_%H%M%S)}
[[ ${EXP_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXP_NAME' >&2; exit 1; }
[[ ${CONCURRENCY} =~ ^[0-9]+$ ]] || { echo 'CONCURRENCY must be non-negative; 0 disables the runner cap.' >&2; exit 1; }
for value in "${TRAINING_STEPS}" "${SAVE_FREQ}" "${CLUSTER_NNODES}" "${NNODES_ROLLOUT}" "${TP}" "${PP}" "${GEN_TP}"; do
    [[ ${value} =~ ^[1-9][0-9]*$ ]] || { echo 'Concurrency, steps, and save frequency must be positive integers.' >&2; exit 1; }
done
trainer_gpus=$(((CLUSTER_NNODES - NNODES_ROLLOUT) * 4))
if (( trainer_gpus <= 0 || trainer_gpus % (TP * PP) != 0 || (NNODES_ROLLOUT * 4) % GEN_TP != 0 )); then
    echo 'Invalid node split or tensor/pipeline parallelism for four GPUs per node.' >&2
    exit 1
fi
RUN_DIR=${WORK_ROOT}/runs/${EXP_NAME}
export RUN_DIR
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/prepare_run.py"
job_id=$(cd "${RUN_DIR}" && sbatch --parsable --account="${SLURM_ACCOUNT}" --partition="${PARTITION}" \
    --nodes="${CLUSTER_NNODES}" --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 \
    --exclusive --mem="${SLURM_MEM_PER_NODE:-850000M}" \
    --time="${TIME_LIMIT}" --no-requeue --job-name="${EXP_NAME}" \
    --chdir="${RUN_DIR}" --output="${RUN_DIR}/training.out" --export=ALL "${RUN_DIR}/launcher.sh")
printf '%s\n' "${job_id}" >"${RUN_DIR}/job_id"
printf 'Submitted %s\nRun: %s\n' "${job_id}" "${RUN_DIR}"
