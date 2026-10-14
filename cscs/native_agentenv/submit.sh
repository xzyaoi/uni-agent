#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in config.env}"
: "${TRAINING_IMAGE:?Set TRAINING_IMAGE in config.env}"
: "${MODEL_PATH:?Set MODEL_PATH in config.env}"
for path in "${TRAINING_IMAGE}" "${MODEL_PATH}/config.json" "${WORK_ROOT}/data/train.parquet" "${AGENTENV_CREDENTIALS}"; do
    [[ -r ${path} && -s ${path} ]] || { echo "Missing or unreadable input: ${path}" >&2; exit 1; }
done
[[ -x ${WORK_ROOT}/venv/bin/harbor ]] || { echo 'Run setup.sh first.' >&2; exit 1; }
export EXP_NAME=${EXP_NAME:-qwen35_27b_pro_v2_native_$(date +%Y%m%d_%H%M%S)}
[[ ${EXP_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXP_NAME' >&2; exit 1; }
for value in "${CONCURRENCY}" "${TRAINING_STEPS}" "${SAVE_FREQ}"; do
    [[ ${value} =~ ^[1-9][0-9]*$ ]] || { echo 'Concurrency, steps, and save frequency must be positive integers.' >&2; exit 1; }
done
RUN_DIR=${WORK_ROOT}/runs/${EXP_NAME}
export RUN_DIR
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/prepare_run.py"
job_id=$(cd "${RUN_DIR}" && sbatch --parsable --account="${SLURM_ACCOUNT}" --partition="${PARTITION}" \
    --nodes=10 --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 \
    --time="${TIME_LIMIT}" --no-requeue --job-name="${EXP_NAME}" \
    --chdir="${RUN_DIR}" --output="${RUN_DIR}/training.out" --export=ALL "${RUN_DIR}/launcher.sh")
printf '%s\n' "${job_id}" >"${RUN_DIR}/job_id"
printf 'Submitted %s\nRun: %s\n' "${job_id}" "${RUN_DIR}"
