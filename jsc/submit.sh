#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
for path in "${TRAINING_IMAGE}" "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${AGENTENV_CREDENTIALS}"; do
    [[ -r ${path} && -s ${path} ]] || { echo "Missing or unreadable input: ${path}" >&2; exit 1; }
done
[[ -x ${WORK_ROOT}/venv/bin/harbor ]] || { echo 'Run jsc/setup.sh first.' >&2; exit 1; }
for value in "${CLUSTER_NNODES}" "${NNODES_ROLLOUT}" "${TP}" "${PP}" "${GEN_TP}" "${TRAINING_STEPS}" "${SAVE_FREQ}" "${ROLLOUT_PROMPT_BSZ}"; do
    [[ ${value} =~ ^[1-9][0-9]*$ ]] || { echo 'Topology, steps and batch settings must be positive integers.' >&2; exit 1; }
done
[[ ${CONCURRENCY} =~ ^[0-9]+$ ]] || { echo 'CONCURRENCY must be non-negative.' >&2; exit 1; }
trainer_gpus=$(((CLUSTER_NNODES - NNODES_ROLLOUT) * 4))
(( trainer_gpus > 0 && trainer_gpus % (TP * PP) == 0 && (NNODES_ROLLOUT * 4) % GEN_TP == 0 )) || { echo 'Invalid four-GPU node split/parallelism.' >&2; exit 1; }
export EXP_NAME=${EXP_NAME:-qwen35_27b_pro_v2_jsc_refill_$(date +%Y%m%d_%H%M%S)}
[[ ${EXP_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXP_NAME.' >&2; exit 1; }
export RUN_DIR=${WORK_ROOT}/runs/${EXP_NAME}
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/apply_rollout_batch.py" "${WORK_ROOT}/runtime"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/apply_continuous_rollout.py" "${WORK_ROOT}/runtime"
"${WORK_ROOT}/venv/bin/python" "${JSC_BUNDLE_ROOT}/prepare_run.py"
job_id=$(cd "${RUN_DIR}" && sbatch --parsable --account="${SLURM_ACCOUNT}" --partition="${PARTITION}" \
    --nodes="${CLUSTER_NNODES}" --ntasks-per-node=1 --cpus-per-task=288 --gpus-per-node=4 \
    --exclusive --mem=0 --time="${TIME_LIMIT}" --no-requeue --job-name="${EXP_NAME}" \
    --chdir="${RUN_DIR}" --output="${RUN_DIR}/training.out" --export=ALL "$@" "${RUN_DIR}/launcher.sh")
printf '%s\n' "${job_id}" > "${RUN_DIR}/job_id"
printf 'Submitted %s\nRun: %s\n' "${job_id}" "${RUN_DIR}"
