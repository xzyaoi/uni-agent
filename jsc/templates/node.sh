#!/usr/bin/env bash
set -euo pipefail
set +x
set -a
source "${RUN_DIR}/launch.env"
source "${AGENTENV_CREDENTIALS}"
set +a
: "${AGENTENV_API_KEY:?Missing API key in private credential file}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TMPDIR=/tmp
export PYTHONPATH=${REPO_DIR}:${REPO_DIR}/verl:${NATIVE_SHELL_ROOT}
export PYTHONNOUSERSITE=1
export HF_HOME=${WORK_ROOT}/hf
export RLLM_HOME=${WORK_ROOT}/rllm_home
export PIP_CACHE_DIR=${WORK_ROOT}/pip-cache
export WANDB_DIR=${WORK_ROOT}/wandb
export MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron-config-locks
export XDG_CACHE_HOME=/tmp/xdg-cache
export TOKENIZERS_PARALLELISM=true
export VLLM_LOGGING_LEVEL=WARN
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${WORK_ROOT}/home" "${HF_HOME}" "${RLLM_HOME}" "${PIP_CACHE_DIR}" "${WANDB_DIR}" "${AGENTENV_TRACE_DIR}"
# Set image environment overrides explicitly; inherit job-specific Ray/topology values.
while IFS= read -r line; do
    [[ ${line} = 'export '* ]] || continue
    key=${line#export }
    key=${key%%=*}
    export "APPTAINERENV_${key}=${!key}"
done < "${RUN_DIR}/launch.env"
for key in CUDA_VISIBLE_DEVICES AGENTENV_API_KEY PYTHONPATH PYTHONNOUSERSITE HF_HOME RLLM_HOME PIP_CACHE_DIR WANDB_DIR MEGATRON_CONFIG_LOCK_DIR XDG_CACHE_HOME TOKENIZERS_PARALLELISM VLLM_LOGGING_LEVEL NCCL_DEBUG CUDA_DEVICE_MAX_CONNECTIONS CUBLAS_WORKSPACE_CONFIG TMPDIR NNODES NNODES_T HEAD_NODE RAY_PORT RAY_TMPDIR LOGGER; do
    export "APPTAINERENV_${key}=${!key}"
done
# HOME is an Apptainer environment override; do not modify the host account HOME.
export APPTAINERENV_HOME=${WORK_ROOT}/home
if [[ -n ${WANDB_API_KEY:-} ]]; then
    export APPTAINERENV_WANDB_API_KEY=${WANDB_API_KEY}
    export APPTAINERENV_WANDB_ENTITY=${WANDB_ENTITY:-}
    export APPTAINERENV_LOGGER="['console','wandb']"
fi
# Keep host metrics even if the container's driver/DCGM access fails.
"${HARBOR_VENV}/bin/python" "${METRICS_COLLECTOR}" --role gpu \
    --output "${RUN_DIR}/telemetry/host" --disk "${WORK_ROOT}" &
host_metrics_pid=$!
cleanup() {
    kill "${host_metrics_pid}" 2>/dev/null || true
    wait "${host_metrics_pid}" 2>/dev/null || true
}
trap cleanup EXIT
# Shared paths preserve absolute venv interpreter and native SDK references.
binds=("${WORK_ROOT}" "${MODEL_PATH%/*}" "${TRAINING_IMAGE%/*}" "${NATIVE_SHELL_ROOT}")
bind_args=()
for path in "${binds[@]}"; do
    [[ ${path} != *:* && ${path} != *,* ]] || { echo "Unsupported bind path: ${path}" >&2; exit 1; }
    bind_args+=(--bind "${path}:${path}")
done
"${APPTAINER_BIN}" exec --nv "${bind_args[@]}" --pwd "${REPO_DIR}" \
    "${TRAINING_IMAGE}" bash "${RUN_DIR}/run_ray_node.sh"
