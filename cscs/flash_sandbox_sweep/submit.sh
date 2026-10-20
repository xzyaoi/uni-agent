#!/usr/bin/env bash
# Run from the Clariden checkout. SWEEP_PREPARE_ONLY=True prepares without sbatch.
set -euo pipefail
set +x
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_DIR=$(cd "${source_dir}/../.." && pwd)
export SCRATCH=${SCRATCH:-/iopsstor/scratch/cscs/xyao}
export MODEL_PATH=${MODEL_PATH:-${SCRATCH}/data/models/Qwen3.5-9B}
export BENCHMARK=${BENCHMARK:-swebench-pro-v2}
export FLASH_SECRET_FILE=${FLASH_SECRET_FILE:-${SCRATCH}/secrets/flash-sandbox/client.env}
: "${FLASH_SDK_PATH:?Set FLASH_SDK_PATH to the isolated staged SDK and runtime dependencies}"
export FLASH_SDK_PATH
export FLASH_SANDBOX_TYPE=${FLASH_SANDBOX_TYPE:-firecracker}
export SWEEP_DIR=${SWEEP_DIR:-${REPO_DIR}/runs/flash_${BENCHMARK}_$(date -u +%Y%m%d_%H%M%SZ)}
export SWEEP_LEVELS=${SWEEP_LEVELS:-8 16 32 64}
# Use the dataset-capable environment for read-only cohort validation.
python=${SWEEP_PYTHON:-/iopsstor/scratch/cscs/xyao/venvs/uni-agentenv-v2/bin/python}
"${python}" "${source_dir}/prepare.py"
if [[ ${SWEEP_PREPARE_ONLY:-False} = True ]]; then
    printf 'Prepared sweep: %s\n' "${SWEEP_DIR}"
    exit 0
fi
python3 "${REPO_DIR}/cscs/sandbox_sweep/queue_points.py" "${SWEEP_DIR}" --once
nohup python3 "${REPO_DIR}/cscs/sandbox_sweep/queue_points.py" "${SWEEP_DIR}" > "${SWEEP_DIR}/queue-controller.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${SWEEP_DIR}/queue-controller.pid"
printf 'Sweep directory: %s\n' "${SWEEP_DIR}"
