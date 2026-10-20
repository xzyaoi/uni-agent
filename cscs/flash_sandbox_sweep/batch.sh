#!/usr/bin/env bash
#SBATCH --account=infra02
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@60
#SBATCH --exclusive
#SBATCH --mem=850000M
set -euo pipefail
set +x
set -a
source "${SLURM_SUBMIT_DIR}/launch.env"
# Only the API key is secret; an explicit endpoint overrides the private file.
sweep_flash_endpoint=${FLASH_SANDBOX_ENDPOINT:-}
source "${FLASH_SECRET_FILE}"
set +a
export FLASH_SANDBOX_ENDPOINT=${sweep_flash_endpoint:-${FLASH_SANDBOX_ENDPOINT:-}}
: "${FLASH_SANDBOX_ENDPOINT:?Private client.env must set FLASH_SANDBOX_ENDPOINT}"
: "${OTELA_API_KEY:?Private client.env must set OTELA_API_KEY}"
export EDF_PATH="${REPO_DIR}" TMPDIR=/tmp
export PYTHONPATH="${FLASH_SDK_PATH}:${REPO_DIR}:${REPO_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"
# Never apply AgentENV's Verified finalizer or alter shared dataset metadata.
if [[ -n ${PREVIOUS_RUN_DIR:-} ]]; then
    "${HARBOR_VENV}/bin/python" "${RUN_DIR}/cleanup_sandboxes.py" "${PREVIOUS_RUN_DIR}" \
        --report-file "${RUN_DIR}/predecessor-sandbox-cleanup.json"
fi
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export HEAD_NODE=${nodes[0]}
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
    --environment=harbor-swe-megatron-clariden bash "${RUN_DIR}/node.sh"
