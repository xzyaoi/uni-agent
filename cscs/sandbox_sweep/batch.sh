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
sweep_agentenv_endpoint=${AGENTENV_ENDPOINT}
source "${AGENTENV_SECRET_DIR}/client.env"
set +a
# Explicit run endpoint takes precedence over the credential file's endpoint.
export AGENTENV_ENDPOINT=${sweep_agentenv_endpoint}
export EDF_PATH="${REPO_DIR}"
export TMPDIR=/tmp
if [[ -n ${PREVIOUS_RUN_DIR:-} ]]; then
    python3 "${RUN_DIR}/cleanup_sandboxes.py" "${PREVIOUS_RUN_DIR}"
fi
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export HEAD_NODE=${nodes[0]}
set +e
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL --environment=harbor-swe-megatron-clariden bash "${RUN_DIR}/node.sh"
run_rc=$?
set -e
if [[ ${SWEEP_DIAGNOSTICS:-False} = True && -d ${RUN_DIR}/diagnostics ]]; then
    tar --exclude='*.pipe' -czf "${RUN_DIR}/diagnostics.tar.gz" -C "${RUN_DIR}" diagnostics
fi
exit "${run_rc}"
