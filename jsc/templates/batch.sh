#!/usr/bin/env bash
set -euo pipefail
set +x
set -a
source "${SLURM_SUBMIT_DIR}/launch.env"
set +a
export SLURM_EXPORT_ENV=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CLUSTER_NNODES=${SLURM_NNODES}
export NNODES_T=${CLUSTER_NNODES}
export NNODES=$((CLUSTER_NNODES - NNODES_ROLLOUT))
(( NNODES > 0 )) || { echo 'Allocation must exceed rollout node count.' >&2; exit 1; }
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export HEAD_NODE=${allocated_nodes[0]}
export RAY_PORT=$((10000 + SLURM_JOB_ID % 40000))
export RAY_TMPDIR=/tmp/ray-${SLURM_JOB_ID}
export LOGGER="['console']"
echo "[launch] JUPITER native AgentENV: ${NNODES} trainer + ${NNODES_ROLLOUT} rollout nodes"
srun --export=ALL --nodes="${CLUSTER_NNODES}" --ntasks="${CLUSTER_NNODES}" \
    --ntasks-per-node=1 --cpus-per-task="${CPUS_PER_NODE}" --gpus-per-task="${GPUS_PER_NODE}" \
    --kill-on-bad-exit=1 bash "${RUN_DIR}/jsc-node.sh"
