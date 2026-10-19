#!/usr/bin/env bash
#SBATCH --job-name=qwen35-27b-vfy-t25-metrics
#SBATCH --partition=preemptable
#SBATCH --account=infra02
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --mem=850000M
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/training.out
set -euo pipefail
set +x
source "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/run.env"
set -a
source "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/credentials.env"
set +a
: "${OTELA_API_KEY:?Missing Flash scheduler credential}"
unset WANDB_API_KEY WANDB_ENTITY
export SLURM_EXPORT_ENV=ALL
export EDF_PATH="/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018:${EDF_PATH:-}"
export TMPDIR=/tmp
export NO_PROXY="172.28.38.18,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"
export RAY_PORT=$((10000 + SLURM_JOB_ID % 40000))
mkdir -p "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/checkpoints" "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/agent-logs" "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/harbor"
echo '[preflight] Checking frozen imports and scheduler connectivity on both allocated nodes'
srun --export=ALL,TMPDIR=/tmp --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    --environment=flash-training python "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/preflight.py"
exec bash "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_verified_flash_2n45m_c512_t25_metrics_20261018/launcher.sh"
