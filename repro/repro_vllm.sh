#!/bin/bash
#SBATCH --job-name=repro-vllm
#SBATCH --partition=debug
#SBATCH --account=infra02
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --output=repro-vllm-%j.out

# Minimal standalone vLLM repro of the `sample_tokens` RPC timeout /
# EngineDeadError that killed training job 3209975 on clariden (GH200).
# Runs vllm.LLM (V1, multiproc executor, TP2, enable_sleep_mode +
# prefix_caching + cudagraphs) with a GDN model and loops
# generate -> sleep -> wake_up -> reset_prefix_cache, mirroring verl's
# free_cache_engine cycle between rollout batches.
#
# Why --qos=normal: debug's QoS caps at 1 job/user; normal QoS has no cap
# and AllowQos=normal,stop on the debug partition (proven in job 3208161).
#
# Tune the repro via env (see repro/vllm_mp_sleep_repro.py):
#   sbatch --export=ALL,CONCURRENCY=16,MAX_TOKENS=8192,DURATION_S=1200 \
#     repro/repro_vllm.sh

set -euo pipefail
[ "${1:-}" = "--" ] && shift
: "${SCRATCH:?SCRATCH is not set -- run on a clariden login node}"

export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
mkdir -p "${PIP_CACHE_DIR}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export EDF_PATH="${REPO_DIR}:${EDF_PATH:-}"
ENV_NAME="${ENV_NAME:-harbor-swe-megatron-bristen}"

echo "[repro] launching standalone vLLM repro in enroot env=${ENV_NAME}"
echo "[repro] config: TP=${TP:-2} CONCURRENCY=${CONCURRENCY:-8} "
echo "        MAX_TOKENS=${MAX_TOKENS:-16384} DURATION_S=${DURATION_S:-1500}"

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
    --gpus-per-task=2 \
    --environment="${ENV_NAME}" \
    python "${REPO_DIR}/repro/vllm_mp_sleep_repro.py"
RC=$?

echo "[repro] done (rc=${RC})"
exit ${RC}
