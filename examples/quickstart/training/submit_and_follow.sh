#!/usr/bin/env bash
# Runs INSIDE the enroot container on the Ray head node. Submits
# train_qwen3p5_dense.sh (which does `ray job submit --no-wait`) and then
# blocks on that Ray job's logs + status until it finishes.
#
# Kept separate from the SLURM launcher to avoid quoting a Ray-submit pipeline
# through `srun ... bash -c`. All tunables come from the environment
# (RUN_ENV in run_train_qwen3p5_bristen.sh).
set -euo pipefail

cd "${REPO_DIR:-/iopsstor/scratch/cscs/xyao/uni-agent}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-examples/quickstart/training/train_qwen3p5_dense.sh}"
JOB_OUT="${SCRATCH:-/iopsstor/scratch/cscs/xyao}/uni-agent/job-${EXP_NAME:-train}-$(date +%s).out"

echo "[submit] cwd=$(pwd) verl=$(python -c 'import verl;print(verl.__version__)' 2>/dev/null || echo unknown)"
echo "[submit] model=${MODEL_PATH:-?} sandbox=flash_sandbox(OTELA_API_KEY=${OTELA_API_KEY:+set}) exp=${EXP_NAME:-?}"

# `ray job submit --no-wait` returns immediately after the job is accepted,
# printing "Submitted Job ID: raysubmit_xxx" to stdout.
bash "${TRAIN_SCRIPT}" 2>&1 | tee "${JOB_OUT}"

JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9_-]+' "${JOB_OUT}" | head -1)"
if [ -z "${JOB_ID}" ]; then
    echo "[submit] ERROR: no Ray job id found in ${JOB_OUT}" >&2
    exit 1
fi
echo "[submit] following logs for ${JOB_ID}..."
ray job logs -f "${JOB_ID}" || true
# ray job status returns non-zero if the job failed; capture and propagate.
set +e
ray job status "${JOB_ID}"
JOB_RC=$?
set -e
echo "[submit] ${JOB_ID} finished with rc=${JOB_RC}"
exit ${JOB_RC}
