#!/usr/bin/env bash
# Five training steps without validation; run on a Clariden login node after the oracle preflight.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_DIR}"
: "${SCRATCH:?Run on a Clariden login node with SCRATCH set}"
export EXP_NAME=${EXP_NAME:-qwen35_27b_swe_pro_v1_$(date +%Y%m%d_%H%M%S)}
export PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-27B-SWE-bench-Pro}
export TRAINING_STEPS=${TRAINING_STEPS:-5}
export CKPTS_DIR=/iopsstor/scratch/cscs/xyao/ckpts/${PROJECT_NAME}/${EXP_NAME}
export MODEL_PATH=/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-27B
export TRAIN_FILE=${TRAIN_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_pro_v1_ghcr.parquet}
export TEST_FILE=${TEST_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_verified.parquet}
export TASK_CONFIG=${TASK_CONFIG:-${REPO_DIR}/examples/quickstart/training/task_config_react_swe_bench_pro_flash_cluster.yaml}
export TRAINER_MODE=separate_async NNODES_ROLLOUT=2 PARAMETER_SYNC_STEP=4
export GEN_TP=4 TP=4 PP=2 CP=1 CONCURRENCY=64
export TRAIN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=8 PPO_MINI_BATCH_SIZE=16
export VLLM_GPU_MEM_UTIL=0.45 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=32768 SAVE_FREQ=5
export FLASH_SANDBOX_ENDPOINT=https://api.opentela.ai/v1/service/flash-sandbox/
export FLASH_SANDBOX_TYPE=libkrun
export FLASH_SANDBOX_IMAGE_REGISTRY=ghcr.io/swe-images

[[ ${EXP_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXP_NAME' >&2; exit 1; }
[[ ${TRAINING_STEPS} =~ ^[1-9][0-9]*$ ]] || { echo 'TRAINING_STEPS must be positive' >&2; exit 1; }
[[ -s ${MODEL_PATH}/config.json ]] || { echo 'Missing model config' >&2; exit 1; }
for path in "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG}"; do
    [[ -s ${path} ]] || { echo "Missing or empty input: ${path}" >&2; exit 1; }
done
command -v sbatch >/dev/null

if [[ ${REQUIRE_ORACLE_PREFLIGHT:-1} == 1 ]]; then
python3 - "${SCRATCH}/tmp/swe-pro-preflight/oracle_result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
if not isinstance(result, dict) or result.get('reward') != 1:
    raise SystemExit(f'Oracle preflight must have reward=1: {path}')
PY
fi

base_launcher=${REPO_DIR}/run_train_qwen3p5_clariden.sh
base_train=${REPO_DIR}/examples/quickstart/training/train_qwen3p5_dense.sh
for path in "${base_launcher}" "${base_train}"; do
    grep -q 'separate_async' "${path}" || {
        echo "Remote separate_async support is required in ${path}" >&2
        exit 1
    }
done

run_dir=${REPO_DIR}/runs/${EXP_NAME}
mkdir -p "${REPO_DIR}/runs"
mkdir "${run_dir}" # Fail if a run with this name already exists.
cp "${base_launcher}" "${run_dir}/launcher.base.sh"
cp "${base_train}" "${run_dir}/train.base.sh"
cp "${TASK_CONFIG}" "${run_dir}/task_config.yaml"
if [[ ${REQUIRE_ORACLE_PREFLIGHT:-1} == 1 ]]; then
    cp "${SCRATCH}/tmp/swe-pro-preflight/oracle_result.json" "${run_dir}/oracle_result.json"
fi
export TASK_CONFIG=${run_dir}/task_config.yaml

python3 - "${run_dir}" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
launcher = (run_dir / 'launcher.base.sh').read_text()
train = (run_dir / 'train.base.sh').read_text()
replacement = f'    TRAIN_SCRIPT="runs/{run_dir.name}/train.sh"'
launcher, count = re.subn(r'^\s+TRAIN_SCRIPT="examples/quickstart/training/train_qwen3p5_dense\.sh"$', replacement, launcher, flags=re.M)
if count != 1:
    raise SystemExit('Expected exactly one hardcoded TRAIN_SCRIPT in launcher')
if launcher.count('RUN_ENV=(\n') != 1:
    raise SystemExit('Expected exactly one RUN_ENV array in launcher')
launcher = launcher.replace('RUN_ENV=(\n', 'RUN_ENV=(\n    TRAINING_STEPS="${TRAINING_STEPS:-5}"\n    PROJECT_NAME="${PROJECT_NAME}"\n    CKPTS_DIR="${CKPTS_DIR}"\n', 1)
if 'trainer.total_training_steps=' in train:
    raise SystemExit('Base script already sets total_training_steps; review before submission')
marker = '    trainer.total_epochs=${total_epochs} \\\n'
if train.count(marker) != 1:
    raise SystemExit('Expected exactly one trainer.total_epochs override')
train = train.replace(marker, marker + '    trainer.total_training_steps=${TRAINING_STEPS:-5} \\\n', 1)
validation_marker = '    trainer.test_freq="${test_freq}"'
if train.count(validation_marker) != 1:
    raise SystemExit('Expected exactly one trainer.test_freq override')
train = train.replace(validation_marker, '    trainer.test_freq=-1', 1)
(run_dir / 'launcher.sh').write_text(launcher)
(run_dir / 'train.sh').write_text(train)
PY

bash -n "${run_dir}/launcher.sh"
bash -n "${run_dir}/train.sh"
# Record only public launch settings; credentials stay with the existing launcher.
for name in EXP_NAME PROJECT_NAME TRAINING_STEPS CKPTS_DIR MODEL_PATH TRAIN_FILE TEST_FILE TASK_CONFIG \
    TRAINER_MODE NNODES_ROLLOUT PARAMETER_SYNC_STEP GEN_TP TP PP CP CONCURRENCY \
    TRAIN_PROMPT_BSZ N_RESP_PER_PROMPT PPO_MINI_BATCH_SIZE VLLM_GPU_MEM_UTIL \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH SAVE_FREQ FLASH_SANDBOX_ENDPOINT FLASH_SANDBOX_TYPE \
    FLASH_SANDBOX_IMAGE_REGISTRY; do
    printf '%s=%q\n' "${name}" "${!name}"
done >"${run_dir}/launch.env"

job_id=$(sbatch --parsable --partition=normal --nodes=10 --time=11:30:00 \
    --job-name="${EXP_NAME}" --output="${run_dir}/training.out" --export=ALL "${run_dir}/launcher.sh")
printf '%s\n' "${job_id}" >"${run_dir}/job_id"
printf 'Submitted %s\nRun directory: %s\nOutput: %s/training.out\n' "${job_id}" "${run_dir}" "${run_dir}"
