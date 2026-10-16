#!/usr/bin/env bash
# Native AgentENV resident baseline: five training steps without validation.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_DIR}"
: "${SCRATCH:?Run on a Clariden login node with SCRATCH set}"
export EXP_NAME=${EXP_NAME:-qwen35_27b_swe_pro_v2_agentenv_$(date +%Y%m%d_%H%M%S)}
export PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-27B-SWE-bench-Pro-V2-AgentENV}
export TRAINING_STEPS=${TRAINING_STEPS:-5}
export CKPTS_DIR=/iopsstor/scratch/cscs/xyao/ckpts/${PROJECT_NAME}/${EXP_NAME}
export MODEL_PATH=${MODEL_PATH:-/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-27B}
export TRAIN_FILE=${TRAIN_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_pro_v2_agentenv/swe_bench_pro_v2_agentenv_train.parquet}
export TEST_FILE=${TRAIN_FILE}
export TASK_CONFIG=${TASK_CONFIG:-${REPO_DIR}/examples/quickstart/training/task_config_harbor_swe_bench_pro_v2_agentenv_clariden.yaml}
export TRAINER_MODE=separate_async NNODES_ROLLOUT=${NNODES_ROLLOUT:-2} PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-4}
export GEN_TP=${GEN_TP:-4} TP=${TP:-4} PP=${PP:-2} CP=${CP:-1} CONCURRENCY=${CONCURRENCY:-64}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-64} N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_PROMPT_BSZ=${ROLLOUT_PROMPT_BSZ:-64}
export CONTINUOUS_ROLLOUT=${CONTINUOUS_ROLLOUT:-True}
export ROLLOUT_TARGET_PROMPTS=${ROLLOUT_TARGET_PROMPTS:-128}
export ROLLOUT_READY_PROMPTS=${ROLLOUT_READY_PROMPTS:-64}
export ROLLOUT_REFILL_PROMPTS=${ROLLOUT_REFILL_PROMPTS:-16}
export ROLLOUT_REFILL_INTERVAL=${ROLLOUT_REFILL_INTERVAL:-1.0}
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.45} MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768} SAVE_FREQ=${SAVE_FREQ:-5}
export TOTAL_NODES=${TOTAL_NODES:-10} TIME_LIMIT=${TIME_LIMIT:-04:00:00}
export AGENT_RUNNER_DISPATCH_MODE=${AGENT_RUNNER_DISPATCH_MODE:-ray_task}
export CPU_MEMORY_BIND=${CPU_MEMORY_BIND:-False}
export SANDBOX_START_ATTEMPT_TIMEOUT=600 SANDBOX_START_RETRIES=1
export AGENTENV_SECRET_DIR=${AGENTENV_SECRET_DIR:-/iopsstor/scratch/cscs/xyao/secrets/agentenv-native}
export AGENTENV_ENDPOINT=${AGENTENV_ENDPOINT:-http://clnode286.clemson.cloudlab.us:8000}
export AGENTENV_TRACE_DIR=${AGENTENV_TRACE_DIR:-/iopsstor/scratch/cscs/xyao/logs/${PROJECT_NAME}/native-traces/${EXP_NAME}}
[[ -s ${AGENTENV_SECRET_DIR}/client.env ]] || { echo 'Missing native AgentENV client.env' >&2; exit 1; }
[[ -x /iopsstor/scratch/cscs/xyao/native-agentenv/harbor-native ]] || { echo 'Missing native Harbor wrapper' >&2; exit 1; }

[[ ${EXP_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid EXP_NAME' >&2; exit 1; }
[[ ${TRAINING_STEPS} =~ ^[1-9][0-9]*$ ]] || { echo 'TRAINING_STEPS must be positive' >&2; exit 1; }
[[ -s ${MODEL_PATH}/config.json ]] || { echo 'Missing model config' >&2; exit 1; }
for path in "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG}"; do
    [[ -s ${path} ]] || { echo "Missing or empty input: ${path}" >&2; exit 1; }
done
command -v sbatch >/dev/null


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
export TASK_CONFIG=${run_dir}/task_config.yaml
export UNI_AGENT_TRACE_DIR=${run_dir}/telemetry/model-calls
export METRICS_DIR=${run_dir}/telemetry/gpu
export METRICS_COLLECTOR=${run_dir}/collect.py
export SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-850000M}
[[ -f ${REPO_DIR}/cscs/observability/collect.py ]] || { echo 'Missing continuous telemetry collector' >&2; exit 1; }
cp "${REPO_DIR}/cscs/observability/collect.py" "${METRICS_COLLECTOR}"
cp "${REPO_DIR}/cscs/native_agentenv/bind_cpu_memory.sh" "${run_dir}/bind_cpu_memory.sh"
python3 "${REPO_DIR}/cscs/observability/bundle_dcgm.py" --output "${run_dir}/dcgm-client"
mkdir -p "${UNI_AGENT_TRACE_DIR}" "${METRICS_DIR}"
cp "${REPO_DIR}/examples/quickstart/training/run_ray_node.sh" "${run_dir}/run_ray_node.base.sh"

python3 - "${run_dir}" <<'PY'
import re
import sys
import os
from pathlib import Path

run_dir = Path(sys.argv[1])
launcher = (run_dir / 'launcher.base.sh').read_text()
launcher = '\n'.join(line for line in launcher.split('\n') if not line.strip().startswith(': "${OTELA_API_KEY:?'))
launcher = launcher.replace('OTELA_API_KEY="${OTELA_API_KEY}"', 'OTELA_API_KEY="${OTELA_API_KEY:-}"')
train = (run_dir / 'train.base.sh').read_text()
node = (run_dir / 'run_ray_node.base.sh').read_text()
monitor = '''
if [[ -n ${METRICS_DIR:-} ]]; then
    mkdir -p "${METRICS_DIR}"
    python3 "${METRICS_COLLECTOR}" --role gpu --output "${METRICS_DIR}" --disk "${RUNTIME_DIR}" &
    _metrics_pid=$!
    _metrics_cleanup() {
        kill "${_metrics_pid}" 2>/dev/null || true
        wait "${_metrics_pid}" 2>/dev/null || true
        if [[ -d ${RAY_TMPDIR}/session_latest/logs ]]; then
            cp -a "${RAY_TMPDIR}/session_latest/logs" "${METRICS_DIR}/$(hostname)-ray-logs" || true
        fi
    }
    trap _metrics_cleanup EXIT
fi
'''
if node.count('set -euo pipefail\n') != 1:
    raise SystemExit('Expected one strict-mode line in node bootstrap')
node = node.replace('set -euo pipefail\n', 'set -euo pipefail\n' + monitor, 1)
bind_guard = '\nif [[ ${CPU_MEMORY_BIND:-False} = True && ${UNI_AGENT_CPU_MEMORY_BOUND:-0} != 1 ]]; then\n    exec bash "$(dirname "${BASH_SOURCE[0]}")/bind_cpu_memory.sh" bash "$0" "$@"\nfi\n'
node = node.replace('set -euo pipefail\n', 'set -euo pipefail\n' + bind_guard, 1)
proxy_bypass = '\n_agentenv_host=$(python3 -c \'import os; from urllib.parse import urlsplit; print(urlsplit(os.environ["AGENTENV_ENDPOINT"]).hostname)\')\nexport NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_agentenv_host}"\nexport no_proxy="${NO_PROXY}"\n'
node = node.replace('set -euo pipefail\n', 'set -euo pipefail\n' + proxy_bypass, 1)

status_marker = '    set +e; ray job status "${JOB_ID}"; JOB_RC=$?; set -e'
status_replacement = """    set +e
    ray job status "${JOB_ID}"
    _job_status=$(python -c 'import sys; from ray.job_submission import JobSubmissionClient; print(JobSubmissionClient("http://127.0.0.1:8265").get_job_status(sys.argv[1]).value)' "${JOB_ID}")
    JOB_RC=1
    [[ ${_job_status} = SUCCEEDED ]] && JOB_RC=0
    set -e"""
if node.count(status_marker) != 1:
    raise SystemExit('Expected one Ray job status check')
node = node.replace(status_marker, status_replacement, 1)

train = train.replace('dispatch_mode=ray_task', 'dispatch_mode=${AGENT_RUNNER_DISPATCH_MODE:-ray_task}')

node_command = '    bash examples/quickstart/training/run_ray_node.sh'
if launcher.count(node_command) != 1:
    raise SystemExit('Expected one node bootstrap command in launcher')
launcher = launcher.replace(node_command, '    bash "' + str(run_dir / 'run_ray_node.sh') + '"', 1)
task_config_path = run_dir / 'task_config.yaml'
task_config, count = re.subn(r'^(\s*)artifacts_dir:.*$', lambda m: m.group(1) + 'artifacts_dir: ' + str(run_dir / 'harbor'), task_config_path.read_text(), flags=re.M)
if count != 1:
    raise SystemExit('Expected one Harbor artifacts directory')
task_config_path.write_text(task_config)
secret_setup = 'set +x\n_agentenv_endpoint=${AGENTENV_ENDPOINT:-}\nset -a\nsource "${AGENTENV_SECRET_DIR}/client.env"\nset +a\nexport AGENTENV_ENDPOINT=${_agentenv_endpoint:-${AGENTENV_ENDPOINT:-}}\n: "${AGENTENV_API_KEY:?Missing native AgentENV API key}"\n: "${AGENTENV_ENDPOINT:?Missing native AgentENV endpoint}"\n'
if launcher.count('set -euo pipefail\n') != 1:
    raise SystemExit('Expected one launcher strict-mode line')
launcher = launcher.replace('set -euo pipefail\n', 'set -euo pipefail\n' + secret_setup, 1)

replacement = f'    TRAIN_SCRIPT="runs/{run_dir.name}/train.sh"'
launcher, count = re.subn(r'^\s+TRAIN_SCRIPT="examples/quickstart/training/train_qwen3p5_dense\.sh"$', replacement, launcher, flags=re.M)
if count != 1:
    raise SystemExit('Expected exactly one hardcoded TRAIN_SCRIPT in launcher')
if launcher.count('RUN_ENV=(\n') != 1:
    raise SystemExit('Expected exactly one RUN_ENV array in launcher')
launcher = launcher.replace('RUN_ENV=(\n', 'RUN_ENV=(\n    AGENTENV_ENDPOINT="${AGENTENV_ENDPOINT}"\n    AGENTENV_API_KEY="${AGENTENV_API_KEY}"\n    AGENTENV_TRACE_DIR="${AGENTENV_TRACE_DIR}"\n    TRAINING_STEPS="${TRAINING_STEPS:-5}"\n    PROJECT_NAME="${PROJECT_NAME}"\n    CKPTS_DIR="${CKPTS_DIR}"\n', 1)
launcher = launcher.replace('RUN_ENV=(\n', 'RUN_ENV=(\n    UNI_AGENT_TRACE_DIR="${UNI_AGENT_TRACE_DIR}"\n    METRICS_DIR="${METRICS_DIR}"\n    METRICS_COLLECTOR="${METRICS_COLLECTOR}"\n', 1)
batch_marker = 'trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}'
if train.count(batch_marker) != 1:
    raise SystemExit('Expected exactly one separate_async parameter sync override')
train = train.replace(batch_marker, batch_marker + ' ++trainer.v1.separate_async.rollout_prompt_batch_size=${ROLLOUT_PROMPT_BSZ:-64}', 1)
launcher = launcher.replace('RUN_ENV=(\n', 'RUN_ENV=(\n    ROLLOUT_PROMPT_BSZ="${ROLLOUT_PROMPT_BSZ:-64}"\n', 1)
refill_names = ['AGENT_RUNNER_DISPATCH_MODE', 'CPU_MEMORY_BIND', 'CONTINUOUS_ROLLOUT', 'ROLLOUT_TARGET_PROMPTS', 'ROLLOUT_READY_PROMPTS', 'ROLLOUT_REFILL_PROMPTS', 'ROLLOUT_REFILL_INTERVAL']
refill_env = ''.join('    ' + name + '="${' + name + '}"\n' for name in refill_names)
launcher = launcher.replace('RUN_ENV=(\n', 'RUN_ENV=(\n' + refill_env, 1)
refill_flags = ' ++trainer.v1.separate_async.continuous_refill.enabled=${CONTINUOUS_ROLLOUT} ++trainer.v1.separate_async.continuous_refill.target_inflight_prompts=${ROLLOUT_TARGET_PROMPTS} ++trainer.v1.separate_async.continuous_refill.max_ready_prompts=${ROLLOUT_READY_PROMPTS} ++trainer.v1.separate_async.continuous_refill.submit_prompts=${ROLLOUT_REFILL_PROMPTS} ++trainer.v1.separate_async.continuous_refill.poll_interval=${ROLLOUT_REFILL_INTERVAL}'
train = train.replace(batch_marker, batch_marker + refill_flags, 1)
if os.environ.get('CONTINUOUS_ROLLOUT', 'False').lower() == 'true':
    train = train.replace(batch_marker, batch_marker + ' data.dataloader_num_workers=0', 1)
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
stats_marker = '    actor_rollout_ref.rollout.enable_chunked_prefill=True'
if train.count(stats_marker) != 1:
    raise SystemExit('Expected exactly one vLLM chunked prefill override')
train = train.replace(stats_marker, '    actor_rollout_ref.rollout.disable_log_stats=False \\\n' + stats_marker, 1)
entropy_marker = '    actor_rollout_ref.actor.entropy_coeff=0 \\\n'
if train.count(entropy_marker) != 1:
    raise SystemExit('Expected exactly one entropy coefficient override')
train = train.replace(entropy_marker, entropy_marker + '    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=True \\\n    actor_rollout_ref.actor.megatron.entropy_from_logits_chunk_size=1024 \\\n', 1)
if train.count('    trainer.val_before_train=False') != 1:
    raise SystemExit('Expected validation before training to be disabled')

(run_dir / 'launcher.sh').write_text(launcher)
(run_dir / 'train.sh').write_text(train)
(run_dir / 'run_ray_node.sh').write_text(node)
PY

bash -n "${run_dir}/launcher.sh"
bash -n "${run_dir}/train.sh"
bash -n "${run_dir}/run_ray_node.sh"
# Record only public launch settings; credentials stay with the existing launcher.
for name in EXP_NAME PROJECT_NAME TRAINING_STEPS CKPTS_DIR MODEL_PATH TRAIN_FILE TEST_FILE TASK_CONFIG \
    TRAINER_MODE NNODES_ROLLOUT PARAMETER_SYNC_STEP GEN_TP TP PP CP CONCURRENCY \
    CONTINUOUS_ROLLOUT ROLLOUT_TARGET_PROMPTS ROLLOUT_READY_PROMPTS ROLLOUT_REFILL_PROMPTS ROLLOUT_REFILL_INTERVAL \
    TRAIN_PROMPT_BSZ ROLLOUT_PROMPT_BSZ N_RESP_PER_PROMPT PPO_MINI_BATCH_SIZE VLLM_GPU_MEM_UTIL \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH SAVE_FREQ AGENTENV_ENDPOINT \
    AGENTENV_SECRET_DIR AGENTENV_TRACE_DIR UNI_AGENT_TRACE_DIR METRICS_DIR METRICS_COLLECTOR SLURM_MEM_PER_NODE \
    SANDBOX_START_ATTEMPT_TIMEOUT SANDBOX_START_RETRIES TOTAL_NODES TIME_LIMIT AGENT_RUNNER_DISPATCH_MODE CPU_MEMORY_BIND; do
    printf '%s=%q\n' "${name}" "${!name}"
done >"${run_dir}/launch.env"

job_id=$(sbatch --parsable --partition="${PARTITION:-preemptable}" --nodes="${TOTAL_NODES}" --time="${TIME_LIMIT}" --exclusive --mem="${SLURM_MEM_PER_NODE}" \
    --job-name="${EXP_NAME}" --output="${run_dir}/training.out" --export=ALL "$@" "${run_dir}/launcher.sh")
printf '%s\n' "${job_id}" >"${run_dir}/job_id"
printf 'Submitted %s\nRun directory: %s\nOutput: %s/training.out\n' "${job_id}" "${run_dir}" "${run_dir}"
