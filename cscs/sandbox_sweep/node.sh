#!/usr/bin/env bash
set -euo pipefail
if [[ ${UNI_AGENT_CPU_MEMORY_BOUND:-0} != 1 ]]; then
    exec bash "${RUN_DIR}/bind_cpu_memory.sh" bash "$0" "$@"
fi
unset ROCR_VISIBLE_DEVICES
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1 TMPDIR=/tmp
if [[ ${SWEEP_DIAGNOSTICS:-False} = True ]]; then
    export SWEEP_DIAGNOSTIC_DIR="${RUN_DIR}/diagnostics/$(hostname)"
    mkdir -p "${SWEEP_DIAGNOSTIC_DIR}"
    export PYTHONPATH="${RUN_DIR}/diagnostics_python:${PYTHONPATH}"
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
    export NCCL_DEBUG_FILE="${SWEEP_DIAGNOSTIC_DIR}/nccl.%h.%p.log"
    export TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_TRACE_CPP_STACK=0 TORCH_NCCL_DUMP_ON_TIMEOUT=1
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${SWEEP_DIAGNOSTIC_DIR}/torch-nccl-rank-"
    export TORCH_FR_DUMP_TEMP_FILE="${SWEEP_DIAGNOSTIC_DIR}/torch-nccl-rank-"
    export TORCH_NCCL_DEBUG_INFO_PIPE_FILE="${SWEEP_DIAGNOSTIC_DIR}/torch-nccl-pipe-"
fi
export TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor HF_HOME=/tmp/hf
export XDG_CACHE_HOME=/tmp/xdg-cache MEGATRON_CONFIG_LOCK_DIR=/tmp/megatron-config-locks
sweep_agentenv_host=${AGENTENV_ENDPOINT#*://}
sweep_agentenv_host=${sweep_agentenv_host%%/*}
sweep_agentenv_host=${sweep_agentenv_host%%:*}
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${sweep_agentenv_host},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
if [[ ${AGENTENV_EXPECTED_NODE_COUNT:-0} -gt 0 ]]; then
    python3 "${RUN_DIR}/gateway_preflight.py" --expected-nodes "${AGENTENV_EXPECTED_NODE_COUNT}"
fi
export UNI_AGENT_TRACE_DIR="${RUN_DIR}/telemetry/model-calls"
export AGENTENV_TRACE_DIR="${RUN_DIR}/native-traces"
export RAY_ADDRESS=auto
export RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID}"
export DCGMI_PATH="${RUN_DIR}/dcgm-client/bin/dcgmi"
mkdir -p "${UNI_AGENT_TRACE_DIR}" "${AGENTENV_TRACE_DIR}" "${RUN_DIR}/telemetry/gpu"
python3 "${RUN_DIR}/collect.py" --role gpu --output "${RUN_DIR}/telemetry/gpu" --disk "${SCRATCH}" --duration 3000 &
collector_pid=$!
diagnostic_pid=
if [[ ${SWEEP_DIAGNOSTICS:-False} = True ]]; then
    python3 "${RUN_DIR}/diagnostic_watch.py" > "${SWEEP_DIAGNOSTIC_DIR}/watchdog.log" 2>&1 &
    diagnostic_pid=$!
fi
cleanup() {
    set +e
    ray stop --force > "${RUN_DIR}/$(hostname)-ray-stop.log" 2>&1
    if [[ ${SLURM_NODEID:-0} = 0 ]]; then
        python3 "${RUN_DIR}/cleanup_sandboxes.py" "${RUN_DIR}" > "${RUN_DIR}/cleanup.log" 2>&1
    fi
    if [[ -d ${RAY_TMPDIR}/session_latest/logs ]]; then
        cp -a "${RAY_TMPDIR}/session_latest/logs" "${RUN_DIR}/telemetry/gpu/$(hostname)-ray-logs"
    fi
    if [[ -n ${diagnostic_pid} ]]; then
        kill "${diagnostic_pid}" 2>/dev/null
        wait "${diagnostic_pid}" 2>/dev/null
    fi
    kill "${collector_pid}" 2>/dev/null
    wait "${collector_pid}" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 143' TERM INT
# One Ray cluster spans both nodes, with two TP4 rollout replicas.
if [[ ${SLURM_NODEID:-0} != 0 ]]; then
    sleep 8
    ray start --address="${HEAD_NODE}:6379" --temp-dir="${RAY_TMPDIR}" --num-cpus=288 --num-gpus=4 \
        --object-store-memory=8589934592 --memory=343597383680 --block > "${RUN_DIR}/$(hostname)-ray-worker.log" 2>&1 &
    ray_cli_pid=$!
    while [[ ! -f ${RUN_DIR}/driver-exit-code ]]; do
        if ! kill -0 "${ray_cli_pid}" 2>/dev/null; then
            echo 'Worker Ray bootstrap exited before driver completion' >&2
            exit 1
        fi
        sleep 2
    done
    exit "$(cat "${RUN_DIR}/driver-exit-code")"
fi
ray start --head --port=6379 --temp-dir="${RAY_TMPDIR}" --num-cpus=288 --num-gpus=4 \
    --object-store-memory=8589934592 --memory=343597383680 --include-dashboard=true
ready=0
for _ in $(seq 1 120); do
    count=$(ray status 2>/dev/null | grep -cE 'node_') || count=0
    if [[ ${count} -ge 2 ]]; then ready=1; break; fi
    sleep 3
done
[[ ${ready} = 1 ]] || { echo 'Both Ray nodes did not join' >&2; exit 1; }
cd "${REPO_DIR}"
set +e
python3 "${RUN_DIR}/run_point.py" --data-path "${DATA_PATH}" --model-path "${MODEL_PATH}" \
    --task-config "${RUN_DIR}/task_config.yaml" --run-dir "${RUN_DIR}" \
    --concurrency "${GROUP_CONCURRENCY}" --groups 128 --responses 8 --seed 42 --nnodes 2 --benchmark "${BENCHMARK:-swebench-pro-v2}"
driver_rc=$?
set -e
printf '%s\n' "${driver_rc}" > "${RUN_DIR}/driver-exit-code.tmp"
mv "${RUN_DIR}/driver-exit-code.tmp" "${RUN_DIR}/driver-exit-code"
exit "${driver_rc}"
