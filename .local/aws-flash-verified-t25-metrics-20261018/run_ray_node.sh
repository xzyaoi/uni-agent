#!/usr/bin/env bash
# Per-node Ray bootstrap + training driver. Invoked ONCE across all nodes by
#   srun --nodes=N --ntasks=N --ntasks-per-node=1 --environment=ENV bash run_ray_node.sh
# so the Ray daemons live in a SINGLE enroot step for the whole training run
# (same lifecycle model as rllm's `ray symmetric-run`, but explicit: we can't
# use symmetric-run directly because train_qwen3p5_dense.sh does
# `ray job submit --no-wait`, which returns immediately -- symmetric-run would
# tear the cluster down before the submitted job runs).
#
#   node 0 (head)  : ray start --head, wait for the cluster, run the training
#                    script (which submits the real Ray job --no-wait), then
#                    `ray job logs -f <id>` + `ray job status <id>` to block
#                    until it finishes, then `ray stop -f`.
#   node 1..N      : ray start --address <head> --block (kept alive until the
#                    head exits and SLURM tears the step down).
#
# Auto-detection: none of the scripts set RAY_ADDRESS, so `ray job submit` and
# `ray job logs` resolve `address=auto` against the local head on :RAY_PORT.
set -euo pipefail

# Bristen's Slurm injects BOTH CUDA_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES
# (=0,1,2,3) for GPU jobs, for CUDA/ROCm script portability. On these A100
# (NVIDIA) nodes ROCR_VISIBLE_DEVICES is meaningless, but verl 0.9.0.dev's
# WorkerDict._setup_env_cuda_visible_devices raises "Please don't set
# ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set" when both are
# present. Unset ROCR here (before any Ray/python import) on BOTH head and
# worker so the Ray daemon + every actor inherits only CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES

PORT="${RAY_PORT:-6379}"
TMP="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-$$}}"
CPUS="${CPUS_PER_NODE:-128}"
GPUS="${GPUS_PER_NODE:-4}"
HEAD="${HEAD_NODE:?HEAD_NODE must be exported by the launcher}"
ADDR="${HEAD}:${PORT}"
NNODES_T="${NNODES_T:-${NNODES:-1}}"


: "${RUN_DIR:?Missing run directory}"
: "${TELEMETRY_DIR:?Missing telemetry directory}"
: "${VLLM_PROMETHEUS_CONFIG:?Missing vLLM discovery path}"
mkdir -p "${TELEMETRY_DIR}/gpu" "${TELEMETRY_DIR}/vllm"
COLLECTOR_PIDS=()
stop_collectors() {
    local result=$?
    trap - EXIT
    for pid in "${COLLECTOR_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    for pid in "${COLLECTOR_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
    exit "${result}"
}
trap stop_collectors EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python "${RUN_DIR}/observability/collect_usage.py" \
    --mode gpu --output-dir "${TELEMETRY_DIR}/gpu" --interval-seconds 5 \
    >"${TELEMETRY_DIR}/gpu/collector-${SLURM_JOB_ID}-${SLURM_NODEID}.log" 2>&1 &
COLLECTOR_PIDS+=("$!")
if [ "${SLURM_NODEID:-0}" = "0" ]; then
    python "${RUN_DIR}/observability/collect_usage.py" \
        --mode prometheus --discovery-format prometheus-yaml \
        --discovery-file "${VLLM_PROMETHEUS_CONFIG}" \
        --output-dir "${TELEMETRY_DIR}/vllm" --interval-seconds 5 \
        >"${TELEMETRY_DIR}/vllm/collector-${SLURM_JOB_ID}.log" 2>&1 &
    COLLECTOR_PIDS+=("$!")
fi
sleep 1
for pid in "${COLLECTOR_PIDS[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "[metrics] Collector failed during startup; inspect ${TELEMETRY_DIR}" >&2
        exit 1
    fi
done
echo "[metrics] Persistent collectors started on $(hostname), interval=5s"

if [ "${SLURM_NODEID:-0}" = "0" ]; then
    # ---- Head ----
    # Runtime import check: libcuda.so.1 is a host-driver component absent
    # from the image BUILD, so the full Megatron import is deferred to here
    # (real GPU node, driver injected via nvidia-container-runtime). Fail fast
    # before starting Ray so we don't burn a multi-node alloc on a bad image.
    #
    # Bridge selection mirrors train_qwen3p5_dense.sh: use_mbridge=True with
    # vanilla_mbridge=False (default) -> the NeMo `megatron.bridge` package
    # (installed in the image). vanilla_mbridge=True would need the separate
    # `mbridge` wheel (github.com/ISEEKYAN/mbridge), NOT in the image -- set
    # VANILLA_MBRIDGE=True only if you install it.
    _bridge_imp="import megatron.bridge"
    if [ "${VANILLA_MBRIDGE:-False}" = "True" ]; then _bridge_imp="import mbridge"; fi
    echo "[head] verifying Megatron stack imports (torch/megatron/${_bridge_imp#import }/TE/apex)..."
    if ! python -c "import torch; import megatron.core; ${_bridge_imp}; import transformer_engine; import apex; print('OK megatron stack: torch', torch.__version__, 'TE', transformer_engine.__version__, 'cuda avail', torch.cuda.is_available())" 2>"${TMP}.import.err"; then
        echo "[head] ERROR: Megatron stack import failed -- aborting before Ray start:" >&2
        cat "${TMP}.import.err" >&2
        exit 1
    fi
    rm -f "${TMP}.import.err"
    echo "[head] ray start --head on :${PORT} (cpus=${CPUS} gpus=${GPUS} tmp=${TMP})"
    ray start --head --port="${PORT}" --temp-dir="${TMP}" \
        --num-cpus="${CPUS}" --num-gpus="${GPUS}" --include-dashboard=true

    echo "[head] waiting for ${NNODES_T} node(s) to join..."
    n=0
    for _ in $(seq 1 120); do
        n=$(ray status 2>/dev/null | grep -cE "node_") || n=0
        if [ "${n}" -ge "${NNODES_T}" ]; then echo "[head] cluster ready (${n} node(s))"; break; fi
        sleep 3
    done
    if [ "${n}" -lt "${NNODES_T}" ]; then
        echo "[head] WARN: only ${n}/${NNODES_T} nodes after 360s; continuing anyway"
        ray status || true
    fi

    REPO="${REPO_DIR:-/iopsstor/scratch/cscs/xyao/uni-agent}"
    TRAIN="${TRAIN_SCRIPT:-examples/quickstart/training/train_qwen3p5_dense.sh}"
    # JOB_OUT MUST be node-local (/tmp), NOT on ${SCRATCH} (Lustre). The
    # `bash "${TRAIN}" | tee "${JOB_OUT}"` line below runs under
    # `set -euo pipefail`: if tee EIOs writing to Lustre (as it did on job
    # 81982 at 11:38), tee keeps the pipe alive but exits NON-ZERO at EOF,
    # which under `set -e` kills run_ray_node.sh on the HEAD -> the head's
    # GCS server dies -> 60s later every worker exits (cluster teardown).
    # JOB_OUT is ONLY grepped locally on the head for the ray job id (line
    # below); it is never read off-node. The real training logs come via
    # `ray job logs -f` (Ray session dir on /tmp) + the SLURM-managed main
    # *.out on scratch (append-only, EIO-tolerant). /tmp is 369 GB tmpfs.
    JOB_OUT="/tmp/job-${EXP_NAME:-train}-$(date +%s).out"

    cd "${REPO}"
    echo "[head] cwd=$(pwd) verl=$(python -c 'import verl;print(verl.__version__)' 2>/dev/null || echo unknown)"
    echo "[head] model=${MODEL_PATH:-?} sandbox=flash_sandbox(OTELA_API_KEY=${OTELA_API_KEY:+set}) exp=${EXP_NAME:-?}"

    # train_qwen3p5_dense.sh: `ray job submit --no-wait` prints the job id and returns.
    bash "${TRAIN}" 2>&1 | tee "${JOB_OUT}"
    JOB_ID="$(grep -oE 'raysubmit_[A-Za-z0-9_-]+' "${JOB_OUT}" | head -1)"
    if [ -z "${JOB_ID}" ]; then
        echo "[head] ERROR: no Ray job id found in ${JOB_OUT}" >&2
        ray stop -f 2>/dev/null || true
        exit 1
    fi
    echo "[head] following logs for ${JOB_ID}..."
    ray job logs -f "${JOB_ID}" || true
    set +e; ray job status "${JOB_ID}"; JOB_RC=$?; set -e
    echo "[head] ${JOB_ID} finished with rc=${JOB_RC}"
    ray stop -f 2>/dev/null || true
    exit "${JOB_RC}"
else
    # ---- Worker ----
    echo "[worker nid=${SLURM_NODEID}] joining ${ADDR}"
    sleep 8  # let the head come up first
    ray start --address="${ADDR}" --temp-dir="${TMP}" \
        --num-cpus="${CPUS}" --num-gpus="${GPUS}" --block
fi
