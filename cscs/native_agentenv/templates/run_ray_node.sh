#!/usr/bin/env bash
set -euo pipefail

unset ROCR_VISIBLE_DEVICES

PORT="${RAY_PORT:-6379}"
TMP="${RAY_TMPDIR:-/tmp/ray-${SLURM_JOB_ID:-$$}}"
CPUS="${CPUS_PER_NODE:-128}"
GPUS="${GPUS_PER_NODE:-4}"
HEAD="${HEAD_NODE:?HEAD_NODE must be exported by the launcher}"
ADDR="${HEAD}:${PORT}"
NNODES_T="${NNODES_T:-${NNODES:-1}}"

if [ "${SLURM_NODEID:-0}" = "0" ]; then
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

    REPO="${REPO_DIR:?Missing REPO_DIR}"
    TRAIN="${TRAIN_SCRIPT:?Missing TRAIN_SCRIPT}"
    JOB_OUT="/tmp/job-${EXP_NAME:-train}-$(date +%s).out"

    cd "${REPO}"
    echo "[head] cwd=$(pwd) verl=$(python -c 'import verl;print(verl.__version__)' 2>/dev/null || echo unknown)"
    echo "[head] model=${MODEL_PATH:-?} sandbox=native-agentenv exp=${EXP_NAME:-?}"

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
    echo "[worker nid=${SLURM_NODEID}] joining ${ADDR}"
    sleep 8  # let the head come up first
    ray start --address="${ADDR}" --temp-dir="${TMP}" \
        --num-cpus="${CPUS}" --num-gpus="${GPUS}" --block
fi
