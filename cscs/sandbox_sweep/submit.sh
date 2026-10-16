#!/usr/bin/env bash
# Run on Clariden from the repository checkout.
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_DIR=$(cd "${source_dir}/../.." && pwd)
export SCRATCH=${SCRATCH:-/iopsstor/scratch/cscs/xyao}
export MODEL_PATH=${MODEL_PATH:-${SCRATCH}/data/models/Qwen3.5-9B}
export BENCHMARK=${BENCHMARK:-swebench-pro-v2}
export TASK_CONFIG_SOURCE=${TASK_CONFIG_SOURCE:-${REPO_DIR}/examples/quickstart/training/task_config_harbor_swe_bench_pro_v2_agentenv_clariden.yaml}
export AGENTENV_UV_ARCHIVE=${AGENTENV_UV_ARCHIVE:-}
export AGENTENV_UV_ARCHIVE_SHA256=${AGENTENV_UV_ARCHIVE_SHA256:-}
export AGENTENV_FIXED_CPU_CORES=${AGENTENV_FIXED_CPU_CORES:-4}
export SWEEP_DIAGNOSTICS=False
if [[ ${BENCHMARK} = swebench-verified ]]; then
    : "${VERIFIED_DATASET_ROOT:?Set VERIFIED_DATASET_ROOT}"
    : "${VERIFIED_WARMUP_REPORT:?Warm the selected images first}"
    /iopsstor/scratch/cscs/xyao/venvs/uni-agentenv-v2/bin/python "${source_dir}/finalize_verified.py" --dataset-root "${VERIFIED_DATASET_ROOT}" --warmup-report "${VERIFIED_WARMUP_REPORT}"
    export TASK_CONFIG_SOURCE="${source_dir}/task_config_verified.yaml"
    export DATA_PATH=${DATA_PATH:-${VERIFIED_DATASET_ROOT}/swebench_verified_harbor.parquet}
    mapfile -t native_asset < <(/iopsstor/scratch/cscs/xyao/venvs/uni-agentenv-v2/bin/python -c 'import json,sys; m=json.load(open(sys.argv[1]))["native_bootstrap"]; print(m["asset_path"]); print(m["sha256"])' "${VERIFIED_DATASET_ROOT}/manifest.json")
    export AGENTENV_UV_ARCHIVE=${AGENTENV_UV_ARCHIVE:-${native_asset[0]}}
    export AGENTENV_UV_ARCHIVE_SHA256=${AGENTENV_UV_ARCHIVE_SHA256:-${native_asset[1]}}
    export AGENTENV_FIXED_CPU_CORES=1
fi
export DATA_PATH=${DATA_PATH:-${SCRATCH}/data/uni_agent/swe_bench_pro_v2_agentenv/swe_bench_pro_v2_agentenv_train.parquet}
export AGENTENV_SECRET_DIR=${AGENTENV_SECRET_DIR:-${SCRATCH}/secrets/agentenv-native}
export AGENTENV_ENDPOINT=http://ec2-3-229-113-27.compute-1.amazonaws.com:8000
sweep_dir=${SWEEP_DIR:-${REPO_DIR}/runs/sandbox_sweep_$(date -u +%Y%m%d_%H%M%SZ)}
mkdir "${sweep_dir}"
limits=(8 16 32 64)
if [[ ${SWEEP_INCLUDE_128:-False} = True ]]; then
    limits+=(128)
fi
python3 - "${sweep_dir}" "${limits[@]}" <<'PY'
import json,sys
from pathlib import Path
(Path(sys.argv[1])/'sweep-plan.json').write_text(json.dumps({'levels':[int(v) for v in sys.argv[2:]], 'conditional_128':'requires clean 64-group completion'},indent=2)+'\n')
PY
previous_job=
previous_run=
for limit in "${limits[@]}"; do
    run_dir="${sweep_dir}/groups-${limit}"
    mkdir -p "${run_dir}/telemetry/model-calls" "${run_dir}/native-traces" "${run_dir}/analysis-tools"
    cp "${source_dir}/run_point.py" "${source_dir}/node.sh" "${source_dir}/cleanup_sandboxes.py" "${run_dir}/"
    cp "${REPO_DIR}/cscs/native_agentenv/bind_cpu_memory.sh" "${run_dir}/"
    cp "${REPO_DIR}/cscs/observability/collect.py" "${run_dir}/"
    cp "${REPO_DIR}/cscs/observability/export_metrics.py" "${run_dir}/analysis-tools/"
    cp "${TASK_CONFIG_SOURCE}" "${run_dir}/task_config.yaml"
    python3 "${REPO_DIR}/cscs/observability/bundle_dcgm.py" --output "${run_dir}/dcgm-client"
    python3 - "${run_dir}" <<'PY'
from pathlib import Path
import re,sys
p=Path(sys.argv[1])/'task_config.yaml'
s,n=re.subn(r'^(\s*)artifacts_dir:.*$',lambda m:m.group(1)+'artifacts_dir: '+str(p.parent/'harbor'),p.read_text(),flags=re.M)
assert n==1
p.write_text(s)
PY
    {
        for name in REPO_DIR SCRATCH MODEL_PATH DATA_PATH AGENTENV_SECRET_DIR AGENTENV_ENDPOINT BENCHMARK AGENTENV_UV_ARCHIVE AGENTENV_UV_ARCHIVE_SHA256 AGENTENV_FIXED_CPU_CORES SWEEP_DIAGNOSTICS; do
            printf '%s=%q\n' "${name}" "${!name}"
        done
        printf 'RUN_DIR=%q\nGROUP_CONCURRENCY=%q\nPREVIOUS_RUN_DIR=%q\n' "${run_dir}" "${limit}" "${previous_run}"
        printf 'AGENTENV_TRACE_DIR=%q\nAGENT_LOG_DIR=%q\nNNODES_ROLLOUT=2\nTRAINING_STEPS=0\nTIME_LIMIT=00:45:00\n' "${run_dir}/native-traces" "${run_dir}/sessions"
    } > "${run_dir}/launch.env"
    cp "${source_dir}/batch.sh" "${run_dir}/launcher.sh"
    previous_run=${run_dir}
done
python3 "${source_dir}/queue_points.py" "${sweep_dir}" --once
nohup python3 "${source_dir}/queue_points.py" "${sweep_dir}" > "${sweep_dir}/queue-controller.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${sweep_dir}/queue-controller.pid"
printf 'Sweep directory: %s\n' "${sweep_dir}"
