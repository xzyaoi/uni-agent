#!/usr/bin/env bash
# One-node environment discovery before allocating the ten-node experiment.
set -euo pipefail
set +x
source "${SLURM_SUBMIT_DIR}/common.sh"
source "${AGENTENV_CREDENTIALS}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3
export APPTAINERENV_AGENTENV_API_KEY=${AGENTENV_API_KEY}
export APPTAINERENV_AGENTENV_ENDPOINT=${AGENTENV_ENDPOINT}
hostname
date -u
free -h
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Read the requested profiling counters from the site's existing hostengine.
dcgmi dmon -e 1001,1002,1003,1004,1005 -c 2 -d 1000
# The container executes only a native API readiness request, not model validation.
"${APPTAINER_BIN}" exec --nv --bind "${WORK_ROOT}:${WORK_ROOT}" "${TRAINING_IMAGE}" \
    "${WORK_ROOT}/venv/bin/python" - <<'PY'
import os
import urllib.request
request = urllib.request.Request(os.environ['AGENTENV_ENDPOINT'].rstrip('/') + '/v2/sandboxes',
                                 headers={'X-API-Key': os.environ['AGENTENV_API_KEY']})
with urllib.request.urlopen(request, timeout=20) as response:
    print('Native authenticated API from compute container:', response.status)
PY
