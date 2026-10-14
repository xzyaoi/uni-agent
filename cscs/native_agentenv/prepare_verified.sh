#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export PYTHONPATH="${WORK_ROOT}/runtime:${WORK_ROOT}/runtime/verl"
export HF_HOME="${WORK_ROOT}/hf"
export HF_DATASETS_CACHE="/tmp/swe-verified-datasets-${UID}"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/prepare_verified.py"
