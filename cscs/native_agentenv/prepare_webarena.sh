#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${WEBARENA_IMAGE:?Set WEBARENA_IMAGE to a WebArena-capable browser image}"
export PYTHONPATH="${WORK_ROOT}/runtime:${WORK_ROOT}/runtime/verl"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/prepare_webarena.py"
