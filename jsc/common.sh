#!/usr/bin/env bash
set -euo pipefail
JSC_BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CONFIG_FILE=${CONFIG_FILE:-${JSC_BUNDLE_ROOT}/config.env}
# Reuse the portable runtime/data configuration and defaults.
source "${JSC_BUNDLE_ROOT}/../cscs/native_agentenv/common.sh"
export JSC_BUNDLE_ROOT APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
