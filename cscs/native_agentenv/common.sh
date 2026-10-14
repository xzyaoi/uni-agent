#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE=${CONFIG_FILE:-${BUNDLE_ROOT}/config.env}
[[ -f ${CONFIG_FILE} ]] || { echo "Copy ${BUNDLE_ROOT}/config.env.example to ${CONFIG_FILE} and edit it." >&2; exit 1; }
set -a
source "${CONFIG_FILE}"
: "${SCRATCH:?Set SCRATCH to your shared scratch directory}"
WORK_ROOT=${WORK_ROOT:-${SCRATCH}/swe-pro-native}
PARTITION=${PARTITION:-preemptable}
TIME_LIMIT=${TIME_LIMIT:-11:30:00}
CONCURRENCY=${CONCURRENCY:-64}
TRAINING_STEPS=${TRAINING_STEPS:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
AGENTENV_ENDPOINT=${AGENTENV_ENDPOINT:-http://clnode286.clemson.cloudlab.us:8000}
AGENTENV_CREDENTIALS=${AGENTENV_CREDENTIALS:-${WORK_ROOT}/secrets/agentenv.env}
set +a
export BUNDLE_ROOT WORK_ROOT
[[ ${WORK_ROOT} = /* ]] || { echo 'WORK_ROOT must be absolute.' >&2; exit 1; }
