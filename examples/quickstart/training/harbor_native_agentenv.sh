#!/usr/bin/env bash
set -euo pipefail
# The SDK shell helper is a small, pinned snapshot of the experiment client.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/iopsstor/scratch/cscs/xyao/native-agentenv/src"
export AGENTENV_TRACE_DIR=${AGENTENV_TRACE_DIR:-/iopsstor/scratch/cscs/xyao/logs/Uni-Agent-Qwen3.5-27B-SWE-bench-Pro-V2-AgentENV/native-traces/${SLURM_JOB_ID:-unassigned}}
exec /iopsstor/scratch/cscs/xyao/venvs/uni-agentenv-v2/bin/harbor "$@"
