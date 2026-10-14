#!/usr/bin/env bash
set -euo pipefail
export MODEL_SIZE=${1:?Usage: submit_preset.sh 4B|9B|27B swebench-verified|swebench-pro|webarena}
export BENCHMARK=${2:?Specify a benchmark}
exec bash "$(dirname "${BASH_SOURCE[0]}")/submit.sh"
