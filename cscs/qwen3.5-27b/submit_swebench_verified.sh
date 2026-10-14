#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../native_agentenv" && pwd)
exec bash "${recipe_dir}/submit_preset.sh" 27B swebench-verified
