#!/usr/bin/env bash
# Harbor V2 training through Flash/libkrun, with no validation episodes.
set -euo pipefail
recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXP_NAME=${EXP_NAME:-qwen35_27b_swe_pro_v2_$(date +%Y%m%d_%H%M%S)}
export PROJECT_NAME=Uni-Agent-Qwen3.5-27B-SWE-bench-Pro-V2
export TRAIN_FILE=/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_pro_v2/swe_bench_pro_v2_train.parquet
# The older trainer constructs a validation loader even when validation is
# disabled. Reuse the training file; no validation episodes are executed.
export TEST_FILE=${TRAIN_FILE}
export TASK_CONFIG=${recipe_dir}/task_config_harbor_swe_bench_pro_v2_flash.yaml
export REQUIRE_ORACLE_PREFLIGHT=0
exec bash "${recipe_dir}/submit_qwen35_27b_pro_v1_clariden.sh"
