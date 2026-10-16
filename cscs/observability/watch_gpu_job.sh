#!/usr/bin/env bash
# Capture host DCGM/vLLM after the main container step owns its CPU allocation.
set -euo pipefail
job_id=${1:?Usage: watch_gpu_job.sh JOB_ID RUN_DIR}
run_dir=${2:?Specify the run directory}
[[ $job_id =~ ^[0-9]+$ ]] || { echo 'Invalid job ID' >&2; exit 1; }
while true; do
    state=$(squeue -h -j "$job_id" -o '%T' 2>/dev/null || true)
    if [[ -z $state ]]; then
        echo 'Job left queue before host monitoring started'
        exit 0
    fi
    if [[ $state = RUNNING ]] && [[ -f $run_dir/training.out ]] && \
       grep -q '\[head\] ray start --head' "$run_dir/training.out"; then
        break
    fi
    sleep 10
done
nodes=$(squeue -h -j "$job_id" -o '%D')
mkdir -p "$run_dir/telemetry/host-supplement"
echo "Starting host monitoring for job $job_id on $nodes nodes"
srun --jobid="$job_id" --overlap --kill-on-bad-exit=0 -N "$nodes" -n "$nodes" \
    --ntasks-per-node=1 -c 1 python3 "$run_dir/collect.py" --role gpu \
    --output "$run_dir/telemetry/host-supplement" --disk "$(dirname "$run_dir")" \
    --duration 21600
