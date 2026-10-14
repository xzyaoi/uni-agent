#!/usr/bin/env bash
set -euo pipefail
run=${1:?Usage: ./status.sh /absolute/path/to/run}
job_id=$(cat "${run}/job_id")
squeue -j "${job_id}" -o '%.18i %.14P %.12T %.10M %.6D %R'
sacct -j "${job_id}" --format=JobID,State,Elapsed,ExitCode -n
if [[ -f ${run}/training.out ]]; then tail -n 40 "${run}/training.out"; fi
if [[ -d ${run}/native-traces ]]; then
    printf 'Trace files: '
    find "${run}/native-traces" -type f -name '*.jsonl' | wc -l
fi
