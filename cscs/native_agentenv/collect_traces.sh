#!/usr/bin/env bash
set -euo pipefail
run=${1:?Usage: ./collect_traces.sh /absolute/path/to/run /absolute/path/to/output.tar.gz}
output=${2:?Specify an output archive outside the run directory}
[[ ${output} = /* ]] || { echo 'Output path must be absolute.' >&2; exit 1; }
[[ ! -e ${output} ]] || { echo "Output already exists: ${output}" >&2; exit 1; }
entries=()
for name in native-traces harbor agent-logs training.out launch.env task_config.yaml runtime_env.yaml job_id; do
    if [[ -e ${run}/${name} ]]; then entries+=("${name}"); fi
done
[[ ${#entries[@]} -gt 0 ]] || { echo 'No run artifacts found.' >&2; exit 1; }
tar -czf "${output}" -C "${run}" "${entries[@]}"
printf 'Collected: %s\n' "${output}"
