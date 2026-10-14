#!/usr/bin/env bash
# Lists registry IDs only; does not start a browser or evaluate tasks.
set -euo pipefail
image=${1:?Usage: export_task_ids.sh browser-image /absolute/path/webarena-task-ids.json}
output=${2:?Specify an output JSON manifest}
[[ ! -e ${output} ]] || { echo "Output already exists: ${output}" >&2; exit 1; }
docker run --rm --platform linux/amd64 "${image}" python -c \
    'import json, browsergym.webarena, gymnasium as gym; print(json.dumps(sorted(x for x in gym.envs.registry if x.startswith("browsergym/webarena."))))' >"${output}"
printf 'Task registry manifest: %s\n' "${output}"
