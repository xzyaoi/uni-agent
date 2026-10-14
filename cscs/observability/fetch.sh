#!/usr/bin/env bash
# Download an analysis snapshot; run again after training for the final files.
set -euo pipefail
cluster=${CLUSTER_SSH:-clariden}
sandbox=${SANDBOX_SSH:-xzyao@clnode286.clemson.cloudlab.us}
run=${1:?Usage: fetch.sh /remote/run/directory /remote/cloudlab/telemetry/directory /local/output}
cloudlab=${2:?Specify the remote CloudLab telemetry directory}
output=${3:?Specify a new or existing local output directory}
mkdir -p "${output}/run" "${output}/cloudlab" "${output}/native-traces"
rsync -a --exclude=native-traces --exclude=secrets --exclude=client.env \
    "${cluster}:${run}/" "${output}/run/"
native=$(python3 - "${output}/run/launch.env" <<'PY'
import shlex
import sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith('AGENTENV_TRACE_DIR='):
        print(shlex.split(line.partition('=')[2])[0])
        break
else:
    raise SystemExit('Missing native trace path in launch.env')
PY
)
rsync -a "${cluster}:${native}/" "${output}/native-traces/"
rsync -a --exclude=collector.pid "${sandbox}:${cloudlab}/" "${output}/cloudlab/"
printf 'Analysis snapshot: %s\n' "${output}"
