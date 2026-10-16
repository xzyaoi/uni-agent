#!/usr/bin/env bash
# Run on the JSC login host after submission, while an image transfer is active.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
training_job=${1:?Usage: release_when_ready.sh TRAINING_JOB PLATFORM_JOB IMAGE_BYTES IMAGE_SHA256}
platform_job=${2:?Specify the one-node readiness job}
expected_bytes=${3:?Specify the source image size}
expected_sha=${4:?Specify the source SHA256}
[[ $training_job =~ ^[0-9]+$ && $platform_job =~ ^[0-9]+$ && $expected_bytes =~ ^[0-9]+$ && $expected_sha =~ ^[a-f0-9]{64}$ ]] || exit 1
for _ in $(seq 1 1440); do
    bytes=$(stat -c %s "${TRAINING_IMAGE}" 2>/dev/null || echo 0)
    if [[ $bytes = "$expected_bytes" ]]; then
        actual_sha=$(sha256sum "${TRAINING_IMAGE}" | awk '{print $1}')
        [[ $actual_sha = "$expected_sha" ]] || { echo 'Image SHA256 mismatch; jobs remain held' >&2; exit 1; }
        mkdir -p "${WORK_ROOT}/diagnostics"
        printf '%s  %s\n' "$actual_sha" "${TRAINING_IMAGE}" > "${WORK_ROOT}/diagnostics/image.sha256"
        scontrol release "$platform_job"
        scontrol release "$training_job"
        echo "Released readiness job $platform_job and dependent training job $training_job"
        exit 0
    fi
    sleep 30
done
echo 'Image transfer did not complete within twelve hours; jobs remain held' >&2
exit 1
