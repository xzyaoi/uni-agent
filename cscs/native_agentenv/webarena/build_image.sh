#!/usr/bin/env bash
# Run on an amd64 machine with Docker; push the resulting image separately.
set -euo pipefail
image=${1:?Usage: build_image.sh ghcr.io/your-account/webarena:version}
directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docker build --platform linux/amd64 -f "${directory}/Containerfile" -t "${image}" "${directory}"
printf 'Built %s\nPublish with: docker push %q\n' "${image}" "${image}"
