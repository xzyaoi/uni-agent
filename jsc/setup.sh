#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# Install the user-local uv binary without modifying shell startup files.
if ! command -v uv >/dev/null; then
    mkdir -p "${WORK_ROOT}/tools"
    installer=$(mktemp)
    trap 'rm -f "${installer}"' EXIT
    curl --fail --silent --show-error --location https://astral.sh/uv/install.sh --output "${installer}"
    UV_INSTALL_DIR="${WORK_ROOT}/tools" UV_NO_MODIFY_PATH=1 sh "${installer}"
    export PATH="${WORK_ROOT}/tools:${PATH}"
fi
exec bash "${BUNDLE_ROOT}/setup.sh"
