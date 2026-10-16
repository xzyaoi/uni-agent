#!/usr/bin/env bash
# Run on a Clariden login node; installs only the native Harbor client environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
command -v uv >/dev/null || { echo 'Install uv or add it to PATH first.' >&2; exit 1; }
(cd "${BUNDLE_ROOT}" && sha256sum -c assets/SHA256SUMS)
mkdir -p "${WORK_ROOT}/runtime" "${WORK_ROOT}/data" "${WORK_ROOT}/native-shell" "${WORK_ROOT}/runs"
if [[ ! -f ${WORK_ROOT}/.source-ready ]]; then
    tar -xzf "${BUNDLE_ROOT}/assets/runtime-source.tar.gz" -C "${WORK_ROOT}/runtime"
    tar -xzf "${BUNDLE_ROOT}/assets/native-shell.tar.gz" -C "${WORK_ROOT}/native-shell"
    touch "${WORK_ROOT}/.source-ready"
fi
if [[ ! -f ${WORK_ROOT}/.data-ready ]]; then
    tar -xzf "${BUNDLE_ROOT}/assets/tasks.tar.gz" -C "${WORK_ROOT}/data"
    touch "${WORK_ROOT}/.data-ready"
fi
export UV_PYTHON_INSTALL_DIR="${WORK_ROOT}/python"
export UV_CACHE_DIR="${WORK_ROOT}/uv-cache"
uv python install --no-bin 3.12
native_python=$(uv python find --managed-python 3.12)
uv venv --python "${native_python}" --allow-existing "${WORK_ROOT}/venv"
uv pip install --python "${WORK_ROOT}/venv/bin/python" -r "${BUNDLE_ROOT}/requirements-native.txt"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/prepare_data.py"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/apply_runtime_support.py"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/apply_rollout_batch.py" "${WORK_ROOT}/runtime"
"${WORK_ROOT}/venv/bin/python" "${BUNDLE_ROOT}/apply_continuous_rollout.py" "${WORK_ROOT}/runtime"
printf 'Ready: %s\nNext: create your credential file, then run ./submit.sh\n' "${WORK_ROOT}"
