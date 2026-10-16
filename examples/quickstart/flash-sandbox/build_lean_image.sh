#!/usr/bin/env bash
#
# Build the LEAN verl+Harbor+Megatron image from examples/quickstart/flash-sandbox/
# Containerfile.lean into a distinct .sqsh so the proven image (built by
# rllm/build_megatron_image.sh -> harbor-swe-megatron.x86_64.sqsh) is kept as a
# fallback.
#
# Lean = the install layer depends ONLY on the LOCAL pyproject.toml + the
# empty lean_image_deps/ package (both in examples/quickstart/flash-sandbox/),
# NOT on a separate rllm checkout -- so editing rllm/verl/uni_agent never
# invalidates the ~30-min pip layer. See Containerfile.lean for the full
# rationale (the [verl]+[harbor] extras + [tool.uv] knobs are a byte-for-byte
# copy of rllm/pyproject.toml).
#
# The build context is a THROWAWAY temp dir containing only:
#   - Containerfile.lean  (copied to ./Containerfile)
#   - pyproject.toml
#   - lean_image_deps/
# Built via podman -> enroot .sqsh, mirroring the proven build_megatron_image.sh.
#
# Usage:  sbatch build_lean_image.sh        (recommended; see SLURM block below)
#    or:  bash build_lean_image.sh          (interactive, on a node with podman)
# Env:    SCRATCH   (default: /iopsstor/scratch/cscs/xyao)
#         IMAGE_TAG (default: harbor-swe-megatron-lean:latest)
#         TORCH_CUDA_ARCH_LIST (default: 8.0 for A100)
#
# ---- SLURM submission header (must be at top, before any code) ------------
#SBATCH --job-name=lean-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --account=infra02
#SBATCH --time=02:30:00
# Build runs podman on the bare host (no --environment: this job does NOT
# run inside the enroot container, it builds one). The .toml is only needed
# by the training jobs that run inside the image.
#SBATCH --output=lean-image-%j.out
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[build] SLURM job ${SLURM_JOB_ID} on $(hostname)"
fi

export SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/xyao}"
# Under sbatch, BASH_SOURCE[0] points to Slurm's spool copy, not the submit
# dir -- so use SLURM_SUBMIT_DIR to locate the Containerfile.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HERE="${SLURM_SUBMIT_DIR}/examples/quickstart/flash-sandbox"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    HERE="$(pwd)"
fi
IMAGE_TAG="${IMAGE_TAG:-harbor-swe-megatron-lean:latest}"
# The dependency manifest (pyproject.toml) + the empty lean_image_deps/
# package live IN this directory -- the build no longer depends on a separate
# rllm checkout (RLLM_DIR). The [verl]+[harbor] extras + [tool.uv] resolution
# knobs are a byte-for-byte copy of rllm/pyproject.toml; re-sync this
# pyproject when upgrading.
HERE="$(cd "${HERE}" && pwd)"
[ -f "${HERE}/pyproject.toml" ] || { echo "[build] missing ${HERE}/pyproject.toml" >&2; exit 1; }
[ -f "${HERE}/lean_image_deps/__init__.py" ] || { echo "[build] missing ${HERE}/lean_image_deps/__init__.py" >&2; exit 1; }
SQSH="${SCRATCH}/harbor-swe-megatron-lean.x86_64.sqsh"
ARCH="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "[build] host=$(hostname) arch=sm_${ARCH} image=${IMAGE_TAG} sqsh=${SQSH}"

# podman storage (mirror proven build_megatron_image.sh setup).
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg-runtime"
export TMPDIR="/dev/shm/${USER}/tmp"
mkdir -p "${XDG_RUNTIME_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
rm -rf "/dev/shm/${USER}/runroot" "${HOME}/.tmp/storage-run-$(id -u)/libpod/tmp"
mkdir -p "${HOME}/.config/containers"
if [ ! -f "${HOME}/.config/containers/storage.conf" ]; then
    cp "${HERE}/../harbor_swe/cluster/storage.conf" "${HOME}/.config/containers/storage.conf" 2>/dev/null || true
fi

# ---- throwaway build context (Containerfile + LOCAL manifest) -----------
# Everything the build needs ships from examples/quickstart/flash-sandbox/ --
# no rllm checkout, no rllm-model-gateway copy. pyproject.toml is the manifest;
# lean_image_deps/ is the empty package `uv pip install '.[verl,harbor]'`
# builds (never imported at runtime).
CTX="$(mktemp -d "${TMPDIR}/lean-ctx.XXXXXX")"
trap 'rm -rf "${CTX}"' EXIT
cp "${HERE}/Containerfile.lean" "${CTX}/Containerfile"
cp "${HERE}/pyproject.toml" "${CTX}/pyproject.toml"
cp -a "${HERE}/lean_image_deps" "${CTX}/lean_image_deps"
echo "[build] context size: $(du -sh "${CTX}" | cut -f1)"

# ---- build ----------------------------------------------------------------
cd "${CTX}"
podman --cgroup-manager=cgroupfs build \
    --network=host \
    --build-arg TORCH_CUDA_ARCH_LIST="${ARCH}" \
    --build-arg MAX_JOBS="${MAX_JOBS:-8}" \
    -f Containerfile \
    -t "${IMAGE_TAG}" \
    .

# ---- export to .sqsh (enroot) ---------------------------------------------
# CSCS compute nodes have NO persistent node-local disk -- /tmp and /dev/shm
# are both tmpfs (RAM, ~350G each); the only persistent fs is Lustre
# (${SCRATCH}). The previous build wrote the ~21G mksquashfs output straight
# to Lustre and hit a transient 'Input/output error' on the large sequential
# write, losing the whole ~2.5h build. So: write the .sqsh to tmpfs FIRST
# (pure RAM -- mksquashfs never touches Lustre, so no OST write errors), then
# copy tmpfs -> Lustre with retries. tmpfs has ~350G and podman storage (also
# /dev/shm) is ~30-40G at this point, so the 21G staging file fits easily.
# enroot import writes the squashfs via mksquashfs, then unmounts/cleans up
# its temp dir (printed as 'enroot.XXXXXX'). In this slurm env the cleanup
# unmount can return non-zero EVEN AFTER mksquashfs fully completed -- the
# .sqsh is already written, enroot's exit code is just unreliable here. So
# we do NOT fail on a non-zero exit code; instead we trust the output file
# with a size sanity check (the full image is ~22.9G, so anything >= 20G is
# a complete squashfs; a truncated/partial one is rejected).
STAGE_DIR="/dev/shm/${USER}/sqsh-stage"
mkdir -p "${STAGE_DIR}" "${SCRATCH}"
rm -f "${SQSH}"
LOCAL_SQSH="${STAGE_DIR}/$(basename "${SQSH}")"
rm -f "${LOCAL_SQSH}"
set +e
enroot import -x mount -o "${LOCAL_SQSH}" "podman://${IMAGE_TAG}"
ENROOT_RC=$?
set -e
# A complete squashfs is ~22.9G (21,861 MiB on the 81676 build). Reject
# anything under 20G as truncated; accept the file even if enroot returned
# non-zero, since mksquashfs completes before enroot's cleanup step.
MIN_SQSH_BYTES=20000000000
if [ ! -s "${LOCAL_SQSH}" ]; then
    echo "[build] ERROR: enroot import (rc=${ENROOT_RC}) did not create ${LOCAL_SQSH}" >&2
    exit 1
fi
LOCAL_SZ="$(stat -c%s "${LOCAL_SQSH}")"
if [ "${LOCAL_SZ}" -lt "${MIN_SQSH_BYTES}" ]; then
    echo "[build] ERROR: ${LOCAL_SQSH} is only ${LOCAL_SZ} bytes (< ${MIN_SQSH_BYTES});" >&2
    echo "[build] likely truncated (enroot rc=${ENROOT_RC}). Re-run the build." >&2
    exit 1
fi
ls -lh "${LOCAL_SQSH}"
echo "[build] squashfs staged OK on tmpfs ($(numfmt --to=iec "${LOCAL_SZ}"); enroot rc=${ENROOT_RC})"
# Copy tmpfs -> Lustre with up to 5 retries. cp exits non-zero on a write
# I/O error (transient OST failures are common on large sequential writes); on
# success the byte count matches the source (O(1) stat check, no full re-read).
for attempt in 1 2 3 4 5; do
    rm -f "${SQSH}"
    if cp "${LOCAL_SQSH}" "${SQSH}" && [ "$(stat -c%s "${SQSH}")" = "${LOCAL_SZ}" ]; then
        echo "[build] sqsh landed on Lustre (attempt ${attempt}, ${LOCAL_SZ} bytes)"
        break
    fi
    echo "[build] cp attempt ${attempt} failed (Lustre I/O); retrying..." >&2
    rm -f "${SQSH}"
    sleep "$((attempt * 5))"
    if [ "$attempt" = 5 ]; then
        echo "[build] ERROR: failed to land ${SQSH} on Lustre after 5 attempts." >&2
        echo "[build] The .sqsh is staged in tmpfs at ${LOCAL_SQSH} (RAM, wiped at job end)." >&2
        echo "[build] Recover it NOW inside this allocation:  cp ${LOCAL_SQSH} ${SQSH}" >&2
        exit 1
    fi
done
rm -f "${LOCAL_SQSH}"
ls -lh "${SQSH}"
echo "[build] DONE: ${SQSH}"
echo "[build] To use it: set IMAGE=harbor-swe-megatron-lean.x86_64.sqsh in your"
echo "[build] .toml, and remove the 'pip:' section from runtime_env_flash.yaml"
echo "[build] (the 6 verl-0.9.0.dev deps are now baked into the image)."
