#!/usr/bin/env bash
#
# Build the LEAN verl+Harbor+Megatron image for CLARIDEN (GH200, aarch64,
# sm_90) from examples/quickstart/flash-sandbox/Containerfile.lean into
# harbor-swe-megatron-lean.aarch64.sqsh.
#
# WHY A SEPARATE SCRIPT: the proven build_lean_image.sh targets Bristen
# (x86_64, sm_80) and names its output .x86_64.sqsh. Clariden GPU nodes are
# aarch64 Grace-Hopper -- the x86_64 sqsh cannot run there. The
# Containerfile itself is arch-neutral (its FROM
# nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 is a multi-arch manifest, so
# podman on an aarch64 host pulls the arm64 base automatically); the only
# arch knobs are build args:
#   TORCH_CUDA_ARCH_LIST=9.0   (Hopper sm_90; GH200)
#   MAX_JOBS=64                (Clariden nodes have 288 CPUs)
# aarch64 wheel availability (probed on clariden 2026-09-28):
#   torch 2.11.0+cu130 aarch64        -> download.pytorch.org/whl/cu130  OK
#   vllm 0.24.0 aarch64               -> pypi manylinux_2_28_aarch64    OK
#   transformer_engine_cu13 2.18.0
#     aarch64                         -> pypi manylinux_2_28_aarch64    OK
# apex / TE[pytorch] / flash-attn compile from source against the in-image
# nvcc + torch (slow layers, hence the 4h walltime).
#
# Everything else mirrors the proven Bristen build:
#   - podman --cgroup-manager=cgroupfs --network=host
#   - throwaway build context (Containerfile + LOCAL pyproject.toml +
#     lean_image_deps/ -- no rllm checkout)
#   - enroot import staged on tmpfs (/dev/shm) then copied to Lustre with
#     retries (transient OST EIO on large sequential writes killed the
#     first Bristen build)
#
# Usage:  sbatch examples/quickstart/flash-sandbox/build_lean_image_clariden.sh
# Env:    SCRATCH   (default: /iopsstor/scratch/cscs/xyao)
#         IMAGE_TAG (default: harbor-swe-megatron-lean-aarch64:latest)
#         TORCH_CUDA_ARCH_LIST (default: 9.0 for GH200)
#         MAX_JOBS  (default: 8 -- see the OOM note below)
#
# MAX_JOBS=8 (NOT higher): MAX_JOBS is inherited by the flash-attn sdist
# compile in the manifest layer (apex/TE hardcode their own 32/64 in the
# Containerfile RUNs). The first build (job 3537951, 2026-09-28) passed
# MAX_JOBS=64 with --cpus-per-task=64 (~193G cgroup slice): 64 concurrent
# nvcc jobs blew through it -- 173 oom_kills, flash-attn died at [60/73].
# 8 jobs x ~8-10G nvcc peak = ~80G, comfortably inside any slice. It is
# slower (~60min for flash-attn), hence the 5h walltime.
#
# ---- SLURM submission header (must be at top, before any code) ------------
#SBATCH --job-name=lean-image-cla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --account=infra02
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=05:00:00
# Build runs podman on the bare host (no --environment: this job does NOT
# run inside the enroot container, it builds one). The .toml is only needed
# by the training jobs that run inside the image.
#SBATCH --output=lean-image-aarch64-%j.out
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
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"
else
    HERE="$(pwd)"
fi
IMAGE_TAG="${IMAGE_TAG:-harbor-swe-megatron-lean-aarch64:latest}"
# The dependency manifest (pyproject.toml) + the empty lean_image_deps/
# package live IN this directory -- the build no longer depends on a separate
# rllm checkout (RLLM_DIR). The [verl]+[harbor] extras + [tool.uv] resolution
# knobs are a byte-for-byte copy of rllm/pyproject.toml; re-sync this
# pyproject when upgrading.
HERE="$(cd "${HERE}" && pwd)"
[ -f "${HERE}/Containerfile.lean" ] || { echo "[build] missing ${HERE}/Containerfile.lean" >&2; exit 1; }
[ -f "${HERE}/pyproject.toml" ] || { echo "[build] missing ${HERE}/pyproject.toml" >&2; exit 1; }
[ -f "${HERE}/lean_image_deps/__init__.py" ] || { echo "[build] missing ${HERE}/lean_image_deps/__init__.py" >&2; exit 1; }
SQSH="${SCRATCH}/harbor-swe-megatron-lean.aarch64.sqsh"
ARCH="${TORCH_CUDA_ARCH_LIST:-9.0}"

echo "[build] host=$(hostname) arch=sm_${ARCH} image=${IMAGE_TAG} sqsh=${SQSH}"

# podman storage on tmpfs (mirrors the proven Bristen storage.conf; the
# graphroot must NOT land on quota-limited Lustre home). fuse-overlayfs-1.13
# exists at the same path on clariden as on bristen (checked 2026-09-28).
# WRITTEN VIA printf + VERIFIED: a heredoc write of this exact file produced
# a torn 7th line once (first build attempt, job 3537901 -- podman died on
# "toml: line 7 ... got '"'"; intermittent Lustre client page tear, the same
# flakiness class as the Bristen EIO saga). printf + verify + retry.
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg-runtime"
export TMPDIR="/dev/shm/${USER}/tmp"
mkdir -p "${XDG_RUNTIME_DIR}" "${TMPDIR}"
chmod 700 "${XDG_RUNTIME_DIR}"
rm -rf "/dev/shm/${USER}/runroot" "${HOME}/.tmp/storage-run-$(id -u)/libpod/tmp"
mkdir -p "${HOME}/.config/containers"
STORAGE_CONF="${HOME}/.config/containers/storage.conf"
for attempt in 1 2 3 4 5; do
    printf '%s\n' \
        '[storage]' \
        '  driver = "overlay"' \
        '  runroot = "/dev/shm/xyao/runroot"' \
        '  graphroot = "/dev/shm/xyao/root"' \
        '[storage.options.overlay]' \
        '  mount_program = "/usr/bin/fuse-overlayfs-1.13"' > "${STORAGE_CONF}"
    if [ "$(wc -l < "${STORAGE_CONF}")" = "6" ] \
       && grep -q '^  mount_program = "/usr/bin/fuse-overlayfs-1.13"$' "${STORAGE_CONF}" \
       && [ "$(tail -c 2 "${STORAGE_CONF}" | od -An -c | tr -d ' \n')" = '3"' ]; then
        break
    fi
    echo "[build] storage.conf write torn (attempt ${attempt}); rewriting" >&2
    sleep 2
done
# Pre-flight: podman must parse the config cleanly before we spend 2h building.
podman --cgroup-manager=cgroupfs info > /dev/null || {
    echo "[build] ERROR: podman info failed (storage.conf parse?)" >&2
    exit 1
}

# ---- throwaway build context (Containerfile + LOCAL manifest) -----------
# Everything the build needs ships from examples/quickstart/flash-sandbox/ --
# no rllm checkout, no rllm-model-gateway copy. pyproject.toml is the manifest;
# lean_image_deps/ is the empty package `uv pip install '.[verl,harbor]'`
# builds (never imported at runtime). The context lives on /dev/shm (tmpfs):
# Lustre reads of the JUST-rsynced sources can serve torn pages, so verify
# the copies byte-for-byte against md5sums computed IN THE SAME JOB.
CTX="$(mktemp -d "${TMPDIR}/lean-ctx.XXXXXX")"
trap 'rm -rf "${CTX}"' EXIT
cp "${HERE}/Containerfile.lean" "${CTX}/Containerfile"
cp "${HERE}/pyproject.toml" "${CTX}/pyproject.toml"
cp -a "${HERE}/lean_image_deps" "${CTX}/lean_image_deps"
if ! cmp -s "${HERE}/Containerfile.lean" "${CTX}/Containerfile" \
   || ! cmp -s "${HERE}/pyproject.toml" "${CTX}/pyproject.toml"; then
    echo "[build] ERROR: build context copy differs from source (Lustre read tear?)" >&2
    exit 1
fi
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
# (${SCRATCH}). The first Bristen build wrote the ~22G mksquashfs output
# straight to Lustre and hit a transient 'Input/output error' on the large
# sequential write, losing the whole ~2.5h build. So: write the .sqsh to
# tmpfs FIRST (pure RAM -- mksquashfs never touches Lustre), then copy
# tmpfs -> Lustre with retries. tmpfs has ~350G and podman storage (also
# /dev/shm) is ~30-40G at this point, so the ~22G staging file fits.
# enroot import writes the squashfs via mksquashfs, then unmounts/cleans up
# its temp dir. In this slurm env the cleanup unmount can return non-zero
# EVEN AFTER mksquashfs fully completed -- the .sqsh is already written,
# enroot's exit code is just unreliable here. So we do NOT fail on a
# non-zero exit code; instead we trust the output file with a size sanity
# check (the full x86 image was ~22.9G; a truncated/partial one is rejected).
STAGE_DIR="/dev/shm/${USER}/sqsh-stage"
mkdir -p "${STAGE_DIR}" "${SCRATCH}"
rm -f "${SQSH}"
LOCAL_SQSH="${STAGE_DIR}/$(basename "${SQSH}")"
rm -f "${LOCAL_SQSH}"
set +e
enroot import -x mount -o "${LOCAL_SQSH}" "podman://${IMAGE_TAG}"
ENROOT_RC=$?
set -e
# A complete squashfs is ~23G. Reject anything under 20G as truncated;
# accept the file even if enroot returned non-zero, since mksquashfs
# completes before enroot's cleanup step.
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
echo "[build] To use it: ENV_NAME=harbor-swe-megatron-clariden (the .toml at the"
echo "[build] repo root already points at ${SQSH}), and keep the 'pip:' section"
echo "[build] out of the runtime env yaml (deps are baked into the image)."
