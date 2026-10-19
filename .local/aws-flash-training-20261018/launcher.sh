#!/bin/bash
#SBATCH --job-name=uni-qwen35-cla
#SBATCH --partition=normal
#SBATCH --account=infra02
#SBATCH --time=11:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --output=uni-qwen35-clariden-%j.out

# Launch uni-agent's examples/quickstart/training/train_qwen3p5_dense.sh on
# CLARIDEN (GH200 aarch64, 4 GPUs/node, 288 CPUs/node) inside the AARCH64
# Megatron enroot image, with the OpenTela Flash Sandbox as the tool boundary
# (provider swap vs vefaas).
#
# Adapted from run_train_qwen3p5_bristen.sh (the proven x86 A100 Bristen
# pipeline, jobs 80205/82077/82104). Structural changes for Clariden:
#   0. ARCH: Clariden GPU nodes are aarch64 (Grace) with sm_90 Hopper GPUs --
#      the Bristen x86_64 sqsh CANNOT run here. Image:
#      harbor-swe-megatron-lean.aarch64.sqsh, built by
#      examples/quickstart/flash-sandbox/build_lean_image_clariden.sh
#      (TORCH_CUDA_ARCH_LIST=9.0). Same baked-in stack: megatron-core 0.17.1
#      + mbridge@691a377f + TE 2.18 + apex (sm_90) + vLLM 0.24.0 (GDN fix)
#      + swebench 4.1.0 + flash-linear-attention 0.5.2.
#   1. SLURM: --account=infra02 (Bristen used a-infra02), --cpus-per-task=288
#      (whole Grace node), --time=11:30:00 (the `normal` partition caps at
#      12h; Bristen allowed 24h). Topology unchanged: TP4 x PP2 x CP1 =
#      8 GPUs/replica; 8 nodes x 4 GPU = 32 GPUs -> 4 replicas (4 DP).
#   2. NCCL: the Bristen toml forced NCCL_NET=Socket (Bristen eni issue).
#      Here the toml enables the host aws-ofi-nccl hook (variant cuda13)
#      over Slurm's Slingshot/CXI libfabric netstack instead -- multi-node
#      NCCL goes over RDMA, not TCP. Do NOT add NCCL_NET=Socket back.
#   3. Same shared scratch: /iopsstor is CSCS cross-cluster scratch, so
#      REPO_DIR, MODEL_PATH, TRAIN_FILE etc. resolve to the same absolute
#      paths as on Bristen. The repo is shipped to workers by
#      `ray job submit --runtime-env working_dir: ./` + PYTHONPATH.
#   4. Manual head/worker bootstrap via run_ray_node.sh instead of
#      `ray symmetric-run` (train_qwen3p5_dense.sh uses `ray job submit
#      --no-wait`, which returns immediately; symmetric-run would tear the
#      cluster down before the submitted job runs). Unchanged from Bristen.
#
# GH200 memory note (96GB HBM3 vs Bristen A100 80GB): slightly MORE headroom
# per GPU. Keep the Bristen-validated actor_ppo_max_token_len math in mind:
#   actor_ppo_max_token_len = (max_prompt + max_response) / train_cp, with
#   CP=1 (forced by the Qwen3.5 GDN layers -- megatron-bridge
#   qwen35_vl_provider asserts context_parallel_size==1) and
#   MAX_RESPONSE_LENGTH=32768 -> (8192+32768)/1 = 40960 tok/micro-batch,
#   the setting job 80205's successor 82104-class runs validated at on
#   80GB. MAX_RESPONSE_LENGTH=65536 (73728 tok/micro-batch) died on vLLM
#   cudaErrorIllegalAddress on A100 (job 80231) -- start at 32768 here too,
#   scale up only after a step lands.
#
# Usage (from a Clariden login node -- the OTELA key is read from ~/.zshrc):
#   sbatch run_train_qwen3p5_clariden.sh
#   # 2-node smoke (~1 h) before committing 32 GPUs:
#   sbatch --nodes=2 --export=NNODES=2,MAX_RESPONSE_LENGTH=16384,CONCURRENCY=64 run_train_qwen3p5_clariden.sh
#   # Qwen3.8-27B with an explicit reasoning effort (template default is xhigh);
#   # recorded in sandbox exec metadata as llm.reasoning_effort for trace bucketing:
#   sbatch --export=MODEL_PATH=$SCRATCH/data/models/Qwen3.8-27B,REASONING_EFFORT=xhigh run_train_qwen3p5_clariden.sh
#
# Override any tunable with `sbatch --export=VAR=val,...`. See run_ray_node.sh
# for the per-node Ray lifecycle.

set -euo pipefail
[ "${1:-}" = "--" ] && shift
: "${SCRATCH:?SCRATCH is not set -- run on a Clariden login node}"

# All temp files (mktemp, Python tempfile, pip cache/build) go to scratch,
# NOT /users/xyao/.tmp -- home is quota-limited and Bristen job 80155 died on
# "Disk quota exceeded" from the bare `mktemp` in the EDF check below.
# Exported here (launcher) AND propagated to workers via RUN_ENV below.
export TMPDIR="${SCRATCH}/tmp"
mkdir -p "${TMPDIR}"
export PIP_CACHE_DIR="${SCRATCH}/pip-cache"
mkdir -p "${PIP_CACHE_DIR}"
# In-container HOME (set to ${SCRATCH}/home by the enroot .toml [env]).
mkdir -p "${SCRATCH}/home"

# Pull the Flash Sandbox key from the user's interactive shell rc if it wasn't
# already in the SLURM submit environment (non-interactive ssh / cron-style
# submits don't source ~/.zshrc). Only the OTELA_API_KEY line is extracted.
if [ -z "${OTELA_API_KEY:-}" ] && [ -f "${HOME}/.zshrc" ]; then
    _key="$(sed -nE 's/^[[:space:]]*export[[:space:]]+OTELA_API_KEY=([^[:space:]]+).*/\1/p' "${HOME}/.zshrc" | head -1)"
    case "${_key}" in
        \"*\") _key="${_key#\"}"; _key="${_key%\"}" ;;
        \'*\') _key="${_key#\'}"; _key="${_key%\'}" ;;
    esac
    [ -n "${_key}" ] && export OTELA_API_KEY="${_key}"
fi
: "${OTELA_API_KEY:?OTELA_API_KEY is required for the Flash Sandbox service (set it in your env or ~/.zshrc)}"

unset WANDB_API_KEY WANDB_ENTITY
LOGGER="['console']"

REPO_DIR="/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_swepro_flash_2n45m_20261018/source"
HERE="/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_swepro_flash_2n45m_20261018"
ENV_NAME="${ENV_NAME:-harbor-swe-megatron-clariden}"
# pyxis resolves --environment=NAME by searching EDF_PATH for NAME.toml.
export EDF_PATH="${HERE}:${EDF_PATH:-}"

CLUSTER_NNODES="${SLURM_NNODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-288}"

# The EDF toml (harbor-swe-megatron-clariden.toml) resolves
# ${SCRATCH}/harbor-swe-megatron-lean.aarch64.sqsh at container start from the
# JOB env. sbatch --export=ALL forwards the submitter's env, so a caller with
# a non-canonical SCRATCH (e.g. pointing at the repo dir) breaks image
# resolution with a cryptic pyxis "Invalid argument". Pin the canonical
# scratch root -- this wrapper is clariden-specific and hardcodes xyao paths
# throughout anyway.
SCRATCH="/iopsstor/scratch/cscs/xyao"
export SCRATCH

# ---- Trainer topology: pool split for separate_async (fully-async) ----
# colocate_async: one hybrid pool on ALL cluster nodes.
# separate_async: sbatch must request train+rollout nodes; the ray cluster
# spans ALL of them (NNODES_T = SLURM nodes), the trainer/hybrid pool gets
# NNODES = cluster - NNODES_ROLLOUT, and NNODES_ROLLOUT nodes run standalone
# vLLM replicas fed by the nccl checkpoint engine.
TRAINER_MODE="${TRAINER_MODE:-colocate_async}"
# Respect a caller-provided NNODES_ROLLOUT (sbatch --export). The old
# hard `NNODES_ROLLOUT=0` reset clobbered the exported value, which made
# the branch's "${NNODES_ROLLOUT:-2}" a permanent no-op (var already set
# to 0) -> the train launcher asserted NNODES_ROLLOUT > 0 and the
# separate_async legs died at boot (job 3546097, 20:36).
NNODES_ROLLOUT="${NNODES_ROLLOUT:-0}"
NNODES="${CLUSTER_NNODES}"
NNODES_T="${CLUSTER_NNODES}"
if [ "${TRAINER_MODE}" = "separate_async" ]; then
    # unset OR explicitly 0 -> the standalone-pool default
    if [ "${NNODES_ROLLOUT}" -le 0 ]; then NNODES_ROLLOUT=2; fi
    if [ "${CLUSTER_NNODES}" -le "${NNODES_ROLLOUT}" ]; then
        echo "separate_async needs sbatch nodes (${CLUSTER_NNODES}) > NNODES_ROLLOUT (${NNODES_ROLLOUT})" >&2
        exit 1
    fi
    NNODES=$((CLUSTER_NNODES - NNODES_ROLLOUT))
fi

# ---- Per-run env for run_ray_node.sh + train_qwen3p5_dense.sh ----
RUN_ENV=(
    TRAINING_STEPS="${TRAINING_STEPS}"
    PROJECT_NAME="${PROJECT_NAME}"
    CKPTS_DIR="${CKPTS_DIR}"
    AGENT_LOG_DIR="${AGENT_LOG_DIR}"
    PYTHONPATH="${PYTHONPATH}"
    TEST_FREQ="${TEST_FREQ}"
    OFFLOAD="${OFFLOAD}"
    OFFLOAD_FRACTION="${OFFLOAD_FRACTION}"
    USER="${USER:-$(id -un)}"
    OTELA_API_KEY="${OTELA_API_KEY}"
    # Required on the rollout workers (not just the head): _resolve_image
    # in flash_sandbox.py reads this at actor-construction time to rewrite
    # the bare swebench/sweb.eval.x86_64.<id> ref to the GHCR mirror with
    # /testbed + git + the SWE harness. get_ppo_ray_runtime_env does NOT
    # forward it from the job-level runtime_env_flash.yaml, so it must be a
    # real process env here (inherited by the raylet -> actors, same path
    # that delivers OTELA_API_KEY above).
    FLASH_SANDBOX_IMAGE_REGISTRY="${FLASH_SANDBOX_IMAGE_REGISTRY-}"
    # Point the flash-sandbox client at the INTERNAL SwissAI mirror (root
    # path, trailing slash REQUIRED). Probed from clariden 2026-09-29 ~01:30:
    #   api.opentela.ai (public relay): HTTP 400 "no providers found" for ALL
    #     backends (libkrun/docker/firecracker) -- full relay outage, 15k
    #     failed start_sandbox in smoke job 3538610.
    #   sandbox.swissai.cscs.ch: reachable from clariden (the 08-30 outage,
    #     bristen job 3209975, was Bristen-specific). Serving the API at ROOT
    #     (POST /sandboxes), NOT /v1/service/flash-sandbox/ (404 there).
    #     backend=docker: HEALTHY -- real start_sandbox succeeded with both
    #     swebench/sweb.eval.x86_64.astropy_1776_astropy-12907 (Docker Hub)
    #     and the ghcr.io/swe-images mirror ref. libkrun/firecracker:
    #     "no healthy nodes available" on the mirror right now.
    # TRAILING SLASH REQUIRED (the SDK urljoins onto it).
    FLASH_SANDBOX_ENDPOINT="${FLASH_SANDBOX_ENDPOINT:-https://api.opentela.ai/v1/service/flash-sandbox/}"
    # Backend selection (2026-09-29 ~10:00): opentela libkrun is the only live
    # combination -- the SwissAI mirror lost ALL backends ("no healthy nodes"
    # for docker/libkrun/firecracker) mid-run of job 3540514, and opentela's
    # docker/firecracker are also dead. Both endpoints have been flapping all
    # day (mirror: healthy 01:30-04:00, dead after; opentela: dead 01:30,
    # libkrun alive ~09:00). Probe before submitting:
    #   repro/probe_libkrun_now.py (FLASH_SANDBOX_ENDPOINT/TYPE env)
    # Mirror fallback (root path, trailing slash REQUIRED):
    #   FLASH_SANDBOX_ENDPOINT=https://sandbox.swissai.cscs.ch/
    #   FLASH_SANDBOX_TYPE=docker
    FLASH_SANDBOX_TYPE="${FLASH_SANDBOX_TYPE:-libkrun}"
    REPO_DIR="${REPO_DIR}"
    TRAIN_SCRIPT="/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_swepro_flash_2n45m_20261018/train.sh"
    RAY_PORT="${RAY_PORT:-6379}"
    RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
    # TMPDIR: NODE-LOCAL /tmp for ALL Python tempfile (Triton JIT, torch
    # inductor, vLLM compile). ${SCRATCH}/tmp (Lustre) EIO'd under 8-way
    # concurrent Triton JIT compiles and killed Bristen job 82077 -- /tmp is
    # the same node-local tmpfs RAY_TMPDIR already uses. Do NOT merge
    # RAY_TMPDIR into this (ray object store wants its own dir).
    TMPDIR="/tmp"
    PIP_CACHE_DIR="${PIP_CACHE_DIR}"
    NNODES="${NNODES}"
    NNODES_T="${NNODES_T}"
    NNODES_ROLLOUT="${NNODES_ROLLOUT}"
    TRAINER_MODE="${TRAINER_MODE}"
    PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-4}"
    NGPUS_PER_NODE="${GPUS_PER_NODE}"
    CPUS_PER_NODE="${CPUS_PER_NODE}"
    # Data (script defaults append an extra /data/uni_agent segment, so set
    # DATA_DIR to the parent and the parquets explicitly).
    DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/xyao/data}"
    MODEL_PATH="${MODEL_PATH:-/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-4B}"
    # TRAIN on swebench-verified for now: only the Verified subset is mirrored
    # on ghcr.io/swe-images (the registry FLASH_SANDBOX_IMAGE_REGISTRY rewrites
    # to). SWE-rebench rows carry a `swerebench/...` ref whose ghcr.io mirror
    # does NOT exist -> sandbox-create HTTP 400 (Bristen job 80162).
    TRAIN_FILE="${TRAIN_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_verified.parquet}"
    TEST_FILE="${TEST_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_verified.parquet}"
    RUNTIME_DIR="${RUNTIME_DIR:-/iopsstor/scratch/cscs/xyao}"
    RUNTIME_ENV="${RUNTIME_ENV:-examples/quickstart/training/runtime_env_flash.yaml}"
    # react (not claude_code): react is uni_agent builtin (pydantic + uni_agent.tools,
    # both in the Megatron image) and its stateful_shell uses the SDK persistent
    # shell_exec. claude_code shells out to install the external `claude` CLI
    # in-sandbox, which fails on the verified sweb images (Bristen job 80164).
    TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/training/task_config_react_clariden.yaml}"
    # Reasoning-effort knob (Qwen3.5+/3.8 chat template). Empty = model default
    # (xhigh on Qwen3.8; no-op on 3.5). Flows via train_qwen3p5_dense.sh to
    # data.apply_chat_template_kwargs (gateway generation) and is mirrored into
    # sandbox exec metadata as llm.reasoning_effort (framework.py _rlvr_base_metadata).
    REASONING_EFFORT="${REASONING_EFFORT:-}"
    # Topology: TP4 x PP2 x CP1 = 8/replica -> 4 DP on 32 GPUs. CP=1 forced by
    # GDN (upstream's CP=2 asserts in megatron-bridge).
    GEN_TP="${GEN_TP:-2}" TP="${TP:-4}" PP="${PP:-2}" CP="${CP:-1}"
    # Rollout / batch.
    # CONCURRENCY: sandbox-start load drives shared-backend stability. 256
    # caused retry storms on BOTH providers (mirror job 3540514: 1049 sessions
    # for 330 traj then total outage; opentela job 3540695: 3520 sessions for
    # 19 traj). 64 is the proven level (smoke 3539089 AND full-run job
    # 3540792: clean 517-traj batch, PPO step 1 at 49min). Across 16 engines
    # (8 nodes) 64 concurrency still fills GPUs better than the smoke did.
    CONCURRENCY="${CONCURRENCY:-64}"
    GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
    # vLLM KV-cache budget as a fraction of GPU memory (train_qwen3p5_dense.sh).
    # Smoke 3539089 measured: actor training peaks at 18.6GB alloc / 23GB
    # reserved (colocate_async sleeps vLLM during the train phase, so the two
    # do not overlap); rollout phase ran ~77GB total with ZERO KV pressure or
    # preemptions at 0.7. GDN's fixed-size per-sequence state means KV cost
    # barely grows with context, so raising this is cheap headroom. 0.8 adds
    # ~10GB of KV per GPU with ~19GB still free for the actor's reserved
    # cache. Override: sbatch --export=...VLLM_GPU_MEM_UTIL=0.8
    VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.8}"
    TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
    N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
    # GH200-safe context (see header). Same start point as the validated
    # Bristen A100 setting; 65536 faulted there (80231) -- do not go up until
    # a step lands at 32768.
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
    # Per-step scheduler budget (train_qwen3p5_dense.sh max_num_batched_tokens).
    # Cap < max_model_len forces large react-transcript prefills to chunk,
    # avoiding the vLLM GDN/M-RoPE cudaErrorIllegalAddress (80231/44/48).
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
    # Algorithm (upstream GRPO baseline -- matches origin/main exactly).
    ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
    LOSS_MODE="${LOSS_MODE:-vanilla}"
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
    CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
    CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
    EXP_NAME="${EXP_NAME:-qwen35_react_flash_clariden_$(date +%Y%m%d%H%M)}"
    # Disable periodic checkpointing (verl default save_freq=10 EIO'd on the
    # first Megatron dist_ckpt write to Lustre, killing Bristen job 82104 at
    # step 10 after 3h14m). save_freq=1000 = no save during this run.
    # Override by sbatch --export=SAVE_FREQ=...; flows to train_qwen3p5_dense.sh:89.
    SAVE_FREQ="${SAVE_FREQ:-1000}"
    # Sandbox-start hardening for relay flakiness (transient HTTP 400s at
    # create): 5 attempts, 10s/20s/30s/40s linear backoff between them, 120s
    # per-attempt cap (was 3 attempts, 2s/4s, 60s). Override via --export.
    SANDBOX_START_RETRIES="${SANDBOX_START_RETRIES:-5}"
    SANDBOX_START_RETRY_BACKOFF="${SANDBOX_START_RETRY_BACKOFF:-10}"
    SANDBOX_START_ATTEMPT_TIMEOUT="${SANDBOX_START_ATTEMPT_TIMEOUT:-120}"
)
# Forward wandb creds + LOGGER to workers. The key is only added when present
# (LOGGER is console-only otherwise, so verl never calls wandb.init and the
# empty key can't trip wandb's "must pass API key" guard).
if [ -n "${WANDB_API_KEY:-}" ]; then
    RUN_ENV+=(
        WANDB_API_KEY="${WANDB_API_KEY}"
        WANDB_ENTITY="${WANDB_ENTITY:-}"
        LOGGER="${LOGGER}"
    )
else
    RUN_ENV+=(LOGGER="${LOGGER}")
fi

# ---- Verify the enroot environment starts (image + mounts) ----
# Host-side enroot (runtime.sh, /etc/enroot/hooks.d/10-netstack.sh) creates
# tiny here-doc temp files in $TMPDIR. With TMPDIR on Lustre, transient EIO
# there kills every enroot bootstrap (Bristen jobs 81650/81661/81676/81726/
# 81958). Pin the HOST-side TMPDIR to node-local /tmp for every srun/enroot
# start below. The IN-CONTAINER TMPDIR is ALSO /tmp (set in RUN_ENV above) --
# both layers node-local, zero temp writes to Lustre during the whole run.
EDF_ERROR="$(TMPDIR=/tmp mktemp)"
cleanup() { rm -f "${EDF_ERROR}"; }
trap cleanup EXIT
if ! TMPDIR=/tmp srun --export=ALL,TMPDIR=/tmp --nodes=1 --ntasks=1 --environment="${ENV_NAME}" true 2>"${EDF_ERROR}"; then
    echo "ERROR: could not start the ${ENV_NAME} enroot env;" >&2
    echo "       build the image (examples/quickstart/flash-sandbox/" >&2
    echo "       build_lean_image_clariden.sh) and check ${HERE}/harbor-swe-megatron-clariden.toml" >&2
    cat "${EDF_ERROR}" >&2
    exit 1
fi

# ---- Head node name (for worker --address) ----
mapfile -t ALLOCATED_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${ALLOCATED_NODES[0]}"
RUN_ENV+=(HEAD_NODE="${HEAD_NODE}")
echo "[launch] nnodes=${CLUSTER_NNODES} gpus=${GPUS_PER_NODE}/node head=${HEAD_NODE}"

# Pre-scrub leftover Ray daemons from prior jobs on the freshly allocated nodes.
TMPDIR=/tmp srun --export=ALL,TMPDIR=/tmp --nodes="${CLUSTER_NNODES}" --ntasks="${CLUSTER_NNODES}" --ntasks-per-node=1 \
    --environment="${ENV_NAME}" \
    bash -c 'pkill -9 -x raylet 2>/dev/null; pkill -9 -f "[r]ay::" 2>/dev/null; pkill -9 -f "[g]cs_server" 2>/dev/null; sleep 1; true' || true

# ---- Single long-lived srun step: head + workers (Ray lives until training ends) ----
TMPDIR=/tmp srun --export=ALL,TMPDIR=/tmp --nodes="${CLUSTER_NNODES}" --ntasks="${CLUSTER_NNODES}" --ntasks-per-node=1 \
    --cpus-per-task="${CPUS_PER_NODE}" \
    --gpus-per-task="${GPUS_PER_NODE}" \
    --environment="${ENV_NAME}" \
    env "${RUN_ENV[@]}" \
    bash "/iopsstor/scratch/cscs/xyao/uni-agent/runs/qwen35_27b_swepro_flash_2n45m_20261018/run_ray_node.sh"
TRAIN_RC=$?

echo "[launch] done (rc=${TRAIN_RC})"
exit ${TRAIN_RC}
