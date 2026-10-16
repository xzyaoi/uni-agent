#!/bin/bash
#SBATCH --job-name=uni-qwen35-train
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --output=uni-qwen35-train-%j.out

# Launch uni-agent's examples/quickstart/training/train_qwen3p5_dense.sh on
# Bristen (x86 A100, 4 GPUs/node) inside the Megatron enroot image, with the
# OpenTela Flash Sandbox as the tool boundary (provider swap vs vefaas).
#
# Mirrors examples/harbor_swe/bristen/run_train.sh (proven pyxis --environment
# + single srun step lifecycle) with two structural changes for uni-agent:
#   1. workdir + PYTHONPATH -> the uni-agent repo + its BUNDLED verl 0.9.0.dev
#      (not the image's /workspace/rllm/rllm-bundled verl 0.8.0). The repo is
#      shipped to workers by `ray job submit --runtime-env working_dir: ./`.
#   2. Manual head/worker bootstrap via run_ray_node.sh instead of
#      `ray symmetric-run`, because train_qwen3p5_dense.sh uses
#      `ray job submit --no-wait` (returns immediately) and symmetric-run
#      would tear the cluster down before the submitted job runs.
#
# Topology (upstream GRPO baseline): TP4 x PP2 x CP1 = 8 GPUs/replica; 8 nodes x
# 4 GPU = 32 GPUs -> 4 replicas (4 DP). adv=grpo loss=vanilla agg=token-mean,
# clip 0.2/0.28, c-clip 10.0. NOTE: upstream uses PP1 x CP2, but the Qwen3.5
# GDN (gated-delta-net) layers assert context_parallel_size==1 in the image's
# megatron-bridge (qwen35_vl_provider.finalize -> MCoreTransformerConfig), so
# CP must be 1 here -- the only deviation from origin/main's topology. Matches
# the rest of train_qwen3p5_dense.sh except the 3 GDN patches + flash sandbox.
#
# A100 80GB memory note: the script computes actor_ppo_max_token_len =
# (max_prompt_length + max_response_length) / train_cp (no env override). With
# CP=1 (forced by GDN, see above) and MAX_RESPONSE_LENGTH=32768 this is
# (8192+32768)/1 = 40960 tok/micro-batch -- 2x the 24576 tok/micro-batch that
# job 80205 ran 20 steps at (MAX_RESPONSE_LENGTH=16384). Job 80231 tried
# MAX_RESPONSE_LENGTH=65536 -> 73728 tok/micro-batch and died on the FIRST
# train batch with vLLM cudaErrorIllegalAddress (Worker_TP0, EngineDeadError;
# 17x "All rollouts failed at global_steps=1" landing on the dead GPU), so
# 73728 is NOT safe on these A100s with enforce_eager=False. Start at 32768;
# once a step lands, scale up via --export=MAX_RESPONSE_LENGTH=49152
# (-> 57344 tok/micro-batch) and recheck headroom. Do NOT go back to 65536.
#
# Usage (from a Bristen login node -- the OTELA key is read from ~/.zshrc):
#   sbatch run_train_qwen3p5_bristen.sh
#   # 2-node smoke (~1 h) before committing 16 GPUs:
#   sbatch --nodes=2 --export=NNODES=2,MAX_RESPONSE_LENGTH=16384,CONCURRENCY=64 run_train_qwen35_bristen.sh
#
# Override any tunable with `sbatch --export=VAR=val,...`. See run_ray_node.sh
# for the per-node Ray lifecycle.

set -euo pipefail
[ "${1:-}" = "--" ] && shift
: "${SCRATCH:?SCRATCH is not set -- run on a Bristen login node}"

# All temp files (mktemp, Python tempfile, pip cache/build) go to scratch,
# NOT /users/xyao/.tmp -- home is quota-limited and job 80155 died on
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

# ---- wandb: pull API key from ~/.netrc (machine api.wandb.ai) ----
# The lean enroot .toml sets in-container HOME=${SCRATCH}/home (empty), so
# wandb cannot auto-read ~/.netrc there -- the key MUST be a real process env
# (wandb prefers WANDB_API_KEY over ~/.netrc). Optional WANDB_ENTITY overrides
# the default entity (your personal account). LOGGER is set here
# (console,wandb when a key is present, else console-only) and forwarded to
# the train script via RUN_ENV below; train_qwen3p5_dense.sh passes it to verl
# as trainer.logger=${LOGGER:-['console','wandb']}.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.netrc" ]; then
    _wkey="$(awk '/^machine[[:space:]]+api\.wandb\.ai[[:space:]]*$/{f=1; next} f && /^[[:space:]]*password[[:space:]]+/{print $2; exit}' "${HOME}/.netrc" | head -1)"
    [ -n "${_wkey}" ] && export WANDB_API_KEY="${_wkey}"
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGER="${LOGGER:-['console','wandb']}"
    export WANDB_ENTITY="${WANDB_ENTITY:-}"
    echo "[launch] wandb ENABLED (entity=${WANDB_ENTITY:-<default>} project=${PROJECT_NAME:-Uni-Agent-Qwen3.5-4B-megatron})"
else
    LOGGER="${LOGGER:-['console']}"
    echo "[launch] wandb DISABLED (no WANDB_API_KEY in env or ~/.netrc) -- console-only logging" >&2
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
HERE="${REPO_DIR}"
ENV_NAME="${ENV_NAME:-harbor-swe-megatron-bristen}"
# pyxis resolves --environment=NAME by searching EDF_PATH for NAME.toml.
export EDF_PATH="${HERE}:${EDF_PATH:-}"

CLUSTER_NNODES="${SLURM_NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-128}"

# ---- Per-run env for run_ray_node.sh + train_qwen3p5_dense.sh ----
RUN_ENV=(
    USER="${USER:-$(id -un)}"
    OTELA_API_KEY="${OTELA_API_KEY}"
    # Required on the rollout workers (not just the head): _resolve_image
    # in flash_sandbox.py reads this at actor-construction time to rewrite
    # the bare swebench/sweb.eval.x86_64.<id> ref to the GHCR mirror with
    # /testbed + git + the SWE harness. get_ppo_ray_runtime_env does NOT
    # forward it from the job-level runtime_env_flash.yaml, so it must be a
    # real process env here (inherited by the raylet -> actors, same path
    # that delivers OTELA_API_KEY above).
    FLASH_SANDBOX_IMAGE_REGISTRY="${FLASH_SANDBOX_IMAGE_REGISTRY:-ghcr.io/swe-images}"
    # Point the flash-sandbox client at the public OpenTela relay. The
    # internal mirror sandbox.swissai.cscs.ch became unreachable from compute
    # nodes mid-run on 08-30 (job 3209975: "Cannot connect to host
    # sandbox.swissai.cscs.ch:443" from ~07:33, every sandbox failed to start
    # after 3 attempts -> zero completed episodes).
    #
    # TRAILING SLASH IS REQUIRED. The no-slash URL returns HTTP 307 (redirect
    # to add the slash) for POSTs. The flash-sandbox SDK is aiohttp, whose
    # default allow_redirects=False for non-GET means it does NOT follow the
    # 307 -> empty response body -> SandboxHTTPError('HTTP 400: unexpected
    # end of JSON input' / "invalid character '}' after top-level value") on
    # EVERY start_sandbox (jobs 3234161/3234479/3234571: zero episodes; a
    # print() diagnostic at start_sandbox confirmed type='libkrun' was sent
    # so this was purely the 307-on-POST, not backend selection). The WITH-
    # slash URL returns directly: clean 200 when a provider is bound, or
    # clean retryable 503 "No provider found for the requested service" /
    # "no healthy nodes for backend X" during degraded windows (so the SDK's
    # 3-attempt backoff can recover during a healthy sub-window, instead of
    # a hard unparseable-400 failure every time). A light-load SDK probe at
    # 17:05 momentarily got 200 on no-slash (server served directly instead
    # of 307ing); that transient window must not be read as "no-slash is fine".
    FLASH_SANDBOX_ENDPOINT="${FLASH_SANDBOX_ENDPOINT:-https://api.opentela.ai/v1/service/flash-sandbox/}"
    # Force the libkrun microVM backend. The flash-sandbox client sends
    # start_sandbox(type=$FLASH_SANDBOX_TYPE); empty (the old default) lets
    # the cluster pick, which api.opentela.ai currently resolves to
    # "docker" -> HTTP 503 "no healthy nodes available for backend docker"
    # (job 3234161: every start_sandbox failed, zero completed episodes).
    # SDK probe 08-30 inside this image confirmed only libkrun has healthy
    # nodes on api.opentela.ai right now (docker + firecracker both 503).
    # Revert to empty / try 'docker' again once OpenTela's Docker backend
    # comes back (it cycled down ~08-30 like SwissAI did at 07:33).
    FLASH_SANDBOX_TYPE="${FLASH_SANDBOX_TYPE:-libkrun}"
    REPO_DIR="${REPO_DIR}"
    TRAIN_SCRIPT="examples/quickstart/training/train_qwen3p5_dense.sh"
    RAY_PORT="${RAY_PORT:-6379}"
    RAY_TMPDIR="/tmp/ray-${SLURM_JOB_ID:-$$}"
    # TMPDIR: NODE-LOCAL /tmp for ALL Python tempfile (Triton JIT, torch
    # inductor, vLLM compile). Was ${SCRATCH}/tmp (Lustre) -- the ORIGINAL
    # intent was "Python/pip want scratch for large tempfiles" (see old
    # comment below), but (a) pip cache is already PIP_CACHE_DIR=${SCRATCH}
    # /pip-cache (separate, NOT TMPDIR), and (b) training temp files are
    # tiny (KB-MB per kernel). With 4 vLLM servers x 2 workers = 8 concurrent
    # Triton JIT compiles, tempfile.TemporaryDirectory() (build.py:89) ->
    # open(src_path,"w") (build.py:91) EIO'd on ${SCRATCH}/tmp (Lustre),
    # killing job 82077 at 13:25 -- the FIRST run to reach model init. /tmp
    # is the same node-local SSD RAY_TMPDIR already uses; align them. Do NOT
    # merge RAY_TMPDIR into this (ray object store wants its own dir).
    TMPDIR="/tmp"
    PIP_CACHE_DIR="${PIP_CACHE_DIR}"
    NNODES="${CLUSTER_NNODES}"
    NNODES_T="${CLUSTER_NNODES}"
    NGPUS_PER_NODE="${GPUS_PER_NODE}"
    CPUS_PER_NODE="${CPUS_PER_NODE}"
    # Data (script defaults append an extra /data/uni_agent segment, so set
    # DATA_DIR to the parent and the parquets explicitly).
    DATA_DIR="${DATA_DIR:-/iopsstor/scratch/cscs/xyao/data}"
    MODEL_PATH="${MODEL_PATH:-/iopsstor/scratch/cscs/xyao/data/models/Qwen3.5-4B}"
    # TRAIN on swebench-verified for now: only the Verified subset is mirrored
    # on ghcr.io/swe-images (the registry FLASH_SANDBOX_IMAGE_REGISTRY rewrites
    # to). SWE-rebench rows carry a `swerebench/...` ref whose ghcr.io mirror
    # does NOT exist -> sandbox-create HTTP 400 (job 80162). Override with
    # TRAIN_FILE=.../swe_rebench_filtered_1150.parquet only once a rebench mirror
    # is up. (TEST stays verified; for the smoke train==test, just proves e2e.)
    TRAIN_FILE="${TRAIN_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_verified.parquet}"
    TEST_FILE="${TEST_FILE:-/iopsstor/scratch/cscs/xyao/data/uni_agent/swe_bench_verified.parquet}"
    RUNTIME_DIR="${RUNTIME_DIR:-/iopsstor/scratch/cscs/xyao}"
    RUNTIME_ENV="${RUNTIME_ENV:-examples/quickstart/training/runtime_env_flash.yaml}"
    # react (not claude_code): react is uni_agent builtin (pydantic + uni_agent.tools,
    # both in the Megatron image) and its stateful_shell uses the SDK persistent
    # shell_exec (the open_shell fix we verified end-to-end). claude_code shells out
    # to install the external `claude` CLI in-sandbox, which on the verified sweb
    # images fails in _ensure_claude (/tmp/claude-0 owned by the microVM uid vs the
    # installer-expected root -- job 80164). react exercises the full
    # rollout->reward->PPO loop without that external-CLI dependency.
    TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/training/task_config_react_flash.yaml}"
    # Topology: TP4 x PP2 x CP1 = 8/replica -> 4 DP on 32 GPUs. CP=1 forced by
    # GDN (upstream's CP=2 asserts in megatron-bridge -- see header).
    GEN_TP="${GEN_TP:-2}" TP="${TP:-4}" PP="${PP:-2}" CP="${CP:-1}"
    # Rollout / batch.
    CONCURRENCY="${CONCURRENCY:-256}"
    GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
    TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
    N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
    NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
    # A100-safe context (see header). 32768 -> max_model_len 40960 (2x 80205's
    # proven 24576). 65536 faulted on 80231 (cudaErrorIllegalAddress) -- do not
    # go back up until a step lands at 32768, then try 49152.
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
    # Per-step scheduler budget (train_qwen3p5_dense.sh max_num_batched_tokens).
    # Cap < max_model_len forces large react-transcript prefills to chunk,
    # avoiding the vLLM 0.22.1 GDN/M-RoPE cudaErrorIllegalAddress (80231/44/48).
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
    # Algorithm (upstream GRPO baseline -- matches origin/main exactly).
    ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
    LOSS_MODE="${LOSS_MODE:-vanilla}"
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
    CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
    CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
    EXP_NAME="${EXP_NAME:-qwen35_react_flash_$(date +%Y%m%d%H%M)}"
    # Disable periodic checkpointing. The verl default save_freq=10 triggered
    # the FIRST distributed ckpt at step 10 (82104, 17:17:40): Megatron
    # dist_ckpt wrote 8.3GB of model shards to the Lustre default_local_dir
    # across 8 ranks in parallel and EIO'd after 5min ([Errno 5] on WorkerDict
    # pid=47679, ip=172.28.8.19) -> worker died -> raylet lost the node ->
    # GCS cascade -> job FAILED at 17:24:09 after 3h14m (9 steps complete, no
    # earlier ckpt to resume). save_freq=1000 = no save during this run
    # (total_epochs=10 << 1000 steps). Same EIO class as fixes 1-5 (Lustre
    # writes under N-way concurrency), new operation. Override by
    # sbatch --export=SAVE_FREQ=...; flows to train_qwen3p5_dense.sh:89.
    SAVE_FREQ="${SAVE_FREQ:-1000}"
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
# tiny here-doc temp files in $TMPDIR. With TMPDIR=${SCRATCH}/tmp (Lustre),
# transient EIO there kills every enroot bootstrap (jobs 81650/81661/81676/
# 81726/81958 all died at runtime.sh:342/355 or 10-netstack.sh:63). Pin the
# HOST-side TMPDIR to node-local /tmp for every srun/enroot start below. The
# IN-CONTAINER TMPDIR is ALSO /tmp now (set in RUN_ENV above, was ${SCRATCH}
# /tmp until 82077's Triton-JIT EIO at build.py:91) -- both layers node-local,
# zero temp writes to Lustre during the whole run.
EDF_ERROR="$(TMPDIR=/tmp mktemp)"
cleanup() { rm -f "${EDF_ERROR}"; }
trap cleanup EXIT
if ! TMPDIR=/tmp srun --export=ALL,TMPDIR=/tmp --nodes=1 --ntasks=1 --environment="${ENV_NAME}" true 2>"${EDF_ERROR}"; then
    echo "ERROR: could not start the ${ENV_NAME} enroot env;" >&2
    echo "       build the image and check ${HERE}/harbor-swe-megatron-bristen.toml" >&2
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
    bash examples/quickstart/training/run_ray_node.sh
TRAIN_RC=$?

echo "[launch] done (rc=${TRAIN_RC})"
exit ${TRAIN_RC}
