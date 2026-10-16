#!/bin/bash
# xhigh @ 2x context (81,920-token trajectory capacity) for Qwen3.8-27B.
#
# WHY THIS RUN: the reasoning-effort sequence (orchestrator_reasoning.sh) showed
# xhigh is capacity-starved, not low-quality. Token-level analysis of the 40960
# legs' step_1 trajectories (think-tag parsing of trajectory.npz):
#   xhigh : median 455-506 think tokens/turn, 78.5% of generated tokens are
#           thinking, 57.5% of episodes die MID-THINK (unterminated think block
#           when the capacity hits), finished 4.3%, reward 0.228
#   medium: median 210-253 think tokens/turn, 63.9% thinking, 21.2% mid-think
#           deaths, finished 49.8%, reward 0.561
# Episodes cap at prompt_length + response_length = 8192 + 32768 = 40960
# (gateway session._trajectory_capacity). This run doubles that to
# 8192 + 73728 = 81920 so xhigh's deliberation has room, and re-runs the xhigh
# leg under the exact same sequence config for a clean A/B vs the 40960 xhigh.
#
# MEMORY RISK (the one untested jump): actor_ppo_max_token_len =
# (8192+73728)/CP=1 = 81920 tok/micro-batch on the TRAIN nodes -- 2x the
# validated 40960. GH200 96GB, 27B TP4xPP2, offload+optimizer-offload on.
# Bristen A100s died at 73728 tok/micro-batch in the vLLM-0.22.1 era, but that
# was the rollout engines (cudaErrorIllegalAddress), not the train side, and
# Clariden runs vLLM 0.24.0 + separate_async (engines on their own 2 nodes at
# VLLM_GPU_MEM_UTIL=0.7, train nodes have the full 96GB). If the FIRST train
# batch OOMs on the train nodes, fall back to MAX_RESPONSE_LENGTH=49152
# (57344 total, the header-sanctioned scale-up step in run_train_qwen3p5_*.sh).
#
# CONTEXT SAFETY: Qwen3.8-27B native context 262144 (config text_config
# max_position_embeddings); task agent max_total_tokens 131072 and max_steps
# 200 do NOT bind at 81920. vLLM max_model_len and rollout.max_num_batched_
# tokens both scale to 81920 automatically (train script derives them from
# MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH). GDN layers have fixed-size
# per-sequence state; only 16/64 layers are full-attention, so KV at 82k x 8
# seqs/engine is a few GB -- well inside the 0.7-util pool.
#
# TIME: episodes that used to truncate at 40960 now run longer AND train
# micro-batches are 2x wide -> expect ~70 min/step vs ~50: budget 08:30:00
# walltime and a 480-min RUNNING-only hang deadline (was 07:00/360 at 40960).
#
# Single leg (no sequence): submit, watch for step 5, scancel. Retry up to
# MAX_LEG_RETRIES times on early-exit/hang. Modeled on orchestrator_reasoning.sh.
#
# Usage (clariden login node):
#   nohup bash orchestrator_xhigh_80k.sh > /dev/null 2>&1 &
#   # or attach a pre-submitted job:  nohup bash orchestrator_xhigh_80k.sh 3561234 &
REPO=/iopsstor/scratch/cscs/xyao/uni-agent
SCRATCH=/iopsstor/scratch/cscs/xyao
export SCRATCH
M38B=$SCRATCH/data/models/Qwen3.8-27B
LOG=$SCRATCH/logs/orchestrator-xhigh80k.log
PRE_SUBMITTED=${1:-}
MAX_LEG_RETRIES=2
LEG_DEADLINE_MIN=${LEG_DEADLINE_MIN:-480}
STEPS=${STEPS:-5}
# vLLM engine memory share on the rollout nodes. Attempt 1 (job 3567235, ctx
# 81920) ran 0.7 and died at the FIRST post-train weight sync: CheckpointEngine
# update_weights OOM'd allocating 2.37 GiB (engine process had ballooned to
# 86.1/95 GiB -- the 81920-token max_num_batched_tokens makes vLLM's
# activation high-water mark ~20 GiB larger than at 40960, blowing past the
# 0.7 budget; the 40960 legs survived 0.7 with half the spike). Fix combo:
# 0.65 leaves the CheckpointEngine + sync buffers ~8 GiB more room, and
# MAX_NUM_BATCHED_TOKENS=16384 forces chunked prefill (82k contexts split
# into ~6 scheduler steps), shrinking the activation spike itself. Train-side
# 81,920 forward/backward COMPLETED fine before the sync died -- only the
# rollout-side budget needed fixing.
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.65}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
# Sandbox provider (SDK-probed 2026-10-05 ~13:20 after the user restarted it):
# opentela + libkrun + a REAL image ref = HTTP 200 + clean start/stop
# (probe_libkrun_now.py with the real SDK -- a raw urllib probe 503'd, but the
# SDK-level test passes; also note a BOGUS image name fails at the OCI pull
# stage, which is NOT a provider outage). Wrapper defaults restored.
FLASH_SANDBOX_ENDPOINT="${FLASH_SANDBOX_ENDPOINT:-https://api.opentela.ai/v1/service/flash-sandbox/}"
FLASH_SANDBOX_TYPE="${FLASH_SANDBOX_TYPE:-libkrun}"
# The doubling: 8192 + 73728 = 81920 = 2 x 40960. Flows through the wrapper to
# train_qwen3p5_dense.sh -> rollout.prompt_length/response_length (gateway
# trajectory capacity), vLLM max_model_len, max_num_batched_tokens, and
# actor/ref ppo_max_token_len_per_gpu.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-73728}"
# Walltime: 40960 legs fit 5 steps + boot in 07:00; 81920 needs ~08:30.
TIME_LIMIT="${TIME_LIMIT:-08:30:00}"

# Self-contained env for sbatch --export=ALL (non-interactive shells do not
# source ~/.zshrc).
OTELA_API_KEY=$(grep -oE "OTELA_API_KEY=[^\" ]+" ~/.zshrc | head -1 | cut -d= -f2)
export OTELA_API_KEY
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}

log() { echo "$(date "+%Y-%m-%d %H:%M:%S") $*" >> "$LOG"; }

# Resolve the sbatch output file: poll both submit-dir candidates for up to
# 90 min (mirrors orchestrator_reasoning.sh).
wait_logfile() { # $1=jobid; echoes found path
  local i f
  for i in $(seq 1 90); do
    for f in "$REPO/uni-qwen35-clariden-$1.out" "$SCRATCH/uni-qwen35-clariden-$1.out"; do
      [ -f "$f" ] && { echo "$f"; return 0; }
    done
    sleep 60
  done
  return 1
}

wait_for_step() { # $1=jobid $2=logfile $3=step $4=label $5=deadline_min
  # Job-gone detection needs 3 consecutive empty squeue polls; the deadline
  # counts RUNNING time only (queued time is free).
  local miss=0 i=0 ST
  while true; do
    ST=$(squeue -h -j "$1" -o %T 2>/dev/null | head -1)
    if [ "$ST" = "RUNNING" ]; then
      i=$((i+1))
      if [ "$i" -gt "${5:-480}" ]; then
        log "$4: deadline (${5} min RUNNING) reached before step $3"
        return 2
      fi
    fi
    if [ -z "$ST" ]; then
      miss=$((miss + 1))
      if [ "$miss" -ge 3 ]; then
        log "$4: job $1 left queue before step $3"
        return 1
      fi
      sleep 60
      continue
    fi
    miss=0
    if [ "$ST" = "RUNNING" ] && grep -q "step:$3 - " "$2" 2>/dev/null; then
      log "$4: reached step $3"
      return 0
    fi
    sleep 60
  done
}

run_leg() { # $1=label $2=presubmitted_jobid (optional)
  local label="$1" J="${2:-}" L EXP TRY=0 rc
  while true; do
    if [ -z "$J" ]; then
      EXP="qwen38_re_${label}80k_$(date +%Y%m%d%H%M)"
      J=$(cd "$REPO" && sbatch -p normal --nodes=10 --time="$TIME_LIMIT" --job-name="uni-qwen38-re-$label80k" \
          --export=ALL,TRAINER_MODE=separate_async,NNODES_ROLLOUT=2,PARAMETER_SYNC_STEP=4,MODEL_PATH="$M38B",REASONING_EFFORT="$label",VLLM_GPU_MEM_UTIL="$VLLM_GPU_MEM_UTIL",MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS",FLASH_SANDBOX_ENDPOINT="$FLASH_SANDBOX_ENDPOINT",FLASH_SANDBOX_TYPE="$FLASH_SANDBOX_TYPE",MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH",MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH",EXP_NAME="$EXP" \
          run_train_qwen3p5_clariden.sh 2>&1 | tail -1)
      J=${J##* }
      case "$J" in ""|*[!0-9]*)
        log "$label: sbatch FAILED (output: $J)"; return 1 ;;
      esac
      log "$label: submitted job $J (effort=$label, ctx=${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH}=$((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})), attempt $((TRY+1)), EXP=$EXP)"

    else
      log "$label: watching pre-submitted job $J (effort=$label, ctx=$MAX_PROMPT_LENGTH+$MAX_RESPONSE_LENGTH)"
    fi
    L=$(wait_logfile "$J")
    if [ -z "$L" ]; then
      log "$label: WARN no output file after 90 min (pending?)"
      L="$REPO/uni-qwen35-clariden-$J.out"
    fi
    rc=0
    wait_for_step "$J" "$L" "$STEPS" "$label" "$LEG_DEADLINE_MIN" || rc=$?
    scancel "$J" 2>/dev/null
    if [ "$rc" -eq 0 ]; then
      log "$label: stopped after step $STEPS"
      return 0
    fi
    TRY=$((TRY+1))
    if [ "$TRY" -gt "$MAX_LEG_RETRIES" ]; then
      log "$label: exhausted retries; giving up (rc=$rc)"
      return 1
    fi
    log "$label: failed before step $STEPS (rc=$rc); retrying as attempt $((TRY+1))"
    J=""
    sleep 60
  done
}

log "=== xhigh-80k run started (pid $$): effort=xhigh, trajectory capacity $((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})) tokens, $STEPS steps ==="
run_leg xhigh "$PRE_SUBMITTED"
log "=== ALL DONE: Qwen3.8-27B xhigh @ $((${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH})) context ==="
