#!/bin/bash
# Wait for the reasoning-effort sequence's low leg to finish, then launch the
# xhigh-80k run (orchestrator_xhigh_80k.sh). Designed to run under nohup on a
# clariden login node:
#   nohup bash wait_and_run_xhigh80k.sh > /dev/null 2>&1 &
#
# Exit conditions for the wait (whichever comes first):
#   1. orchestrator-reasoning.log prints "ALL DONE" (low reached step 5, or
#      exhausted retries -- either way the sequence is over), or
#   2. the orchestrator_reasoning.sh process itself dies (crash/reboot path).
# Then: block until squeue shows NO uni-qwen38-re-* job left (the low job's
# 10 nodes are actually freed) and start the 80k leg.
LOG=/iopsstor/scratch/cscs/xyao/logs/orchestrator-xhigh80k.log
RLOG=/iopsstor/scratch/cscs/xyao/logs/orchestrator-reasoning.log
TS() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "$(TS) $*" >> "$LOG"; }

log "watcher: waiting for reasoning sequence (low leg) to finish"
while true; do
  if grep -q "ALL DONE: Qwen3.8-27B reasoning-effort sequence" "$RLOG" 2>/dev/null; then
    log "watcher: reasoning sequence ALL DONE"
    break
  fi
  if ! pgrep -f "bash orchestrator_reasoning.sh" > /dev/null 2>&1; then
    log "watcher: orchestrator_reasoning.sh process gone (crashed?); proceeding"
    break
  fi
  sleep 120
done

# Belt and suspenders: the scancel after step 5 can take a minute to drain
# the 10 nodes; do not contend for them.
while [ -n "$(squeue -h -u "$USER" -n uni-qwen38-re-low,uni-qwen38-re-medium,uni-qwen38-re-xhigh -o %i 2>/dev/null)" ]; do
  log "watcher: sequence job still draining; waiting"
  sleep 60
done

log "watcher: launching xhigh-80k leg"
exec bash /iopsstor/scratch/cscs/xyao/tmp/orchestrator_xhigh_80k.sh
