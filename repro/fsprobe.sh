#!/bin/bash
# Probe Lustre vs node-local /tmp small-file write health RIGHT NOW,
# the exact path enroot runtime.sh:342 (here-doc) chokes on.
#SBATCH --job-name=fsprobe
#SBATCH --partition=normal
#SBATCH --account=infra02
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=fsprobe-%j.out
set -uo pipefail
SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/xyao}"
echo "[fsprobe] host=$(hostname) $(date -u +%H:%M:%SZ)"
echo "=== write small file to each target (the operation that failed) ==="
for tgt in "/tmp/fsprobe_$$" "${SCRATCH}/tmp/fsprobe_$$" "${SCRATCH}/home/fsprobe_$$" "${SCRATCH}/fsprobe_$$"; do
  d=$(dirname "$tgt"); mkdir -p "$d" 2>/dev/null
  # replicate the here-doc temp-file create that failed
  if out=$(cat <<HD > "$tgt" 2>&1
enroot-here-doc-probe-line-1
enroot-here-doc-probe-line-2
HD
  ); then
    sz=$(stat -c%s "$tgt" 2>/dev/null)
    echo "  OK   $tgt (${sz} bytes)"
    rm -f "$tgt"
  else
    echo "  FAIL $tgt -> $out"
    rm -f "$tgt" 2>/dev/null
  fi
done
echo "=== df (free space on each) ==="
df -h /tmp "${SCRATCH}" 2>/dev/null
echo "=== repeated small writes to scratch (10x) — EIO rate? ==="
fail=0
for i in $(seq 1 10); do
  f="${SCRATCH}/tmp/fsrepeat_${$}_${i}"
  if echo "x$i" > "$f" 2>/dev/null && [ -s "$f" ]; then rm -f "$f"; else fail=$((fail+1)); echo "  write $i FAILED"; fi
done
echo "=== scratch small-write failures: $fail / 10 ==="
echo DONE
