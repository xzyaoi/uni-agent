#!/usr/bin/env bash
# Bind ordinary host allocations without requiring numactl in the container.
set -euo pipefail
exec python3 - "$@" <<'PY'
import ctypes
import os
import platform
import sys
from pathlib import Path

allowed_line = next(line for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('Mems_allowed_list:'))
allowed = set()
for item in allowed_line.split(':', 1)[1].strip().split(','):
    bounds = item.split('-')
    allowed.update(range(int(bounds[0]), int(bounds[-1]) + 1))
nodes = sorted(int(p.name[4:]) for p in Path('/sys/devices/system/node').glob('node[0-9]*')
               if int(p.name[4:]) in allowed and (p / 'cpulist').read_text().strip())
if not nodes:
    raise SystemExit('No permitted CPU-attached NUMA memory nodes found')
# Linux syscall numbers from asm-generic/unistd.h and asm/unistd_64.h.
number = {'aarch64': 237, 'x86_64': 238}.get(platform.machine())
if number is None:
    raise SystemExit('Unsupported architecture for CPU memory binding')
word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
mask = (ctypes.c_ulong * (max(nodes) // word_bits + 1))()
for node in nodes:
    mask[node // word_bits] |= 1 << (node % word_bits)
libc = ctypes.CDLL(None, use_errno=True)
if libc.syscall(ctypes.c_long(number), ctypes.c_int(2), ctypes.byref(mask),
                ctypes.c_ulong(len(mask) * word_bits)) != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
print('[memory] Host allocations bound to CPU NUMA nodes: ' + ','.join(map(str, nodes)), flush=True)
for node in nodes:
    print(Path('/sys/devices/system/node/node%d/meminfo' % node).read_text(), flush=True)
os.environ['UNI_AGENT_CPU_MEMORY_BOUND'] = '1'
if len(sys.argv) < 2:
    raise SystemExit('Specify the node bootstrap command')
os.execvp(sys.argv[1], sys.argv[1:])
PY
