#!/usr/bin/env python3
"""Capture process memory and armed vLLM worker stacks without changing execution knobs."""
import json
import os
import signal
import socket
import time
from pathlib import Path

root = Path(os.environ['SWEEP_DIAGNOSTIC_DIR'])
root.mkdir(parents=True, exist_ok=True)
stop = False

def terminate(*_):
    global stop
    stop = True

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
last_dump = 0
with (root / 'process-samples.jsonl').open('a') as output:
    while not stop:
        now = time.time()
        rows = []
        for registration in root.glob('armed-*.json'):
            try:
                metadata = json.loads(registration.read_text())
                proc = Path('/proc') / str(metadata['pid'])
                if proc.stat().st_uid != os.getuid():
                    continue
                stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                if stat[19] != metadata['start_ticks']:
                    continue
                name = (proc / 'comm').read_text().strip()
                status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
                row = {'pid': metadata['pid'], 'name': name,
                       'rss_kib': int(status.get('VmRSS', '0').split()[0]),
                       'threads': int(status.get('Threads', '0').strip()),
                       'cpu_ticks': int(stat[11]) + int(stat[12])}
                target = name.startswith(('VLLM::Worker', 'EngineCore'))
                # Never signal Ray actors: Ray replaces SIGUSR1 handling and
                # its native stack-dump path segfaulted in the diagnostic run.
                caught = int(status.get('SigCgt', '0').strip(), 16) & (1 << (signal.SIGUSR1 - 1))
                if target and caught and now - last_dump >= 15:
                    os.kill(metadata['pid'], signal.SIGUSR1)
                    row['stack_signal_sent'] = True
                rows.append(row)
            except (OSError, ValueError, KeyError, ProcessLookupError):
                continue
        if now - last_dump >= 15:
            last_dump = now
        record = {'timestamp': now, 'node': socket.gethostname(), 'process_count': len(rows),
                  'processes': rows, 'cpu_stat': Path('/proc/stat').read_text().splitlines()[0],
                  'meminfo': Path('/proc/meminfo').read_text(),
                  'numa_meminfo': {p.name: (p / 'meminfo').read_text()
                                   for p in Path('/sys/devices/system/node').glob('node[0-9]*')}}
        output.write(json.dumps(record) + '\n')
        output.flush()
        time.sleep(5)
