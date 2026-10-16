#!/usr/bin/env python3
"""Read-only CPU attribution from /proc tick deltas; no guest commands executed."""
import argparse
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path


def sample():
    cpu = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            raw = path.read_text()
            name = raw[raw.index('(') + 1:raw.rindex(')')]
            fields = raw[raw.rindex(')') + 2:].split()
            # Include start time in identity to avoid PID-reuse errors.
            identity = (path.parent.name, int(fields[19]))
            processes[identity] = (name, int(fields[11]) + int(fields[12]), int(fields[40]) if len(fields) > 40 else 0)
        except (OSError, ValueError, IndexError):
            continue
    return time.monotonic(), time.time(), cpu, processes


def category(name):
    if name == 'firecracker':
        return 'vm'
    if name == 'regctl':
        return 'registry'
    if name.startswith('overlaybd-'):
        return 'image_prepare'
    if name.startswith(('uvm-ublk', 'ublk', 'overlaybd-tcmu')):
        return 'vm_storage'
    if name == 'server':
        return 'agentenv'
    if name.startswith(('kworker', 'kswapd', 'kcompact', 'ksoftirq', 'rcu', 'migration')):
        return 'kernel_threads'
    return 'other_processes'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--duration', type=float, default=86400)
    args = parser.parse_args()
    if args.interval <= 0 or args.duration <= 0:
        parser.error('interval and duration must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    hz = os.sysconf('SC_CLK_TCK')
    cpus = os.cpu_count()
    node = socket.gethostname().split('.')[0]
    previous = sample()
    end = time.monotonic() + args.duration
    with (args.output / (node + '-cpu-attribution.jsonl')).open('a') as stream:
        while not stopping.wait(args.interval) and time.monotonic() < end:
            current = sample()
            elapsed = current[0] - previous[0]
            delta = [b - a for a, b in zip(previous[2], current[2])]
            total = sum(delta[:8])  # guest/guest_nice are already included in user/nice.
            if total <= 0 or min(delta) < 0:
                previous = current
                continue
            groups = {}
            for identity, (name, ticks, guest) in current[3].items():
                if identity not in previous[3]:
                    continue
                old = previous[3][identity]
                entry = groups.setdefault(category(name), {'cpu_cores': 0., 'guest_cores': 0., 'processes': 0})
                entry['cpu_cores'] += max(0, ticks - old[1]) / hz / elapsed
                entry['guest_cores'] += max(0, guest - old[2]) / hz / elapsed
                entry['processes'] += 1
            for entry in groups.values():
                entry['host_capacity_pct'] = entry['cpu_cores'] / cpus * 100
                entry['guest_capacity_pct'] = entry['guest_cores'] / cpus * 100
            busy = (total - delta[3] - delta[4]) / total * 100
            observed = sum(group['host_capacity_pct'] for group in groups.values())
            record = {'kind': 'cpu_attribution', 'timestamp': current[1], 'node': node,
                      'logical_cpus': cpus, 'elapsed_seconds': elapsed, 'groups': groups,
                      'host_busy_pct': busy, 'host_guest_pct': sum(delta[8:10]) / total * 100,
                      'unattributed_busy_pct': max(0., busy - observed)}
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            previous = current


if __name__ == '__main__':
    main()
