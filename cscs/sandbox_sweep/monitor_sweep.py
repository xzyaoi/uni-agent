#!/usr/bin/env python3
"""Discover sweep job IDs as slots open and start their resource monitors."""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--remote-root', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--duration', type=int, default=86400)
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
sweep_name = Path(args.remote_root).name
children = {}
started = time.monotonic()
while time.monotonic() - started < args.duration:
    try:
        plan = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'clariden',
                               shlex.join(['cat', args.remote_root + '/sweep-plan.json'])],
                              capture_output=True, text=True, timeout=25)
        if plan.returncode == 0:
            temporary = root / 'sweep-plan.tmp'
            temporary.write_text(plan.stdout)
            temporary.replace(root / 'sweep-plan.json')
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'clariden',
                                 shlex.join(['cat', args.remote_root + '/jobs.tsv'])],
                                capture_output=True, text=True, timeout=25, check=True)
        (root / 'jobs.tsv').write_text(result.stdout)
        for line in result.stdout.splitlines():
            level, job, remote = line.split('\t')
            point = root / ('groups-' + level)
            point.mkdir(parents=True, exist_ok=True)
            if (point / 'scheduler-final.json').exists():
                continue
            child = children.get(job)
            if child is not None and child.poll() is None:
                continue
            command = [sys.executable, str(repo / 'cscs/observability/monitor_run.py'),
                       '--job-id', job, '--nodes', '2', '--rollout-only', '--run', remote,
                       '--host-metrics', 'telemetry/gpu', '--sandbox', 'ubuntu@3.229.113.27',
                       '--sandbox-identity', str(Path.home() / '.ssh/agentenv-xiayao.pem'),
                       '--sandbox-disk', '/nvme', '--cloudlab-dir',
                       '/home/ubuntu/observability/' + sweep_name + '/groups-' + level,
                       '--output', str(point), '--manage-sandbox-collector', '--interval', '300',
                       '--duration', str(args.duration)]
            with (point / 'monitor.log').open('a') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
            children[job] = child
            (point / 'monitor.pid').write_text(str(child.pid) + '\n')
            print('Monitoring groups=%s job=%s' % (level, job), flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print('Monitor discovery retry: ' + type(exc).__name__, flush=True)
    time.sleep(30)
