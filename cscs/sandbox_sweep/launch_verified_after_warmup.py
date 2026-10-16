#!/usr/bin/env python3
"""Gate the Verified sweep on image warmup, then launch its existing monitors."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--sweep-dir', required=True)
parser.add_argument('--dataset-root', required=True)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[2]
identity = str(Path.home() / '.ssh/agentenv-xiayao.pem')
aws_base = ['ssh', '-i', identity, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'ubuntu@3.229.113.27']
cluster_base = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'clariden']
server_root = '/home/ubuntu/observability/' + Path(args.sweep_dir).name
warm_path = '/home/ubuntu/observability/verified-warmup/full/status.json'

def command(argv, timeout=120):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(r.stderr[-1600:])
    return r.stdout

while True:
    try:
        raw = command(aws_base + [shlex.join(['cat', warm_path])])
        warm = json.loads(raw)
        (output / 'warmup-report.json').write_text(raw)
        (output / 'WARMUP.md').write_text('# Verified image warmup\n\nReady: **%s/%s**; failed: **%s**. GPU measurements wait for successful completion.\n' % (warm.get('ready_images', 0), warm['total_images'], warm.get('failed_images', 0)))
        if warm.get('finished_at'):
            if warm.get('failed_images'):
                raise SystemExit('Warmup failed; no GPU jobs submitted')
            break
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('Warmup poll retry: ' + type(exc).__name__, flush=True)
    time.sleep(30)

# Warmup evidence is public metadata, not the API-key file.
remote_report = args.dataset_root + '/warmup-report.json'
command(['scp', '-q', str(output / 'warmup-report.json'), 'clariden:' + remote_report])
settings = {'BENCHMARK': 'swebench-verified', 'VERIFIED_DATASET_ROOT': args.dataset_root,
            'VERIFIED_WARMUP_REPORT': remote_report, 'DATA_PATH': args.dataset_root + '/swebench_verified_harbor.parquet',
            'SWEEP_DIR': args.sweep_dir}
script = 'cd /iopsstor/scratch/cscs/xyao/uni-agent\n' + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in settings.items()) + '\nbash cscs/sandbox_sweep/submit.sh'
text = command(cluster_base + [shlex.join(['bash', '-c', script])], timeout=180)
(output / 'submission.log').write_text(text)
print(text, flush=True)
for name, argv in [
    ('sweep-monitor', [sys.executable, str(repo / 'cscs/sandbox_sweep/monitor_sweep.py'), '--remote-root', args.sweep_dir, '--output', str(output), '--duration', '86400']),
    ('summary-monitor', [sys.executable, str(repo / 'cscs/sandbox_sweep/summarize.py'), str(output), '--watch-seconds', '86400']),
]:
    with (output / (name + '.log')).open('a') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=os.environ.copy())
    (output / (name + '.pid')).write_text(str(child.pid) + '\n')
(output / 'pipeline-state.json').write_text(json.dumps({'state': 'submitted', 'sweep_dir': args.sweep_dir}, indent=2) + '\n')
