#!/usr/bin/env python3
"""Archive a monitored sweep before terminating its dedicated EC2 host."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def run(argv, timeout=120):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('%s failed: %s' % (argv[0], result.stderr[-1000:]))
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--host', default='ubuntu@3.229.113.27')
    parser.add_argument('--expected-ip', default='3.229.113.27')
    parser.add_argument('--terminate-instance', action='store_true', required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'teardown.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chmod(output, 0o700)
    state_file = output / 'teardown-state.json'
    if state_file.exists() and json.loads(state_file.read_text()).get('state') == 'terminated':
        return
    identity = str(Path.home() / '.ssh/agentenv-xiayao.pem')
    ssh = ['ssh', '-i', identity, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', args.host]
    cluster = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'clariden']
    aws = ['aws', 'ec2', '--region', args.region]
    sweep = Path(args.remote_root).name

    def record(state, **fields):
        data = dict(state=state, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    instance_id=args.instance_id, remote_root=args.remote_root, **fields)
        temporary = state_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(state_file)
        print(json.dumps(data), flush=True)

    while True:
        try:
            jobs = run(cluster + [shlex.join(['cat', args.remote_root + '/jobs.tsv'])])
            points = {int(line.split('\t')[0]): line.split('\t')[1]
                      for line in jobs.splitlines() if line.strip()}
            plan_file = output / 'sweep-plan.json'
            expected = set(json.loads(plan_file.read_text())['levels']) if plan_file.exists() else {8, 16, 32, 64}
            skipped = set()
            if 128 in expected and 128 not in points:
                probe = run(cluster + [shlex.join(['bash', '-c',
                    'if [ -f ' + shlex.quote(args.remote_root + '/groups-128/skipped.json') +
                    ' ]; then cat ' + shlex.quote(args.remote_root + '/groups-128/skipped.json') + '; else echo null; fi'])])
                skip = json.loads(probe)
                if skip and skip.get('state') == 'skipped' and skip.get('gated_by_groups') == 64:
                    skipped.add(128)
                    point = output / 'groups-128'
                    point.mkdir(exist_ok=True)
                    for name in ('skipped.json', 'result.json'):
                        (point / name).write_text(json.dumps(skip, indent=2) + '\n')
            if set(points) | skipped != expected:
                record('waiting_for_points', submitted=points)
                time.sleep(60)
                continue
            active = run(cluster + [shlex.join(['squeue', '-h', '-j', ','.join(points.values()), '-o', '%i|%T'])]).strip()
            missing = [level for level in points
                       if not (output / ('groups-%d' % level) / 'scheduler-final.json').exists()]
            if active or missing:
                record('monitoring', active_jobs=active, awaiting_final_snapshot=missing)
                time.sleep(60)
                continue
            # Final monitor files are only written after telemetry export succeeds.
            for level in points:
                point = output / ('groups-%d' % level)
                outcome = json.loads((point / 'result.json').read_text())
                if outcome.get('state') == 'running':
                    raise RuntimeError('Point %s has no terminal driver result' % level)
                if not (point / 'inputs/cloudlab.jsonl.gz').exists():
                    # Older monitor versions use a different input directory name.
                    if not list(point.rglob('cloudlab.jsonl.gz')):
                        raise RuntimeError('Point %s is missing sandbox telemetry' % level)

            description = json.loads(run(aws + ['describe-instances', '--instance-ids', args.instance_id]))
            instance = description['Reservations'][0]['Instances'][0]
            if instance['State']['Name'] == 'terminated':
                record('terminated')
                return
            if instance.get('PublicIpAddress') != args.expected_ip:
                raise RuntimeError('Dedicated instance IP no longer matches; refusing teardown')
            (output / 'instance-before-teardown.json').write_text(json.dumps(description, indent=2) + '\n')
            record('archiving', jobs=points)
            remote_archive = '/home/ubuntu/' + sweep + '-agentenv.tar.gz'
            # Preserve older experiment telemetry too: the instance's root disk
            # and NVMe storage are both lost on termination.
            prepare = """import pathlib,os,signal,subprocess,time
root=pathlib.Path('/home/ubuntu/observability')
for path in root.rglob('*.pid'):
    try:
        pid=int(path.read_text().strip())
        cmd=pathlib.Path('/proc/%s/cmdline' % pid).read_bytes().replace(b'\\0',b' ').decode()
        if str(path.parent) in cmd and ('collect.py' in cmd or 'collect_cpu.py' in cmd):
            os.kill(pid,signal.SIGTERM)
    except (OSError,ValueError):
        pass
subprocess.run(['sudo','-n','systemctl','stop','aenv-recorder'],check=True)
time.sleep(5)
"""
            run(ssh + ['python3 -c ' + shlex.quote(prepare)])
            tar_command = ['tar', '-czf', remote_archive, '--exclude=api-key', '--exclude=*.env',
                           '-C', '/', 'home/ubuntu/observability', 'nvme/aenv-recorder']
            run(ssh + [shlex.join(tar_command)], timeout=1800)
            expected_hash = run(ssh + [shlex.join(['sha256sum', remote_archive])], timeout=300).split()[0]
            local_archive = output / 'agentenv-archive.tar.gz'
            run(['rsync', '-a', '--partial', '-e', shlex.join(ssh[:-1]),
                 args.host + ':' + remote_archive, str(local_archive)], timeout=3600)
            digest = hashlib.sha256()
            with local_archive.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                os.fsync(stream.fileno())
            if digest.hexdigest() != expected_hash:
                raise RuntimeError('Archive checksum mismatch; retaining EC2 host')
            run(['tar', '-tzf', str(local_archive)], timeout=600)
            (output / 'agentenv-archive.sha256').write_text(expected_hash + '  agentenv-archive.tar.gz\n')
            record('archive_verified', archive=str(local_archive), sha256=expected_hash,
                   cluster_traces=args.remote_root)
            # Explicit user authorization is supplied by --terminate-instance.
            response = run(aws + ['terminate-instances', '--instance-ids', args.instance_id])
            (output / 'termination-response.json').write_text(response)
            record('terminating', archive=str(local_archive), sha256=expected_hash)
            while True:
                status = json.loads(run(aws + ['describe-instances', '--instance-ids', args.instance_id]))
                status = status['Reservations'][0]['Instances'][0]['State']['Name']
                if status == 'terminated':
                    record('terminated', archive=str(local_archive), sha256=expected_hash)
                    return
                time.sleep(30)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            record('retrying_instance_retained', error=str(exc))
            time.sleep(60)


if __name__ == '__main__':
    main()
