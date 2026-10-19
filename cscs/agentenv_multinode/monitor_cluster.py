#!/usr/bin/env python3
"""Monitor a static AgentENV fleet and a sequential rollout sweep.

No AWS API calls or instance teardown. Successful completion writes READY_TO_ARCHIVE.json
only after final snapshots and checksum-verified raw per-node archives are local.
Run with Python 3.10+. Credentials are never read or copied by this script.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.request


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def records(path):
    if not path.exists():
        return
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def node_agent(args):
    """Additional local control-plane metrics and journals, without API credentials."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    units = subprocess.run(['systemctl', 'list-units', '--all', '--type=service',
                            '--plain', '--no-legend'], capture_output=True, text=True,
                           timeout=15).stdout.splitlines()
    selected = [row.split()[0] for row in units if row.split()
                and any(word in row.split()[0].lower() for word in
                        ('agentenv', 'aenv', 'scheduler', 'gateway', 'recorder'))]
    journal = (output / 'controlplane-journal.jsonl').open('a')
    command = ['sudo', '-n', 'journalctl', '--since', '@' + str(int(args.since)),
               '--follow', '--no-pager', '-o', 'json']
    for unit in selected:
        command += ['-u', unit]
    child = subprocess.Popen(command, stdout=journal, stderr=journal) if selected else None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.duration
    try:
        with (output / 'controlplane-metrics.jsonl').open('a') as stream:
            while not stop and time.monotonic() < deadline:
                for endpoint in args.metric_endpoint or [
                        'http://127.0.0.1:8000/metrics', 'http://127.0.0.1:8001/metrics',
                        'http://127.0.0.1:8080/metrics', 'http://127.0.0.1:9090/metrics',
                        'http://127.0.0.1:9101/metrics', 'http://127.0.0.1:9102/metrics',
                        'http://127.0.0.1:9103/metrics']:
                    record = {'timestamp': time.time(), 'endpoint': endpoint}
                    try:
                        with opener.open(endpoint, timeout=2) as response:
                            record['prometheus'] = response.read(4 * 1024 * 1024).decode(errors='replace')
                            record['status'] = response.status
                    except Exception as exc:
                        record['error_class'] = type(exc).__name__
                    stream.write(json.dumps(record) + '\n')
                stream.flush()
                for _ in range(5):
                    if stop:
                        break
                    time.sleep(1)
    finally:
        if child:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        journal.close()


def aggregate(point, nodes):
    """Equal-node minute means, weighted CPU capacity; totals require full coverage."""
    bins = {}
    for node in nodes:
        previous = None
        for row in records(point / 'inputs' / node / 'host.jsonl.gz'):
            if row.get('kind') != 'host':
                continue
            stamp = row.get('timestamp')
            if not isinstance(stamp, (int, float)):
                continue
            bucket = bins.setdefault(int(stamp // 60) * 60, {}).setdefault(node, {})
            memory = {}
            for line in (row.get('meminfo') or '').splitlines():
                fields = line.split()
                if len(fields) > 1:
                    try:
                        memory[fields[0].rstrip(':')] = float(fields[1]) / 1048576
                    except ValueError:
                        pass
            values = {'ram_available_gib': memory.get('MemAvailable'),
                      'ram_total_gib': memory.get('MemTotal'), 'vm_count': row.get('vm_count')}
            cpu = (row.get('cpu_stat') or '').split()[1:9]
            if len(cpu) == 8:
                ticks = list(map(int, cpu))
                if previous:
                    delta = [new - old for new, old in zip(ticks, previous)]
                    total = sum(delta)
                    if total > 0 and min(delta) >= 0:
                        values['cpu_busy_pct'] = 100 * (total - delta[3] - delta[4]) / total
                previous = ticks
            for key, value in values.items():
                if value is not None:
                    bucket.setdefault(key, []).append(value)
    gpu_rows = {}
    gpu_csv = point / 'gpu' / 'metrics_1min.csv'
    if gpu_csv.exists():
        with gpu_csv.open() as stream:
            gpu_rows = {int(float(row['epoch'])): row for row in csv.DictReader(stream)}
    fields = ['timestamp_utc', 'epoch', 'nodes_observed', 'expected_nodes', 'full_node_coverage',
              'cluster_cpu_busy_pct', 'cluster_ram_available_gib', 'cluster_ram_total_gib',
              'cluster_vm_count', 'rollout_gpu_busy_pct', 'rollout_sm_active_pct',
              'rollout_tensor_active_pct', 'rollout_vllm_active_requests']
    result = []
    for epoch, observed in sorted(bins.items()):
        row = {'epoch': epoch, 'timestamp_utc': dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat(),
               'nodes_observed': len(observed), 'expected_nodes': len(nodes),
               'full_node_coverage': int(len(observed) == len(nodes))}
        for key in ('cpu_busy_pct', 'ram_available_gib', 'ram_total_gib', 'vm_count'):
            values = [statistics.mean(samples[key]) for samples in observed.values() if samples.get(key)]
            # All nodes have 96 vCPUs. Never impute missing nodes as zero.
            if len(values) == len(nodes):
                row['cluster_' + key] = statistics.mean(values) if key == 'cpu_busy_pct' else sum(values)
        for key in fields[9:]:
            row[key] = gpu_rows.get(epoch, {}).get(key, '')
        result.append(row)
    with (point / 'cluster_metrics_1min.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    atomic(point / 'cluster_metric_definitions.json', {
        'cpu': 'Mean per-node minute busy percentage, weighted by equal 96-vCPU capacity',
        'ram_and_vms': 'Sum of per-node minute means, only with all nodes reporting metric',
        'coverage': 'A node has at least one host sample in the minute; edge bins may be partial',
        'gpu': 'Existing plot_resources sample means across eight rollout GPUs; missing GPUs are not zero-filled',
        'expected_nodes': nodes, 'host_vcpus_each': 96,
        'gpu_coverage_limitation': 'GPU source does not require all eight GPUs to report each bin'})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        x = [dt.datetime.fromtimestamp(row['epoch'], dt.timezone.utc) for row in result]
        for key, label in [('cluster_cpu_busy_pct', f'Sandbox CPU ({96 * len(nodes)} vCPUs)'),
                           ('rollout_gpu_busy_pct', 'Rollout GPU busy'),
                           ('rollout_sm_active_pct', 'Rollout SM active')]:
            axes[0].plot(x, [float(row[key]) if row.get(key) not in (None, '') else float('nan')
                             for row in result], label=label)
        axes[0].set_ylabel('Activity (%)')
        axes[0].legend()
        for ax, key, label in zip(axes[1:], ['cluster_vm_count', 'cluster_ram_available_gib'],
                                  ['Cluster VMs (sum)', 'Available RAM (GiB, sum)']):
            ax.plot(x, [float(row[key]) if row.get(key) not in (None, '') else float('nan')
                        for row in result])
            ax.set_ylabel(label)
        axes[-1].set_xlabel('UTC; gaps indicate incomplete node coverage')
        fig.tight_layout()
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(point / ('cluster_resources.' + suffix), dpi=180)
        plt.close(fig)
    except ImportError:
        pass


class Monitor:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        os.umask(0o077)
        manifest = json.loads(args.cluster_manifest.read_text())
        runtime_count = manifest.get('runtime_count', len(manifest['instances']))
        if isinstance(runtime_count, bool) or not isinstance(runtime_count, int) or runtime_count <= 0:
            raise ValueError('runtime_count must be a positive integer')
        self.nodes = []
        for index, item in enumerate(manifest['instances']):
            self.nodes.append({'id': str(item.get('node_id', 'node-' + str(index + 1))),
                               'host': item.get('public_dns') or item.get('PublicDnsName') or item.get('public_ip') or item.get('PublicIpAddress'),
                               'instance_id': item.get('instance_id', item.get('InstanceId'))})
        if len(self.nodes) != runtime_count or any(not node['host'] for node in self.nodes):
            raise ValueError(f'{runtime_count} reachable nodes are required')
        if any('/' in node['id'] or node['id'] in ('.', '..') for node in self.nodes):
            raise ValueError('node_id must be a path component')
        if len({node['id'] for node in self.nodes}) != runtime_count:
            raise ValueError('node_id values must be unique')
        atomic(self.root / 'cluster_inventory.json', {'nodes': self.nodes,
               'runtime_count': runtime_count,
               'gateway_endpoint': manifest.get('gateway_endpoint')})
        self.remote = '/home/ubuntu/observability/' + Path(args.remote_sweep_root).name
        self.tools = Path(__file__).resolve().parents[1] / 'observability'
        origin = self.root / 'monitor-origin.json'
        self.started = json.loads(origin.read_text())['timestamp'] if origin.exists() else time.time()
        atomic(origin, {'timestamp': self.started})
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def command(self, argv, timeout=90, data=None):
        result = subprocess.run(argv, input=data, text=True, capture_output=True, timeout=timeout)
        if result.returncode:
            # Avoid copying remote output, logs, credentials or command lines into errors.
            raise RuntimeError('%s exit %s' % (argv[0], result.returncode))
        return result.stdout

    def ssh(self, host, command, timeout=90, data=None, sandbox=False):
        options = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                   '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
        if sandbox:
            options += ['-i', str(self.args.identity.expanduser()), '-o', 'IdentitiesOnly=yes']
        elif self.args.ssh_hostname:
            options += ['-o', 'Hostname=' + self.args.ssh_hostname]
        return self.command(options + [host, command], timeout, data)

    def python(self, host, source, timeout=90, sandbox=False):
        return self.ssh(host, 'python3 -c ' + shlex.quote(source), timeout, sandbox=sandbox)

    def copy(self, host, remote, local, sandbox=False):
        local.parent.mkdir(parents=True, exist_ok=True)
        options = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
        if sandbox:
            options += ['-i', str(self.args.identity.expanduser()), '-o', 'IdentitiesOnly=yes']
        if not sandbox and self.args.ssh_hostname:
            options += ['-o', 'Hostname=' + self.args.ssh_hostname]
        temporary = local.with_suffix(local.suffix + '.tmp')
        self.command(options + [host + ':' + remote, str(temporary)], timeout=1800)
        temporary.replace(local)

    def start_node(self, node):
        host = 'ubuntu@' + node['host']
        remote = self.remote + '/' + node['id']
        self.python(host, 'from pathlib import Path; p=Path(%r); p.mkdir(parents=True,exist_ok=True); p.chmod(0o700)' % remote, sandbox=True)
        for path in [self.tools / 'collect.py', self.tools / 'collect_cpu.py',
                     self.tools / 'export_metrics.py', Path(__file__).resolve()]:
            self.ssh(host, 'cat > ' + shlex.quote(remote + '/' + path.name),
                     data=path.read_text(), sandbox=True)
        commands = {'collector': ['python3', remote + '/collect.py', '--role', 'sandbox', '--output', remote,
                                   '--disk', '/nvme', '--duration', str(self.args.duration), '--sudo-journal'],
                    'cpu-collector': ['python3', remote + '/collect_cpu.py', '--output', remote,
                                      '--duration', str(self.args.duration)],
                    'controlplane': ['python3', remote + '/monitor_cluster.py', '--node-agent', '--output', remote,
                                     '--duration', str(self.args.duration), '--since', str(self.started)]}
        code = 'import os,json,subprocess\nfrom pathlib import Path\nos.umask(0o077)\n'
        code += 'root=Path(%r)\ncommands=%r\n' % (remote, commands)
        code += '''for name,argv in commands.items():
 p=root/(name+'.pid')
 alive=False
 if p.exists():
  try:
   pid=int(p.read_text()); os.kill(pid,0)
   alive=argv[1].encode() in Path('/proc/%d/cmdline'%pid).read_bytes()
  except (OSError,ValueError): pass
 if not alive:
  with (root/(name+'.log')).open('a') as out:
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=out,start_new_session=True)
  p.write_text(str(child.pid))
'''
        self.python(host, code, sandbox=True)

    def snapshot(self, level, job, remote):
        point = self.root / ('groups-' + level)
        point.mkdir(parents=True, exist_ok=True)
        inputs = point / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        # Filter at the source; never transfer arbitrary launch environment values.
        code = 'import json,shlex\nfrom pathlib import Path\nr={}\n'
        code += 'for line in Path(%r).read_text().splitlines():\n' % (remote + '/launch.env')
        code += ' key,_,value=(line[7:] if line.startswith("export ") else line).partition("=")\n'
        code += ' if key in ("AGENTENV_TRACE_DIR","BENCHMARK","GROUP_CONCURRENCY"):\n'
        code += '  fields=shlex.split(value)\n  if fields: r[key]=fields[0]\n'
        code += 'print(json.dumps(r))\n'
        config = json.loads(self.python(self.args.cluster, code))
        atomic(inputs / 'public-run-config.json', config)
        export = [ 'python3', remote + '/analysis-tools/export_metrics.py',
                  '--telemetry', remote + '/telemetry/gpu', '--models', remote + '/telemetry/model-calls',
                  '--native', config.get('AGENTENV_TRACE_DIR', remote + '/native-traces'),
                  '--output', remote + '/analysis-tools/multinode-demand.jsonl.gz']
        self.ssh(self.args.cluster, shlex.join(export), timeout=600)
        self.copy(self.args.cluster, remote + '/analysis-tools/multinode-demand.jsonl.gz', inputs / 'demand.jsonl.gz')
        for name in ('progress.json', 'result.json', 'settings.json', 'manifest.json'):
            try:
                self.copy(self.args.cluster, remote + '/' + name, point / name)
            except RuntimeError:
                pass
        result = json.loads((point / 'result.json').read_text()) if (point / 'result.json').exists() else {}
        start = result.get('model_ready_at', 0)
        end = result.get('finished_at', time.time())
        for node in self.nodes:
            host = 'ubuntu@' + node['host']
            directory = self.remote + '/' + node['id']
            self.ssh(host, shlex.join(['python3', directory + '/export_metrics.py', '--telemetry', directory,
                                     '--output', directory + '/snapshot.jsonl.gz']), sandbox=True)
            path = inputs / node['id'] / 'host.jsonl.gz'
            self.copy(host, directory + '/snapshot.jsonl.gz', path, sandbox=True)
            temporary = path.with_suffix('.tmp')
            with gzip.open(temporary, 'wt') as out:
                for row in records(path):
                    if start <= row.get('timestamp', 0) <= end:
                        out.write(json.dumps(row) + '\n')
            temporary.replace(path)
        # GPU-only figures never merge sandbox hosts using the old mean semantics.
        empty = inputs / 'empty-sandbox.jsonl.gz'
        with gzip.open(empty, 'wt'):
            pass
        plot = [sys.executable, str(self.tools / 'plot_resources.py'), str(inputs / 'demand.jsonl.gz'),
                str(empty), '--output', str(point / 'gpu'), '--trainer-nodes', '',
                '--title', 'Job ' + job + ': eight rollout GPUs']
        if start:
            plot += ['--start', dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat()]
        try:
            self.command(plot, timeout=180)
        except RuntimeError:
            pass  # DCGM and model servers may not exist during bootstrap.
        for node in self.nodes:
            per_node_plot = [sys.executable, str(self.tools / 'plot_resources.py'),
                             str(inputs / 'demand.jsonl.gz'), str(inputs / node['id'] / 'host.jsonl.gz'),
                             '--output', str(point / 'nodes' / node['id']), '--trainer-nodes', '',
                             '--title', 'Sandbox node ' + node['id'] + ' / eight rollout GPUs']
            if start:
                per_node_plot += ['--start', dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat()]
            try:
                self.command(per_node_plot, timeout=180)
            except RuntimeError:
                pass
        aggregate(point, [node['id'] for node in self.nodes])

    def archive_node(self, node):
        host = 'ubuntu@' + node['host']
        remote = self.remote + '/' + node['id']
        # Capture remaining journal entries from the experiment start. No configuration/secrets directories.
        code = '''import pathlib,subprocess,tarfile,hashlib,json,os,time,signal
os.umask(0o077)
p=pathlib.Path(%r)
for file in p.glob('*.pid'):
 try:
  pid=int(file.read_text())
  if str(p).encode() in pathlib.Path('/proc/%%d/cmdline'%%pid).read_bytes():
   os.kill(pid,signal.SIGTERM)
 except (OSError,ValueError): pass
time.sleep(8)
subprocess.run(['sudo','-n','systemctl','stop','aenv-recorder'],check=True,timeout=60)
with (p/'final-controlplane-journal.jsonl').open('w') as out:
 subprocess.run(['sudo','-n','journalctl','--since',%r,'--no-pager','-o','json','-u','aenv','-u','agentenv','-u','agentenv-gateway','-u','agentenv-scheduler','-u','aenv-recorder'],stdout=out,stderr=subprocess.DEVNULL,check=True,timeout=120)
archive=p.parent/(p.name+'-raw.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
 tar.add(str(p),arcname=p.name)
 recorder=pathlib.Path('/nvme/aenv-recorder')
 if recorder.exists():
  for file in recorder.glob('*.jsonl*'):
   if file.stat().st_mtime >= %r: tar.add(str(file),arcname='recorder/'+file.name)
digest=hashlib.sha256()
with archive.open('rb') as stream:
 for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
(archive.with_suffix('.sha256')).write_text(digest.hexdigest()+chr(10))
''' % (remote, '@' + str(int(self.started)), self.started)
        self.python(host, code, timeout=3600, sandbox=True)
        archive = self.root / 'archives' / (node['id'] + '-raw.tar.gz')
        self.copy(host, self.remote + '/' + node['id'] + '-raw.tar.gz', archive, sandbox=True)
        self.copy(host, self.remote + '/' + node['id'] + '-raw.tar.sha256', archive.with_suffix('.sha256'), sandbox=True)
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        expected = archive.with_suffix('.sha256').read_text().strip()
        if digest.hexdigest() != expected:
            raise RuntimeError('Archive checksum mismatch for ' + node['id'])
        return {'node_id': node['id'], 'archive': str(archive), 'sha256': expected}

    def run(self):
        initialized = set()
        finalized = set()
        deadline = time.monotonic() + self.args.duration
        while not self.stop and time.monotonic() < deadline:
            try:
                for node in self.nodes:
                    if node['id'] not in initialized:
                        self.start_node(node)
                        initialized.add(node['id'])
                plan = json.loads(self.ssh(self.args.cluster, 'cat ' + shlex.quote(self.args.remote_sweep_root + '/sweep-plan.json')))
                atomic(self.root / 'sweep-plan.json', plan)
                text = self.ssh(self.args.cluster, 'cat ' + shlex.quote(self.args.remote_sweep_root + '/jobs.tsv'))
                (self.root / 'jobs.tsv').write_text(text)
                jobs = [line.split('\t') for line in text.splitlines() if line.strip()]
                for level, job, remote in jobs:
                    if level in finalized:
                        continue
                    state = self.ssh(self.args.cluster, shlex.join(['squeue', '-h', '-j', job, '-o', '%T'])).strip()
                    self.snapshot(level, job, remote)
                    if not state:
                        accounting = self.ssh(self.args.cluster, shlex.join(['sacct', '-X', '-n', '-P', '-j', job, '--format=State'])).strip()
                        if accounting:
                            atomic(self.root / ('groups-' + level) / 'scheduler-final.json', {'job_id': job, 'state': accounting})
                            finalized.add(level)
                # Include explicitly skipped optional points, but never infer missing points finished.
                for level in plan['levels']:
                    if str(level) not in finalized:
                        try:
                            skipped = json.loads(self.ssh(self.args.cluster, 'cat ' + shlex.quote(self.args.remote_sweep_root + '/groups-%s/skipped.json' % level)))
                            if skipped.get('state') == 'skipped':
                                point = self.root / ('groups-' + str(level))
                                point.mkdir(exist_ok=True)
                                atomic(point / 'result.json', skipped)
                                finalized.add(str(level))
                        except RuntimeError:
                            pass
                try:
                    self.command([sys.executable, str(self.tools.parent / 'sandbox_sweep/summarize.py'), str(self.root)], timeout=180)
                except RuntimeError:
                    pass
                atomic(self.root / 'monitor-status.json', {'timestamp': time.time(), 'finalized_levels': sorted(finalized),
                                                         'expected_levels': plan['levels'], 'state': 'monitoring'})
                if set(map(str, plan['levels'])) <= finalized:
                    archives = [self.archive_node(node) for node in self.nodes]
                    atomic(self.root / 'READY_TO_ARCHIVE.json', {'state': 'all_points_terminal_and_host_archives_verified',
                           'timestamp': time.time(), 'archives': archives,
                           'cscs_raw_retained_at': self.args.remote_sweep_root,
                           'note': 'CSCS native/model replay traces remain on persistent cluster storage; no instance termination performed'})
                    return
            except Exception as exc:
                # Exception text from remote logs is intentionally omitted.
                atomic(self.root / 'monitor-error.json', {'timestamp': time.time(), 'error_class': type(exc).__name__})
                print('Monitor retry: ' + type(exc).__name__, flush=True)
            for _ in range(self.args.interval):
                if self.stop:
                    break
                time.sleep(1)
        atomic(self.root / 'monitor-status.json', {'state': 'stopped_before_archive_completion', 'timestamp': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster-manifest', type=Path)
    parser.add_argument('--remote-sweep-root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cluster', default='clariden')
    parser.add_argument('--ssh-hostname', default='clariden-ln001.cscs.ch',
                        help='Explicit SSH Hostname override for cluster alias (empty disables override)')
    parser.add_argument('--identity', type=Path, default=Path('~/.ssh/agentenv-xiayao.pem'))
    parser.add_argument('--duration', type=int, default=86400)
    parser.add_argument('--interval', type=int, default=180)
    parser.add_argument('--node-agent', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--since', type=float, default=time.time(), help=argparse.SUPPRESS)
    parser.add_argument('--metric-endpoint', action='append', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error('duration and interval must be positive')
    if args.node_agent:
        node_agent(args)
    else:
        if not args.cluster_manifest or not args.remote_sweep_root:
            parser.error('--cluster-manifest and --remote-sweep-root required')
        Monitor(args).run()


if __name__ == '__main__':
    main()
