"""Render C=128 resource timelines; --refresh-data reads existing local exports."""

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from plot_cpu_host_scaling import REPO, ROOT, SOURCES, digest

DATA = ROOT / 'data/cpu_host_timeline_c128_20261019.json'
OUTPUT = ROOT / 'figure/cpu_host_timeline_c128_pro'


def refresh_data():
    runs = []
    for hosts, campaign in enumerate(SOURCES, 1):
        point = campaign / 'results/groups-16'
        result = json.loads((point / 'result.json').read_text())
        assert result['state'] == 'completed'
        start, end = result['model_ready_at'], result['finished_at']
        duration = end - start
        count = math.ceil(duration / 10)
        samples = [defaultdict(list) for _ in range(count)]
        begins, ends = {}, {}
        paths = [point / name for name in ('result.json', 'cluster_metrics_1min.csv', 'inputs/demand.jsonl.gz')]
        kinds = defaultdict(int)
        with gzip.open(paths[-1], 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                kind = event.get('kind')
                kinds[kind] += 1
                stamp = event.get('timestamp', 0)
                if kind == 'model':
                    key = (event['trace_file'], event['operation_id'])
                    if event['phase'] == 'begin':
                        begins[key] = stamp
                    elif event['phase'] in ('end', 'error'):
                        ends[key] = event
                if not start <= stamp < end:
                    continue
                bucket = int((stamp - start) // 10)
                if kind == 'dcgm':
                    fields = event.get('line', '').split()
                    if len(fields) >= 7 and fields[0] == 'GPU':
                        for key, value in [('gpu_sm_pct', fields[3]), ('gpu_tensor_pct', fields[5])]:
                            try:
                                value = float(value) * 100
                            except ValueError:
                                continue
                            if math.isfinite(value):
                                samples[bucket][key].append(value)
                elif kind == 'host' and event.get('role') == 'gpu':
                    for line in event.get('gpus', {}).get('stdout', '').splitlines():
                        fields = line.split(',')
                        if len(fields) >= 6:
                            try:
                                samples[bucket]['gpu_busy_pct'].append(float(fields[2]))
                            except ValueError:
                                pass
        # Exact interval overlap gives the mean client concurrency in each bin;
        # it is not a snapshot and is not the vLLM scheduler's running queue.
        assert begins.keys() == ends.keys(), 'Unpaired client requests require explicit censoring'
        request_seconds = [0.0] * count
        output_tokens = [0] * count
        changes = defaultdict(int)
        for key, begin in begins.items():
            event = ends[key]
            left, right = max(start, begin), min(end, event['timestamp'])
            assert event['timestamp'] >= begin
            if right > left:
                changes[left] += 1
                changes[right] -= 1
                for bucket in range(int((left - start) // 10), min(count, int((right - start) // 10) + 1)):
                    request_seconds[bucket] += max(0, min(right, start + (bucket + 1) * 10)
                                                  - max(left, start + bucket * 10))
            if start <= event['timestamp'] < end and event['phase'] == 'end':
                output_tokens[int((event['timestamp'] - start) // 10)] += (
                    event.get('usage') or {}).get('completion_tokens', 0)
        events = sorted(changes.items())
        index, active = 0, 0
        rows = []
        for bucket in range(count):
            left, right = start + bucket * 10, min(end, start + (bucket + 1) * 10)
            while index < len(events) and events[index][0] <= left:
                active += events[index][1]
                index += 1
            peak = active
            while index < len(events) and events[index][0] < right:
                active += events[index][1]
                peak = max(peak, active)
                index += 1
            rows.append({
                'elapsed_minutes': ((left + right) / 2 - start) / 60,
                'bin_seconds': right - left,
                'client_requests_mean': request_seconds[bucket] / (right - left),
                'client_requests_peak': peak,
                'delivered_output_tokens_s': output_tokens[bucket] / (right - left),
                **{key: statistics.mean(values) for key, values in samples[bucket].items()},
            })
        with paths[1].open() as stream:
            resources = [
                {'elapsed_minutes': (float(b['epoch']) + 30 - start) / 60,
                 'cpu_busy_pct': float(b['cluster_cpu_busy_pct']),
                 'sandbox_vm_count': float(b['cluster_vm_count'])}
                for b in csv.DictReader(stream)
                if start <= float(b['epoch']) and float(b['epoch']) + 60 <= end
                and b['full_node_coverage'] == '1'
            ]
        runs.append({
            'runtime_count': hosts, 'duration_minutes': duration / 60,
            'rows': rows, 'resources_1min': resources,
            'summary': {
                'client_requests_mean': sum(request_seconds) / duration,
                'client_requests_peak': max(r['client_requests_peak'] for r in rows),
                'model_calls': len(begins),
                'last_model_completion_minutes': (max(e['timestamp'] for e in ends.values()) - start) / 60,
                'vllm_scrapes': kinds['vllm'],
            },
            'source_hashes': {str(p.relative_to(REPO)): digest(p) for p in paths},
        })
    data = {
        'runs': runs,
        'scope': 'Completed Qwen3.5-9B Pro C=128 rollout-only observations; two-host source is preceding sweep.',
        'definitions': {
            'window': 'model_ready_at through finished_at; no training updates in these runs.',
            'requests': 'Client begin/end interval overlap: mean and peak per ten-second bin. '
                        'Includes gateway/network waits; does not measure engine-running or queued requests.',
            'gpu': 'Ten-second sample means across recorded GPUs; missing samples omitted, not zero-filled.',
            'tokens': 'Output tokens assigned to client completion time, divided by bin duration; '
                      'delivered output, not instantaneous engine generation.',
            'cpu_and_vms': 'Full-node-coverage one-minute means wholly inside the rollout window; '
                            'CPU normalized to the active fleet capacity.',
        },
    }
    DATA.write_text(json.dumps(data, indent=2) + '\n')


def plot():
    data = json.loads(DATA.read_text())
    sns.set_theme(style='ticks', context='paper', font_scale=1.25,
                  font='DejaVu Sans', rc={'font.family': ['Roboto', 'DejaVu Sans'],
                                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(5, 2, figsize=(12, 10), sharex=True, sharey='row')
    for col, run in enumerate(data['runs']):
        rows = run['rows']
        x = [r['elapsed_minutes'] for r in rows]
        for key, label, color in [('gpu_busy_pct', 'GPU busy', '#CFCECE'),
                                  ('gpu_sm_pct', 'SM active', '#0F4D92'),
                                  ('gpu_tensor_pct', 'Tensor active', '#9A4D8E')]:
            axes[0, col].plot(x, [r.get(key, float('nan')) for r in rows], label=label, color=color)
        for key, label, color in [('client_requests_peak', 'Peak / 10 s', '#AADCA9'),
                                  ('client_requests_mean', 'Mean / 10 s', '#0F4D92')]:
            axes[1, col].step(x, [r[key] for r in rows], where='mid', label=label, color=color)
        axes[2, col].plot(x, [r['delivered_output_tokens_s'] for r in rows], color='#3775BA')
        resources = run['resources_1min']
        t = [r['elapsed_minutes'] for r in resources]
        axes[3, col].plot(t, [r['cpu_busy_pct'] for r in resources], color='#8BCF8B')
        axes[4, col].plot(t, [r['sandbox_vm_count'] for r in resources], color='#9A4D8E')
        for row, label in enumerate(['GPU activity (%)', 'Client requests\nin flight',
                                     'Delivered output\n(tokens/s)', 'CPU utilization (%)', 'Sandbox VMs']):
            ax = axes[row, col]
            if col == 0:
                ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.set_xlim(0, max(r['duration_minutes'] for r in data['runs']))
            ax.grid(axis='y', alpha=.2)
            sns.despine(ax=ax)
        axes[0, col].set_ylim(0, 100)
        axes[3, col].set_ylim(0, 105)
        axes[0, col].set_title(f"{run['runtime_count']} CPU host{'s' if col else ''}", loc='left', fontsize=14)
        axes[0, col].legend(ncol=3, fontsize=9, frameon=False)
        axes[1, col].legend(ncol=2, fontsize=9, frameon=False)
        axes[4, col].set_xlabel('Minutes since model readiness')
    fig.suptitle('SWE-bench Pro V2 · Qwen3.5-9B · C=128 · eight rollout GPUs', fontsize=15)
    fig.text(.5, .015, 'Requests are measured at the model client; vLLM running/waiting queues were not captured.\n'
             'GPU/request/token bins: 10 s; CPU/VM bins: 1 min. Two-host observation is from the preceding sweep.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .96))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUTPUT.with_suffix('.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    with OUTPUT.with_suffix('.csv').open('w') as stream:
        fields = ['runtime_count', *dict.fromkeys(key for run in data['runs'] for row in run['rows'] for key in row)]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in data['runs']:
            writer.writerows({'runtime_count': run['runtime_count'], **row} for row in run['rows'])
    for run in data['runs']:
        print(run['runtime_count'], 'hosts:', run['summary'])
    print('Saved', OUTPUT.with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    plot()
