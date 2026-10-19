"""Plot the completed Pro C=128 host comparison in campaign_throughput style.

Render frozen data: python meta/plot/plot_cpu_host_scaling.py
Refresh from local exports: python meta/plot/plot_cpu_host_scaling.py --refresh-data
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = ROOT / 'data/cpu_host_scaling_c128_20261019.json'
OUTPUT = ROOT / 'figure/cpu_host_scaling_c128_pro'
SOURCES = [
    REPO / 'outputs/pro_overnight_20261019_recovery2/points/pro_overnight_20261019_p01s_n1_c128',
    REPO / 'outputs/pro_multinode_2x_20261018',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_data():
    rows, sources, manifests, settings, measurements = [], [], [], [], []
    for hosts, campaign in enumerate(SOURCES, 1):
        point = campaign / 'results/groups-16'
        result = json.loads((point / 'result.json').read_text())
        manifest = json.loads((point / 'manifest.json').read_text())
        config = json.loads((point / 'settings.json').read_text())
        measurement = json.loads((campaign / 'preparation/measurement-manifest.json').read_text())
        definitions = json.loads((point / 'cluster_metric_definitions.json').read_text())
        assert result['state'] == 'completed' and result['max_concurrent_episodes'] == 128
        assert result['progress']['groups_terminal'] == 128 and result['progress']['groups_failed'] == 0
        assert len(definitions['expected_nodes']) == hosts and definitions['host_vcpus_each'] == 96
        start, end = result['model_ready_at'], result['finished_at']
        with (point / 'cluster_metrics_1min.csv').open() as stream:
            bins = [b for b in csv.DictReader(stream)
                    if start <= float(b['epoch']) and float(b['epoch']) + 60 <= end
                    and b['full_node_coverage'] == '1']
        cpu = [float(b['cluster_cpu_busy_pct']) for b in bins]
        # Rebuild DCGM minutes from raw exports: the two-host summary CSV has
        # blank GPU columns even though the raw samples are present.
        gpu = defaultdict(list)
        gpu_devices = defaultdict(set)
        demand = point / 'inputs/demand.jsonl.gz'
        with gzip.open(demand, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('kind') != 'dcgm':
                    continue
                stamp = event['timestamp']
                bucket = int(stamp // 60) * 60
                fields = event.get('line', '').split()
                if start <= bucket and bucket + 60 <= end and len(fields) >= 7 and fields[0] == 'GPU':
                    try:
                        value = float(fields[3]) * 100
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        gpu[bucket].append(value)
                        gpu_devices[bucket].add((event['node'], fields[1]))
        assert cpu and gpu
        # Align independent host sampling on five-second bins. Require every
        # active host, then take the maximum fleet sum of per-host bin means.
        memory = defaultdict(lambda: defaultdict(list))
        host_paths = [point / f'inputs/runtime-{n}/host.jsonl.gz' for n in range(1, hosts + 1)]
        for node, path in enumerate(host_paths, 1):
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get('kind') != 'host' or not start <= event['timestamp'] <= end:
                        continue
                    cgroup = event['cgroup']
                    assert cgroup['path'] == '/system.slice/aenv.service'
                    stat = dict(line.split() for line in cgroup['files']['memory.stat'].splitlines())
                    memory[int(event['timestamp'] // 5) * 5][node].append(int(stat['anon']) / 2**30)
        fleet_memory = [sum(statistics.mean(samples) for samples in nodes.values())
                        for nodes in memory.values() if len(nodes) == hosts]
        assert fleet_memory
        harbor = result['harbor']
        seconds = result['rollout_wall_seconds']
        row = {
            'runtime_count': hosts, 'vcpus': 96 * hosts, 'episode_ceiling': 128,
            'state': result['state'], 'wall_seconds': seconds,
            'completed_episodes': harbor['terminal_results'],
            'verified_episodes': harbor['evaluation_completed'],
            'incomplete_verifications': harbor.get('evaluation_incomplete', 0),
            'usable_episodes': result['progress']['episodes_with_usable_trajectory'],
            'sandbox_cpu_busy_pct_mean': statistics.mean(cpu),
            'sandbox_cpu_busy_pct_peak_1min': max(cpu), 'cpu_minutes': len(cpu),
            'rollout_sm_active_pct_mean': statistics.mean(statistics.mean(v) for v in gpu.values()),
            'gpu_minutes': len(gpu),
            'gpu_minutes_with_all_eight_devices': sum(len(v) == 8 for v in gpu_devices.values()),
            'peak_anonymous_memory_gib': max(fleet_memory),
            'memory_full_coverage_5s_bins': len(fleet_memory),
        }
        for count in ('completed', 'verified'):
            row[f'{count}_episodes_per_minute'] = row[f'{count}_episodes'] * 60 / seconds
        rows.append(row)
        job = next(line.split()[1] for line in (campaign / 'results/jobs.tsv').read_text().splitlines()
                   if line.split()[0] == '16')
        paths = [point / name for name in ('result.json', 'manifest.json', 'settings.json',
                                         'cluster_metrics_1min.csv', 'cluster_metric_definitions.json')]
        paths += [campaign / 'preparation/measurement-manifest.json', demand, *host_paths]
        sources.append({
            'runtime_count': hosts, 'campaign': str(campaign.relative_to(REPO)), 'job_id': job,
            'model_ready_at': start, 'finished_at': end,
            'files': {str(p.relative_to(REPO)): digest(p) for p in paths},
        })
        manifests.append(manifest)
        settings.append({k: v for k, v in config.items() if k not in ('run_dir', 'log_dir', 'task_config')})
        measurements.append(measurement)
    assert settings[0] == settings[1] and measurements[0] == measurements[1]
    for key in ('selection_sha256', 'model_config_sha256', 'selected_tasks'):
        assert manifests[0][key] == manifests[1][key]
    data = {
        'benchmark': 'SWE-bench Pro V2', 'model': 'Qwen3.5-9B', 'rollout_gpus': 8,
        'groups': 128, 'responses': 8, 'rows': rows,
    }
    DATA.write_text(json.dumps(data, indent=2) + '\n')
    provenance = {
        'sha256': digest(DATA), 'sources': sources,
        'comparison': 'One completed observation per host count; two-host point is from the preceding sweep. '
                      'The planned overnight two-host repetition has no completed result in local exports.',
        'matching': 'Identical selected task order, model config hash, pinned measurement manifest, '
                    'and settings except output/task-config paths. Task-config hashes differ between runs.',
        'throughput': 'Terminal attempts or completed reward verifications / rollout_wall_seconds * 60; '
                      'model initialization excluded, native sandbox setup included. '
                      'Verification includes reward zero.',
        'cpu': 'Mean full-node-coverage one-minute CPU busy bins wholly within the rollout window; '
               'normalized to 96 or 192 total vCPUs.',
        'gpu': 'Mean of DCGM SM-active sample means per complete minute within the rollout window, '
               'rebuilt from raw demand exports for both runs. Missing samples are not zero-filled.',
        'memory': 'Peak sum across active hosts of aenv.service memory.stat anon, using per-host means '
                  'in five-second bins with all hosts present; rollout-window samples only; file cache excluded.',
        'selection_sha256': manifests[0]['selection_sha256'],
        'model_config_sha256': manifests[0]['model_config_sha256'],
    }
    DATA.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def plot():
    provenance = json.loads(DATA.with_suffix('.provenance.json').read_text())
    assert digest(DATA) == provenance['sha256']
    data = json.loads(DATA.read_text())
    rows = data['rows']
    assert [r['runtime_count'] for r in rows] == [1, 2]
    for row in rows:
        assert row['state'] == 'completed' and row['episode_ceiling'] == 128
        assert row['completed_episodes'] == 1024
        assert row['verified_episodes'] + row['incomplete_verifications'] == 1024
        for count in ('completed', 'verified'):
            assert math.isclose(row[f'{count}_episodes_per_minute'],
                                row[f'{count}_episodes'] * 60 / row['wall_seconds'])
    sns.set_style('ticks')
    sns.set_style({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 12})
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 3, 'lines.markersize': 10})
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (throughput, activity) = plt.subplots(1, 2, figsize=(9, 3.75))
    fig.subplots_adjust(left=.15, right=.83, bottom=.28, top=.72, wspace=.65)
    memory = activity.twinx()
    x = np.arange(2)
    finished = throughput.bar(x - .17, [r['completed_episodes_per_minute'] for r in rows], .34,
                              color='#B64342', label='Finished attempts', zorder=3)
    verified = throughput.bar(x + .17, [r['verified_episodes_per_minute'] for r in rows], .34,
                              color='#3775BA', label='Verified attempts', zorder=3)
    gpu = activity.bar(x - .24, [r['rollout_sm_active_pct_mean'] for r in rows], .24,
                       color='#0F4D92', label='GPU SM Activity', zorder=3)
    cpu = activity.bar(x, [r['sandbox_cpu_busy_pct_mean'] for r in rows], .24,
                       color='#8BCF8B', label='CPU utilization', zorder=3)
    ram = memory.bar(x + .24, [r['peak_anonymous_memory_gib'] for r in rows], .24,
                      color='#9A4D8E', label='Peak anon. memory', zorder=3)
    for axis in (throughput, activity):
        axis.set_xticks(x, ['1', '2'])
        axis.set_xlabel('# CPU Hosts', fontsize=16)
    throughput.set_title('SWE-bench Pro V2 · C=128', loc='left', fontsize=14, pad=12)
    throughput.set(ylabel='Throughput\n(trajectories / min)', ylim=(0, 100))
    activity.set(ylabel='Utilization (%)', ylim=(0, 100))
    memory.set(ylabel='Peak anon. memory\n(GiB)', ylim=(0, 250))
    for axis in (throughput, activity, memory):
        axis.yaxis.label.set_fontsize(17.6)
    memory.yaxis.label.set_color('#9A4D8E')
    memory.tick_params(axis='y', colors='#9A4D8E')
    sns.despine(ax=throughput)
    sns.despine(ax=activity)
    sns.despine(ax=memory, left=True, right=False)
    memory.spines['right'].set_color('#9A4D8E')
    throughput.bar_label(finished, fmt='%.1f', padding=4, fontsize=12)
    throughput.bar_label(verified, fmt='%.1f', padding=16, fontsize=12)
    fig.legend(handles=[finished, gpu, verified, cpu, ram], loc='upper center',
               bbox_to_anchor=(.5, 1.13), ncol=3, fontsize=17.6, columnspacing=.8,
               handlelength=1.0, labelspacing=.3, frameon=True)
    fig.text(.5, -.04,
             'Same 128 tasks × 8 attempts; 8 rollout GPUs; 96 vCPUs per CPU host. One observation each.\n'
             'Incomplete verifications: 4 / 0. Two-host source: preceding sweep (job 3694759).',
             ha='center', fontsize=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(OUTPUT.with_suffix('.' + extension), dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    with OUTPUT.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', OUTPUT.with_suffix('.png'))
    for row in rows:
        print(f"{row['runtime_count']} hosts: {row['verified_episodes_per_minute']:.2f} verified/min, "
              f"CPU {row['sandbox_cpu_busy_pct_mean']:.2f}%, SM {row['rollout_sm_active_pct_mean']:.2f}%, "
              f"peak anon {row['peak_anonymous_memory_gib']:.2f} GiB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true', help='Re-extract frozen data from local outputs/')
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    plot()
