"""Explore the studied CPU/memory-time roofline for the two Pro C=128 runs.

python meta/plot/plot_cpu_memory_roofline.py --refresh-data
The default memory basis is memory.stat anon; --memory-basis charged retains
the memory.current accounting view.
Subsequent renders use frozen data only. These exports lack the identity and
monotonic-clock evidence required by the strict roofline profiler.
"""

import argparse
import gzip
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from plot_cpu_host_scaling import REPO, ROOT, SOURCES, digest

DATA = ROOT / 'data/cpu_memory_roofline_c128_20261019.json'
OUTPUT = ROOT / 'figure/cpu_memory_roofline_c128_pro'
COLORS = ['#0F4D92', '#9A4D8E']


def integrate_host(samples, start, end):
    """Integrate CPU and memory on identical, linearly interpolated boundaries."""
    assert samples[0]['timestamp'] <= start < end <= samples[-1]['timestamp']
    times = [s['timestamp'] for s in samples]
    points = [start, *[t for t in times if start < t < end], end]
    cpu = np.interp([start, end], times, [s['cpu_seconds'] for s in samples])
    result = {'cpu_seconds': float(cpu[1] - cpu[0])}
    for field in ('charged_gib', 'anonymous_gib', 'file_gib'):
        values = np.interp(points, times, [s[field] for s in samples])
        result[field + '_seconds'] = float(sum(
            (b - a) * (left + right) / 2
            for a, b, left, right in zip(points[:-1], points[1:], values[:-1], values[1:], strict=True)
        ))
    return result


def refresh_data():
    runs = []
    for hosts, campaign in enumerate(SOURCES, 1):
        point = campaign / 'results/groups-16'
        result = json.loads((point / 'result.json').read_text())
        definitions = json.loads((point / 'cluster_metric_definitions.json').read_text())
        manifest = json.loads((point / 'manifest.json').read_text())
        assert result['state'] == 'completed' and result['max_concurrent_episodes'] == 128
        assert definitions['host_vcpus_each'] == 96 and len(definitions['expected_nodes']) == hosts
        sources = [point / name for name in ('result.json', 'cluster_metric_definitions.json', 'manifest.json')]
        nodes = []
        for node in range(1, hosts + 1):
            source = point / f'inputs/runtime-{node}/host.jsonl.gz'
            sources.append(source)
            samples = []
            with gzip.open(source, 'rt') as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get('kind') != 'host':
                        continue
                    assert event['cgroup']['path'] == '/system.slice/aenv.service'
                    files = event['cgroup']['files']
                    cpu = dict(line.split() for line in files['cpu.stat'].splitlines())
                    memory = dict(line.split() for line in files['memory.stat'].splitlines())
                    meminfo = dict((line.split()[0].rstrip(':'), line.split()[1])
                                   for line in event['meminfo'].splitlines())
                    samples.append({
                        'timestamp': event['timestamp'], 'cpu_seconds': int(cpu['usage_usec']) / 1e6,
                        'charged_gib': int(files['memory.current']) / 2**30,
                        'anonymous_gib': int(memory['anon']) / 2**30,
                        'file_gib': int(memory['file']) / 2**30,
                        'capacity_gib': int(meminfo['MemTotal']) / 2**20,
                    })
            assert len(samples) >= 2
            pairs = list(zip(samples[:-1], samples[1:], strict=True))
            gaps = [b['timestamp'] - a['timestamp'] for a, b in pairs]
            assert all(0 < gap <= 6 for gap in gaps), 'Unsupported sampling gap in exploratory integration'
            assert all(b['cpu_seconds'] >= a['cpu_seconds'] for a, b in pairs)
            assert all(s['capacity_gib'] == samples[0]['capacity_gib'] for s in samples)
            nodes.append({'id': f'runtime-{node}', 'samples': samples, 'max_gap_seconds': max(gaps)})
        start = max(node['samples'][0]['timestamp'] for node in nodes)
        end = min(node['samples'][-1]['timestamp'] for node in nodes)
        duration = end - start
        assert result['model_ready_at'] <= start < end <= result['finished_at']
        integrals = [integrate_host(node['samples'], start, end) for node in nodes]
        cpu = sum(r['cpu_seconds'] for r in integrals)
        area = sum(r['charged_gib_seconds'] for r in integrals)
        anonymous_area = sum(r['anonymous_gib_seconds'] for r in integrals)
        nominal_cpu = 96 * hosts
        nominal_ram = sum(node['samples'][0]['capacity_gib'] for node in nodes)
        assert cpu > 0 and area > 0 and anonymous_area > 0
        # Session scores are read once per final (UID, session) trajectory.
        # Reconcile to the independently retained usable-trajectory count, then
        # attribute availability at the group's observed terminal timestamp.
        retained = {uid: len(scores) for uid, scores in result['scores_by_uid'].items()}
        assert all(0 <= count <= manifest['responses'] for count in retained.values())
        assert sum(retained.values()) == result['scored_sessions'] == (
            result['progress']['episodes_with_usable_trajectory'])
        demand = point / 'inputs/demand.jsonl.gz'
        sources.append(demand)
        terminals = {}
        with gzip.open(demand, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if event.get('kind') == 'pipeline' and event.get('event') == 'group_terminal':
                    assert event['uid'] not in terminals, 'Duplicate group terminal would double count trajectories'
                    terminals[event['uid']] = event['timestamp']
        assert set(terminals) == set(retained) and len(terminals) == manifest['groups']
        in_window = {uid for uid, stamp in terminals.items() if start <= stamp <= end}
        useful = sum(retained[uid] for uid in in_window)
        assert useful > 0
        cpu_per_useful, memory_per_useful = cpu / useful, area / useful
        rows = {
            'runtime_count': hosts, 'benchmark': 'SWE-bench Pro V2', 'episode_ceiling': 128,
            'selection_sha256': manifest['selection_sha256'],
            'common_window_start_epoch': start, 'common_window_end_epoch': end,
            'window_seconds': duration,
            'sampled_fraction_of_rollout_wall_window': duration / (result['finished_at'] - result['model_ready_at']),
            'cpu_seconds': cpu, 'charged_memory_gib_seconds': area,
            'anonymous_memory_gib_seconds': anonymous_area,
            'mean_cpu_service': cpu / duration, 'mean_charged_memory_gib': area / duration,
            'mean_anonymous_memory_gib': anonymous_area / duration,
            'mean_file_memory_gib': sum(r['file_gib_seconds'] for r in integrals) / duration,
            'intensity_cpu_eq_per_gib': cpu / area,
            'anonymous_intensity_cpu_eq_per_gib': cpu / anonymous_area,
            'useful_trajectories_in_window': useful,
            'retained_usable_trajectories_full_run': sum(retained.values()),
            'terminal_groups_in_window': len(in_window),
            'excluded_group_terminals': [
                {'uid': uid, 'timestamp': stamp, 'retained_trajectories': retained[uid]}
                for uid, stamp in terminals.items() if uid not in in_window
            ],
            'useful_trajectories_per_minute': 60 * useful / duration,
            'cpu_seconds_per_useful_trajectory': cpu_per_useful,
            'charged_memory_gib_seconds_per_useful_trajectory': memory_per_useful,
            'anonymous_memory_gib_seconds_per_useful_trajectory': anonymous_area / useful,
            'conditional_cpu_roof_trajectories_per_minute': 60 * nominal_cpu / cpu_per_useful,
            'conditional_memory_roof_at_observation_trajectories_per_minute': 60 * nominal_ram / memory_per_useful,
            'conditional_anonymous_memory_roof_trajectories_per_minute': 60 * nominal_ram * useful / anonymous_area,
            'nominal_cpu_logical_processors': nominal_cpu, 'nominal_host_ram_gib': nominal_ram,
            'nominal_knee_cpu_eq_per_gib': nominal_cpu / nominal_ram,
            'max_sample_gap_seconds': max(node['max_gap_seconds'] for node in nodes),
            'strict_profiler_compatible': False,
            'nodes': [{
                'id': node['id'], 'sample_count': len(node['samples']),
                'first_charged_memory_gib': node['samples'][0]['charged_gib'],
                'first_file_memory_gib': node['samples'][0]['file_gib'],
                **integral,
            } for node, integral in zip(nodes, integrals, strict=True)],
            'source_hashes': {str(p.relative_to(REPO)): digest(p) for p in sources},
        }
        runs.append(rows)
    assert runs[0]['selection_sha256'] == runs[1]['selection_sha256']
    frozen = {
        'model': 'I=C/A; Y=C/T; Y <= min(P, M*I)',
        'default_memory_basis': 'anon',
        'useful_trajectory_model': 'N=retained useful trajectories attributed to in-window group terminals; '
                                   'c=C/N; d=A/N; R=60*N/T; conditional roof R(I)=60*min(P,M*I)/c.',
        'reference': '../infersim/scripts/analyzer/plot_aws_replay_roofline.py; '
                     '../../flash-sandbox/meta/tools/roofline_profiler/README.md',
        'definitions': {
            'C': 'Sum of aenv.service cumulative cpu.stat usage_usec deltas, converted to CPU seconds.',
            'A': 'Trapezoidal integral of memory.current in GiB, summed across active hosts.',
            'A_anon': 'Trapezoidal integral of memory.stat anon in GiB over the same common host window.',
            'T': 'Intersection of first/last host sample windows; interpolate each host at common boundaries.',
            'P_and_M': 'Nominal 96 logical CPUs and sampled MemTotal per host, not calibrated or enforced budgets.',
            'useful_trajectory': 'A final stored scored trajectory, reconciled with progress usable count; '
                                 'reward zero is included. Availability is credited at group_terminal.',
            'c_and_d': 'Same-window CPU and charged-memory-time cost per emitted useful trajectory, '
                       'including resource costs of unfinished and unsuccessful work.',
        },
        'limitations': [
            'Exploratory accounting only; hollow markers are not strict-profiler-validated measurements.',
            'Only wall timestamps and cgroup paths were retained; no monotonic clock, boot ID, or cgroup inode.',
            'Approximately five-second sampling exceeds the reference renderer strict three-second gap rule; '
            'exploratory integration checks positive gaps no larger than six seconds.',
            'Charged memory includes substantial file cache present before the run; no baseline subtraction.',
            'Exclusive cgroup ownership and effective ancestor CPU quotas were not established from these exports.',
            'Whole-fleet placements for the same Pro workload at two host counts; individual tasks are not attributed.',
            'Nearness to the nominal envelope or side of the knee does not identify a proven operational bottleneck.',
            'Useful trajectories are credited at group termination, not at an individually measured persistence time.',
            'Resource windows exclude unsampled run edges; the final group of each run terminates outside the window.',
            'Each useful-trajectory roof assumes that run-specific measured cost per trajectory remains fixed '
            'as intensity or capacity changes. This is a conditional accounting projection, not a calibrated roof.',
            'The conditional roof includes only CPU and the selected memory basis; no GPU-serving roof was measured.',
            'The anonymous-memory variant excludes file cache and kernel costs; '
            'it is not a complete working-set model.',
        ],
        'runs': runs,
    }
    DATA.write_text(json.dumps(frozen, indent=2) + '\n')


def plot(y_axis='useful-trajectories', memory_basis='anon'):
    useful_axis = y_axis == 'useful-trajectories'
    anonymous = memory_basis == 'anon'
    stem = 'cpu_memory_roofline' + ('_anon' if anonymous else '') + ('' if useful_axis else '_cpu_service')
    output = OUTPUT.with_name(stem + '_c128_pro')
    runs = json.loads(DATA.read_text())['runs']
    sns.set_style('ticks')
    sns.set_style({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 12})
    sns.set_context('paper', font_scale=2)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.30, top=.72, wspace=.30)
    xmax = 1.2 if anonymous else .6
    x = np.linspace(0, xmax, 301)
    for axis, run, color in zip(axes, runs, COLORS, strict=True):
        p, m = run['nominal_cpu_logical_processors'], run['nominal_host_ram_gib']
        curve_x = np.sort(np.append(x, p / m))
        scale = 60 / run['cpu_seconds_per_useful_trajectory'] if useful_axis else 1
        axis.plot(curve_x, np.minimum(p, m * curve_x) * scale, color=color, ls='--', lw=2)
        axis.axvline(p / m, color='#999999', ls=':', lw=1)
        area = run['anonymous_memory_gib_seconds'] if anonymous else run['charged_memory_gib_seconds']
        intensity, service = run['cpu_seconds'] / area, run['mean_cpu_service']
        ordinate = run['useful_trajectories_per_minute'] if useful_axis else service
        label = 'R' if useful_axis else 'Y'
        axis.scatter([intensity], [ordinate], s=90, facecolors='white', edgecolors=color, linewidths=2, zorder=4)
        axis.annotate(f'I={intensity:.3f}, {label}={ordinate:.1f}', (intensity, ordinate),
                      xytext=(-10 if anonymous else 10, -2), textcoords='offset points',
                      ha='right' if anonymous else 'left', va='top', fontsize=12, color=color,
                      bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        capacity_on_right = useful_axis and run['runtime_count'] == 2
        capacity_y = (.55 if anonymous else .05) if capacity_on_right else .95
        axis.text(.97 if capacity_on_right else .04, capacity_y,
                  f'P={p}; M={m:.1f} GiB', transform=axis.transAxes,
                  va='bottom' if capacity_on_right else 'top',
                  ha='right' if capacity_on_right else 'left', fontsize=12,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        axis.set(xscale='linear', yscale='linear', xlim=(0, xmax), ylim=(0, 160 if useful_axis else 240))
        axis.set_xticks([0, .4, .8, 1.2] if anonymous else [0, .2, .4, .6])
        axis.set_yticks([0, 50, 100, 150] if useful_axis else [0, 100, 200])
        axis.set_xlabel('CPU / anon-memory-time intensity\n(CPU-eq / GiB)' if anonymous else
                        'CPU / memory-time intensity\n(CPU-eq / GiB)', fontsize=14)
        hosts = run['runtime_count']
        axis.set_title(f"Pro C=128 · {hosts} CPU host{'s' if hosts > 1 else ''}", loc='left', fontsize=14, pad=12)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.grid(which='major', alpha=.15)
        sns.despine(ax=axis)
        print(f"{hosts} hosts: I={intensity:.4f}, {run['useful_trajectories_in_window']} useful trajectories; "
              f"R={run['useful_trajectories_per_minute']:.2f}/min; "
              f"conditional CPU roof={run['conditional_cpu_roof_trajectories_per_minute']:.2f}/min")
    axes[0].set_ylabel('Useful trajectories\n(trajectories / min)' if useful_axis else
                       'Mean CPU service\n(CPU-s / s)', fontsize=14)
    fig.legend(handles=[
        Line2D([], [], color='#555555', ls='--', lw=2,
               label=('Conditional CPU/anon roof' if anonymous else 'Conditional CPU/RAM roof')
               if useful_axis else 'Nominal reference envelope'),
        Line2D([], [], color='#555555', marker='o', markerfacecolor='white', ls='none',
               label='Observed useful trajectories' if useful_axis else 'Exploratory cgroup mean'),
    ], loc='upper center', bbox_to_anchor=(.5, .99), ncol=2, fontsize=14, frameon=True)
    caption = (
        'Roof uses measured CPU cost per trajectory; charged memory includes retained cache.' if useful_axis else
        'Charged-memory accounting includes retained file cache; capacities are nominal.'
    )
    if anonymous:
        caption = 'Anonymous memory only; kernel and file-cache costs excluded. Roofs remain conditional.'
    fig.text(.5, .015, caption,
             ha='center', fontsize=12)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=300, facecolor='white')
    plt.close(fig)
    print('Saved', output.with_suffix('.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true')
    parser.add_argument('--y-axis', choices=['useful-trajectories', 'cpu-service'], default='useful-trajectories')
    parser.add_argument('--memory-basis', choices=['anon', 'charged'], default='anon')
    args = parser.parse_args()
    if args.refresh_data:
        refresh_data()
    plot(args.y_axis, args.memory_basis)
