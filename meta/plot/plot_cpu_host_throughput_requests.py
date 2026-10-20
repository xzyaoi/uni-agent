"""Plot throughput by CPU host count beside model demand over time at C=128.

Run: python meta/plot/plot_cpu_host_throughput_requests.py
Output-token grouped-bar variant: add --metric output-tokens
Uses frozen inputs only; no cluster or raw exports required.
"""

import argparse
import json
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from plot_cpu_host_scaling import DATA, ROOT, digest
from plot_cpu_host_timeline import DATA as TIMELINE_DATA

OUTPUT = ROOT / 'figure/cpu_host_throughput_requests_c128_pro'
HOST_COLORS = {1: '#0F4D92', 2: '#9A4D8E'}
METRICS = [  # x tick label, mean key, value-label format
    ('Usable\ntrajectories / min', 'Usable / min', '{:.1f}'),
    ('Mean active\nrequests', 'Mean active requests', '{:.1f}'),
    ('Mean output\ntokens / s', 'Mean output (tokens/s)', '{:.0f}'),
]


def host_stats(observations, runs):
    """Validate both frozen sources and derive per-host summary metrics."""
    assert [r['runtime_count'] for r in observations] == [r['runtime_count'] for r in runs] == [1, 2]
    summary = []
    for observation, run in zip(observations, runs, strict=True):
        assert observation['state'] == 'completed' and observation['episode_ceiling'] == 128
        assert abs(observation['wall_seconds'] - run['duration_minutes'] * 60) < 2
        timeline_rows = run['rows']
        assert all(0 <= row['client_requests_mean'] <= 128 for row in timeline_rows)
        assert 0 < observation['usable_episodes'] <= observation['completed_episodes']
        seconds = sum(row['bin_seconds'] for row in timeline_rows)
        request_mean = sum(row['client_requests_mean'] * row['bin_seconds']
                           for row in timeline_rows) / seconds
        assert request_mean == run['summary']['client_requests_mean']
        assert max(row['client_requests_peak'] for row in timeline_rows) == \
            run['summary']['client_requests_peak']
        summary.append({
            'CPU hosts': observation['runtime_count'],
            'Usable / min': observation['usable_episodes'] * 60 / observation['wall_seconds'],
            'Mean active requests': request_mean,
            'Peak active requests': max(row['client_requests_peak'] for row in timeline_rows),
            'Mean output (tokens/s)': sum(row['delivered_output_tokens_s'] * row['bin_seconds']
                                          for row in timeline_rows) / seconds,
            'Peak output (tokens/s)': max(row['delivered_output_tokens_s'] for row in timeline_rows),
        })
    return summary


def draw_timeline(axis, runs):
    """Aggregate delivered output tokens/s stairs with duration-weighted means."""
    lines, mean_rates = [], []
    for run in runs:
        rows = run['rows']
        edges = [0, *[(row['elapsed_minutes'] * 60 + row['bin_seconds'] / 2) / 60 for row in rows]]
        host_count = run['runtime_count']
        line = axis.stairs([row['delivered_output_tokens_s'] for row in rows], edges, baseline=None,
                           color=HOST_COLORS[host_count], linewidth=1.8,
                           label=f"{host_count} CPU host{'s' if host_count > 1 else ''}")
        lines.append(line)
        seconds = sum(row['bin_seconds'] for row in rows)
        mean_rate = sum(row['delivered_output_tokens_s'] * row['bin_seconds']
                        for row in rows) / seconds
        mean_rates.append(mean_rate)
        axis.hlines(mean_rate, 0, run['duration_minutes'], color=HOST_COLORS[host_count],
                    linestyle='--', linewidth=1.5, zorder=4)
        axis.annotate(f'{mean_rate:.1f}', (run['duration_minutes'], mean_rate),
                      xytext=(14 if host_count == 2 else -3, 5), textcoords='offset points',
                      ha='right', va='bottom', color=HOST_COLORS[host_count], fontsize=13,
                      zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
        print(f'{host_count} CPU hosts: mean output rate {mean_rate:.2f} tokens/s '
              f'over {seconds:.2f} s')
    axis.set(xlim=(0, max(r['duration_minutes'] for r in runs)), xticks=[0, 10, 20, 30],
             ylim=(0, 4000), yticks=[0, 1000, 2000, 3000, 4000], ylabel='Output tokens / s')
    axis.annotate(f'+{(mean_rates[1] / mean_rates[0] - 1) * 100:.1f}%\nmean output',
                  xy=(20, mean_rates[1]), xytext=(26, 2400), ha='center', va='center',
                  fontsize=14, color=HOST_COLORS[2], fontweight='bold',
                  arrowprops={'arrowstyle': '->', 'color': HOST_COLORS[2], 'lw': 1.5}, zorder=6)
    axis.set_xlabel('Elapsed time (min)', fontsize=16)
    axis.set_title('Aggregate output tokens', loc='left', fontsize=14, pad=12)
    return lines


def draw_output_token_grouped(host_rows, runs):
    """Mean-rate host bars (three metrics on the x-axis, one- and two-CPU-host
    bars dodged within each group; log rate axis) beside the delivered output
    tokens/s timeline."""
    fig, (rates, timeline) = plt.subplots(
        1, 2, figsize=(9, 3.75), gridspec_kw={'width_ratios': (1.35, 1)})
    fig.subplots_adjust(left=.15, right=.98, bottom=.28, top=.72, wspace=.3)
    x = np.arange(len(METRICS))
    for host_index, host_count in enumerate((1, 2)):
        color = HOST_COLORS[host_count]
        positions = x + (-.22 if host_count == 1 else .22)
        for pos, (_, mean_key, fmt) in zip(positions, METRICS):
            mean = host_rows[host_index][mean_key]
            bars = rates.bar(pos, mean, width=.2, color=color, linewidth=0, zorder=3)
            rates.bar_label(bars, labels=[fmt.format(mean)], padding=3, fontsize=13)

    rates.set_yscale('log')
    rates.set(xlim=(-.5, 2.5), xticks=x, ylim=(2.2, 10000))
    rates.set_xticklabels([metric[0] for metric in METRICS], fontsize=11)
    rates.set_yticks([10, 100, 1000], labels=['10', '100', '1000'])
    for group_x, mean_key, arrow_y in ((0, 'Usable / min', 200),
                                       (1, 'Mean active requests', 180),
                                       (2, 'Mean output (tokens/s)', 1900)):
        gain = (host_rows[1][mean_key] / host_rows[0][mean_key] - 1) * 100
        rates.annotate('', xy=(group_x + .22, arrow_y),
                       xytext=(group_x - .22, arrow_y),
                       arrowprops={'arrowstyle': '->', 'color': HOST_COLORS[2],
                                   'lw': 1.5}, zorder=6)
        rates.text(group_x, arrow_y * 1.12, f'+{gain:.1f}%', ha='center', va='bottom',
                   color=HOST_COLORS[2], fontsize=14, fontweight='bold', zorder=6)
    rates.set_title('SWE-bench Pro V2 · C=128', loc='left', fontsize=14, pad=12)
    rates.set_ylabel('Rate (log scale)', fontsize=16)
    rates.tick_params(labelsize=12)
    rates.grid(axis='y', color='#CFCECE', linewidth=.8, alpha=.4, zorder=0)
    rates.set_axisbelow(True)

    draw_timeline(timeline, runs)

    for axis in (rates, timeline):
        axis.yaxis.label.set_fontsize(17.6)
        sns.despine(ax=axis)
    fig.legend(handles=[Patch(facecolor=HOST_COLORS[1], label='1 CPU host'),
                        Patch(facecolor=HOST_COLORS[2], label='2 CPU hosts')],
               loc='upper center', bbox_to_anchor=(.5, .99), ncol=2, fontsize=12.5,
               columnspacing=1, handlelength=1.2, labelspacing=.3, frameon=True)
    return fig


def draw_requests_compact(observations, runs):
    """Original compact layout: usable throughput beside requests in flight."""
    fig, (throughput, requests) = plt.subplots(1, 2, figsize=(9, 3.75))
    fig.subplots_adjust(left=.15, right=.98, bottom=.28, top=.72, wspace=.65)
    x = np.arange(len(observations))
    usable_rates = [r['usable_episodes'] * 60 / r['wall_seconds'] for r in observations]
    usable = throughput.bar(x, usable_rates, width=.55, color='#3775BA',
                            label='Usable trajectories', zorder=3)
    throughput.set_xticks(x, ['1', '2'])
    throughput.set_xlabel('# CPU Hosts', fontsize=16)
    throughput.set(ylabel='Throughput\n(trajectories / min)', ylim=(0, 100))
    throughput.set_title('SWE-bench Pro V2 · C=128', loc='left', fontsize=14, pad=12)
    throughput.bar_label(usable, fmt='%.1f', padding=4, fontsize=12)
    throughput_gain = (usable_rates[1] / usable_rates[0] - 1) * 100
    throughput.annotate('', xy=(1, usable_rates[1] + 20), xytext=(0, usable_rates[0] + 20),
                        arrowprops={'arrowstyle': '->', 'color': '#3775BA', 'lw': 1.8})
    throughput.text(.5, 69, f'+{throughput_gain:.1f}%', ha='center', va='bottom',
                    color='#3775BA', fontsize=13, fontweight='bold')

    lines = []
    for run in runs:
        rows = run['rows']
        edges = [0, *[(r['elapsed_minutes'] * 60 + r['bin_seconds'] / 2) / 60 for r in rows]]
        host_count = run['runtime_count']
        line = requests.stairs([r['client_requests_mean'] for r in rows], edges, baseline=None,
                               color=HOST_COLORS[host_count], linewidth=1.8,
                               label=f"{host_count} CPU host{'s' if host_count > 1 else ''}")
        lines.append(line)
    requests.set(xlim=(0, max(r['duration_minutes'] for r in runs)), xticks=[0, 10, 20, 30],
                 ylim=(0, 128), yticks=[0, 32, 64, 96, 128], ylabel='Active client requests')
    requests.set_xlabel('Elapsed time (min)', fontsize=16)
    requests.set_title('Model requests in flight', loc='left', fontsize=14, pad=12)
    for axis in (throughput, requests):
        axis.yaxis.label.set_fontsize(17.6)
        sns.despine(ax=axis)
    fig.legend(handles=[usable, *lines], loc='upper center',
               bbox_to_anchor=(.5, .99), ncol=3, fontsize=17.6,
               columnspacing=1, handlelength=1.2, labelspacing=.3, frameon=True)
    fig.text(.5, -.04,
             'Client requests: 10-second means since model readiness; includes gateway/network waits.\n'
             'One observation per host count; two-host result from the preceding sweep. No training updates.',
             ha='center', fontsize=8.5)
    return fig


def main(metric='requests'):
    token_rate = metric == 'output-tokens'
    output = OUTPUT.with_name('cpu_host_throughput_output_tokens_c128_pro') if token_rate else OUTPUT
    provenance = json.loads(DATA.with_suffix('.provenance.json').read_text())
    assert digest(DATA) == provenance['sha256']
    observations = json.loads(DATA.read_text())['rows']
    runs = json.loads(TIMELINE_DATA.read_text())['runs']
    host_rows = host_stats(observations, runs)
    if token_rate:
        assert [row['Peak active requests'] for row in host_rows] == [124, 127]
        assert [round(row['Mean active requests'], 2) for row in host_rows] == [4.16, 6.55]
        assert [row['Peak output (tokens/s)'] for row in host_rows] == [3557.2, 3684.9]

    sns.set_style('ticks')
    sns.set_style({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 12})
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 3, 'lines.markersize': 10})
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig = draw_output_token_grouped(host_rows, runs) if token_rate else \
        draw_requests_compact(observations, runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=300, facecolor='white',
                    bbox_inches=None if token_rate else 'tight')
    plt.close(fig)
    print('Saved', output.with_suffix('.png'))
    for observation, run, host_row in zip(observations, runs, host_rows, strict=True):
        bins = run['resources_1min']
        mean_sandboxes = statistics.mean(bin['sandbox_vm_count'] for bin in bins)
        print(f"{observation['runtime_count']} CPU hosts: {host_row['Usable / min']:.2f} usable "
              f"trajectories/min; {mean_sandboxes:.2f} mean resident sandbox VMs across "
              f"{len(bins)} complete rollout minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=['requests', 'output-tokens'], default='requests')
    args = parser.parse_args()
    main(args.metric)
