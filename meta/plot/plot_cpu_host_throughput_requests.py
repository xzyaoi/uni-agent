"""Plot throughput by CPU host count beside model demand over time at C=128.

Run: python meta/plot/plot_cpu_host_throughput_requests.py
Output-token rate: add --metric output-tokens
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
from plot_cpu_host_scaling import DATA, ROOT, digest
from plot_cpu_host_timeline import DATA as TIMELINE_DATA

OUTPUT = ROOT / 'figure/cpu_host_throughput_requests_c128_pro'
HOST_COLORS = {1: '#0F4D92', 2: '#9A4D8E'}


def main(metric='requests'):
    token_rate = metric == 'output-tokens'
    output = OUTPUT.with_name('cpu_host_throughput_output_tokens_c128_pro') if token_rate else OUTPUT
    key = 'delivered_output_tokens_s' if token_rate else 'client_requests_mean'
    provenance = json.loads(DATA.with_suffix('.provenance.json').read_text())
    assert digest(DATA) == provenance['sha256']
    observations = json.loads(DATA.read_text())['rows']
    runs = json.loads(TIMELINE_DATA.read_text())['runs']
    assert [r['runtime_count'] for r in observations] == [r['runtime_count'] for r in runs] == [1, 2]
    for observation, run in zip(observations, runs, strict=True):
        assert observation['state'] == 'completed' and observation['episode_ceiling'] == 128
        assert abs(observation['wall_seconds'] - run['duration_minutes'] * 60) < 2
        assert all(0 <= row['client_requests_mean'] <= 128 for row in run['rows'])
        assert 0 < observation['usable_episodes'] <= observation['completed_episodes']

    sns.set_style('ticks')
    sns.set_style({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 12})
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 3, 'lines.markersize': 10})
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none'})
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
    mean_rates = []
    for run in runs:
        rows = run['rows']
        edges = [0, *[(r['elapsed_minutes'] * 60 + r['bin_seconds'] / 2) / 60 for r in rows]]
        host_count = run['runtime_count']
        line = requests.stairs([r[key] for r in rows], edges, baseline=None,
                               color=HOST_COLORS[host_count], linewidth=1.8,
                               label=f"{host_count} CPU host{'s' if host_count > 1 else ''}")
        lines.append(line)
        if token_rate:
            seconds = sum(row['bin_seconds'] for row in rows)
            mean_rate = sum(row[key] * row['bin_seconds'] for row in rows) / seconds
            mean_rates.append(mean_rate)
            requests.hlines(mean_rate, 0, run['duration_minutes'], color=HOST_COLORS[host_count],
                            linestyle='--', linewidth=1.5, zorder=4)
            requests.annotate(f'{mean_rate:.1f}', (run['duration_minutes'], mean_rate),
                              xytext=(14 if host_count == 2 else -3, 5), textcoords='offset points',
                              ha='right', va='bottom',
                              color=HOST_COLORS[host_count], fontsize=12, zorder=5,
                              bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
            print(f'{host_count} CPU hosts: mean output rate {mean_rate:.2f} tokens/s over {seconds:.2f} s')
    requests.set(xlim=(0, max(r['duration_minutes'] for r in runs)), xticks=[0, 10, 20, 30])
    if token_rate:
        requests.set(ylim=(0, 4000), yticks=[0, 1000, 2000, 3000, 4000],
                     ylabel='Output tokens / s')
        output_gain = (mean_rates[1] / mean_rates[0] - 1) * 100
        requests.annotate(f'+{output_gain:.1f}%\nmean output', xy=(20, mean_rates[1]), xytext=(26, 2400),
                          ha='center', va='center', fontsize=12, color=HOST_COLORS[2], fontweight='bold',
                          arrowprops={'arrowstyle': '->', 'color': HOST_COLORS[2], 'lw': 1.5}, zorder=6)
    else:
        requests.set(ylim=(0, 128), yticks=[0, 32, 64, 96, 128], ylabel='Active client requests')
    requests.set_xlabel('Elapsed time (min)', fontsize=16)
    requests.set_title('Aggregate output tokens' if token_rate else 'Model requests in flight',
                       loc='left', fontsize=14, pad=12)
    for axis in (throughput, requests):
        axis.yaxis.label.set_fontsize(17.6)
        sns.despine(ax=axis)
    fig.legend(handles=[usable, *lines], loc='upper center',
               bbox_to_anchor=(.5, .99), ncol=3, fontsize=17.6,
               columnspacing=1, handlelength=1.2, labelspacing=.3, frameon=True)
    if not token_rate:
        fig.text(.5, -.04,
                 'Client requests: 10-second means since model readiness; includes gateway/network waits.\n'
                 'One observation per host count; two-host result from the preceding sweep. No training updates.',
                 ha='center', fontsize=8.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=300, facecolor='white',
                    bbox_inches=None if token_rate else 'tight')
    plt.close(fig)
    print('Saved', output.with_suffix('.png'))
    for observation, run, rate in zip(observations, runs, usable_rates, strict=True):
        bins = run['resources_1min']
        mean_sandboxes = statistics.mean(row['sandbox_vm_count'] for row in bins)
        print(f"{observation['runtime_count']} CPU hosts: {rate:.2f} usable trajectories/min; "
              f"{mean_sandboxes:.2f} mean resident sandbox VMs across {len(bins)} complete rollout minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=['requests', 'output-tokens'], default='requests')
    args = parser.parse_args()
    main(args.metric)
