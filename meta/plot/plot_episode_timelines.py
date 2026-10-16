"""Render Pro and Verified timelines from frozen data, without cluster access.

Run: python meta/plot/plot_episode_timelines.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
COLORS = {
    'LLM Wait': '#DDF3DE', 'Cold Start': '#B64342', 'Setup': '#CFCECE',
    'Sandbox Interaction': '#0F4D92', 'Verification': '#9A4D8E',
}


def plot(sources, output):
    datasets = [json.loads(path.read_text()) for path in sources]
    plt.rcParams.update({'font.family': ['Roboto', 'DejaVu Sans'],
                         'font.size': 11, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    rows = max(len(data['points']) for data in datasets)
    xmax = max(e['lifetime_s'] for data in datasets for p in data['points'] for e in p['episodes']) * 1.025
    fig, axes = plt.subplots(rows, len(datasets), figsize=(9 * len(datasets), 2 * rows + 1.6),
                             squeeze=False, sharex=True)
    selections = []
    starts = []
    for col, data in enumerate(datasets):
        benchmark = data.get('benchmark', 'SWE-bench Pro V2')
        for row, point in enumerate(data['points']):
            ax = axes[row, col]
            for index, episode in enumerate(point['episodes'], 1):
                for phase, color in COLORS.items():
                    intervals = [(s['start_s'], s['end_s'] - s['start_s'])
                                 for s in episode['segments'] if s['category'] == phase]
                    if intervals:
                        ax.broken_barh(intervals, (4 - index - .32, .64),
                                       facecolors=color, edgecolors='none')
                selections.append({'benchmark': benchmark, 'campaign': data['campaign'],
                                   'concurrent_trajectories': point['episode_ceiling'],
                                   'lane': 'e%d' % index, 'sandbox_id': episode['sandbox_id'],
                                   'trace_file': episode['trace_file'], 'lifetime_s': episode['lifetime_s'],
                                   'model_calls': episode['model_calls'],
                                   'eligible_episodes': point['eligible_episodes'],
                                   **{phase: episode['phase_seconds'].get(phase, 0) for phase in COLORS}})
            starts.append({'benchmark': benchmark, 'concurrent_trajectories': point['episode_ceiling'],
                           'successful_starts': point['successful_start_calls'],
                           'median_cold_start_s': point['successful_start_median_s'],
                           'p90_cold_start_s': point['successful_start_p90_s']})
            title = f"{point['episode_ceiling']} concurrent trajectories"
            if benchmark == 'SWE-bench Pro V2' and point['episode_ceiling'] == 512:
                title += ' (run had engine failure)'
            ax.set_title(title, loc='left', fontsize=12)
            ax.set(xlim=(0, xmax), ylim=(-.55, 3.55), yticks=[0, 1, 2, 3],
                   yticklabels=['e4', 'e3', 'e2', 'e1'])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_axisbelow(True)
            ax.grid(axis='x', linestyle=':', color='#CFCECE')
        for row in range(len(data['points']), rows):
            axes[row, col].set_visible(False)
        axes[len(data['points']) - 1, col].set_xlabel('Time since sandbox request (s)')
        fig.text((col + .5) / len(datasets), .88, benchmark, ha='center', fontsize=17)
    handles = [Patch(facecolor=color, label=phase) for phase, color in COLORS.items()]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(.5, .98))
    fig.text(.5, .025,
             'Four median-near, complete, error-free-call traces per panel; episodes are not task-matched.\n'
             'Verification is an inferred phase; LLM Wait includes untraced overhead. Deletion time excluded.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.055, right=.985, bottom=.14, top=.81, hspace=.58, wspace=.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    for suffix, records in [('selection', selections), ('cold_start_summary', starts)]:
        with output.with_name(output.name + '_' + suffix + '.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print('Saved', output.with_suffix('.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', default=[
        ROOT / 'data/episode_timelines_20261015.json',
        ROOT / 'data/episode_timelines_verified_20261016.json'])
    parser.add_argument('--output', type=Path, default=ROOT / 'figure/episode_timelines')
    args = parser.parse_args()
    plot(args.data, args.output)
