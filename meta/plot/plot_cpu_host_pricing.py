"""Compare modeled active-resource costs at the user's supplied hourly rates.

Run: python meta/plot/plot_cpu_host_pricing.py
The GPU rate is for the entire eight-GPU allocation. Timings are from GH200
rollout measurements; applying an H100 price does not measure H100 performance.
"""

import argparse
import csv
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot_cpu_host_scaling import DATA, ROOT, digest

OUTPUT = ROOT / 'figure/cpu_host_pricing_c128_pro'


def main(cpu_hourly=5.424, gpu_hourly=5.970):
    if cpu_hourly < 0 or gpu_hourly < 0:
        raise ValueError('Hourly prices must be nonnegative')
    provenance = json.loads(DATA.with_suffix('.provenance.json').read_text())
    assert digest(DATA) == provenance['sha256']
    observations = json.loads(DATA.read_text())['rows']
    assert [r['runtime_count'] for r in observations] == [1, 2]
    rows = []
    for observation in observations:
        assert observation['state'] == 'completed' and observation['episode_ceiling'] == 128
        hours = observation['wall_seconds'] / 3600
        usable = observation['usable_episodes']
        cpu_rate = cpu_hourly * observation['runtime_count']
        total_rate = cpu_rate + gpu_hourly
        rows.append({
            'runtime_count': observation['runtime_count'],
            'cpu_host_usd_per_hour': cpu_hourly,
            'cpu_fleet_usd_per_hour': cpu_rate,
            'eight_gpu_allocation_usd_per_hour': gpu_hourly,
            'total_usd_per_hour': total_rate,
            'rollout_seconds': observation['wall_seconds'],
            'usable_trajectories': usable,
            'usable_trajectories_per_minute': usable / (hours * 60),
            'modeled_rollout_usd': total_rate * hours,
            'cpu_usd_per_1000_usable': cpu_rate * hours / usable * 1000,
            'gpu_usd_per_1000_usable': gpu_hourly * hours / usable * 1000,
            'total_usd_per_1000_usable': total_rate * hours / usable * 1000,
        })

    sns.set_style('ticks')
    sns.set_style({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 12})
    sns.set_context('paper', font_scale=2, rc={'lines.linewidth': 3, 'lines.markersize': 10})
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.75))
    fig.subplots_adjust(left=.13, right=.98, bottom=.28, top=.72, wspace=.65)
    x = np.arange(2)
    handles = None
    for axis, cpu_key, gpu_key, total_key, title, ylabel, ymax, margin, annotation_y in [
        (axes[0], 'cpu_fleet_usd_per_hour', 'eight_gpu_allocation_usd_per_hour', 'total_usd_per_hour',
         'Hourly allocation', 'Hourly cost\n(USD / h)', 30, 4, 26),
        (axes[1], 'cpu_usd_per_1000_usable', 'gpu_usd_per_1000_usable', 'total_usd_per_1000_usable',
         'Cost per usable trajectory', 'Cost / 1,000 usable\ntrajectories (USD)', 12, 2, 10.2),
    ]:
        cpu_values = [r[cpu_key] for r in rows]
        gpu_values = [r[gpu_key] for r in rows]
        totals = [r[total_key] for r in rows]
        cpu = axis.bar(x, cpu_values, width=.55, color='#3775BA', label='CPU hosts', zorder=3)
        gpu = axis.bar(x, gpu_values, bottom=cpu_values, width=.55, color='#9A4D8E',
                       label='8-GPU allocation', zorder=3)
        if handles is None:
            handles = [cpu, gpu]
        axis.bar_label(gpu, labels=[f'${v:.2f}' for v in totals], padding=4, fontsize=12)
        gain = (totals[1] / totals[0] - 1) * 100
        axis.annotate('', xy=(1, totals[1] + margin), xytext=(0, totals[0] + margin),
                      arrowprops={'arrowstyle': '->', 'color': '#272727', 'lw': 1.8})
        axis.text(.5, annotation_y, f'+{gain:.1f}%', ha='center', va='bottom',
                  fontsize=13, fontweight='bold', color='#272727')
        axis.set(ylim=(0, ymax), xticks=x, xticklabels=['1', '2'], ylabel=ylabel)
        axis.set_xlabel('# CPU Hosts', fontsize=16)
        axis.yaxis.label.set_fontsize(17.6)
        axis.set_title(title, loc='left', fontsize=14, pad=12)
        sns.despine(ax=axis)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .99),
               ncol=2, fontsize=17.6, columnspacing=1, handlelength=1.2, frameon=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(OUTPUT.with_suffix('.' + extension), dpi=300, facecolor='white')
    plt.close(fig)
    with OUTPUT.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assumptions = {
        'price_source': 'User-provided USD scenario rates, not fetched provider quotes.',
        'cpu_host': 'm5d.metal', 'cpu_host_usd_per_hour': cpu_hourly,
        'gpu_price_label': '8 × H100', 'eight_gpu_allocation_usd_per_hour': gpu_hourly,
        'measured_gpu_hardware': '8 × GH200',
        'scope': 'Modeled active configuration during rollout only. '
                 'Charges one or two CPU hosts plus the full eight-GPU allocation concurrently. '
                 'Excludes model initialization, image warmup, inactive allocated hosts, '
                 'provisioning, storage, transfer, and taxes. This is not an experiment billing statement.',
        'hardware_assumption': 'Apply supplied GPU rate to measured GH200 runtimes; '
                               'no H100 throughput equivalence was measured.',
        'source': str(DATA.relative_to(ROOT)), 'source_sha256': digest(DATA), 'rows': rows,
    }
    (ROOT / 'data/cpu_host_pricing_c128_20261019.json').write_text(json.dumps(assumptions, indent=2) + '\n')
    print('Saved', OUTPUT.with_suffix('.png'))
    for row in rows:
        print(f"{row['runtime_count']} CPU hosts: ${row['total_usd_per_hour']:.3f}/h; "
              f"${row['modeled_rollout_usd']:.2f} per measured rollout; "
              f"${row['total_usd_per_1000_usable']:.2f}/1,000 usable trajectories")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-hourly', type=float, default=5.424, help='USD per CPU host per hour')
    parser.add_argument('--gpu-hourly', type=float, default=5.970, help='USD for all eight GPUs per hour')
    args = parser.parse_args()
    main(args.cpu_hourly, args.gpu_hourly)
