"""Plot the final Verified sweep from frozen results; no remote access required."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / 'data/verified_sweep_20261016.json').read_text())
    complete = [r for r in rows if r['state'] == 'completed']
    x = [r['active_groups'] for r in complete]
    plt.rcParams.update({'font.family': ['Roboto', 'DejaVu Sans'], 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    throughput, duration, activity, outcomes = axes.flat
    for key, label in [('completed_episodes_per_minute', 'Terminal attempts'),
                       ('verified_episodes_per_minute', 'Reward verification completed'),
                       ('usable_episodes_per_minute', 'Usable trajectories')]:
        throughput.plot(x, [r[key] for r in complete], 'o-', label=label)
    throughput.set(title='Throughput after model initialization', ylabel='Episodes / minute', xticks=x)
    throughput.legend(fontsize=8)
    duration.bar([str(v) for v in x], [r['wall_seconds'] / 60 for r in complete], color='#3775BA')
    duration.set(title='Completed batch duration', ylabel='Minutes for 1,024 attempts')
    for key, label in [('sandbox_cpu_busy_pct_mean', 'Sandbox CPU busy'),
                       ('rollout_gpu_busy_pct_mean', 'GPU busy'),
                       ('rollout_sm_active_pct_mean', 'GPU SM active')]:
        activity.plot(x, [r[key] for r in complete], 'o-', label=label)
    activity.set(title='Distinct CPU and GPU counter definitions', ylabel='Mean activity (%)', ylim=(0, 100), xticks=x)
    activity.legend(fontsize=8)
    positions = list(range(len(x)))
    outcomes.bar([v - .2 for v in positions], [r['incomplete_verifications'] for r in complete], .4,
                 label='Incomplete verification', color='#9A4D8E')
    outcomes.bar([v + .2 for v in positions], [1024 - r['usable_episodes'] for r in complete], .4,
                 label='Missing usable trajectory', color='#B64342')
    outcomes.set_xticks(positions, [str(v) for v in x])
    outcomes.set(title='Incomplete or missing results', ylabel='Episodes (categories may overlap)')
    outcomes.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Active groups (8 episodes each)')
        ax.grid(axis='y', alpha=.25)
    fig.suptitle('Qwen3.5-9B / SWE-bench Verified / native AgentENV')
    fig.text(.5, .02, 'Same 128 tasks × 8 episodes; 8 GH200 rollout GPUs. One run per point; 128 groups skipped.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    output = ROOT / 'figure/verified_sweep_overview'
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=220, facecolor='white')
    plt.close(fig)
    print('Saved', output.with_suffix('.svg'))


if __name__ == '__main__':
    main()
