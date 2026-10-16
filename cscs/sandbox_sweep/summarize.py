#!/usr/bin/env python3
"""Summarize completed sweep points; never treat partial runs as full throughput."""
import argparse
import csv
import datetime as dt
import json
import statistics
import time
from pathlib import Path


def summarize(root):
    rows = []
    plan_file = root / 'sweep-plan.json'
    levels = json.loads(plan_file.read_text())['levels'] if plan_file.exists() else [8, 16, 32, 64]
    for limit in levels:
        point = root / ('groups-%d' % limit)
        path = point / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        progress_path = point / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else (result.get('progress') or {})
        row = {'active_groups': limit, 'episode_ceiling': limit * 8,
               'state': result.get('state', 'pending'), 'benchmark': result.get('benchmark'),
               'wall_seconds': result.get('rollout_wall_seconds', progress.get('rollout_elapsed_seconds')),
               'groups_terminal': progress.get('groups_terminal', 0)}
        harbor = result.get('harbor') or progress.get('harbor') or {}
        row['harbor_counts'] = harbor
        row['completed_episodes'] = harbor.get('terminal_results', 0)
        row['verified_episodes'] = harbor.get('evaluation_completed', 0)
        row['incomplete_verifications'] = harbor.get('evaluation_incomplete', 0)
        row['usable_episodes'] = progress.get('episodes_with_usable_trajectory', 0)
        if row['wall_seconds'] and row['wall_seconds'] > 0:
            minutes = row['wall_seconds'] / 60
            row['completed_episodes_per_minute'] = row['completed_episodes'] / minutes
            row['verified_episodes_per_minute'] = row['verified_episodes'] / minutes
            row['usable_episodes_per_minute'] = row['usable_episodes'] / minutes
        if row['state'] == 'completed' and row['groups_terminal'] == 128 and row['wall_seconds']:
            row['terminal_groups_per_hour'] = 128 * 3600 / row['wall_seconds']
            row['usable_episodes'] = result['progress'].get('episodes_with_usable_trajectory', 0)
            row['usable_episodes_per_hour'] = row['usable_episodes'] * 3600 / row['wall_seconds']
        start, end = result.get('model_ready_at'), result.get('finished_at', progress.get('timestamp'))
        metrics = point / 'metrics_1min.csv'
        if start and end and metrics.exists():
            with metrics.open() as stream:
                bins = [r for r in csv.DictReader(stream) if start <= float(r['epoch']) and float(r['epoch']) + 60 <= end]
            for key in ('sandbox_cpu_busy_pct', 'sandbox_cpu_psi_pct', 'sandbox_vm_count',
                        'sandbox_ram_available_gib', 'rollout_gpu_busy_pct', 'rollout_sm_active_pct',
                        'rollout_tensor_active_pct', 'rollout_vllm_active_requests'):
                values = [float(r[key]) for r in bins if r.get(key)]
                if values:
                    row[key + '_mean'] = statistics.mean(values)
                    row[key + '_max_1min_mean'] = max(values)
        rows.append(row)
    (root / 'sweep_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    fields = ['active_groups', 'episode_ceiling', 'state', 'groups_terminal', 'wall_seconds',
              'terminal_groups_per_hour', 'completed_episodes', 'completed_episodes_per_minute',
              'verified_episodes', 'verified_episodes_per_minute', 'incomplete_verifications',
              'usable_episodes', 'usable_episodes_per_minute', 'usable_episodes_per_hour',
              'sandbox_cpu_busy_pct_mean', 'rollout_gpu_busy_pct_mean', 'rollout_sm_active_pct_mean']
    with (root / 'sweep_results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    text = '# Sweep progress\n\nUpdated: ' + dt.datetime.now(dt.timezone.utc).isoformat() + '\n\n'
    text += '| Active groups | Episode ceiling | State | Terminal groups | Rollout minutes | Episodes/min | Verified/min |\n|---:|---:|---|---:|---:|---:|---:|\n'
    for row in rows:
        minutes = '—' if row['wall_seconds'] is None else '%.1f' % (row['wall_seconds'] / 60)
        rates = ['—' if key not in row else '%.1f' % row[key] for key in ('completed_episodes_per_minute', 'verified_episodes_per_minute')]
        text += '| %s | %s | %s | %s / 128 | %s | %s | %s |\n' % (row['active_groups'], row['episode_ceiling'], row['state'], row['groups_terminal'], minutes, *rates)
    text += '\nEpisodes/min counts all terminal Harbor attempts, including errors; verified/min counts completed reward verification. Rates use elapsed rollout time excluding model initialization and remain available for running or interrupted points. Unfinished episodes are excluded from the numerator. Full-batch time is available only when all 128 groups finish. Utilization uses complete one-minute bins after model readiness; maxima of those bins are not instantaneous peaks.\n'
    (root / 'SWEEP_STATUS.md').write_text(text)
    finished = [row for row in rows if 'completed_episodes_per_minute' in row]
    if finished:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        x = [row['active_groups'] for row in finished]
        axes[0].plot(x, [row['completed_episodes_per_minute'] for row in finished], 'o-')
        axes[0].set_ylabel('Terminal episode attempts / minute')
        axes[1].plot(x, [row['verified_episodes_per_minute'] for row in finished], 'o-')
        axes[1].set_ylabel('Verified episodes / minute')
        for key, label in [('sandbox_cpu_busy_pct_mean', 'Sandbox CPU busy'), ('rollout_sm_active_pct_mean', 'GPU SM active')]:
            points = [row for row in finished if key in row]
            if points:
                axes[2].plot([r['active_groups'] for r in points], [r[key] for r in points], 'o-', label=label)
        axes[2].set_ylabel('Activity (%) — different counter definitions')
        if axes[2].lines:
            axes[2].legend()
        for ax in axes:
            ax.set_xlabel('Active groups (8 episodes each)')
            ax.set_xticks(levels)
            ax.grid(alpha=.25)
        benchmark = next((r.get("benchmark") for r in rows if r.get("benchmark")), "SWE-bench")
        fig.suptitle("Qwen3.5-9B / " + benchmark + " / AgentENV — observed rates")
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(root / ('sweep_throughput.' + suffix), dpi=160)
        plt.close(fig)

        # Include outcome quality alongside throughput so terminal failures do
        # not look like an improvement in capacity.
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        throughput, duration, activity, outcomes = axes.flat
        for key, label in [('completed_episodes_per_minute', 'Terminal attempts'),
                           ('verified_episodes_per_minute', 'Reward verification completed'),
                           ('usable_episodes_per_minute', 'Usable trajectories')]:
            throughput.plot(x, [r.get(key, float('nan')) for r in finished], 'o-', label=label)
        throughput.set_ylabel('Episodes / minute')
        throughput.set_title('Throughput after model initialization')
        throughput.legend(fontsize=8)
        complete = [r for r in finished if r['state'] == 'completed' and r['groups_terminal'] == 128]
        duration.bar([str(r['active_groups']) for r in complete], [r['wall_seconds'] / 60 for r in complete], color='#4878a8')
        duration.set_ylabel('Minutes for 1,024 attempts')
        duration.set_title('Completed batch duration')
        for key, label in [('sandbox_cpu_busy_pct_mean', 'Sandbox CPU busy'),
                           ('rollout_gpu_busy_pct_mean', 'GPU busy'),
                           ('rollout_sm_active_pct_mean', 'GPU SM active')]:
            points = [r for r in finished if key in r]
            activity.plot([r['active_groups'] for r in points], [r[key] for r in points], 'o-', label=label)
        activity.set_ylim(0, 100)
        activity.set_ylabel('Mean activity (%)')
        activity.set_title('Distinct CPU and GPU counter definitions')
        activity.legend(fontsize=8)
        positions = list(range(len(complete)))
        outcomes.bar([v - .2 for v in positions], [r['incomplete_verifications'] for r in complete], .4, label='Incomplete verification', color='#d57a40')
        outcomes.bar([v + .2 for v in positions], [1024 - r['usable_episodes'] for r in complete], .4, label='Missing usable trajectory', color='#b54b62')
        outcomes.set_xticks(positions, [str(r['active_groups']) for r in complete])
        outcomes.set_ylabel('Episodes (categories may overlap)')
        outcomes.set_title('Incomplete or missing results')
        outcomes.legend(fontsize=8)
        for ax in (throughput, activity):
            ax.set_xticks([r['active_groups'] for r in finished])
        for ax in axes.flat:
            ax.set_xlabel('Active groups (8 episodes each)')
            ax.grid(axis='y', alpha=.25)
        skipped = [str(r['active_groups']) for r in rows if r['state'] == 'skipped']
        footnote = 'Same 128 tasks × 8 episodes; two GH200 nodes / 8 rollout GPUs. One run per point; no error bars.'
        if skipped:
            footnote += ' Skipped: ' + ', '.join(skipped) + ' groups.'
        fig.suptitle('Qwen3.5-9B / ' + benchmark + ' / native AgentENV')
        fig.text(.5, .015, footnote, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .96))
        for suffix in ('png', 'pdf'):
            fig.savefig(root / ('sweep_overview.' + suffix), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--watch-seconds', type=int, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.watch_seconds
    while True:
        summarize(args.study)
        if time.monotonic() >= deadline:
            break
        time.sleep(60)
