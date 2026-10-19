#!/usr/bin/env python3
"""Submit a sequential sweep as debug's per-user submission slots become free."""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('sweep_dir', type=Path)
parser.add_argument('--once', action='store_true')
args = parser.parse_args()
root = args.sweep_dir.resolve()


def deadline_reached(plan):
    deadline = plan.get('submission_deadline_epoch')
    if deadline is None or time.time() < deadline:
        return False
    state = {'state': 'submission_deadline_reached', 'timestamp': time.time(),
             'submission_deadline_epoch': deadline,
             'unsubmitted_levels': [level for level in plan['levels']
                                    if not (root / ('groups-%d' % level) / 'job_id').exists()]}
    temporary = root / 'submission-deadline.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(root / 'submission-deadline.json')
    print('Submission deadline reached; retaining submitted jobs and all sweep evidence', flush=True)
    return True


while True:
    plan_file = root / 'sweep-plan.json'
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {'levels': [8, 16, 32, 64]}
    levels = plan['levels']
    previous = None
    blocked = False
    expired = False
    for level in levels:
        run = root / ('groups-%d' % level)
        job_file = run / 'job_id'
        if not job_file.exists() and deadline_reached(plan):
            expired = True
            break
        if not job_file.exists() and previous:
            state = subprocess.run(['squeue', '-h', '-j', previous, '-o', '%T'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if state.stdout.strip():
                blocked = True
                break
            previous_result = root / ('groups-%d' % levels[levels.index(level) - 1]) / 'result.json'
            if not previous_result.exists():
                raise RuntimeError('Previous point has no result; inspect setup before continuing')
            outcome = json.loads(previous_result.read_text())
            if outcome.get('state') != 'completed' and not outcome.get('progress'):
                raise RuntimeError('Previous point failed before rollout; refusing to repeat setup failure')
            if level == 128 and plan.get('conditional_128', 'requires clean 64-group completion'):
                progress = outcome.get('progress') or {}
                harbor = outcome.get('harbor') or progress.get('harbor') or {}
                accounting = subprocess.run(['sacct', '-X', '-n', '-P', '-j', previous, '--format=State'],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if accounting.returncode or not accounting.stdout.strip():
                    blocked = True
                    break
                healthy = (outcome.get('state') == 'completed'
                           and progress.get('groups_terminal') == 128
                           and progress.get('groups_failed', 0) == 0
                           and progress.get('episodes_with_usable_trajectory') == 1024
                           and harbor.get('evaluation_completed') == 1024
                           and harbor.get('evaluation_incomplete', 0) == 0
                           and accounting.stdout.strip().splitlines()[0].split('|')[0] == 'COMPLETED')
                if not healthy:
                    skip = {'state': 'skipped', 'gated_by_groups': 64,
                            'reason': '64-group point did not complete cleanly with 1024 verified and usable episodes',
                            'predecessor_job': previous, 'predecessor_progress': progress,
                            'predecessor_scheduler_state': accounting.stdout.strip()}
                    (run / 'skipped.json').write_text(json.dumps(skip, indent=2) + '\n')
                    (run / 'result.json').write_text(json.dumps(skip, indent=2) + '\n')
                    print('Skipped groups=128: ' + skip['reason'], flush=True)
                    continue
        if not job_file.exists():
            if deadline_reached(plan):
                expired = True
                break
            command = ['sbatch', '--parsable', '--job-name=swe9b-g%d-128groups' % level,
                       '--output=' + str(run / 'rollout.out'), '--export=ALL']
            if previous:
                command.append('--dependency=afterany:' + previous)
            result = subprocess.run(command + [str(run / 'launcher.sh')], cwd=str(run),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode:
                if 'QOSMaxSubmitJobPerUserLimit' not in result.stderr:
                    raise RuntimeError(result.stderr)
                blocked = True
                break
            job_file.write_text(result.stdout.strip() + '\n')
            print('Submitted groups=%d job=%s' % (level, result.stdout.strip()), flush=True)
        previous = job_file.read_text().strip()
    # Rebuild from durable job IDs, including points beyond a deadline break.
    submitted = ['%d\t%s\t%s\n' % (level, (root / ('groups-%d' % level) / 'job_id').read_text().strip(), root / ('groups-%d' % level))
                 for level in levels if (root / ('groups-%d' % level) / 'job_id').exists()]
    temporary = root / 'jobs.tsv.tmp'
    temporary.write_text(''.join(submitted))
    temporary.replace(root / 'jobs.tsv')
    if expired or args.once or not blocked:
        break
    time.sleep(30)
