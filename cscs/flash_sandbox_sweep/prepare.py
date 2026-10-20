#!/usr/bin/env python3
"""Prepare four reviewable Flash points, without contacting Slurm or AWS."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

repository = Path(os.environ.get('REPO_DIR', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(repository))
from cscs.flash_sandbox_sweep.cohort import digest, validate  # noqa: E402


def prepare() -> Path:
    source = repository / 'cscs/flash_sandbox_sweep'
    benchmark = os.environ.get('BENCHMARK', 'swebench-pro-v2')
    if benchmark not in ('swebench-pro-v2', 'swebench-verified'):
        raise ValueError('Unsupported benchmark')
    if os.environ.get('FLASH_SANDBOX_TYPE', 'firecracker') != 'firecracker':
        raise ValueError('Matched comparison requires FLASH_SANDBOX_TYPE=firecracker')
    levels = [int(value) for value in os.environ.get('SWEEP_LEVELS', '8 16 32 64').split()]
    if not levels or len(set(levels)) != len(levels) or any(v not in (8, 16, 32, 64) for v in levels):
        raise ValueError('SWEEP_LEVELS must be unique group counts from 8,16,32,64')
    pin_path = source / f'cohort_{benchmark}.json'
    pin = json.loads(pin_path.read_text())
    data = Path(pin['dataset'])
    if os.environ.get('DATA_PATH', str(data)) != str(data):
        raise ValueError('DATA_PATH override differs from the pinned baseline')
    model = Path(os.environ['MODEL_PATH'])
    secret = Path(os.environ['FLASH_SECRET_FILE']).expanduser().resolve()
    mode = secret.stat().st_mode
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600 or secret.stat().st_uid != os.getuid():
        raise ValueError('FLASH_SECRET_FILE must be a mode-0600 regular file owned by this user')
    sdk = Path(os.environ['FLASH_SDK_PATH']).expanduser().resolve()
    for package in ('flash_sandbox', 'httpx2'):
        if not (sdk / package / '__init__.py').is_file():
            raise ValueError('FLASH_SDK_PATH must contain isolated flash_sandbox and httpx2 packages')
    sdk_hashes = {str(p.relative_to(sdk)): digest(p) for p in sdk.rglob('*')
                  if p.is_file() and (p.suffix == '.py' or p.name == 'METADATA')}
    proof = validate(pin_path, data, model)
    root = Path(os.environ['SWEEP_DIR']).expanduser().resolve()
    root.mkdir(mode=0o700)  # Refuse to overwrite an existing experiment.
    (root / 'cohort-validation.json').write_text(json.dumps(proof, indent=2) + '\n')
    (root / 'sweep-plan.json').write_text(json.dumps({
        'backend': 'flash-sandbox', 'benchmark': benchmark, 'levels': levels,
        'episode_concurrency': [v * 8 for v in levels], 'conditional_128': None,
        'time_limit': '00:45:00', 'groups': 128, 'responses': 8, 'seed': 42,
        'aws_stop_allowed': False, 'host_creation_allowed': False,
    }, indent=2) + '\n')
    tracked = [source / name for name in ('run_point.py', 'environment.py', 'cohort.py', 'client.py')]
    tracked += [repository / 'cscs/sandbox_sweep' / name for name in ('run_point.py', 'group_framework.py')]
    hashes = {str(path.relative_to(repository)): digest(path) for path in tracked}
    previous = ''
    for level in levels:
        run = root / f'groups-{level}'
        run.mkdir(mode=0o700)
        for name in ('telemetry/model-calls', 'telemetry/gpu', 'flash-traces', 'analysis-tools'):
            (run / name).mkdir(parents=True, exist_ok=True)
        for name in ('run_point.py', 'node.sh', 'cleanup_sandboxes.py',
                     'backend_preflight.py', 'harbor-flash', 'client.py'):
            shutil.copyfile(source / name, run / name)
        (run / 'harbor-flash').chmod(0o700)
        shutil.copyfile(pin_path, run / 'cohort.json')
        (run / 'cohort-frozen.json').write_text(json.dumps(proof, indent=2) + '\n')
        for original, target in [('cscs/native_agentenv/bind_cpu_memory.sh', 'bind_cpu_memory.sh'),
                                 ('cscs/observability/collect.py', 'collect.py'),
                                 ('cscs/observability/export_metrics.py', 'analysis-tools/export_metrics.py')]:
            shutil.copyfile(repository / original, run / target)
        subprocess.run([sys.executable, str(repository / 'cscs/observability/bundle_dcgm.py'),
                        '--output', str(run / 'dcgm-client')], check=True)
        config = source / ('task_config_verified.yaml' if benchmark == 'swebench-verified' else 'task_config_pro.yaml')
        (run / 'task_config.yaml').write_text(config.read_text().replace('artifacts_dir: harbor',
                                              'artifacts_dir: ' + json.dumps(str(run / 'harbor'))).replace(
                                                  'harbor_executable: harbor-flash',
                                                  'harbor_executable: ' + json.dumps(str(run / 'harbor-flash'))))
        values = {'REPO_DIR': str(repository), 'SCRATCH': os.environ['SCRATCH'],
                  'MODEL_PATH': str(model), 'DATA_PATH': str(data), 'BENCHMARK': benchmark,
                  'FLASH_SECRET_FILE': str(secret), 'FLASH_SDK_PATH': str(sdk),
                  'FLASH_SANDBOX_TYPE': os.environ.get('FLASH_SANDBOX_TYPE', 'firecracker'),
                  'HARBOR_VENV': os.environ.get('HARBOR_VENV', os.environ['SCRATCH'] + '/venvs/uni-agentenv-v2'),
                  'RUN_DIR': str(run), 'GROUP_CONCURRENCY': str(level), 'PREVIOUS_RUN_DIR': previous,
                  'AGENT_LOG_DIR': str(run / 'sessions'), 'NNODES_ROLLOUT': '2',
                  'AGENTENV_UV_ARCHIVE': pin.get('native_bootstrap', {}).get('asset_path', ''),
                  'AGENTENV_UV_ARCHIVE_SHA256': pin.get('native_bootstrap', {}).get('sha256', ''),
                  'TRAINING_STEPS': '0', 'TIME_LIMIT': '00:45:00', 'SWEEP_DIAGNOSTICS': 'False'}
        if os.environ.get('FLASH_SANDBOX_ENDPOINT'):
            values['FLASH_SANDBOX_ENDPOINT'] = os.environ['FLASH_SANDBOX_ENDPOINT']
        launch = run / 'launch.env'
        launch.write_text(''.join(f'{name}={shlex.quote(value)}\n' for name, value in values.items()))
        launch.chmod(0o600)
        shutil.copyfile(source / 'batch.sh', run / 'launcher.sh')
        (run / 'sdk-hashes.json').write_text(json.dumps(sdk_hashes, indent=2) + '\n')
        (run / 'source-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
        previous = str(run)
    return root


if __name__ == '__main__':
    print(prepare())
