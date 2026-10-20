"""Read-only validation of the exact ordered AgentENV comparison cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import tomllib


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows, pin: dict) -> list[dict]:
    if len(rows) != pin['rows']:
        raise ValueError('Dataset row count changed')
    indices = list(range(len(rows)))
    random.Random(pin['seed']).shuffle(indices)
    identities = []
    for index in indices[:pin['groups']]:
        metadata = rows[index]['extra_info']['tools_kwargs']['task']['metadata']
        task = Path(metadata['task_path']) / 'task.toml'
        config = tomllib.loads(task.read_text())
        image = config['environment']['docker_image']
        if '@sha256:' not in image:
            raise ValueError('Selected task image is not pinned to a digest')
        if (task.parent / 'environment' / 'docker-compose.yaml').exists() or (
            task.parent / 'environment' / 'docker-compose.yml'
        ).exists():
            raise ValueError('Flash does not support Docker Compose tasks')
        for relative, expected in pin.get('selected_file_sha256', {}).get(metadata['instance_id'], {}).items():
            if digest(task.parent / relative) != expected:
                raise ValueError('Official verifier or instruction file changed')
        identities.append({'dataset_index': index, 'instance_id': metadata['instance_id'],
                           'task_toml_sha256': digest(task)})
    if identities != pin['selected_tasks']:
        raise ValueError('Selected task identity, row order or TOML digest changed')
    selection = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    if selection != pin['selection_sha256']:
        raise ValueError('Selection digest changed')
    return identities


def validate(pin_path: Path, data_path: Path, model_path: Path | None = None) -> dict:
    pin = json.loads(pin_path.read_text())
    if (pin['groups'], pin['responses'], pin['seed']) != (128, 8, 42):
        raise ValueError('Comparison requires 128 groups x 8 responses and seed 42')
    if str(data_path.resolve()) != pin['dataset'] or digest(data_path) != pin['dataset_sha256']:
        raise ValueError('Parquet path or digest differs from the matched baseline')
    if model_path is not None and digest(model_path / 'config.json') != pin['model_config_sha256']:
        raise ValueError('Model config differs from the matched baseline')
    import pyarrow.parquet as pq
    rows = pq.read_table(data_path).to_pylist()
    identities = validate_rows(rows, pin)
    files = {}
    for identity in identities:
        metadata = rows[identity['dataset_index']]['extra_info']['tools_kwargs']['task']['metadata']
        root = Path(metadata['task_path'])
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[str(path)] = {'sha256': digest(path), 'mode': path.stat().st_mode & 0o777}
    bootstrap = pin.get('native_bootstrap')
    if bootstrap and digest(Path(bootstrap['asset_path'])) != bootstrap['sha256']:
        raise ValueError('Pinned baseline uv archive changed')
    return {'state': 'matched_cohort_validated', 'dataset': str(data_path),
            'dataset_sha256': pin['dataset_sha256'], 'rows': len(rows),
            'groups': 128, 'responses': 8, 'seed': 42,
            'selection_sha256': pin['selection_sha256'], 'selected_tasks': identities,
            'pin_sha256': digest(pin_path), 'task_files': files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.pin, args.data_path, args.model_path), indent=2) + '\n')
