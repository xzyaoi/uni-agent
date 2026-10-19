#!/usr/bin/env python3
"""Record the exact prior Pro cohort and images without modifying task definitions."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tomllib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).to_pylist()
    baseline = json.loads(args.baseline_manifest.read_text())
    indices = list(range(len(rows)))
    random.Random(baseline['seed']).shuffle(indices)
    indices = indices[:baseline['groups']]
    selected = []
    for index in indices:
        metadata = rows[index]['extra_info']['tools_kwargs']['task']['metadata']
        task = Path(metadata['task_path']) / 'task.toml'
        definition = tomllib.loads(task.read_text())
        selected.append({'dataset_index': index, 'instance_id': metadata['instance_id'],
                         'task_path': str(task.parent), 'task_toml_sha256': sha(task),
                         'image': definition['environment']['docker_image']})
    # An accidental cohort/data change must not silently become a repeat.
    fields = ('dataset_index', 'instance_id', 'task_toml_sha256')
    if [[item[k] for k in fields] for item in selected] != [[item[k] for k in fields] for item in baseline['selected_tasks']]:
        raise ValueError('Selected Pro tasks differ from the recorded baseline')
    result = {'seed': baseline['seed'], 'groups': baseline['groups'],
              'responses': baseline['responses'], 'dataset': str(args.data),
              'dataset_sha256': sha(args.data), 'selected_tasks': selected,
              'baseline_manifest_sha256': sha(args.baseline_manifest),
              'images': [{'original_ref': image} for image in sorted({s['image'] for s in selected})],
              'warmup_policy': 'Resolve once to immutable digests; warm every selected image on all four runtimes before GPU submission',
              'gpu_launch_allowed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'selected_tasks': len(selected), 'unique_images': len(result['images']),
                      'baseline_matches': True, 'manifest': str(args.output)}))


if __name__ == '__main__':
    main()
