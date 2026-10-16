#!/usr/bin/env python3
"""Require completed native image warmup and pin the measured Verified task images."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset-root', type=Path, required=True)
parser.add_argument('--warmup-report', type=Path, required=True)
args = parser.parse_args()
root = args.dataset_root.resolve()
manifest = json.loads((root / 'manifest.json').read_text())
warm = json.loads(args.warmup_report.read_text())
if not warm.get('finished_at') or warm.get('failed_images') or warm.get('ready_images') != len(manifest['selected_images']):
    raise RuntimeError('Selected image warmup is incomplete or failed')
resolved = {r['original_ref']: r['resolved_ref'] for r in warm['results']
            if r.get('status') == 'ready' and r.get('cleanup') == 'deleted'}
if set(resolved) != set(manifest['selected_images']):
    raise RuntimeError('Warmup images do not match the selected Verified cohort')
readiness = json.loads((root / 'provenance/native-tools-readiness/result.json').read_text())
if readiness.get('return_code') != 0 or 'uv 0.7.13' not in readiness.get('stdout', ''):
    raise RuntimeError('Required native uv tool has not been prepared successfully')
asset = manifest['native_bootstrap']
if hashlib.sha256(Path(asset['asset_path']).read_bytes()).hexdigest() != asset['sha256']:
    raise RuntimeError('Native tool asset checksum changed')
for item in manifest['selected_tasks']:
    task = Path(item['task_path'])
    if hashlib.sha256((task / 'tests/test.sh').read_bytes()).hexdigest() != item['verifier_sha256']:
        raise RuntimeError('Official verifier was modified: ' + str(task))
    p = task / 'task.toml'
    old = item['image'];new = resolved[old]
    source = p.read_text()
    if new not in source:
        if source.count(old) != 1:
            raise RuntimeError('Unexpected task image field: ' + str(p))
        p.write_text(source.replace(old, new, 1))
    item['resolved_image'] = new
    item['measured_task_toml_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
manifest['uv_provisioning_verified'] = True
manifest['warmup_report_sha256'] = hashlib.sha256(args.warmup_report.read_bytes()).hexdigest()
manifest['measured_images'] = sorted(resolved.values())
manifest['actual_sandbox_shape'] = {'cpu_count': 1, 'memory_mb': 4096}
(root / 'measurement-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Pinned %d ready images; official verifier hashes preserved' % len(resolved))
