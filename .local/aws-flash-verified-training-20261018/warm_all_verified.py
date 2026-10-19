#!/usr/bin/env python3
"""Materialize Verified images with native pulls; never create a sandbox."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace

GIB = 1024 ** 3
CACHE_CAP_BYTES = 800 * GIB
RESERVE_BYTES = 64 * GIB


def atomic_json(path, value):
    path = Path(path)
    previous_stat = path.stat() if path.exists() else None
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        if previous_stat is not None:
            os.fchown(stream.fileno(), previous_stat.st_uid, previous_stat.st_gid)
            os.fchmod(stream.fileno(), previous_stat.st_mode & 0o7777)
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_template(directory, check_size=True):
    """Validate identity, rootfs header/config, and optionally allocation accounting."""
    try:
        if not re.fullmatch('[0-9a-f]{64}', directory.name):
            return None
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata.get('image_id') != directory.name:
            return None
        if not isinstance(metadata.get('image_config'), dict):
            return None
        rootfs = directory / 'scratch-template.ext4'
        stat = rootfs.stat()
        if stat.st_size <= 1081:
            return None
        with rootfs.open('rb') as stream:
            stream.seek(1080)
            if stream.read(2) != b'\x53\xef':
                return None
        size = metadata.get('size_bytes', 0)
        if not isinstance(size, int) or (check_size and size > 0 and size != stat.st_blocks * 512):
            return None
        refs = metadata.get('image_refs', [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            return None
        return metadata
    except (OSError, ValueError, TypeError):
        return None



def valid_template(directory):
    """Return metadata only when the prepared rootfs and allocation size agree."""
    return _read_template(directory, check_size=True)


def refresh_allocations(cache, emit, run=subprocess.run):
    """Reconcile settled XFS CoW accounting only after a read-only clean fsck.

    Call at boundaries with no active warming requests and no running VMs.
    This never repairs rootfs files or accepts a bad filesystem.
    """
    for directory in sorted(Path(cache).iterdir()):
        if not directory.is_dir():
            continue
        metadata = _read_template(directory, check_size=False)
        if metadata is None:
            continue
        rootfs = directory / 'scratch-template.ext4'
        old_size = metadata.get('size_bytes', 0)
        if old_size <= 0 or old_size == rootfs.stat().st_blocks * 512:
            continue
        result = run(['e2fsck', '-f', '-n', str(rootfs)], capture_output=True, text=True)
        if result.returncode != 0:
            emit({'kind': 'allocation_refresh', 'image_id': directory.name, 'ok': False,
                  'old_size_bytes': old_size, 'fsck_returncode': result.returncode,
                  'stdout': result.stdout[-4096:], 'error': result.stderr[-4096:]})
            raise RuntimeError(f'read-only e2fsck failed for {directory.name}: {result.returncode}')
        # Re-read after fsck so unknown fields are retained and identity is still valid.
        current = _read_template(directory, check_size=False)
        if current is None:
            raise RuntimeError(f'template changed during allocation refresh: {directory.name}')
        new_size = rootfs.stat().st_blocks * 512
        current['size_bytes'] = new_size
        atomic_json(directory / 'metadata.json', current)
        emit({'kind': 'allocation_refresh', 'image_id': directory.name, 'ok': True,
              'old_size_bytes': old_size, 'new_size_bytes': new_size,
              'fsck_returncode': result.returncode})


def cache_index(cache):
    by_id, by_ref = {}, {}
    for directory in sorted(Path(cache).iterdir()):
        if not directory.is_dir():
            continue
        metadata = valid_template(directory)
        if metadata is None:
            continue
        by_id[directory.name] = metadata
        refs = [metadata.get('image_ref')] + metadata.get('image_refs', [])
        for ref in refs:
            if isinstance(ref, str) and ref:
                by_ref[ref] = directory.name
    return by_id, by_ref


def docker_rows(run=subprocess.run):
    result = run(['docker', 'image', 'ls', '--digests', '--no-trunc', '--format', '{{json .}}'],
                 check=True, capture_output=True, text=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def docker_refs(rows):
    refs = {}
    for row in rows:
        image_id = row.get('ID', '').removeprefix('sha256:')
        repository = row.get('Repository', '')
        if not re.fullmatch('[0-9a-f]{64}', image_id) or not repository or repository == '<none>':
            continue
        tag, digest = row.get('Tag'), row.get('Digest')
        if tag and tag != '<none>':
            refs[f'{repository}:{tag}'] = image_id
        if digest and digest != '<none>':
            refs[f'{repository}@{digest}'] = image_id
    return refs


def ensure_no_docker_containers(run=subprocess.run):
    result = run(['docker', 'ps', '-aq'], check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError('Docker containers exist; staging reclamation requires zero containers')


def reclaim_prepared(cache, emit, run=subprocess.run, *, image_ids=None, stop_when=None):
    """Only remove named Docker refs whose exact config ID has a valid FC file."""
    ensure_no_docker_containers(run)
    valid_ids, _ = cache_index(cache)
    for ref, image_id in sorted(docker_refs(docker_rows(run)).items()):
        if stop_when is not None and stop_when():
            break
        if image_ids is not None and image_id not in image_ids:
            continue
        if image_id not in valid_ids or valid_template(Path(cache) / image_id) is None:
            continue
        result = run(['docker', 'image', 'rm', '--no-prune', ref],
                     capture_output=True, text=True)
        emit({'kind': 'docker_reclaim', 'ref': ref, 'image_id': image_id,
              'ok': result.returncode == 0, 'returncode': result.returncode,
              'stdout': result.stdout.strip(), 'error': result.stderr.strip()})


def register_alias(cache, image, image_id, emit, run=subprocess.run):
    inspected = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                    check=True, capture_output=True, text=True)
    if inspected.stdout.strip().removeprefix('sha256:') != image_id:
        raise RuntimeError(f'Docker config ID changed for {image}')
    directory = Path(cache) / image_id
    metadata = valid_template(directory)
    if metadata is None:
        raise RuntimeError(f'cannot register alias on invalid template {image_id}')
    old_refs = list(metadata.get('image_refs', []))
    if image in old_refs or metadata.get('image_ref') == image:
        return
    metadata['image_refs'] = list(dict.fromkeys(old_refs + [image]))
    atomic_json(directory / 'metadata.json', metadata)
    emit({'kind': 'alias_registration', 'image': image, 'image_id': image_id,
          'old_refs': old_refs, 'new_refs': metadata['image_refs']})



def persist_pull_aliases(cache, outcomes, aliases, emit, run=subprocess.run):
    """Make every successful pull's exact ref durable before Docker reclamation."""
    _, references = cache_index(cache)
    for result in outcomes:
        if not result['ok']:
            continue
        ref = result['image']
        if aliases and ref in aliases:
            # The initial phase must retain its originally observed ID guard.
            image_id = aliases[ref]
        elif ref not in references:
            inspected = run(['docker', 'image', 'inspect', ref, '--format', '{{.Id}}'],
                            check=True, capture_output=True, text=True)
            image_id = inspected.stdout.strip().removeprefix('sha256:')
        else:
            continue
        register_alias(cache, ref, image_id, emit, run)
        references[ref] = image_id


def volume_stats(path):
    stat = os.statvfs(path)
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    return {'volume_used_bytes': (stat.f_blocks - stat.f_bfree) * stat.f_frsize,
            'volume_free_bytes': free, 'volume_total_bytes': total}


def enough_headroom(stats, cap=CACHE_CAP_BYTES, reserve=RESERVE_BYTES):
    return stats['volume_used_bytes'] < cap - reserve and stats['volume_free_bytes'] >= reserve


def inventory_ready(status, expected, now=None, freshness=120):
    """Exact refs on every available FC worker; aliases never substitute for refs."""
    if status.get('sandbox_count') != 0:
        return False, 'sandbox_count is not zero'
    nodes = [node for node in status.get('nodes', [])
             if node.get('available') and 'firecracker' in node.get('backend_types', [])]
    if not nodes:
        return False, 'no available Firecracker worker'
    now = now or datetime.now(timezone.utc)
    for node in nodes:
        inventory = node.get('image_inventory') or {}
        if not inventory.get('complete') or not inventory.get('physical_size_complete'):
            return False, f'incomplete inventory: {node.get("id")}'
        try:
            refreshed = datetime.fromisoformat(inventory['refreshed_at'].replace('Z', '+00:00'))
            age = (now - refreshed).total_seconds()
            if not -5 <= age <= freshness:
                return False, 'stale inventory'
        except (KeyError, ValueError, TypeError):
            return False, 'inventory timestamp unavailable'
        missing = set(expected) - set(inventory.get('images', []))
        if missing:
            return False, f'{len(missing)} exact references absent from inventory'
    return True, None


def execute(args):
    # The adjacent tool owns the HTTP contract and result decoding. No create APIs used.
    from prewarm_cluster_images import ClusterClient, warm_by_pull

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    expected = list(dict.fromkeys(manifest['images']))
    if len(expected) != 500 or any(not isinstance(ref, str) or not ref for ref in expected):
        raise ValueError('manifest must contain exactly 500 unique image references')
    client = ClusterClient(args.coordinator, 900, args.api_key)
    pull_args = SimpleNamespace(retries=0, retry_delay=0)
    failures = []
    last_image = None
    phase = 'initializing'

    def emit(record):
        record = {'at': datetime.now(timezone.utc).isoformat(), **record}
        with (output / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    def progress(extra=None):
        _, references = cache_index(args.cache)
        record = {'expected': len(expected), 'ready_count': len(set(expected) & set(references)),
                  'phase': phase, 'last_image': last_image, 'failures': failures,
                  **volume_stats(args.volume)}
        record.update(extra or {})
        atomic_json(output / 'progress.json', record)
        return record

    def cleanup(image_ids=None, *, pressure=False):
        refresh_allocations(args.cache, emit)
        if args.prune_prepared_docker:
            reclaim_prepared(args.cache, emit, image_ids=image_ids,
                             stop_when=(lambda: enough_headroom(volume_stats(args.volume)))
                             if pressure else None)

    def check_capacity():
        refresh_allocations(args.cache, emit)
        stats = volume_stats(args.volume)
        if not enough_headroom(stats):
            cleanup(pressure=True)
            stats = volume_stats(args.volume)
        if not enough_headroom(stats):
            raise RuntimeError('capacity blocked: cannot retain 64 GiB headroom below 800 GiB whole-volume cache cap')

    def batch_pull(images, aliases=None):
        nonlocal last_image
        if aliases is None:
            check_capacity()
        outcomes = []
        # Cleanup happens only after the executor has joined every future.
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            pending = {executor.submit(warm_by_pull, client, ref, 'firecracker', 'all', pull_args): ref
                       for ref in images}
            for future in as_completed(pending):
                ref = pending[future]
                try:
                    result = asdict(future.result())
                except Exception as error:
                    result = {'image': ref, 'ok': False, 'error': str(error)}
                outcomes.append(result)
                emit({'kind': 'native_pull', 'phase': phase, **result})
                last_image = ref
                if not result['ok']:
                    failures.append(result)
                progress()
        # Existing-template aliases are not persisted by the eager cache-hit API.
        # Persist only after a successful native pull, and preserve other metadata.
        refresh_allocations(args.cache, emit)
        persist_pull_aliases(args.cache, outcomes, aliases, emit)
        progress()
        if aliases is None:
            _, references = cache_index(args.cache)
            successful_ids = {references[result['image']] for result in outcomes
                              if result['ok'] and result['image'] in references}
            cleanup(image_ids=successful_ids)
        progress()

    try:
        ensure_no_docker_containers()
        status = client.status()
        if status.get('sandbox_count') != 0:
            raise RuntimeError('prewarm requires zero active sandboxes')
        refresh_allocations(args.cache, emit)
        by_id, by_ref = cache_index(args.cache)
        existing = docker_refs(docker_rows())
        aliases = {ref: existing[ref] for ref in expected
                   if ref in existing and existing[ref] in by_id}
        # Register every prepared Docker alias before any Docker removal.
        phase = 'registering_existing_aliases'
        candidates = [ref for ref in expected if ref in aliases and ref not in by_ref]
        for offset in range(0, len(candidates), args.batch_size):
            batch_pull(candidates[offset:offset + args.batch_size], aliases)
        check_capacity()
        for pass_number in (1, 2):
            phase = f'materializing_pass_{pass_number}'
            _, by_ref = cache_index(args.cache)
            missing = [ref for ref in expected if ref not in by_ref]
            if not missing:
                break
            if pass_number == 2:
                # Timeout outcomes may still be completing on the worker.
                time.sleep(5)
                _, by_ref = cache_index(args.cache)
                missing = [ref for ref in missing if ref not in by_ref]
            for offset in range(0, len(missing), args.batch_size):
                batch_pull(missing[offset:offset + args.batch_size])
        phase = 'verifying'
        refresh_allocations(args.cache, emit)
        _, by_ref = cache_index(args.cache)
        missing = [ref for ref in expected if ref not in by_ref]
        inventory_ok, reason = False, 'cache templates missing'
        status = {}
        if not missing:
            deadline = time.monotonic() + 120
            while True:
                status = client.status()
                inventory_ok, reason = inventory_ready(status, expected)
                if inventory_ok or time.monotonic() >= deadline:
                    break
                time.sleep(5)
        ensure_no_docker_containers()
        passed = not missing and inventory_ok
        phase = 'complete' if passed else 'failed'
        summary = progress({'passed': passed, 'missing': missing,
                            'inventory_ready': inventory_ok, 'inventory_error': reason,
                            'coordinator_status': status})
        atomic_json(output / 'summary.json', summary)
        return 0 if passed else 1
    except Exception as error:
        phase = 'blocked' if 'capacity blocked' in str(error) else 'failed'
        summary = progress({'passed': False, 'error': str(error)})
        atomic_json(output / 'summary.json', summary)
        emit({'kind': 'fatal', 'error': str(error)})
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(Path(__file__).with_name('warm-images.json')))
    parser.add_argument('--cache', default='/sandbox/firecracker-cache/images')
    parser.add_argument('--volume', default='/sandbox')
    parser.add_argument('--coordinator', default='http://127.0.0.1:8080')
    parser.add_argument('--output-dir', default=str(Path(__file__).with_name('warming')))
    parser.add_argument('--api-key', default=os.environ.get('FLASH_API_KEY'))
    parser.add_argument('--concurrency', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--batch-size', type=int, choices=(4, 8), default=8)
    parser.add_argument('--prune-prepared-docker', action='store_true')
    raise SystemExit(execute(parser.parse_args()))


if __name__ == '__main__':
    main()
