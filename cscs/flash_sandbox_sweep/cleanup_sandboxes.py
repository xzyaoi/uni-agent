#!/usr/bin/env python3
"""Stop only IDs durably recorded by one Flash sweep point."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from cscs.flash_sandbox_sweep.client import SynchronousDeleteClient


def owned_ids(run: Path, endpoint: str) -> set[str]:
    expected = hashlib.sha256(endpoint.rstrip('/').encode()).hexdigest()
    ids = set()
    for path in (run / 'flash-traces').glob('owned-*.jsonl'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # An interrupted final write cannot authorize deletion.
            if (row.get('event') in {'sandbox_claimed', 'sandbox_created'} and row.get('run_id') == str(run.resolve())
                    and row.get('endpoint_sha256') == expected and isinstance(row.get('sandbox_id'), str)
                    and row['sandbox_id']):
                ids.add(row['sandbox_id'])
    return ids


async def cleanup(run: Path, report_file: Path | None = None) -> dict:
    from flash_sandbox import SandboxNotFoundError
    endpoint = os.environ['FLASH_SANDBOX_ENDPOINT']
    ids = owned_ids(run, endpoint)
    counts = {'recorded_ids': len(ids), 'deleted': 0, 'already_gone': 0, 'errors': 0}
    client = SynchronousDeleteClient(endpoint, api_key=os.environ['OTELA_API_KEY'], timeout=150)
    try:
        for sid in sorted(ids):
            try:
                await client.stop_sandbox(sid, cleanup=True, timeout=120)
                counts['deleted'] += 1
            except SandboxNotFoundError:
                counts['already_gone'] += 1
            except Exception:
                counts['errors'] += 1
    finally:
        await client.close()
    (report_file or run / 'sandbox-cleanup.json').write_text(json.dumps(counts, indent=2) + '\n')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--report-file', type=Path, help='Write a predecessor cleanup report into the current run')
    args = parser.parse_args()
    result = asyncio.run(cleanup(args.run_dir.resolve(), args.report_file))
    print(json.dumps(result))
    raise SystemExit(bool(result['errors']))
