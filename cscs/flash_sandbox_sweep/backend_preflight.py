#!/usr/bin/env python3
"""Check authenticated public SDK cluster availability without creating a VM."""
import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path


def validate_sdk() -> None:
    sdk = Path(os.environ['FLASH_SDK_PATH']).resolve()
    manifest = Path(os.environ['RUN_DIR']) / 'sdk-hashes.json'
    for relative, expected in json.loads(manifest.read_text()).items():
        if hashlib.sha256((sdk / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Staged SDK changed after preparation')
    for name in ('flash_sandbox', 'httpx2'):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(sdk):
            raise RuntimeError('Flash SDK dependency did not resolve from the isolated target')


async def check() -> dict:
    from flash_sandbox import AsyncHTTPClient
    client = AsyncHTTPClient(os.environ['FLASH_SANDBOX_ENDPOINT'],
                             api_key=os.environ['OTELA_API_KEY'], timeout=30)
    try:
        status = await client.cluster_status()
        backend = os.environ.get('FLASH_SANDBOX_TYPE', 'firecracker')
        healthy = [node for node in status.nodes if node.available and backend in node.backend_types]
        if (status.node_count, status.available_node_count, status.unavailable_node_count,
            len(status.nodes), len(healthy), status.sandbox_count) != (1, 1, 0, 1, 1, 0):
            raise RuntimeError('Matched Flash comparison requires one available Firecracker node and zero sandboxes')
        if healthy[0].running_count != 0:
            raise RuntimeError('Flash node still has running sandboxes')
        return {'state': 'available', 'node_count': status.node_count,
                'available_node_count': status.available_node_count,
                'unavailable_node_count': status.unavailable_node_count,
                'sandbox_count': status.sandbox_count, 'backend': backend,
                'available_backend_nodes': len(healthy)}
    finally:
        await client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_sdk()
        result = asyncio.run(check())
    except Exception as exc:
        # Exception text can contain the private endpoint or HTTP headers.
        args.output.write_text(json.dumps({'state': 'failed', 'error_class': type(exc).__name__}) + '\n')
        raise SystemExit('Flash SDK preflight failed; see sanitized preflight report') from None
    args.output.write_text(json.dumps(result, indent=2) + '\n')
