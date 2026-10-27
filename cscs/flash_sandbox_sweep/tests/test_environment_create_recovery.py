"""The benchmark must record and preserve the provider's logical create fence."""
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'harbor.environments.base',
                        SimpleNamespace(BaseEnvironment=object, ExecResult=SimpleNamespace))
    from cscs.flash_sandbox_sweep import environment
    monkeypatch.setenv('FLASH_SWEEP_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('FLASH_SWEEP_RUN_ID', 'fixture-run')
    client = object.__new__(environment.OwnedShapeClient)
    client._sweep_endpoint = 'http://fixture'
    fields = dict(sandbox_id='owned-sid', operation_id='owned-operation',
                  infrastructure_deadline_unix_ms=2000000000000,
                  type='firecracker', cpu_cores=1, memory_mb=4096)

    def records():
        return [json.loads(line) for path in tmp_path.glob('owned-*.jsonl') for line in path.read_text().splitlines()]

    return environment, client, fields, records


def test_wrapper_honors_provider_fence_and_records_before_dispatch(wrapper, monkeypatch):
    module, client, fields, records = wrapper
    seen = []

    async def create(self, *, sandbox_id, operation_id, infrastructure_deadline_unix_ms, **kwargs):
        assert records()[0]['event'] == 'sandbox_claimed'
        assert records()[0]['sandbox_id'] == sandbox_id == 'owned-sid'
        assert records()[0]['operation_id'] == operation_id == 'owned-operation'
        assert records()[0]['infrastructure_deadline_unix_ms'] == infrastructure_deadline_unix_ms
        assert kwargs['vcpu_count'] == 1
        seen.append(sandbox_id)
        return SimpleNamespace(id=sandbox_id)

    monkeypatch.setattr(module.OwnedShapeClient.__bases__[0], 'start_sandbox', create)
    assert asyncio.run(client.start_sandbox(**fields)).id == 'owned-sid'
    assert seen == ['owned-sid']
    assert [r['event'] for r in records()] == ['sandbox_claimed', 'sandbox_created']


def test_lost_reply_preserves_claim_with_operation_and_deadline(wrapper, monkeypatch):
    module, client, fields, records = wrapper

    async def create(self, *, sandbox_id, operation_id, infrastructure_deadline_unix_ms, **kwargs):
        raise TimeoutError('lost create reply')

    monkeypatch.setattr(module.OwnedShapeClient.__bases__[0], 'start_sandbox', create)
    with pytest.raises(TimeoutError):
        asyncio.run(client.start_sandbox(**fields))
    assert len(records()) == 1
    assert records()[0]['sandbox_id'] == fields['sandbox_id']
    assert records()[0]['operation_id'] == fields['operation_id']
    assert records()[0]['infrastructure_deadline_unix_ms'] == fields['infrastructure_deadline_unix_ms']


def test_wrapper_rejects_unfenced_request_and_old_sdk(wrapper, monkeypatch):
    module, client, fields, records = wrapper
    with pytest.raises(ValueError, match='stable sandbox ID'):
        asyncio.run(client.start_sandbox(type='firecracker', cpu_cores=1, memory_mb=4096))

    async def old_create(self, **kwargs):
        raise AssertionError('old SDK must not dispatch or silently discard fields')

    monkeypatch.setattr(module.OwnedShapeClient.__bases__[0], 'start_sandbox', old_create)
    with pytest.raises(RuntimeError, match='SDK'):
        asyncio.run(client.start_sandbox(**fields))
    assert records() == []


def test_mismatched_reply_never_authorizes_deleting_another_sandbox(wrapper, monkeypatch):
    module, client, fields, records = wrapper

    async def create(self, *, sandbox_id, operation_id, infrastructure_deadline_unix_ms, **kwargs):
        return SimpleNamespace(id='unrelated-sid')

    async def forbidden_delete(*args, **kwargs):
        raise AssertionError('mismatched response is not ownership proof')

    monkeypatch.setattr(module.OwnedShapeClient.__bases__[0], 'start_sandbox', create)
    monkeypatch.setattr(client, 'stop_sandbox', forbidden_delete)
    with pytest.raises(RuntimeError, match='different from the claimed ID'):
        asyncio.run(client.start_sandbox(**fields))
    assert {r['sandbox_id'] for r in records()} == {'owned-sid'}
    assert records()[-1]['event'] == 'sandbox_create_identity_mismatch'


def test_real_modern_sdk_provider_wrapper_reconcile_one_backend_create(wrapper, monkeypatch):
    import inspect
    from unittest.mock import AsyncMock

    from uni_agent.sandbox import base
    from uni_agent.sandbox.flash_sandbox import FlashSandboxSandbox

    module, _, _, records = wrapper
    original = module.flash_sandbox.AsyncHTTPClient
    if 'infrastructure_deadline_unix_ms' not in inspect.signature(original.start_sandbox).parameters:
        pytest.skip('Run with the current flash-sandbox SDK on PYTHONPATH')
    from flash_sandbox._transport import AsyncTransport

    created, requests, deleted = {}, [], []

    async def post(self, path, *, json, **kwargs):
        assert path == 'sandboxes'
        assert records()[-1]['sandbox_id'] == json['id']
        assert records()[-1]['operation_id'] == json['operation_id']
        requests.append((json.copy(), kwargs['headers'].copy(), kwargs['timeout']))
        created.setdefault(json['operation_id'], json['id'])
        if len(requests) == 1:
            raise OSError('lost HTTP reply after backend accepted create')
        return dict(id=created[json['operation_id']], operation_id=json['operation_id'], lifecycle_state='running')

    async def delete(self, path, **kwargs):
        deleted.append(path)

    monkeypatch.setattr(AsyncTransport, '_post', post)
    monkeypatch.setattr(AsyncTransport, '_delete', delete)
    monkeypatch.setattr(module.flash_sandbox, 'AsyncHTTPClient', module.OwnedShapeClient)
    monkeypatch.setenv('OTELA_API', 'fixture')
    local_asyncio = SimpleNamespace(**vars(asyncio))
    local_asyncio.sleep = AsyncMock()
    monkeypatch.setattr(base, 'asyncio', local_asyncio)
    sandbox = FlashSandboxSandbox(endpoint='http://fixture', sandbox_type='firecracker',
                                   cpu_cores=1, memory_mb=4096, keep_alive=False, start_timeout=60)
    monkeypatch.setattr(sandbox, '_bootstrap', AsyncMock())

    async def scenario():
        await sandbox.__aenter__()
        assert len(created) == 1 and len(requests) == 2
        assert requests[0] == requests[1]
        assert deleted == []
        assert [r['event'] for r in records()] == ['sandbox_claimed', 'sandbox_claimed', 'sandbox_created']
        await sandbox.stop()
        assert deleted == ['sandboxes/' + next(iter(created.values()))]

    asyncio.run(scenario())
